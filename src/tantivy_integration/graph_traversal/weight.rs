//! OptimizedGraphTraversalWeight implementation.
//!
//! This module provides the weight type for graph traversal queries,
//! responsible for building scorers per segment.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tantivy::{
    query::{Weight, Scorer, Bm25Weight},
    schema::{Field, Schema, IndexRecordOption},
    DocId, DocSet, Score, SegmentReader,
    Result as TantivyResult,
    Term,
};
use tantivy_fst::Regex;

use crate::query::ast::{Pattern, FlatPatternStep, Constraint, Matcher};
use super::types::{
    prof_inc,
    CollapsedSpec, PositionPrefilterPlan, ConstraintTermReq,
    DEFAULT_MAX_TERM_EXPANSIONS,
};
#[allow(unused_imports)]
use super::types::{REGEX_EXPANSION_COUNT, REGEX_EXPANSION_TERMS};
use super::candidate_driver::{
    CandidateDriver, EmptyDriver, CombinedPositionDriver,
    UnionPositionsIterator, UnionAndIntersectDriver,
    expand_matcher, get_or_compile_regex,
};
use super::pattern_utils::{build_constraint_requirements, unwrap_constraint_pattern_static};
use super::scorer::OptimizedGraphTraversalScorer;

/// Bounded container for compiled regex automata and expanded BM25 terms.
/// Evicts all entries when full to cap memory.
pub(crate) struct RegexCacheInner {
    /// Compiled FST regex automata (pattern → automaton)
    map: HashMap<String, Arc<Regex>>,
    /// Expanded terms for BM25 scoring ((pattern, field) → terms)
    /// Cached to avoid redundant FST traversals across repeated queries.
    expanded_terms: HashMap<(String, Field), Vec<Term>>,
    max_entries: usize,
}

impl RegexCacheInner {
    pub fn new(max_entries: usize) -> Self {
        Self {
            map: HashMap::with_capacity(max_entries.min(256)),
            expanded_terms: HashMap::with_capacity(max_entries.min(256)),
            max_entries,
        }
    }

    pub fn get(&self, pattern: &str) -> Option<Arc<Regex>> {
        self.map.get(pattern).cloned()
    }

    pub fn insert(&mut self, pattern: String, regex: Arc<Regex>) {
        if self.map.len() >= self.max_entries && !self.map.contains_key(&pattern) {
            self.map.clear();
        }
        self.map.insert(pattern, regex);
    }

    /// Get cached expanded terms for a (pattern, field) combination.
    pub fn get_expanded_terms(&self, pattern: &str, field: Field) -> Option<Vec<Term>> {
        self.expanded_terms.get(&(pattern.to_string(), field)).cloned()
    }

    /// Cache expanded terms for a (pattern, field) combination.
    pub fn insert_expanded_terms(&mut self, pattern: String, field: Field, terms: Vec<Term>) {
        if self.expanded_terms.len() >= self.max_entries {
            self.expanded_terms.clear();
        }
        self.expanded_terms.insert((pattern, field), terms);
    }
}

/// Thread-safe shared regex cache (bounded size).
pub(crate) type RegexCache = Arc<RwLock<RegexCacheInner>>;

/// Optimized weight for graph traversal queries using Odinson-style collapsed optimization.
/// No BooleanQuery weights - candidate generation driven exclusively by CombinedPositionDriver.
pub(crate) struct OptimizedGraphTraversalWeight {
    dependencies_binary_field: Field,
    /// Pre-computed flattened pattern steps (cached once per query)
    flat_steps: Vec<FlatPatternStep>,
    /// Position prefilter plan for edge-based position restrictions
    prefilter_plan: PositionPrefilterPlan,
    /// Collapse spec for src constraint - required for CombinedPositionDriver
    src_collapse: Option<CollapsedSpec>,
    /// Collapse spec for dst constraint - required for CombinedPositionDriver
    dst_collapse: Option<CollapsedSpec>,
    /// Cached compiled regex automata (shared across segments, thread-safe)
    regex_cache: RegexCache,
    /// BM25 weight for scoring (IDF + field norms). None when no terms or scoring disabled.
    bm25_weight: Option<Bm25Weight>,
    /// Field used for field norms (e.g. word). Must have indexed field norms.
    score_field: Field,
}

impl OptimizedGraphTraversalWeight {
    /// Build weight without BM25 (e.g. when only cache is available). Scorer will use Option B.
    pub fn new(
        dependencies_binary_field: Field,
        flat_steps: Vec<FlatPatternStep>,
        prefilter_plan: PositionPrefilterPlan,
        src_collapse: Option<CollapsedSpec>,
        dst_collapse: Option<CollapsedSpec>,
        regex_cache: RegexCache,
        bm25_weight: Option<Bm25Weight>,
        score_field: Field,
    ) -> Self {
        Self {
            dependencies_binary_field,
            flat_steps,
            prefilter_plan,
            src_collapse,
            dst_collapse,
            regex_cache,
            bm25_weight,
            score_field,
        }
    }

    /// Build a CandidateDriver from a CollapsedSpec.
    /// Handles both exact matches (fast path) and regex patterns (term enumeration).
    ///
    /// Returns None if:
    /// - Postings don't exist (term not in segment)
    /// - Regex expansion exceeds max_expansions limit
    /// - No matching terms for regex pattern
    fn build_combined_driver(
        &self,
        reader: &SegmentReader,
        spec: &CollapsedSpec,
    ) -> Option<Box<dyn CandidateDriver>> {
        // Expand constraint matcher
        let constraint_postings = expand_matcher(
            reader,
            spec.constraint_field,
            &spec.constraint_matcher,
            DEFAULT_MAX_TERM_EXPANSIONS,
            self.regex_cache.clone(),
        )?;

        // Expand edge matcher
        let edge_postings = expand_matcher(
            reader,
            spec.edge_field,
            &spec.edge_matcher,
            DEFAULT_MAX_TERM_EXPANSIONS,
            self.regex_cache.clone(),
        )?;

        // Fast path: both exact (single postings each) - use CombinedPositionDriver
        if constraint_postings.len() == 1 && edge_postings.len() == 1 {
            return Some(Box::new(CombinedPositionDriver::new(
                constraint_postings.into_iter().next().unwrap(),
                edge_postings.into_iter().next().unwrap(),
            )));
        }

        // Regex path: use UnionAndIntersectDriver
        let lhs = UnionPositionsIterator::new(constraint_postings);
        let rhs = UnionPositionsIterator::new(edge_postings);

        Some(Box::new(UnionAndIntersectDriver::new(lhs, rhs)))
    }

    /// Expand regex patterns in constraints to their matching terms for prefiltering
    /// Returns additional ConstraintTermReq entries for regex patterns
    fn expand_regex_constraints(
        &self,
        reader: &SegmentReader,
        flat_steps: &[FlatPatternStep],
        schema: &Schema,
    ) -> Vec<ConstraintTermReq> {
        let mut expanded_reqs = Vec::new();
        let mut constraint_idx = 0;

        for step in flat_steps.iter() {
            if let FlatPatternStep::Constraint(pat) = step {
                let inner = unwrap_constraint_pattern_static(pat);

                if let Pattern::Constraint(Constraint::Field { name, matcher }) = inner {
                    // Only expand regex patterns (exact strings are already handled by build_constraint_requirements)
                    if let Matcher::Regex { pattern, .. } = matcher {
                        if let Ok(field) = schema.get_field(name) {
                            // Check if field is indexed with positions
                            let field_entry = schema.get_field_entry(field);
                            let has_positions = field_entry.field_type().get_index_record_option()
                                .map(|opt| opt.has_positions())
                                .unwrap_or(false);

                            if has_positions {
                                // Strip /.../ delimiters
                                let clean_pattern = pattern.trim_start_matches('/').trim_end_matches('/');

                                // Expand regex using FST automaton
                                if let Some(automaton) = get_or_compile_regex(&self.regex_cache, clean_pattern) {
                                    if let Ok(inverted_index) = reader.inverted_index(field) {
                                        let term_dict = inverted_index.terms();

                                        if let Some(mut stream) = term_dict.search(automaton.as_ref()).into_stream().ok() {
                                            let mut count = 0;
                                            // Odinson-style: expand ALL matching terms without cap.
                                            // FST automaton search is exact -- every matching term
                                            // in the index is enumerated with zero false positives.
                                            while stream.advance() {
                                                let term_bytes = stream.key();
                                                let term_str = String::from_utf8_lossy(term_bytes);

                                                expanded_reqs.push(ConstraintTermReq {
                                                    field,
                                                    term: term_str.to_string(),
                                                    constraint_idx,
                                                });
                                                count += 1;
                                            }

                                            if count > 0 {
                                                prof_inc!(REGEX_EXPANSION_COUNT);
                                                prof_inc!(REGEX_EXPANSION_TERMS, count);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                constraint_idx += 1;
            }
        }

        expanded_reqs
    }

    /// Build the concrete scorer for this segment. Used by execution to avoid boxing and enable take_current_doc_matches.
    pub(crate) fn concrete_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
    ) -> TantivyResult<OptimizedGraphTraversalScorer> {
        // Odinson-style: Build drivers exclusively from collapse specs
        // No GenericDriver fallback - use EmptyDriver when postings unavailable

        let src_driver: Box<dyn CandidateDriver> = if let Some(ref spec) = self.src_collapse {
            if let Some(driver) = self.build_combined_driver(reader, spec) {
                driver
            } else {
                Box::new(EmptyDriver)
            }
        } else {
            Box::new(EmptyDriver)
        };

        let dst_driver: Box<dyn CandidateDriver> = if let Some(ref spec) = self.dst_collapse {
            if let Some(driver) = self.build_combined_driver(reader, spec) {
                driver
            } else {
                Box::new(EmptyDriver)
            }
        } else {
            Box::new(EmptyDriver)
        };

        // Get fast field column for O(1) access to dependency graph bytes
        // This avoids document store decompression overhead for each graph access
        let schema = reader.schema();
        let field_name = schema.get_field_name(self.dependencies_binary_field);
        let dependencies_fast_field = reader
            .fast_fields()
            .bytes(field_name)
            .ok()
            .flatten();

        // Pre-extract constraint field names from flat_steps (computed once, not per document)
        fn unwrap_pattern_for_field_name(pat: &crate::query::ast::Pattern) -> String {
            use crate::query::ast::Pattern;
            match pat {
                Pattern::NamedCapture { pattern, .. } => unwrap_pattern_for_field_name(pattern),
                Pattern::Repetition { pattern, .. } => unwrap_pattern_for_field_name(pattern),
                Pattern::Constraint(crate::query::ast::Constraint::Field { name, .. }) => name.clone(),
                _ => "word".to_string(),
            }
        }

        let constraint_field_names: Vec<String> = self.flat_steps.iter()
            .filter_map(|step| {
                if let FlatPatternStep::Constraint(pat) = step {
                    Some(unwrap_pattern_for_field_name(pat))
                } else {
                    None
                }
            })
            .collect();

        // Get fast field columns for constraint fields (O(1) columnar access)
        // This avoids document store decompression overhead for constraint token access
        // Text fields use StrColumn with dictionary encoding for fast field access
        let constraint_fast_fields: Vec<Option<tantivy::columnar::StrColumn>> = constraint_field_names
            .iter()
            .map(|field_name| {
                reader
                    .fast_fields()
                    .str(field_name)
                    .ok()
                    .flatten()
            })
            .collect();

        // Build constraint requirements from flat_steps (need schema from reader)
        let schema = reader.schema();
        let mut constraint_reqs = build_constraint_requirements(&self.flat_steps, schema);

        // Expand regex patterns for constraint prefiltering (per-segment expansion)
        let expanded_reqs = self.expand_regex_constraints(reader, &self.flat_steps, schema);
        constraint_reqs.extend(expanded_reqs);

        // Log warning if no constraint prefiltering available
        if constraint_reqs.is_empty() {
        }

        // Create postings cursors for edge terms (one per segment)
        let mut edge_postings = Vec::new();

        for req in &self.prefilter_plan.edge_reqs {
            let term = Term::from_field_text(req.field, &req.label);
            let inverted_index = reader.inverted_index(req.field);

            let postings_result = if let Ok(inv_idx) = inverted_index {
                inv_idx.read_postings(&term, IndexRecordOption::WithFreqsAndPositions)
            } else {
                Ok(None)
            };

            match postings_result {
                Ok(Some(postings)) => edge_postings.push(Some(postings)),
                _ => edge_postings.push(None),
            }
        }

        // Create postings cursors for constraint terms (one per segment)
        let mut constraint_postings = Vec::new();

        for req in &constraint_reqs {
            let term = Term::from_field_text(req.field, &req.term);
            let inverted_index = reader.inverted_index(req.field);

            let postings_result = if let Ok(inv_idx) = inverted_index {
                inv_idx.read_postings(&term, IndexRecordOption::WithFreqsAndPositions)
            } else {
                Ok(None)
            };

            match postings_result {
                Ok(Some(postings)) => constraint_postings.push(Some(postings)),
                _ => constraint_postings.push(None),
            }
        }

        let fieldnorm_reader = reader
            .get_fieldnorms_reader(self.score_field)
            .ok();
        let boosted_bm25 = self
            .bm25_weight
            .as_ref()
            .map(|w| w.clone().boost_by(boost));

        let mut scorer = OptimizedGraphTraversalScorer::new(
            src_driver,
            dst_driver,
            dependencies_fast_field,
            boost,
            self.flat_steps.clone(),
            constraint_field_names,
            self.prefilter_plan.clone(),
            edge_postings,
            constraint_reqs,
            constraint_postings,
            self.src_collapse.clone(),
            self.dst_collapse.clone(),
            constraint_fast_fields,
            boosted_bm25,
            fieldnorm_reader,
        );

        // Advance to the first document
        let _ = scorer.advance();

        Ok(scorer)
    }
}

impl Weight for OptimizedGraphTraversalWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> TantivyResult<Box<dyn Scorer>> {
        Ok(Box::new(self.concrete_scorer(reader, boost)?))
    }

    fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> TantivyResult<tantivy::query::Explanation> {
        Ok(tantivy::query::Explanation::new("OptimizedGraphTraversalQuery", Score::default()))
    }
}

// Compile-time assertion: OptimizedGraphTraversalWeight must be Send + Sync to be shared across
// segments via Arc in execute_graph_traversal. If this fails, fix the offending field rather
// than adding unsafe impl.
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn check() {
        assert_send_sync::<OptimizedGraphTraversalWeight>();
    }
};
