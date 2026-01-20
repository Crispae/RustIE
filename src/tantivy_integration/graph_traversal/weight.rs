//! OptimizedGraphTraversalWeight implementation.
//!
//! This module provides the weight type for graph traversal queries,
//! responsible for building scorers per segment.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::sync::atomic::Ordering;

use tantivy::{
    query::{Weight, Scorer},
    schema::{Field, Schema, IndexRecordOption},
    DocId, DocSet, Score, SegmentReader,
    Result as TantivyResult,
    Term,
};
use tantivy_fst::Regex;

use crate::query::ast::{Pattern, FlatPatternStep, Constraint, Matcher};
use super::types::{
    CollapsedSpec, PositionPrefilterPlan, ConstraintTermReq,
    DEFAULT_MAX_TERM_EXPANSIONS, REGEX_EXPANSION_COUNT, REGEX_EXPANSION_TERMS,
};
use super::candidate_driver::{
    CandidateDriver, EmptyDriver, CombinedPositionDriver,
    UnionPositionsIterator, UnionAndIntersectDriver,
    expand_matcher, get_or_compile_regex,
};
use super::pattern_utils::{build_constraint_requirements, unwrap_constraint_pattern_static};
use super::scorer::{OptimizedGraphTraversalScorer, is_exact_skippable};

/// Optimized weight for graph traversal queries using Odinson-style collapsed optimization.
/// No BooleanQuery weights - candidate generation driven exclusively by CombinedPositionDriver.
pub(crate) struct OptimizedGraphTraversalWeight {
    #[allow(dead_code)]
    traversal: crate::query::ast::Traversal,
    dependencies_binary_field: Field,
    #[allow(dead_code)]
    incoming_edges_field: Field,
    #[allow(dead_code)]
    outgoing_edges_field: Field,
    #[allow(dead_code)]
    src_pattern: crate::query::ast::Pattern,
    #[allow(dead_code)]
    dst_pattern: crate::query::ast::Pattern,
    /// Pre-computed flattened pattern steps (cached once per query)
    flat_steps: Vec<FlatPatternStep>,
    /// Position prefilter plan for edge-based position restrictions
    prefilter_plan: PositionPrefilterPlan,
    /// Collapse spec for src constraint - required for CombinedPositionDriver
    src_collapse: Option<CollapsedSpec>,
    /// Collapse spec for dst constraint - required for CombinedPositionDriver
    dst_collapse: Option<CollapsedSpec>,
    /// Cached compiled regex automata (shared across segments, thread-safe)
    regex_cache: Arc<RwLock<HashMap<String, Arc<Regex>>>>,
    /// Pre-computed constraint exact flags (true if constraint is exact string match)
    /// OPTIMIZATION: Computed once at query creation, not per document
    constraint_exact_flags: Vec<bool>,
    /// Pre-extracted constraint field names (computed once, not per document)
    constraint_field_names: Vec<String>,
}

impl OptimizedGraphTraversalWeight {
    pub fn new(
        src_pattern: crate::query::ast::Pattern,
        dst_pattern: crate::query::ast::Pattern,
        traversal: crate::query::ast::Traversal,
        dependencies_binary_field: Field,
        incoming_edges_field: Field,
        outgoing_edges_field: Field,
        flat_steps: Vec<FlatPatternStep>,
        prefilter_plan: PositionPrefilterPlan,
        src_collapse: Option<CollapsedSpec>,
        dst_collapse: Option<CollapsedSpec>,
    ) -> Self {
        // OPTIMIZATION: Pre-compute constraint_exact_flags ONCE at query creation
        let constraint_exact_flags: Vec<bool> = flat_steps.iter()
            .filter_map(|step| {
                if let FlatPatternStep::Constraint(constraint_pat) = step {
                    let unwrapped = unwrap_constraint_pattern_static(constraint_pat);
                    if let Pattern::Constraint(constraint) = unwrapped {
                        Some(is_exact_skippable(constraint))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        // OPTIMIZATION: Pre-extract constraint field names ONCE at query creation
        fn extract_field_name(pat: &Pattern) -> String {
            match pat {
                Pattern::NamedCapture { pattern, .. } => extract_field_name(pattern),
                Pattern::Repetition { pattern, .. } => extract_field_name(pattern),
                Pattern::Constraint(Constraint::Field { name, .. }) => name.clone(),
                _ => "word".to_string(),
            }
        }

        let constraint_field_names: Vec<String> = flat_steps.iter()
            .filter_map(|step| {
                if let FlatPatternStep::Constraint(pat) = step {
                    Some(extract_field_name(pat))
                } else {
                    None
                }
            })
            .collect();

        Self {
            src_pattern,
            dst_pattern,
            traversal,
            dependencies_binary_field,
            incoming_edges_field,
            outgoing_edges_field,
            flat_steps,
            prefilter_plan,
            src_collapse,
            dst_collapse,
            regex_cache: Arc::new(RwLock::new(HashMap::<String, Arc<Regex>>::new())),
            constraint_exact_flags,
            constraint_field_names,
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
                                            while stream.advance() && count < DEFAULT_MAX_TERM_EXPANSIONS {
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
                                                REGEX_EXPANSION_COUNT.fetch_add(1, Ordering::Relaxed);
                                                REGEX_EXPANSION_TERMS.fetch_add(count, Ordering::Relaxed);
                                            } else {
                                                log::warn!(
                                                    "Regex constraint '{}' expanded to 0 terms - pattern may not match any terms in segment",
                                                    pattern
                                                );
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
}

impl Weight for OptimizedGraphTraversalWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> TantivyResult<Box<dyn Scorer>> {
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

        // Cache the store reader (created once, reused for all documents in this segment)
        let store_reader = reader.get_store_reader(1)?;

        // OPTIMIZATION: Use pre-computed constraint_field_names from Weight (computed once at query creation)
        // No need to re-compute per segment

        // Build constraint requirements from flat_steps (need schema from reader)
        let schema = reader.schema();
        let mut constraint_reqs = build_constraint_requirements(&self.flat_steps, schema);

        // Expand regex patterns for constraint prefiltering (per-segment expansion)
        let expanded_reqs = self.expand_regex_constraints(reader, &self.flat_steps, schema);
        constraint_reqs.extend(expanded_reqs);

        // Log warning if no constraint prefiltering available
        if constraint_reqs.is_empty() {
            log::warn!("CONSTRAINT PREFILTERING DISABLED: No constraint fields indexed with positions!");
        }

        // Determine which constraint indices are collapsed (src/dst)
        let src_collapsed_idx = self.src_collapse.as_ref().map(|s| s.constraint_idx);
        let dst_collapsed_idx = self.dst_collapse.as_ref().map(|s| s.constraint_idx);

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
                Ok(Some(postings)) => {
                    edge_postings.push(Some(postings));
                }
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

        let mut scorer = OptimizedGraphTraversalScorer::new(
            src_driver,
            dst_driver,
            self.traversal.clone(),
            self.dependencies_binary_field,
            reader.clone(),
            store_reader,
            self.src_pattern.clone(),
            self.dst_pattern.clone(),
            boost,
            self.flat_steps.clone(),
            self.constraint_field_names.clone(),  // OPTIMIZATION: Use pre-computed from Weight
            self.prefilter_plan.clone(),
            edge_postings,
            constraint_reqs,
            constraint_postings,
            self.src_collapse.clone(),
            self.dst_collapse.clone(),
            self.constraint_exact_flags.clone(),  // OPTIMIZATION: Pass pre-computed flags
        );

        // Advance to the first document
        let _ = scorer.advance();

        Ok(Box::new(scorer))
    }

    fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> TantivyResult<tantivy::query::Explanation> {
        Ok(tantivy::query::Explanation::new("OptimizedGraphTraversalQuery", Score::default()))
    }
}
