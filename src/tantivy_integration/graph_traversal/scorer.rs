//! Scorer implementation for graph traversal queries.
//!
//! This module contains the `OptimizedGraphTraversalScorer` which performs
//! the actual document matching and scoring during query execution.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use rayon::prelude::*;
use tantivy::{
    query::Scorer,
    schema::Field,
    DocId, Score,
    DocSet, SegmentReader,
    store::StoreReader,
    postings::{SegmentPostings, Postings},
    columnar::{BytesColumn, StrColumn},
};

use crate::query::ast::{FlatPatternStep, Pattern, Constraint, Matcher};
use crate::digraph::zero_copy::ZeroCopyGraph;
use crate::digraph::traversal::PARALLEL_START_POSITIONS_THRESHOLD;

use super::types::{
    prof_inc, capture_name,
    ConstraintTermReq, PositionPrefilterPlan, PositionRequirement, CollapsedSpec, CollapsedMatcher,
};
// Profiling counters - only mutated through prof_inc! macro
#[allow(unused_imports)]
use super::types::{
    CALL_COUNT, GRAPH_DESER_COUNT, GRAPH_DESER_SKIPPED,
    PREFILTER_DOCS, PREFILTER_KILLED, PREFILTER_ALLOWED_POS_SUM, PREFILTER_ALLOWED_POS_COUNT,
    DRIVER_ALIGNMENT_DOCS, DRIVER_INTERSECTION_SUM, DRIVER_INTERSECTION_COUNT,
    PREFILTER_SKIPPED_ALL_COLLAPSED, TOKEN_EXTRACTION_SKIPPED,
};
use super::candidate_driver::CandidateDriver;
use super::intersection::TwoPhaseIntersection;

/// Optimized scorer for graph traversal queries
/// Uses CandidateDriver abstraction for Odinson-style collapsed query optimization
/// and TwoPhaseIntersection for efficient document-level alignment (Lucene TwoPhaseIterator pattern)
pub struct OptimizedGraphTraversalScorer {
    /// Two-phase intersection of src/dst drivers
    /// Implements Lucene's ConjunctionDISI.intersectSpans() pattern:
    /// Phase 1: Document alignment via skip-list optimized intersection
    /// Phase 2: Graph traversal verification (check_graph_traversal)
    pub(crate) intersection: TwoPhaseIntersection,
    #[allow(dead_code)]
    pub(crate) traversal: crate::query::ast::Traversal,
    #[allow(dead_code)]
    pub(crate) dependencies_binary_field: Field,
    pub(crate) reader: SegmentReader,
    /// Fast field column for O(1) access to dependency graph bytes (columnar storage)
    /// This avoids document store decompression overhead for graph access
    pub(crate) dependencies_fast_field: Option<BytesColumn>,
    /// Store reader for constraint token extraction (still needed for lazy token loading)
    pub(crate) store_reader: StoreReader,
    pub(crate) current_doc: Option<DocId>,
    pub(crate) current_matches: Vec<(DocId, Score)>,
    pub(crate) match_index: usize,
    #[allow(dead_code)]
    pub(crate) src_pattern: crate::query::ast::Pattern,
    #[allow(dead_code)]
    pub(crate) dst_pattern: crate::query::ast::Pattern,
    pub(crate) current_doc_matches: Vec<crate::types::SpanWithCaptures>,
    /// Boost factor from weight creation
    pub(crate) boost: Score,
    /// Pre-computed flattened pattern steps (cached from Weight)
    pub(crate) flat_steps: Vec<FlatPatternStep>,
    /// Pre-extracted constraint field names (cached from flat_steps)
    pub(crate) constraint_field_names: Vec<String>,
    /// Position prefilter plan
    pub(crate) prefilter_plan: PositionPrefilterPlan,
    /// Postings cursors for edge terms (one per EdgeTermReq)
    pub(crate) edge_postings: Vec<Option<SegmentPostings>>,
    /// Constraint term requirements (built in scorer)
    pub(crate) constraint_reqs: Vec<ConstraintTermReq>,
    /// Postings cursors for constraint terms (one per ConstraintTermReq)
    pub(crate) constraint_postings: Vec<Option<SegmentPostings>>,
    /// Optional collapse spec for src (for position handoff)
    pub(crate) src_collapse: Option<CollapsedSpec>,
    /// Optional collapse spec for dst (for position handoff)
    pub(crate) dst_collapse: Option<CollapsedSpec>,
    /// Tracks which constraints are fully covered by postings intersection
    /// and can skip token parsing entirely. True = positions from postings are sufficient,
    /// no need to parse tokens from document store.
    pub(crate) constraints_covered_by_postings: Vec<bool>,
    /// Fast field columns for O(1) access to constraint field tokens (columnar storage)
    /// One entry per constraint field name, None if fast field not available
    /// Text fields use StrColumn with dictionary encoding for fast field access
    pub(crate) constraint_fast_fields: Vec<Option<StrColumn>>,
    /// Scratch buffer for source driver positions (reused across documents to avoid allocation)
    pub(crate) src_positions_buf: Vec<u32>,
    /// Scratch buffer for destination driver positions (reused across documents to avoid allocation)
    pub(crate) dst_positions_buf: Vec<u32>,
}

impl OptimizedGraphTraversalScorer {
    /// Create a new scorer
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        src_driver: Box<dyn CandidateDriver>,
        dst_driver: Box<dyn CandidateDriver>,
        traversal: crate::query::ast::Traversal,
        dependencies_binary_field: Field,
        reader: SegmentReader,
        dependencies_fast_field: Option<BytesColumn>,
        store_reader: StoreReader,
        src_pattern: crate::query::ast::Pattern,
        dst_pattern: crate::query::ast::Pattern,
        boost: Score,
        flat_steps: Vec<FlatPatternStep>,
        constraint_field_names: Vec<String>,
        prefilter_plan: PositionPrefilterPlan,
        edge_postings: Vec<Option<SegmentPostings>>,
        constraint_reqs: Vec<ConstraintTermReq>,
        constraint_postings: Vec<Option<SegmentPostings>>,
        src_collapse: Option<CollapsedSpec>,
        dst_collapse: Option<CollapsedSpec>,
        constraint_fast_fields: Vec<Option<StrColumn>>,
    ) -> Self {
        // Wrap drivers in TwoPhaseIntersection for Lucene-style document alignment
        let intersection = TwoPhaseIntersection::new(src_driver, dst_driver);

        // Compute which constraints are fully covered by postings intersection and
        // DON'T need token verification. A constraint is covered ONLY if:
        // 1. It's an EXACT string match (not regex!) with postings, OR
        // 2. It's src/dst with a collapse spec that's exact (positions come from CombinedPositionDriver), OR
        // 3. It's a wildcard (any position is valid - positions come from edges)
        //
        // IMPORTANT: Regex constraints CANNOT be marked as covered even if they have
        // constraint_reqs, because the traversal engine still needs to verify the token
        // actually matches the regex pattern (term expansion may include false positives).
        let num_constraints = prefilter_plan.num_constraints;
        let mut constraints_covered_by_postings = vec![false; num_constraints];

        // First, determine which constraints are exact string matches (not regex)
        let mut constraint_is_exact = vec![false; num_constraints];
        let mut constraint_idx = 0;
        for step in &flat_steps {
            if let FlatPatternStep::Constraint(pat) = step {
                if constraint_idx < num_constraints {
                    let inner = unwrap_constraint_pattern_for_coverage(pat);
                    constraint_is_exact[constraint_idx] = match inner {
                        Pattern::Constraint(Constraint::Field { matcher, .. }) => {
                            matches!(matcher, Matcher::String(_))
                        }
                        Pattern::Constraint(Constraint::Wildcard) => true, // Wildcards don't need verification
                        _ => false,
                    };
                }
                constraint_idx += 1;
            }
        }

        // Mark constraints covered by constraint_reqs ONLY if they're exact string matches
        for req in &constraint_reqs {
            if req.constraint_idx < num_constraints && constraint_is_exact[req.constraint_idx] {
                constraints_covered_by_postings[req.constraint_idx] = true;
            }
        }

        // Mark src/dst constraints covered by collapse specs ONLY if they're exact
        if let Some(ref spec) = src_collapse {
            if spec.constraint_idx < num_constraints {
                // Check if the collapse spec constraint matcher is exact (String, not Regex)
                let is_exact = match &spec.constraint_matcher {
                    CollapsedMatcher::Exact(_) => true,
                    CollapsedMatcher::RegexPattern(_) => false,
                };
                if is_exact {
                    constraints_covered_by_postings[spec.constraint_idx] = true;
                }
            }
        }
        if let Some(ref spec) = dst_collapse {
            if spec.constraint_idx < num_constraints {
                // Check if the collapse spec constraint matcher is exact (String, not Regex)
                let is_exact = match &spec.constraint_matcher {
                    CollapsedMatcher::Exact(_) => true,
                    CollapsedMatcher::RegexPattern(_) => false,
                };
                if is_exact {
                    constraints_covered_by_postings[spec.constraint_idx] = true;
                }
            }
        }

        // Mark wildcard constraints as covered (they don't need token verification)
        constraint_idx = 0;
        for step in &flat_steps {
            if let FlatPatternStep::Constraint(pat) = step {
                let inner = unwrap_constraint_pattern_for_coverage(pat);
                if matches!(inner, Pattern::Constraint(Constraint::Wildcard)) {
                    if constraint_idx < num_constraints {
                        constraints_covered_by_postings[constraint_idx] = true;
                    }
                }
                constraint_idx += 1;
            }
        }

        Self {
            intersection,
            traversal,
            dependencies_binary_field,
            reader,
            dependencies_fast_field,
            store_reader,
            current_doc: None,
            current_matches: Vec::new(),
            match_index: 0,
            src_pattern,
            dst_pattern,
            current_doc_matches: Vec::new(),
            boost,
            flat_steps,
            constraint_field_names,
            prefilter_plan,
            edge_postings,
            constraint_reqs,
            constraint_postings,
            src_collapse,
            dst_collapse,
            constraints_covered_by_postings,
            constraint_fast_fields,
            src_positions_buf: Vec::with_capacity(16),
            dst_positions_buf: Vec::with_capacity(16),
        }
    }
}

/// Helper to unwrap NamedCapture/Repetition for coverage check
fn unwrap_constraint_pattern_for_coverage(pat: &Pattern) -> &Pattern {
    match pat {
        Pattern::NamedCapture { pattern, .. } => unwrap_constraint_pattern_for_coverage(pattern),
        Pattern::Repetition { pattern, .. } => unwrap_constraint_pattern_for_coverage(pattern),
        _ => pat,
    }
}

/// Extract tokens from a constraint field using fast fields (O(1) columnar access)
/// Returns empty vector if fast field not available
/// Uses StrColumn for text fields with dictionary encoding
fn extract_constraint_tokens_from_fast_field(
    constraint_idx: usize,
    field_name: &str,
    fast_field_columns: &[Option<StrColumn>],
    doc_id: DocId,
) -> Vec<String> {
    if let Some(Some(str_col)) = fast_field_columns.get(constraint_idx) {
        // StrColumn uses dictionary encoding - get term ordinal for this document
        // ords() returns a Column<u64> with term ordinals for each document
        let ords_col = str_col.ords();
        let term_ord = ords_col.values.get_val(doc_id);
        
        // Get the string value from the dictionary using the term ordinal
        // ord_to_str() requires a mutable String buffer
        let mut encoded_buffer = String::new();
        let ord_to_str_result = str_col.ord_to_str(term_ord, &mut encoded_buffer);
        if ord_to_str_result.as_ref().map(|b| *b).unwrap_or(false) && !encoded_buffer.is_empty() {
            // Single-pass decode: split directly without pre-counting pipe chars
            let token_fields = ["word", "lemma", "pos", "tag", "chunk", "entity", "norm", "raw"];
            if token_fields.contains(&field_name) {
                return encoded_buffer.split('|').map(|s| s.to_string()).collect();
            } else {
                return vec![encoded_buffer];
            }
        }
    }
    Vec::new()
}

/// Lazy constraint token loader - parses constraint fields only when first accessed
/// This avoids parsing constraint fields that are never reached during graph traversal
/// Uses fast fields (columnar storage) for O(1) access
pub(crate) struct LazyConstraintTokens<'a> {
    constraint_field_names: &'a [String],
    schema: &'a tantivy::schema::Schema,
    cache: Vec<Option<Vec<String>>>,
    /// Fast field columns for O(1) columnar access to constraint tokens
    /// One entry per constraint field name, None if fast field not available
    /// Text fields use StrColumn with dictionary encoding
    fast_field_columns: &'a [Option<StrColumn>],
    /// Current document ID for fast field access
    doc_id: DocId,
}

/// Trait for token accessors used in parallel processing.
/// This allows both ImmutableTokenAccessor (with actual tokens) and
/// EmptyTokenAccessor (when all constraints are covered by postings) to be used.
pub(crate) trait TokenAccessor: Sync {
    /// Get token at a specific position for a constraint (returns owned String)
    fn get(&self, constraint_idx: usize, position: usize) -> Option<String>;
}

/// Immutable token accessor for thread-safe parallel processing
/// Provides read-only access to pre-loaded tokens from LazyConstraintTokens
pub(crate) struct ImmutableTokenAccessor<'a> {
    tokens: Vec<Option<&'a [String]>>,
}

impl<'a> ImmutableTokenAccessor<'a> {
    /// Create a new immutable token accessor from a LazyConstraintTokens instance
    /// All tokens must be pre-loaded before calling this
    pub fn new(lazy_tokens: &'a LazyConstraintTokens) -> Self {
        let tokens: Vec<Option<&'a [String]>> = lazy_tokens
            .cache
            .iter()
            .map(|opt_tokens| opt_tokens.as_ref().map(|tokens| tokens.as_slice()))
            .collect();
        Self { tokens }
    }
}

impl<'a> TokenAccessor for ImmutableTokenAccessor<'a> {
    fn get(&self, constraint_idx: usize, position: usize) -> Option<String> {
        self.tokens
            .get(constraint_idx)?
            .and_then(|tokens| tokens.get(position))
            .cloned()
    }
}

/// Empty token accessor for when all constraints are covered by postings.
/// Always returns None - the traversal engine will use allowed_positions_hashset
/// and constraint_exact_flags to validate positions without token verification.
pub(crate) struct EmptyTokenAccessor {
    _num_constraints: usize,
}

impl EmptyTokenAccessor {
    /// Create an empty token accessor for the given number of constraints
    pub fn new(num_constraints: usize) -> Self {
        Self { _num_constraints: num_constraints }
    }
}

impl TokenAccessor for EmptyTokenAccessor {
    fn get(&self, _constraint_idx: usize, _position: usize) -> Option<String> {
        None
    }
}

impl<'a> LazyConstraintTokens<'a> {
    /// Create a new lazy token loader with fast field support
    pub fn new(
        constraint_field_names: &'a [String],
        schema: &'a tantivy::schema::Schema,
        fast_field_columns: &'a [Option<StrColumn>],
        doc_id: DocId,
    ) -> Self {
        let cache = vec![None; constraint_field_names.len()];
        Self {
            constraint_field_names,
            schema,
            cache,
            fast_field_columns,
            doc_id,
        }
    }

    /// Ensure tokens for a constraint are loaded (parses on first access)
    pub fn ensure_loaded(&mut self, constraint_idx: usize) {
        if constraint_idx >= self.cache.len() {
            return;
        }

        if self.cache[constraint_idx].is_none() {
            let field_name = &self.constraint_field_names[constraint_idx];
            let tokens = self.extract_tokens_from_field(constraint_idx, field_name);
            self.cache[constraint_idx] = Some(tokens);
        }
    }

    /// Get token at a specific position for a constraint (returns owned String)
    /// Parses the constraint field on first access
    pub fn get(&mut self, constraint_idx: usize, position: usize) -> Option<String> {
        self.ensure_loaded(constraint_idx);
        self.cache[constraint_idx]
            .as_ref()
            .and_then(|tokens| tokens.get(position))
            .cloned()
    }

    /// Get all tokens for a constraint (returns reference after ensuring loaded)
    /// Used for position calculation that needs full token list
    pub fn get_all_tokens(&mut self, constraint_idx: usize) -> Option<&[String]> {
        self.ensure_loaded(constraint_idx);
        self.cache[constraint_idx].as_ref().map(|tokens| tokens.as_slice())
    }

    /// Extract tokens from a field using fast fields (O(1) columnar access)
    /// Uses StrColumn for text fields with dictionary encoding
    fn extract_tokens_from_field(&self, constraint_idx: usize, field_name: &str) -> Vec<String> {
        // Use fast field access (O(1) columnar access) with StrColumn
        if let Some(Some(str_col)) = self.fast_field_columns.get(constraint_idx) {
            // StrColumn uses dictionary encoding - get term ordinal for this document
            // ords() returns a Column<u64> with term ordinals for each document
            let ords_col = str_col.ords();
            let term_ord = ords_col.values.get_val(self.doc_id);
            
            // Get the string value from the dictionary using the term ordinal
            // ord_to_str() requires a mutable String buffer
            let mut encoded_buffer = String::new();
            if str_col.ord_to_str(term_ord, &mut encoded_buffer).is_ok() && !encoded_buffer.is_empty() {
                // Single-pass decode: split directly without pre-counting pipe chars
                let token_fields = ["word", "lemma", "pos", "tag", "chunk", "entity", "norm", "raw"];
                if token_fields.contains(&field_name) {
                    return encoded_buffer.split('|').map(|s| s.to_string()).collect();
                } else {
                    return vec![encoded_buffer];
                }
            }
        }

        // Fast field not available - return empty vector (fast fields must be available after reindexing)
        Vec::new()
    }
}

/// Build destination inverted index (HashMap) from destination positions.
/// This matches Odinson's mkInvIndex(getAllSpansWithCaptures(dstSpans)) approach.
/// 
/// Destination positions come from Tantivy's inverted index (via CombinedPositionDriver),
/// then we build a HashMap on-the-fly for O(1) span lookup.
/// 
/// Returns HashMap mapping position (u32) to pre-computed SpanWithCaptures.
fn build_dst_inverted_index(
    dst_positions: &[u32],
    flat_steps: &[FlatPatternStep],
) -> HashMap<u32, Vec<crate::types::SpanWithCaptures>> {
    let mut index: HashMap<u32, Vec<crate::types::SpanWithCaptures>> = HashMap::new();
    
    if dst_positions.is_empty() {
        return index;
    }
    
    // Find the last constraint step (destination constraint)
    let dst_constraint_step = flat_steps.iter()
        .rev()
        .find(|step| matches!(step, FlatPatternStep::Constraint(_)));
    
    // Count total constraints to derive the destination constraint index
    let total_constraints = flat_steps.iter()
        .filter(|s| matches!(s, FlatPatternStep::Constraint(_)))
        .count();
    let dst_constraint_idx = total_constraints.saturating_sub(1);

    let cap_name = if let Some(FlatPatternStep::Constraint(ref pat)) = dst_constraint_step {
        match pat {
            Pattern::NamedCapture { name, .. } => name.clone(),
            _ => capture_name(dst_constraint_idx),
        }
    } else {
        capture_name(dst_constraint_idx)
    };
    
    // Pre-compute SpanWithCaptures for each destination position
    for &dst_pos in dst_positions {
        let span = crate::types::Span {
            start: dst_pos as usize,
            end: dst_pos as usize + 1,
        };
        let capture = crate::types::NamedCapture::new(cap_name.clone(), span.clone());
        let span_with_captures = crate::types::SpanWithCaptures::with_captures(span, vec![capture]);
        
        index.entry(dst_pos).or_default().push(span_with_captures);
    }
    
    index
}

/// Process a single start position and return all matching paths as spans
/// This is used for parallel processing where each thread processes one start position
/// graph_bytes: The raw bytes of the graph (zero-copy, can be shared across threads)
/// dst_set: Pre-computed destination positions for O(1) endpoint validation (Odinson-style)
/// dst_index: Optional pre-built destination inverted index for O(1) span lookup
pub(crate) fn process_single_start_position<T: TokenAccessor>(
    graph_bytes: &[u8],
    flat_steps: &[FlatPatternStep],
    start_pos: usize,
    constraint_field_names: &[String],
    token_accessor: &T,
    allowed_positions: &[Option<HashSet<u32>>],
    constraint_exact_flags: &[bool],
    dst_set: &HashSet<u32>,
) -> Vec<crate::types::SpanWithCaptures> {
    process_single_start_position_with_index(
        graph_bytes,
        flat_steps,
        start_pos,
        constraint_field_names,
        token_accessor,
        allowed_positions,
        constraint_exact_flags,
        dst_set,
        None,
    )
}

/// Internal function that accepts optional destination inverted index
fn process_single_start_position_with_index<T: TokenAccessor>(
    graph_bytes: &[u8],
    flat_steps: &[FlatPatternStep],
    start_pos: usize,
    constraint_field_names: &[String],
    token_accessor: &T,
    allowed_positions: &[Option<HashSet<u32>>],
    constraint_exact_flags: &[bool],
    dst_set: &HashSet<u32>,
    dst_index_opt: Option<&Arc<HashMap<u32, Vec<crate::types::SpanWithCaptures>>>>,
) -> Vec<crate::types::SpanWithCaptures> {
    // Recreate ZeroCopyGraph from bytes for this thread (zero-copy, no allocation)
    let graph = match ZeroCopyGraph::from_bytes(graph_bytes) {
        Ok(g) => g,
        Err(_) => return Vec::new(),
    };

    let traversal_engine = crate::digraph::traversal::GraphTraversal::new(graph);

    // Create mutable closure for token access (wraps immutable accessor)
    let mut get_token = |constraint_idx: usize, position: usize| -> Option<String> {
        token_accessor.get(constraint_idx, position)
    };

    // Count total constraints to identify destination constraint index
    let total_constraints: usize = flat_steps.iter()
        .filter(|s| matches!(s, FlatPatternStep::Constraint(_)))
        .count();
    let dst_constraint_idx = total_constraints.saturating_sub(1);

    // Use optimized traversal if inverted index is available, otherwise use batch traversal
    let matches = if let Some(dst_index) = dst_index_opt {
        // OPTIMIZED: Use inverted index for O(1) destination lookup
        let reachable_pairs = traversal_engine.automaton_find_reachable_destinations(
            flat_steps,
            &[start_pos],
            dst_set,
            constraint_field_names,
            &mut get_token,
            allowed_positions,
            constraint_exact_flags,
        );

        let mut matches = Vec::with_capacity(reachable_pairs.len());
        for (_src_pos, dst_pos, path) in &reachable_pairs {
            if path.is_empty() {
                continue;
            }

            let mut captures = Vec::with_capacity(path.len());
            let mut c_idx = 0;
            let mut dst_capture_added = false;
            
            for step in flat_steps.iter() {
                if let FlatPatternStep::Constraint(ref pat) = step {
                    if let Some(&node_idx) = path.get(c_idx) {
                        let is_dst_constraint = c_idx == dst_constraint_idx && node_idx == *dst_pos;
                        
                        if is_dst_constraint {
                            // Use pre-built destination span from inverted index (O(1) lookup)
                            if let Some(dst_spans) = dst_index.get(&(*dst_pos as u32)) {
                                if let Some(dst_span) = dst_spans.first() {
                                    captures.extend(dst_span.captures.iter().cloned());
                                    dst_capture_added = true;
                                }
                            }
                        }
                        
                        // Build capture if not using inverted index or for intermediate constraints
                        if !is_dst_constraint || !dst_capture_added {
                            let span = crate::types::Span { start: node_idx, end: node_idx + 1 };
                            let name = match pat {
                                Pattern::NamedCapture { name, .. } => name.clone(),
                                _ => capture_name(c_idx),
                            };
                            captures.push(crate::types::NamedCapture::new(name, span));
                        }
                    }
                    c_idx += 1;
                }
            }
            
            let min_pos = *path.iter().min().unwrap();
            let max_pos = *path.iter().max().unwrap();
            matches.push(
                crate::types::SpanWithCaptures::with_captures(
                    crate::types::Span { start: min_pos, end: max_pos + 1 },
                    captures
                )
            );
        }
        matches
    } else {
        // FALLBACK: Use batch traversal and build from paths (backward compatibility)
        let all_paths = traversal_engine.automaton_query_paths_batch(
            flat_steps,
            &[start_pos],
            dst_set,
            constraint_field_names,
            &mut get_token,
            allowed_positions,
            constraint_exact_flags,
        );

        let mut matches = Vec::new();
        for path in &all_paths {
            if !path.is_empty() {
                let mut captures = Vec::with_capacity(path.len());
                let mut c_idx = 0;
                for step in flat_steps.iter() {
                    if let FlatPatternStep::Constraint(ref pat) = step {
                        if let Some(&node_idx) = path.get(c_idx) {
                            let span = crate::types::Span { start: node_idx, end: node_idx + 1 };
                            let name = match pat {
                                Pattern::NamedCapture { name, .. } => name.clone(),
                                _ => capture_name(c_idx),
                            };
                            captures.push(crate::types::NamedCapture::new(name, span));
                        }
                        c_idx += 1;
                    }
                }
                let min_pos = *path.iter().min().unwrap();
                let max_pos = *path.iter().max().unwrap();
                matches.push(
                    crate::types::SpanWithCaptures::with_captures(
                        crate::types::Span { start: min_pos, end: max_pos + 1 },
                        captures
                    )
                );
            }
        }
        matches
    };
    
    matches
}

/// Check if a constraint is exact and can skip matches() when prefilter confirms
/// Only simple Field { Matcher::String } constraints are skippable
pub(crate) fn is_exact_skippable(constraint: &Constraint) -> bool {
    match constraint {
        Constraint::Field {
            matcher: Matcher::String(_),
            ..
        } => true,
        Constraint::Field {
            matcher: Matcher::Regex { .. },
            ..
        } => false,
        Constraint::Negated(_)
        | Constraint::Conjunctive(_)
        | Constraint::Disjunctive(_)
        | Constraint::Wildcard
        | Constraint::Fuzzy { .. } => false,
    }
}

impl OptimizedGraphTraversalScorer {
    /// Helper method to run traversal with any GraphTraversal<G: GraphAccess>
    /// This allows us to use ZeroCopyGraph directly without conversion
    #[allow(dead_code)]
    pub(crate) fn run_traversal_with_engine<G: crate::digraph::graph_trait::GraphAccess>(
        &mut self,
        traversal_engine: &crate::digraph::traversal::GraphTraversal<G>,
        flat_steps: &[FlatPatternStep],
        src_positions: &[usize],
        lazy_tokens: &mut LazyConstraintTokens,
        allowed_positions_hashset: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
    ) -> bool {
        let constraint_field_names = &self.constraint_field_names;

        let mut get_token = |constraint_idx: usize, position: usize| -> Option<String> {
            lazy_tokens.get(constraint_idx, position)
        };

        let mut all_matches = Vec::new();

        for &src_pos in src_positions {
            let all_paths = traversal_engine.automaton_query_paths(
                flat_steps,
                &[src_pos],
                constraint_field_names,
                &mut get_token,
                allowed_positions_hashset,
                constraint_exact_flags,
            );

            for path in &all_paths {
                if !path.is_empty() {
                    let mut captures = Vec::with_capacity(path.len());
                    let mut c_idx = 0;
                    for step in flat_steps.iter() {
                        if let FlatPatternStep::Constraint(ref pat) = step {
                            if let Some(&node_idx) = path.get(c_idx) {
                                let span = crate::types::Span { start: node_idx, end: node_idx + 1 };
                                let name = match pat {
                                    Pattern::NamedCapture { name, .. } => name.clone(),
                                    _ => capture_name(c_idx),
                                };
                                captures.push(crate::types::NamedCapture::new(name, span));
                            }
                            c_idx += 1;
                        }
                    }
                    let min_pos = *path.iter().min().unwrap();
                    let max_pos = *path.iter().max().unwrap();
                    all_matches.push(
                        crate::types::SpanWithCaptures::with_captures(
                            crate::types::Span { start: min_pos, end: max_pos + 1 },
                            captures
                        )
                    );
                }
            }

            if !all_paths.is_empty() {
                self.current_doc_matches.extend(all_matches);
                return true;
            }
        }

        false
    }

    /// Unwrap constraint pattern by removing NamedCapture and Repetition wrappers
    /// Returns the underlying constraint pattern
    pub(crate) fn unwrap_constraint_pattern<'a>(&self, pat: &'a Pattern) -> &'a Pattern {
        match pat {
            Pattern::NamedCapture { pattern, .. } => self.unwrap_constraint_pattern(pattern),
            Pattern::Repetition { pattern, .. } => self.unwrap_constraint_pattern(pattern),
            _ => pat,
        }
    }

    /// Compute sloppy frequency factor based on span width (Odinson-style)
    /// Uses the formula: 1.0 / (1.0 + distance) where distance = span width
    /// Shorter spans get higher scores
    fn compute_slop_factor(span_width: usize) -> Score {
        1.0 / (1.0 + span_width as f32)
    }

    /// Compute Odinson-style score for the current document
    /// Accumulates sloppy frequency for all matches, similar to Lucene/Odinson
    fn compute_odinson_score(&self) -> Score {
        if self.current_doc_matches.is_empty() {
            return 0.0;
        }

        let mut acc_sloppy_freq: Score = 0.0;

        for span_match in &self.current_doc_matches {
            let span_width = span_match.span.end.saturating_sub(span_match.span.start);
            acc_sloppy_freq += Self::compute_slop_factor(span_width);
        }

        let src_score = 1.0;
        let dst_score = 1.0;
        let base_score = (src_score + dst_score) / 2.0;

        let final_score = base_score * acc_sloppy_freq * self.boost;
        final_score.max(1.0)
    }

    /// Log final statistics when iteration completes
    fn log_final_stats(&self) {
        // Logging disabled for performance
    }

    /// Build allowed positions combining src_driver, dst_driver, and prefilter positions.
    /// Converts to HashSet<u32> with pre-allocated capacity to avoid rehashing.
    pub(crate) fn build_allowed_positions(
        &self,
        src_driver_positions: Option<&[u32]>,
        dst_driver_positions: Option<&[u32]>,
        prefilter_positions: &[Option<Vec<u32>>],
        num_constraints: usize,
    ) -> Vec<Option<HashSet<u32>>> {
        let mut result: Vec<Option<HashSet<u32>>> = vec![None; num_constraints];

        if let Some(src_positions) = src_driver_positions {
            if num_constraints > 0 {
                let mut set = HashSet::with_capacity(src_positions.len());
                set.extend(src_positions.iter().copied());
                result[0] = Some(set);
            }
        }

        if let Some(dst_positions) = dst_driver_positions {
            let last_idx = num_constraints.saturating_sub(1);
            if last_idx > 0 && last_idx < num_constraints {
                let mut set = HashSet::with_capacity(dst_positions.len());
                set.extend(dst_positions.iter().copied());
                result[last_idx] = Some(set);
            }
        }

        for (idx, prefilter) in prefilter_positions.iter().enumerate() {
            if idx < num_constraints {
                if result[idx].is_none() {
                    if let Some(positions) = prefilter {
                        let mut set = HashSet::with_capacity(positions.len());
                        set.extend(positions.iter().copied());
                        result[idx] = Some(set);
                    }
                }
            }
        }

        result
    }

    /// Intersect two sorted vectors of u32, storing result in the first vector
    /// O(n+m) time using two-pointer merge
    fn intersect_sorted_in_place(a: &mut Vec<u32>, b: &[u32]) {
        let mut out = Vec::with_capacity(a.len().min(b.len()));
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    out.push(a[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        *a = out;
    }

    /// Intersect two sorted slices into a new vector (avoids cloning)
    /// O(n+m) time using two-pointer merge
    fn intersect_sorted_slices(a: &[u32], b: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(a.len().min(b.len()));
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    out.push(a[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        out
    }

    /// Compute allowed positions for each constraint, using driver positions for collapsed constraints
    pub(crate) fn compute_allowed_positions(
        &mut self,
        doc_id: DocId,
        src_driver_positions: Option<&[u32]>,
        dst_driver_positions: Option<&[u32]>,
    ) -> Option<Vec<Option<Vec<u32>>>> {
        let mut allowed: Vec<Option<Vec<u32>>> = vec![None; self.prefilter_plan.num_constraints];

        let src_collapsed_idx = self.src_collapse.as_ref().map(|s| s.constraint_idx);
        let dst_collapsed_idx = self.dst_collapse.as_ref().map(|s| s.constraint_idx);
        let last_constraint_idx = self.prefilter_plan.num_constraints.saturating_sub(1);
        let dst_is_last = dst_collapsed_idx.map(|idx| idx == last_constraint_idx).unwrap_or(false);

        if let Some(src_positions) = src_driver_positions {
            if let Some(idx) = src_collapsed_idx {
                allowed[idx] = Some(src_positions.to_vec());
            }
        }
        if let Some(dst_positions) = dst_driver_positions {
            if dst_is_last {
                if let Some(idx) = dst_collapsed_idx {
                    allowed[idx] = Some(dst_positions.to_vec());
                }
            }
        }

        let mut requirements_by_constraint: HashMap<usize, Vec<PositionRequirement>> = HashMap::new();

        for (req_idx, req) in self.prefilter_plan.edge_reqs.iter().enumerate() {
            if Some(req.constraint_idx) == src_collapsed_idx || Some(req.constraint_idx) == dst_collapsed_idx {
                continue;
            }

            requirements_by_constraint
                .entry(req.constraint_idx)
                .or_insert_with(Vec::new)
                .push(PositionRequirement::Edge {
                    req_idx,
                    field: req.field,
                    label: req.label.clone(),
                });
        }

        let mut constraint_indices_with_reqs: HashSet<usize> = HashSet::new();

        for (req_idx, req) in self.constraint_reqs.iter().enumerate() {
            if Some(req.constraint_idx) == src_collapsed_idx || Some(req.constraint_idx) == dst_collapsed_idx {
                continue;
            }

            constraint_indices_with_reqs.insert(req.constraint_idx);

            requirements_by_constraint
                .entry(req.constraint_idx)
                .or_insert_with(Vec::new)
                .push(PositionRequirement::Constraint {
                    req_idx,
                    field: req.field,
                    term: req.term.clone(),
                });
        }

        let mut buf: Vec<u32> = Vec::with_capacity(32);

        for (constraint_idx, mut requirements) in requirements_by_constraint {
            requirements.sort_by_key(|req| match req {
                PositionRequirement::Edge { .. } => 0,
                PositionRequirement::Constraint { .. } => 1,
            });

            let mut edge_intersection: Option<Vec<u32>> = None;
            let mut constraint_union: Vec<u32> = Vec::with_capacity(64);
            let mut has_constraint_reqs = false;

            for req in requirements {
                buf.clear();

                match req {
                    PositionRequirement::Edge { req_idx, .. } => {
                        let postings_opt = self.edge_postings.get_mut(req_idx).and_then(|p| p.as_mut());

                        let postings = match postings_opt {
                            Some(p) => p,
                            None => {
                                return None;
                            }
                        };

                        if postings.doc() < doc_id {
                            postings.seek(doc_id);
                        }

                        if postings.doc() != doc_id {
                            return None;
                        }

                        postings.positions(&mut buf);

                        if buf.is_empty() {
                            return None;
                        }

                        match &mut edge_intersection {
                            None => {
                                edge_intersection = Some(std::mem::take(&mut buf));
                            }
                            Some(existing) => {
                                Self::intersect_sorted_in_place(existing, &buf);
                                if existing.is_empty() {
                                    return None;
                                }
                            }
                        }
                    }
                    PositionRequirement::Constraint { req_idx, .. } => {
                        has_constraint_reqs = true;

                        let postings_opt = self.constraint_postings.get_mut(req_idx).and_then(|p| p.as_mut());

                        let postings = match postings_opt {
                            Some(p) => p,
                            None => {
                                continue;
                            }
                        };

                        if postings.doc() < doc_id {
                            postings.seek(doc_id);
                        }

                        if postings.doc() != doc_id {
                            continue;
                        }

                        postings.positions(&mut buf);

                        if !buf.is_empty() {
                            if let Some(ref edge_positions) = edge_intersection {
                                let filtered = Self::intersect_sorted_slices(&buf, edge_positions);
                                if !filtered.is_empty() {
                                    constraint_union.extend_from_slice(&filtered);
                                }
                            } else {
                                constraint_union.extend_from_slice(&buf);
                            }
                        }
                    }
                }
            }

            if constraint_indices_with_reqs.contains(&constraint_idx) && constraint_union.is_empty() {
                return None;
            }

            let final_positions = if has_constraint_reqs {
                if !constraint_union.is_empty() {
                    constraint_union.sort_unstable();
                    constraint_union.dedup();
                }

                if constraint_union.is_empty() {
                    return None;
                }

                match edge_intersection {
                    None => {
                        constraint_union
                    }
                    Some(mut edge_positions) => {
                        Self::intersect_sorted_in_place(&mut edge_positions, &constraint_union);
                        if edge_positions.is_empty() {
                            return None;
                        }
                        edge_positions
                    }
                }
            } else {
                match edge_intersection {
                    None => {
                        continue;
                    }
                    Some(edge_positions) => edge_positions,
                }
            };

            allowed[constraint_idx] = Some(final_positions);
        }

        for constraint_idx in &constraint_indices_with_reqs {
            if Some(*constraint_idx) != src_collapsed_idx && Some(*constraint_idx) != dst_collapsed_idx {
                if allowed[*constraint_idx].is_none() || allowed[*constraint_idx].as_ref().unwrap().is_empty() {
                    return None;
                }
            }
        }

        Some(allowed)
    }

    /// Copy driver positions into reusable scratch buffers (avoids per-document heap allocation).
    /// Must be called before check_graph_traversal to populate src_positions_buf / dst_positions_buf.
    fn snapshot_driver_positions(&mut self) {
        self.src_positions_buf.clear();
        if let Some(positions) = self.intersection.src_driver().matching_positions() {
            self.src_positions_buf.extend_from_slice(positions);
        }
        self.dst_positions_buf.clear();
        if let Some(positions) = self.intersection.dst_driver().matching_positions() {
            self.dst_positions_buf.extend_from_slice(positions);
        }
    }

    /// Check if a document has valid graph traversal from source to destination
    pub(crate) fn check_graph_traversal(&mut self, doc_id: DocId) -> bool {
        let call_num = prof_inc!(CALL_COUNT);
        self.current_doc_matches.clear();

        let num_constraints = self.prefilter_plan.num_constraints;

        // Use scratch buffers (populated by snapshot_driver_positions before this call)
        let src_has_positions = !self.src_positions_buf.is_empty();
        let dst_has_positions = !self.dst_positions_buf.is_empty();
        let all_collapsed = num_constraints == 2
            && self.src_collapse.is_some()
            && self.dst_collapse.is_some()
            && src_has_positions
            && dst_has_positions;

        prof_inc!(PREFILTER_DOCS);

        // Phase A: compute allowed_positions.
        // compute_allowed_positions needs &mut self (for postings seek), so we must
        // take ownership of scratch buffer contents before the call to avoid borrow conflicts.
        // We use std::mem::take to move the data out of self without allocation,
        // then put it back afterward so the buffers are reused on the next call.
        let src_buf_owned = std::mem::take(&mut self.src_positions_buf);
        let dst_buf_owned = std::mem::take(&mut self.dst_positions_buf);

        let allowed_positions = if all_collapsed {
            prof_inc!(PREFILTER_SKIPPED_ALL_COLLAPSED);

            let mut allowed: Vec<Option<Vec<u32>>> = vec![None; num_constraints];
            if src_has_positions {
                allowed[0] = Some(src_buf_owned.clone());
            }
            if dst_has_positions {
                allowed[num_constraints - 1] = Some(dst_buf_owned.clone());
            }
            allowed
        } else {
            let src_slice = if src_has_positions { Some(src_buf_owned.as_slice()) } else { None };
            let dst_slice = if dst_has_positions { Some(dst_buf_owned.as_slice()) } else { None };
            match self.compute_allowed_positions(
                doc_id,
                src_slice,
                dst_slice,
            ) {
                Some(ap) => ap,
                None => {
                    // Put buffers back before returning
                    self.src_positions_buf = src_buf_owned;
                    self.dst_positions_buf = dst_buf_owned;
                    prof_inc!(PREFILTER_KILLED);
                    prof_inc!(GRAPH_DESER_SKIPPED);
                    return false;
                }
            }
        };

        // Put scratch buffers back into self (preserves allocation for next call)
        self.src_positions_buf = src_buf_owned;
        self.dst_positions_buf = dst_buf_owned;

        for ap in &allowed_positions {
            if let Some(ref positions) = ap {
                prof_inc!(PREFILTER_ALLOWED_POS_SUM, positions.len());
                prof_inc!(PREFILTER_ALLOWED_POS_COUNT);
            }
        }

        for ap in &allowed_positions {
            if let Some(ref positions) = ap {
                if positions.is_empty() {
                    prof_inc!(GRAPH_DESER_SKIPPED);
                    return false;
                }
            }
        }

        let flat_steps = &self.flat_steps;
        if flat_steps.is_empty() {
            prof_inc!(GRAPH_DESER_SKIPPED);
            return false;
        }

        let total_constraints = flat_steps.iter()
            .filter(|s| matches!(s, FlatPatternStep::Constraint(_)))
            .count();

        // Phase B: After the &mut self call is done, we can borrow scratch buffers immutably.
        let src_driver_positions: Option<&[u32]> = if src_has_positions { Some(&self.src_positions_buf) } else { None };
        let dst_driver_positions: Option<&[u32]> = if dst_has_positions { Some(&self.dst_positions_buf) } else { None };

        let allowed_positions_hashset = self.build_allowed_positions(
            src_driver_positions,
            dst_driver_positions,
            &allowed_positions,
            num_constraints,
        );

        let mut constraint_idx_counter = 0;
        let constraint_exact_flags: Vec<bool> = flat_steps.iter()
            .filter_map(|step| {
                if let FlatPatternStep::Constraint(constraint_pat) = step {
                    let unwrapped = self.unwrap_constraint_pattern(constraint_pat);
                    if let Pattern::Constraint(constraint) = unwrapped {
                        let is_exact = is_exact_skippable(constraint);
                        constraint_idx_counter += 1;
                        Some(is_exact)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        // OPTIMIZATION: Check if all constraints are fully covered by postings intersection.
        // If so, we can skip document store access entirely for token parsing.
        // IMPORTANT: Only skip token extraction if ALL constraints are exact matches,
        // because regex constraints still need token verification even with postings positions.
        let all_constraints_covered = self.constraints_covered_by_postings.len() == total_constraints
            && self.constraints_covered_by_postings.iter().all(|&covered| covered)
            && constraint_exact_flags.len() == total_constraints
            && constraint_exact_flags.iter().all(|&is_exact| is_exact)
            && (0..total_constraints).all(|idx| allowed_positions.get(idx).map(|p| p.is_some()).unwrap_or(false));

        // No need to load document - we use fast fields only for both constraint tokens and graph bytes
        if all_constraints_covered {
            prof_inc!(TOKEN_EXTRACTION_SKIPPED);
        }

        let mut constraint_count = 0;
        let mut cached_positions: Vec<Vec<usize>> = Vec::new();

        for step in flat_steps.iter() {
            if let FlatPatternStep::Constraint(constraint_pat) = step {
                if constraint_count >= self.constraint_field_names.len() {
                    prof_inc!(GRAPH_DESER_SKIPPED);
                    return false;
                }

                let unwrapped = self.unwrap_constraint_pattern(constraint_pat);

                let is_wildcard = matches!(
                    unwrapped,
                    Pattern::Constraint(Constraint::Wildcard)
                );

                let is_first_constraint = constraint_count == 0;
                let is_last_constraint = constraint_count == total_constraints - 1;

                // OPTIMIZATION: Check if this constraint is covered by postings and has positions
                let is_covered = self.constraints_covered_by_postings
                    .get(constraint_count)
                    .copied()
                    .unwrap_or(false);
                let has_postings_positions = allowed_positions
                    .get(constraint_count)
                    .map(|p| p.is_some())
                    .unwrap_or(false);

                // Fast path: Use postings positions directly without token parsing
                let positions = if is_covered && has_postings_positions {
                    // Positions come from postings intersection - no token parsing needed
                    if let Some(ref allowed) = allowed_positions[constraint_count] {
                        allowed.iter().map(|&p| p as usize).collect()
                    } else {
                        // Should not happen if has_postings_positions is true
                        Vec::new()
                    }
                } else if is_first_constraint && self.src_collapse.is_some() {
                    // Src collapse provides positions directly
                    if let Some(ref allowed) = allowed_positions[constraint_count] {
                        allowed.iter().map(|&p| p as usize).collect()
                    } else {
                        // Use fast field for token parsing
                        let tokens = extract_constraint_tokens_from_fast_field(
                            constraint_count,
                            &self.constraint_field_names[constraint_count],
                            &self.constraint_fast_fields,
                            doc_id,
                        );
                        self.find_positions_in_tokens(&tokens, constraint_pat)
                    }
                } else if is_last_constraint && self.dst_collapse.is_some() {
                    // Dst collapse provides positions directly
                    if let Some(ref allowed) = allowed_positions[constraint_count] {
                        allowed.iter().map(|&p| p as usize).collect()
                    } else {
                        // Use fast field for token parsing
                        let tokens = extract_constraint_tokens_from_fast_field(
                            constraint_count,
                            &self.constraint_field_names[constraint_count],
                            &self.constraint_fast_fields,
                            doc_id,
                        );
                        self.find_positions_in_tokens(&tokens, constraint_pat)
                    }
                } else if is_wildcard {
                    // Wildcards use edge positions or all positions
                    if let Some(ref allowed) = allowed_positions[constraint_count] {
                        allowed.iter().map(|&p| p as usize).collect()
                    } else {
                        // Need to know token count for wildcard - use fast field
                        let tokens = extract_constraint_tokens_from_fast_field(
                            constraint_count,
                            &self.constraint_field_names[constraint_count],
                            &self.constraint_fast_fields,
                            doc_id,
                        );
                        (0..tokens.len()).collect()
                    }
                } else if let Some(ref allowed) = allowed_positions[constraint_count] {
                    // Have prefilter positions but constraint not fully covered - need token verification
                    let tokens = extract_constraint_tokens_from_fast_field(
                        constraint_count,
                        &self.constraint_field_names[constraint_count],
                        &self.constraint_fast_fields,
                        doc_id,
                    );
                    self.find_positions_in_tokens_limited(&tokens, constraint_pat, allowed)
                } else {
                    // No prefilter positions - full token scan using fast field
                    let tokens = extract_constraint_tokens_from_fast_field(
                        constraint_count,
                        &self.constraint_field_names[constraint_count],
                        &self.constraint_fast_fields,
                        doc_id,
                    );
                    self.find_positions_in_tokens(&tokens, constraint_pat)
                };

                if positions.is_empty() {
                    prof_inc!(GRAPH_DESER_SKIPPED);
                    return false;
                }
                cached_positions.push(positions);
                constraint_count += 1;
            }
        }

        // Get graph bytes from fast field (O(1) columnar access) - no fallback
        // This avoids decompressing the entire document just to get the graph bytes
        let graph_bytes_buffer: Vec<u8>;
        let binary_data: &[u8] = if let Some(ref bytes_col) = self.dependencies_fast_field {
            let mut buffer = Vec::new();
            let mut term_ords = bytes_col.term_ords(doc_id);
            if let Some(ord) = term_ords.next() {
                match bytes_col.ord_to_bytes(ord, &mut buffer) {
                    Ok(true) => {
                        graph_bytes_buffer = buffer;
                        &graph_bytes_buffer
                    }
                    _ => {
                        prof_inc!(GRAPH_DESER_SKIPPED);
                        return false;
                    }
                }
            } else {
                prof_inc!(GRAPH_DESER_SKIPPED);
                return false;
            }
        } else {
            // Fast field not available - cannot proceed without fallback
            prof_inc!(GRAPH_DESER_SKIPPED);
            return false;
        };

        prof_inc!(GRAPH_DESER_COUNT);

        let src_positions_slice: &[usize] = cached_positions.get(0).map(|v| v.as_slice()).unwrap_or(&[]);
        if constraint_count > 0 && src_positions_slice.is_empty() {
            return false;
        }

        let src_positions_iter: Box<dyn Iterator<Item = usize> + '_> = if let Some(iter) = self.intersection.src_driver().matching_positions_iter() {
            Box::new(iter.map(|p| p as usize))
        } else {
            Box::new(src_positions_slice.iter().copied())
        };

        let estimated_size = src_positions_slice.len();
        let constraint_field_names = &self.constraint_field_names;

        if !ZeroCopyGraph::is_valid_format(binary_data) {
            prof_inc!(GRAPH_DESER_SKIPPED);
            return false;
        }

        let traversal_result = match ZeroCopyGraph::from_bytes(binary_data) {
            Ok(zc_graph) => {
                if estimated_size >= PARALLEL_START_POSITIONS_THRESHOLD {
                    let src_positions: Vec<usize> = src_positions_iter.collect();

                    // OPTIMIZATION: Build destination set once, share across all parallel threads
                    // Odinson-style: O(1) destination validation at traversal endpoints
                    let dst_set: HashSet<u32> = dst_driver_positions
                        .map(|positions| positions.iter().copied().collect())
                        .unwrap_or_default();

                    // OPTIMIZATION: Pre-build destination inverted index once, share across threads
                    let dst_index: Arc<HashMap<u32, Vec<crate::types::SpanWithCaptures>>> =
                        Arc::new(if let Some(positions) = dst_driver_positions {
                            build_dst_inverted_index(positions, flat_steps)
                        } else {
                            HashMap::new()
                        });

                    // OPTIMIZATION: Use EmptyTokenAccessor when all constraints are covered by postings
                    let all_matches: Vec<crate::types::SpanWithCaptures> = if all_constraints_covered {
                        let empty_accessor = EmptyTokenAccessor::new(total_constraints);
                        src_positions
                            .par_iter()
                            .flat_map(|&src_pos| {
                                process_single_start_position_with_index(
                                    binary_data,
                                    flat_steps,
                                    src_pos,
                                    constraint_field_names,
                                    &empty_accessor,
                                    &allowed_positions_hashset,
                                    &constraint_exact_flags,
                                    &dst_set,
                                    Some(&dst_index),
                                )
                            })
                            .collect()
                    } else {
                        // Need actual tokens - use fast fields to create token accessor
                        let mut lazy_tokens = LazyConstraintTokens::new(
                            &self.constraint_field_names,
                            self.reader.schema(),
                            &self.constraint_fast_fields,
                            doc_id,
                        );
                        for constraint_idx in 0..constraint_field_names.len() {
                            lazy_tokens.ensure_loaded(constraint_idx);
                        }
                        let token_accessor = ImmutableTokenAccessor::new(&lazy_tokens);

                        src_positions
                            .par_iter()
                            .flat_map(|&src_pos| {
                                process_single_start_position_with_index(
                                    binary_data,
                                    flat_steps,
                                    src_pos,
                                    constraint_field_names,
                                    &token_accessor,
                                    &allowed_positions_hashset,
                                    &constraint_exact_flags,
                                    &dst_set,
                                    Some(&dst_index),
                                )
                            })
                            .collect()
                    };

                    if !all_matches.is_empty() {
                        self.current_doc_matches.extend(all_matches);
                        return true;
                    }
                } else {
                    // OPTIMIZATION: Odinson-style batch traversal with destination index
                    // Instead of O(S×E) per-source traversals, do O(E+S) total:
                    // 1. Pre-build destination set for O(1) endpoint validation
                    // 2. Single batch traversal from all sources
                    // 3. O(1) destination check at pattern completion

                    let traversal_engine = crate::digraph::traversal::GraphTraversal::new(zc_graph);

                    // Create token getter closure - returns None when all constraints covered
                    let schema = self.reader.schema();
                    let field_names = &self.constraint_field_names;
                    let mut lazy_tokens_opt: Option<LazyConstraintTokens> = if all_constraints_covered {
                        None
                    } else {
                        Some(LazyConstraintTokens::new(
                            field_names,
                            schema,
                            &self.constraint_fast_fields,
                            doc_id,
                        ))
                    };

                    let mut get_token = |constraint_idx: usize, position: usize| -> Option<String> {
                        if let Some(ref mut tokens) = lazy_tokens_opt {
                            tokens.get(constraint_idx, position)
                        } else {
                            None
                        }
                    };

                    // Collect all source positions for batch processing
                    let src_positions: Vec<usize> = src_positions_iter.collect();

                    // Build destination set for O(1) membership check during traversal
                    // If dst_driver has positions, use them; otherwise use empty set (no filtering)
                    let dst_set: HashSet<u32> = dst_driver_positions
                        .map(|positions| positions.iter().copied().collect())
                        .unwrap_or_default();

                    // OPTIMIZATION: Pre-build destination inverted index for O(1) span lookup
                    let dst_index: HashMap<u32, Vec<crate::types::SpanWithCaptures>> = 
                        if let Some(positions) = dst_driver_positions {
                            build_dst_inverted_index(positions, flat_steps)
                        } else {
                            HashMap::new()
                        };

                    // Single batch traversal returning (src, dst, path) tuples
                    let reachable_pairs = traversal_engine.automaton_find_reachable_destinations(
                        flat_steps,
                        &src_positions,
                        &dst_set,
                        constraint_field_names,
                        &mut get_token,
                        &allowed_positions_hashset,
                        &constraint_exact_flags,
                    );

                    // Convert to SpanWithCaptures using inverted index for destination + path for intermediates
                    let mut all_matches = Vec::with_capacity(reachable_pairs.len());
                    
                    // Count total constraints to identify destination constraint index
                    let total_constraints: usize = flat_steps.iter()
                        .filter(|s| matches!(s, FlatPatternStep::Constraint(_)))
                        .count();
                    let dst_constraint_idx = total_constraints.saturating_sub(1);
                    
                    for (src_pos, dst_pos, path) in &reachable_pairs {
                        if path.is_empty() {
                            continue;
                        }

                        // Build captures for all constraints in the path
                        let mut captures = Vec::with_capacity(path.len());
                        let mut c_idx = 0;
                        let mut dst_capture_added = false;
                        
                        for step in flat_steps.iter() {
                            if let FlatPatternStep::Constraint(ref pat) = step {
                                if let Some(&node_idx) = path.get(c_idx) {
                                    // Check if this is the destination constraint (last constraint in pattern)
                                    let is_dst_constraint = c_idx == dst_constraint_idx && 
                                        node_idx == *dst_pos;
                                    
                                    if is_dst_constraint && !dst_index.is_empty() {
                                        // Use pre-built destination span from inverted index (O(1) lookup)
                                        if let Some(dst_spans) = dst_index.get(&(*dst_pos as u32)) {
                                            // Extract captures from pre-built destination span
                                            if let Some(dst_span) = dst_spans.first() {
                                                captures.extend(dst_span.captures.iter().cloned());
                                                dst_capture_added = true;
                                            }
                                        }
                                    }
                                    
                                    // If we didn't use inverted index, or for intermediate constraints,
                                    // build capture from path position
                                    if !is_dst_constraint || !dst_capture_added {
                                        let span = crate::types::Span { start: node_idx, end: node_idx + 1 };
                                        let name = match pat {
                                            Pattern::NamedCapture { name, .. } => name.clone(),
                                            _ => capture_name(c_idx),
                                        };
                                        captures.push(crate::types::NamedCapture::new(name, span));
                                    }
                                }
                                c_idx += 1;
                            }
                        }
                        
                        let min_pos = *path.iter().min().unwrap();
                        let max_pos = *path.iter().max().unwrap();
                        all_matches.push(
                            crate::types::SpanWithCaptures::with_captures(
                                crate::types::Span { start: min_pos, end: max_pos + 1 },
                                captures
                            )
                        );
                    }

                    if !all_matches.is_empty() {
                        self.current_doc_matches.extend(all_matches);
                        return true;
                    }
                }

                false
            }
            Err(_e) => {
                prof_inc!(GRAPH_DESER_SKIPPED);
                false
            }
        };

        if call_num > 0 && call_num % 100 == 0 {
            // Logging disabled for performance
        }

        traversal_result
    }

    #[allow(dead_code)]
    fn get_field_name_from_pattern<'a>(&self, pattern: &'a Pattern) -> &'a str {
        match pattern {
            Pattern::Constraint(Constraint::Field { name, .. }) => {
                name.as_str()
            }
            _ => "word",
        }
    }

    #[allow(dead_code)]
    fn extract_tokens_from_field(&self, doc: &tantivy::schema::TantivyDocument, field_name: &str) -> Vec<String> {
        crate::tantivy_integration::utils::extract_field_values(self.reader.schema(), doc, field_name)
    }

    #[allow(dead_code)]
    fn find_positions_matching_pattern(&self, tokens: &[String], pattern: &Pattern) -> Vec<usize> {
        self.find_positions_in_tokens(tokens, pattern)
    }

    pub fn get_current_doc_matches(&self) -> &[crate::types::SpanWithCaptures] {
        &self.current_doc_matches
    }

    #[allow(dead_code)]
    fn traversal_to_pattern(&self) -> Pattern {
        Pattern::GraphTraversal {
            src: Box::new(self.src_pattern.clone()),
            traversal: self.traversal.clone(),
            dst: Box::new(self.dst_pattern.clone()),
        }
    }

    /// Find positions in tokens that match a given pattern (string, regex, or wildcard for any field)
    pub(crate) fn find_positions_in_tokens(&self, tokens: &[String], pattern: &Pattern) -> Vec<usize> {
        let pattern = self.unwrap_constraint_pattern(pattern);

        let mut positions = Vec::new();
        match pattern {
            Pattern::Constraint(Constraint::Field { name: _, matcher }) => {
                match matcher {
                    Matcher::String(s) => {
                        for (i, token) in tokens.iter().enumerate() {
                            if token == s {
                                positions.push(i);
                            }
                        }
                    }
                    Matcher::Regex { regex, .. } => {
                        for (i, token) in tokens.iter().enumerate() {
                            if regex.is_match(token) {
                                positions.push(i);
                            }
                        }
                    }
                }
            }
            Pattern::Constraint(Constraint::Wildcard) => {
                positions = (0..tokens.len()).collect();
            }
            _ => {}
        }
        positions
    }

    /// Find positions in tokens that match a pattern, restricted to allowed positions
    pub(crate) fn find_positions_in_tokens_limited(
        &self,
        tokens: &[String],
        pattern: &Pattern,
        allowed: &[u32],
    ) -> Vec<usize> {
        let pattern = self.unwrap_constraint_pattern(pattern);

        let mut positions = Vec::new();

        match pattern {
            Pattern::Constraint(Constraint::Field { name: _, matcher }) => {
                match matcher {
                    Matcher::String(s) => {
                        for &pos in allowed {
                            let pos_usize = pos as usize;
                            if pos_usize < tokens.len() && tokens[pos_usize] == *s {
                                positions.push(pos_usize);
                            }
                        }
                    }
                    Matcher::Regex { regex, .. } => {
                        for &pos in allowed {
                            let pos_usize = pos as usize;
                            if pos_usize < tokens.len() && regex.is_match(&tokens[pos_usize]) {
                                positions.push(pos_usize);
                            }
                        }
                    }
                }
            }
            Pattern::Constraint(Constraint::Wildcard) => {
                positions = allowed.iter().map(|&p| p as usize).collect();
            }
            _ => {}
        }
        positions
    }
}

impl Scorer for OptimizedGraphTraversalScorer {
    fn score(&mut self) -> Score {
        if let Some((_, score)) = self.current_matches.get(self.match_index) {
            *score
        } else {
            Score::default()
        }
    }
}

impl DocSet for OptimizedGraphTraversalScorer {
    /// Advance to the next matching document using TwoPhaseIntersection.
    ///
    /// This implements the Lucene TwoPhaseIterator pattern:
    /// - Phase 1: TwoPhaseIntersection.advance() finds candidates where both drivers align
    /// - Phase 2: check_graph_traversal() verifies actual graph path exists
    fn advance(&mut self) -> DocId {
        loop {
            // Phase 1: Get next candidate from TwoPhaseIntersection
            let candidate = self.intersection.advance();

            if candidate == tantivy::TERMINATED {
                self.current_doc = None;
                self.log_final_stats();
                return tantivy::TERMINATED;
            }

            // Statistics tracking
            prof_inc!(DRIVER_ALIGNMENT_DOCS);

            if let Some(src_pos) = self.intersection.src_driver().matching_positions() {
                prof_inc!(DRIVER_INTERSECTION_SUM, src_pos.len());
                prof_inc!(DRIVER_INTERSECTION_COUNT);
            }
            if let Some(dst_pos) = self.intersection.dst_driver().matching_positions() {
                prof_inc!(DRIVER_INTERSECTION_SUM, dst_pos.len());
                prof_inc!(DRIVER_INTERSECTION_COUNT);
            }

            // Snapshot driver positions into scratch buffers before mutable check_graph_traversal
            self.snapshot_driver_positions();

            // Phase 2: Expensive verification - check if graph traversal succeeds
            if self.check_graph_traversal(candidate) {
                self.current_doc = Some(candidate);
                let score = self.compute_odinson_score();
                self.current_matches.push((candidate, score));
                self.match_index = self.current_matches.len() - 1;
                return candidate;
            }
            // Verification failed, continue to next candidate
        }
    }

    /// Seek to the first matching document >= target using TwoPhaseIntersection.
    ///
    /// This implements the Lucene TwoPhaseIterator pattern with a target hint:
    /// - Phase 1: TwoPhaseIntersection.seek() finds candidates >= target where both drivers align
    /// - Phase 2: check_graph_traversal() verifies actual graph path exists
    fn seek(&mut self, target: DocId) -> DocId {
        let current = self.doc();
        if current != tantivy::TERMINATED && current >= target {
            return current;
        }

        // Phase 1: Seek to first candidate >= target
        let mut candidate = self.intersection.seek(target);

        loop {
            if candidate == tantivy::TERMINATED {
                self.current_doc = None;
                return tantivy::TERMINATED;
            }

            // Snapshot driver positions into scratch buffers
            self.snapshot_driver_positions();

            // Phase 2: Expensive verification - check if graph traversal succeeds
            if self.check_graph_traversal(candidate) {
                self.current_doc = Some(candidate);
                let score = self.compute_odinson_score();
                self.current_matches.push((candidate, score));
                self.match_index = self.current_matches.len() - 1;
                return candidate;
            }

            // Verification failed, advance to next candidate
            candidate = self.intersection.advance_past_current();
        }
    }

    fn doc(&self) -> DocId {
        self.current_doc.unwrap_or(tantivy::TERMINATED)
    }

    fn size_hint(&self) -> u32 {
        0
    }
}
