//! Scorer implementation for graph traversal queries.
//!
//! This module contains the `OptimizedGraphTraversalScorer` which performs
//! the actual document matching and scoring during query execution.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use rayon::prelude::*;
use tantivy::{
    query::Scorer,
    schema::{Field, Value},
    DocId, Score,
    DocSet, SegmentReader,
    store::StoreReader,
    postings::{SegmentPostings, Postings},
};

use crate::query::ast::{FlatPatternStep, Pattern, Constraint, Matcher};
use crate::digraph::zero_copy::ZeroCopyGraph;
use crate::digraph::traversal::PARALLEL_START_POSITIONS_THRESHOLD;

use super::types::{
    CAPTURE_COUNTER, CALL_COUNT, GRAPH_DESER_COUNT, GRAPH_DESER_SKIPPED,
    PREFILTER_DOCS, PREFILTER_KILLED, PREFILTER_ALLOWED_POS_SUM, PREFILTER_ALLOWED_POS_COUNT,
    DST_DRIVER_DOCS, DRIVER_ALIGNMENT_DOCS,
    DRIVER_INTERSECTION_SUM, DRIVER_INTERSECTION_COUNT,
    PREFILTER_SKIPPED_ALL_COLLAPSED,
    ConstraintTermReq, PositionPrefilterPlan, PositionRequirement, CollapsedSpec,
};
use super::candidate_driver::CandidateDriver;

/// Optimized scorer for graph traversal queries
/// Uses CandidateDriver abstraction for Odinson-style collapsed query optimization
pub struct OptimizedGraphTraversalScorer {
    /// Source candidate driver (may be CombinedPositionDriver or GenericDriver)
    pub(crate) src_driver: Box<dyn CandidateDriver>,
    /// Destination candidate driver (may be CombinedPositionDriver or GenericDriver)
    pub(crate) dst_driver: Box<dyn CandidateDriver>,
    #[allow(dead_code)]
    pub(crate) traversal: crate::query::ast::Traversal,
    pub(crate) dependencies_binary_field: Field,
    pub(crate) reader: SegmentReader,
    /// Cached store reader (created once, reused for all documents)
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
    ) -> Self {
        Self {
            src_driver,
            dst_driver,
            traversal,
            dependencies_binary_field,
            reader,
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
        }
    }
}

/// Lazy constraint token loader - parses constraint fields only when first accessed
/// This avoids parsing constraint fields that are never reached during graph traversal
pub(crate) struct LazyConstraintTokens<'a> {
    doc: &'a tantivy::schema::TantivyDocument,
    constraint_field_names: &'a [String],
    schema: &'a tantivy::schema::Schema,
    cache: Vec<Option<Vec<String>>>,
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

    /// Get token at a specific position for a constraint (returns owned String)
    /// Returns None if constraint not loaded or position out of bounds
    pub fn get(&self, constraint_idx: usize, position: usize) -> Option<String> {
        self.tokens
            .get(constraint_idx)?
            .and_then(|tokens| tokens.get(position))
            .cloned()
    }
}

impl<'a> LazyConstraintTokens<'a> {
    /// Create a new lazy token loader
    pub fn new(
        doc: &'a tantivy::schema::TantivyDocument,
        constraint_field_names: &'a [String],
        schema: &'a tantivy::schema::Schema,
    ) -> Self {
        let cache = vec![None; constraint_field_names.len()];
        Self {
            doc,
            constraint_field_names,
            schema,
            cache,
        }
    }

    /// Ensure tokens for a constraint are loaded (parses on first access)
    pub fn ensure_loaded(&mut self, constraint_idx: usize) {
        if constraint_idx >= self.cache.len() {
            return;
        }

        if self.cache[constraint_idx].is_none() {
            let field_name = &self.constraint_field_names[constraint_idx];
            let tokens = self.extract_tokens_from_field(field_name);
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

    /// Extract tokens from a field (reuses existing logic)
    fn extract_tokens_from_field(&self, field_name: &str) -> Vec<String> {
        crate::tantivy_integration::utils::extract_field_values(self.schema, self.doc, field_name)
    }
}

/// Process a single start position and return all matching paths as spans
/// This is used for parallel processing where each thread processes one start position
/// graph_bytes: The raw bytes of the graph (zero-copy, can be shared across threads)
pub(crate) fn process_single_start_position(
    graph_bytes: &[u8],
    flat_steps: &[FlatPatternStep],
    start_pos: usize,
    constraint_field_names: &[String],
    token_accessor: &ImmutableTokenAccessor,
    allowed_positions: &[Option<HashSet<u32>>],
    constraint_exact_flags: &[bool],
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

    let all_paths = traversal_engine.automaton_query_paths(
        flat_steps,
        &[start_pos],
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
                            _ => format!("c{}", CAPTURE_COUNTER.fetch_add(1, Ordering::Relaxed)),
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
                                    _ => format!("c{}", CAPTURE_COUNTER.fetch_add(1, Ordering::Relaxed)),
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

    /// Build allowed positions combining src_driver, dst_driver, and prefilter positions
    /// Converts all to HashSet<u32> for O(1) lookup
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
                result[0] = Some(src_positions.iter().copied().collect());
            }
        }

        if let Some(dst_positions) = dst_driver_positions {
            let last_idx = num_constraints.saturating_sub(1);
            if last_idx > 0 && last_idx < num_constraints {
                result[last_idx] = Some(dst_positions.iter().copied().collect());
            }
        }

        for (idx, prefilter) in prefilter_positions.iter().enumerate() {
            if idx < num_constraints {
                if result[idx].is_none() {
                    if let Some(positions) = prefilter {
                        result[idx] = Some(positions.iter().copied().collect());
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

    /// Check if a document has valid graph traversal from source to destination
    pub(crate) fn check_graph_traversal(&mut self, doc_id: DocId) -> bool {
        let call_num = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        self.current_doc_matches.clear();

        let num_constraints = self.prefilter_plan.num_constraints;

        // Optimization: Check if all constraints are collapsed first (avoids cloning in fast path)
        // We need to check driver positions availability before deciding the path
        let src_has_positions = self.src_driver.matching_positions().is_some();
        let dst_has_positions = self.dst_driver.matching_positions().is_some();
        let all_collapsed = num_constraints == 2
            && self.src_collapse.is_some()
            && self.dst_collapse.is_some()
            && src_has_positions
            && dst_has_positions;

        PREFILTER_DOCS.fetch_add(1, Ordering::Relaxed);

        // Clone positions once - needed for both branches due to mutable borrow in compute_allowed_positions
        // In the all_collapsed branch, this is the only clone needed
        // In the non-collapsed branch, compute_allowed_positions needs &mut self, so we must clone first
        let src_driver_positions: Option<Vec<u32>> = self.src_driver.matching_positions().map(|p| p.to_vec());
        let dst_driver_positions: Option<Vec<u32>> = self.dst_driver.matching_positions().map(|p| p.to_vec());

        let allowed_positions = if all_collapsed {
            PREFILTER_SKIPPED_ALL_COLLAPSED.fetch_add(1, Ordering::Relaxed);

            let mut allowed: Vec<Option<Vec<u32>>> = vec![None; num_constraints];
            // Move positions directly instead of cloning again
            if let Some(positions) = src_driver_positions.clone() {
                allowed[0] = Some(positions);
            }
            if let Some(positions) = dst_driver_positions.clone() {
                allowed[num_constraints - 1] = Some(positions);
            }
            allowed
        } else {
            match self.compute_allowed_positions(
                doc_id,
                src_driver_positions.as_deref(),
                dst_driver_positions.as_deref(),
            ) {
                Some(ap) => ap,
                None => {
                    PREFILTER_KILLED.fetch_add(1, Ordering::Relaxed);
                    GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                    return false;
                }
            }
        };

        for ap in &allowed_positions {
            if let Some(ref positions) = ap {
                PREFILTER_ALLOWED_POS_SUM.fetch_add(positions.len(), Ordering::Relaxed);
                PREFILTER_ALLOWED_POS_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }

        for ap in &allowed_positions {
            if let Some(ref positions) = ap {
                if positions.is_empty() {
                    GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                    return false;
                }
            }
        }

        let flat_steps = &self.flat_steps;
        if flat_steps.is_empty() {
            GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let total_constraints = flat_steps.iter()
            .filter(|s| matches!(s, FlatPatternStep::Constraint(_)))
            .count();

        let doc = match self.store_reader.get(doc_id) {
            Ok(doc) => doc,
            Err(_) => return false,
        };

        let mut lazy_tokens = LazyConstraintTokens::new(
            &doc,
            &self.constraint_field_names,
            self.reader.schema(),
        );

        let allowed_positions_hashset = self.build_allowed_positions(
            src_driver_positions.as_deref(),
            dst_driver_positions.as_deref(),
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

        let mut constraint_count = 0;
        let mut cached_positions: Vec<Vec<usize>> = Vec::new();

        for step in flat_steps.iter() {
            if let FlatPatternStep::Constraint(constraint_pat) = step {
                if constraint_count >= self.constraint_field_names.len() {
                    GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                    return false;
                }

                let tokens = match lazy_tokens.get_all_tokens(constraint_count) {
                    Some(t) => t,
                    None => {
                        GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                };

                let unwrapped = self.unwrap_constraint_pattern(constraint_pat);

                let is_wildcard = matches!(
                    unwrapped,
                    Pattern::Constraint(Constraint::Wildcard)
                );

                let is_first_constraint = constraint_count == 0;
                let is_last_constraint = constraint_count == total_constraints - 1;

                let positions = if is_first_constraint && self.src_collapse.is_some() {
                    if let Some(ref allowed) = allowed_positions[constraint_count] {
                        allowed.iter().map(|&p| p as usize).collect()
                    } else {
                        self.find_positions_in_tokens(tokens, constraint_pat)
                    }
                } else if is_last_constraint && self.dst_collapse.is_some() {
                    if let Some(ref allowed) = allowed_positions[constraint_count] {
                        allowed.iter().map(|&p| p as usize).collect()
                    } else {
                        self.find_positions_in_tokens(tokens, constraint_pat)
                    }
                } else if is_wildcard {
                    if let Some(ref allowed) = allowed_positions[constraint_count] {
                        allowed.iter().map(|&p| p as usize).collect()
                    } else {
                        (0..tokens.len()).collect()
                    }
                } else if let Some(ref allowed) = allowed_positions[constraint_count] {
                    self.find_positions_in_tokens_limited(tokens, constraint_pat, allowed)
                } else {
                    self.find_positions_in_tokens(tokens, constraint_pat)
                };

                if positions.is_empty() {
                    GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                    return false;
                }
                cached_positions.push(positions);
                constraint_count += 1;
            }
        }

        let binary_data = match doc.get_first(self.dependencies_binary_field).and_then(|v| v.as_bytes()) {
            Some(data) => data,
            None => {
                GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                return false;
            }
        };

        GRAPH_DESER_COUNT.fetch_add(1, Ordering::Relaxed);

        let src_positions_slice: &[usize] = cached_positions.get(0).map(|v| v.as_slice()).unwrap_or(&[]);
        if constraint_count > 0 && src_positions_slice.is_empty() {
            return false;
        }

        let src_positions_iter: Box<dyn Iterator<Item = usize> + '_> = if let Some(iter) = self.src_driver.matching_positions_iter() {
            Box::new(iter.map(|p| p as usize))
        } else {
            Box::new(src_positions_slice.iter().copied())
        };

        let estimated_size = src_positions_slice.len();
        let constraint_field_names = &self.constraint_field_names;

        if !ZeroCopyGraph::is_valid_format(binary_data) {
            GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let traversal_result = match ZeroCopyGraph::from_bytes(binary_data) {
            Ok(zc_graph) => {
                if estimated_size >= PARALLEL_START_POSITIONS_THRESHOLD {
                    let src_positions: Vec<usize> = src_positions_iter.collect();

                    for constraint_idx in 0..constraint_field_names.len() {
                        lazy_tokens.ensure_loaded(constraint_idx);
                    }

                    let token_accessor = ImmutableTokenAccessor::new(&lazy_tokens);

                    let all_matches: Vec<crate::types::SpanWithCaptures> = src_positions
                        .par_iter()
                        .flat_map(|&src_pos| {
                            process_single_start_position(
                                binary_data,
                                flat_steps,
                                src_pos,
                                constraint_field_names,
                                &token_accessor,
                                &allowed_positions_hashset,
                                &constraint_exact_flags,
                            )
                        })
                        .collect();

                    if !all_matches.is_empty() {
                        self.current_doc_matches.extend(all_matches);
                        return true;
                    }
                } else {
                    let traversal_engine = crate::digraph::traversal::GraphTraversal::new(zc_graph);
                    let mut get_token = |constraint_idx: usize, position: usize| -> Option<String> {
                        lazy_tokens.get(constraint_idx, position)
                    };

                    let mut all_matches = Vec::new();

                    for src_pos in src_positions_iter {
                        let all_paths = traversal_engine.automaton_query_paths(
                            flat_steps,
                            &[src_pos],
                            constraint_field_names,
                            &mut get_token,
                            &allowed_positions_hashset,
                            &constraint_exact_flags,
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
                                                _ => format!("c{}", CAPTURE_COUNTER.fetch_add(1, Ordering::Relaxed)),
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
                    }

                    if !all_matches.is_empty() {
                        self.current_doc_matches.extend(all_matches);
                        return true;
                    }
                }

                false
            }
            Err(_e) => {
                GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
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
    fn advance(&mut self) -> DocId {
        let current_doc = self.doc();

        let mut candidate = if current_doc == tantivy::TERMINATED {
            self.src_driver.seek(0)
        } else {
            self.src_driver.advance()
        };

        loop {
            if candidate == tantivy::TERMINATED {
                self.current_doc = None;
                self.log_final_stats();
                return tantivy::TERMINATED;
            }

            let dst_current = self.dst_driver.doc();

            const SEEK_THRESHOLD: DocId = 10;
            let dst_doc = if dst_current < candidate {
                let gap = candidate - dst_current;
                if gap >= SEEK_THRESHOLD {
                    self.dst_driver.seek(candidate)
                } else {
                    let mut doc = dst_current;
                    while doc < candidate {
                        doc = self.dst_driver.advance();
                        if doc == tantivy::TERMINATED {
                            break;
                        }
                    }
                    doc
                }
            } else {
                dst_current
            };

            if dst_doc == tantivy::TERMINATED {
                self.current_doc = None;
                self.log_final_stats();
                return tantivy::TERMINATED;
            }

            if dst_doc > candidate {
                DST_DRIVER_DOCS.fetch_add(1, Ordering::Relaxed);
                let gap = dst_doc - candidate;

                const SEEK_THRESHOLD: DocId = 10;
                if gap >= SEEK_THRESHOLD {
                    candidate = self.src_driver.seek(dst_doc);
                } else {
                    while candidate < dst_doc {
                        candidate = self.src_driver.advance();
                        if candidate == tantivy::TERMINATED {
                            break;
                        }
                    }
                }
                continue;
            }

            debug_assert_eq!(candidate, dst_doc);
            DRIVER_ALIGNMENT_DOCS.fetch_add(1, Ordering::Relaxed);

            if let Some(src_pos) = self.src_driver.matching_positions() {
                DRIVER_INTERSECTION_SUM.fetch_add(src_pos.len(), Ordering::Relaxed);
                DRIVER_INTERSECTION_COUNT.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(dst_pos) = self.dst_driver.matching_positions() {
                DRIVER_INTERSECTION_SUM.fetch_add(dst_pos.len(), Ordering::Relaxed);
                DRIVER_INTERSECTION_COUNT.fetch_add(1, Ordering::Relaxed);
            }

            if self.check_graph_traversal(candidate) {
                self.current_doc = Some(candidate);
                let score = self.compute_odinson_score();
                self.current_matches.push((candidate, score));
                self.match_index = self.current_matches.len() - 1;
                return candidate;
            }
            candidate = self.src_driver.advance();
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        let current = self.doc();
        if current != tantivy::TERMINATED && current >= target {
            return current;
        }

        let mut candidate = self.src_driver.seek(target);

        loop {
            if candidate == tantivy::TERMINATED {
                self.current_doc = None;
                return tantivy::TERMINATED;
            }

            let dst_doc = self.dst_driver.seek(candidate);

            if dst_doc == tantivy::TERMINATED {
                self.current_doc = None;
                return tantivy::TERMINATED;
            }

            if dst_doc > candidate {
                const SEEK_THRESHOLD: DocId = 10;
                let gap = dst_doc - candidate;
                if gap >= SEEK_THRESHOLD {
                    candidate = self.src_driver.seek(dst_doc);
                } else {
                    while candidate < dst_doc {
                        candidate = self.src_driver.advance();
                        if candidate == tantivy::TERMINATED {
                            break;
                        }
                    }
                }
                continue;
            }

            if self.check_graph_traversal(candidate) {
                self.current_doc = Some(candidate);
                let score = self.compute_odinson_score();
                self.current_matches.push((candidate, score));
                self.match_index = self.current_matches.len() - 1;
                return candidate;
            }

            candidate = self.src_driver.advance();
        }
    }

    fn doc(&self) -> DocId {
        self.current_doc.unwrap_or(tantivy::TERMINATED)
    }

    fn size_hint(&self) -> u32 {
        0
    }
}
