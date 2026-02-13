//! Automaton-based graph traversal engine.
//!
//! This module implements the core automaton traversal algorithm with:
//! - Cycle detection via memoization
//! - Pre-computed label matchers for O(1) matching
//! - Lazy token loading for constraint evaluation
//! - Support for all traversal patterns (wildcard, exact, regex, optional, kleene star, etc.)
//! - Optional destination-set filtering for batch traversal queries

use std::collections::HashSet;

use crate::query::ast::{Traversal, FlatPatternStep};
use crate::digraph::graph_trait::GraphAccess;
use super::GraphTraversal;
use super::matcher::ResolvedTraversalMatcher;

impl<G: GraphAccess> GraphTraversal<G> {

    // ---- Internal helpers ----

    /// Collect reachable neighbor nodes from `node` following the given traversal direction.
    ///
    /// Handles all four edge-direction variants (Outgoing, Incoming, OutgoingWildcard,
    /// IncomingWildcard) in a single place, applying the resolved label matcher for filtering.
    /// Returns a Vec of neighbor node IDs. For non-edge traversal variants, returns empty.
    fn collect_neighbors(
        &self,
        node: usize,
        traversal: &Traversal,
        resolved: &ResolvedTraversalMatcher,
    ) -> Vec<usize> {
        match traversal {
            Traversal::Outgoing(_) => {
                if let Some(edges) = self.graph.outgoing(node) {
                    edges.filter_map(|(target, label_id)| {
                        if self.matches_label(resolved, label_id) { Some(target) } else { None }
                    }).collect()
                } else {
                    Vec::new()
                }
            }
            Traversal::Incoming(_) => {
                if let Some(edges) = self.graph.incoming(node) {
                    edges.filter_map(|(source, label_id)| {
                        if self.matches_label(resolved, label_id) { Some(source) } else { None }
                    }).collect()
                } else {
                    Vec::new()
                }
            }
            Traversal::OutgoingWildcard => {
                if let Some(edges) = self.graph.outgoing(node) {
                    edges.map(|(target, _)| target).collect()
                } else {
                    Vec::new()
                }
            }
            Traversal::IncomingWildcard => {
                if let Some(edges) = self.graph.incoming(node) {
                    edges.map(|(source, _)| source).collect()
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        }
    }

    /// Core automaton traversal with early fail and memoization.
    ///
    /// Unified traversal function that supports both plain and destination-filtered modes:
    /// - `dst_set: None` — record all completed paths (no destination filtering)
    /// - `dst_set: Some(set)` — only record paths whose final node is in the destination set
    ///
    /// Uses pre-computed label matchers for O(1) matching instead of repeated HashMap lookups.
    /// Uses closure for lazy token loading and skips matches() for exact prefilter-confirmed constraints.
    /// `memo` uses generation counter: memo[idx] == generation means "currently visiting".
    fn automaton_traverse_core<F>(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        step_idx: usize,
        memo: &mut Vec<u32>,
        generation: u32,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
        resolved_matchers: &[Option<ResolvedTraversalMatcher>],
        path: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
        dst_set: Option<&HashSet<u32>>,
    ) where
        F: FnMut(usize, usize) -> Option<String>,
    {
        let num_steps = pattern.len();
        let idx = node * num_steps + step_idx;

        // Terminal condition: reached end of pattern
        if step_idx == num_steps {
            // When dst_set is None, always record the path.
            // When dst_set is Some(set), only record if final node is in the set (or set is empty).
            if dst_set.map_or(true, |ds| ds.is_empty() || ds.contains(&(node as u32))) {
                results.push(path.clone());
            }
            return;
        }

        // Cycle detection: skip if already visiting this node at this step in current path
        if memo[idx] == generation {
            return;
        }
        memo[idx] = generation; // Mark as visiting

        match &pattern[step_idx] {
            FlatPatternStep::Constraint(constraint_pat) => {
                self.handle_constraint_step(
                    pattern, node, step_idx, idx, memo, generation, constraint_field_names,
                    get_token, allowed_positions, constraint_exact_flags,
                    resolved_matchers, path, results, constraint_pat, dst_set
                );
            }
            FlatPatternStep::Traversal(traversal) => {
                self.handle_traversal_step(
                    pattern, node, step_idx, memo, generation, constraint_field_names,
                    get_token, allowed_positions, constraint_exact_flags,
                    resolved_matchers, path, results, traversal, dst_set
                );
            }
        }

        memo[idx] = 0; // Backtrack: allow other paths through this node
    }

    /// Handle a constraint step in the automaton traversal.
    fn handle_constraint_step<F>(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        step_idx: usize,
        idx: usize,
        memo: &mut Vec<u32>,
        generation: u32,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
        resolved_matchers: &[Option<ResolvedTraversalMatcher>],
        path: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
        constraint_pat: &crate::query::ast::Pattern,
        dst_set: Option<&HashSet<u32>>,
    ) where
        F: FnMut(usize, usize) -> Option<String>,
    {
        // Count the number of Constraint steps in pattern[0..step_idx] to get the correct index
        // into constraint_fields_and_tokens (which only contains Constraint entries, not Traversals)
        let constraint_idx = pattern[..step_idx].iter()
            .filter(|s| matches!(s, FlatPatternStep::Constraint(_)))
            .count();

        if constraint_idx >= constraint_field_names.len() {
            memo[idx] = 0; // Backtrack
            return;
        }

        let field_name = &constraint_field_names[constraint_idx];

        if let crate::query::ast::Pattern::Constraint(constraint) = constraint_pat {
            // Optimization: Check if exact constraint + prefilter confirmed
            let is_exact = constraint_exact_flags.get(constraint_idx).copied().unwrap_or(false);
            let prefilter_confirmed = allowed_positions.get(constraint_idx)
                .and_then(|opt| opt.as_ref())
                .map(|set| set.contains(&(node as u32)))  // O(1) HashSet lookup
                .unwrap_or(false);

            if is_exact && prefilter_confirmed {
                // Skip BOTH get_token() AND constraint.matches()
                // Postings already confirmed exact match - zero work needed
            } else {
                // Need token for matching (regex/wildcard) or verification
                let token = match get_token(constraint_idx, node) {
                    Some(t) => t,
                    None => {
                        memo[idx] = 0; // Backtrack
                        return;
                    }
                };
                let matches = constraint.matches(field_name, &token);
                if !matches {
                    memo[idx] = 0; // Backtrack
                    return;
                }
            }
        } else {
            memo[idx] = 0; // Backtrack
            return;
        }

        path.push(node);
        self.automaton_traverse_core(
            pattern, node, step_idx + 1, memo, generation, constraint_field_names,
            get_token, allowed_positions, constraint_exact_flags,
            resolved_matchers, path, results, dst_set
        );
        path.pop();
    }

    /// Handle a traversal step in the automaton traversal.
    ///
    /// Dispatches to the appropriate traversal strategy based on the traversal variant.
    /// Uses `collect_neighbors` to avoid duplicating the Outgoing/Incoming/Wildcard dispatch.
    fn handle_traversal_step<F>(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        step_idx: usize,
        memo: &mut Vec<u32>,
        generation: u32,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
        resolved_matchers: &[Option<ResolvedTraversalMatcher>],
        path: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
        traversal: &Traversal,
        dst_set: Option<&HashSet<u32>>,
    ) where
        F: FnMut(usize, usize) -> Option<String>,
    {
        // Get pre-computed matcher for this step (O(1) lookup)
        let resolved_matcher = resolved_matchers.get(step_idx)
            .and_then(|m| m.as_ref())
            .cloned()
            .unwrap_or(ResolvedTraversalMatcher::Wildcard);

        match traversal {
            Traversal::Optional(inner_traversal) => {
                // Try skipping the traversal
                self.automaton_traverse_core(
                    pattern, node, step_idx + 1, memo, generation, constraint_field_names,
                    get_token, allowed_positions, constraint_exact_flags,
                    resolved_matchers, path, results, dst_set
                );

                // Try taking the traversal - use inner traversal's matcher
                let inner_resolved = self.resolve_traversal_matcher(inner_traversal);
                let neighbors = self.collect_neighbors(node, inner_traversal, &inner_resolved);
                for neighbor in neighbors {
                    self.automaton_traverse_core(
                        pattern, neighbor, step_idx + 1, memo, generation, constraint_field_names,
                        get_token, allowed_positions, constraint_exact_flags,
                        resolved_matchers, path, results, dst_set
                    );
                }
            }
            Traversal::Outgoing(_) | Traversal::Incoming(_) |
            Traversal::OutgoingWildcard | Traversal::IncomingWildcard => {
                let neighbors = self.collect_neighbors(node, traversal, &resolved_matcher);
                for neighbor in neighbors {
                    self.automaton_traverse_core(
                        pattern, neighbor, step_idx + 1, memo, generation, constraint_field_names,
                        get_token, allowed_positions, constraint_exact_flags,
                        resolved_matchers, path, results, dst_set
                    );
                }
            }
            Traversal::Disjunctive(alternatives) => {
                // Union of all alternative traversals - try each and collect all results
                for alt in alternatives {
                    let alt_resolved = self.resolve_traversal_matcher(alt);
                    let neighbors = self.collect_neighbors(node, alt, &alt_resolved);
                    for neighbor in neighbors {
                        self.automaton_traverse_core(
                            pattern, neighbor, step_idx + 1, memo, generation, constraint_field_names,
                            get_token, allowed_positions, constraint_exact_flags,
                            resolved_matchers, path, results, dst_set
                        );
                    }
                }
            }
            Traversal::Concatenated(steps) => {
                self.execute_concatenated(
                    pattern, node, step_idx, steps, 0, memo, generation, constraint_field_names,
                    get_token, allowed_positions, constraint_exact_flags,
                    resolved_matchers, path, results, dst_set
                );
            }
            Traversal::KleeneStar(inner) => {
                // Kleene star: zero or more repetitions
                // First, try zero repetitions (skip to next step)
                self.automaton_traverse_core(
                    pattern, node, step_idx + 1, memo, generation, constraint_field_names,
                    get_token, allowed_positions, constraint_exact_flags,
                    resolved_matchers, path, results, dst_set
                );

                // Then try one or more repetitions
                let inner_resolved = self.resolve_traversal_matcher(inner);
                self.execute_kleene_star(
                    pattern, node, step_idx, inner, &inner_resolved, memo, generation,
                    constraint_field_names, get_token, allowed_positions,
                    constraint_exact_flags, resolved_matchers, path, results,
                    &mut HashSet::new(), dst_set
                );
            }
            Traversal::NoTraversal => {
                // No traversal means stay at current node, advance step
                self.automaton_traverse_core(
                    pattern, node, step_idx + 1, memo, generation, constraint_field_names,
                    get_token, allowed_positions, constraint_exact_flags,
                    resolved_matchers, path, results, dst_set
                );
            }
        }
    }

    /// Execute concatenated traversal steps within the automaton.
    fn execute_concatenated<F>(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        outer_step_idx: usize,
        concat_steps: &[Traversal],
        concat_idx: usize,
        memo: &mut Vec<u32>,
        generation: u32,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
        resolved_matchers: &[Option<ResolvedTraversalMatcher>],
        path: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
        dst_set: Option<&HashSet<u32>>,
    ) where
        F: FnMut(usize, usize) -> Option<String>,
    {
        // If we've completed all concatenated steps, continue to next pattern step
        if concat_idx >= concat_steps.len() {
            self.automaton_traverse_core(
                pattern, node, outer_step_idx + 1, memo, generation, constraint_field_names,
                get_token, allowed_positions, constraint_exact_flags,
                resolved_matchers, path, results, dst_set
            );
            return;
        }

        let current_trav = &concat_steps[concat_idx];
        let trav_resolved = self.resolve_traversal_matcher(current_trav);
        let neighbors = self.collect_neighbors(node, current_trav, &trav_resolved);

        for neighbor in neighbors {
            self.execute_concatenated(
                pattern, neighbor, outer_step_idx, concat_steps, concat_idx + 1,
                memo, generation, constraint_field_names, get_token, allowed_positions,
                constraint_exact_flags, resolved_matchers, path, results, dst_set
            );
        }
    }

    /// Execute Kleene star traversal within the automaton.
    fn execute_kleene_star<F>(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        outer_step_idx: usize,
        inner: &Traversal,
        inner_resolved: &ResolvedTraversalMatcher,
        memo: &mut Vec<u32>,
        generation: u32,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
        resolved_matchers: &[Option<ResolvedTraversalMatcher>],
        path: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
        visited: &mut HashSet<usize>,
        dst_set: Option<&HashSet<u32>>,
    ) where
        F: FnMut(usize, usize) -> Option<String>,
    {
        // Prevent infinite loops
        if visited.contains(&node) {
            return;
        }
        visited.insert(node);

        // Collect neighbors once, then try completing and repeating for each
        let neighbors = self.collect_neighbors(node, inner, inner_resolved);

        for neighbor in neighbors {
            // After one traversal, try completing (go to next step)
            self.automaton_traverse_core(
                pattern, neighbor, outer_step_idx + 1, memo, generation,
                constraint_field_names, get_token, allowed_positions,
                constraint_exact_flags, resolved_matchers, path, results, dst_set
            );
            // Also try repeating
            self.execute_kleene_star(
                pattern, neighbor, outer_step_idx, inner, inner_resolved, memo, generation,
                constraint_field_names, get_token, allowed_positions,
                constraint_exact_flags, resolved_matchers, path, results, visited, dst_set
            );
        }
    }

    // ---- Public API wrappers (signatures unchanged) ----

    /// Wrapper: Run automaton traversal for all start nodes matching the first constraint, collect all paths.
    ///
    /// OPTIMIZED: Pre-computes all label matchers ONCE before iterating.
    /// Uses closure for lazy token loading and skips matches() for exact prefilter-confirmed constraints.
    pub fn automaton_query_paths<F>(
        &self,
        pattern: &[FlatPatternStep],
        candidate_nodes: &[usize],
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
    ) -> Vec<Vec<usize>>
    where
        F: FnMut(usize, usize) -> Option<String>,
    {
        let mut all_results = Vec::new();
        // Use graph's node count for proper memo sizing (fixes potential out-of-bounds)
        let node_count = self.graph.node_count();
        if node_count == 0 || pattern.is_empty() {
            return all_results;
        }

        // OPTIMIZATION: Pre-compute all matchers ONCE before any traversal
        let resolved_matchers = self.precompute_matchers(pattern);

        // Pre-allocate memo buffer once, reuse across all start nodes
        let memo_size = node_count * pattern.len();
        let mut memo = vec![0u32; memo_size];
        let mut generation: u32 = 0;
        let mut path = Vec::new();

        for &start_node in candidate_nodes {
            if start_node >= node_count {
                continue;
            }
            generation = generation.wrapping_add(1);
            if generation == 0 { generation = 1; }
            path.clear();
            self.automaton_traverse_core(
                pattern, start_node, 0, &mut memo, generation, constraint_field_names,
                get_token, allowed_positions, constraint_exact_flags,
                &resolved_matchers, &mut path, &mut all_results, None
            );
        }
        all_results
    }

    /// Wrapper: Run automaton traversal for all start nodes matching the first constraint.
    ///
    /// OPTIMIZED: Pre-computes all label matchers ONCE before iterating.
    /// Uses closure for lazy token loading and skips matches() for exact prefilter-confirmed constraints.
    /// Returns true as soon as any match is found (early exit).
    pub fn automaton_query<F>(
        &self,
        pattern: &[FlatPatternStep],
        candidate_nodes: &[usize],
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
    ) -> bool
    where
        F: FnMut(usize, usize) -> Option<String>,
    {
        // Use graph's node count for proper memo sizing (fixes potential out-of-bounds)
        let node_count = self.graph.node_count();
        if node_count == 0 || pattern.is_empty() {
            return false;
        }

        // OPTIMIZATION: Pre-compute all matchers ONCE before any traversal
        let resolved_matchers = self.precompute_matchers(pattern);

        // Pre-allocate memo buffer once, reuse across all start nodes
        let memo_size = node_count * pattern.len();
        let mut memo = vec![0u32; memo_size];
        let mut generation: u32 = 0;
        let mut path = Vec::new();
        let mut results = Vec::new();

        for &start_node in candidate_nodes {
            if start_node >= node_count {
                continue;
            }
            // Increment generation instead of clearing memo (avoids memset)
            generation = generation.wrapping_add(1);
            if generation == 0 { generation = 1; } // Skip 0 (default value)
            path.clear();
            results.clear();
            self.automaton_traverse_core(
                pattern, start_node, 0, &mut memo, generation, constraint_field_names,
                get_token, allowed_positions, constraint_exact_flags,
                &resolved_matchers, &mut path, &mut results, None
            );
            if !results.is_empty() {
                return true;
            }
        }
        false
    }

    /// Odinson-style batch traversal: O(E) total instead of O(S*E).
    ///
    /// Key optimization: Instead of doing a separate traversal for each source position,
    /// this performs a single traversal from ALL source positions simultaneously.
    /// The destination set enables O(1) validation at path endpoints.
    ///
    /// Complexity comparison:
    /// - Original: For S sources, O(S * E) where E = edges traversed per source
    /// - Optimized: O(E + S) - single traversal + source iteration overhead
    pub fn automaton_query_paths_batch<F>(
        &self,
        pattern: &[FlatPatternStep],
        src_positions: &[usize],
        dst_set: &HashSet<u32>,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
    ) -> Vec<Vec<usize>>
    where
        F: FnMut(usize, usize) -> Option<String>,
    {
        let mut all_results = Vec::new();
        let node_count = self.graph.node_count();
        if node_count == 0 || pattern.is_empty() {
            return all_results;
        }

        // OPTIMIZATION: Pre-compute all matchers ONCE (shared across all sources)
        let resolved_matchers = self.precompute_matchers(pattern);

        // Single memo buffer, reused across all start nodes
        let memo_size = node_count * pattern.len();
        let mut memo = vec![0u32; memo_size];
        let mut generation: u32 = 0;
        let mut path = Vec::new();

        // Batch process all source positions
        for &start_node in src_positions {
            if start_node >= node_count {
                continue;
            }
            generation = generation.wrapping_add(1);
            if generation == 0 { generation = 1; }
            path.clear();

            self.automaton_traverse_core(
                pattern, start_node, 0, &mut memo, generation, constraint_field_names,
                get_token, allowed_positions, constraint_exact_flags,
                &resolved_matchers, &mut path, &mut all_results, Some(dst_set)
            );
        }
        all_results
    }

    /// Find reachable destinations from all source positions, returning (src, dst, path) tuples.
    /// This enables O(1) destination span lookup via inverted index while preserving
    /// full path information for intermediate constraint captures.
    ///
    /// Returns Vec<(usize, usize, Vec<usize>)> where:
    /// - First usize is source position
    /// - Second usize is destination position
    /// - Vec<usize> is the full path (for building intermediate captures)
    pub fn automaton_find_reachable_destinations<F>(
        &self,
        pattern: &[FlatPatternStep],
        src_positions: &[usize],
        dst_set: &HashSet<u32>,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
    ) -> Vec<(usize, usize, Vec<usize>)>
    where
        F: FnMut(usize, usize) -> Option<String>,
    {
        let mut all_results = Vec::new();
        let node_count = self.graph.node_count();
        if node_count == 0 || pattern.is_empty() {
            return all_results;
        }

        // OPTIMIZATION: Pre-compute all matchers ONCE (shared across all sources)
        let resolved_matchers = self.precompute_matchers(pattern);

        // Single memo buffer, reused across all start nodes
        let memo_size = node_count * pattern.len();
        let mut memo = vec![0u32; memo_size];
        let mut generation: u32 = 0;
        let mut path = Vec::new();
        let mut results_for_src = Vec::new();

        // Batch process all source positions
        for &start_node in src_positions {
            if start_node >= node_count {
                continue;
            }
            generation = generation.wrapping_add(1);
            if generation == 0 { generation = 1; }
            path.clear();
            results_for_src.clear();

            self.automaton_traverse_core(
                pattern, start_node, 0, &mut memo, generation, constraint_field_names,
                get_token, allowed_positions, constraint_exact_flags,
                &resolved_matchers, &mut path, &mut results_for_src, Some(dst_set)
            );

            // Convert paths to (src, dst, path) tuples
            for path_result in &results_for_src {
                if !path_result.is_empty() {
                    let dst = *path_result.last().unwrap();
                    all_results.push((start_node, dst, path_result.clone()));
                }
            }
        }
        all_results
    }
}
