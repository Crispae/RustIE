//! Automaton-based graph traversal engine.
//!
//! This module implements the core automaton traversal algorithm with:
//! - Cycle detection via memoization
//! - Pre-computed label matchers for O(1) matching
//! - Lazy token loading for constraint evaluation
//! - Support for all traversal patterns (wildcard, exact, regex, optional, kleene star, etc.)

use std::collections::HashSet;

use crate::query::ast::{Traversal, FlatPatternStep};
use crate::digraph::graph_trait::GraphAccess;
use super::GraphTraversal;
use super::matcher::ResolvedTraversalMatcher;

impl<G: GraphAccess> GraphTraversal<G> {
    /// Automaton-based traversal with early fail and memoization.
    ///
    /// NOTE: This function is deprecated - use automaton_query_paths or automaton_query instead.
    /// Kept for backward compatibility but requires default empty parameters.
    pub fn automaton_traverse<F>(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        step_idx: usize,
        memo: &mut Vec<bool>,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
    ) -> bool
    where
        F: FnMut(usize, usize) -> Option<String>,
    {
        let mut dummy_path = Vec::new();
        let mut dummy_results = Vec::new();
        // Pre-compute matchers once
        let resolved_matchers = self.precompute_matchers(pattern);
        self.automaton_traverse_paths_optimized(
            pattern, node, step_idx, memo, constraint_field_names, get_token,
            allowed_positions, constraint_exact_flags, &resolved_matchers,
            &mut dummy_path, &mut dummy_results
        );
        !dummy_results.is_empty()
    }

    /// OPTIMIZED: Traverse and collect all token index paths for matches.
    ///
    /// Uses pre-computed label matchers for O(1) matching instead of repeated HashMap lookups.
    /// Uses closure for lazy token loading and skips matches() for exact prefilter-confirmed constraints.
    pub(crate) fn automaton_traverse_paths_optimized<F>(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        step_idx: usize,
        memo: &mut Vec<bool>,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
        resolved_matchers: &[Option<ResolvedTraversalMatcher>],
        path: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
    ) where
        F: FnMut(usize, usize) -> Option<String>,
    {
        let num_steps = pattern.len();
        let idx = node * num_steps + step_idx;

        // Terminal condition: reached end of pattern
        if step_idx == pattern.len() {
            results.push(path.clone());
            return;
        }

        // Cycle detection: skip if already visiting this node at this step in current path
        if memo[idx] {
            return;
        }
        memo[idx] = true; // Mark as visiting

        match &pattern[step_idx] {
            FlatPatternStep::Constraint(constraint_pat) => {
                self.handle_constraint_step(
                    pattern, node, step_idx, idx, memo, constraint_field_names,
                    get_token, allowed_positions, constraint_exact_flags,
                    resolved_matchers, path, results, constraint_pat
                );
            }
            FlatPatternStep::Traversal(traversal) => {
                self.handle_traversal_step(
                    pattern, node, step_idx, idx, memo, constraint_field_names,
                    get_token, allowed_positions, constraint_exact_flags,
                    resolved_matchers, path, results, traversal
                );
            }
        }

        memo[idx] = false; // Backtrack: allow other paths through this node
    }

    /// Handle a constraint step in the automaton traversal.
    fn handle_constraint_step<F>(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        step_idx: usize,
        idx: usize,
        memo: &mut Vec<bool>,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
        resolved_matchers: &[Option<ResolvedTraversalMatcher>],
        path: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
        constraint_pat: &crate::query::ast::Pattern,
    ) where
        F: FnMut(usize, usize) -> Option<String>,
    {
        // Count the number of Constraint steps in pattern[0..step_idx] to get the correct index
        // into constraint_fields_and_tokens (which only contains Constraint entries, not Traversals)
        let constraint_idx = pattern[..step_idx].iter()
            .filter(|s| matches!(s, FlatPatternStep::Constraint(_)))
            .count();

        if constraint_idx >= constraint_field_names.len() {
            memo[idx] = false; // Backtrack
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
                        memo[idx] = false; // Backtrack
                        return;
                    }
                };
                let matches = constraint.matches(field_name, &token);
                if !matches {
                    memo[idx] = false; // Backtrack
                    return;
                }
            }
        } else {
            memo[idx] = false; // Backtrack
            return;
        }

        path.push(node);
        self.automaton_traverse_paths_optimized(
            pattern, node, step_idx + 1, memo, constraint_field_names,
            get_token, allowed_positions, constraint_exact_flags,
            resolved_matchers, path, results
        );
        path.pop();
    }

    /// Handle a traversal step in the automaton traversal.
    fn handle_traversal_step<F>(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        step_idx: usize,
        _idx: usize,
        memo: &mut Vec<bool>,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
        resolved_matchers: &[Option<ResolvedTraversalMatcher>],
        path: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
        traversal: &Traversal,
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
                self.handle_optional_traversal(
                    pattern, node, step_idx, memo, constraint_field_names,
                    get_token, allowed_positions, constraint_exact_flags,
                    resolved_matchers, path, results, inner_traversal
                );
            }
            Traversal::Outgoing(_) => {
                if let Some(edges) = self.graph.outgoing(node) {
                    for (target_node, label_id) in edges {
                        if self.matches_label(&resolved_matcher, label_id) {
                            self.automaton_traverse_paths_optimized(
                                pattern, target_node, step_idx + 1, memo, constraint_field_names,
                                get_token, allowed_positions, constraint_exact_flags,
                                resolved_matchers, path, results
                            );
                        }
                    }
                }
            }
            Traversal::Incoming(_) => {
                if let Some(edges) = self.graph.incoming(node) {
                    for (source_node, label_id) in edges {
                        if self.matches_label(&resolved_matcher, label_id) {
                            self.automaton_traverse_paths_optimized(
                                pattern, source_node, step_idx + 1, memo, constraint_field_names,
                                get_token, allowed_positions, constraint_exact_flags,
                                resolved_matchers, path, results
                            );
                        }
                    }
                }
            }
            Traversal::OutgoingWildcard => {
                if let Some(edges) = self.graph.outgoing(node) {
                    for (target_node, _label_id) in edges {
                        self.automaton_traverse_paths_optimized(
                            pattern, target_node, step_idx + 1, memo, constraint_field_names,
                            get_token, allowed_positions, constraint_exact_flags,
                            resolved_matchers, path, results
                        );
                    }
                }
            }
            Traversal::IncomingWildcard => {
                if let Some(edges) = self.graph.incoming(node) {
                    for (source_node, _label_id) in edges {
                        self.automaton_traverse_paths_optimized(
                            pattern, source_node, step_idx + 1, memo, constraint_field_names,
                            get_token, allowed_positions, constraint_exact_flags,
                            resolved_matchers, path, results
                        );
                    }
                }
            }
            Traversal::Disjunctive(alternatives) => {
                self.handle_disjunctive_traversal(
                    pattern, node, step_idx, memo, constraint_field_names,
                    get_token, allowed_positions, constraint_exact_flags,
                    resolved_matchers, path, results, alternatives
                );
            }
            Traversal::Concatenated(steps) => {
                self.execute_concatenated_in_automaton(
                    pattern, node, step_idx, steps, 0, memo, constraint_field_names,
                    get_token, allowed_positions, constraint_exact_flags,
                    resolved_matchers, path, results
                );
            }
            Traversal::KleeneStar(inner) => {
                // Kleene star: zero or more repetitions
                // First, try zero repetitions (skip to next step)
                self.automaton_traverse_paths_optimized(
                    pattern, node, step_idx + 1, memo, constraint_field_names,
                    get_token, allowed_positions, constraint_exact_flags,
                    resolved_matchers, path, results
                );

                // Then try one or more repetitions
                let inner_resolved = self.resolve_traversal_matcher(inner);
                self.execute_kleene_star_in_automaton(
                    pattern, node, step_idx, inner, &inner_resolved, memo,
                    constraint_field_names, get_token, allowed_positions,
                    constraint_exact_flags, resolved_matchers, path, results,
                    &mut HashSet::new()
                );
            }
            Traversal::NoTraversal => {
                // No traversal means stay at current node, advance step
                self.automaton_traverse_paths_optimized(
                    pattern, node, step_idx + 1, memo, constraint_field_names,
                    get_token, allowed_positions, constraint_exact_flags,
                    resolved_matchers, path, results
                );
            }
        }
    }

    /// Handle optional traversal within the automaton.
    fn handle_optional_traversal<F>(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        step_idx: usize,
        memo: &mut Vec<bool>,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
        resolved_matchers: &[Option<ResolvedTraversalMatcher>],
        path: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
        inner_traversal: &Box<Traversal>,
    ) where
        F: FnMut(usize, usize) -> Option<String>,
    {
        // Try skipping the traversal
        self.automaton_traverse_paths_optimized(
            pattern, node, step_idx + 1, memo, constraint_field_names,
            get_token, allowed_positions, constraint_exact_flags,
            resolved_matchers, path, results
        );

        // Try taking the traversal - use inner traversal's matcher
        let inner_resolved = self.resolve_traversal_matcher(inner_traversal);
        match &**inner_traversal {
            Traversal::Outgoing(_) => {
                if let Some(edges) = self.graph.outgoing(node) {
                    for (target_node, label_id) in edges {
                        if self.matches_label(&inner_resolved, label_id) {
                            self.automaton_traverse_paths_optimized(
                                pattern, target_node, step_idx + 1, memo, constraint_field_names,
                                get_token, allowed_positions, constraint_exact_flags,
                                resolved_matchers, path, results
                            );
                        }
                    }
                }
            }
            Traversal::Incoming(_) => {
                if let Some(edges) = self.graph.incoming(node) {
                    for (source_node, label_id) in edges {
                        if self.matches_label(&inner_resolved, label_id) {
                            self.automaton_traverse_paths_optimized(
                                pattern, source_node, step_idx + 1, memo, constraint_field_names,
                                get_token, allowed_positions, constraint_exact_flags,
                                resolved_matchers, path, results
                            );
                        }
                    }
                }
            }
            Traversal::OutgoingWildcard => {
                if let Some(edges) = self.graph.outgoing(node) {
                    for (target_node, _label_id) in edges {
                        self.automaton_traverse_paths_optimized(
                            pattern, target_node, step_idx + 1, memo, constraint_field_names,
                            get_token, allowed_positions, constraint_exact_flags,
                            resolved_matchers, path, results
                        );
                    }
                }
            }
            Traversal::IncomingWildcard => {
                if let Some(edges) = self.graph.incoming(node) {
                    for (source_node, _label_id) in edges {
                        self.automaton_traverse_paths_optimized(
                            pattern, source_node, step_idx + 1, memo, constraint_field_names,
                            get_token, allowed_positions, constraint_exact_flags,
                            resolved_matchers, path, results
                        );
                    }
                }
            }
            _ => {}
        }
    }

    /// Handle disjunctive traversal within the automaton.
    fn handle_disjunctive_traversal<F>(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        step_idx: usize,
        memo: &mut Vec<bool>,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
        resolved_matchers: &[Option<ResolvedTraversalMatcher>],
        path: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
        alternatives: &[Traversal],
    ) where
        F: FnMut(usize, usize) -> Option<String>,
    {
        // Union of all alternative traversals - try each and collect all results
        for alt in alternatives {
            let alt_resolved = self.resolve_traversal_matcher(alt);
            match alt {
                Traversal::Outgoing(_) => {
                    if let Some(edges) = self.graph.outgoing(node) {
                        for (target_node, label_id) in edges {
                            if self.matches_label(&alt_resolved, label_id) {
                                self.automaton_traverse_paths_optimized(
                                    pattern, target_node, step_idx + 1, memo, constraint_field_names,
                                    get_token, allowed_positions, constraint_exact_flags,
                                    resolved_matchers, path, results
                                );
                            }
                        }
                    }
                }
                Traversal::Incoming(_) => {
                    if let Some(edges) = self.graph.incoming(node) {
                        for (source_node, label_id) in edges {
                            if self.matches_label(&alt_resolved, label_id) {
                                self.automaton_traverse_paths_optimized(
                                    pattern, source_node, step_idx + 1, memo, constraint_field_names,
                                    get_token, allowed_positions, constraint_exact_flags,
                                    resolved_matchers, path, results
                                );
                            }
                        }
                    }
                }
                Traversal::OutgoingWildcard => {
                    if let Some(edges) = self.graph.outgoing(node) {
                        for (target_node, _label_id) in edges {
                            self.automaton_traverse_paths_optimized(
                                pattern, target_node, step_idx + 1, memo, constraint_field_names,
                                get_token, allowed_positions, constraint_exact_flags,
                                resolved_matchers, path, results
                            );
                        }
                    }
                }
                Traversal::IncomingWildcard => {
                    if let Some(edges) = self.graph.incoming(node) {
                        for (source_node, _label_id) in edges {
                            self.automaton_traverse_paths_optimized(
                                pattern, source_node, step_idx + 1, memo, constraint_field_names,
                                get_token, allowed_positions, constraint_exact_flags,
                                resolved_matchers, path, results
                            );
                        }
                    }
                }
                // Nested disjunctive/concatenated handled by recursive resolve
                _ => {}
            }
        }
    }

    /// Helper: Execute concatenated traversal steps within the automaton.
    pub(crate) fn execute_concatenated_in_automaton<F>(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        outer_step_idx: usize,
        concat_steps: &[Traversal],
        concat_idx: usize,
        memo: &mut Vec<bool>,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
        resolved_matchers: &[Option<ResolvedTraversalMatcher>],
        path: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
    ) where
        F: FnMut(usize, usize) -> Option<String>,
    {
        // If we've completed all concatenated steps, continue to next pattern step
        if concat_idx >= concat_steps.len() {
            self.automaton_traverse_paths_optimized(
                pattern, node, outer_step_idx + 1, memo, constraint_field_names,
                get_token, allowed_positions, constraint_exact_flags,
                resolved_matchers, path, results
            );
            return;
        }

        let current_trav = &concat_steps[concat_idx];
        let trav_resolved = self.resolve_traversal_matcher(current_trav);

        match current_trav {
            Traversal::Outgoing(_) => {
                if let Some(edges) = self.graph.outgoing(node) {
                    for (target_node, label_id) in edges {
                        if self.matches_label(&trav_resolved, label_id) {
                            self.execute_concatenated_in_automaton(
                                pattern, target_node, outer_step_idx, concat_steps, concat_idx + 1,
                                memo, constraint_field_names, get_token, allowed_positions,
                                constraint_exact_flags, resolved_matchers, path, results
                            );
                        }
                    }
                }
            }
            Traversal::Incoming(_) => {
                if let Some(edges) = self.graph.incoming(node) {
                    for (source_node, label_id) in edges {
                        if self.matches_label(&trav_resolved, label_id) {
                            self.execute_concatenated_in_automaton(
                                pattern, source_node, outer_step_idx, concat_steps, concat_idx + 1,
                                memo, constraint_field_names, get_token, allowed_positions,
                                constraint_exact_flags, resolved_matchers, path, results
                            );
                        }
                    }
                }
            }
            Traversal::OutgoingWildcard => {
                if let Some(edges) = self.graph.outgoing(node) {
                    for (target_node, _label_id) in edges {
                        self.execute_concatenated_in_automaton(
                            pattern, target_node, outer_step_idx, concat_steps, concat_idx + 1,
                            memo, constraint_field_names, get_token, allowed_positions,
                            constraint_exact_flags, resolved_matchers, path, results
                        );
                    }
                }
            }
            Traversal::IncomingWildcard => {
                if let Some(edges) = self.graph.incoming(node) {
                    for (source_node, _label_id) in edges {
                        self.execute_concatenated_in_automaton(
                            pattern, source_node, outer_step_idx, concat_steps, concat_idx + 1,
                            memo, constraint_field_names, get_token, allowed_positions,
                            constraint_exact_flags, resolved_matchers, path, results
                        );
                    }
                }
            }
            // For nested complex traversals, fall back to the basic execute approach
            _ => {}
        }
    }

    /// Helper: Execute Kleene star traversal within the automaton.
    pub(crate) fn execute_kleene_star_in_automaton<F>(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        outer_step_idx: usize,
        inner: &Traversal,
        inner_resolved: &ResolvedTraversalMatcher,
        memo: &mut Vec<bool>,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
        resolved_matchers: &[Option<ResolvedTraversalMatcher>],
        path: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
        visited: &mut HashSet<usize>,
    ) where
        F: FnMut(usize, usize) -> Option<String>,
    {
        // Prevent infinite loops
        if visited.contains(&node) {
            return;
        }
        visited.insert(node);

        // Try traversing once and then either continue to next step or repeat
        match inner {
            Traversal::Outgoing(_) => {
                if let Some(edges) = self.graph.outgoing(node) {
                    for (target_node, label_id) in edges {
                        if self.matches_label(inner_resolved, label_id) {
                            // After one traversal, try completing (go to next step)
                            self.automaton_traverse_paths_optimized(
                                pattern, target_node, outer_step_idx + 1, memo,
                                constraint_field_names, get_token, allowed_positions,
                                constraint_exact_flags, resolved_matchers, path, results
                            );
                            // Also try repeating
                            self.execute_kleene_star_in_automaton(
                                pattern, target_node, outer_step_idx, inner, inner_resolved, memo,
                                constraint_field_names, get_token, allowed_positions,
                                constraint_exact_flags, resolved_matchers, path, results, visited
                            );
                        }
                    }
                }
            }
            Traversal::Incoming(_) => {
                if let Some(edges) = self.graph.incoming(node) {
                    for (source_node, label_id) in edges {
                        if self.matches_label(inner_resolved, label_id) {
                            self.automaton_traverse_paths_optimized(
                                pattern, source_node, outer_step_idx + 1, memo,
                                constraint_field_names, get_token, allowed_positions,
                                constraint_exact_flags, resolved_matchers, path, results
                            );
                            self.execute_kleene_star_in_automaton(
                                pattern, source_node, outer_step_idx, inner, inner_resolved, memo,
                                constraint_field_names, get_token, allowed_positions,
                                constraint_exact_flags, resolved_matchers, path, results, visited
                            );
                        }
                    }
                }
            }
            Traversal::OutgoingWildcard => {
                if let Some(edges) = self.graph.outgoing(node) {
                    for (target_node, _label_id) in edges {
                        self.automaton_traverse_paths_optimized(
                            pattern, target_node, outer_step_idx + 1, memo,
                            constraint_field_names, get_token, allowed_positions,
                            constraint_exact_flags, resolved_matchers, path, results
                        );
                        self.execute_kleene_star_in_automaton(
                            pattern, target_node, outer_step_idx, inner, inner_resolved, memo,
                            constraint_field_names, get_token, allowed_positions,
                            constraint_exact_flags, resolved_matchers, path, results, visited
                        );
                    }
                }
            }
            Traversal::IncomingWildcard => {
                if let Some(edges) = self.graph.incoming(node) {
                    for (source_node, _label_id) in edges {
                        self.automaton_traverse_paths_optimized(
                            pattern, source_node, outer_step_idx + 1, memo,
                            constraint_field_names, get_token, allowed_positions,
                            constraint_exact_flags, resolved_matchers, path, results
                        );
                        self.execute_kleene_star_in_automaton(
                            pattern, source_node, outer_step_idx, inner, inner_resolved, memo,
                            constraint_field_names, get_token, allowed_positions,
                            constraint_exact_flags, resolved_matchers, path, results, visited
                        );
                    }
                }
            }
            _ => {}
        }
    }

    /// Legacy wrapper for backward compatibility.
    ///
    /// NOTE: This function is deprecated - use automaton_query_paths or automaton_query instead.
    /// Kept for backward compatibility but requires default empty parameters.
    pub fn automaton_traverse_paths<F>(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        step_idx: usize,
        memo: &mut Vec<bool>,
        constraint_field_names: &[String],
        get_token: &mut F,
        allowed_positions: &[Option<HashSet<u32>>],
        constraint_exact_flags: &[bool],
        path: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
    ) where
        F: FnMut(usize, usize) -> Option<String>,
    {
        // Pre-compute matchers and delegate to optimized version
        let resolved_matchers = self.precompute_matchers(pattern);
        self.automaton_traverse_paths_optimized(
            pattern, node, step_idx, memo, constraint_field_names,
            get_token, allowed_positions, constraint_exact_flags,
            &resolved_matchers, path, results
        );
    }

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
        let mut memo = vec![false; memo_size];
        let mut path = Vec::new();

        for &start_node in candidate_nodes {
            // Validate start_node is within bounds
            if start_node >= node_count {
                continue;
            }
            // Clear memo for this iteration (faster than reallocating)
            memo.fill(false);
            path.clear();
            self.automaton_traverse_paths_optimized(
                pattern, start_node, 0, &mut memo, constraint_field_names,
                get_token, allowed_positions, constraint_exact_flags,
                &resolved_matchers, &mut path, &mut all_results
            );
        }
        all_results
    }

    /// Wrapper: Run automaton traversal for all start nodes matching the first constraint.
    ///
    /// OPTIMIZED: Pre-computes all label matchers ONCE before iterating.
    /// Uses closure for lazy token loading and skips matches() for exact prefilter-confirmed constraints.
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
        let mut memo = vec![false; memo_size];
        let mut path = Vec::new();
        let mut results = Vec::new();

        for &start_node in candidate_nodes {
            // Validate start_node is within bounds
            if start_node >= node_count {
                continue;
            }
            // Clear memo for this iteration (faster than reallocating)
            memo.fill(false);
            path.clear();
            results.clear();
            self.automaton_traverse_paths_optimized(
                pattern, start_node, 0, &mut memo, constraint_field_names,
                get_token, allowed_positions, constraint_exact_flags,
                &resolved_matchers, &mut path, &mut results
            );
            if !results.is_empty() {
                return true;
            }
        }
        false
    }
}
