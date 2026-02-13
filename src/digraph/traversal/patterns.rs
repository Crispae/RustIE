//! Pattern composition for complex graph traversals.
//!
//! This module implements composite traversal patterns: concatenation,
//! disjunction (union), optional, and Kleene star operations.

use std::collections::HashSet;

use crate::query::ast::Traversal;
use crate::digraph::graph_trait::GraphAccess;
use super::{GraphTraversal, TraversalResult};

impl<G: GraphAccess> GraphTraversal<G> {
    /// Concatenate multiple traversals (sequence).
    /// Scala: Concatenation(List[GraphTraversal])
    pub(crate) fn concatenated_traversal(&self, start_nodes: &[usize], traversals: &[Traversal]) -> TraversalResult {
        let mut current_nodes = start_nodes.to_vec();
        for traversal in traversals.iter() {
            match self.execute(traversal, &current_nodes) {
                TraversalResult::Success(nodes) => {
                    current_nodes = nodes;
                }
                TraversalResult::FailTraversal => {
                    return TraversalResult::FailTraversal;
                }
                TraversalResult::NoTraversal => {
                    // Continue with current nodes (no change)
                }
            }
        }
        current_nodes.sort_unstable();
        current_nodes.dedup();
        TraversalResult::Success(current_nodes)
    }

    /// Disjunctive traversal (union).
    /// Scala: Union(List[GraphTraversal])
    pub(crate) fn disjunctive_traversal(&self, start_nodes: &[usize], traversals: &[Traversal]) -> TraversalResult {
        let mut all_results = Vec::new();
        for traversal in traversals {
            match self.execute(traversal, start_nodes) {
                TraversalResult::Success(nodes) => {
                    all_results.extend(nodes);
                }
                TraversalResult::FailTraversal => {
                    // Continue trying other traversals
                }
                TraversalResult::NoTraversal => {
                    // Continue with current nodes
                    all_results.extend(start_nodes);
                }
            }
        }
        TraversalResult::from_nodes(all_results)
    }

    /// Optional traversal (0 or 1 occurrence).
    /// Scala: Optional(GraphTraversal)
    pub(crate) fn optional_traversal(&self, start_nodes: &[usize], traversal: &Traversal) -> TraversalResult {
        let mut nodes = match self.execute(traversal, start_nodes) {
            TraversalResult::Success(nodes) => nodes,
            _ => Vec::new(),
        };
        nodes.extend_from_slice(start_nodes);
        nodes.sort_unstable();
        nodes.dedup();
        TraversalResult::Success(nodes)
    }

    /// Kleene star traversal (0 or more occurrences).
    /// Scala: KleeneStar(GraphTraversal)
    pub(crate) fn kleene_star_traversal(&self, start_nodes: &[usize], traversal: &Traversal) -> TraversalResult {
        let mut all_nodes: HashSet<usize> = start_nodes.iter().cloned().collect();
        let mut current_nodes = start_nodes.to_vec();
        let mut visited = HashSet::new();

        loop {
            let key = format!("{:?}", current_nodes);
            if visited.contains(&key) {
                break; // Avoid infinite loops
            }
            visited.insert(key);

            match self.execute(traversal, &current_nodes) {
                TraversalResult::Success(nodes) => {
                    let mut new_nodes = false;
                    for &node in &nodes {
                        if all_nodes.insert(node) {
                            new_nodes = true;
                        }
                    }
                    if !new_nodes {
                        break; // No new nodes found
                    }
                    current_nodes = nodes;
                }
                TraversalResult::FailTraversal | TraversalResult::NoTraversal => {
                    break;
                }
            }
        }
        // Return all unique nodes reached
        let mut result: Vec<usize> = all_nodes.into_iter().collect();
        result.sort_unstable();
        TraversalResult::Success(result)
    }
}
