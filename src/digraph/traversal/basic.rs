//! Core graph traversal structure and basic traversal operations.
//!
//! This module provides the `GraphTraversal` struct and implements basic
//! traversal operations like wildcard and label-specific traversals.

use crate::query::ast::{Traversal, Matcher};
use crate::digraph::graph_trait::GraphAccess;
use super::TraversalResult;

/// Graph traversal engine for dependency parsing.
///
/// This implementation exactly matches the Scala GraphTraversal logic:
/// https://github.com/lum-ai/odinson/blob/master/core/src/main/scala/ai/lum/odinson/digraph/GraphTraversal.scala
pub struct GraphTraversal<G: GraphAccess> {
    pub(crate) graph: G,
}

impl<G: GraphAccess> GraphTraversal<G> {
    /// Create a new graph traversal engine.
    pub fn new(graph: G) -> Self {
        Self { graph }
    }

    /// Execute a traversal pattern.
    /// Mirrors the Scala GraphTraversal trait and its case classes.
    pub fn execute(&self, traversal: &Traversal, start_nodes: &[usize]) -> TraversalResult {

        match traversal {
            Traversal::NoTraversal => TraversalResult::NoTraversal,
            Traversal::OutgoingWildcard => self.outgoing_wildcard(start_nodes),
            Traversal::IncomingWildcard => self.incoming_wildcard(start_nodes),
            Traversal::Outgoing(matcher) => self.outgoing_traversal(start_nodes, matcher),
            Traversal::Incoming(matcher) => self.incoming_traversal(start_nodes, matcher),
            Traversal::Concatenated(traversals) => self.concatenated_traversal(start_nodes, traversals),
            Traversal::Disjunctive(traversals) => self.disjunctive_traversal(start_nodes, traversals),
            Traversal::Optional(traversal) => self.optional_traversal(start_nodes, traversal),
            Traversal::KleeneStar(traversal) => self.kleene_star_traversal(start_nodes, traversal),
        }
    }

    /// Traverse all outgoing edges.
    /// Scala: OutgoingWildcard
    pub(crate) fn outgoing_wildcard(&self, start_nodes: &[usize]) -> TraversalResult {
        let mut result_nodes = Vec::new();
        for &start_node in start_nodes {
            if let Some(edges) = self.graph.outgoing(start_node) {
                // Iterator returns (target_node, label_id) pairs
                for (target_node, _label_id) in edges {
                    result_nodes.push(target_node);
                }
            }
        }
        // Deduplicate results as in Scala
        result_nodes.sort_unstable();
        result_nodes.dedup();
        if result_nodes.is_empty() {
            TraversalResult::FailTraversal
        } else {
            TraversalResult::Success(result_nodes)
        }
    }

    /// Traverse all incoming edges.
    /// Scala: IncomingWildcard
    pub(crate) fn incoming_wildcard(&self, start_nodes: &[usize]) -> TraversalResult {
        let mut result_nodes = Vec::new();
        for &start_node in start_nodes {
            if let Some(edges) = self.graph.incoming(start_node) {
                // Iterator returns (source_node, label_id) pairs
                for (source_node, _label_id) in edges {
                    result_nodes.push(source_node);
                }
            }
        }
        // Deduplicate results as in Scala
        result_nodes.sort_unstable();
        result_nodes.dedup();
        if result_nodes.is_empty() {
            TraversalResult::FailTraversal
        } else {
            TraversalResult::Success(result_nodes)
        }
    }

    /// Traverse outgoing edges matching a specific label.
    /// OPTIMIZED: Pre-resolve matcher ONCE before iterating edges
    pub(crate) fn outgoing_traversal(&self, start_nodes: &[usize], matcher: &Matcher) -> TraversalResult {
        let mut result_nodes = Vec::new();

        // OPTIMIZATION: Pre-resolve label ID for exact matches (O(1) comparison)
        let pre_resolved_label_id = match matcher {
            Matcher::String(s) => self.graph.get_label_id(s),
            Matcher::Regex { .. } => None, // Will use regex matching per edge
        };

        for &start_node in start_nodes {
            if let Some(edges) = self.graph.outgoing(start_node) {
                // Iterator returns (target_node, label_id) pairs
                for (target_node, label_id) in edges {
                    // Fast path: exact match with pre-resolved ID
                    if let Some(expected_id) = pre_resolved_label_id {
                        if label_id == expected_id {
                            result_nodes.push(target_node);
                        }
                    } else {
                        // Regex path: need to get label string
                        if let Some(label_str) = self.graph.get_label(label_id) {
                            match matcher {
                                Matcher::String(s) => {
                                    if label_str == s {
                                        result_nodes.push(target_node);
                                    }
                                }
                                Matcher::Regex { regex, .. } => {
                                    if regex.is_match(label_str) {
                                        result_nodes.push(target_node);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Deduplicate results as in Scala
        result_nodes.sort_unstable();
        result_nodes.dedup();
        if result_nodes.is_empty() {
            TraversalResult::FailTraversal
        } else {
            TraversalResult::Success(result_nodes)
        }
    }

    /// Traverse incoming edges matching a specific label.
    /// OPTIMIZED: Pre-resolve matcher ONCE before iterating edges
    pub(crate) fn incoming_traversal(&self, start_nodes: &[usize], matcher: &Matcher) -> TraversalResult {
        let mut result_nodes = Vec::new();

        // OPTIMIZATION: Pre-resolve label ID for exact matches (O(1) comparison)
        let pre_resolved_label_id = match matcher {
            Matcher::String(s) => self.graph.get_label_id(s),
            Matcher::Regex { .. } => None, // Will use regex matching per edge
        };

        for &start_node in start_nodes {
            if let Some(edges) = self.graph.incoming(start_node) {
                // Iterator returns (source_node, label_id) pairs
                for (source_node, label_id) in edges {
                    // Fast path: exact match with pre-resolved ID
                    if let Some(expected_id) = pre_resolved_label_id {
                        if label_id == expected_id {
                            result_nodes.push(source_node);
                        }
                    } else {
                        // Regex path: need to get label string
                        if let Some(label_str) = self.graph.get_label(label_id) {
                            match matcher {
                                Matcher::String(s) => {
                                    if label_str == s {
                                        result_nodes.push(source_node);
                                    }
                                }
                                Matcher::Regex { regex, .. } => {
                                    if regex.is_match(label_str) {
                                        result_nodes.push(source_node);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Deduplicate results as in Scala
        result_nodes.sort_unstable();
        result_nodes.dedup();
        if result_nodes.is_empty() {
            TraversalResult::FailTraversal
        } else {
            TraversalResult::Success(result_nodes)
        }
    }
}
