//! Graph traversal module for dependency parsing.
//!
//! This module provides graph traversal operations for pattern matching
//! on dependency graphs. It supports various traversal patterns including
//! wildcards, label-specific traversals, concatenation, disjunction,
//! optional, and Kleene star operations.
//!
//! # Module Structure
//!
//! - [`types`] - Core types: `TraversalResult` and constants
//! - [`basic`] - Basic traversal operations: wildcards and label traversals
//! - [`patterns`] - Pattern composition: concatenation, disjunction, optional, kleene star
//! - [`query`] - Graph query utilities: shortest path, reachability
//! - [`matcher`] - Label matcher optimization for efficient traversal
//! - [`automaton`] - Automaton-based traversal engine with memoization
//!
//! # Example
//!
//! ```ignore
//! use crate::digraph::{DirectedGraph, GraphTraversal, TraversalResult};
//! use crate::query::ast::{Traversal, Matcher};
//!
//! let mut graph = DirectedGraph::new();
//! graph.add_edge(0, 1, "nsubj");
//! graph.add_edge(1, 2, "prep");
//!
//! let traversal = GraphTraversal::new(graph);
//! let result = traversal.execute(&Traversal::OutgoingWildcard, &[0]);
//! ```

mod types;
mod basic;
mod patterns;
mod query;
pub(crate) mod matcher;
mod automaton;

#[cfg(test)]
mod tests;

// Re-export public API
pub use types::{TraversalResult, PARALLEL_START_POSITIONS_THRESHOLD};
pub use basic::GraphTraversal;
