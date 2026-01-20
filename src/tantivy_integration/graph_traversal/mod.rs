//! Graph traversal query implementation for Tantivy.
//!
//! This module provides Odinson-style collapsed query optimization for
//! efficient graph traversal queries over dependency parse graphs.
//!
//! # Architecture
//!
//! The module is organized into several submodules:
//!
//! - [`types`]: Core data structures and atomic counters for statistics
//! - [`logging`]: Debug and performance logging helpers
//! - [`intersection`]: Position intersection algorithms (galloping, linear)
//! - [`candidate_driver`]: CandidateDriver trait and implementations
//! - [`query`]: OptimizedGraphTraversalQuery implementation
//! - [`weight`]: OptimizedGraphTraversalWeight implementation
//! - [`scorer`]: OptimizedGraphTraversalScorer implementation
//! - [`pattern_utils`]: Pattern flattening and prefilter building
//! - [`stats`]: Performance statistics collection and reporting

pub mod types;
pub mod logging;
pub mod intersection;
pub mod candidate_driver;
pub mod query;
pub mod weight;
pub mod scorer;
pub mod pattern_utils;
pub mod stats;

#[cfg(test)]
mod tests;

// Re-export public types at module level for convenient access
pub use types::{CollapsedMatcher, CollapsedSpec, DEFAULT_MAX_TERM_EXPANSIONS};
pub use query::OptimizedGraphTraversalQuery;
pub use scorer::OptimizedGraphTraversalScorer;
pub use stats::GraphTraversalStats;
pub use pattern_utils::flatten_graph_traversal_pattern;
