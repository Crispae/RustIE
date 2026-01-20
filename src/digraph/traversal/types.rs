//! Types and constants for graph traversal operations.

/// Minimum number of start positions to enable parallel processing.
/// Below this threshold, sequential processing is faster due to overhead.
///
/// Typical use case: 3-10 start positions per sentence, so threshold of 4-8
/// allows parallelization for single sentences with many matches or multi-sentence documents.
pub const PARALLEL_START_POSITIONS_THRESHOLD: usize = 4;

#[derive(Debug, Clone)]
pub enum TraversalResult {
    /// No traversal operation performed
    NoTraversal,
    /// Traversal failed to find matches
    FailTraversal,
    /// Successfully found matching nodes
    Success(Vec<usize>),
}
