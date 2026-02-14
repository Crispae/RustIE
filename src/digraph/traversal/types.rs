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

impl TraversalResult {
    /// Deduplicate and convert a node list into a TraversalResult.
    /// Returns `FailTraversal` if the list is empty, `Success` otherwise.
    pub fn from_nodes(mut nodes: Vec<usize>) -> Self {
        nodes.sort_unstable();
        nodes.dedup();
        if nodes.is_empty() {
            TraversalResult::FailTraversal
        } else {
            TraversalResult::Success(nodes)
        }
    }
}

/// Set of allowed positions for a constraint (e.g. prefilter or driver positions).
/// Stored as a sorted vec for cache-friendly O(log n) lookup; avoids HashSet
/// allocation and hashing for sentence-sized position sets.
#[derive(Debug, Clone)]
pub struct AllowedPositions {
    sorted: Vec<u32>,
}

impl AllowedPositions {
    /// Build from a slice of positions. Sorts and deduplicates.
    /// Use for sentence-sized sets; debug_assert validates expected size.
    pub fn from_positions(positions: &[u32]) -> Self {
        debug_assert!(
            positions.len() < 1000,
            "Unexpectedly large position set ({}); consider HashSet variant if this fires",
            positions.len()
        );
        let mut sorted = positions.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        Self { sorted }
    }

    #[inline]
    pub fn contains(&self, v: u32) -> bool {
        self.sorted.binary_search(&v).is_ok()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }
}
