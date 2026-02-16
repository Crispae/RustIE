//! Position intersection utilities for sorted position lists.
//!
//! This module provides efficient algorithms for intersecting sorted position
//! lists, using adaptive strategies based on list size ratios.
//!
//! Also provides `TwoPhaseIntersection` for document-level intersection with
//! verification callbacks, implementing the Lucene TwoPhaseIterator pattern.

use tantivy::DocId;
use super::candidate_driver::CandidateDriver;

/// Lazy iterator wrapper for driver positions
/// Enables on-the-fly consumption without materialization
#[derive(Clone)]
pub(crate) struct PositionIterator<'a> {
    positions: &'a [u32],
    index: usize,
}

impl<'a> PositionIterator<'a> {
    pub fn new(positions: &'a [u32]) -> Self {
        Self { positions, index: 0 }
    }
}

impl<'a> Iterator for PositionIterator<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.positions.len() {
            let pos = self.positions[self.index];
            self.index += 1;
            Some(pos)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.positions.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for PositionIterator<'a> {
    fn len(&self) -> usize {
        self.positions.len().saturating_sub(self.index)
    }
}

/// Perform galloping search (exponential search) for target in arr starting at start_idx.
/// Returns the index where target is found, or where it would be inserted.
/// Guaranteed to return a value >= start_idx.
pub(crate) fn galloping_search(arr: &[u32], target: u32, start_idx: usize) -> usize {
    if start_idx >= arr.len() {
        return arr.len();
    }

    // Check first element
    if arr[start_idx] >= target {
        return start_idx;
    }

    let mut step = 1;
    let mut current = start_idx;

    // Gallop forward: 1, 2, 4, 8...
    while current + step < arr.len() && arr[current + step] < target {
        current += step;
        step *= 2;
    }

    // Binary search in the identified range [current + 1, min(current + step + 1, len)]
    // We strictly know arr[current] < target, so we search after it.
    let upper_bound = std::cmp::min(current + step + 1, arr.len());
    let range = &arr[current + 1..upper_bound];
    match range.binary_search(&target) {
        Ok(idx) => current + 1 + idx,
        Err(idx) => current + 1 + idx,
    }
}

/// Helper for linear two-pointer intersection (fast for similar list sizes)
fn intersect_linear(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
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
}

/// Helper for skewed intersection (one short list, one long list)
fn intersect_skewed(short: &[u32], long: &[u32], out: &mut Vec<u32>) {
    let mut long_idx = 0;

    for &target in short {
        // Find position of target in long list (or Next Greater Element)
        long_idx = galloping_search(long, target, long_idx);

        if long_idx >= long.len() {
            break;
        }

        if long[long_idx] == target {
            out.push(target);
            long_idx += 1; // Advance past the match
        }
    }
}

/// Intersection with hybrid Linear/Galloping strategy
pub(crate) fn intersect_sorted_into(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    if a.is_empty() || b.is_empty() {
        return;
    }

    // Heuristic: If size ratio > 10, use galloping search.
    // Galloping is O(N log M) which beats O(N+M) when M >> N.
    // Linear is faster for dense/similar lists due to CPU cache locality.
    const SKEW_RATIO: usize = 10;

    if a.len() > b.len() * SKEW_RATIO {
        intersect_skewed(b, a, out); // b is short, a is long
    } else if b.len() > a.len() * SKEW_RATIO {
        intersect_skewed(a, b, out); // a is short, b is long
    } else {
        intersect_linear(a, b, out); // Standard two-pointer match
    }
}

// =============================================================================
// TwoPhaseIntersection: Lucene-style TwoPhaseIterator for document intersection
// =============================================================================

/// Threshold for adaptive seek vs. advance decision.
/// For gaps smaller than this, sequential advance() is faster due to skip-list overhead.
const SEEK_THRESHOLD: DocId = 10;

/// Two-phase document intersection implementing the Lucene TwoPhaseIterator pattern.
///
/// This abstraction separates document-level intersection (Phase 1: approximate)
/// from expensive verification (Phase 2: verify). This is the pattern used by
/// Lucene's `ConjunctionDISI.intersectSpans()`.
///
/// # Two-Phase Pattern
///
/// 1. **Phase 1 (Approximate)**: Find candidate documents where both drivers match.
///    This uses skip-list optimized intersection with adaptive seek/advance.
///
/// 2. **Phase 2 (Verify)**: The caller provides a verification callback that
///    performs expensive validation (e.g., graph traversal) only on candidates.
///
/// # Example
///
/// ```ignore
/// let mut intersection = TwoPhaseIntersection::new(src_driver, dst_driver);
///
/// loop {
///     let candidate = intersection.advance();
///     if candidate == tantivy::TERMINATED { break; }
///
///     // Phase 2: Expensive verification
///     if check_graph_traversal(candidate) {
///         // Match found
///     }
/// }
/// ```
pub(crate) struct TwoPhaseIntersection {
    /// Primary driver (typically the rarer/more selective one)
    src_driver: Box<dyn CandidateDriver>,
    /// Secondary driver for intersection
    dst_driver: Box<dyn CandidateDriver>,
    /// Current aligned document (TERMINATED if exhausted)
    current_doc: DocId,
    /// Whether we've started iteration
    started: bool,
}

impl TwoPhaseIntersection {
    /// Create a new two-phase intersection over two candidate drivers.
    ///
    /// The drivers should be positioned before their first document
    /// (i.e., freshly created from postings).
    pub fn new(
        src_driver: Box<dyn CandidateDriver>,
        dst_driver: Box<dyn CandidateDriver>,
    ) -> Self {
        Self {
            src_driver,
            dst_driver,
            current_doc: tantivy::TERMINATED,
            started: false,
        }
    }

    /// Get the source driver (for position access after alignment).
    #[inline]
    pub fn src_driver(&self) -> &dyn CandidateDriver {
        self.src_driver.as_ref()
    }

    /// Get the destination driver (for position access after alignment).
    #[inline]
    pub fn dst_driver(&self) -> &dyn CandidateDriver {
        self.dst_driver.as_ref()
    }

    /// Advance to the next candidate document where both drivers align.
    ///
    /// This is Phase 1 of the two-phase pattern. After this returns a valid
    /// document ID, the caller should perform Phase 2 verification.
    ///
    /// Returns `tantivy::TERMINATED` when exhausted.
    pub fn advance(&mut self) -> DocId {
        // Get next candidate from source
        let candidate = if !self.started {
            self.started = true;
            self.src_driver.seek(0)
        } else {
            self.src_driver.advance()
        };

        self.current_doc = self.align_drivers(candidate);
        self.current_doc
    }

    /// Seek to the first candidate document >= target where both drivers align.
    ///
    /// This is Phase 1 of the two-phase pattern with a target hint.
    ///
    /// Returns `tantivy::TERMINATED` when exhausted.
    pub fn seek(&mut self, target: DocId) -> DocId {
        // Check if already at or past target
        if self.current_doc != tantivy::TERMINATED && self.current_doc >= target {
            return self.current_doc;
        }

        self.started = true;
        let candidate = self.src_driver.seek(target);
        self.current_doc = self.align_drivers(candidate);
        self.current_doc
    }

    /// Advance the source driver and realign (used after verification failure).
    ///
    /// Call this when Phase 2 verification fails to move to the next candidate.
    #[inline]
    pub fn advance_past_current(&mut self) -> DocId {
        let candidate = self.src_driver.advance();
        self.current_doc = self.align_drivers(candidate);
        self.current_doc
    }

    /// Core alignment loop: find the next document where both drivers match.
    ///
    /// Uses adaptive seek/advance based on gap size for optimal performance:
    /// - Large gaps (>= SEEK_THRESHOLD): Use skip-list seek (O(log n))
    /// - Small gaps: Use sequential advance (O(1) amortized, better cache locality)
    #[inline]
    fn align_drivers(&mut self, mut candidate: DocId) -> DocId {
        loop {
            if candidate == tantivy::TERMINATED {
                return tantivy::TERMINATED;
            }

            let dst_current = self.dst_driver.doc();

            // Align dst_driver to candidate using adaptive strategy
            let dst_doc = if dst_current < candidate {
                let gap = candidate - dst_current;
                if gap >= SEEK_THRESHOLD {
                    self.dst_driver.seek(candidate)
                } else {
                    // Sequential advance for small gaps
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

            // Check for exhaustion
            if dst_doc == tantivy::TERMINATED {
                return tantivy::TERMINATED;
            }

            // If dst overshot, realign src
            if dst_doc > candidate {
                let gap = dst_doc - candidate;
                if gap >= SEEK_THRESHOLD {
                    candidate = self.src_driver.seek(dst_doc);
                } else {
                    // Sequential advance for small gaps
                    while candidate < dst_doc {
                        candidate = self.src_driver.advance();
                        if candidate == tantivy::TERMINATED {
                            break;
                        }
                    }
                }
                continue;
            }

            // Documents aligned!
            debug_assert_eq!(candidate, dst_doc);
            return candidate;
        }
    }


}
