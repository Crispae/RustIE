//! Position intersection utilities for sorted position lists.
//!
//! This module provides efficient algorithms for intersecting sorted position
//! lists, using adaptive strategies based on list size ratios.

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
