//! CandidateDriver abstraction for Odinson-style collapsed query optimization.
//!
//! This module provides the core abstraction for document iteration with optional
//! position access, enabling efficient position-based query optimization.

use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, RwLock};

use tantivy::{
    DocId, DocSet,
    schema::{Field, IndexRecordOption},
    postings::{SegmentPostings, Postings},
    SegmentReader,
    Term,
};
use tantivy_fst::Regex;

use super::types::CollapsedMatcher;
use super::intersection::{PositionIterator, intersect_sorted_into};

// =============================================================================
// CandidateDriver Trait and Implementations (Odinson-style collapsed queries)
// =============================================================================

/// Abstraction for candidate document iteration with optional position access.
/// Avoids downcasting from Box<dyn Scorer> by making position-aware drivers explicit.
pub(crate) trait CandidateDriver: Send {
    /// Current document ID
    fn doc(&self) -> DocId;

    /// Advance to the next matching document
    fn advance(&mut self) -> DocId;

    /// Seek to the first document >= target (uses skip lists for O(log n) performance)
    /// This is the key optimization over sequential advance
    fn seek(&mut self, target: DocId) -> DocId;

    /// If this driver is position-aware (collapsed constraint + edge),
    /// return the positions that matched (intersection). Otherwise None.
    fn matching_positions(&self) -> Option<&[u32]>;

    /// Return a lazy iterator over matching positions (Odinson-style)
    /// Returns None if not position-aware
    fn matching_positions_iter(&self) -> Option<PositionIterator<'_>> {
        self.matching_positions().map(|positions| PositionIterator::new(positions))
    }
}

/// EmptyDriver: Immediately returns TERMINATED.
/// Used when CombinedPositionDriver cannot be built (required term missing from segment)
/// or when CollapsedSpec exists but postings are unavailable.
/// This skips entire segments when required terms are missing, avoiding wasted computation.
pub(crate) struct EmptyDriver;

impl CandidateDriver for EmptyDriver {
    fn doc(&self) -> DocId {
        tantivy::TERMINATED
    }

    fn advance(&mut self) -> DocId {
        tantivy::TERMINATED
    }

    fn seek(&mut self, _target: DocId) -> DocId {
        tantivy::TERMINATED
    }

    fn matching_positions(&self) -> Option<&[u32]> {
        None
    }
}

/// Drives iteration using two postings cursors (constraint + edge).
/// Computes position intersection at index level via skip lists.
/// This is the core of the Odinson-style collapsed query optimization.
pub(crate) struct CombinedPositionDriver {
    constraint_postings: SegmentPostings,
    edge_postings: SegmentPostings,
    constraint_buf: Vec<u32>,
    edge_buf: Vec<u32>,
    intersection: Vec<u32>,
    current_doc: DocId,
}

impl CombinedPositionDriver {
    pub fn new(constraint_postings: SegmentPostings, edge_postings: SegmentPostings) -> Self {
        // Position both postings at their first document
        // SegmentPostings starts positioned at first doc after creation
        let mut driver = Self {
            constraint_postings,
            edge_postings,
            constraint_buf: Vec::with_capacity(16),
            edge_buf: Vec::with_capacity(16),
            intersection: Vec::with_capacity(8),
            current_doc: tantivy::TERMINATED,
        };
        // Advance to first matching doc (where positions intersect)
        driver.advance_to_next_match();
        driver
    }

    /// Internal advance that finds next doc with overlapping positions
    fn advance_to_next_match(&mut self) -> DocId {
        loop {
            let d1 = self.constraint_postings.doc();
            let d2 = self.edge_postings.doc();

            // If either is exhausted, we're done
            if d1 == tantivy::TERMINATED || d2 == tantivy::TERMINATED {
                self.current_doc = tantivy::TERMINATED;
                return self.current_doc;
            }

            // OPTIMIZATION: Adaptive alignment - use seek() for large gaps, advance() for small gaps
            // Skip lists have overhead, so for small gaps (<10 docs), sequential advance() is faster
            const SEEK_THRESHOLD: DocId = 10;
            if d1 < d2 {
                let gap = d2 - d1;
                if gap >= SEEK_THRESHOLD && self.constraint_postings.doc() < d2 {
                    // Large gap: use skip-list seeking (O(log n))
                    self.constraint_postings.seek(d2);
                } else {
                    // Small gap: use sequential advance() (O(1) for small gaps)
                    self.constraint_postings.advance();
                }
                continue;
            } else if d2 < d1 {
                let gap = d1 - d2;
                if gap >= SEEK_THRESHOLD && self.edge_postings.doc() < d1 {
                    // Large gap: use skip-list seeking (O(log n))
                    self.edge_postings.seek(d1);
                } else {
                    // Small gap: use sequential advance() (O(1) for small gaps)
                    self.edge_postings.advance();
                }
                continue;
            }

            // Same doc - compute position intersection
            self.constraint_buf.clear();
            self.edge_buf.clear();
            self.constraint_postings.positions(&mut self.constraint_buf);
            self.edge_postings.positions(&mut self.edge_buf);

            self.intersection.clear();
            intersect_sorted_into(&self.constraint_buf, &self.edge_buf, &mut self.intersection);

            let doc = d1;

            // Advance both for next iteration
            self.constraint_postings.advance();
            self.edge_postings.advance();

            if !self.intersection.is_empty() {
                self.current_doc = doc;
                return self.current_doc;
            }
            // No position overlap - continue to next doc
        }
    }

    /// Internal seek that finds next doc >= current with overlapping positions
    /// Used after seeking both postings to a target
    fn seek_to_next_match(&mut self) -> DocId {
        loop {
            let d1 = self.constraint_postings.doc();
            let d2 = self.edge_postings.doc();

            // If either is exhausted, we're done
            if d1 == tantivy::TERMINATED || d2 == tantivy::TERMINATED {
                self.current_doc = tantivy::TERMINATED;
                return self.current_doc;
            }

            // OPTIMIZATION: Adaptive alignment - use seek() for large gaps, advance() for small gaps
            const SEEK_THRESHOLD: DocId = 10;
            if d1 < d2 {
                let gap = d2 - d1;
                if gap >= SEEK_THRESHOLD && self.constraint_postings.doc() < d2 {
                    self.constraint_postings.seek(d2);
                } else {
                    self.constraint_postings.advance();
                }
                continue;
            } else if d2 < d1 {
                let gap = d1 - d2;
                if gap >= SEEK_THRESHOLD && self.edge_postings.doc() < d1 {
                    self.edge_postings.seek(d1);
                } else {
                    self.edge_postings.advance();
                }
                continue;
            }

            // Same doc - compute position intersection
            self.constraint_buf.clear();
            self.edge_buf.clear();
            self.constraint_postings.positions(&mut self.constraint_buf);
            self.edge_postings.positions(&mut self.edge_buf);

            self.intersection.clear();
            intersect_sorted_into(&self.constraint_buf, &self.edge_buf, &mut self.intersection);

            let doc = d1;

            // Advance both for next iteration
            self.constraint_postings.advance();
            self.edge_postings.advance();

            if !self.intersection.is_empty() {
                self.current_doc = doc;
                return self.current_doc;
            }
            // No position overlap - continue to next doc
        }
    }
}

impl CandidateDriver for CombinedPositionDriver {
    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn advance(&mut self) -> DocId {
        self.advance_to_next_match()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        // If already at or past target, return current
        if self.current_doc != tantivy::TERMINATED && self.current_doc >= target {
            return self.current_doc;
        }

        // OPTIMIZATION: Adaptive seeking
        const SEEK_THRESHOLD: DocId = 10;

        if self.constraint_postings.doc() < target {
            let gap = target - self.constraint_postings.doc();
            if gap >= SEEK_THRESHOLD {
                self.constraint_postings.seek(target);
            } else {
                while self.constraint_postings.doc() < target {
                    self.constraint_postings.advance();
                }
            }
        }
        if self.edge_postings.doc() < target {
            let gap = target - self.edge_postings.doc();
            if gap >= SEEK_THRESHOLD {
                self.edge_postings.seek(target);
            } else {
                while self.edge_postings.doc() < target {
                    self.edge_postings.advance();
                }
            }
        }

        // Now find the next matching document (with position intersection)
        self.seek_to_next_match()
    }

    fn matching_positions(&self) -> Option<&[u32]> {
        Some(&self.intersection)
    }
}

// =============================================================================
// Regex Support: Term Expansion and Union Position Iteration
// =============================================================================

/// Helper function to expand terms using a cached regex automaton.
/// Extracted to avoid code duplication and enable caching.
pub(crate) fn expand_with_automaton(
    term_dict: &tantivy::termdict::TermDictionary,
    automaton: &Regex,
    field: Field,
    inverted_index: &tantivy::InvertedIndexReader,
    max_expansions: usize,
) -> Option<Vec<SegmentPostings>> {
    let mut stream = match term_dict.search(automaton).into_stream() {
        Ok(s) => s,
        Err(e) => {
            log::warn!("Failed to search term dict: {:?}", e);
            return None;
        }
    };

    let mut postings_list = Vec::new();
    let mut count = 0;

    while stream.advance() {
        if count >= max_expansions {
            log::warn!(
                "collapse regex disabled: expanded_terms={} > max={}",
                count, max_expansions
            );
            return None;
        }

        let term_bytes = stream.key();
        let term = Term::from_field_bytes(field, term_bytes);
        if let Ok(Some(postings)) = inverted_index
            .read_postings(&term, IndexRecordOption::WithFreqsAndPositions)
        {
            postings_list.push(postings);
        }
        count += 1;
    }

    if postings_list.is_empty() {
        return None;
    }

    Some(postings_list)
}

/// Get or compile a regex automaton from the cache (thread-safe).
pub(crate) fn get_or_compile_regex(
    cache: &Arc<RwLock<HashMap<String, Arc<Regex>>>>,
    pattern: &str,
) -> Option<Arc<Regex>> {
    // Fast path: read lock
    {
        let read_guard = cache.read().ok()?;
        if let Some(regex) = read_guard.get(pattern) {
            return Some(Arc::clone(regex));
        }
    } // Read lock released here

    // Slow path: write lock
    let mut write_guard = cache.write().ok()?;

    // Double-check after acquiring write lock (another thread might have compiled it)
    if let Some(regex) = write_guard.get(pattern) {
        return Some(Arc::clone(regex));
    }

    // Compile and cache
    let regex = match Regex::new(pattern) {
        Ok(r) => Arc::new(r),
        Err(e) => {
            log::warn!("Invalid regex pattern '{}': {}", pattern, e);
            return None;
        }
    };
    write_guard.insert(pattern.to_string(), Arc::clone(&regex));
    Some(regex)
}

/// Expand a CollapsedMatcher to Vec<SegmentPostings>.
/// Returns None if:
/// - Field doesn't support positions
/// - Regex exceeds max_expansions (logs warning and bails out)
/// - No matching terms in segment
pub(crate) fn expand_matcher(
    reader: &SegmentReader,
    field: Field,
    matcher: &CollapsedMatcher,
    max_expansions: usize,
    regex_cache: Arc<RwLock<HashMap<String, Arc<Regex>>>>,
) -> Option<Vec<SegmentPostings>> {
    let inverted_index = reader.inverted_index(field).ok()?;

    match matcher {
        CollapsedMatcher::Exact(term_str) => {
            // Single term lookup (fast path)
            let term = Term::from_field_text(field, term_str);
            let postings = inverted_index
                .read_postings(&term, IndexRecordOption::WithFreqsAndPositions)
                .ok()??;
            Some(vec![postings])
        }
        CollapsedMatcher::RegexPattern(pattern) => {
            // Automaton-based term enumeration
            let term_dict = inverted_index.terms();

            // Get or compile regex automaton from cache (thread-safe)
            let automaton = get_or_compile_regex(&regex_cache, pattern)?;

            // Expand terms using the cached automaton
            expand_with_automaton(&term_dict, automaton.as_ref(), field, &inverted_index, max_expansions)
        }
    }
}

/// Entry in the min-heap for k-way merge of postings
struct PostingsEntry {
    doc: DocId,
    idx: usize,  // Index into postings vec
}

impl Ord for PostingsEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: smaller doc first (reverse comparison)
        other.doc.cmp(&self.doc)
    }
}

impl PartialOrd for PostingsEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for PostingsEntry {}

impl PartialEq for PostingsEntry {
    fn eq(&self, other: &Self) -> bool {
        self.doc == other.doc
    }
}

/// Union iterator over multiple postings lists.
/// Yields (doc_id, merged_positions) for each doc that appears in any postings.
/// Uses heap-based k-way merge for efficient iteration.
pub(crate) struct UnionPositionsIterator {
    postings: Vec<SegmentPostings>,
    heap: BinaryHeap<PostingsEntry>,
    current_doc: DocId,
    merged_positions: Vec<u32>,
    position_buf: Vec<u32>,
}

impl UnionPositionsIterator {
    pub fn new(postings: Vec<SegmentPostings>) -> Self {
        let mut heap = BinaryHeap::with_capacity(postings.len());

        // Initialize heap with first doc from each postings
        for (idx, p) in postings.iter().enumerate() {
            let doc = p.doc();
            if doc != tantivy::TERMINATED {
                heap.push(PostingsEntry { doc, idx });
            }
        }

        let mut iter = Self {
            postings,
            heap,
            current_doc: tantivy::TERMINATED,
            merged_positions: Vec::with_capacity(32),
            position_buf: Vec::with_capacity(16),
        };

        // Advance to first doc
        iter.advance_to_next_doc();
        iter
    }

    pub fn doc(&self) -> DocId {
        self.current_doc
    }

    pub fn positions(&self) -> &[u32] {
        &self.merged_positions
    }

    pub fn advance_to_next_doc(&mut self) -> DocId {
        self.merged_positions.clear();

        if self.heap.is_empty() {
            self.current_doc = tantivy::TERMINATED;
            return self.current_doc;
        }

        // Get minimum doc
        let min_doc = self.heap.peek().unwrap().doc;
        self.current_doc = min_doc;

        // Collect positions from all postings at min_doc directly into merged_positions
        while let Some(entry) = self.heap.peek() {
            if entry.doc != min_doc {
                break;
            }

            let entry = self.heap.pop().unwrap();
            let postings = &mut self.postings[entry.idx];

            // Get positions into temp buffer, then extend merged_positions
            self.position_buf.clear();
            postings.positions(&mut self.position_buf);
            self.merged_positions.extend_from_slice(&self.position_buf);

            // Advance this postings and re-insert if not exhausted
            let next_doc = postings.advance();
            if next_doc != tantivy::TERMINATED {
                self.heap.push(PostingsEntry { doc: next_doc, idx: entry.idx });
            }
        }

        // Sort and dedup the merged positions
        if !self.merged_positions.is_empty() {
            self.merged_positions.sort_unstable();
            self.merged_positions.dedup();
        }

        self.current_doc
    }

    /// Seek to the first document >= target
    /// Uses seek() on underlying postings for O(log n) skip-list based seeking
    pub fn seek_to_doc(&mut self, target: DocId) -> DocId {
        // If already at or past target, return current
        if self.current_doc != tantivy::TERMINATED && self.current_doc >= target {
            return self.current_doc;
        }

        // Clear and rebuild heap by seeking all postings to target
        self.heap.clear();
        for (idx, p) in self.postings.iter_mut().enumerate() {
            // CRITICAL: Only seek if current doc < target (tantivy requires self.doc() <= target)
            let doc = if p.doc() < target {
                p.seek(target)
            } else {
                p.doc()
            };
            if doc != tantivy::TERMINATED {
                self.heap.push(PostingsEntry { doc, idx });
            }
        }

        // Advance to next valid doc (which will be >= target)
        self.advance_to_next_doc()
    }
}

/// Driver that combines two UnionPositionsIterators with position intersection.
/// This is the regex-capable version of CombinedPositionDriver.
/// Implements the Lucene "SpanMultiTermQueryWrapper + SpanNear/AND" pattern.
pub(crate) struct UnionAndIntersectDriver {
    lhs: UnionPositionsIterator,  // Constraint side
    rhs: UnionPositionsIterator,  // Edge side
    current_doc: DocId,
    intersection: Vec<u32>,
}

impl UnionAndIntersectDriver {
    pub fn new(lhs: UnionPositionsIterator, rhs: UnionPositionsIterator) -> Self {
        let mut driver = Self {
            lhs,
            rhs,
            current_doc: tantivy::TERMINATED,
            intersection: Vec::with_capacity(16),
        };
        driver.advance_to_next_match();
        driver
    }

    fn advance_to_next_match(&mut self) -> DocId {
        loop {
            let d1 = self.lhs.doc();
            let d2 = self.rhs.doc();

            if d1 == tantivy::TERMINATED || d2 == tantivy::TERMINATED {
                self.current_doc = tantivy::TERMINATED;
                return self.current_doc;
            }

            // OPTIMIZATION: Adaptive alignment
            const SEEK_THRESHOLD: DocId = 10;
            if d1 < d2 {
                let gap = d2 - d1;
                if gap >= SEEK_THRESHOLD {
                    self.lhs.seek_to_doc(d2);
                } else {
                    self.lhs.advance_to_next_doc();
                }
                continue;
            } else if d2 < d1 {
                let gap = d1 - d2;
                if gap >= SEEK_THRESHOLD {
                    self.rhs.seek_to_doc(d1);
                } else {
                    self.rhs.advance_to_next_doc();
                }
                continue;
            }

            // Same doc - intersect positions
            self.intersection.clear();
            intersect_sorted_into(
                self.lhs.positions(),
                self.rhs.positions(),
                &mut self.intersection
            );

            let doc = d1;

            // Advance both for next iteration
            self.lhs.advance_to_next_doc();
            self.rhs.advance_to_next_doc();

            if !self.intersection.is_empty() {
                self.current_doc = doc;
                return self.current_doc;
            }
            // No position overlap - continue
        }
    }

    fn seek_to_next_match(&mut self) -> DocId {
        loop {
            let d1 = self.lhs.doc();
            let d2 = self.rhs.doc();

            if d1 == tantivy::TERMINATED || d2 == tantivy::TERMINATED {
                self.current_doc = tantivy::TERMINATED;
                return self.current_doc;
            }

            const SEEK_THRESHOLD: DocId = 10;
            if d1 < d2 {
                let gap = d2 - d1;
                if gap >= SEEK_THRESHOLD {
                    self.lhs.seek_to_doc(d2);
                } else {
                    self.lhs.advance_to_next_doc();
                }
                continue;
            } else if d2 < d1 {
                let gap = d1 - d2;
                if gap >= SEEK_THRESHOLD {
                    self.rhs.seek_to_doc(d1);
                } else {
                    self.rhs.advance_to_next_doc();
                }
                continue;
            }

            // Same doc - intersect positions
            self.intersection.clear();
            intersect_sorted_into(
                self.lhs.positions(),
                self.rhs.positions(),
                &mut self.intersection
            );

            let doc = d1;

            // Advance both for next iteration
            self.lhs.advance_to_next_doc();
            self.rhs.advance_to_next_doc();

            if !self.intersection.is_empty() {
                self.current_doc = doc;
                return self.current_doc;
            }
            // No position overlap - continue
        }
    }
}

impl CandidateDriver for UnionAndIntersectDriver {
    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn advance(&mut self) -> DocId {
        self.advance_to_next_match()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        // If already at or past target, return current
        if self.current_doc != tantivy::TERMINATED && self.current_doc >= target {
            return self.current_doc;
        }

        // OPTIMIZATION: Seek both union iterators to target (O(log n) per postings)
        self.lhs.seek_to_doc(target);
        self.rhs.seek_to_doc(target);

        // Find next doc with position intersection
        self.seek_to_next_match()
    }

    fn matching_positions(&self) -> Option<&[u32]> {
        Some(&self.intersection)
    }
}

// =============================================================================
// IntermediateEdgeFilter: Document-level filtering for intermediate edges
// =============================================================================

/// Filter that checks document-level presence of intermediate edges.
/// Used during document enumeration to skip documents that can't possibly match.
/// This enables Odinson-style filtering where ALL edges participate in document selection.
pub(crate) struct IntermediateEdgeFilter {
    /// Postings for intermediate edges (edges not collapsed into src/dst drivers)
    edge_postings: Vec<SegmentPostings>,
    /// Current minimum document across all postings (for efficient intersection)
    current_min_doc: DocId,
}

impl IntermediateEdgeFilter {
    /// Create a new filter from intermediate edge postings
    pub fn new(edge_postings: Vec<SegmentPostings>) -> Self {
        let current_min_doc = if edge_postings.is_empty() {
            tantivy::TERMINATED
        } else {
            // Find minimum doc across all postings
            edge_postings.iter()
                .map(|p| p.doc())
                .filter(|&d| d != tantivy::TERMINATED)
                .min()
                .unwrap_or(tantivy::TERMINATED)
        };

        Self {
            edge_postings,
            current_min_doc,
        }
    }

    /// Check if filter has any postings (if empty, all documents pass)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.edge_postings.is_empty()
    }

    /// Check if a document has ALL intermediate edges.
    /// Returns true if document passes the filter (has all edges).
    /// Seeks all postings to the target document.
    pub fn document_matches(&mut self, doc_id: DocId) -> bool {
        if self.edge_postings.is_empty() {
            return true; // No intermediate edges to check
        }

        for postings in &mut self.edge_postings {
            // Seek to doc_id if not already there or past
            if postings.doc() < doc_id {
                postings.seek(doc_id);
            }

            // If this postings doesn't have doc_id, document fails the filter
            if postings.doc() != doc_id {
                return false;
            }
        }

        true
    }

    /// Find the next document >= target that has ALL intermediate edges.
    /// Uses adaptive seeking for efficiency.
    /// Returns TERMINATED if no such document exists.
    pub fn seek_to_matching_doc(&mut self, target: DocId) -> DocId {
        if self.edge_postings.is_empty() {
            return target; // No filter, any target is valid
        }

        let mut candidate = target;

        loop {
            // Seek all postings to candidate
            let mut max_doc = candidate;
            let mut all_match = true;

            for postings in &mut self.edge_postings {
                if postings.doc() < candidate {
                    postings.seek(candidate);
                }

                let doc = postings.doc();
                if doc == tantivy::TERMINATED {
                    self.current_min_doc = tantivy::TERMINATED;
                    return tantivy::TERMINATED;
                }

                if doc != candidate {
                    all_match = false;
                    if doc > max_doc {
                        max_doc = doc;
                    }
                }
            }

            if all_match {
                self.current_min_doc = candidate;
                return candidate;
            }

            // At least one posting is ahead - advance candidate to max_doc
            candidate = max_doc;
        }
    }

    /// Get the current minimum document (for skip-ahead optimization)
    #[inline]
    pub fn current_doc(&self) -> DocId {
        self.current_min_doc
    }
}
