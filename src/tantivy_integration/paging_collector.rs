//! Odinson-style pagination collectors: count total_hits and keep only one page of (DocAddress, score).
//!
//! - [SimpleCollector]: first page; no cursor.
//! - [PagingCollector]: next pages; skip docs at or before the cursor.

use crate::types::SearchCursor;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use tantivy::collector::{Collector, SegmentCollector};
use tantivy::{DocAddress, DocId, Score, SegmentReader};
use tantivy::Result as TantivyResult;

/// A scored document address, ordered by (score DESC, segment_ord ASC, doc_id ASC)
/// so that "smaller" means "earlier in result order" (better).
#[derive(Debug, Clone)]
pub struct ScoredDoc {
    pub score: Score,
    pub address: DocAddress,
}

impl PartialEq for ScoredDoc {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
            && self.address.segment_ord == other.address.segment_ord
            && self.address.doc_id == other.address.doc_id
    }
}

impl Eq for ScoredDoc {}

impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredDoc {
    /// Order: score DESC, then segment_ord ASC, then doc_id ASC.
    /// Smaller = earlier in result list = better.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.address.segment_ord.cmp(&other.address.segment_ord))
            .then_with(|| self.address.doc_id.cmp(&other.address.doc_id))
    }
}

impl ScoredDoc {
    /// Returns true if this doc appeared on a previous page
    /// (at or before the cursor in global result order).
    ///
    /// Global order: score DESC, segment_ord ASC, doc_id ASC
    ///
    /// Skip when:
    ///   - Higher score (appeared earlier in results)
    ///   - Same score + earlier (segment_ord, doc_id) tuple
    ///   - Exact match (same score, segment, doc = the cursor itself)
    pub fn should_skip(&self, cursor: &SearchCursor) -> bool {
        match self.score.partial_cmp(&cursor.score) {
            // Higher score → earlier in results → skip
            Some(Ordering::Greater) => true,
            // Lower score → later in results → keep
            Some(Ordering::Less) => false,
            // Same score (or NaN) → compare (segment_ord, doc_id) ASC
            _ => {
                let self_pos = (self.address.segment_ord, self.address.doc_id);
                let cursor_pos = (cursor.segment_ord, cursor.doc_id);
                self_pos <= cursor_pos
            }
        }
    }
}

/// Result of a paginated search: total hit count and the page of scored docs.
pub struct PaginatedSearchResult {
    pub total_hits: usize,
    pub scored_docs: Vec<ScoredDoc>,
}

// ─── First-page collector (like Odinson's SimpleOdinsonCollector) ───

/// Collector for the first page: counts all hits, keeps top `page_size` by (score DESC, segment_ord, doc_id).
pub struct SimpleCollector {
    page_size: usize,
}

impl SimpleCollector {
    pub fn new(page_size: usize) -> Self {
        Self { page_size }
    }
}

impl Collector for SimpleCollector {
    type Fruit = PaginatedSearchResult;
    type Child = SimpleSegmentCollector;

    fn requires_scoring(&self) -> bool {
        true
    }

    fn for_segment(
        &self,
        segment_ord: u32,
        _segment: &SegmentReader,
    ) -> TantivyResult<Self::Child> {
        Ok(SimpleSegmentCollector {
            segment_ord,
            page_size: self.page_size,
            total_hits: 0,
            heap: BinaryHeap::new(),
        })
    }

    fn merge_fruits(
        &self,
        segment_fruits: Vec<PaginatedSearchResult>,
    ) -> TantivyResult<PaginatedSearchResult> {
        let total_hits: usize = segment_fruits.iter().map(|f| f.total_hits).sum();
        let mut all_docs: Vec<ScoredDoc> = segment_fruits
            .into_iter()
            .flat_map(|f| f.scored_docs)
            .collect();
        all_docs.sort(); // score DESC, segment_ord ASC, doc_id ASC
        all_docs.truncate(self.page_size);
        Ok(PaginatedSearchResult {
            total_hits,
            scored_docs: all_docs,
        })
    }
}

/// Segment-level collector for the first page: count hits, keep top page_size in a min-heap (worst at top).
pub struct SimpleSegmentCollector {
    segment_ord: u32,
    page_size: usize,
    total_hits: usize,
    heap: BinaryHeap<std::cmp::Reverse<ScoredDoc>>,
}

impl SegmentCollector for SimpleSegmentCollector {
    type Fruit = PaginatedSearchResult;

    fn collect(&mut self, doc: DocId, score: Score) {
        self.total_hits += 1;
        let scored_doc = ScoredDoc {
            score,
            address: DocAddress::new(self.segment_ord, doc),
        };
        if self.heap.len() < self.page_size {
            self.heap.push(std::cmp::Reverse(scored_doc));
        } else if let Some(worst) = self.heap.peek() {
            if scored_doc < worst.0 {
                self.heap.pop();
                self.heap.push(std::cmp::Reverse(scored_doc));
            }
        }
    }

    fn harvest(self) -> PaginatedSearchResult {
        let mut scored_docs: Vec<ScoredDoc> = self
            .heap
            .into_iter()
            .map(|r| r.0)
            .collect();
        scored_docs.sort();
        PaginatedSearchResult {
            total_hits: self.total_hits,
            scored_docs,
        }
    }
}

// ─── Paging collector (like Odinson's PagingOdinsonCollector) ───

/// Collector for next pages: same as SimpleCollector but skips docs at or before the cursor.
pub struct PagingCollector {
    page_size: usize,
    after: SearchCursor,
}

impl PagingCollector {
    pub fn new(page_size: usize, after: SearchCursor) -> Self {
        Self { page_size, after }
    }
}

impl Collector for PagingCollector {
    type Fruit = PaginatedSearchResult;
    type Child = PagingSegmentCollector;

    fn requires_scoring(&self) -> bool {
        true
    }

    fn for_segment(
        &self,
        segment_ord: u32,
        _segment: &SegmentReader,
    ) -> TantivyResult<Self::Child> {
        Ok(PagingSegmentCollector {
            segment_ord,
            page_size: self.page_size,
            after: self.after.clone(),
            total_hits: 0,
            heap: BinaryHeap::new(),
        })
    }

    fn merge_fruits(
        &self,
        segment_fruits: Vec<PaginatedSearchResult>,
    ) -> TantivyResult<PaginatedSearchResult> {
        let total_hits: usize = segment_fruits.iter().map(|f| f.total_hits).sum();
        let mut all_docs: Vec<ScoredDoc> = segment_fruits
            .into_iter()
            .flat_map(|f| f.scored_docs)
            .collect();
        all_docs.sort();
        all_docs.truncate(self.page_size);
        Ok(PaginatedSearchResult {
            total_hits,
            scored_docs: all_docs,
        })
    }
}

/// Segment-level collector for next pages: skip docs at or before cursor, then same heap logic.
pub struct PagingSegmentCollector {
    segment_ord: u32,
    page_size: usize,
    after: SearchCursor,
    total_hits: usize,
    heap: BinaryHeap<std::cmp::Reverse<ScoredDoc>>,
}

impl SegmentCollector for PagingSegmentCollector {
    type Fruit = PaginatedSearchResult;

    fn collect(&mut self, doc: DocId, score: Score) {
        self.total_hits += 1;
        let scored_doc = ScoredDoc {
            score,
            address: DocAddress::new(self.segment_ord, doc),
        };
        if scored_doc.should_skip(&self.after) {
            return;
        }
        if self.heap.len() < self.page_size {
            self.heap.push(std::cmp::Reverse(scored_doc));
        } else if let Some(worst) = self.heap.peek() {
            if scored_doc < worst.0 {
                self.heap.pop();
                self.heap.push(std::cmp::Reverse(scored_doc));
            }
        }
    }

    fn harvest(self) -> PaginatedSearchResult {
        let mut scored_docs: Vec<ScoredDoc> = self.heap.into_iter().map(|r| r.0).collect();
        scored_docs.sort();
        PaginatedSearchResult {
            total_hits: self.total_hits,
            scored_docs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scored_doc_ord_higher_score_better() {
        let a = ScoredDoc {
            score: 5.0,
            address: DocAddress::new(0, 10),
        };
        let b = ScoredDoc {
            score: 3.0,
            address: DocAddress::new(0, 20),
        };
        assert!(a < b); // a is better (higher score) => smaller in our Ord
    }

    #[test]
    fn scored_doc_ord_same_score_then_segment_then_doc() {
        let a = ScoredDoc {
            score: 4.0,
            address: DocAddress::new(0, 5),
        };
        let b = ScoredDoc {
            score: 4.0,
            address: DocAddress::new(0, 10),
        };
        assert!(a < b);

        let c = ScoredDoc {
            score: 4.0,
            address: DocAddress::new(1, 0),
        };
        assert!(a < c);
        assert!(b < c);
    }

    #[test]
    fn should_skip_higher_score() {
        let cursor = SearchCursor::new(0, 20, 3.0);
        let doc = ScoredDoc {
            score: 5.0,
            address: DocAddress::new(0, 10),
        };
        assert!(doc.should_skip(&cursor));
    }

    #[test]
    fn should_skip_same_score_before_cursor() {
        let cursor = SearchCursor::new(0, 20, 4.0);
        let doc = ScoredDoc {
            score: 4.0,
            address: DocAddress::new(0, 15),
        };
        assert!(doc.should_skip(&cursor));
    }

    #[test]
    fn should_skip_same_score_at_cursor() {
        let cursor = SearchCursor::new(0, 20, 4.0);
        let doc = ScoredDoc {
            score: 4.0,
            address: DocAddress::new(0, 20),
        };
        assert!(doc.should_skip(&cursor));
    }

    #[test]
    fn should_not_skip_same_score_after_cursor() {
        let cursor = SearchCursor::new(0, 20, 4.0);
        let doc = ScoredDoc {
            score: 4.0,
            address: DocAddress::new(0, 25),
        };
        assert!(!doc.should_skip(&cursor));
    }

    #[test]
    fn should_not_skip_lower_score() {
        let cursor = SearchCursor::new(0, 20, 4.0);
        let doc = ScoredDoc {
            score: 2.0,
            address: DocAddress::new(1, 0),
        };
        assert!(!doc.should_skip(&cursor));
    }

    // Edge-case tests for should_skip (cursor at score=5.0, seg=0, doc=20)
    #[test]
    fn skip_same_score_same_seg_earlier_doc() {
        let cursor = SearchCursor::new(0, 20, 5.0);
        let d = ScoredDoc {
            score: 5.0,
            address: DocAddress::new(0, 10),
        };
        assert!(d.should_skip(&cursor));
    }

    #[test]
    fn keep_same_score_same_seg_later_doc() {
        let cursor = SearchCursor::new(0, 20, 5.0);
        let d = ScoredDoc {
            score: 5.0,
            address: DocAddress::new(0, 25),
        };
        assert!(!d.should_skip(&cursor));
    }

    #[test]
    fn keep_same_score_later_segment() {
        let cursor = SearchCursor::new(0, 20, 5.0);
        let d = ScoredDoc {
            score: 5.0,
            address: DocAddress::new(1, 5),
        };
        assert!(!d.should_skip(&cursor));
    }

    #[test]
    fn skip_same_score_earlier_segment() {
        let cursor = SearchCursor::new(1, 5, 5.0);
        let d = ScoredDoc {
            score: 5.0,
            address: DocAddress::new(0, 100),
        };
        assert!(d.should_skip(&cursor));
    }

    #[test]
    fn nan_score_does_not_panic() {
        let nan_doc = ScoredDoc {
            score: f32::NAN,
            address: DocAddress::new(0, 10),
        };
        let c = SearchCursor::new(0, 20, 5.0);
        let _ = nan_doc.should_skip(&c);
    }

    fn doc(score: f32, seg: u32, id: u32) -> ScoredDoc {
        ScoredDoc {
            score,
            address: DocAddress::new(seg, id),
        }
    }

    #[test]
    fn scored_doc_ordering_matches_global_order() {
        let mut docs = vec![
            doc(3.0, 1, 2),
            doc(5.0, 0, 20),
            doc(5.0, 0, 10),
            doc(3.0, 0, 5),
            doc(5.0, 1, 5),
        ];
        docs.sort();

        assert_eq!(docs[0], doc(5.0, 0, 10));
        assert_eq!(docs[1], doc(5.0, 0, 20));
        assert_eq!(docs[2], doc(5.0, 1, 5));
        assert_eq!(docs[3], doc(3.0, 0, 5));
        assert_eq!(docs[4], doc(3.0, 1, 2));
    }

    #[test]
    fn simple_then_paging_covers_all_docs() {
        let seg0_docs = vec![
            doc(5.0, 0, 1),
            doc(4.0, 0, 2),
            doc(3.0, 0, 3),
        ];
        let seg1_docs = vec![doc(4.5, 1, 1), doc(2.0, 1, 2)];

        let page_size = 2;
        let mut all: Vec<ScoredDoc> = seg0_docs
            .into_iter()
            .chain(seg1_docs.into_iter())
            .collect();
        all.sort();
        let page1: Vec<ScoredDoc> = all.iter().take(page_size).cloned().collect();

        assert_eq!(page1.len(), 2);
        assert_eq!(page1[0].score, 5.0);
        assert_eq!(page1[1].score, 4.5);

        let cursor = SearchCursor::new(
            page1[1].address.segment_ord,
            page1[1].address.doc_id,
            page1[1].score,
        );

        let page2: Vec<ScoredDoc> = all
            .iter()
            .filter(|d| !d.should_skip(&cursor))
            .take(page_size)
            .cloned()
            .collect();

        assert_eq!(page2.len(), 2);
        assert_eq!(page2[0].score, 4.0);
        assert_eq!(page2[1].score, 3.0);

        let cursor2 = SearchCursor::new(
            page2[1].address.segment_ord,
            page2[1].address.doc_id,
            page2[1].score,
        );

        let page3: Vec<ScoredDoc> = all
            .iter()
            .filter(|d| !d.should_skip(&cursor2))
            .take(page_size)
            .cloned()
            .collect();

        assert_eq!(page3.len(), 1);
        assert_eq!(page3[0].score, 2.0);

        let mut all_seen: Vec<ScoredDoc> = page1
            .iter()
            .chain(page2.iter())
            .chain(page3.iter())
            .cloned()
            .collect();
        all_seen.sort_by_key(|d| (d.address.segment_ord, d.address.doc_id));
        assert_eq!(all_seen.len(), 5);
    }
}
