//! Shared pagination logic for basic (collector) and graph (segment loop) execution paths.
//! Both paths feed sorted items into [PaginationDriver::paginate] and get identical
//! cursor/skip/take behavior.

use crate::tantivy_integration::paging_collector::ScoredDoc;
use crate::types::SearchCursor;
use std::collections::HashSet;
use tantivy::DocAddress;

/// Trait for anything that can be paginated.
/// Both [ScoredDoc] (basic path) and [crate::engine::execution::ScoredSentence] (graph path) implement this.
pub trait Paginatable: Ord + Clone {
    /// Score for cursor ordering.
    fn score(&self) -> f32;

    /// DocAddress for cursor construction and ordering.
    fn doc_address(&self) -> DocAddress;

    /// Dedup key — None if dedup is not needed (basic path).
    fn dedup_key(&self) -> Option<(String, String)> {
        None
    }

    /// Convert to ScoredDoc for cursor comparison.
    fn as_scored_doc(&self) -> ScoredDoc {
        ScoredDoc {
            score: self.score(),
            address: self.doc_address(),
        }
    }

    /// Whether this item should be skipped given the cursor (at or before cursor in global order).
    fn should_skip(&self, cursor: &SearchCursor) -> bool {
        self.as_scored_doc().should_skip(cursor)
    }
}

impl Paginatable for ScoredDoc {
    fn score(&self) -> f32 {
        self.score
    }

    fn doc_address(&self) -> DocAddress {
        self.address
    }
}

/// Result of pagination — generic over the item type.
pub struct PaginatedPage<T> {
    pub total_hits: usize,
    pub items: Vec<T>,
    pub next_cursor: Option<SearchCursor>,
}

/// Shared pagination driver. Both basic and graph paths feed sorted items into this
/// and get back one page with cursor and total_hits.
pub struct PaginationDriver {
    pub page_size: usize,
    pub after: Option<SearchCursor>,
}

impl PaginationDriver {
    pub fn new(page_size: usize, after: Option<SearchCursor>) -> Self {
        Self { page_size, after }
    }

    /// Paginate a pre-sorted list of items.
    ///
    /// Items MUST be sorted in global order (score DESC, segment_ord ASC, doc_id ASC)
    /// before calling this method.
    ///
    /// Steps: dedup (if items have dedup keys), count total_hits, skip past cursor,
    /// take page_size, build next_cursor.
    pub fn paginate<T: Paginatable>(&self, mut items: Vec<T>) -> PaginatedPage<T> {
        // Step 1: Dedup if items support it
        let has_dedup = items.first().and_then(|i| i.dedup_key()).is_some();
        if has_dedup {
            let mut seen = HashSet::new();
            items.retain(|item| {
                match item.dedup_key() {
                    Some(key) => seen.insert(key),
                    None => true,
                }
            });
        }

        let total_hits = items.len();

        // Step 2: Skip past cursor
        let start_index = match &self.after {
            Some(cursor) => items
                .iter()
                .position(|item| !item.should_skip(cursor))
                .unwrap_or(items.len()),
            None => 0,
        };

        // Step 3: Take page_size
        let page: Vec<T> = items
            .into_iter()
            .skip(start_index)
            .take(self.page_size)
            .collect();

        // Step 4: Build next_cursor only when there is at least one more item after this page
        let next_cursor =
            if page.len() == self.page_size && start_index + page.len() < total_hits {
                page.last().map(|last| {
                    SearchCursor::new(
                        last.doc_address().segment_ord,
                        last.doc_address().doc_id,
                        last.score(),
                    )
                })
            } else {
                None
            };

        PaginatedPage {
            total_hits,
            items: page,
            next_cursor,
        }
    }

    /// Build next_cursor for an already-fetched page (e.g. from collectors).
    /// Use the same rule as [paginate]: emit cursor iff there is a next page.
    pub fn next_cursor_for_page(
        last: &ScoredDoc,
        total_hits: usize,
        page_size: usize,
        current_page_len: usize,
    ) -> Option<SearchCursor> {
        if current_page_len == page_size && total_hits > page_size {
            Some(SearchCursor::new(
                last.address.segment_ord,
                last.address.doc_id,
                last.score,
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tantivy::DocAddress;

    fn doc(score: f32, seg: u32, id: u32) -> ScoredDoc {
        ScoredDoc {
            score,
            address: DocAddress::new(seg, id),
        }
    }

    #[test]
    fn empty_input_produces_no_cursor() {
        let driver = PaginationDriver::new(10, None);

        let page: PaginatedPage<ScoredDoc> = driver.paginate(vec![]);
        assert_eq!(page.total_hits, 0);
        assert!(page.items.is_empty());
        assert!(page.next_cursor.is_none());
    }

    #[test]
    fn page_size_larger_than_results() {
        let items = vec![doc(5.0, 0, 1), doc(4.0, 0, 2)];
        let driver = PaginationDriver::new(10, None);
        let page = driver.paginate(items);

        assert_eq!(page.total_hits, 2);
        assert_eq!(page.items.len(), 2);
        assert!(page.next_cursor.is_none());
    }

    #[test]
    fn cursor_past_all_items_returns_empty() {
        let items = vec![doc(5.0, 0, 1), doc(4.0, 0, 2)];
        let cursor = SearchCursor::new(1, 999, 1.0);
        let driver = PaginationDriver::new(10, Some(cursor));
        let page = driver.paginate(items);

        assert_eq!(page.total_hits, 2);
        assert!(page.items.is_empty());
        assert!(page.next_cursor.is_none());
    }

    #[test]
    fn first_page_returns_next_cursor_when_more_items() {
        let items = vec![
            doc(5.0, 0, 1),
            doc(4.5, 1, 1),
            doc(4.0, 0, 2),
            doc(3.0, 0, 3),
        ];
        let driver = PaginationDriver::new(2, None);
        let page = driver.paginate(items);

        assert_eq!(page.total_hits, 4);
        assert_eq!(page.items.len(), 2);
        let c = page.next_cursor.unwrap();
        assert_eq!(c.segment_ord, 1);
        assert_eq!(c.doc_id, 1);
        assert!((c.score - 4.5).abs() < 1e-6);
    }

    #[test]
    fn second_page_using_cursor() {
        let items = vec![
            doc(5.0, 0, 1),
            doc(4.5, 1, 1),
            doc(4.0, 0, 2),
            doc(3.0, 0, 3),
        ];
        let cursor = SearchCursor::new(1, 1, 4.5);
        let driver = PaginationDriver::new(2, Some(cursor));
        let page = driver.paginate(items);

        assert_eq!(page.total_hits, 4);
        assert_eq!(page.items.len(), 2);
        assert!((page.items[0].score - 4.0).abs() < 1e-6);
        assert!((page.items[1].score - 3.0).abs() < 1e-6);
    }
}
