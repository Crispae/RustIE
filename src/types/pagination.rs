// src/types/pagination.rs

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Identifies a position in search results for cursor-based pagination.
///
/// Results are ordered by (score DESC, segment_ord ASC, doc_id ASC).
/// The cursor points to the last result of the previous page; the next page
/// starts immediately after this position in the global order.
///
/// Used by:
/// - `paging_collector`: to skip docs at or before the cursor
/// - `engine/execution`: to build and return cursors
/// - `api/models`: serialized as the `after` field in requests/responses
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct SearchCursor {
    /// Tantivy segment ordinal of the document
    pub segment_ord: u32,
    /// Document ID within the segment
    pub doc_id: u32,
    /// BM25 (or constant) score of the document
    pub score: f32,
}

impl SearchCursor {
    pub fn new(segment_ord: u32, doc_id: u32, score: f32) -> Self {
        Self {
            segment_ord,
            doc_id,
            score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_cursor_new() {
        let cursor = SearchCursor::new(0, 42, 3.5);
        assert_eq!(cursor.segment_ord, 0);
        assert_eq!(cursor.doc_id, 42);
        assert!((cursor.score - 3.5).abs() < f32::EPSILON);
    }

    #[test]
    fn search_cursor_clone() {
        let cursor = SearchCursor::new(1, 10, 2.0);
        let cloned = cursor.clone();
        assert_eq!(cursor, cloned);
    }

    #[test]
    fn search_cursor_serialize_roundtrip() {
        let cursor = SearchCursor::new(0, 456, 3.14);
        let json = serde_json::to_string(&cursor).unwrap();
        let deserialized: SearchCursor = serde_json::from_str(&json).unwrap();
        assert_eq!(cursor, deserialized);
    }

    #[test]
    fn search_cursor_deserialize_from_json() {
        let json = r#"{"segment_ord": 2, "doc_id": 100, "score": 5.5}"#;
        let cursor: SearchCursor = serde_json::from_str(json).unwrap();
        assert_eq!(cursor.segment_ord, 2);
        assert_eq!(cursor.doc_id, 100);
        assert!((cursor.score - 5.5).abs() < f32::EPSILON);
    }
}
