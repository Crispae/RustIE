use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use crate::types::{Span, SpanWithCaptures, NamedCapture, SearchCursor};

/// Individual document result with detailed information
#[derive(Debug, Serialize, ToSchema)]
pub struct DocumentResult {
    /// Internal Odinson document ID
    #[schema(example = 1)]
    pub odinson_doc: u32,
    /// Document score
    #[schema(example = 1.5)]
    pub score: f32,
    /// Document ID from the original document
    #[schema(example = "doc_simple")]
    pub document_id: String,
    /// Sentence index within the document
    #[schema(example = 0)]
    pub sentence_index: u32,
    /// Array of tokens (words) in the sentence
    #[schema(example = json!(["John", "eats", "pizza"]))]
    pub words: Vec<String>,
    /// Array of matches found in this document
    pub matches: Vec<MatchResult>,
}

/// Individual match result with spans and captures
#[derive(Debug, Serialize, ToSchema)]
pub struct MatchResult {
    /// The span of the match
    pub span: SpanResult,
    /// Named captures within this match
    pub captures: Vec<NamedCaptureResult>,
}

/// Span information
#[derive(Debug, Serialize, ToSchema)]
pub struct SpanResult {
    /// Start position (inclusive)
    #[schema(example = 0)]
    pub start: usize,
    /// End position (exclusive)
    #[schema(example = 2)]
    pub end: usize,
}

/// Named capture result
#[derive(Debug, Serialize, ToSchema)]
pub struct NamedCaptureResult {
    /// Name of the capture
    #[schema(example = "subject")]
    pub name: String,
    /// Span of the captured text
    pub span: SpanResult,
}

/// Error response model
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    /// Error message
    #[schema(example = "Query cannot be empty")]
    pub error: String,
    /// Error type
    #[schema(example = "ValidationError")]
    pub error_type: String,
}

/// Health check response
#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    /// Service status
    #[schema(example = "healthy")]
    pub status: String,
    /// Service name
    #[schema(example = "RustIE Query API")]
    pub service: String,
}

/// Index statistics response
#[derive(Debug, Serialize, ToSchema)]
pub struct StatsResponse {
    /// Total number of documents in the index
    #[schema(example = 100)]
    pub total_docs: usize,
    /// Path to the index
    #[schema(example = "./test_api_index")]
    pub index_path: String,
    /// List of indexed fields
    #[schema(example = json!(["word", "pos", "lemma", "entity"]))]
    pub fields: Vec<String>,
}

/// Cursor for the next page of paginated search (alias for API; same shape as SearchCursor).
pub type AfterCursor = SearchCursor;

/// Request for Odinson-style paginated search.
#[derive(Debug, Deserialize, ToSchema)]
pub struct PaginatedQueryRequest {
    /// The Odinson query string to execute.
    #[schema(example = "[entity=/B-Gene/]")]
    pub query: String,
    /// Page size (default 15).
    #[serde(default = "default_page_size")]
    #[schema(example = 15, default = 15)]
    pub page_size: usize,
    /// Cursor from the previous page's response; omit for the first page.
    pub after: Option<AfterCursor>,
}

fn default_page_size() -> usize {
    15
}

/// Response for paginated search with total_hits and next_cursor.
#[derive(Debug, Serialize, ToSchema)]
pub struct PaginatedQueryResponse {
    /// The original query string
    pub query: String,
    /// Query execution duration in seconds
    pub duration: f32,
    /// Total number of matching documents (all pages)
    pub total_hits: usize,
    /// Number of results in this page
    pub result_count: usize,
    /// Maximum score among results in this page
    pub max_score: Option<f32>,
    /// Results for the current page only
    pub results: Vec<DocumentResult>,
    /// Cursor for the next page; absent if this is the last page
    pub next_cursor: Option<AfterCursor>,
}

// Helper conversion functions
impl From<Span> for SpanResult {
    fn from(span: Span) -> Self {
        Self {
            start: span.start,
            end: span.end,
        }
    }
}

impl From<NamedCapture> for NamedCaptureResult {
    fn from(capture: NamedCapture) -> Self {
        Self {
            name: capture.name,
            span: capture.span.into(),
        }
    }
}

impl From<SpanWithCaptures> for MatchResult {
    fn from(span_with_captures: SpanWithCaptures) -> Self {
        Self {
            span: span_with_captures.span.into(),
            captures: span_with_captures.captures.into_iter().map(|c| c.into()).collect(),
        }
    }
}
