use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use crate::results::RustIeResult;
use crate::types::{Span, SpanWithCaptures, NamedCapture};

/// Request model for querying documents
#[derive(Debug, Deserialize, ToSchema)]
pub struct QueryRequest {
    /// The Odinson query string to execute
    #[schema(example = "[word=John] >nsubj [pos=/VB.*/]")]
    pub query: String,
    /// Maximum number of results to return (optional)
    #[serde(default = "default_limit")]
    #[schema(example = 10, default = 10)]
    pub limit: usize,
}

/// Response model for query results
#[derive(Debug, Serialize, ToSchema)]
pub struct QueryResponse {
    /// The original query string
    #[schema(example = "[word=John]")]
    pub query: String,
    /// Query execution duration in seconds
    #[schema(example = 0.023)]
    pub duration: f32,
    /// Total number of documents in the index
    #[schema(example = 100)]
    pub total_docs: usize,
    /// Number of results returned
    #[schema(example = 5)]
    pub result_count: usize,
    /// Maximum score among results
    #[schema(example = 1.5)]
    pub max_score: Option<f32>,
    /// The query results
    pub results: Vec<DocumentResult>,
}

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

fn default_limit() -> usize {
    10
}

impl From<RustIeResult> for QueryResponse {
    fn from(odin_results: RustIeResult) -> Self {
        let results: Vec<DocumentResult> = odin_results
            .score_docs()
            .iter()
            .map(|score_doc| DocumentResult {
                odinson_doc: score_doc.doc.doc_id,
                score: score_doc.score,
                document_id: format!("doc_{}", score_doc.doc.doc_id),
                sentence_index: 0,
                words: Vec::new(),
                matches: Vec::new(),
            })
            .collect();

        QueryResponse {
            query: "".to_string(),
            duration: 0.0,
            total_docs: odin_results.total_hits,
            result_count: results.len(),
            max_score: odin_results.max_score,
            results,
        }
    }
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
