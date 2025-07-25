use serde::{Deserialize, Serialize};
use crate::results::RustIeResult;
use crate::types::{Span, SpanWithCaptures, NamedCapture};

/// Request model for querying documents
#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    /// The Odinson query string to execute
    pub query: String,
    /// Maximum number of results to return (optional)
    #[serde(default = "default_limit")]
    pub limit: usize,
}

/// Response model for query results
#[derive(Debug, Serialize)]
pub struct QueryResponse {
    /// The original query string
    pub query: String,
    /// Query execution duration in seconds
    pub duration: f32,
    /// Total number of documents in the index
    pub total_docs: usize,
    /// Number of results returned
    pub result_count: usize,
    /// Maximum score among results
    pub max_score: Option<f32>,
    /// The query results
    pub results: Vec<DocumentResult>,
}

/// Individual document result with detailed information
#[derive(Debug, Serialize)]
pub struct DocumentResult {
    /// Internal Odinson document ID
    pub odinson_doc: u32,
    /// Document score
    pub score: f32,
    /// Document ID from the original document
    pub document_id: String,
    /// Sentence index within the document
    pub sentence_index: u32,
    /// Array of tokens (words) in the sentence
    pub words: Vec<String>,
    /// Array of matches found in this document
    pub matches: Vec<MatchResult>,
}

/// Individual match result with spans and captures
#[derive(Debug, Serialize)]
pub struct MatchResult {
    /// The span of the match
    pub span: SpanResult,
    /// Named captures within this match
    pub captures: Vec<NamedCaptureResult>,
}

/// Span information
#[derive(Debug, Serialize)]
pub struct SpanResult {
    /// Start position (inclusive)
    pub start: usize,
    /// End position (exclusive)
    pub end: usize,
}

/// Named capture result
#[derive(Debug, Serialize)]
pub struct NamedCaptureResult {
    /// Name of the capture
    pub name: String,
    /// Span of the captured text
    pub span: SpanResult,
}

/// Error response model
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    /// Error message
    pub error: String,
    /// Error type
    pub error_type: String,
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
                sentence_index: 0, // TODO: Extract actual sentence index
                words: Vec::new(), // TODO: Extract actual tokens
                matches: Vec::new(), // TODO: Extract actual matches
            })
            .collect();

        QueryResponse {
            query: "".to_string(), // TODO: Pass the original query
            duration: 0.0, // TODO: Calculate actual duration
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