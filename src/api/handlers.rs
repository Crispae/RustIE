use actix_web::{web, HttpResponse, Result};
use crate::engine::ExtractorEngine;
use crate::api::models::{QueryRequest, QueryResponse, ErrorResponse, DocumentResult, MatchResult, HealthResponse, StatsResponse};
use std::time::Instant;

/// Health check endpoint
#[utoipa::path(
    get,
    path = "/api/v1/health",
    tag = "Health",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse)
    )
)]
pub async fn health_check() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(HealthResponse {
        status: "healthy".to_string(),
        service: "RustIE Query API".to_string(),
    }))
}

/// Query documents with full options
#[utoipa::path(
    post,
    path = "/api/v1/query",
    tag = "Query",
    request_body = QueryRequest,
    responses(
        (status = 200, description = "Query executed successfully", body = QueryResponse),
        (status = 400, description = "Invalid query", body = ErrorResponse),
        (status = 500, description = "Query execution failed", body = ErrorResponse)
    )
)]
pub async fn query_documents(
    engine: web::Data<ExtractorEngine>,
    request: web::Json<QueryRequest>,
) -> Result<HttpResponse> {
    // Validate request
    if request.query.trim().is_empty() {
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: "Query cannot be empty".to_string(),
            error_type: "ValidationError".to_string(),
        }));
    }

    // Execute query with timing
    let start_time = Instant::now();
    match engine.query_with_limit(&request.query, request.limit) {
        Ok(odin_results) => {
            let duration = start_time.elapsed().as_secs_f32();

            // Convert to detailed response
            let results = convert_to_detailed_results(&engine, odin_results);

            let response = QueryResponse {
                query: request.query.clone(),
                duration,
                result_count: results.len(),
                max_score: results.iter().map(|r| r.score).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
                results,
            };

            Ok(HttpResponse::Ok().json(response))
        }
        Err(e) => {
            log::error!("Query execution failed: {}", e);
            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Query execution failed: {}", e),
                error_type: "QueryError".to_string(),
            }))
        }
    }
}

/// Simple query endpoint that accepts query as URL parameter
#[utoipa::path(
    get,
    path = "/api/v1/query/{query}",
    tag = "Query",
    params(
        ("query" = String, Path, description = "The Odinson query string (URL encoded)")
    ),
    responses(
        (status = 200, description = "Query executed successfully", body = QueryResponse),
        (status = 400, description = "Invalid query", body = ErrorResponse),
        (status = 500, description = "Query execution failed", body = ErrorResponse)
    )
)]
pub async fn simple_query(
    engine: web::Data<ExtractorEngine>,
    path: web::Path<String>,
) -> Result<HttpResponse> {
    let query = path.into_inner();

    if query.trim().is_empty() {
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: "Query cannot be empty".to_string(),
            error_type: "ValidationError".to_string(),
        }));
    }

    // Execute query with timing
    let start_time = Instant::now();
    match engine.query_with_limit(&query, 10) {
        Ok(odin_results) => {
            let duration = start_time.elapsed().as_secs_f32();

            // Convert to detailed response
            let results = convert_to_detailed_results(&engine, odin_results);

            let response = QueryResponse {
                query: query.clone(),
                duration,
                result_count: results.len(),
                max_score: results.iter().map(|r| r.score).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
                results,
            };

            Ok(HttpResponse::Ok().json(response))
        }
        Err(e) => {
            log::error!("Query execution failed: {}", e);
            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Query execution failed: {}", e),
                error_type: "QueryError".to_string(),
            }))
        }
    }
}

/// Get index statistics
#[utoipa::path(
    get,
    path = "/api/v1/stats",
    tag = "Index",
    responses(
        (status = 200, description = "Index statistics", body = StatsResponse)
    )
)]
pub async fn index_stats(engine: web::Data<ExtractorEngine>) -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(StatsResponse {
        total_docs: engine.num_docs(),
        index_path: "index".to_string(),
        fields: vec![
            "word".to_string(),
            "lemma".to_string(),
            "pos".to_string(),
            "tag".to_string(),
            "doc_id".to_string(),
            "sentence_id".to_string(),
            "sentence_length".to_string(),
            "dependencies".to_string()
        ],
    }))
}

/// Convert RustIeResult to detailed DocumentResult with tokens and matches
fn convert_to_detailed_results(engine: &ExtractorEngine, odin_results: crate::results::RustIeResult) -> Vec<DocumentResult> {
    let mut detailed_results = Vec::new();

    // Prefer sentence_results as they contain richer data (words, matches, fields)
    if !odin_results.sentence_results.is_empty() {
        for sentence in odin_results.sentence_results {
            let matches: Vec<MatchResult> = sentence.matches
                .iter()
                .map(|m| m.clone().into())
                .collect();
            
            // Get words from fields
            let words = sentence.fields.get("word").cloned().unwrap_or_default();

            let detailed_result = DocumentResult {
                odinson_doc: 0, // Doc ID might not be preserved in SentenceResult directly as u32
                score: sentence.score,
                document_id: sentence.document_id,
                sentence_index: sentence.sentence_id.parse().unwrap_or(0),
                words,
                matches,
            };
            detailed_results.push(detailed_result);
        }
        return detailed_results;
    }

    // Fallback to score_docs if sentence_results are missing (legacy path)
    for score_doc in odin_results.score_docs() {
        // Try to get document content
        let (document_id, sentence_index, words) = match engine.doc(score_doc.doc) {
            Ok(_doc) => {
                // For now, use fallback values since we need to implement proper field access
                let doc_id = format!("doc_{}", score_doc.doc.doc_id);
                let sent_idx = 0u32;
                let tokens = Vec::new();

                (doc_id, sent_idx, tokens)
            }
            Err(_) => {
                // Fallback if document retrieval fails
                (format!("doc_{}", score_doc.doc.doc_id), 0, Vec::new())
            }
        };

        // Convert matches
        let matches: Vec<MatchResult> = score_doc.get_matches()
            .iter()
            .map(|span_with_captures| span_with_captures.clone().into())
            .collect();

        let detailed_result = DocumentResult {
            odinson_doc: score_doc.doc.doc_id,
            score: score_doc.score,
            document_id,
            sentence_index,
            words,
            matches,
        };

        detailed_results.push(detailed_result);
    }

    detailed_results
}
