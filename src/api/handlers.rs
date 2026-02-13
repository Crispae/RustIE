use actix_web::{web, HttpResponse, Result};
use crate::engine::ExtractorEngine;
use crate::api::models::{QueryRequest, QueryResponse, ErrorResponse, DocumentResult, MatchResult, HealthResponse, StatsResponse};
use std::time::Instant;

/// Send-safe result type returned from the blocking closure (HttpResponse is not Send in actix-web 4).
enum QueryOutcome {
    Success(QueryResponse),
    Error(ErrorResponse),
}

/// Validate that the query string is non-empty, returning a BadRequest response if invalid.
fn validate_query(query: &str) -> Option<HttpResponse> {
    if query.trim().is_empty() {
        Some(HttpResponse::BadRequest().json(ErrorResponse {
            error: "Query cannot be empty".to_string(),
            error_type: "ValidationError".to_string(),
        }))
    } else {
        None
    }
}

/// Execute a query on the blocking pool — returns Send-safe data, NOT HttpResponse.
fn execute_query(engine: &ExtractorEngine, query: &str, limit: usize) -> QueryOutcome {
    let start_time = Instant::now();
    match engine.query_with_limit(query, limit) {
        Ok(odin_results) => {
            let duration = start_time.elapsed().as_secs_f32();
            let results = convert_to_detailed_results(engine, odin_results);

            QueryOutcome::Success(QueryResponse {
                query: query.to_string(),
                duration,
                result_count: results.len(),
                max_score: results.iter().map(|r| r.score).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
                results,
            })
        }
        Err(e) => {
            log::error!("Query execution failed: {}", e);
            QueryOutcome::Error(ErrorResponse {
                error: format!("Query execution failed: {}", e),
                error_type: "QueryError".to_string(),
            })
        }
    }
}

/// Convert QueryOutcome → HttpResponse (runs on async worker, cheap).
fn outcome_to_response(outcome: QueryOutcome) -> HttpResponse {
    match outcome {
        QueryOutcome::Success(resp) => HttpResponse::Ok().json(resp),
        QueryOutcome::Error(err) => HttpResponse::InternalServerError().json(err),
    }
}

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
    if let Some(err_response) = validate_query(&request.query) {
        return Ok(err_response);
    }

    let engine = engine.clone();
    let query = request.query.clone();
    let limit = request.limit;

    let outcome = web::block(move || execute_query(&engine, &query, limit)).await?;
    Ok(outcome_to_response(outcome))
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

    if let Some(err_response) = validate_query(&query) {
        return Ok(err_response);
    }

    let engine = engine.clone();
    let limit = 10usize;

    let outcome = web::block(move || execute_query(&engine, &query, limit)).await?;
    Ok(outcome_to_response(outcome))
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
    let engine = engine.clone();

    let stats = web::block(move || {
        let fields: Vec<String> = engine
            .schema()
            .fields()
            .map(|(_field, entry)| entry.name().to_string())
            .collect();

        StatsResponse {
            total_docs: engine.num_docs(),
            index_path: "index".to_string(),
            fields,
        }
    })
    .await?;

    Ok(HttpResponse::Ok().json(stats))
}

/// Convert RustIeResult to detailed DocumentResult with tokens and matches
fn convert_to_detailed_results(engine: &ExtractorEngine, odin_results: crate::results::RustIeResult) -> Vec<DocumentResult> {
    let mut detailed_results = Vec::new();

    if !odin_results.sentence_results.is_empty() {
        for sentence in odin_results.sentence_results {
            let matches: Vec<MatchResult> = sentence.matches
                .iter()
                .map(|m| m.clone().into())
                .collect();

            let words = sentence.fields.get("word").cloned().unwrap_or_default();

            let detailed_result = DocumentResult {
                odinson_doc: 0,
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

    for score_doc in odin_results.score_docs() {
        let (document_id, sentence_index, words) = match engine.doc(score_doc.doc) {
            Ok(_doc) => {
                let doc_id = format!("doc_{}", score_doc.doc.doc_id);
                let sent_idx = 0u32;
                let tokens = Vec::new();
                (doc_id, sent_idx, tokens)
            }
            Err(_) => {
                (format!("doc_{}", score_doc.doc.doc_id), 0, Vec::new())
            }
        };

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