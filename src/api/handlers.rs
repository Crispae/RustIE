use actix_web::{web, HttpResponse, Result};
use crate::engine::ExtractorEngine;
use crate::api::models::{QueryRequest, QueryResponse, ErrorResponse, DocumentResult, MatchResult};
use std::time::Instant;

/// Health check endpoint
pub async fn health_check() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "service": "RustIE Query API"
    })))
}

/// Query documents endpoint
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
                total_docs: engine.num_docs(),
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
                total_docs: engine.num_docs(),
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
pub async fn index_stats(engine: web::Data<ExtractorEngine>) -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "total_documents": engine.num_docs(),
        "index_path": "index", // TODO: Make this configurable
        "schema_fields": vec![
            "word", "lemma", "pos", "tag", "doc_id", 
            "sentence_id", "sentence_length", "dependencies"
        ]
    })))
}

/// Convert RustIeResult to detailed DocumentResult with tokens and matches
fn convert_to_detailed_results(engine: &ExtractorEngine, odin_results: crate::results::RustIeResult) -> Vec<DocumentResult> {
    let mut detailed_results = Vec::new();
    
    for score_doc in odin_results.score_docs() {
        // Try to get document content
        let (document_id, sentence_index, words) = match engine.doc(score_doc.doc) {
            Ok(doc) => {
                // For now, use fallback values since we need to implement proper field access
                let doc_id = format!("doc_{}", score_doc.doc.doc_id);
                let sent_idx = 0u32;
                let tokens = Vec::new(); // TODO: Implement proper token extraction
                
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