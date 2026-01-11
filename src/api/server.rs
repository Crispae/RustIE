use actix_web::{web, App, HttpServer, middleware};
use anyhow::Result;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;
use crate::engine::ExtractorEngine;
use crate::api::handlers::{health_check, query_documents, simple_query, index_stats};
use crate::api::models::{
    QueryRequest, QueryResponse, ErrorResponse, DocumentResult,
    MatchResult, SpanResult, NamedCaptureResult, HealthResponse, StatsResponse
};

/// OpenAPI documentation
#[derive(OpenApi)]
#[openapi(
    info(
        title = "RustIE API",
        version = "1.0.0",
        description = "Information Extraction API for querying documents using Odinson-style patterns.

## Query Syntax

### Basic Constraints
- `[word=John]` - Match exact word
- `[pos=NN]` - Match POS tag
- `[lemma=eat]` - Match lemma
- `[word=/J.*/]` - Regex match

### Boolean Logic
- `[pos=NN & word=cat]` - AND
- `[word=John | word=Mary]` - OR
- `[pos=NN & !word=cat]` - NOT

### Sequences
- `[pos=DT] [pos=NN]` - Sequence
- `[pos=JJ]* [pos=NN]` - Zero or more
- `[pos=JJ]+ [pos=NN]` - One or more
- `[pos=DT]? [pos=NN]` - Optional

### Graph Traversal
- `[word=eats] >nsubj [word=John]` - Outgoing edge
- `[word=pizza] <dobj [word=eats]` - Incoming edge
- `[pos=/VB.*/] >nsubj [pos=/NN.*/]` - Any field works

### Named Captures
- `(?<subject>[pos=NNP])` - Capture with name
",
        license(name = "MIT")
    ),
    paths(
        crate::api::handlers::health_check,
        crate::api::handlers::query_documents,
        crate::api::handlers::simple_query,
        crate::api::handlers::index_stats
    ),
    components(schemas(
        QueryRequest, QueryResponse, ErrorResponse, DocumentResult,
        MatchResult, SpanResult, NamedCaptureResult, HealthResponse, StatsResponse
    )),
    tags(
        (name = "Health", description = "Health check endpoints"),
        (name = "Query", description = "Document query endpoints"),
        (name = "Index", description = "Index management endpoints")
    )
)]
pub struct ApiDoc;

/// Configuration for the API server
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub index_path: String,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            index_path: "index".to_string(),
        }
    }
}

/// Start the API server
pub async fn start_server(config: ApiConfig) -> Result<()> {
    // Initialize the extractor engine
    let engine = ExtractorEngine::from_path(&config.index_path)?;
    let engine_data = web::Data::new(engine);

    log::info!("Starting RustIE API server on {}:{}", config.host, config.port);
    log::info!("Index path: {}", config.index_path);
    log::info!("Swagger UI: http://{}:{}/swagger-ui/", config.host, config.port);

    // Generate OpenAPI spec
    let openapi = ApiDoc::openapi();

    // Start the HTTP server
    HttpServer::new(move || {
        App::new()
            .wrap(middleware::Logger::default())
            .app_data(engine_data.clone())
            // Swagger UI
            .service(
                SwaggerUi::new("/swagger-ui/{_:.*}")
                    .url("/api-docs/openapi.json", openapi.clone())
            )
            // API routes
            .service(
                web::scope("/api/v1")
                    .route("/health", web::get().to(health_check))
                    .route("/query", web::post().to(query_documents))
                    .route("/query/{query}", web::get().to(simple_query))
                    .route("/stats", web::get().to(index_stats))
            )
    })
    .bind(format!("{}:{}", config.host, config.port))?
    .run()
    .await?;

    Ok(())
}

/// Start the API server with default configuration
pub async fn start_server_default() -> Result<()> {
    start_server(ApiConfig::default()).await
}
