use actix_web::{web, App, HttpServer, middleware};
use anyhow::Result;
use crate::engine::ExtractorEngine;
use crate::api::handlers::{health_check, query_documents, simple_query, index_stats};

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

    // Start the HTTP server
    HttpServer::new(move || {
        App::new()
            .wrap(middleware::Logger::default())
            .app_data(engine_data.clone())
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