//! RustIE API Server
//!
//! Starts an HTTP server for querying the RustIE index.
//!
//! Usage:
//!   cargo run --bin api_server -- --index ./my_index
//!   cargo run --bin api_server -- --index ./my_index --host 0.0.0.0 --port 3000

use clap::Parser;
use rustie::api::server::{start_server, ApiConfig};

#[derive(Parser, Debug)]
#[command(name = "rustie-api")]
#[command(about = "RustIE Information Extraction API Server")]
#[command(version)]
struct Args {
    /// Path to the Tantivy index directory
    #[arg(short, long, required = true)]
    index: String,

    /// Host address to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 8080)]
    port: u16,
}

#[actix_web::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse command-line arguments
    let args = Args::parse();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║              RustIE API Server                                ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Index:    {}", args.index);
    println!("  Server:   http://{}:{}", args.host, args.port);
    println!();
    println!("  Endpoints:");
    println!("    GET  /api/v1/health       - Health check");
    println!("    GET  /api/v1/stats        - Index statistics");
    println!("    GET  /api/v1/query/{{q}}    - Simple query");
    println!("    POST /api/v1/query        - Full query (JSON body)");
    println!();

    // Create configuration
    let config = ApiConfig {
        host: args.host,
        port: args.port,
        index_path: args.index,
    };

    // Start the server
    start_server(config).await
}
