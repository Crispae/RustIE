use rustie::{start_server, ApiConfig};
use anyhow::Result;
use log;
use clap::Parser;

#[derive(Parser)]
#[command(name = "rustie-api")]
#[command(about = "RustIE Query API Server")]
struct Args {
    /// Host to bind the server to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    
    /// Port to bind the server to
    #[arg(long, default_value = "8080")]
    port: u16,
    
    /// Path to the index directory
    #[arg(long, default_value = "index")]
    index_path: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    // Parse command line arguments
    let args = Args::parse();
    
    log::info!("Starting RustIE API Server...");
    log::info!("Configuration: host={}, port={}, index_path={}", 
               args.host, args.port, args.index_path);
    
    // Create configuration
    let config = ApiConfig {
        host: args.host,
        port: args.port,
        index_path: args.index_path,
    };
    
    // Start the server
    start_server(config).await?;
    
    Ok(())
} 