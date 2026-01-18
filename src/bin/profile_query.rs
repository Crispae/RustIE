//! Simple profiling test for a specific query
//!
//! Usage:
//!   cargo build --release
//!   samply record ./target/release/profile_query.exe --index ./index
//!
//! Or with a specific index:
//!   samply record ./target/release/profile_query.exe --index ./test_api_index

use rustie::ExtractorEngine;
use rustie::tantivy_integration::graph_traversal::GraphTraversalStats;
use std::path::Path;
use clap::Parser;
use anyhow::Result;

// Initialize logging to see debug/info messages
fn init_logging() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
}

#[derive(Parser)]
#[command(name = "profile_query")]
#[command(about = "Profile a specific query")]
struct Args {
    /// Index directory path
    #[arg(short, long, default_value = "./index")]
    index: String,
    
    /// Number of iterations to run (for better profiling data)
    #[arg(short, long, default_value_t = 100)]
    iterations: u32,
}

fn main() -> Result<()> {
    // Initialize logging to see regex expansion messages
    init_logging();
    
    let args = Args::parse();
    
    println!("Loading index from: {}", args.index);
    let engine = ExtractorEngine::from_path(&args.index)?;
    
    println!("Index loaded: {} documents", engine.num_docs());
    println!("\nQuery: ([tag=/N.*/]) <nsubj ([tag=/V.*/]) >dobj ([tag=/N.*/])");
    println!("Running {} iterations...\n", args.iterations);
    
    let query = "([tag=/N.*/]) <nsubj ([tag=/V.*/]) >dobj ([tag=/N.*/])";
    
    // Reset stats before starting
    GraphTraversalStats::reset();
    
    let start = std::time::Instant::now();
    let mut total_hits = 0;
    
    for i in 0..args.iterations {
        match engine.query(query) {
            Ok(result) => {
                total_hits += result.total_hits;
                if i == 0 {
                    println!("First run: {} hits", result.total_hits);
                    if let Some(max_score) = result.max_score {
                        println!("Max score: {:.4}", max_score);
                    }
                    println!("Sample results: {} sentences", result.sentence_results.len());
                }
            }
            Err(e) => {
                eprintln!("Query error: {}", e);
                return Err(e.into());
            }
        }
    }
    
    let elapsed = start.elapsed();
    let avg_time = elapsed.as_millis() as f64 / args.iterations as f64;
    
    println!("\n=== Results ===");
    println!("Total iterations: {}", args.iterations);
    println!("Total time: {:.2}ms", elapsed.as_millis());
    println!("Average time per query: {:.2}ms", avg_time);
    println!("Total hits across all runs: {}", total_hits);
    println!("Average hits per query: {:.2}", total_hits as f64 / args.iterations as f64);
    
    // Print detailed performance statistics
    let stats = GraphTraversalStats::get();
    stats.print_summary();
    
    Ok(())
}