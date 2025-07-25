use rustie::compiler::ast::{Matcher, Traversal};
use rustie::digraph::graph::DirectedGraph;
use rustie::digraph::traversal::{GraphTraversal, TraversalResult};
use rustie::compiler::QueryParser;
use rustie::engine::ExtractorEngine;
use std::path::Path;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "query_parser")]
#[command(about = "Parse and execute graph traversal queries")]
#[command(version)]
struct Args {
    /// Custom query to execute
    #[arg(short, long)]
    query: Option<String>,
    
    /// Index directory path
    #[arg(short, long, default_value = "test_index_output")]
    index: String,
}

fn main() {
    let args = Args::parse();
    
    // If a custom query is provided, run it directly
    if let Some(query) = args.query {
        println!("=== Executing Custom Query ===\n");
        println!("Query: {}", query);
        println!("Index: {}", args.index);
        println!();
        
        execute_custom_query(&query, &args.index);
        return;
    }

    println!("=== RustIE Graph Traversal Examples ===\n");

    // 1. Build a small dependency graph in memory
    // Example: John --nsubj--> works
    //          works --prep--> Microsoft
    //          John --amod--> smart
    let mut graph = DirectedGraph::new();
    // Add nodes
    graph.add_node(0);
    graph.add_node(1);
    graph.add_node(2);
    graph.add_node(3);

    // Add edges
    graph.add_edge(0, 1, "nsubj"); // John --nsubj--> works
    graph.add_edge(1, 2, "prep");  // works --prep--> Microsoft
    graph.add_edge(0, 3, "amod");  // John --amod--> smart

    // 2. Define a traversal pattern: outgoing nsubj from 'John'
    let traversal = Traversal::Outgoing(Matcher::String("nsubj".to_string()));
    let start_node = 0; // John

    // 3. Run the traversal using the traversal engine
    let engine = GraphTraversal::new(graph);
    let result = engine.execute(&traversal, &[start_node]);

    // 4. Print the results
    match result {
        TraversalResult::Success(nodes) => {
            println!("Nodes reached from 'John' via outgoing 'nsubj':");
            for node_id in nodes {
                println!("  Node {}", node_id);
            }
        }
        TraversalResult::NoTraversal => println!("No traversal performed."),
        TraversalResult::FailTraversal => println!("No nodes reached via this traversal."),
    }

    // 5. Show a more complex traversal: outgoing nsubj then outgoing prep (John --nsubj--> works --prep--> Microsoft)
    let complex_traversal = Traversal::Concatenated(vec![
        Traversal::Outgoing(Matcher::String("nsubj".to_string())),
        Traversal::Outgoing(Matcher::String("prep".to_string())),
    ]);
    
    let complex_result = engine.execute(&complex_traversal, &[start_node]);
    match complex_result {
        TraversalResult::Success(nodes) => {
            println!("\nNodes reached from 'John' via outgoing 'nsubj' then 'prep':");
            for node_id in nodes {
                println!("  Node {}", node_id);
            }
        }
        _ => println!("Complex traversal failed."),
    }

    println!("\n=== Odinson-Style Pattern Parsing ===\n");

    // Test pattern parsing
    let patterns = vec![
        "[word=John] >nsubj [pos=VBZ]",
        "[word=works] <nsubj [word=John]",
        "[word=John] >> [pos=NNP]",
        "[word=John] << [pos=VBZ]",
    ];

    for pattern in patterns {
        println!("--- Parsing: '{}' ---", pattern);
        match parse_and_compile_pattern(pattern) {
            Ok(ast) => {
                println!("AST: {:?}", ast);
                println!("Successfully compiled to Tantivy query");
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
        println!();
    }

    // Now test against the actual indexed data
    println!("=== Querying Indexed Data ===\n");
    test_indexed_queries();
}

fn parse_and_compile_pattern(pattern: &str) -> Result<rustie::compiler::ast::Pattern, Box<dyn std::error::Error>> {
    let parser = QueryParser::new("word".to_string());
    let ast = parser.parse_query(pattern)?;
    Ok(ast)
}

fn test_indexed_queries() {
    // Open the index
    let index_path = Path::new("test_index_output");
    let schema_path = Path::new("configs/schema.yaml");
    match ExtractorEngine::new(index_path, schema_path) {
        Ok(engine) => {
            println!("Successfully opened index with {} documents", engine.num_docs());
            
            // Test some queries against the indexed data
            let queries = vec![
                "word:John",
                "word:works", 
                "word:pizza",
                "pos:NNP",
                "pos:VBZ",
            ];

            for query in queries {
                println!("--- Query: '{}' ---", query);
                match engine.query(query) {
                    Ok(results) => {
                        println!("Found {} matches", results.total_hits);
                        if let Some(max_score) = results.max_score {
                            println!("Max score: {:.3}", max_score);
                        }
                        
                        for (i, score_doc) in results.score_docs.iter().take(3).enumerate() {
                            println!("  {}. Document {} (score: {:.3})", 
                                i + 1, score_doc.doc.segment_ord, score_doc.score);
                        }
                    }
                    Err(e) => {
                        println!("Error executing query: {}", e);
                    }
                }
                println!();
            }

            // Test a graph traversal query
            println!("--- Testing Graph Traversal Query ---");
            let graph_query = "[word=John] >nsubj [word=works]";
            println!("Query: {}", graph_query);
            
            match engine.query(graph_query) {
                Ok(results) => {
                    println!("Found {} matches", results.total_hits);
                    if let Some(max_score) = results.max_score {
                        println!("Max score: {:.3}", max_score);
                    }
                    
                    for (i, score_doc) in results.score_docs.iter().take(3).enumerate() {
                        println!("  {}. Document {} (score: {:.3})", 
                            i + 1, score_doc.doc.segment_ord, score_doc.score);
                    }
                }
                Err(e) => {
                    println!("Error executing graph traversal query: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Failed to open index: {}", e);
        }
    }
}

fn execute_custom_query(query: &str, index_path: &str) {
    let path = Path::new(index_path);
    let schema_path = Path::new("configs/schema.yaml");
    match ExtractorEngine::new(path, schema_path) {
        Ok(engine) => {
            println!("Successfully opened index with {} documents", engine.num_docs());
            
            // First try to parse the query
            let parser = QueryParser::new("word".to_string());
            match parser.parse_query(query) {
                Ok(ast) => {
                    println!("Parsed AST: {:?}", ast);
                    println!();
                }
                Err(e) => {
                    println!("Error parsing query: {}", e);
                    return;
                }
            }
            
            // Execute the query
            match engine.query(query) {
                Ok(results) => {
                    println!("Query Results:");
                    println!("  Total hits: {}", results.total_hits);
                    if let Some(max_score) = results.max_score {
                        println!("  Max score: {:.3}", max_score);
                    }
                    
                    if results.score_docs.is_empty() {
                        println!("  No matching documents found");
                    } else {
                        println!("  Top results:");
                        for (i, score_doc) in results.score_docs.iter().take(10).enumerate() {
                            println!("    {}. Document {} (score: {:.3})", 
                                i + 1, score_doc.doc.segment_ord, score_doc.score);
                        }
                    }
                }
                Err(e) => {
                    println!("Error executing query: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Failed to open index: {}", e);
        }
    }
}