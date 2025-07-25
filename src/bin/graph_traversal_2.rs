use rustie::engine::ExtractorEngine;
use std::fs;
use std::path::Path;
use anyhow::Result;
use rustie::data;
use tantivy::directory::MmapDirectory;
use tantivy::Index;
use tantivy::Term;
use tantivy::schema::Field;
use tantivy::query::{Query, TermQuery};
use rustie::digraph::graph::DirectedGraph;

fn main() -> Result<()> {
    println!("=== GRAPH TRAVERSAL DEBUG ===");
    
    let index_path = Path::new("manual_index");
    let schema_path = Path::new("configs/schema.yaml");
    
    let engine = ExtractorEngine::new(index_path, schema_path)?;
    println!("Index contains {} documents", engine.num_docs());

    let (schema, _) = ExtractorEngine::create_schema_from_yaml(schema_path)?;
    let dir = MmapDirectory::open(index_path)?;
    let index = Index::open(dir)?;

    let reader = index.reader()?;
    let searcher = reader.searcher();

    // Get fields
    let word_field = schema.get_field("word").expect("word field should exist");
    let dependencies_field = schema.get_field("dependencies_binary").expect("dependencies_binary field should exist");

    // Get the first document and analyze its graph
    let segment_reader = searcher.segment_reader(0);
    let store_reader = segment_reader.get_store_reader(1)?;
    
    for doc_id in 0..segment_reader.max_doc() {
        if let Ok(doc) = store_reader.get(doc_id) {
            println!("\n=== Analyzing Document {} ===", doc_id);
            
            // Get all words
            let words: Vec<&str> = doc.get_all(word_field)
                .filter_map(|v| v.as_str())
                .collect();
            
            // Find positions of TAZ and interacts
            let taz_positions: Vec<usize> = words.iter()
                .enumerate()
                .filter(|(_, &w)| w == "TAZ")
                .map(|(i, _)| i)
                .collect();
            
            let interacts_positions: Vec<usize> = words.iter()
                .enumerate()
                .filter(|(_, &w)| w == "interacts")
                .map(|(i, _)| i)
                .collect();
            
            println!("TAZ positions: {:?}", taz_positions);
            println!("interacts positions: {:?}", interacts_positions);
            
            // Get the dependency graph
            if let Some(binary_data) = doc.get_first(dependencies_field).and_then(|v| v.as_bytes()) {
                let graph = DirectedGraph::from_bytes(binary_data)?;
                
                println!("\nGraph structure:");
                println!("  Number of nodes: {}", graph.nodes.len());
                println!("  Number of edges: {}", graph.adjacency_list.len());
                
                // For each TAZ position, check what edges it has
                for &taz_pos in &taz_positions {
                    println!("\n  Edges from TAZ at position {}:", taz_pos);
                    if let Some(edges) = graph.adjacency_list.get(&taz_pos) {
                        for edge in edges {
                            println!("    -> to position {} with label '{}'", edge.destination, edge.label);
                            if edge.destination < words.len() {
                                println!("       (word at {}: '{}')", edge.destination, words[edge.destination]);
                            }
                        }
                    } else {
                        println!("    No outgoing edges");
                    }
                    
                    // Check incoming edges
                    println!("  Edges to TAZ at position {}:", taz_pos);
                    for (src_pos, edges) in &graph.adjacency_list {
                        for edge in edges {
                            if edge.destination == taz_pos {
                                println!("    <- from position {} with label '{}'", src_pos, edge.label);
                                if *src_pos < words.len() {
                                    println!("       (word at {}: '{}')", src_pos, words[*src_pos]);
                                }
                            }
                        }
                    }
                }
                
                // For each interacts position, check what edges it has
                for &interacts_pos in &interacts_positions {
                    println!("\n  Edges from interacts at position {}:", interacts_pos);
                    if let Some(edges) = graph.adjacency_list.get(&interacts_pos) {
                        for edge in edges {
                            println!("    -> to position {} with label '{}'", edge.destination, edge.label);
                            if edge.destination < words.len() {
                                println!("       (word at {}: '{}')", edge.destination, words[edge.destination]);
                            }
                        }
                    } else {
                        println!("    No outgoing edges");
                    }
                    
                    // Check incoming edges
                    println!("  Edges to interacts at position {}:", interacts_pos);
                    for (src_pos, edges) in &graph.adjacency_list {
                        for edge in edges {
                            if edge.destination == interacts_pos {
                                println!("    <- from position {} with label '{}'", src_pos, edge.label);
                                if *src_pos < words.len() {
                                    println!("       (word at {}: '{}')", src_pos, words[*src_pos]);
                                }
                            }
                        }
                    }
                }
                
                // Now test the specific traversal
                println!("\n=== Testing Traversal Logic ===");
                let traversal_engine = rustie::digraph::traversal::GraphTraversal::new(graph.clone());
                
                // Test: TAZ -> interacts with any edge
                for &taz_pos in &taz_positions {
                    let result = traversal_engine.execute(
                        &rustie::compiler::ast::Traversal::Any,
                        &[taz_pos]
                    );
                    
                    match result {
                        rustie::digraph::traversal::TraversalResult::Success(positions) => {
                            println!("From TAZ at {}, can reach positions: {:?}", taz_pos, positions);
                            for pos in &positions {
                                if *pos < words.len() {
                                    println!("  Position {} = '{}'", pos, words[*pos]);
                                }
                            }
                        }
                        _ => println!("Traversal failed from TAZ at {}", taz_pos),
                    }
                }
                
                // Test specific nsubj traversal
                println!("\n=== Testing nsubj Traversal ===");
                
                // Test TAZ -nsubj-> ?
                for &taz_pos in &taz_positions {
                    let result = traversal_engine.execute(
                        &rustie::compiler::ast::Traversal::Outgoing(
                            rustie::compiler::ast::Matcher::String("nsubj".to_string())
                        ),
                        &[taz_pos]
                    );
                    
                    match result {
                        rustie::digraph::traversal::TraversalResult::Success(positions) => {
                            println!("TAZ at {} -nsubj-> positions: {:?}", taz_pos, positions);
                        }
                        _ => println!("No nsubj edges from TAZ at {}", taz_pos),
                    }
                }
                
                // Test interacts -nsubj-> ?
                for &interacts_pos in &interacts_positions {
                    let result = traversal_engine.execute(
                        &rustie::compiler::ast::Traversal::Outgoing(
                            rustie::compiler::ast::Matcher::String("nsubj".to_string())
                        ),
                        &[interacts_pos]
                    );
                    
                    match result {
                        rustie::digraph::traversal::TraversalResult::Success(positions) => {
                            println!("interacts at {} -nsubj-> positions: {:?}", interacts_pos, positions);
                        }
                        _ => println!("No nsubj edges from interacts at {}", interacts_pos),
                    }
                }
                
                // Test ? -nsubj-> TAZ
                println!("\nTesting incoming nsubj to TAZ:");
                for &taz_pos in &taz_positions {
                    let result = traversal_engine.execute(
                        &rustie::compiler::ast::Traversal::Incoming(
                            rustie::compiler::ast::Matcher::String("nsubj".to_string())
                        ),
                        &[taz_pos]
                    );
                    
                    match result {
                        rustie::digraph::traversal::TraversalResult::Success(positions) => {
                            println!("Positions -nsubj-> TAZ at {}: {:?}", taz_pos, positions);
                            for pos in &positions {
                                if *pos < words.len() {
                                    println!("  Position {} = '{}'", pos, words[*pos]);
                                }
                            }
                        }
                        _ => println!("No incoming nsubj edges to TAZ at {}", taz_pos),
                    }
                }
            }
        }
    }
    
    Ok(())
}