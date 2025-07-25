use rustie::engine::ExtractorEngine;
use std::fs;
use std::path::Path;
use anyhow::Result;
use rustie::data::{self, DirectedGraph};
use rustie::compiler::parser::QueryParser;
use rustie::compiler::compiler::QueryCompiler;
use tantivy::directory::MmapDirectory;
use tantivy::{Index, DocId, Document as TantivyDoc};
use tantivy::collector::TopDocs;
use tantivy::query::Query;
use std::collections::HashSet;

fn main() -> Result<()> {
    println!("=== TRAVERSAL DEBUG TEST ===");
    
    // Create or open index
    let index_path = Path::new("traversal_debug_index");
    let schema_path = Path::new("configs/schema.yaml");
    
    let mut engine = if index_path.exists() {
        println!("Opening existing index...");
        ExtractorEngine::new(index_path, schema_path)?
    } else {
        println!("Creating new index...");
        fs::create_dir_all(index_path)?;
        ExtractorEngine::new(index_path, schema_path)?
    };

    println!("Index contains {} documents", engine.num_docs());

    let (schema, _output_fields) = ExtractorEngine::create_schema_from_yaml(schema_path)?;
    let dir = MmapDirectory::open(index_path)?;
    let index = Index::open_or_create(dir, schema.clone())?;

    let parser = QueryParser::new("word".to_string());
    let compiler = QueryCompiler::new(schema);

    // Test the specific traversal that's failing
    let traversal_query = "[word=TAZ] >nsubj [word=interacts]";
    println!("\n=== TESTING TRAVERSAL: {} ===", traversal_query);
    
    let query = parser.parse_query(traversal_query)?;
    println!("\nStep 1: Parsed query:\n{:#?}", query);
    
    let compiled_query = compiler.compile_pattern(&query)?;
    println!("\nStep 2: Compiled query structure:");
    debug_compiled_query(&compiled_query);

    // Now let's debug the execution step by step
    let reader = index.reader()?;
    let searcher = reader.searcher();

    println!("\n=== STEP-BY-STEP TRAVERSAL EXECUTION ===");
    
    // Step 3: Create Weight
    println!("\nStep 3: Creating Weight from Query...");
    let weight = compiled_query.weight(&searcher, false)?;
    
    // Step 4: Get all segments and debug each
    let segment_readers = searcher.segment_readers();
    println!("\nStep 4: Processing {} segments", segment_readers.len());
    
    for (seg_idx, segment_reader) in segment_readers.iter().enumerate() {
        println!("\n--- Segment {} ---", seg_idx);
        
        // Create scorer for this segment
        let scorer = weight.scorer(segment_reader, 1.0)?;
        
        // Debug the scoring process
        debug_traversal_scoring(segment_reader, seg_idx)?;
    }

    // Step 5: Execute the full search
    println!("\n\nStep 5: Executing full search...");
    let top_docs = searcher.search(&compiled_query, &TopDocs::with_limit(10))?;
    println!("Found {} matching documents", top_docs.len());

    // Step 6: Debug each matching document
    for (score, doc_address) in top_docs {
        println!("\n=== MATCHING DOCUMENT ===");
        println!("Score: {}, DocAddress: {:?}", score, doc_address);
        
        // Load and debug the document
        debug_matching_document(&searcher, doc_address)?;
    }

    Ok(())
}

/// Debug the compiled query structure
fn debug_compiled_query(query: &Box<dyn Query>) {
    // This will show the structure but we can't easily downcast
    println!("{:#?}", query);
}

/// Debug traversal scoring for a segment
fn debug_traversal_scoring(segment_reader: &tantivy::SegmentReader, seg_idx: usize) -> Result<()> {
    println!("\nDebug scoring for segment {}:", seg_idx);
    
    // Get the fields we're interested in
    let word_field = tantivy::schema::Field::from_field_id(0);
    let deps_field = tantivy::schema::Field::from_field_id(6);
    
    // Check how many documents in this segment
    let max_doc = segment_reader.max_doc();
    println!("  Max doc ID in segment: {}", max_doc);
    
    // Sample first few documents to see their content
    let doc_store = segment_reader.get_store_reader()?;
    
    for doc_id in 0..std::cmp::min(3, max_doc) {
        println!("\n  Document {}: ", doc_id);
        if let Ok(doc) = doc_store.get(doc_id) {
            // Get word tokens
            if let Some(word_value) = doc.get_first(word_field) {
                if let tantivy::schema::Value::Str(text) = word_value {
                    println!("    Words: {}", text);
                }
            }
            
            // Check if it has dependency data
            if let Some(deps_value) = doc.get_first(deps_field) {
                if let tantivy::schema::Value::Bytes(bytes) = deps_value {
                    println!("    Has dependency graph: {} bytes", bytes.len());
                    
                    // Try to deserialize and debug the graph
                    if let Ok(graph) = bincode::deserialize::<DirectedGraph>(bytes) {
                        debug_graph_structure(&graph)?;
                    }
                }
            }
        }
    }
    
    Ok(())
}

/// Debug a matching document
fn debug_matching_document(searcher: &tantivy::Searcher, doc_address: tantivy::DocAddress) -> Result<()> {
    let doc = searcher.doc(doc_address)?;
    
    let word_field = tantivy::schema::Field::from_field_id(0);
    let deps_field = tantivy::schema::Field::from_field_id(6);
    
    // Show document content
    if let Some(word_value) = doc.get_first(word_field) {
        if let tantivy::schema::Value::Str(text) = word_value {
            let tokens: Vec<&str> = text.split_whitespace().collect();
            println!("Tokens: {:?}", tokens);
            
            // Find positions of TAZ and interacts
            for (pos, token) in tokens.iter().enumerate() {
                if *token == "TAZ" || *token == "interacts" {
                    println!("  Found '{}' at position {}", token, pos);
                }
            }
        }
    }
    
    // Debug the dependency graph
    if let Some(deps_value) = doc.get_first(deps_field) {
        if let tantivy::schema::Value::Bytes(bytes) = deps_value {
            if let Ok(graph) = bincode::deserialize::<DirectedGraph>(bytes) {
                println!("\nDependency Graph:");
                debug_full_traversal(&graph, "TAZ", "interacts")?;
            }
        }
    }
    
    Ok(())
}

/// Debug the graph structure
fn debug_graph_structure(graph: &DirectedGraph) -> Result<()> {
    println!("    Graph stats:");
    println!("      Nodes: {}", graph.outgoing.len());
    println!("      Vocabulary size: {}", graph.vocabulary.len());
    
    // Show some edges
    for (node_id, edges) in graph.outgoing.iter().enumerate().take(3) {
        if !edges.is_empty() {
            println!("      Node {}: {} outgoing edges", node_id, edges.len() / 2);
        }
    }
    
    Ok(())
}

/// Debug full traversal from source to destination
fn debug_full_traversal(graph: &DirectedGraph, src_word: &str, dst_word: &str) -> Result<()> {
    println!("\nTraversal Debug: {} >nsubj {}", src_word, dst_word);
    
    // First, we need to find positions of these words
    // In real implementation, this would come from the document
    // For now, let's trace through the graph
    
    // Find nsubj label ID
    let nsubj_id = graph.vocabulary.get_id("nsubj");
    println!("  'nsubj' label ID: {:?}", nsubj_id);
    
    // Debug some edges to understand the structure
    println!("\n  Sample edges:");
    for (node_id, edges) in graph.outgoing.iter().enumerate() {
        if !edges.is_empty() {
            println!("    Node {}: ", node_id);
            for i in (0..edges.len()).step_by(2) {
                let target = edges[i];
                let label_id = edges[i + 1];
                if let Some(label) = graph.vocabulary.get_term(label_id) {
                    println!("      -> {} ({})", target, label);
                    
                    // If this is an nsubj edge, note it
                    if label == "nsubj" {
                        println!("        [Found nsubj edge!]");
                    }
                }
            }
        }
    }
    
    Ok(())
}

/// Alternative: Manual traversal simulation
fn simulate_traversal(index: &Index, query_str: &str) -> Result<()> {
    println!("\n=== MANUAL TRAVERSAL SIMULATION ===");
    
    let reader = index.reader()?;
    let searcher = reader.searcher();
    
    // Field definitions
    let word_field = tantivy::schema::Field::from_field_id(0);
    let deps_field = tantivy::schema::Field::from_field_id(6);
    
    // Step 1: Find documents containing "TAZ"
    let taz_query = tantivy::query::TermQuery::new(
        tantivy::schema::Term::from_field_text(word_field, "TAZ"),
        tantivy::schema::IndexRecordOption::Basic
    );
    
    let taz_docs = searcher.search(&taz_query, &TopDocs::with_limit(10))?;
    println!("Found {} documents containing 'TAZ'", taz_docs.len());
    
    // Step 2: For each document, load graph and traverse
    for (_, doc_addr) in taz_docs {
        let doc = searcher.doc(doc_addr)?;
        
        println!("\nDocument {:?}:", doc_addr);
        
        // Get tokens
        if let Some(tantivy::schema::Value::Str(text)) = doc.get_first(word_field) {
            let tokens: Vec<&str> = text.split_whitespace().collect();
            
            // Find TAZ position
            let taz_positions: Vec<usize> = tokens.iter()
                .enumerate()
                .filter(|(_, t)| **t == "TAZ")
                .map(|(i, _)| i)
                .collect();
                
            println!("  TAZ found at positions: {:?}", taz_positions);
            
            // Load graph
            if let Some(tantivy::schema::Value::Bytes(bytes)) = doc.get_first(deps_field) {
                if let Ok(graph) = bincode::deserialize::<DirectedGraph>(bytes) {
                    // For each TAZ position, check incoming edges
                    for &pos in &taz_positions {
                        println!("  Checking position {}:", pos);
                        
                        if let Some(incoming) = graph.incoming(pos) {
                            println!("    Incoming edges:");
                            for i in (0..incoming.len()).step_by(2) {
                                let source = incoming[i];
                                let label_id = incoming[i + 1];
                                
                                if let Some(label) = graph.vocabulary.get_term(label_id) {
                                    println!("      <- {} from position {}", label, source);
                                    
                                    // Check if this is nsubj and source is "interacts"
                                    if label == "nsubj" && source < tokens.len() && tokens[source] == "interacts" {
                                        println!("      *** MATCH FOUND! ***");
                                    }
                                }
                            }
                        }
                    }
                }
            }