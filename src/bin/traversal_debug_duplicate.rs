use rustie::engine::ExtractorEngine;
use std::fs;
use std::path::Path;
use anyhow::Result;
use rustie::data;
use rustie::compiler::parser::QueryParser;
use rustie::compiler::compiler::QueryCompiler;
use tantivy::directory::MmapDirectory;
use tantivy::Index;
use rustie::compiler::ast::FlatPatternStep;

fn main() -> Result<()> {
    println!("=== TRAVERSAL DEBUG TEST ===");
    
    // Create or open index
    let index_path = Path::new("manual_index");
    let schema_path = Path::new("configs/schema.yaml");
    
    let mut engine = if index_path.exists() {
        println!("Opening existing index...");
        ExtractorEngine::new(index_path, schema_path)?
    } else {
        println!("Creating new index...");

        // Ensure the index directory exists
        fs::create_dir_all(index_path)?;
        ExtractorEngine::new(index_path, schema_path)?
    };

    // Load and index the document
    let doc1 = fs::read_to_string("sample_documents/pattern_coverage.json")?;
    let json_parsed_doc = serde_json::from_str::<data::Document>(&doc1)?;

    // Actually index the document
    //engine.add_document(&json_parsed_doc)?;
    //engine.commit()?;

    println!("Index contains {} documents", engine.num_docs());

    // Get the schema from the engine instead of creating it separately
    let schema = engine.schema().clone();
    let dir = MmapDirectory::open(index_path)?;
    let index = Index::open_or_create(dir, schema.clone())?;

    // Test 2: Step-by-step traversal debug
    println!("\n=== TEST 2: Step-by-Step Traversal Debug ===");

    let parser = QueryParser::new("word".to_string());
    
    // Use the engine's QueryCompiler instead of creating a new one
    let compiler = engine.compiler();

    // Test the specific traversal that's failing
    let traversal_query = "[word=TAZ] >nsubj [tag=VBZ] >nmod [tag=NNS]";
    println!("Testing traversal: {}", traversal_query);


    //let result = engine.query(traversal_query)?;
    //let result_json = serde_json::to_string_pretty(&result)?;
    //println!("  Result: {}", result_json);
    
     let query = parser.parse_query(traversal_query)?;
    println!("  Parsed query: {:?}", query);
    
    println!("----------------------------\n");

    let compiled_query = compiler.compile_pattern(&query)?;
    println!("  Compiled query: {:?}", compiled_query);

    println!("----------------------------\n");

   
    //let reader = index.reader()?;
    //let searcher = reader.searcher();

    // Create field mappings
    let word_field = schema.get_field("word").expect("word field should exist");
    let dependencies_field = schema.get_field("dependencies_binary").expect("dependencies_binary field should exist");


     // Flattent the compiled query
     use rustie::tantivy_integration::graph_traversal::flatten_graph_traversal_pattern;
     let mut flattened_query = Vec::new();
     flatten_graph_traversal_pattern(&query, &mut flattened_query);
     println!("  Flattened query: {:?}", flattened_query);


    // Simulate traversal using the flattened query
    let sentence_idx = 0; // or whichever sentence you want to debug
    let tokens = json_parsed_doc.get_tokens(sentence_idx, "word").unwrap();
    let tags = json_parsed_doc.get_tokens(sentence_idx, "tag").unwrap();
    let deps = json_parsed_doc.get_dependencies(sentence_idx).unwrap();


    let mut current_nodes: Vec<usize> = (0..tokens.len()).collect();
    for step in &flattened_query {
        
        match step {


            // Constraint step
            FlatPatternStep::Constraint(pattern) => {
                
                current_nodes = current_nodes.into_iter().filter(|&i| {
                    // Filter nodes by constraint
                    match pattern {
                        rustie::compiler::ast::Pattern::Constraint(rustie::compiler::ast::Constraint::Field { name, matcher }) => {
                            let value = match name.as_str() {
                                "word" => &tokens[i],
                                "tag" => &tags[i],
                                _ => return false,
                            };
                            matcher.matches(value)
                        }
                        _ => false,
                    }
                }).collect();
                println!("After constraint {:?}: {:?}", pattern, current_nodes);

            }

            // Traversal step
            FlatPatternStep::Traversal(traversal) => {
                
                let mut next_nodes = Vec::new();
                for &i in &current_nodes {
                    match traversal {
                        
                        rustie::compiler::ast::Traversal::Outgoing(matcher) => {
                            for (to, rel) in deps.outgoing_edges(i as u32) {
                                if matcher.matches(&rel) {
                                    next_nodes.push(to as usize);
                                }
                            }
                        }
                        
                        rustie::compiler::ast::Traversal::Incoming(matcher) => {
                            for (from, rel) in deps.incoming_edges(i as u32) {
                                if matcher.matches(&rel) {
                                    next_nodes.push(from as usize);
                                }
                            }
                        }
                        // Handle other traversal types as no-ops or print a message
                        _ => {
                            println!("Traversal type {:?} not handled in simulation, skipping.", traversal);
                        }
                    }
                }
                current_nodes = next_nodes;
                println!("After traversal {:?}: {:?}", traversal, current_nodes);
            }


        }
    }
    println!("Final nodes: {:?}", current_nodes);

  



    // // Create weight from the graph traversal query
    //let weight = compiled_query.weight(tantivy::query::EnableScoring::Enabled { 
    //     searcher: &searcher, 
    //     statistics_provider: &searcher 
    // })?;
    // println!("  Created weight successfully");

    // // Create scorer from weight
    //let reader = searcher.segment_reader(0); // Get first segment reader
    //let mut scorer = weight.scorer(&reader, 1.0)?;

    //println!("  Created scorer successfully");

    // // Execute the search step by step
    // let mut matching_docs = Vec::new();

    // // First check if the scorer is already at a valid document
    // let mut doc_id = scorer.doc();
    // println!("  [DEBUG] Initial doc() returned: {}", doc_id);

    // // Process the current document if valid
    // if doc_id != tantivy::TERMINATED {
    //     println!("  Found matching document: {}", doc_id);
    //     matching_docs.push(doc_id);
    // }

    // // Then continue with advance() to find more documents
    // let mut doc_id = scorer.advance();
    // println!("  [DEBUG] First advance() returned: {}", doc_id);

    // while doc_id != tantivy::TERMINATED {
    //     println!("  Found matching document: {}", doc_id);
    //     matching_docs.push(doc_id);
    //     doc_id = scorer.advance();
    //     println!("  [DEBUG] Next advance() returned: {}", doc_id);
    // }

    // println!("Found {} matching documents", matching_docs.len());

    Ok(())
}



