use rustie::engine::ExtractorEngine;
use std::path::Path;
use std::fs;
use anyhow::Result;
use rustie::data;
use rustie::compiler::parser::QueryParser;
use rustie::compiler::compiler::QueryCompiler;

fn main() -> Result<()> {

    // Create or open index
    let index_path = Path::new("manual_index");
    let schema_path = Path::new("configs/schema.yaml");
    
    let engine = if index_path.exists() {
        println!("Opening existing index...");
        ExtractorEngine::new(index_path, schema_path)?
    } else {
        println!("Creating new index...");

        // Ensure the index directory exists
        fs::create_dir_all(index_path)?;
        ExtractorEngine::new(index_path,schema_path)?

    };

    let  doc1 = fs::read_to_string("sample_documents/biomed.json")?;
    let json_parsed_doc = serde_json::from_str::<data::Document>(&doc1)?;

    //engine.add_document(&json_parsed_doc)?;
    //engine.commit()?;

    println!("Index contains {} documents", engine.num_docs());

    let parser = QueryParser::new("word".to_string());
    let compiler = engine.compiler();

    // Parsing query

    

    let traversal_patterns = vec![
        // Simple word queries that should exist in the document
        
        /**
        "[word=TAZ]",
        "[word=interacts]",
        "[word=TEF-1]",
        "[word=proteins]",
        
        */

        
        
        // Single hop traversals using dependency relations from the document
        "[word=TAZ] [] [word=interacts]",
        //"[word=interacts] <nsubj [word=TAZ]",
        //"[word=proteins] <compound [word=TEF-1]",
        //"[word=TEF-1] <compound [word=proteins]",
        
        // Multi-hop traversals
        //"[word=TAZ] >nsubj [word=interacts] >nmod [word=members]",
       //[word=members] <nmod [word=interacts] <nsubj [word=TAZ]",
        
        // Using POS tags that exist in the document
        //pos=NN]",
        //pos=VBZ]",
        //pos=JJ]",
        
        // Using lemmas
        //lemma=/interact/]",
        //lemma=/protein/]",
        //lemma=/member/]",
        
        // Wildcard traversals
        //word=TAZ] >> [word=interacts]",
        //word=interacts] << [word=TAZ]",

        
    ];

   
    
    for (index, pattern) in traversal_patterns.iter().enumerate() {

        // Parsing query
        let query = parser.parse_query(pattern)?;
     
        // Compiling query
        let compiled_query = compiler.compile_pattern(&query)?;
    

        // Search query
        let search_results = engine.execute_query(compiled_query.as_ref(), 10)?;
        
        
        // Show JSON output with proper indentation
        println!("{}", search_results.to_json_standard());
        

        
        println!("-----------------------\n");

    }
    
    Ok(())
}