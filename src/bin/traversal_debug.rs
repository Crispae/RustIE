use rustie::engine::ExtractorEngine;
use std::fs;
use std::path::Path;
use anyhow::Result;
use rustie::data;
use rustie::compiler::parser::QueryParser;
use rustie::compiler::compiler::QueryCompiler;
use tantivy::directory::MmapDirectory;
use tantivy::Index;

fn main() -> Result<()> {
    println!("=== TRAVERSAL DEBUG TEST ===");
    
    // Create or open index
    let index_path = Path::new("complex_logic_index");
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

    // Patterns to test
    let patterns = vec![
        //"[word=dog] | [word=cat]",
        //"[word=ROOT] >> [pos=/JJ.*/] >> [word=eta]",
        //"[word=TAZ] >nmod [word=/mem.*/]",
        "[word=TAZ] >nsubj [tag=VBZ] >nmod [tag=NNS]",
        //"([word=the] [word=dog]) | ([word=a] [word=cat])"
        //"[word=TAZ] [] [word=transcriptional]",
        //"[pos=/nn.*/] >amod [word=/transcri.*/] >> [pos=/nn.*/]",
        //"[word=TAZ] [*] [word=transcriptional]",
        //"[word=dog]",
        //"[word=TAZ] >nsubj [word=interacts]",
        //"[word=TAZ] >compound [word=family]",
        //"[word=TAZ] >nmod [word=members]",
        //"[word=TAZ] >amod? [word=transcriptional]",
        //"[word=TAZ] >amod* [word=transcriptional]",
        //"[word=TAZ] >/amod|nsubj/ [word=transcriptional]",
        //"[word=TAZ] >> [word=transcriptional]",
        //"[word=TAZ] << [word=transcriptional]",
        //"[word=ROOT] >rel1 [word=MID] >rel2 [word=END]"
    ];

    let mut results = Vec::new();
    let schema = engine.schema();
    let parser = QueryParser::new("word".to_string());
    let compiler = QueryCompiler::new(schema.clone());
    for pattern in &patterns {
        
        println!("Testing pattern: {}", pattern);
        // Print parsed AST
        match parser.parse_query(pattern) {
            Ok(ast) => {
                //println!("  Parsed AST: {:?}", ast);
                // Print compiled AST
                //match compiler.compile_pattern(&ast) {
                //    Ok(compiled) => {
                //        println!("  Compiled AST: {:?}", compiled);
                //    }
                //    Err(e) => {
                //        println!("  Compile error: {}", e);
                //    }
                //}
            }
            Err(e) => {
                println!("  Parse error: {}", e);
            }
        }
        
        // Run the query
        let result = engine.query(pattern);
        
        match result {
            Ok(res) => {
                let matched = res.total_hits > 0;
                let result_json = serde_json::to_string_pretty(&res)?;
                println!("  Result: {}", result_json);
                println!("  Matched? {} (total_hits: {})", matched, res.total_hits);
                results.push((*pattern, matched, res.total_hits));
            }
            Err(e) => {
                println!("  Error: {}", e);
                results.push((*pattern, false, 0));
            }
        }
    }
    
    println!("\n=== Pattern Match Summary ===");
    for (pattern, matched, hits) in results {
        println!("{:50} | Matched: {:5} | Hits: {}", pattern, matched, hits);
    }

    Ok(())
}



