use rustie::compiler::QueryCompiler;
use rustie::compiler::parser::QueryParser;
use tantivy::schema::{Schema, TEXT, STORED};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== Testing Compiler Features ===\n");

    // Create a simple schema
    let mut schema_builder = Schema::builder();
    schema_builder.add_text_field("word", TEXT | STORED);
    schema_builder.add_text_field("pos", TEXT | STORED);
    schema_builder.add_text_field("lemma", TEXT | STORED);
    schema_builder.add_text_field("tag", TEXT | STORED);
    let schema = schema_builder.build();

    let compiler = QueryCompiler::new(schema);
    let parser = QueryParser::new("word".to_string());

    // Test queries
    let test_cases = vec![
        // Basic constraints (should work)
        ("[word=John]", true),
        ("[word=/john|jane/]", true),
        ("[*]", true),
        
        // Negated constraints (implemented)
        ("[!word=John]", true),
        ("[word!=John]", true),
        ("[word=dog & !tag=VB]", true),
        
        // Fuzzy matching (implemented)
        ("[word=hello~]", true),
        
        // Assertions (NOW IMPLEMENTED!)
        ("<s> [word=The]", true),
        ("[word=end] </s>", true),
        ("(?=[word=works])", true),
        ("(?<=[word=The])", true),
        
        // Repetition (NOW IMPLEMENTED!)
        ("[word=the]*", true),
        ("[word=the]+", true),
        ("[word=the]?", true),
        ("[word=the]{2,5}", true),
        
        // Named Captures (NOW IMPLEMENTED!)
        ("(?<subject>[word=John])", true),
        ("(?<verb>[tag=VB])", true),
        
        // Graph traversals (should work if schema has dependencies_binary)
        ("[word=dog] >nsubj [word=barks]", false), // Will fail without dependencies field
    ];

    for (query, should_succeed) in test_cases {
        print!("Testing: '{}' ... ", query);
        
        match parser.parse_query(query) {
            Ok(pattern) => {
                match compiler.compile_pattern(&pattern) {
                    Ok(_) => {
                        if should_succeed {
                            println!("✓ Compiled successfully");
                        } else {
                            println!("✗ Unexpected success (expected failure)");
                        }
                    }
                    Err(e) => {
                        if !should_succeed {
                            println!("✓ Failed as expected: {}", e);
                        } else {
                            println!("✗ Compilation failed: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                println!("✗ Parse error: {}", e);
            }
        }
    }

    println!("\n=== Test Complete ===");
    Ok(())
}

