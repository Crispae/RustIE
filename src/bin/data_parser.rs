use rustie::data::{Document, DocumentParser};
use rustie::engine::ExtractorEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example JSON data matching the structure you provided
    let example_json = r#"{
  "id": "15634067",
  "metadata": [],
  "sentences": [
    {
      "numTokens": 13,
      "fields": [
        {
          "name": "raw",
          "$type": "ai.lum.odinson.TokensField",
          "tokens": [
            "The",
            "diffusivity",
            "is",
            "observed",
            "to",
            "be",
            "very",
            "directly",
            "linked",
            "to",
            "network",
            "imperfection",
            "."
          ]
        },
        {
          "name": "dependencies",
          "$type": "ai.lum.odinson.GraphField",
          "edges": [
            [8, 11, "nmod"],
            [11, 9, "case"],
            [3, 12, "punct"],
            [1, 0, "det"],
            [7, 6, "advmod"],
            [11, 10, "compound"],
            [3, 1, "nsubjpass"],
            [8, 7, "advmod"],
            [3, 8, "xcomp"],
            [8, 5, "auxpass"],
            [3, 2, "auxpass"],
            [8, 4, "mark"]
          ],
          "roots": [3]
        },
        {
          "name": "word",
          "$type": "ai.lum.odinson.TokensField",
          "tokens": [
            "The",
            "diffusivity",
            "is",
            "observed",
            "to",
            "be",
            "very",
            "directly",
            "linked",
            "to",
            "network",
            "imperfection",
            "."
          ]
        },
        {
          "name": "lemma",
          "$type": "ai.lum.odinson.TokensField",
          "tokens": [
            "the",
            "diffusivity",
            "be",
            "observe",
            "to",
            "be",
            "very",
            "directly",
            "link",
            "to",
            "network",
            "imperfection",
            "."
          ]
        },
        {
          "name": "tag",
          "$type": "ai.lum.odinson.TokensField",
          "tokens": [
            "DT",
            "NN",
            "VBZ",
            "VBN",
            "TO",
            "VB",
            "RB",
            "RB",
            "VBN",
            "IN",
            "NN",
            "NN",
            "."
          ]
        },
        {
          "name": "entity",
          "$type": "ai.lum.odinson.TokensField",
          "tokens": [
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O"
          ]
        },
        {
          "name": "chunk",
          "$type": "ai.lum.odinson.TokensField",
          "tokens": [
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O"
          ]
        },
        {
          "name": "norm",
          "$type": "ai.lum.odinson.TokensField",
          "tokens": [
            "The",
            "diffusivity",
            "is",
            "observed",
            "to",
            "be",
            "very",
            "directly",
            "linked",
            "to",
            "network",
            "imperfection",
            "."
          ]
        }
      ]
    }
  ]
}"#;

    // Create schema (using the same schema as ExtractorEngine)
    let (schema, _) = ExtractorEngine::create_schema_from_yaml("configs/schema.yaml")?;
    let mut parser = DocumentParser::new(schema);

    // Parse the JSON string
    println!("Attempting to parse JSON...");
    
    // First, try to parse directly with serde_json to get better error messages
    println!("Testing direct serde_json parsing...");
    match serde_json::from_str::<Document>(example_json) {
        Ok(doc) => {
            println!("✅ Direct parsing successful!");
            println!("Document ID: {}", doc.id);
            println!("Number of sentences: {}", doc.sentences.len());
        }
        Err(e) => {
            println!("❌ Direct parsing failed: {}", e);
            println!("Error details: {:?}", e);
        }
    }
    
    // Now try with the parser
    match parser.parse_json(example_json) {
        Ok(documents) => {
            println!("Successfully parsed {} document(s)", documents.len());
            
            for (doc_idx, doc) in documents.iter().enumerate() {
                println!("\nDocument {}: {}", doc_idx, doc.id);
                println!("Number of sentences: {}", doc.sentences.len());
                
                for (sent_idx, sentence) in doc.sentences.iter().enumerate() {
                    println!("  Sentence {}: {} tokens", sent_idx, sentence.numTokens);
                    
                    // Show some field information
                    if let Some(words) = doc.get_tokens(sent_idx, "word") {
                        println!("    Words: {}", words.join(" "));
                    }
                    
                    if let Some(lemmas) = doc.get_tokens(sent_idx, "lemma") {
                        println!("    Lemmas: {}", lemmas.join(" "));
                    }
                    
                    if let Some(deps) = doc.dependencies_to_string(sent_idx) {
                        println!("    Dependencies: {}", deps);
                    }
                }
                
                // Convert to Tantivy documents
                match parser.to_tantivy_document(doc) {
                    Ok(tantivy_docs) => {
                        println!("  Converted to {} Tantivy document(s)", tantivy_docs.len());
                        
                        // Show the first Tantivy document structure
                        if let Some(first_doc) = tantivy_docs.first() {
                    
                            println!("{:#?}", first_doc);
                        }
                    }
                    Err(e) => {
                        println!("  Error converting to Tantivy documents: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            println!("Error parsing JSON: {}", e);
            println!("JSON content preview: {}", &example_json[..example_json.len().min(200)]);
        }
    }

    // Example of how to parse from a file (if it exists)
    let test_file_path = "test_document.json";
    if std::path::Path::new(test_file_path).exists() {
        println!("\nParsing from file: {}", test_file_path);
        match parser.parse_file(test_file_path) {
            Ok(docs) => {
                println!("Successfully parsed {} document(s) from file", docs.len());
            }
            Err(e) => {
                println!("Error parsing file: {}", e);
            }
        }
    } else {
        println!("\nNo test file found at: {}", test_file_path);
        println!("You can create a test file with your JSON data to test file parsing.");
    }
    Ok(())
} 