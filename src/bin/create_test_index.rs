//! Create a test index for API testing
//!
//! Usage:
//!   cargo run --bin create_test_index
//!
//! This creates an index at ./test_api_index with sample documents.

use rustie::{Document, ExtractorEngine};
use rustie::data::document::{Sentence, Field};
use std::path::Path;
use std::fs;
use anyhow::Result;

fn main() -> Result<()> {
    let index_path = Path::new("./test_api_index");
    let schema_path = Path::new("configs/schema.yaml");

    println!("Creating test index for API...\n");

    // Clean up existing index
    if index_path.exists() {
        println!("Removing existing index...");
        fs::remove_dir_all(index_path)?;
    }
    fs::create_dir_all(index_path)?;

    // Create engine
    let mut engine = ExtractorEngine::new(index_path, schema_path)?;

    // Create sample documents
    let documents = create_sample_documents();

    println!("Indexing {} documents...\n", documents.len());

    for doc in &documents {
        println!("  - {} ({} sentences)", doc.id, doc.sentences.len());
        engine.add_document(doc)?;
    }

    engine.commit()?;

    println!("\nIndex created successfully!");
    println!("  Path: {}", index_path.display());
    println!("  Documents: {}", engine.num_docs());
    println!("\nTo start the API server:");
    println!("  cargo run --bin api_server -- --index ./test_api_index");
    println!("\nExample queries:");
    println!("  curl http://localhost:8080/api/v1/health");
    println!("  curl http://localhost:8080/api/v1/stats");
    println!("  curl \"http://localhost:8080/api/v1/query/%5Bword=John%5D\"");

    Ok(())
}

fn create_sample_documents() -> Vec<Document> {
    vec![
        // Document 1: Simple sentence with subject-verb-object
        Document {
            id: "doc_simple".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 4,
                    fields: vec![
                        Field::TokensField { name: "word".to_string(), tokens: vec!["John".to_string(), "eats".to_string(), "pizza".to_string(), ".".to_string()] },
                        Field::TokensField { name: "pos".to_string(), tokens: vec!["NNP".to_string(), "VBZ".to_string(), "NN".to_string(), ".".to_string()] },
                        Field::TokensField { name: "lemma".to_string(), tokens: vec!["john".to_string(), "eat".to_string(), "pizza".to_string(), ".".to_string()] },
                        Field::TokensField { name: "entity".to_string(), tokens: vec!["B-PERSON".to_string(), "O".to_string(), "B-FOOD".to_string(), "O".to_string()] },
                        Field::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![1],
                            edges: vec![
                                (1, 0, "nsubj".to_string()),
                                (1, 2, "dobj".to_string()),
                                (1, 3, "punct".to_string()),
                            ],
                        },
                    ],
                },
            ],
        },

        // Document 2: Cat sleeps
        Document {
            id: "doc_cat".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 6,
                    fields: vec![
                        Field::TokensField { name: "word".to_string(), tokens: vec!["The".to_string(), "cat".to_string(), "sleeps".to_string(), "on".to_string(), "the".to_string(), "mat".to_string()] },
                        Field::TokensField { name: "pos".to_string(), tokens: vec!["DT".to_string(), "NN".to_string(), "VBZ".to_string(), "IN".to_string(), "DT".to_string(), "NN".to_string()] },
                        Field::TokensField { name: "lemma".to_string(), tokens: vec!["the".to_string(), "cat".to_string(), "sleep".to_string(), "on".to_string(), "the".to_string(), "mat".to_string()] },
                        Field::TokensField { name: "entity".to_string(), tokens: vec!["O".to_string(), "B-ANIMAL".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "O".to_string()] },
                        Field::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![2],
                            edges: vec![
                                (1, 0, "det".to_string()),
                                (2, 1, "nsubj".to_string()),
                                (2, 3, "prep".to_string()),
                                (3, 5, "pobj".to_string()),
                                (5, 4, "det".to_string()),
                            ],
                        },
                    ],
                },
            ],
        },

        // Document 3: Scientists discover
        Document {
            id: "doc_science".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 7,
                    fields: vec![
                        Field::TokensField { name: "word".to_string(), tokens: vec!["Scientists".to_string(), "discovered".to_string(), "a".to_string(), "new".to_string(), "species".to_string(), "in".to_string(), "Amazon".to_string()] },
                        Field::TokensField { name: "pos".to_string(), tokens: vec!["NNS".to_string(), "VBD".to_string(), "DT".to_string(), "JJ".to_string(), "NN".to_string(), "IN".to_string(), "NNP".to_string()] },
                        Field::TokensField { name: "lemma".to_string(), tokens: vec!["scientist".to_string(), "discover".to_string(), "a".to_string(), "new".to_string(), "species".to_string(), "in".to_string(), "amazon".to_string()] },
                        Field::TokensField { name: "entity".to_string(), tokens: vec!["O".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "B-LOC".to_string()] },
                        Field::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![1],
                            edges: vec![
                                (1, 0, "nsubj".to_string()),
                                (1, 4, "dobj".to_string()),
                                (4, 2, "det".to_string()),
                                (4, 3, "amod".to_string()),
                                (1, 5, "prep".to_string()),
                                (5, 6, "pobj".to_string()),
                            ],
                        },
                    ],
                },
            ],
        },

        // Document 4: Mary loves coffee
        Document {
            id: "doc_mary".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 5,
                    fields: vec![
                        Field::TokensField { name: "word".to_string(), tokens: vec!["Mary".to_string(), "loves".to_string(), "hot".to_string(), "coffee".to_string(), ".".to_string()] },
                        Field::TokensField { name: "pos".to_string(), tokens: vec!["NNP".to_string(), "VBZ".to_string(), "JJ".to_string(), "NN".to_string(), ".".to_string()] },
                        Field::TokensField { name: "lemma".to_string(), tokens: vec!["mary".to_string(), "love".to_string(), "hot".to_string(), "coffee".to_string(), ".".to_string()] },
                        Field::TokensField { name: "entity".to_string(), tokens: vec!["B-PERSON".to_string(), "O".to_string(), "O".to_string(), "B-FOOD".to_string(), "O".to_string()] },
                        Field::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![1],
                            edges: vec![
                                (1, 0, "nsubj".to_string()),
                                (1, 3, "dobj".to_string()),
                                (3, 2, "amod".to_string()),
                                (1, 4, "punct".to_string()),
                            ],
                        },
                    ],
                },
            ],
        },

        // Document 5: Company announcement
        Document {
            id: "doc_company".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 8,
                    fields: vec![
                        Field::TokensField { name: "word".to_string(), tokens: vec!["Apple".to_string(), "announced".to_string(), "a".to_string(), "revolutionary".to_string(), "new".to_string(), "product".to_string(), "yesterday".to_string(), ".".to_string()] },
                        Field::TokensField { name: "pos".to_string(), tokens: vec!["NNP".to_string(), "VBD".to_string(), "DT".to_string(), "JJ".to_string(), "JJ".to_string(), "NN".to_string(), "NN".to_string(), ".".to_string()] },
                        Field::TokensField { name: "lemma".to_string(), tokens: vec!["apple".to_string(), "announce".to_string(), "a".to_string(), "revolutionary".to_string(), "new".to_string(), "product".to_string(), "yesterday".to_string(), ".".to_string()] },
                        Field::TokensField { name: "entity".to_string(), tokens: vec!["B-ORG".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "B-DATE".to_string(), "O".to_string()] },
                        Field::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![1],
                            edges: vec![
                                (1, 0, "nsubj".to_string()),
                                (1, 5, "dobj".to_string()),
                                (5, 2, "det".to_string()),
                                (5, 3, "amod".to_string()),
                                (5, 4, "amod".to_string()),
                                (1, 6, "tmod".to_string()),
                                (1, 7, "punct".to_string()),
                            ],
                        },
                    ],
                },
            ],
        },
    ]
}
