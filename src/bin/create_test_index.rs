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
    println!("\n=== EXAMPLE QUERIES ===\n");
    println!("Basic queries:");
    println!("  curl http://localhost:8080/api/v1/health");
    println!("  curl http://localhost:8080/api/v1/stats");
    println!();
    println!("Single-hop graph traversals:");
    println!("  [word=eats] >nsubj [word=John]       - verb with subject");
    println!("  [word=eats] >dobj [word=pizza]       - verb with object");
    println!("  [pos=VBZ] >nsubj [pos=NNP]           - any verb with proper noun subject");
    println!("  [lemma=give] >iobj [entity=B-PERSON] - give with person as indirect object");
    println!();
    println!("Multi-hop graph traversals:");
    println!("  [lemma=chase] >nsubj [pos=NN] <amod [pos=JJ]");
    println!("      -> chased -> dog -> big (verb to subject to modifier)");
    println!();
    println!("  [lemma=sleep] >prep [word=on] >pobj [word=mat]");
    println!("      -> sleeps -> on -> mat (verb to prep to object)");
    println!();
    println!("  [word=believes] >ccomp [word=knows] >dobj [word=truth]");
    println!("      -> believes -> knows -> truth (verb to clause to object)");
    println!();
    println!("  [word=left] >advcl [word=arrived] >nsubj [word=Mary]");
    println!("      -> left -> arrived -> Mary (verb to adverbial clause to subject)");
    println!();
    println!("Field-based queries:");
    println!("  [entity=B-PERSON]                    - find all person entities");
    println!("  [entity=B-FOOD]                      - find all food entities");
    println!("  [pos=JJ] [pos=NN]                    - adjective followed by noun");
    println!();

    Ok(())
}

fn create_sample_documents() -> Vec<Document> {
    vec![
        // ============================================================
        // BASIC SUBJECT-VERB-OBJECT PATTERNS
        // ============================================================

        // Document 1: Simple SVO - "John eats pizza"
        // Test: [word=eats] >nsubj [word=John]
        // Test: [word=eats] >dobj [word=pizza]
        // Test: [pos=VBZ] >nsubj [pos=NNP]
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
                                (1, 0, "nsubj".to_string()),   // eats -> John
                                (1, 2, "dobj".to_string()),    // eats -> pizza
                                (1, 3, "punct".to_string()),   // eats -> .
                            ],
                        },
                    ],
                },
            ],
        },

        // ============================================================
        // MULTI-HOP: VERB -> SUBJECT -> MODIFIER CHAINS
        // ============================================================

        // Document 2: "The big dog chased the small cat"
        // Multi-hop test: [lemma=chase] >nsubj [pos=NN] <amod [pos=JJ]
        // Test: verb -> subject -> adjective modifier
        //       chased(2) -> dog(1) -> big(0)
        // Also: chased(2) -> cat(5) -> small(4)
        Document {
            id: "doc_chase".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 7,
                    fields: vec![
                        Field::TokensField { name: "word".to_string(), tokens: vec!["The".to_string(), "big".to_string(), "dog".to_string(), "chased".to_string(), "the".to_string(), "small".to_string(), "cat".to_string()] },
                        Field::TokensField { name: "pos".to_string(), tokens: vec!["DT".to_string(), "JJ".to_string(), "NN".to_string(), "VBD".to_string(), "DT".to_string(), "JJ".to_string(), "NN".to_string()] },
                        Field::TokensField { name: "lemma".to_string(), tokens: vec!["the".to_string(), "big".to_string(), "dog".to_string(), "chase".to_string(), "the".to_string(), "small".to_string(), "cat".to_string()] },
                        Field::TokensField { name: "entity".to_string(), tokens: vec!["O".to_string(), "O".to_string(), "B-ANIMAL".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "B-ANIMAL".to_string()] },
                        Field::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![3],
                            edges: vec![
                                (2, 0, "det".to_string()),     // dog -> The
                                (2, 1, "amod".to_string()),    // dog -> big
                                (3, 2, "nsubj".to_string()),   // chased -> dog
                                (3, 6, "dobj".to_string()),    // chased -> cat
                                (6, 4, "det".to_string()),     // cat -> the
                                (6, 5, "amod".to_string()),    // cat -> small
                            ],
                        },
                    ],
                },
            ],
        },

        // ============================================================
        // MULTI-HOP: DITRANSITIVE VERB (3 arguments)
        // ============================================================

        // Document 3: "John gave Mary a beautiful gift"
        // Test: [word=gave] >nsubj [entity=B-PERSON] - subject
        // Test: [word=gave] >iobj [word=Mary] - indirect object
        // Test: [word=gave] >dobj [word=gift] - direct object
        // Multi-hop: [lemma=give] >dobj [pos=NN] <amod [pos=JJ]
        //            gave(1) -> gift(5) -> beautiful(4)
        Document {
            id: "doc_ditransitive".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 6,
                    fields: vec![
                        Field::TokensField { name: "word".to_string(), tokens: vec!["John".to_string(), "gave".to_string(), "Mary".to_string(), "a".to_string(), "beautiful".to_string(), "gift".to_string()] },
                        Field::TokensField { name: "pos".to_string(), tokens: vec!["NNP".to_string(), "VBD".to_string(), "NNP".to_string(), "DT".to_string(), "JJ".to_string(), "NN".to_string()] },
                        Field::TokensField { name: "lemma".to_string(), tokens: vec!["john".to_string(), "give".to_string(), "mary".to_string(), "a".to_string(), "beautiful".to_string(), "gift".to_string()] },
                        Field::TokensField { name: "entity".to_string(), tokens: vec!["B-PERSON".to_string(), "O".to_string(), "B-PERSON".to_string(), "O".to_string(), "O".to_string(), "O".to_string()] },
                        Field::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![1],
                            edges: vec![
                                (1, 0, "nsubj".to_string()),   // gave -> John
                                (1, 2, "iobj".to_string()),    // gave -> Mary (indirect object)
                                (1, 5, "dobj".to_string()),    // gave -> gift (direct object)
                                (5, 3, "det".to_string()),     // gift -> a
                                (5, 4, "amod".to_string()),    // gift -> beautiful
                            ],
                        },
                    ],
                },
            ],
        },

        // ============================================================
        // MULTI-HOP: PREPOSITIONAL CHAINS
        // ============================================================

        // Document 4: "The cat sleeps on the soft mat in the corner"
        // Multi-hop: [lemma=sleep] >prep [word=on] >pobj [word=mat]
        //            sleeps(2) -> on(3) -> mat(6)
        // Triple-hop: [lemma=sleep] >prep [word=on] >pobj [pos=NN] <amod [word=soft]
        //             sleeps(2) -> on(3) -> mat(6) -> soft(5)
        Document {
            id: "doc_prep_chain".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 10,
                    fields: vec![
                        Field::TokensField { name: "word".to_string(), tokens: vec!["The".to_string(), "cat".to_string(), "sleeps".to_string(), "on".to_string(), "the".to_string(), "soft".to_string(), "mat".to_string(), "in".to_string(), "the".to_string(), "corner".to_string()] },
                        Field::TokensField { name: "pos".to_string(), tokens: vec!["DT".to_string(), "NN".to_string(), "VBZ".to_string(), "IN".to_string(), "DT".to_string(), "JJ".to_string(), "NN".to_string(), "IN".to_string(), "DT".to_string(), "NN".to_string()] },
                        Field::TokensField { name: "lemma".to_string(), tokens: vec!["the".to_string(), "cat".to_string(), "sleep".to_string(), "on".to_string(), "the".to_string(), "soft".to_string(), "mat".to_string(), "in".to_string(), "the".to_string(), "corner".to_string()] },
                        Field::TokensField { name: "entity".to_string(), tokens: vec!["O".to_string(), "B-ANIMAL".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "B-LOC".to_string()] },
                        Field::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![2],
                            edges: vec![
                                (1, 0, "det".to_string()),     // cat -> The
                                (2, 1, "nsubj".to_string()),   // sleeps -> cat
                                (2, 3, "prep".to_string()),    // sleeps -> on
                                (3, 6, "pobj".to_string()),    // on -> mat
                                (6, 4, "det".to_string()),     // mat -> the
                                (6, 5, "amod".to_string()),    // mat -> soft
                                (6, 7, "prep".to_string()),    // mat -> in (nested prep)
                                (7, 9, "pobj".to_string()),    // in -> corner
                                (9, 8, "det".to_string()),     // corner -> the
                            ],
                        },
                    ],
                },
            ],
        },

        // ============================================================
        // MULTI-HOP: RELATIVE CLAUSE (complex structure)
        // ============================================================

        // Document 5: "Scientists who work at MIT discovered particles"
        // Multi-hop: [word=Scientists] <nsubj [word=discovered]
        //            Plus relative clause: Scientists -> who -> work -> MIT
        // Test: [pos=NNS] >relcl [pos=VBP] >prep [word=at] >pobj [entity=B-ORG]
        Document {
            id: "doc_relative".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 7,
                    fields: vec![
                        Field::TokensField { name: "word".to_string(), tokens: vec!["Scientists".to_string(), "who".to_string(), "work".to_string(), "at".to_string(), "MIT".to_string(), "discovered".to_string(), "particles".to_string()] },
                        Field::TokensField { name: "pos".to_string(), tokens: vec!["NNS".to_string(), "WP".to_string(), "VBP".to_string(), "IN".to_string(), "NNP".to_string(), "VBD".to_string(), "NNS".to_string()] },
                        Field::TokensField { name: "lemma".to_string(), tokens: vec!["scientist".to_string(), "who".to_string(), "work".to_string(), "at".to_string(), "mit".to_string(), "discover".to_string(), "particle".to_string()] },
                        Field::TokensField { name: "entity".to_string(), tokens: vec!["O".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "B-ORG".to_string(), "O".to_string(), "O".to_string()] },
                        Field::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![5],
                            edges: vec![
                                (5, 0, "nsubj".to_string()),   // discovered -> Scientists
                                (0, 2, "relcl".to_string()),   // Scientists -> work (relative clause)
                                (2, 1, "nsubj".to_string()),   // work -> who
                                (2, 3, "prep".to_string()),    // work -> at
                                (3, 4, "pobj".to_string()),    // at -> MIT
                                (5, 6, "dobj".to_string()),    // discovered -> particles
                            ],
                        },
                    ],
                },
            ],
        },

        // ============================================================
        // MULTI-HOP: COORDINATION (AND/OR)
        // ============================================================

        // Document 6: "John and Mary love pizza and pasta"
        // Test: compound subject and compound object
        // Multi-hop: [word=love] >nsubj [word=John] <conj [word=Mary]
        Document {
            id: "doc_coordination".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 7,
                    fields: vec![
                        Field::TokensField { name: "word".to_string(), tokens: vec!["John".to_string(), "and".to_string(), "Mary".to_string(), "love".to_string(), "pizza".to_string(), "and".to_string(), "pasta".to_string()] },
                        Field::TokensField { name: "pos".to_string(), tokens: vec!["NNP".to_string(), "CC".to_string(), "NNP".to_string(), "VBP".to_string(), "NN".to_string(), "CC".to_string(), "NN".to_string()] },
                        Field::TokensField { name: "lemma".to_string(), tokens: vec!["john".to_string(), "and".to_string(), "mary".to_string(), "love".to_string(), "pizza".to_string(), "and".to_string(), "pasta".to_string()] },
                        Field::TokensField { name: "entity".to_string(), tokens: vec!["B-PERSON".to_string(), "O".to_string(), "B-PERSON".to_string(), "O".to_string(), "B-FOOD".to_string(), "O".to_string(), "B-FOOD".to_string()] },
                        Field::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![3],
                            edges: vec![
                                (3, 0, "nsubj".to_string()),   // love -> John
                                (0, 1, "cc".to_string()),      // John -> and
                                (0, 2, "conj".to_string()),    // John -> Mary (conjoined)
                                (3, 4, "dobj".to_string()),    // love -> pizza
                                (4, 5, "cc".to_string()),      // pizza -> and
                                (4, 6, "conj".to_string()),    // pizza -> pasta (conjoined)
                            ],
                        },
                    ],
                },
            ],
        },

        // ============================================================
        // MULTI-HOP: PASSIVE CONSTRUCTION
        // ============================================================

        // Document 7: "The pizza was eaten by John quickly"
        // Test: passive voice with agent
        // Test: [lemma=eat] >nsubjpass [word=pizza]
        // Test: [lemma=eat] >agent [entity=B-PERSON]
        Document {
            id: "doc_passive".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 7,
                    fields: vec![
                        Field::TokensField { name: "word".to_string(), tokens: vec!["The".to_string(), "pizza".to_string(), "was".to_string(), "eaten".to_string(), "by".to_string(), "John".to_string(), "quickly".to_string()] },
                        Field::TokensField { name: "pos".to_string(), tokens: vec!["DT".to_string(), "NN".to_string(), "VBD".to_string(), "VBN".to_string(), "IN".to_string(), "NNP".to_string(), "RB".to_string()] },
                        Field::TokensField { name: "lemma".to_string(), tokens: vec!["the".to_string(), "pizza".to_string(), "be".to_string(), "eat".to_string(), "by".to_string(), "john".to_string(), "quickly".to_string()] },
                        Field::TokensField { name: "entity".to_string(), tokens: vec!["O".to_string(), "B-FOOD".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "B-PERSON".to_string(), "O".to_string()] },
                        Field::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![3],
                            edges: vec![
                                (1, 0, "det".to_string()),        // pizza -> The
                                (3, 1, "nsubjpass".to_string()),  // eaten -> pizza (passive subject)
                                (3, 2, "auxpass".to_string()),    // eaten -> was (passive aux)
                                (3, 4, "agent".to_string()),      // eaten -> by (agent marker)
                                (4, 5, "pobj".to_string()),       // by -> John
                                (3, 6, "advmod".to_string()),     // eaten -> quickly
                            ],
                        },
                    ],
                },
            ],
        },

        // ============================================================
        // MULTI-HOP: NESTED NOUN PHRASES
        // ============================================================

        // Document 8: "The president of the United States signed the bill"
        // Multi-hop: [word=president] >prep [word=of] >pobj [pos=NNP] <compound [word=United]
        // Tests deep noun phrase structure
        Document {
            id: "doc_nested_np".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 9,
                    fields: vec![
                        Field::TokensField { name: "word".to_string(), tokens: vec!["The".to_string(), "president".to_string(), "of".to_string(), "the".to_string(), "United".to_string(), "States".to_string(), "signed".to_string(), "the".to_string(), "bill".to_string()] },
                        Field::TokensField { name: "pos".to_string(), tokens: vec!["DT".to_string(), "NN".to_string(), "IN".to_string(), "DT".to_string(), "NNP".to_string(), "NNP".to_string(), "VBD".to_string(), "DT".to_string(), "NN".to_string()] },
                        Field::TokensField { name: "lemma".to_string(), tokens: vec!["the".to_string(), "president".to_string(), "of".to_string(), "the".to_string(), "united".to_string(), "states".to_string(), "sign".to_string(), "the".to_string(), "bill".to_string()] },
                        Field::TokensField { name: "entity".to_string(), tokens: vec!["O".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "B-LOC".to_string(), "I-LOC".to_string(), "O".to_string(), "O".to_string(), "O".to_string()] },
                        Field::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![6],
                            edges: vec![
                                (1, 0, "det".to_string()),       // president -> The
                                (1, 2, "prep".to_string()),      // president -> of
                                (2, 5, "pobj".to_string()),      // of -> States
                                (5, 3, "det".to_string()),       // States -> the
                                (5, 4, "compound".to_string()),  // States -> United
                                (6, 1, "nsubj".to_string()),     // signed -> president
                                (6, 8, "dobj".to_string()),      // signed -> bill
                                (8, 7, "det".to_string()),       // bill -> the
                            ],
                        },
                    ],
                },
            ],
        },

        // ============================================================
        // MULTI-HOP: CLAUSAL COMPLEMENT
        // ============================================================

        // Document 9: "John believes Mary knows the truth"
        // Multi-hop: [word=believes] >nsubj [entity=B-PERSON]
        // Multi-hop: [word=believes] >ccomp [word=knows] >nsubj [word=Mary]
        // Triple hop: believes -> knows -> truth
        Document {
            id: "doc_clausal".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 6,
                    fields: vec![
                        Field::TokensField { name: "word".to_string(), tokens: vec!["John".to_string(), "believes".to_string(), "Mary".to_string(), "knows".to_string(), "the".to_string(), "truth".to_string()] },
                        Field::TokensField { name: "pos".to_string(), tokens: vec!["NNP".to_string(), "VBZ".to_string(), "NNP".to_string(), "VBZ".to_string(), "DT".to_string(), "NN".to_string()] },
                        Field::TokensField { name: "lemma".to_string(), tokens: vec!["john".to_string(), "believe".to_string(), "mary".to_string(), "know".to_string(), "the".to_string(), "truth".to_string()] },
                        Field::TokensField { name: "entity".to_string(), tokens: vec!["B-PERSON".to_string(), "O".to_string(), "B-PERSON".to_string(), "O".to_string(), "O".to_string(), "O".to_string()] },
                        Field::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![1],
                            edges: vec![
                                (1, 0, "nsubj".to_string()),   // believes -> John
                                (1, 3, "ccomp".to_string()),   // believes -> knows (clausal complement)
                                (3, 2, "nsubj".to_string()),   // knows -> Mary
                                (3, 5, "dobj".to_string()),    // knows -> truth
                                (5, 4, "det".to_string()),     // truth -> the
                            ],
                        },
                    ],
                },
            ],
        },

        // ============================================================
        // MULTI-HOP: ADVERBIAL CLAUSE
        // ============================================================

        // Document 10: "John left because Mary arrived late"
        // Multi-hop: [word=left] >advcl [word=arrived] >nsubj [word=Mary]
        Document {
            id: "doc_adverbial".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 6,
                    fields: vec![
                        Field::TokensField { name: "word".to_string(), tokens: vec!["John".to_string(), "left".to_string(), "because".to_string(), "Mary".to_string(), "arrived".to_string(), "late".to_string()] },
                        Field::TokensField { name: "pos".to_string(), tokens: vec!["NNP".to_string(), "VBD".to_string(), "IN".to_string(), "NNP".to_string(), "VBD".to_string(), "RB".to_string()] },
                        Field::TokensField { name: "lemma".to_string(), tokens: vec!["john".to_string(), "leave".to_string(), "because".to_string(), "mary".to_string(), "arrive".to_string(), "late".to_string()] },
                        Field::TokensField { name: "entity".to_string(), tokens: vec!["B-PERSON".to_string(), "O".to_string(), "O".to_string(), "B-PERSON".to_string(), "O".to_string(), "O".to_string()] },
                        Field::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![1],
                            edges: vec![
                                (1, 0, "nsubj".to_string()),   // left -> John
                                (1, 4, "advcl".to_string()),   // left -> arrived (adverbial clause)
                                (4, 2, "mark".to_string()),    // arrived -> because
                                (4, 3, "nsubj".to_string()),   // arrived -> Mary
                                (4, 5, "advmod".to_string()),  // arrived -> late
                            ],
                        },
                    ],
                },
            ],
        },
    ]
}
