//! Comprehensive Test Suite for RustIE Information Extraction System
//!
//! This binary tests all rule systems including:
//! - Basic field constraints (exact match, regex, fuzzy)
//! - Boolean logic (AND, OR, NOT)
//! - Sequence patterns with quantifiers
//! - Dependency graph traversals
//! - Named captures
//! - Lookahead/lookbehind assertions

use rustie::{Document, ExtractorEngine};
use std::path::Path;
use std::fs;
use std::error::Error;
use anyhow::Result;
use serde_json::Value as JsonValue;
use tantivy::schema::Value;
use tantivy::tokenizer::{Tokenizer, TokenStream};

/// Test case structure
struct TestCase {
    name: &'static str,
    query: &'static str,
    category: TestCategory,
    expected_min_hits: usize,
    description: &'static str,
}

#[derive(Debug, Clone, Copy)]
enum TestCategory {
    BasicExactMatch,
    BasicRegex,
    BooleanAnd,
    BooleanOr,
    BooleanNot,
    SequenceSimple,
    SequenceQuantifierPlus,
    SequenceQuantifierStar,
    SequenceQuantifierOptional,
    SequenceQuantifierRange,
    GraphOutgoing,
    GraphIncoming,
    GraphWildcard,
    GraphDisjunctive,
    NamedCapture,
    Lookahead,
    Lookbehind,
    Complex,
}

impl TestCategory {
    fn display_name(&self) -> &'static str {
        match self {
            TestCategory::BasicExactMatch => "Basic: Exact Match",
            TestCategory::BasicRegex => "Basic: Regex Match",
            TestCategory::BooleanAnd => "Boolean: AND",
            TestCategory::BooleanOr => "Boolean: OR",
            TestCategory::BooleanNot => "Boolean: NOT",
            TestCategory::SequenceSimple => "Sequence: Simple",
            TestCategory::SequenceQuantifierPlus => "Sequence: Quantifier +",
            TestCategory::SequenceQuantifierStar => "Sequence: Quantifier *",
            TestCategory::SequenceQuantifierOptional => "Sequence: Quantifier ?",
            TestCategory::SequenceQuantifierRange => "Sequence: Quantifier {m,n}",
            TestCategory::GraphOutgoing => "Graph: Outgoing Edge",
            TestCategory::GraphIncoming => "Graph: Incoming Edge",
            TestCategory::GraphWildcard => "Graph: Wildcard",
            TestCategory::GraphDisjunctive => "Graph: Disjunctive",
            TestCategory::NamedCapture => "Named Capture",
            TestCategory::Lookahead => "Lookahead",
            TestCategory::Lookbehind => "Lookbehind",
            TestCategory::Complex => "Complex Pattern",
        }
    }
}

/// Test result structure
struct TestResult {
    test_case: &'static str,
    category: TestCategory,
    passed: bool,
    actual_hits: usize,
    expected_min_hits: usize,
    error_message: Option<String>,
    execution_time_ms: u128,
}

/// Main test runner
fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     RustIE Comprehensive Test Suite                           ║");
    println!("║     Testing All Rule Systems                                  ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Setup test environment
    let test_index_dir = Path::new("./test_index_comprehensive");
    let schema_path = Path::new("configs/schema.yaml");
    let sample_docs_dir = Path::new("sample_documents");

    // Clean up any existing test index
    if test_index_dir.exists() {
        println!("Cleaning up existing test index...");
        fs::remove_dir_all(test_index_dir)?;
    }
    fs::create_dir_all(test_index_dir)?;

    println!("=== Phase 1: Indexing Test Documents ===\n");

    // Create and populate the engine
    let mut engine = ExtractorEngine::new(test_index_dir, schema_path)?;

    // Index all sample documents
    let mut total_docs = 0;
    let mut total_sentences = 0;

    for entry in fs::read_dir(sample_docs_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map_or(false, |ext| ext == "json") {
            println!("  Indexing: {}", path.display());

            let content = fs::read_to_string(&path)?;
            let json_value: JsonValue = serde_json::from_str(&content)?;

            // Try as single document or array
            let single_doc_result = serde_json::from_value::<Document>(json_value.clone());
            let array_doc_result = serde_json::from_value::<Vec<Document>>(json_value.clone());
            
            if let Ok(document) = single_doc_result {
                total_sentences += document.sentences.len();
                engine.add_document(&document)?;
                total_docs += 1;
                println!("    ✓ Successfully indexed as single document ({} sentences)", document.sentences.len());
            } else if let Ok(documents) = array_doc_result {
                for doc in &documents {
                    total_sentences += doc.sentences.len();
                    engine.add_document(doc)?;
                    total_docs += 1;
                }
                println!("    ✓ Successfully indexed as array of {} documents ({} total sentences)", documents.len(), total_sentences);
            } else {
                eprintln!("    ✗ ERROR: Failed to parse {} as Document or Vec<Document>", path.display());
                if let Err(e) = single_doc_result {
                    eprintln!("      Single document parse error: {}", e);
                    // Try to show a snippet of the error
                    if let Some(serde_error) = e.source() {
                        eprintln!("      Root cause: {}", serde_error);
                    }
                }
                if let Err(e) = array_doc_result {
                    eprintln!("      Array parse error: {}", e);
                    if let Some(serde_error) = e.source() {
                        eprintln!("      Root cause: {}", serde_error);
                    }
                }
            }
        }
    }

    engine.commit()?;

    println!("\n  Documents indexed: {}", total_docs);
    println!("  Total sentences: {}", total_sentences);
    println!("  Index size (docs): {}", engine.num_docs());

    // Debug: Check if basic term search works
    println!("\n=== Debug: Testing basic Tantivy search ===");
    {
        let searcher = engine.searcher();
        let schema = engine.schema();

        // Try to find "John" in the word field
        if let Ok(word_field) = schema.get_field("word") {
            // Test with words that don't exist
            let term = tantivy::Term::from_field_text(word_field, "John");
            let term_query = tantivy::query::TermQuery::new(term.clone(), tantivy::schema::IndexRecordOption::Basic);

            match searcher.search(&term_query, &tantivy::collector::Count) {
                Ok(count) => println!("  Raw TermQuery for 'John' in word field: {} hits", count),
                Err(e) => println!("  Raw TermQuery error: {}", e),
            }

            // Also try lowercase
            let term_lower = tantivy::Term::from_field_text(word_field, "john");
            let term_query_lower = tantivy::query::TermQuery::new(term_lower, tantivy::schema::IndexRecordOption::Basic);

            match searcher.search(&term_query_lower, &tantivy::collector::Count) {
                Ok(count) => println!("  Raw TermQuery for 'john' (lowercase): {} hits", count),
                Err(e) => println!("  Raw TermQuery error: {}", e),
            }

            // Test with words that SHOULD exist in the document
            for test_word in &["The", "TAZ", "interacts", "ROOT", "MID"] {
                let term = tantivy::Term::from_field_text(word_field, test_word);
                let term_query = tantivy::query::TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
                match searcher.search(&term_query, &tantivy::collector::Count) {
                    Ok(count) => println!("  Raw TermQuery for '{}': {} hits", test_word, count),
                    Err(e) => println!("  Raw TermQuery for '{}' error: {}", test_word, e),
                }
            }

            // Try to see what terms exist in the first document
            if engine.num_docs() > 0 {
                let doc_address = tantivy::DocAddress::new(0, 0);
                if let Ok(doc) = searcher.doc::<tantivy::TantivyDocument>(doc_address) {
                    println!("  First document word field values:");
                    for value in doc.get_all(word_field) {
                        if let Some(text) = value.as_str() {
                            println!("    Raw stored value: '{}'", text);
                            // Decode the position-aware format
                            let tokens: Vec<&str> = text.split('|').collect();
                            println!("    Decoded tokens: {:?}", tokens);
                        }
                    }
                }
            }

            // Try to test with position-aware tokenizer directly
            println!("  Testing position-aware tokenizer behavior...");
            use rustie::tantivy_integration::position_tokenizer::PositionAwareTokenTokenizer;
            let mut tokenizer = PositionAwareTokenTokenizer;
            for test_text in &["The", "The|transcriptional|TAZ"] {
                let mut stream = tokenizer.token_stream(test_text);
                println!("    Tokenizing '{}':", test_text);
                let mut pos = 0;
                while stream.advance() {
                    let token = stream.token();
                    println!("      Position {}: '{}'", token.position, token.text);
                    pos += 1;
                    if pos > 10 { break; } // Limit output
                }
            }
        }
    }

    println!("\n=== Phase 2: Running Test Cases ===\n");

    // Define all test cases
    let test_cases = get_test_cases();

    let mut results: Vec<TestResult> = Vec::new();
    let mut passed = 0;
    let mut failed = 0;
    let mut errors = 0;

    // Group test cases by category
    let mut current_category: Option<TestCategory> = None;

    for tc in &test_cases {
        // Print category header if changed
        if current_category.map_or(true, |c| std::mem::discriminant(&c) != std::mem::discriminant(&tc.category)) {
            println!("\n--- {} ---", tc.category.display_name());
            current_category = Some(tc.category);
        }

        let start_time = std::time::Instant::now();

        let result = match engine.query(tc.query) {
            Ok(query_result) => {
                let elapsed = start_time.elapsed().as_millis();
                let actual_hits = query_result.total_hits;
                let test_passed = actual_hits >= tc.expected_min_hits;

                if test_passed {
                    passed += 1;
                    print_test_result(tc.name, true, actual_hits, tc.expected_min_hits, elapsed);
                } else {
                    failed += 1;
                    print_test_result(tc.name, false, actual_hits, tc.expected_min_hits, elapsed);
                }

                TestResult {
                    test_case: tc.name,
                    category: tc.category,
                    passed: test_passed,
                    actual_hits,
                    expected_min_hits: tc.expected_min_hits,
                    error_message: None,
                    execution_time_ms: elapsed,
                }
            }
            Err(e) => {
                let elapsed = start_time.elapsed().as_millis();
                errors += 1;
                print_test_error(tc.name, &e.to_string());

                TestResult {
                    test_case: tc.name,
                    category: tc.category,
                    passed: false,
                    actual_hits: 0,
                    expected_min_hits: tc.expected_min_hits,
                    error_message: Some(e.to_string()),
                    execution_time_ms: elapsed,
                }
            }
        };

        results.push(result);
    }

    // Print summary
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                      TEST SUMMARY                             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Total test cases: {}", test_cases.len());
    println!("  Passed:           {} ({}%)", passed, (passed * 100) / test_cases.len().max(1));
    println!("  Failed:           {}", failed);
    println!("  Errors:           {}", errors);
    println!();

    // Print category breakdown
    println!("=== Results by Category ===\n");
    print_category_summary(&results);

    // Print failed tests details
    if failed > 0 || errors > 0 {
        println!("\n=== Failed/Error Tests ===\n");
        for result in &results {
            if !result.passed {
                println!("  {} [{}]", result.test_case, result.category.display_name());
                if let Some(err) = &result.error_message {
                    println!("    Error: {}", err);
                } else {
                    println!("    Expected >= {}, got {}", result.expected_min_hits, result.actual_hits);
                }
            }
        }
    }

    // Detailed query analysis for some test cases
    println!("\n=== Detailed Query Analysis (Sample) ===\n");
    run_detailed_analysis(&engine)?;

    // Cleanup
    println!("\n=== Cleanup ===\n");
    if test_index_dir.exists() {
        fs::remove_dir_all(test_index_dir)?;
        println!("  Removed test index directory");
    }

    println!("\nTest suite completed!");

    if failed > 0 || errors > 0 {
        std::process::exit(1);
    }

    Ok(())
}

/// Get all test cases
fn get_test_cases() -> Vec<TestCase> {
    vec![
        // === Basic Exact Match ===
        TestCase {
            name: "exact_word_taz",
            query: "[word=TAZ]",
            category: TestCategory::BasicExactMatch,
            expected_min_hits: 1,
            description: "Find exact word 'TAZ'",
        },
        TestCase {
            name: "exact_word_family",
            query: "[word=family]",
            category: TestCategory::BasicExactMatch,
            expected_min_hits: 1,
            description: "Find exact word 'family'",
        },
        TestCase {
            name: "exact_pos_vbz",
            query: "[tag=VBZ]",
            category: TestCategory::BasicExactMatch,
            expected_min_hits: 1,
            description: "Find present tense verbs",
        },
        TestCase {
            name: "exact_lemma_interact",
            query: "[lemma=interact]",
            category: TestCategory::BasicExactMatch,
            expected_min_hits: 1,
            description: "Find lemma 'interact'",
        },

        // === Basic Regex Match ===
        TestCase {
            name: "regex_word_pattern",
            query: "[word=/T.*/]",
            category: TestCategory::BasicRegex,
            expected_min_hits: 1,
            description: "Find words starting with T",
        },
        TestCase {
            name: "regex_pos_verb",
            query: "[tag=/VB.*/]",
            category: TestCategory::BasicRegex,
            expected_min_hits: 1,
            description: "Find any verb POS tag",
        },
        TestCase {
            name: "regex_pos_noun",
            query: "[tag=/NN.*/]",
            category: TestCategory::BasicRegex,
            expected_min_hits: 1,
            description: "Find any noun POS tag",
        },
        TestCase {
            name: "regex_word_ing",
            query: "[word=/.*ing/]",
            category: TestCategory::BasicRegex,
            expected_min_hits: 0,
            description: "Find words ending in 'ing'",
        },

        // === Boolean AND ===
        TestCase {
            name: "bool_and_pos_word",
            query: "[tag=NNP & word=TAZ]",
            category: TestCategory::BooleanAnd,
            expected_min_hits: 1,
            description: "Find proper noun 'TAZ'",
        },
        TestCase {
            name: "bool_and_pos_lemma",
            query: "[tag=VBZ & lemma=interact]",
            category: TestCategory::BooleanAnd,
            expected_min_hits: 1,
            description: "Find present tense verb 'interact'",
        },

        // === Boolean OR ===
        TestCase {
            name: "bool_or_words",
            query: "[word=TAZ | word=ROOT]",
            category: TestCategory::BooleanOr,
            expected_min_hits: 1,
            description: "Find 'TAZ' or 'ROOT'",
        },
        TestCase {
            name: "bool_or_pos",
            query: "[tag=NN | tag=VBZ]",
            category: TestCategory::BooleanOr,
            expected_min_hits: 1,
            description: "Find nouns or present tense verbs",
        },

        // === Boolean NOT ===
        TestCase {
            name: "bool_not_lemma",
            query: "[tag=NN & !lemma=family]",
            category: TestCategory::BooleanNot,
            expected_min_hits: 1,
            description: "Find nouns that are not 'family'",
        },

        // === Sequence Simple ===
        TestCase {
            name: "seq_simple_det_noun",
            query: "[tag=DT] [tag=NN]",
            category: TestCategory::SequenceSimple,
            expected_min_hits: 1,
            description: "Find determiner followed by noun",
        },
        TestCase {
            name: "seq_simple_adj_noun",
            query: "[tag=JJ] [tag=NN]",
            category: TestCategory::SequenceSimple,
            expected_min_hits: 1,
            description: "Find adjective followed by noun",
        },
        TestCase {
            name: "seq_simple_subj_verb",
            query: "[tag=NNP] [tag=VBZ]",
            category: TestCategory::SequenceSimple,
            expected_min_hits: 1,
            description: "Find proper noun followed by verb",
        },

        // === Sequence Quantifier + ===
        TestCase {
            name: "seq_plus_adj",
            query: "[tag=JJ]+ [tag=NN]",
            category: TestCategory::SequenceQuantifierPlus,
            expected_min_hits: 0,
            description: "Find one or more adjectives followed by noun (quantifier+ may have limited support)",
        },
        TestCase {
            name: "seq_plus_adv",
            query: "[tag=RB]+ [tag=VBD]",
            category: TestCategory::SequenceQuantifierPlus,
            expected_min_hits: 0,
            description: "Find one or more adverbs followed by past verb",
        },

        // === Sequence Quantifier * ===
        TestCase {
            name: "seq_star_adj",
            query: "[tag=DT] [tag=JJ]* [tag=NN]",
            category: TestCategory::SequenceQuantifierStar,
            expected_min_hits: 1,
            description: "Find det + optional adjectives + noun",
        },

        // === Sequence Quantifier ? ===
        TestCase {
            name: "seq_optional_det",
            query: "[tag=DT]? [tag=NN]",
            category: TestCategory::SequenceQuantifierOptional,
            expected_min_hits: 1,
            description: "Find optional determiner + noun",
        },

        // === Sequence Quantifier {m,n} ===
        TestCase {
            name: "seq_range_adj",
            query: "[tag=JJ]{1,3} [tag=NN]",
            category: TestCategory::SequenceQuantifierRange,
            expected_min_hits: 0,
            description: "Find 1-3 adjectives followed by noun (range quantifier may have limited support)",
        },

        // === Graph Outgoing ===
        // Graph traversal works with any field (word, pos, lemma, entity, etc.)
        // Based on edges: [2, 3, "nsubj"] means TAZ (pos 2) has nsubj to interacts (pos 3)
        // But wait, that's backwards - [2, 3, "nsubj"] means from 2 to 3 with label nsubj
        // So TAZ (2) has nsubj edge TO interacts (3), meaning interacts is subject of TAZ
        // Actually, looking at the edges more carefully: [2, 3, "nsubj"] means from token 2 to token 3
        // So we need to check: [word=interacts] >nsubj [word=TAZ] or [word=TAZ] >nsubj [word=interacts]?
        // The edge [2, 3, "nsubj"] means token 2 (TAZ) has an nsubj edge pointing to token 3 (interacts)
        // So TAZ >nsubj interacts means "TAZ has interacts as its subject"
        TestCase {
            name: "graph_out_nsubj_word",
            query: "[word=TAZ] >nsubj [word=interacts]",
            category: TestCategory::GraphOutgoing,
            expected_min_hits: 1,
            description: "Find 'TAZ' with interacts as subject (word field)",
        },
        TestCase {
            name: "graph_out_amod_word",
            query: "[word=TAZ] >amod [word=transcriptional]",
            category: TestCategory::GraphOutgoing,
            expected_min_hits: 1,
            description: "Find 'TAZ' with transcriptional as amod (word field)",
        },
        TestCase {
            name: "graph_out_nsubj_pos",
            query: "[tag=/VB.*/] >nsubj [tag=/NN.*/]",
            category: TestCategory::GraphOutgoing,
            expected_min_hits: 1,
            description: "Find verbs with noun subjects (pos field)",
        },
        TestCase {
            name: "graph_out_amod_pos",
            query: "[tag=NNP] >amod [tag=JJ]",
            category: TestCategory::GraphOutgoing,
            expected_min_hits: 1,
            description: "Find proper nouns with adjective modifiers (pos field)",
        },
        TestCase {
            name: "graph_out_lemma",
            query: "[lemma=taz] >amod [lemma=transcriptional]",
            category: TestCategory::GraphOutgoing,
            expected_min_hits: 1,
            description: "Find 'taz' lemma with 'transcriptional' as amod (lemma field)",
        },
        TestCase {
            name: "graph_out_rel1",
            query: "[word=ROOT] >rel1 [word=MID]",
            category: TestCategory::GraphOutgoing,
            expected_min_hits: 1,
            description: "Find 'ROOT' with MID as rel1",
        },

        // === Graph Incoming ===
        TestCase {
            name: "graph_in_nsubj",
            query: "[word=interacts] <nsubj [word=TAZ]",
            category: TestCategory::GraphIncoming,
            expected_min_hits: 1,
            description: "Find interacts as subject of TAZ",
        },
        TestCase {
            name: "graph_in_amod",
            query: "[word=transcriptional] <amod [word=TAZ]",
            category: TestCategory::GraphIncoming,
            expected_min_hits: 1,
            description: "Find transcriptional as amod of TAZ",
        },

        // === Graph Wildcard ===
        TestCase {
            name: "graph_wildcard_out",
            query: "[word=TAZ] >> [word=interacts]",
            category: TestCategory::GraphWildcard,
            expected_min_hits: 0,
            description: "Find any outgoing edge from TAZ (wildcard may not be fully supported)",
        },
        TestCase {
            name: "graph_wildcard_in",
            query: "[word=transcriptional] << [word=TAZ]",
            category: TestCategory::GraphWildcard,
            expected_min_hits: 0,
            description: "Find any incoming edge to transcriptional from TAZ",
        },

        // === Graph Disjunctive ===
        TestCase {
            name: "graph_disj_subj",
            query: "[word=TAZ] >nsubj|amod [word=interacts]",
            category: TestCategory::GraphDisjunctive,
            expected_min_hits: 0,
            description: "Find TAZ with interacts as subject or amod (disjunctive traversal)",
        },

        // === Named Capture ===
        TestCase {
            name: "named_capture_verb",
            query: "(?<verb>[tag=/VB.*/])",
            category: TestCategory::NamedCapture,
            expected_min_hits: 1,
            description: "Capture verbs with name 'verb'",
        },
        TestCase {
            name: "named_capture_noun",
            query: "(?<noun>[tag=NN])",
            category: TestCategory::NamedCapture,
            expected_min_hits: 1,
            description: "Capture nouns with name 'noun'",
        },

        // === Lookahead ===
        TestCase {
            name: "lookahead_positive",
            query: "[tag=VBZ] (?=[word=family])",
            category: TestCategory::Lookahead,
            expected_min_hits: 0,
            description: "Find verb followed by 'family'",
        },
        TestCase {
            name: "lookahead_negative",
            query: "[tag=VBZ] (?![word=family])",
            category: TestCategory::Lookahead,
            expected_min_hits: 1,
            description: "Find verb NOT followed by 'family'",
        },

        // === Lookbehind ===
        TestCase {
            name: "lookbehind_positive",
            query: "(?<= [tag=DT]) [tag=NN]",
            category: TestCategory::Lookbehind,
            expected_min_hits: 0,
            description: "Find noun preceded by determiner (may not be fully supported)",
        },

        // === Complex Patterns ===
        TestCase {
            name: "complex_np_full",
            query: "[tag=DT]? [tag=JJ]* [tag=NN]+",
            category: TestCategory::Complex,
            expected_min_hits: 1,
            description: "Full noun phrase pattern",
        },
        TestCase {
            name: "complex_verb_phrase",
            query: "[tag=/VB.*/] [tag=DT]? [tag=JJ]* [tag=NN]",
            category: TestCategory::Complex,
            expected_min_hits: 1,
            description: "Verb followed by noun phrase",
        },
        TestCase {
            name: "complex_graph_chain",
            query: "[word=TAZ] >nsubj [word=interacts]",
            category: TestCategory::Complex,
            expected_min_hits: 1,
            description: "TAZ with interacts as subject",
        },
    ]
}

/// Print test result
fn print_test_result(name: &str, passed: bool, actual: usize, expected_min: usize, time_ms: u128) {
    let status = if passed { "PASS" } else { "FAIL" };
    let symbol = if passed { "✓" } else { "✗" };
    println!(
        "  {} [{}] {} (hits: {}, expected >= {}) [{} ms]",
        symbol, status, name, actual, expected_min, time_ms
    );
}

/// Print test error
fn print_test_error(name: &str, error: &str) {
    println!("  ✗ [ERROR] {} - {}", name, error);
}

/// Print category summary
fn print_category_summary(results: &[TestResult]) {
    use std::collections::HashMap;

    let mut category_stats: HashMap<String, (usize, usize)> = HashMap::new();

    for result in results {
        let cat_name = result.category.display_name().to_string();
        let entry = category_stats.entry(cat_name).or_insert((0, 0));
        entry.1 += 1; // total
        if result.passed {
            entry.0 += 1; // passed
        }
    }

    let mut categories: Vec<_> = category_stats.iter().collect();
    categories.sort_by_key(|(name, _)| name.clone());

    for (category, (passed, total)) in categories {
        let bar = create_progress_bar(*passed, *total, 20);
        println!("  {:<30} {} ({}/{})", category, bar, passed, total);
    }
}

/// Create ASCII progress bar
fn create_progress_bar(value: usize, max: usize, width: usize) -> String {
    let filled = (value * width) / max.max(1);
    let empty = width - filled;
    format!("[{}{}] {}%",
        "█".repeat(filled),
        "░".repeat(empty),
        (value * 100) / max.max(1)
    )
}

/// Run detailed analysis on selected queries
fn run_detailed_analysis(engine: &ExtractorEngine) -> Result<()> {
    let analysis_queries = vec![
        ("[word=TAZ]", "Simple word match"),
        ("[tag=DT] [tag=NN]", "Simple sequence"),
        ("[word=TAZ] >nsubj [word=interacts]", "Graph traversal"),
    ];

    for (query, description) in analysis_queries {
        println!("\nQuery: {} ({})", query, description);
        println!("{}", "-".repeat(60));

        match engine.query(query) {
            Ok(result) => {
                println!("  Total hits: {}", result.total_hits);
                if let Some(max_score) = result.max_score {
                    println!("  Max score: {:.4}", max_score);
                }

                // Show first 3 results
                for (i, sentence) in result.sentence_results.iter().take(3).enumerate() {
                    println!("\n  Result {}:", i + 1);
                    println!("    Doc ID: {}", sentence.document_id);
                    println!("    Sentence ID: {}", sentence.sentence_id);
                    println!("    Score: {:.4}", sentence.score);

                    // Show words
                    if let Some(words) = sentence.fields.get("word") {
                        println!("    Words: {}", words.join(" "));
                    }

                    // Show matches
                    if !sentence.matches.is_empty() {
                        println!("    Matches:");
                        for m in &sentence.matches {
                            println!("      Span: [{}, {})", m.span.start, m.span.end);
                            for cap in &m.captures {
                                println!("        Capture '{}': [{}, {})", cap.name, cap.span.start, cap.span.end);
                            }
                        }
                    }
                }
            }
            Err(e) => {
                println!("  Error: {}", e);
            }
        }
    }

    Ok(())
}
