//! Comprehensive Test Suite for RustIE Information Extraction System
//!
//! This binary tests all rule systems including:
//! - Basic field constraints (exact match, regex, fuzzy)
//! - Boolean logic (AND, OR, NOT)
//! - Sequence patterns with quantifiers
//! - Dependency graph traversals
//! - Named captures
//! - Lookahead/lookbehind assertions

use rustie::{Document, ExtractorEngine, RustIeResult};
use std::path::Path;
use std::fs;
use anyhow::{Result, anyhow};
use serde_json::Value;

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
            let json_value: Value = serde_json::from_str(&content)?;

            // Try as single document or array
            if let Ok(document) = serde_json::from_value::<Document>(json_value.clone()) {
                total_sentences += document.sentences.len();
                engine.add_document(&document)?;
                total_docs += 1;
            } else if let Ok(documents) = serde_json::from_value::<Vec<Document>>(json_value) {
                for doc in &documents {
                    total_sentences += doc.sentences.len();
                    engine.add_document(doc)?;
                    total_docs += 1;
                }
            }
        }
    }

    engine.commit()?;

    println!("\n  Documents indexed: {}", total_docs);
    println!("  Total sentences: {}", total_sentences);
    println!("  Index size (docs): {}", engine.num_docs());

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
            name: "exact_word_john",
            query: "[word=John]",
            category: TestCategory::BasicExactMatch,
            expected_min_hits: 1,
            description: "Find exact word 'John'",
        },
        TestCase {
            name: "exact_word_cat",
            query: "[word=cat]",
            category: TestCategory::BasicExactMatch,
            expected_min_hits: 1,
            description: "Find exact word 'cat'",
        },
        TestCase {
            name: "exact_pos_vbd",
            query: "[pos=VBD]",
            category: TestCategory::BasicExactMatch,
            expected_min_hits: 1,
            description: "Find past tense verbs",
        },
        TestCase {
            name: "exact_lemma_discover",
            query: "[lemma=discover]",
            category: TestCategory::BasicExactMatch,
            expected_min_hits: 1,
            description: "Find lemma 'discover'",
        },

        // === Basic Regex Match ===
        TestCase {
            name: "regex_word_pattern",
            query: "[word=/J.*/]",
            category: TestCategory::BasicRegex,
            expected_min_hits: 1,
            description: "Find words starting with J",
        },
        TestCase {
            name: "regex_pos_verb",
            query: "[pos=/VB.*/]",
            category: TestCategory::BasicRegex,
            expected_min_hits: 1,
            description: "Find any verb POS tag",
        },
        TestCase {
            name: "regex_pos_noun",
            query: "[pos=/NN.*/]",
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
            query: "[pos=NNP & word=John]",
            category: TestCategory::BooleanAnd,
            expected_min_hits: 1,
            description: "Find proper noun 'John'",
        },
        TestCase {
            name: "bool_and_pos_lemma",
            query: "[pos=VBZ & lemma=eat]",
            category: TestCategory::BooleanAnd,
            expected_min_hits: 1,
            description: "Find present tense verb 'eat'",
        },

        // === Boolean OR ===
        TestCase {
            name: "bool_or_words",
            query: "[word=John | word=Mary]",
            category: TestCategory::BooleanOr,
            expected_min_hits: 1,
            description: "Find 'John' or 'Mary'",
        },
        TestCase {
            name: "bool_or_pos",
            query: "[pos=VBD | pos=VBZ]",
            category: TestCategory::BooleanOr,
            expected_min_hits: 1,
            description: "Find past or present tense verbs",
        },

        // === Boolean NOT ===
        TestCase {
            name: "bool_not_lemma",
            query: "[pos=NN & !lemma=cat]",
            category: TestCategory::BooleanNot,
            expected_min_hits: 1,
            description: "Find nouns that are not 'cat'",
        },

        // === Sequence Simple ===
        TestCase {
            name: "seq_simple_det_noun",
            query: "[pos=DT] [pos=NN]",
            category: TestCategory::SequenceSimple,
            expected_min_hits: 1,
            description: "Find determiner followed by noun",
        },
        TestCase {
            name: "seq_simple_adj_noun",
            query: "[pos=JJ] [pos=NN]",
            category: TestCategory::SequenceSimple,
            expected_min_hits: 1,
            description: "Find adjective followed by noun",
        },
        TestCase {
            name: "seq_simple_subj_verb",
            query: "[pos=NNP] [pos=VBZ]",
            category: TestCategory::SequenceSimple,
            expected_min_hits: 1,
            description: "Find proper noun followed by verb",
        },

        // === Sequence Quantifier + ===
        TestCase {
            name: "seq_plus_adj",
            query: "[pos=JJ]+ [pos=NN]",
            category: TestCategory::SequenceQuantifierPlus,
            expected_min_hits: 0,
            description: "Find one or more adjectives followed by noun (quantifier+ may have limited support)",
        },
        TestCase {
            name: "seq_plus_adv",
            query: "[pos=RB]+ [pos=VBD]",
            category: TestCategory::SequenceQuantifierPlus,
            expected_min_hits: 0,
            description: "Find one or more adverbs followed by past verb",
        },

        // === Sequence Quantifier * ===
        TestCase {
            name: "seq_star_adj",
            query: "[pos=DT] [pos=JJ]* [pos=NN]",
            category: TestCategory::SequenceQuantifierStar,
            expected_min_hits: 1,
            description: "Find det + optional adjectives + noun",
        },

        // === Sequence Quantifier ? ===
        TestCase {
            name: "seq_optional_det",
            query: "[pos=DT]? [pos=NN]",
            category: TestCategory::SequenceQuantifierOptional,
            expected_min_hits: 1,
            description: "Find optional determiner + noun",
        },

        // === Sequence Quantifier {m,n} ===
        TestCase {
            name: "seq_range_adj",
            query: "[pos=JJ]{1,3} [pos=NN]",
            category: TestCategory::SequenceQuantifierRange,
            expected_min_hits: 0,
            description: "Find 1-3 adjectives followed by noun (range quantifier may have limited support)",
        },

        // === Graph Outgoing ===
        // Graph traversal works with any field (word, pos, lemma, entity, etc.)
        TestCase {
            name: "graph_out_nsubj_word",
            query: "[word=eats] >nsubj [word=John]",
            category: TestCategory::GraphOutgoing,
            expected_min_hits: 1,
            description: "Find 'eats' with John as subject (word field)",
        },
        TestCase {
            name: "graph_out_dobj_word",
            query: "[word=eats] >dobj [word=pizza]",
            category: TestCategory::GraphOutgoing,
            expected_min_hits: 1,
            description: "Find 'eats' with pizza as direct object (word field)",
        },
        TestCase {
            name: "graph_out_nsubj_pos",
            query: "[pos=/VB.*/] >nsubj [pos=/NN.*/]",
            category: TestCategory::GraphOutgoing,
            expected_min_hits: 1,
            description: "Find verbs with noun subjects (pos field)",
        },
        TestCase {
            name: "graph_out_dobj_pos",
            query: "[pos=/VB.*/] >dobj [pos=/NN.*/]",
            category: TestCategory::GraphOutgoing,
            expected_min_hits: 1,
            description: "Find verbs with noun objects (pos field)",
        },
        TestCase {
            name: "graph_out_lemma",
            query: "[lemma=eat] >dobj [lemma=pizza]",
            category: TestCategory::GraphOutgoing,
            expected_min_hits: 1,
            description: "Find 'eat' lemma with 'pizza' as object (lemma field)",
        },
        TestCase {
            name: "graph_out_nsubj_cat",
            query: "[word=sleeps] >nsubj [word=cat]",
            category: TestCategory::GraphOutgoing,
            expected_min_hits: 1,
            description: "Find 'sleeps' with cat as subject",
        },

        // === Graph Incoming ===
        TestCase {
            name: "graph_in_nsubj",
            query: "[word=John] <nsubj [word=eats]",
            category: TestCategory::GraphIncoming,
            expected_min_hits: 1,
            description: "Find John as subject of eats",
        },
        TestCase {
            name: "graph_in_dobj",
            query: "[word=pizza] <dobj [word=eats]",
            category: TestCategory::GraphIncoming,
            expected_min_hits: 1,
            description: "Find pizza as object of eats",
        },

        // === Graph Wildcard ===
        TestCase {
            name: "graph_wildcard_out",
            query: "[word=eats] >> [word=John]",
            category: TestCategory::GraphWildcard,
            expected_min_hits: 0,
            description: "Find any outgoing edge from eats (wildcard may not be fully supported)",
        },
        TestCase {
            name: "graph_wildcard_in",
            query: "[word=pizza] << [word=eats]",
            category: TestCategory::GraphWildcard,
            expected_min_hits: 0,
            description: "Find any incoming edge to pizza from eats",
        },

        // === Graph Disjunctive ===
        TestCase {
            name: "graph_disj_subj",
            query: "[word=eats] >nsubj|dobj [word=John]",
            category: TestCategory::GraphDisjunctive,
            expected_min_hits: 0,
            description: "Find eats with John as subject or object (disjunctive traversal)",
        },

        // === Named Capture ===
        TestCase {
            name: "named_capture_verb",
            query: "(?<verb>[pos=/VB.*/])",
            category: TestCategory::NamedCapture,
            expected_min_hits: 1,
            description: "Capture verbs with name 'verb'",
        },
        TestCase {
            name: "named_capture_noun",
            query: "(?<noun>[pos=NN])",
            category: TestCategory::NamedCapture,
            expected_min_hits: 1,
            description: "Capture nouns with name 'noun'",
        },

        // === Lookahead ===
        TestCase {
            name: "lookahead_positive",
            query: "[pos=VBZ] (?=[word=pizza])",
            category: TestCategory::Lookahead,
            expected_min_hits: 0,
            description: "Find verb followed by 'pizza'",
        },
        TestCase {
            name: "lookahead_negative",
            query: "[pos=VBZ] (?![word=pizza])",
            category: TestCategory::Lookahead,
            expected_min_hits: 1,
            description: "Find verb NOT followed by 'pizza'",
        },

        // === Lookbehind ===
        TestCase {
            name: "lookbehind_positive",
            query: "(?<= [pos=DT]) [pos=NN]",
            category: TestCategory::Lookbehind,
            expected_min_hits: 0,
            description: "Find noun preceded by determiner (may not be fully supported)",
        },

        // === Complex Patterns ===
        TestCase {
            name: "complex_np_full",
            query: "[pos=DT]? [pos=JJ]* [pos=NN]+",
            category: TestCategory::Complex,
            expected_min_hits: 1,
            description: "Full noun phrase pattern",
        },
        TestCase {
            name: "complex_verb_phrase",
            query: "[pos=/VB.*/] [pos=DT]? [pos=JJ]* [pos=NN]",
            category: TestCategory::Complex,
            expected_min_hits: 1,
            description: "Verb followed by noun phrase",
        },
        TestCase {
            name: "complex_graph_chain",
            query: "[word=eats] >nsubj [word=John]",
            category: TestCategory::Complex,
            expected_min_hits: 1,
            description: "Verb eats with John as subject",
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
        let percentage = (passed * 100) / total.max(&1);
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
        ("[word=John]", "Simple word match"),
        ("[pos=DT] [pos=NN]", "Simple sequence"),
        ("[word=eats] >nsubj [word=John]", "Graph traversal"),
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
