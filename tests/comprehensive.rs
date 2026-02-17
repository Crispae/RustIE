//! Comprehensive integration tests for RustIE Information Extraction System
//!
//! Migrated from src/bin/comprehensive_test.rs into proper #[test] functions.
//! Tests all rule systems: basic constraints, boolean logic, sequences,
//! graph traversals, named captures, and assertions.

mod common;

use rstest::rstest;

/// Helper: run a paginated query, return the number of hits (panics on query error).
fn query_hits(engine: &rustie::ExtractorEngine, query: &str) -> usize {
    engine
        .query_paginated(query, 100, None)
        .unwrap_or_else(|e| panic!("Query '{}' failed: {}", query, e))
        .total_hits
}

// ======================== Basic Exact Match ========================

#[rstest]
#[case::word_taz("[word=TAZ]", 1)]
#[case::word_family("[word=family]", 1)]
#[case::tag_vbz("[tag=VBZ]", 1)]
#[case::lemma_interact("[lemma=interact]", 1)]
fn basic_exact_match(#[case] query: &str, #[case] expected_min: usize) {
    let (engine, _tmp) = common::test_engine();
    let hits = query_hits(&engine, query);
    assert!(
        hits >= expected_min,
        "Basic exact '{}': expected >= {} hits, got {}",
        query, expected_min, hits
    );
}

// ======================== Basic Regex Match ========================

#[rstest]
#[case::word_starts_t("[word=/T.*/]", 1)]
#[case::tag_any_verb("[tag=/VB.*/]", 1)]
#[case::tag_any_noun("[tag=/NN.*/]", 1)]
fn basic_regex_match(#[case] query: &str, #[case] expected_min: usize) {
    let (engine, _tmp) = common::test_engine();
    let hits = query_hits(&engine, query);
    assert!(
        hits >= expected_min,
        "Regex '{}': expected >= {} hits, got {}",
        query, expected_min, hits
    );
}

// ======================== Boolean AND ========================

#[rstest]
#[case::and_pos_word("[tag=NNP & word=TAZ]", 1)]
#[case::and_pos_lemma("[tag=VBZ & lemma=interact]", 1)]
fn boolean_and(#[case] query: &str, #[case] expected_min: usize) {
    let (engine, _tmp) = common::test_engine();
    let hits = query_hits(&engine, query);
    assert!(
        hits >= expected_min,
        "Boolean AND '{}': expected >= {} hits, got {}",
        query, expected_min, hits
    );
}

// ======================== Boolean OR ========================

#[rstest]
#[case::or_words("[word=TAZ | word=ROOT]", 1)]
#[case::or_pos("[tag=NN | tag=VBZ]", 1)]
fn boolean_or(#[case] query: &str, #[case] expected_min: usize) {
    let (engine, _tmp) = common::test_engine();
    let hits = query_hits(&engine, query);
    assert!(
        hits >= expected_min,
        "Boolean OR '{}': expected >= {} hits, got {}",
        query, expected_min, hits
    );
}

// ======================== Boolean NOT ========================

#[test]
fn boolean_not_lemma() {
    let (engine, _tmp) = common::test_engine();
    let hits = query_hits(&engine, "[tag=NN & !lemma=family]");
    assert!(hits >= 1, "Boolean NOT: expected >= 1 hits, got {}", hits);
}

// ======================== Sequence Patterns ========================

#[rstest]
#[case::det_noun("[tag=DT] [tag=NN]", 1)]
#[case::adj_noun("[tag=JJ] [tag=NN]", 1)]
#[case::proper_verb("[tag=NNP] [tag=VBZ]", 1)]
fn sequence_simple(#[case] query: &str, #[case] expected_min: usize) {
    let (engine, _tmp) = common::test_engine();
    let hits = query_hits(&engine, query);
    assert!(
        hits >= expected_min,
        "Sequence '{}': expected >= {} hits, got {}",
        query, expected_min, hits
    );
}

#[test]
fn sequence_star_det_adj_noun() {
    let (engine, _tmp) = common::test_engine();
    let hits = query_hits(&engine, "[tag=DT] [tag=JJ]* [tag=NN]");
    assert!(hits >= 1, "Seq DT JJ* NN: expected >= 1 hits, got {}", hits);
}

#[test]
fn sequence_optional_det_noun() {
    let (engine, _tmp) = common::test_engine();
    let hits = query_hits(&engine, "[tag=DT]? [tag=NN]");
    assert!(hits >= 1, "Seq DT? NN: expected >= 1 hits, got {}", hits);
}

// ======================== Graph Outgoing ========================

#[rstest]
#[case::nsubj_word("[word=TAZ] >nsubj [word=interacts]", 1)]
#[case::amod_word("[word=TAZ] >amod [word=transcriptional]", 1)]
#[case::verb_noun_nsubj("[tag=/VB.*/] >nsubj [tag=/NN.*/]", 1)]
#[case::proper_adj_amod("[tag=NNP] >amod [tag=JJ]", 1)]
#[case::lemma_amod("[lemma=taz] >amod [lemma=transcriptional]", 1)]
#[case::rel1("[word=ROOT] >rel1 [word=MID]", 1)]
fn graph_outgoing(#[case] query: &str, #[case] expected_min: usize) {
    let (engine, _tmp) = common::test_engine();
    let hits = query_hits(&engine, query);
    assert!(
        hits >= expected_min,
        "Graph outgoing '{}': expected >= {} hits, got {}",
        query, expected_min, hits
    );
}

// ======================== Graph Incoming ========================

#[rstest]
#[case::in_nsubj("[word=interacts] <nsubj [word=TAZ]", 1)]
#[case::in_amod("[word=transcriptional] <amod [word=TAZ]", 1)]
fn graph_incoming(#[case] query: &str, #[case] expected_min: usize) {
    let (engine, _tmp) = common::test_engine();
    let hits = query_hits(&engine, query);
    assert!(
        hits >= expected_min,
        "Graph incoming '{}': expected >= {} hits, got {}",
        query, expected_min, hits
    );
}

// ======================== Named Captures ========================

#[rstest]
#[case::capture_verb("(?<verb>[tag=/VB.*/])", 1)]
#[case::capture_noun("(?<noun>[tag=NN])", 1)]
fn named_capture(#[case] query: &str, #[case] expected_min: usize) {
    let (engine, _tmp) = common::test_engine();
    let hits = query_hits(&engine, query);
    assert!(
        hits >= expected_min,
        "Named capture '{}': expected >= {} hits, got {}",
        query, expected_min, hits
    );
}

// ======================== Complex Patterns ========================

#[rstest]
#[case::full_np("[tag=DT]? [tag=JJ]* [tag=NN]+", 1)]
#[case::verb_np("[tag=/VB.*/] [tag=DT]? [tag=JJ]* [tag=NN]", 1)]
#[case::graph_chain("[word=TAZ] >nsubj [word=interacts]", 1)]
fn complex_pattern(#[case] query: &str, #[case] expected_min: usize) {
    let (engine, _tmp) = common::test_engine();
    let hits = query_hits(&engine, query);
    assert!(
        hits >= expected_min,
        "Complex '{}': expected >= {} hits, got {}",
        query, expected_min, hits
    );
}

// ======================== Detailed Analysis ========================

/// Verify that detailed query results include words and match spans
#[test]
fn detailed_results_include_words_and_matches() {
    let (engine, _tmp) = common::test_engine();
    let result = engine.query_paginated("[word=TAZ]", 100, None).expect("query should not fail");
    assert!(result.total_hits >= 1, "Should have at least 1 hit for [word=TAZ]");

    for sentence in &result.sentence_results {
        // Each sentence result should have a word field
        assert!(
            sentence.fields.contains_key("word"),
            "Sentence result should contain 'word' field"
        );
    }
}
