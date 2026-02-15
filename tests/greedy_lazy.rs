//! Greedy vs lazy quantifier behavior tests
//!
//! Migrated from src/bin/test_greedy_lazy.rs into proper #[test] functions.
//! Tests that lazy quantifiers produce shorter/more matches and greedy produce longer/fewer.

use rustie::query::ast::{Constraint, Matcher, Pattern, QuantifierKind};
use rustie::tantivy_integration::concat_query::find_constraint_spans_in_sequence;
use std::collections::HashMap;

fn create_field_cache(tokens: Vec<&str>) -> HashMap<String, Vec<String>> {
    let mut cache = HashMap::new();
    cache.insert(
        "word".to_string(),
        tokens.iter().map(|s| s.to_string()).collect(),
    );
    cache
}

/// Build pattern: `a []*? c` (lazy) or `a []* c` (greedy)
fn build_a_wildcard_c(kind: QuantifierKind) -> Pattern {
    Pattern::Concatenated(vec![
        Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::String("a".to_string()),
        }),
        Pattern::Repetition {
            pattern: Box::new(Pattern::Constraint(Constraint::Wildcard)),
            min: 0,
            max: None,
            kind,
        },
        Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::String("c".to_string()),
        }),
    ])
}

#[test]
fn lazy_quantifier_produces_multiple_short_matches() {
    let field_cache = create_field_cache(vec!["a", "b", "c", "a", "b", "c"]);
    let pattern = build_a_wildcard_c(QuantifierKind::Lazy);
    let results = find_constraint_spans_in_sequence(&pattern, &field_cache);

    // Lazy should produce at least 2 matches (one for each "a...c")
    assert!(
        results.len() >= 2,
        "Lazy quantifier should produce >= 2 matches, got {}",
        results.len()
    );

    // The match starting at position 0 should be short (length 3: a b c)
    let short_match = results.iter().find(|r| r.span.start == 0);
    assert!(short_match.is_some(), "Should have a match starting at position 0");
    assert_eq!(
        short_match.unwrap().span.length(),
        3,
        "Lazy match at pos 0 should be shortest (length 3)"
    );
}

#[test]
fn greedy_quantifier_produces_one_long_match() {
    let field_cache = create_field_cache(vec!["a", "b", "c", "a", "b", "c"]);
    let pattern = build_a_wildcard_c(QuantifierKind::Greedy);
    let results = find_constraint_spans_in_sequence(&pattern, &field_cache);

    assert!(
        !results.is_empty(),
        "Greedy quantifier should produce at least 1 match"
    );

    // Greedy should produce a single match of length 6 (the entire sequence)
    assert_eq!(
        results.len(),
        1,
        "Greedy quantifier should produce exactly 1 match, got {}",
        results.len()
    );
    assert_eq!(
        results[0].span.length(),
        6,
        "Greedy match should span the full sequence (length 6), got {}",
        results[0].span.length()
    );
}

#[test]
fn lazy_matches_are_shorter_than_greedy() {
    let field_cache = create_field_cache(vec!["a", "b", "c", "a", "b", "c"]);

    let lazy_results =
        find_constraint_spans_in_sequence(&build_a_wildcard_c(QuantifierKind::Lazy), &field_cache);
    let greedy_results = find_constraint_spans_in_sequence(
        &build_a_wildcard_c(QuantifierKind::Greedy),
        &field_cache,
    );

    assert!(!lazy_results.is_empty(), "Lazy should have results");
    assert!(!greedy_results.is_empty(), "Greedy should have results");

    let avg_lazy_length: usize =
        lazy_results.iter().map(|r| r.span.length()).sum::<usize>() / lazy_results.len();
    let greedy_length = greedy_results[0].span.length();

    assert!(
        avg_lazy_length < greedy_length,
        "Average lazy match length ({}) should be less than greedy length ({})",
        avg_lazy_length,
        greedy_length
    );
}
