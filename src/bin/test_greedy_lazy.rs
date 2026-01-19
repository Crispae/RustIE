/// Test script to verify greedy vs lazy quantifier behavior
/// 
/// Expected behavior:
/// - Lazy `[]*?`: Should produce multiple shorter matches
/// - Greedy `[]*`: Should produce one longer match
/// 
/// Example: Pattern `a []*? c` on text "a b c a b c"
/// - Lazy: Should match "a b c" twice (positions 0-3 and 3-6)
/// - Greedy: Should match "a b c a b c" once (position 0-6)

use rustie::compiler::ast::{Pattern, Constraint, Matcher, QuantifierKind};
use rustie::tantivy_integration::concat_query::find_constraint_spans_in_sequence;
use std::collections::HashMap;

fn create_field_cache(tokens: Vec<&str>) -> HashMap<String, Vec<String>> {
    let mut cache = HashMap::new();
    cache.insert("word".to_string(), tokens.iter().map(|s| s.to_string()).collect());
    cache
}

fn main() {
    println!("=== Testing Greedy vs Lazy Quantifier Behavior ===\n");
    
    // Test text: "a b c a b c"
    let field_cache = create_field_cache(vec!["a", "b", "c", "a", "b", "c"]);
    
    // Test 1: Lazy quantifier `a []*? c`
    println!("Test 1: Lazy quantifier `a []*? c`");
    let lazy_pattern = Pattern::Concatenated(vec![
        Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::String("a".to_string()),
        }),
        Pattern::Repetition {
            pattern: Box::new(Pattern::Constraint(Constraint::Wildcard)),
            min: 0,
            max: None,
            kind: QuantifierKind::Lazy,
        },
        Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::String("c".to_string()),
        }),
    ]);
    
    let lazy_results = find_constraint_spans_in_sequence(&lazy_pattern, &field_cache);
    println!("  Found {} matches:", lazy_results.len());
    for (i, result) in lazy_results.iter().enumerate() {
        println!("    Match {}: span [{}, {}), length {}", 
            i + 1, result.span.start, result.span.end, result.span.length());
    }
    
    // Expected: Should produce 2 matches (one for each "a ... c")
    // For lazy, at position 0, should prefer shortest: [0, 3) not [0, 6)
    if lazy_results.len() >= 2 {
        // Check if we have the correct lazy behavior: shortest match at each position
        let has_short_match = lazy_results.iter().any(|r| r.span.start == 0 && r.span.length() == 3);
        if has_short_match {
            println!("  ✅ PASS: Lazy quantifier produces multiple matches with shortest preferred");
        } else {
            println!("  ⚠️  WARNING: Lazy quantifier produces multiple matches but may not prefer shortest");
        }
    } else {
        println!("  ⚠️  WARNING: Expected at least 2 matches for lazy quantifier");
    }
    
    println!();
    
    // Test 2: Greedy quantifier `a []* c`
    println!("Test 2: Greedy quantifier `a []* c`");
    let greedy_pattern = Pattern::Concatenated(vec![
        Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::String("a".to_string()),
        }),
        Pattern::Repetition {
            pattern: Box::new(Pattern::Constraint(Constraint::Wildcard)),
            min: 0,
            max: None,
            kind: QuantifierKind::Greedy,
        },
        Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::String("c".to_string()),
        }),
    ]);
    
    let greedy_results = find_constraint_spans_in_sequence(&greedy_pattern, &field_cache);
    println!("  Found {} matches:", greedy_results.len());
    for (i, result) in greedy_results.iter().enumerate() {
        println!("    Match {}: span [{}, {}), length {}", 
            i + 1, result.span.start, result.span.end, result.span.length());
    }
    
    // Expected: Should produce 1 match (the longest one)
    if greedy_results.len() == 1 && greedy_results[0].span.length() == 6 {
        println!("  ✅ PASS: Greedy quantifier produces one long match");
    } else if greedy_results.len() == 1 {
        println!("  ⚠️  WARNING: Greedy quantifier produces one match, but length is {}", 
            greedy_results[0].span.length());
    } else {
        println!("  ⚠️  WARNING: Expected 1 match for greedy quantifier, got {}", 
            greedy_results.len());
    }
    
    println!();
    
    // Test 3: Compare lengths
    println!("Test 3: Comparing match lengths");
    if !lazy_results.is_empty() && !greedy_results.is_empty() {
        let avg_lazy_length: usize = lazy_results.iter()
            .map(|r| r.span.length())
            .sum::<usize>() / lazy_results.len();
        let greedy_length = greedy_results[0].span.length();
        
        println!("  Average lazy match length: {}", avg_lazy_length);
        println!("  Greedy match length: {}", greedy_length);
        
        if avg_lazy_length < greedy_length {
            println!("  ✅ PASS: Lazy matches are shorter than greedy match");
        } else {
            println!("  ⚠️  WARNING: Lazy matches should be shorter");
        }
    }
    
    println!();
    println!("=== Test Complete ===");
}
