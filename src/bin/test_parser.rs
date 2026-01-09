use pest::Parser;
use rustie::compiler::pest_parser::{QueryParser, Rule, build_ast};

fn main() {
    println!("=== RustIE Odinson Parser Test Suite ===\n");

    // Test categories
    test_basic_constraints();
    test_negated_constraints();
    test_repetition_quantifiers();
    test_graph_traversals();
    test_named_captures();
    test_assertions();
    test_complex_patterns();
    test_edge_cases();
    test_default_field_queries();

    println!("\n=== Test Suite Complete ===");
}

fn test_basic_constraints() {
    println!("--- Basic Constraints ---");
    let queries = vec![
        "[word=John]",
        "[word=/john|jane/]",
        "[*]",
        "[word=dog & tag=NN]",
        "[word=/j.*/ | tag=NN]",
    ];

    for query in queries {
        test_query(query, "Basic constraint");
    }
}

fn test_negated_constraints() {
    println!("\n--- Negated Constraints ---");
    let queries = vec![
        "[!word=John]",
        "[word!=John]",
        "[word=dog & !tag=VB]",
    ];

    for query in queries {
        test_query(query, "Negated constraint");
    }
}

fn test_repetition_quantifiers() {
    println!("\n--- Repetition Quantifiers ---");
    let queries = vec![
        "[word=the]*",
        "[word=the]+",
        "[word=the]{2,5}",
        "[word=the]{3,}",
        "[word=the]*?",
        "[word=the]+?",
        "[word=the]??",
        "[word=the]{2,5}?",
    ];

    for query in queries {
        test_query(query, "Repetition quantifier");
    }
}

fn test_graph_traversals() {
    println!("\n--- Graph Traversals ---");
    let queries = vec![
        "[word=dog] >nsubj [word=barks]",
        "[word=dog] >> [word=barks]",
        "[word=dog] >nsubj? [word=barks]",
        "[word=dog] >nsubj|dobj> [word=barks]",
        "[word=dog] >nsubj>dobj> [word=cat]",
        "[word=dog] <dobj [word=cat]",
        "[word=dog] << [word=cat]",
    ];

    for query in queries {
        test_query(query, "Graph traversal");
    }
}

fn test_named_captures() {
    println!("\n--- Named Captures ---");
    let queries = vec![
        "(?<subject>[word=John]) >nsubj [word=works]",
        "(?<obj>[word=Microsoft])",
    ];

    for query in queries {
        test_query(query, "Named capture");
    }
}

fn test_assertions() {
    println!("\n--- Assertions ---");
    let queries = vec![
        "<s> [word=The]",
        "[word=end] </s>",
        "[word=John] (?=[word=works])",
        "[word=John] (?![word=works])",
        "(?<=[word=The]) [word=dog]",
        "(?<![word=The]) [word=dog]",
    ];

    for query in queries {
        test_query(query, "Assertion");
    }
}

fn test_complex_patterns() {
    println!("\n--- Complex Patterns ---");
    let queries = vec![
        "[word=the] [word=dog] >nsubj [word=barks]",
        "(?<subject>[word=John]) >nsubj [word=works] <dobj [word=Microsoft]",
        "[word=/j.*/]* [word=works]+",
        "[word=the]* [word=dog]+ [word=barks]",
    ];

    for query in queries {
        test_query(query, "Complex pattern");
    }
}

fn test_edge_cases() {
    println!("\n--- Edge Cases ---");
    let queries = vec![
        "[]",
        "(([word=John]))",
        "[word=John] | [word=Jane] | [word=Bob]",
        "([word=John])",
        "[word=John] [word=Jane] [word=Bob]",
    ];

    for query in queries {
        test_query(query, "Edge case");
    }
}

fn test_default_field_queries() {
    println!("\n--- Default Field Queries ---");
    let queries = vec![
        "John",
        "/john|jane/",
    ];

    for query in queries {
        test_query(query, "Default field query");
    }
}

fn test_query(query: &str, category: &str) {
    match QueryParser::parse(Rule::query, query) {
        Ok(mut pairs) => {
            let ast = build_ast(pairs.next().unwrap());
            println!("✓ {}: '{}'", category, query);
            // Uncomment for detailed AST output:
            // println!("  AST: {:#?}\n", ast);
        },
        Err(e) => {
            println!("✗ {}: '{}'", category, query);
            println!("  Error: {}", e);
        }
    }
}

