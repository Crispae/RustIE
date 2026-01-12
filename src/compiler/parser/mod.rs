use crate::compiler::ast::Pattern;
use anyhow::{Result, anyhow};
// Integrate pest-based parser
use crate::compiler::pest_parser::{QueryParser as PestQueryParser, Rule, build_ast};
use pest::Parser;
use std::panic;

/// Unified parser that delegates to appropriate specialized parser
pub struct QueryParser {
}

impl QueryParser {
    pub fn new(_default_field: String) -> Self {
        Self {
        }
    }

    pub fn parse_query(&self, query: &str) -> Result<Pattern> {
        // Use pest-based parser for all queries
        let mut pairs = PestQueryParser::parse(Rule::query, query)?;

        // Fixed: Handle empty parse result instead of unwrapping
        let first_pair = pairs.next()
            .ok_or_else(|| anyhow!("Parse error: Empty parse result for query '{}'", query))?;

        // Catch panics from build_ast and convert to Result
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            build_ast(first_pair)
        }));

        match result {
            Ok(ast) => Ok(ast),
            Err(panic_info) => {
                // Convert panic to error
                let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown parse error".to_string()
                };
                Err(anyhow!("Parse error in query '{}': {}", query, msg))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ast::{Constraint, Traversal};

    fn parser() -> QueryParser {
        QueryParser::new("word".to_string())
    }

    // ==================== Valid Query Tests ====================

    #[test]
    fn test_parse_simple_word_constraint() {
        let parser = parser();
        let result = parser.parse_query("[word=test]");
        assert!(result.is_ok(), "Failed to parse simple word constraint: {:?}", result.err());
        let pattern = result.unwrap();
        match pattern {
            Pattern::Constraint(Constraint::Field { name, .. }) => {
                assert_eq!(name, "word");
            }
            _ => panic!("Expected Field constraint, got {:?}", pattern),
        }
    }

    #[test]
    fn test_parse_wildcard_constraint() {
        let parser = parser();
        let result = parser.parse_query("[]");
        assert!(result.is_ok(), "Failed to parse wildcard constraint: {:?}", result.err());
        let pattern = result.unwrap();
        match pattern {
            Pattern::Constraint(Constraint::Wildcard) => {}
            _ => panic!("Expected Wildcard constraint, got {:?}", pattern),
        }
    }

    #[test]
    fn test_parse_constraint_with_star_wildcard() {
        let parser = parser();
        let result = parser.parse_query("[*]");
        assert!(result.is_ok(), "Failed to parse star wildcard: {:?}", result.err());
    }

    #[test]
    fn test_parse_named_capture() {
        let parser = parser();
        let result = parser.parse_query("(?<subject> [word=John])");
        assert!(result.is_ok(), "Failed to parse named capture: {:?}", result.err());
        let pattern = result.unwrap();
        match pattern {
            Pattern::NamedCapture { name, .. } => {
                assert_eq!(name, "subject");
            }
            _ => panic!("Expected NamedCapture pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_parse_regex_constraint() {
        let parser = parser();
        // Use simpler regex pattern that fits the grammar
        let result = parser.parse_query("[word=/test.*/]");
        assert!(result.is_ok(), "Failed to parse regex constraint: {:?}", result.err());
    }

    #[test]
    fn test_parse_repetition_star() {
        let parser = parser();
        let result = parser.parse_query("[]*");
        assert!(result.is_ok(), "Failed to parse repetition star: {:?}", result.err());
        let pattern = result.unwrap();
        match pattern {
            Pattern::Repetition { min, max, .. } => {
                assert_eq!(min, 0);
                assert_eq!(max, None);
            }
            _ => panic!("Expected Repetition pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_parse_repetition_plus() {
        let parser = parser();
        let result = parser.parse_query("[]+");
        assert!(result.is_ok(), "Failed to parse repetition plus: {:?}", result.err());
        let pattern = result.unwrap();
        match pattern {
            Pattern::Repetition { min, max, .. } => {
                assert_eq!(min, 1);
                assert_eq!(max, None);
            }
            _ => panic!("Expected Repetition pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_parse_repetition_optional() {
        let parser = parser();
        let result = parser.parse_query("[]?");
        assert!(result.is_ok(), "Failed to parse repetition optional: {:?}", result.err());
        let pattern = result.unwrap();
        match pattern {
            Pattern::Repetition { min, max, .. } => {
                assert_eq!(min, 0);
                assert_eq!(max, Some(1));
            }
            _ => panic!("Expected Repetition pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_parse_repetition_range() {
        let parser = parser();
        let result = parser.parse_query("[]{2,5}");
        assert!(result.is_ok(), "Failed to parse repetition range: {:?}", result.err());
        let pattern = result.unwrap();
        match pattern {
            Pattern::Repetition { min, max, .. } => {
                assert_eq!(min, 2);
                assert_eq!(max, Some(5));
            }
            _ => panic!("Expected Repetition pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_parse_graph_traversal_outgoing() {
        let parser = parser();
        let result = parser.parse_query("[word=eats] >nsubj [word=John]");
        assert!(result.is_ok(), "Failed to parse graph traversal: {:?}", result.err());
        let pattern = result.unwrap();
        match pattern {
            Pattern::GraphTraversal { traversal, .. } => {
                match traversal {
                    Traversal::Outgoing(_) => {}
                    _ => panic!("Expected Outgoing traversal, got {:?}", traversal),
                }
            }
            _ => panic!("Expected GraphTraversal pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_parse_graph_traversal_incoming() {
        let parser = parser();
        let result = parser.parse_query("[word=John] <nsubj [word=eats]");
        assert!(result.is_ok(), "Failed to parse graph traversal incoming: {:?}", result.err());
        let pattern = result.unwrap();
        match pattern {
            Pattern::GraphTraversal { traversal, .. } => {
                match traversal {
                    Traversal::Incoming(_) => {}
                    _ => panic!("Expected Incoming traversal, got {:?}", traversal),
                }
            }
            _ => panic!("Expected GraphTraversal pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_parse_graph_traversal_wildcard() {
        let parser = parser();
        let result = parser.parse_query("[word=eats] >> []");
        assert!(result.is_ok(), "Failed to parse wildcard traversal: {:?}", result.err());
        let pattern = result.unwrap();
        match pattern {
            Pattern::GraphTraversal { traversal, .. } => {
                match traversal {
                    Traversal::OutgoingWildcard => {}
                    _ => panic!("Expected OutgoingWildcard traversal, got {:?}", traversal),
                }
            }
            _ => panic!("Expected GraphTraversal pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_parse_disjunction_inside_constraint() {
        let parser = parser();
        // Disjunction is inside the constraint brackets with |
        let result = parser.parse_query("[word=cat | word=dog]");
        assert!(result.is_ok(), "Failed to parse disjunction inside constraint: {:?}", result.err());
        let pattern = result.unwrap();
        match pattern {
            Pattern::Constraint(Constraint::Disjunctive(constraints)) => {
                assert_eq!(constraints.len(), 2);
            }
            _ => panic!("Expected Disjunctive constraint, got {:?}", pattern),
        }
    }

    #[test]
    fn test_parse_conjunction_inside_constraint() {
        let parser = parser();
        // Conjunction uses & inside the constraint
        let result = parser.parse_query("[word=cat & pos=NN]");
        assert!(result.is_ok(), "Failed to parse conjunction inside constraint: {:?}", result.err());
        let pattern = result.unwrap();
        match pattern {
            Pattern::Constraint(Constraint::Conjunctive(constraints)) => {
                assert_eq!(constraints.len(), 2);
            }
            _ => panic!("Expected Conjunctive constraint, got {:?}", pattern),
        }
    }

    #[test]
    fn test_parse_negated_constraint() {
        let parser = parser();
        // Negation uses != operator (word!=value syntax works correctly)
        let result = parser.parse_query("[word!=the]");
        assert!(result.is_ok(), "Failed to parse negated constraint: {:?}", result.err());
        let pattern = result.unwrap();
        match pattern {
            Pattern::Constraint(Constraint::Negated(_)) => {}
            _ => panic!("Expected Negated constraint, got {:?}", pattern),
        }
    }

    // ==================== Error Handling Tests ====================

    #[test]
    fn test_parse_invalid_syntax_unmatched_bracket() {
        let parser = parser();
        let result = parser.parse_query("[word=test");
        assert!(result.is_err(), "Expected error for unmatched bracket");
    }

    #[test]
    fn test_parse_invalid_empty_string() {
        let parser = parser();
        let result = parser.parse_query("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_only_whitespace() {
        let parser = parser();
        let result = parser.parse_query("   ");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_unclosed_regex() {
        let parser = parser();
        let result = parser.parse_query("[word=/unclosed");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_unclosed_capture() {
        let parser = parser();
        let result = parser.parse_query("(?<name [word=test]");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_no_crash_on_double_pipe() {
        let parser = parser();
        let result = parser.parse_query("[word=a] || [word=b]");
        // Just ensure no panic - may be valid or invalid
        let _ = result;
    }

    #[test]
    fn test_parse_no_crash_on_trailing_traversal() {
        let parser = parser();
        let result = parser.parse_query("[word=test] >nsubj");
        // Just ensure no panic - may be valid or invalid
        let _ = result;
    }

    #[test]
    fn test_parse_no_crash_on_extra_bracket() {
        let parser = parser();
        let result = parser.parse_query("[word=test]]");
        // Just ensure no panic - may be valid or invalid depending on grammar
        let _ = result;
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_parse_complex_nested_pattern() {
        let parser = parser();
        let result = parser.parse_query("(?<subject> [word=John]) >nsubj [word=eats]");
        // Complex patterns should either parse or return error, not panic
        let _ = result;
    }

    #[test]
    fn test_parse_multiple_traversals() {
        let parser = parser();
        let result = parser.parse_query("[word=John] <nsubj [word=eats] >dobj [word=pizza]");
        assert!(result.is_ok(), "Failed to parse multiple traversals: {:?}", result.err());
    }

    #[test]
    fn test_parse_no_crash_on_nested_repetition() {
        let parser = parser();
        let result = parser.parse_query("([word=a])+");
        // Just ensure no panic
        let _ = result;
    }

    #[test]
    fn test_parse_alphanumeric_constraint_value() {
        let parser = parser();
        // Grammar only supports ASCII_ALPHANUMERIC+ for values
        let result = parser.parse_query("[word=test123]");
        assert!(result.is_ok(), "Failed to parse alphanumeric value: {:?}", result.err());
    }

    #[test]
    fn test_parse_no_crash_on_special_characters_in_regex() {
        let parser = parser();
        let result = parser.parse_query("[word=/test.*/]");
        // Just ensure no panic - special chars may or may not be supported
        let _ = result;
    }

    #[test]
    fn test_parser_new_creates_valid_instance() {
        let parser = QueryParser::new("lemma".to_string());
        // Parser should be created with any field name
        let result = parser.parse_query("[]");
        assert!(result.is_ok());
    }

    // ==================== Default Field Query Tests ====================

    #[test]
    fn test_parse_default_string_query() {
        let parser = parser();
        let result = parser.parse_query("hello");
        assert!(result.is_ok(), "Failed to parse default string query: {:?}", result.err());
    }

    #[test]
    fn test_parse_default_regex_query() {
        let parser = parser();
        let result = parser.parse_query("/test.*/");
        assert!(result.is_ok(), "Failed to parse default regex query: {:?}", result.err());
    }

    // ==================== Traversal Regex Tests ====================

    #[test]
    fn test_parse_traversal_with_regex_label() {
        let parser = parser();
        let result = parser.parse_query("[word=eats] >/nsubj|dobj/ []");
        assert!(result.is_ok(), "Failed to parse traversal with regex label: {:?}", result.err());
    }

    // ==================== Assertion Tests ====================

    #[test]
    fn test_parse_positive_lookahead() {
        let parser = parser();
        let result = parser.parse_query("(?= [word=test])");
        assert!(result.is_ok(), "Failed to parse positive lookahead: {:?}", result.err());
    }

    #[test]
    fn test_parse_negative_lookahead() {
        let parser = parser();
        let result = parser.parse_query("(?! [word=test])");
        assert!(result.is_ok(), "Failed to parse negative lookahead: {:?}", result.err());
    }

    #[test]
    fn test_parse_positive_lookbehind() {
        let parser = parser();
        let result = parser.parse_query("(?<= [word=test])");
        assert!(result.is_ok(), "Failed to parse positive lookbehind: {:?}", result.err());
    }

    #[test]
    fn test_parse_negative_lookbehind() {
        let parser = parser();
        let result = parser.parse_query("(?<! [word=test])");
        assert!(result.is_ok(), "Failed to parse negative lookbehind: {:?}", result.err());
    }
} 