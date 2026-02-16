use serde::{Deserialize, Serialize};

/// Matcher for string or regex patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Matcher {
    String(String),
    #[serde(skip)]
    Regex { pattern: String, regex: std::sync::Arc<regex::Regex> },
}

impl Matcher {
    /// Create a string matcher
    pub fn string(s: String) -> Self {
        Matcher::String(s)
    }

    /// Create a regex matcher with validation (returns Result)
    /// Use this for user-provided patterns to handle invalid regex gracefully
    pub fn try_regex(pattern: String) -> Result<Self, regex::Error> {
        let regex = regex::Regex::new(&pattern)?;
        Ok(Matcher::Regex { pattern, regex: std::sync::Arc::new(regex) })
    }

    /// Create a regex matcher (pre-compiles the regex)
    /// Panics if the regex is invalid - use try_regex for user input
    pub fn regex(pattern: String) -> Self {
        Self::try_regex(pattern.clone())
            .unwrap_or_else(|e| panic!("Invalid regex pattern '{}': {}", pattern, e))
    }

    /// Check if a token matches this matcher
    pub fn matches(&self, token: &str) -> bool {
        match self {
            Matcher::String(s) => token == s,
            Matcher::Regex { regex, .. } => regex.is_match(token),
        }
    }

    /// Get the pattern string (for both String and Regex variants)
    pub fn pattern_str(&self) -> &str {
        match self {
            Matcher::String(s) => s,
            Matcher::Regex { pattern, .. } => pattern,
        }
    }
}

/// Constraints for token matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    Wildcard,
    Field { name: String, matcher: Matcher },
    Fuzzy { name: String, matcher: String },
    Negated(Box<Constraint>),
    Conjunctive(Vec<Constraint>),
    Disjunctive(Vec<Constraint>),
}

impl Constraint {
    /// Check if a token matches this constraint
    pub fn matches(&self, field_name: &str, token: &str) -> bool {
        match self {
            Constraint::Wildcard => true,
            Constraint::Field { name, matcher } => {
                if name == field_name {
                    matcher.matches(token)
                } else {
                    false
                }
            }
            Constraint::Fuzzy { name, matcher } => {
                if name == field_name {
                    // Simple fuzzy matching (can be improved with edit distance)
                    token.to_lowercase().contains(&matcher.to_lowercase())
                } else {
                    false
                }
            }
            Constraint::Negated(inner) => !inner.matches(field_name, token),
            Constraint::Conjunctive(constraints) => {
                constraints.iter().all(|c| c.matches(field_name, token))
            }
            Constraint::Disjunctive(constraints) => {
                constraints.iter().any(|c| c.matches(field_name, token))
            }
        }
    }

    /// Returns true if this constraint references the given field name.
    ///
    /// Used to determine whether token-level position extraction is needed
    /// for a given field. For example, a constraint on "entity" does not
    /// reference "word", so we can skip loading word tokens entirely.
    pub fn references_field(&self, field_name: &str) -> bool {
        match self {
            Constraint::Wildcard => true,
            Constraint::Field { name, .. } => name == field_name,
            Constraint::Fuzzy { name, .. } => name == field_name,
            Constraint::Negated(inner) => inner.references_field(field_name),
            Constraint::Conjunctive(constraints) => {
                constraints.iter().any(|c| c.references_field(field_name))
            }
            Constraint::Disjunctive(constraints) => {
                constraints.iter().any(|c| c.references_field(field_name))
            }
        }
    }
}

/// Assertions for position-based matching (lookahead and lookbehind only)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Assertion {
    PositiveLookahead(Box<Pattern>),
    NegativeLookahead(Box<Pattern>),
    PositiveLookbehind(Box<Pattern>),
    NegativeLookbehind(Box<Pattern>),
}

impl Assertion {
    /// Returns true if this assertion's inner pattern references the given field.
    pub fn references_field(&self, field_name: &str) -> bool {
        match self {
            Assertion::PositiveLookahead(pattern) => pattern.references_field(field_name),
            Assertion::NegativeLookahead(pattern) => pattern.references_field(field_name),
            Assertion::PositiveLookbehind(pattern) => pattern.references_field(field_name),
            Assertion::NegativeLookbehind(pattern) => pattern.references_field(field_name),
        }
    }
}

/// Graph traversal patterns for dependency parsing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Traversal {
    NoTraversal,
    OutgoingWildcard,
    IncomingWildcard,
    Incoming(Matcher),
    Outgoing(Matcher),
    Concatenated(Vec<Traversal>),
    Disjunctive(Vec<Traversal>),
    Optional(Box<Traversal>),
    KleeneStar(Box<Traversal>),
}

/// Flat pattern step for automaton-based traversal
#[derive(Debug, Clone)]
pub enum FlatPatternStep {
    Constraint(Pattern),
    Traversal(Traversal),
}

/// Quantifier kind: greedy or lazy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantifierKind {
    Greedy,
    Lazy,
}

/// Main pattern types for Odinson queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Pattern {
    Assertion(Assertion),
    Constraint(Constraint),
    Disjunctive(Vec<Pattern>),
    Concatenated(Vec<Pattern>),
    NamedCapture { name: String, pattern: Box<Pattern> },
    Mention { arg_name: Option<String>, label: String },
    GraphTraversal { src: Box<Pattern>, traversal: Traversal, dst: Box<Pattern> },
    Repetition { pattern: Box<Pattern>, min: usize, max: Option<usize>, kind: QuantifierKind },
}

impl Pattern {
    /// Extract all matching positions for a pattern from a list of tokens
    pub fn extract_matching_positions(&self, field_name: &str, tokens: &[String]) -> Vec<usize> {
        match self {
            Pattern::Assertion(_) => {
                // Assertions don't match individual tokens
                vec![]
            }
            Pattern::Constraint(constraint) => {
                if !constraint.references_field(field_name) {
                    return Vec::new();
                }
                tokens.iter().enumerate()
                    .filter_map(|(i, token)| {
                        if constraint.matches(field_name, token) {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect()
            }
            Pattern::Disjunctive(patterns) => {
                let mut positions = Vec::new();
                for pattern in patterns {
                    positions.extend(pattern.extract_matching_positions(field_name, tokens));
                }
                positions
            }
            Pattern::Concatenated(_) => {
                // Concatenated patterns require sequence matching, not individual token matching
                // For now, we'll treat them as individual constraints
                vec![]
            }
            Pattern::NamedCapture { pattern, .. } => {
                pattern.extract_matching_positions(field_name, tokens)
            }
            Pattern::Mention { .. } => {
                // Mentions are resolved differently
                vec![]
            }
            Pattern::GraphTraversal { src, .. } => {
                // For graph traversal, we only need source positions
                src.extract_matching_positions(field_name, tokens)
            }
            Pattern::Repetition { pattern, .. } => {
                // For now, treat repetition as the base pattern
                pattern.extract_matching_positions(field_name, tokens)
            }
        }
    }
    
    /// Extract destination positions for graph traversal
    pub fn extract_destination_positions(&self, field_name: &str, tokens: &[String]) -> Vec<usize> {
        match self {
            Pattern::GraphTraversal { dst, .. } => {
                dst.extract_matching_positions(field_name, tokens)
            }
            _ => {
                // For non-graph patterns, treat as regular pattern
                self.extract_matching_positions(field_name, tokens)
            }
        }
    }

    /// Returns true if this pattern references the given field name in any
    /// constraint that would require token-level matching against that field.
    pub fn references_field(&self, field_name: &str) -> bool {
        match self {
            Pattern::Assertion(assertion) => assertion.references_field(field_name),
            Pattern::Constraint(constraint) => constraint.references_field(field_name),
            Pattern::Disjunctive(patterns) => {
                patterns.iter().any(|p| p.references_field(field_name))
            }
            Pattern::Concatenated(patterns) => {
                patterns.iter().any(|p| p.references_field(field_name))
            }
            Pattern::NamedCapture { pattern, .. } => pattern.references_field(field_name),
            Pattern::Mention { .. } => false,
            Pattern::GraphTraversal { src, dst, .. } => {
                src.references_field(field_name) || dst.references_field(field_name)
            }
            Pattern::Repetition { pattern, .. } => pattern.references_field(field_name),
        }
    }
}

/// Main AST structure
pub struct Ast;

impl Ast {
    /// Create a wildcard constraint
    pub fn wildcard() -> Constraint {
        Constraint::Wildcard
    }

    /// Create a field constraint
    pub fn field_constraint(name: String, matcher: Matcher) -> Constraint {
        Constraint::Field { name, matcher }
    }

    /// Create a string matcher
    pub fn string_matcher(s: String) -> Matcher {
        Matcher::String(s)
    }

    /// Create a regex matcher
    pub fn regex_matcher(pattern: String) -> Matcher {
        Matcher::regex(pattern)
    }

    /// Create a constraint pattern
    pub fn constraint_pattern(constraint: Constraint) -> Pattern {
        Pattern::Constraint(constraint)
    }

    /// Create a concatenated pattern
    pub fn concatenated_pattern(patterns: Vec<Pattern>) -> Pattern {
        Pattern::Concatenated(patterns)
    }

    /// Create a disjunctive pattern
    pub fn disjunctive_pattern(patterns: Vec<Pattern>) -> Pattern {
        Pattern::Disjunctive(patterns)
    }

    /// Create a named capture pattern
    pub fn named_capture_pattern(name: String, pattern: Pattern) -> Pattern {
        Pattern::NamedCapture {
            name,
            pattern: Box::new(pattern),
        }
    }

    /// Create a repetition pattern
    pub fn repetition_pattern(pattern: Pattern, min: usize, max: Option<usize>, kind: QuantifierKind) -> Pattern {
        Pattern::Repetition {
            pattern: Box::new(pattern),
            min,
            max,
            kind,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Matcher Tests ====================

    #[test]
    fn test_matcher_string_exact_match() {
        let matcher = Matcher::string("hello".to_string());
        assert!(matcher.matches("hello"));
        assert!(!matcher.matches("Hello"));  // Case sensitive
        assert!(!matcher.matches("hello "));  // Exact match
        assert!(!matcher.matches("world"));
    }

    #[test]
    fn test_matcher_string_empty() {
        let matcher = Matcher::string("".to_string());
        assert!(matcher.matches(""));
        assert!(!matcher.matches("a"));
    }

    #[test]
    fn test_matcher_regex_simple() {
        let matcher = Matcher::regex("^hello$".to_string());
        assert!(matcher.matches("hello"));
        assert!(!matcher.matches("hello world"));
        assert!(!matcher.matches("say hello"));
    }

    #[test]
    fn test_matcher_regex_pattern_matching() {
        let matcher = Matcher::regex("^[A-Z][a-z]+$".to_string());
        assert!(matcher.matches("John"));
        assert!(matcher.matches("Alice"));
        assert!(!matcher.matches("john"));  // Doesn't start with uppercase
        assert!(!matcher.matches("JOHN"));  // All uppercase
    }

    #[test]
    fn test_matcher_regex_partial_match() {
        let matcher = Matcher::regex("cat".to_string());
        assert!(matcher.matches("cat"));
        assert!(matcher.matches("category"));
        assert!(matcher.matches("scattered"));
    }

    #[test]
    fn test_matcher_try_regex_valid() {
        let result = Matcher::try_regex("^test$".to_string());
        assert!(result.is_ok());
        let matcher = result.unwrap();
        assert!(matcher.matches("test"));
    }

    #[test]
    fn test_matcher_try_regex_invalid() {
        let result = Matcher::try_regex("[[invalid".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_matcher_pattern_str_string() {
        let matcher = Matcher::string("hello".to_string());
        assert_eq!(matcher.pattern_str(), "hello");
    }

    #[test]
    fn test_matcher_pattern_str_regex() {
        let matcher = Matcher::regex("^[a-z]+$".to_string());
        assert_eq!(matcher.pattern_str(), "^[a-z]+$");
    }

    #[test]
    #[should_panic(expected = "Invalid regex pattern")]
    fn test_matcher_regex_invalid_panics() {
        let _ = Matcher::regex("[[invalid".to_string());
    }

    // ==================== Constraint Tests ====================

    #[test]
    fn test_constraint_wildcard() {
        let constraint = Constraint::Wildcard;
        assert!(constraint.matches("word", "anything"));
        assert!(constraint.matches("word", ""));
        assert!(constraint.matches("other_field", "test"));
    }

    #[test]
    fn test_constraint_field_match() {
        let constraint = Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::string("hello".to_string()),
        };
        assert!(constraint.matches("word", "hello"));
        assert!(!constraint.matches("word", "world"));
        assert!(!constraint.matches("lemma", "hello"));  // Wrong field
    }

    #[test]
    fn test_constraint_field_regex() {
        let constraint = Constraint::Field {
            name: "pos".to_string(),
            matcher: Matcher::regex("^NN.*".to_string()),
        };
        assert!(constraint.matches("pos", "NN"));
        assert!(constraint.matches("pos", "NNS"));
        assert!(constraint.matches("pos", "NNP"));
        assert!(!constraint.matches("pos", "VB"));
    }

    #[test]
    fn test_constraint_fuzzy() {
        let constraint = Constraint::Fuzzy {
            name: "word".to_string(),
            matcher: "cat".to_string(),
        };
        assert!(constraint.matches("word", "cat"));
        assert!(constraint.matches("word", "category"));
        assert!(constraint.matches("word", "CAT"));  // Case insensitive
        assert!(constraint.matches("word", "scattered"));
        assert!(!constraint.matches("word", "dog"));
        assert!(!constraint.matches("lemma", "cat"));  // Wrong field
    }

    #[test]
    fn test_constraint_negated() {
        let inner = Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::string("hello".to_string()),
        };
        let constraint = Constraint::Negated(Box::new(inner));
        assert!(!constraint.matches("word", "hello"));  // Negated
        assert!(constraint.matches("word", "world"));
        assert!(constraint.matches("lemma", "hello"));  // Wrong field, so inner is false, negated is true
    }

    #[test]
    fn test_constraint_conjunctive_all_match() {
        let constraint = Constraint::Conjunctive(vec![
            Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::regex("^[A-Z]".to_string()),  // Starts with uppercase
            },
            Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::regex("n$".to_string()),  // Ends with 'n'
            },
        ]);
        assert!(constraint.matches("word", "John"));
        assert!(constraint.matches("word", "Christian"));
        assert!(!constraint.matches("word", "john"));  // Doesn't start with uppercase
        assert!(!constraint.matches("word", "Jack"));  // Doesn't end with 'n'
    }

    #[test]
    fn test_constraint_conjunctive_empty() {
        let constraint = Constraint::Conjunctive(vec![]);
        assert!(constraint.matches("word", "anything"));  // Empty conjunction is vacuously true
    }

    #[test]
    fn test_constraint_disjunctive_any_match() {
        let constraint = Constraint::Disjunctive(vec![
            Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::string("cat".to_string()),
            },
            Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::string("dog".to_string()),
            },
        ]);
        assert!(constraint.matches("word", "cat"));
        assert!(constraint.matches("word", "dog"));
        assert!(!constraint.matches("word", "bird"));
    }

    #[test]
    fn test_constraint_disjunctive_empty() {
        let constraint = Constraint::Disjunctive(vec![]);
        assert!(!constraint.matches("word", "anything"));  // Empty disjunction is false
    }

    // ==================== Pattern Tests ====================

    #[test]
    fn test_pattern_extract_matching_positions_constraint() {
        let pattern = Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::string("the".to_string()),
        });
        let tokens = vec!["the".to_string(), "cat".to_string(), "the".to_string(), "dog".to_string()];
        let positions = pattern.extract_matching_positions("word", &tokens);
        assert_eq!(positions, vec![0, 2]);
    }

    #[test]
    fn test_pattern_extract_matching_positions_wildcard() {
        let pattern = Pattern::Constraint(Constraint::Wildcard);
        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let positions = pattern.extract_matching_positions("word", &tokens);
        assert_eq!(positions, vec![0, 1, 2]);
    }

    #[test]
    fn test_pattern_extract_matching_positions_no_match() {
        let pattern = Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::string("nonexistent".to_string()),
        });
        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let positions = pattern.extract_matching_positions("word", &tokens);
        assert!(positions.is_empty());
    }

    #[test]
    fn test_pattern_extract_matching_positions_empty_tokens() {
        let pattern = Pattern::Constraint(Constraint::Wildcard);
        let tokens: Vec<String> = vec![];
        let positions = pattern.extract_matching_positions("word", &tokens);
        assert!(positions.is_empty());
    }

    #[test]
    fn test_pattern_extract_matching_positions_disjunctive() {
        let pattern = Pattern::Disjunctive(vec![
            Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::string("cat".to_string()),
            }),
            Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::string("dog".to_string()),
            }),
        ]);
        let tokens = vec!["cat".to_string(), "and".to_string(), "dog".to_string()];
        let positions = pattern.extract_matching_positions("word", &tokens);
        assert!(positions.contains(&0));
        assert!(positions.contains(&2));
    }

    #[test]
    fn test_pattern_extract_matching_positions_named_capture() {
        let inner = Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::string("test".to_string()),
        });
        let pattern = Pattern::NamedCapture {
            name: "capture1".to_string(),
            pattern: Box::new(inner),
        };
        let tokens = vec!["test".to_string(), "data".to_string(), "test".to_string()];
        let positions = pattern.extract_matching_positions("word", &tokens);
        assert_eq!(positions, vec![0, 2]);
    }

    #[test]
    fn test_pattern_extract_matching_positions_assertion() {
        let pattern = Pattern::Assertion(Assertion::PositiveLookahead(Box::new(
            Pattern::Constraint(Constraint::Wildcard)
        )));
        let tokens = vec!["a".to_string(), "b".to_string()];
        let positions = pattern.extract_matching_positions("word", &tokens);
        assert!(positions.is_empty());  // Assertions don't match individual tokens
    }

    #[test]
    fn test_pattern_extract_matching_positions_concatenated() {
        let pattern = Pattern::Concatenated(vec![
            Pattern::Constraint(Constraint::Wildcard),
            Pattern::Constraint(Constraint::Wildcard),
        ]);
        let tokens = vec!["a".to_string(), "b".to_string()];
        let positions = pattern.extract_matching_positions("word", &tokens);
        assert!(positions.is_empty());  // Concatenated requires sequence matching
    }

    #[test]
    fn test_pattern_extract_matching_positions_mention() {
        let pattern = Pattern::Mention {
            arg_name: Some("arg".to_string()),
            label: "Person".to_string(),
        };
        let tokens = vec!["John".to_string(), "Smith".to_string()];
        let positions = pattern.extract_matching_positions("word", &tokens);
        assert!(positions.is_empty());  // Mentions are resolved differently
    }

    #[test]
    fn test_pattern_extract_matching_positions_repetition() {
        let inner = Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::string("x".to_string()),
        });
        let pattern = Pattern::Repetition {
            pattern: Box::new(inner),
            min: 1,
            max: Some(3),
            kind: QuantifierKind::Greedy,
        };
        let tokens = vec!["x".to_string(), "y".to_string(), "x".to_string()];
        let positions = pattern.extract_matching_positions("word", &tokens);
        assert_eq!(positions, vec![0, 2]);
    }

    #[test]
    fn test_pattern_extract_destination_positions_graph_traversal() {
        let src = Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::string("src".to_string()),
        });
        let dst = Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::string("dst".to_string()),
        });
        let pattern = Pattern::GraphTraversal {
            src: Box::new(src),
            traversal: Traversal::OutgoingWildcard,
            dst: Box::new(dst),
        };
        let tokens = vec!["src".to_string(), "dst".to_string(), "other".to_string()];
        let positions = pattern.extract_destination_positions("word", &tokens);
        assert_eq!(positions, vec![1]);
    }

    #[test]
    fn test_pattern_extract_destination_positions_non_graph() {
        let pattern = Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::string("test".to_string()),
        });
        let tokens = vec!["test".to_string(), "data".to_string()];
        let positions = pattern.extract_destination_positions("word", &tokens);
        assert_eq!(positions, vec![0]);  // Falls back to extract_matching_positions
    }

    // ==================== Ast Helper Function Tests ====================

    #[test]
    fn test_ast_wildcard() {
        let constraint = Ast::wildcard();
        assert!(matches!(constraint, Constraint::Wildcard));
    }

    #[test]
    fn test_ast_field_constraint() {
        let matcher = Matcher::string("test".to_string());
        let constraint = Ast::field_constraint("word".to_string(), matcher);
        match constraint {
            Constraint::Field { name, .. } => assert_eq!(name, "word"),
            _ => panic!("Expected Field constraint"),
        }
    }

    #[test]
    fn test_ast_string_matcher() {
        let matcher = Ast::string_matcher("hello".to_string());
        assert!(matcher.matches("hello"));
    }

    #[test]
    fn test_ast_regex_matcher() {
        let matcher = Ast::regex_matcher("^test$".to_string());
        assert!(matcher.matches("test"));
        assert!(!matcher.matches("testing"));
    }

    #[test]
    fn test_ast_constraint_pattern() {
        let constraint = Constraint::Wildcard;
        let pattern = Ast::constraint_pattern(constraint);
        assert!(matches!(pattern, Pattern::Constraint(Constraint::Wildcard)));
    }

    #[test]
    fn test_ast_concatenated_pattern() {
        let patterns = vec![
            Pattern::Constraint(Constraint::Wildcard),
            Pattern::Constraint(Constraint::Wildcard),
        ];
        let pattern = Ast::concatenated_pattern(patterns);
        match pattern {
            Pattern::Concatenated(p) => assert_eq!(p.len(), 2),
            _ => panic!("Expected Concatenated pattern"),
        }
    }

    #[test]
    fn test_ast_disjunctive_pattern() {
        let patterns = vec![
            Pattern::Constraint(Constraint::Wildcard),
            Pattern::Constraint(Constraint::Wildcard),
        ];
        let pattern = Ast::disjunctive_pattern(patterns);
        match pattern {
            Pattern::Disjunctive(p) => assert_eq!(p.len(), 2),
            _ => panic!("Expected Disjunctive pattern"),
        }
    }

    #[test]
    fn test_ast_named_capture_pattern() {
        let inner = Pattern::Constraint(Constraint::Wildcard);
        let pattern = Ast::named_capture_pattern("capture".to_string(), inner);
        match pattern {
            Pattern::NamedCapture { name, .. } => assert_eq!(name, "capture"),
            _ => panic!("Expected NamedCapture pattern"),
        }
    }

    #[test]
    fn test_ast_repetition_pattern() {
        let inner = Pattern::Constraint(Constraint::Wildcard);
        let pattern = Ast::repetition_pattern(inner, 1, Some(5), QuantifierKind::Greedy);
        match pattern {
            Pattern::Repetition { min, max, .. } => {
                assert_eq!(min, 1);
                assert_eq!(max, Some(5));
            }
            _ => panic!("Expected Repetition pattern"),
        }
    }

    #[test]
    fn test_ast_repetition_pattern_unbounded() {
        let inner = Pattern::Constraint(Constraint::Wildcard);
        let pattern = Ast::repetition_pattern(inner, 0, None, QuantifierKind::Greedy);
        match pattern {
            Pattern::Repetition { min, max, .. } => {
                assert_eq!(min, 0);
                assert_eq!(max, None);
            }
            _ => panic!("Expected Repetition pattern"),
        }
    }

    // ==================== Clone and Debug Tests ====================

    #[test]
    fn test_matcher_clone() {
        let matcher = Matcher::regex("^test$".to_string());
        let cloned = matcher.clone();
        assert!(cloned.matches("test"));
    }

    #[test]
    fn test_constraint_clone() {
        let constraint = Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::string("test".to_string()),
        };
        let cloned = constraint.clone();
        assert!(cloned.matches("word", "test"));
    }

    #[test]
    fn test_pattern_clone() {
        let pattern = Pattern::Constraint(Constraint::Wildcard);
        let cloned = pattern.clone();
        let tokens = vec!["a".to_string()];
        assert_eq!(cloned.extract_matching_positions("word", &tokens), vec![0]);
    }

    // ==================== references_field Tests ====================

    #[test]
    fn test_constraint_wildcard_references_any_field() {
        let c = Constraint::Wildcard;
        assert!(c.references_field("word"));
        assert!(c.references_field("entity"));
        assert!(c.references_field("pos"));
        assert!(c.references_field("anything"));
    }

    #[test]
    fn test_constraint_field_references_own_field_only() {
        let c = Constraint::Field {
            name: "entity".to_string(),
            matcher: Matcher::string("B-Gene".to_string()),
        };
        assert!(c.references_field("entity"));
        assert!(!c.references_field("word"));
        assert!(!c.references_field("pos"));
        assert!(!c.references_field("lemma"));
    }

    #[test]
    fn test_constraint_fuzzy_references_own_field_only() {
        let c = Constraint::Fuzzy {
            name: "word".to_string(),
            matcher: "cat".to_string(),
        };
        assert!(c.references_field("word"));
        assert!(!c.references_field("entity"));
    }

    #[test]
    fn test_constraint_negated_delegates() {
        let inner = Constraint::Field {
            name: "entity".to_string(),
            matcher: Matcher::string("B-Gene".to_string()),
        };
        let c = Constraint::Negated(Box::new(inner));
        assert!(c.references_field("entity"));
        assert!(!c.references_field("word"));
    }

    #[test]
    fn test_constraint_conjunctive_any_child() {
        let c = Constraint::Conjunctive(vec![
            Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::string("hello".to_string()),
            },
            Constraint::Field {
                name: "entity".to_string(),
                matcher: Matcher::string("B-Gene".to_string()),
            },
        ]);
        assert!(c.references_field("word"));
        assert!(c.references_field("entity"));
        assert!(!c.references_field("pos"));
    }

    #[test]
    fn test_constraint_disjunctive_any_child() {
        let c = Constraint::Disjunctive(vec![
            Constraint::Field {
                name: "pos".to_string(),
                matcher: Matcher::string("NN".to_string()),
            },
            Constraint::Field {
                name: "pos".to_string(),
                matcher: Matcher::string("NNS".to_string()),
            },
        ]);
        assert!(c.references_field("pos"));
        assert!(!c.references_field("word"));
    }

    #[test]
    fn test_constraint_nested_negated_conjunctive() {
        let c = Constraint::Negated(Box::new(Constraint::Conjunctive(vec![
            Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::string("cat".to_string()),
            },
            Constraint::Field {
                name: "entity".to_string(),
                matcher: Matcher::string("B-Gene".to_string()),
            },
        ])));
        assert!(c.references_field("word"));
        assert!(c.references_field("entity"));
        assert!(!c.references_field("pos"));
    }

    #[test]
    fn test_pattern_constraint_references_field() {
        let p = Pattern::Constraint(Constraint::Field {
            name: "entity".to_string(),
            matcher: Matcher::string("B-Gene".to_string()),
        });
        assert!(p.references_field("entity"));
        assert!(!p.references_field("word"));
    }

    #[test]
    fn test_pattern_wildcard_references_any_field() {
        let p = Pattern::Constraint(Constraint::Wildcard);
        assert!(p.references_field("word"));
        assert!(p.references_field("entity"));
        assert!(p.references_field("anything"));
    }

    #[test]
    fn test_pattern_disjunctive_references_field() {
        let p = Pattern::Disjunctive(vec![
            Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::string("cat".to_string()),
            }),
            Pattern::Constraint(Constraint::Field {
                name: "entity".to_string(),
                matcher: Matcher::string("B-Gene".to_string()),
            }),
        ]);
        assert!(p.references_field("word"));
        assert!(p.references_field("entity"));
        assert!(!p.references_field("pos"));
    }

    #[test]
    fn test_pattern_concatenated_references_field() {
        let p = Pattern::Concatenated(vec![
            Pattern::Constraint(Constraint::Field {
                name: "pos".to_string(),
                matcher: Matcher::string("DET".to_string()),
            }),
            Pattern::Constraint(Constraint::Field {
                name: "pos".to_string(),
                matcher: Matcher::string("NN".to_string()),
            }),
        ]);
        assert!(p.references_field("pos"));
        assert!(!p.references_field("word"));
    }

    #[test]
    fn test_pattern_named_capture_delegates() {
        let p = Pattern::NamedCapture {
            name: "subject".to_string(),
            pattern: Box::new(Pattern::Constraint(Constraint::Field {
                name: "entity".to_string(),
                matcher: Matcher::string("B-Gene".to_string()),
            })),
        };
        assert!(p.references_field("entity"));
        assert!(!p.references_field("word"));
    }

    #[test]
    fn test_pattern_mention_references_no_field() {
        let p = Pattern::Mention {
            arg_name: Some("arg".to_string()),
            label: "Person".to_string(),
        };
        assert!(!p.references_field("word"));
        assert!(!p.references_field("entity"));
        assert!(!p.references_field("anything"));
    }

    #[test]
    fn test_pattern_graph_traversal_checks_src_and_dst() {
        let p = Pattern::GraphTraversal {
            src: Box::new(Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::string("run".to_string()),
            })),
            traversal: Traversal::OutgoingWildcard,
            dst: Box::new(Pattern::Constraint(Constraint::Field {
                name: "entity".to_string(),
                matcher: Matcher::string("B-Gene".to_string()),
            })),
        };
        assert!(p.references_field("word"));
        assert!(p.references_field("entity"));
        assert!(!p.references_field("pos"));
    }

    #[test]
    fn test_pattern_graph_traversal_src_only() {
        let p = Pattern::GraphTraversal {
            src: Box::new(Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::string("run".to_string()),
            })),
            traversal: Traversal::OutgoingWildcard,
            dst: Box::new(Pattern::Constraint(Constraint::Wildcard)),
        };
        assert!(p.references_field("word"));
        assert!(p.references_field("entity"));
    }

    #[test]
    fn test_pattern_repetition_delegates() {
        let p = Pattern::Repetition {
            pattern: Box::new(Pattern::Constraint(Constraint::Field {
                name: "pos".to_string(),
                matcher: Matcher::string("NN".to_string()),
            })),
            min: 1,
            max: Some(3),
            kind: QuantifierKind::Greedy,
        };
        assert!(p.references_field("pos"));
        assert!(!p.references_field("word"));
    }

    #[test]
    fn test_pattern_assertion_delegates_to_inner() {
        let p = Pattern::Assertion(Assertion::PositiveLookahead(Box::new(
            Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::string("test".to_string()),
            }),
        )));
        assert!(p.references_field("word"));
        assert!(!p.references_field("entity"));
    }

    #[test]
    fn test_pattern_assertion_all_variants() {
        let make_inner = || {
            Box::new(Pattern::Constraint(Constraint::Field {
                name: "pos".to_string(),
                matcher: Matcher::string("VB".to_string()),
            }))
        };

        let variants = vec![
            Pattern::Assertion(Assertion::PositiveLookahead(make_inner())),
            Pattern::Assertion(Assertion::NegativeLookahead(make_inner())),
            Pattern::Assertion(Assertion::PositiveLookbehind(make_inner())),
            Pattern::Assertion(Assertion::NegativeLookbehind(make_inner())),
        ];

        for (i, p) in variants.iter().enumerate() {
            assert!(
                p.references_field("pos"),
                "Assertion variant {i} should reference 'pos'"
            );
            assert!(
                !p.references_field("word"),
                "Assertion variant {i} should not reference 'word'"
            );
        }
    }

    #[test]
    fn test_references_field_deeply_nested() {
        let p = Pattern::NamedCapture {
            name: "deep".to_string(),
            pattern: Box::new(Pattern::Repetition {
                pattern: Box::new(Pattern::Disjunctive(vec![
                    Pattern::Constraint(Constraint::Field {
                        name: "entity".to_string(),
                        matcher: Matcher::string("B-Gene".to_string()),
                    }),
                    Pattern::Constraint(Constraint::Field {
                        name: "word".to_string(),
                        matcher: Matcher::string("test".to_string()),
                    }),
                ])),
                min: 0,
                max: None,
                kind: QuantifierKind::Lazy,
            }),
        };
        assert!(p.references_field("entity"));
        assert!(p.references_field("word"));
        assert!(!p.references_field("pos"));
        assert!(!p.references_field("lemma"));
    }

    #[test]
    fn test_extract_positions_short_circuits_on_wrong_field() {
        let pattern = Pattern::Constraint(Constraint::Field {
            name: "entity".to_string(),
            matcher: Matcher::string("B-Gene".to_string()),
        });
        let tokens = vec![
            "The".to_string(),
            "B-Gene".to_string(),
            "protein".to_string(),
        ];
        let positions = pattern.extract_matching_positions("word", &tokens);
        assert!(
            positions.is_empty(),
            "Entity constraint should not match word tokens, got {:?}",
            positions
        );
    }

    #[test]
    fn test_extract_positions_works_when_field_matches() {
        let pattern = Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::string("cat".to_string()),
        });
        let tokens = vec!["the".to_string(), "cat".to_string(), "sat".to_string()];
        let positions = pattern.extract_matching_positions("word", &tokens);
        assert_eq!(positions, vec![1]);
    }

    #[test]
    fn test_extract_positions_wildcard_not_short_circuited() {
        let pattern = Pattern::Constraint(Constraint::Wildcard);
        let tokens = vec!["a".to_string(), "b".to_string()];
        let positions = pattern.extract_matching_positions("word", &tokens);
        assert_eq!(positions, vec![0, 1]);
    }
} 