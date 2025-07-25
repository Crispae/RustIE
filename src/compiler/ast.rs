use serde::{Deserialize, Serialize};
use regex::Regex;

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
    /// Create a regex matcher (pre-compiles the regex)
    pub fn regex(pattern: String) -> Self {
        let regex = std::sync::Arc::new(regex::Regex::new(&pattern).expect("Invalid regex pattern"));
        Matcher::Regex { pattern, regex }
    }
    /// Check if a token matches this matcher
    pub fn matches(&self, token: &str) -> bool {
        match self {
            Matcher::String(s) => token == s,
            Matcher::Regex { regex, .. } => regex.is_match(token),
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
}

/// Assertions for position-based matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Assertion {
    SentenceStart,
    SentenceEnd,
    PositiveLookahead(Box<Pattern>),
    NegativeLookahead(Box<Pattern>),
    PositiveLookbehind(Box<Pattern>),
    NegativeLookbehind(Box<Pattern>),
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
    Repetition { pattern: Box<Pattern>, min: usize, max: Option<usize> },
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

    /// Create a sentence start assertion
    pub fn sentence_start_assertion() -> Pattern {
        Pattern::Assertion(Assertion::SentenceStart)
    }

    /// Create a sentence end assertion
    pub fn sentence_end_assertion() -> Pattern {
        Pattern::Assertion(Assertion::SentenceEnd)
    }

    /// Create a repetition pattern
    pub fn repetition_pattern(pattern: Pattern, min: usize, max: Option<usize>) -> Pattern {
        Pattern::Repetition {
            pattern: Box::new(pattern),
            min,
            max,
        }
    }
} 