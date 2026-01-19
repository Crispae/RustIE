use crate::types::{Span, NamedCapture};

/// Match node with full metadata for selection algorithm
/// This structure preserves the complete match tree needed for Odinson-style
/// greedy/lazy disambiguation.
#[derive(Debug, Clone)]
pub struct MatchWithMetadata {
    pub span: Span,
    pub captures: Vec<NamedCapture>,
    pub kind: MatchKind,
}

impl MatchWithMetadata {
    /// Create a new match with metadata
    pub fn new(span: Span, captures: Vec<NamedCapture>, kind: MatchKind) -> Self {
        Self { span, captures, kind }
    }

    /// Create an atom match (simple constraint match)
    pub fn atom(span: Span, captures: Vec<NamedCapture>) -> Self {
        Self {
            span,
            captures,
            kind: MatchKind::Atom,
        }
    }

    /// Create a repetition match
    pub fn repetition(
        span: Span,
        captures: Vec<NamedCapture>,
        is_greedy: bool,
        sub_matches: Vec<MatchWithMetadata>,
    ) -> Self {
        Self {
            span,
            captures,
            kind: MatchKind::Repetition {
                is_greedy,
                sub_matches,
            },
        }
    }

    /// Create an optional match
    pub fn optional(
        span: Span,
        captures: Vec<NamedCapture>,
        is_greedy: bool,
        matched: bool,
    ) -> Self {
        Self {
            span,
            captures,
            kind: MatchKind::Optional { is_greedy, matched },
        }
    }

    /// Create a disjunction match
    pub fn disjunction(
        span: Span,
        captures: Vec<NamedCapture>,
        clause_index: usize,
    ) -> Self {
        Self {
            span,
            captures,
            kind: MatchKind::Disjunction { clause_index },
        }
    }

    /// Create a sequence match
    pub fn sequence(
        span: Span,
        captures: Vec<NamedCapture>,
        sub_matches: Vec<MatchWithMetadata>,
    ) -> Self {
        Self {
            span,
            captures,
            kind: MatchKind::Sequence { sub_matches },
        }
    }

    /// Get the length of the match span
    pub fn length(&self) -> usize {
        self.span.length()
    }
}

/// Match kind indicating the type of pattern that produced this match
#[derive(Debug, Clone)]
pub enum MatchKind {
    /// Simple constraint match (atom)
    Atom,

    /// Repetition match with greedy/lazy flag
    Repetition {
        is_greedy: bool,
        sub_matches: Vec<MatchWithMetadata>,
    },

    /// Optional match with greedy/lazy flag
    /// `matched` indicates whether the optional pattern matched something or was skipped
    Optional {
        is_greedy: bool,
        matched: bool,
    },

    /// Disjunction - tracks which clause matched
    Disjunction {
        clause_index: usize,
    },

    /// Sequence of matches (concatenated patterns)
    Sequence {
        sub_matches: Vec<MatchWithMetadata>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_with_metadata_atom() {
        let span = Span::new(0, 5);
        let match_node = MatchWithMetadata::atom(span.clone(), Vec::new());
        assert_eq!(match_node.span, span);
        assert!(matches!(match_node.kind, MatchKind::Atom));
    }

    #[test]
    fn test_match_with_metadata_repetition() {
        let span = Span::new(0, 10);
        let sub_match = MatchWithMetadata::atom(Span::new(0, 5), Vec::new());
        let match_node = MatchWithMetadata::repetition(
            span.clone(),
            Vec::new(),
            true, // greedy
            vec![sub_match],
        );
        assert_eq!(match_node.span, span);
        match match_node.kind {
            MatchKind::Repetition { is_greedy, sub_matches } => {
                assert!(is_greedy);
                assert_eq!(sub_matches.len(), 1);
            }
            _ => panic!("Expected Repetition"),
        }
    }

    #[test]
    fn test_match_with_metadata_optional() {
        let span = Span::new(0, 5);
        let match_node = MatchWithMetadata::optional(span.clone(), Vec::new(), false, true);
        assert_eq!(match_node.span, span);
        match match_node.kind {
            MatchKind::Optional { is_greedy, matched } => {
                assert!(!is_greedy); // lazy
                assert!(matched);
            }
            _ => panic!("Expected Optional"),
        }
    }

    #[test]
    fn test_match_length() {
        let span = Span::new(0, 10);
        let match_node = MatchWithMetadata::atom(span, Vec::new());
        assert_eq!(match_node.length(), 10);
    }
}
