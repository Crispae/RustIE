//! Position-aware tokenizer for edge labels (Odinson-style)
//!
//! This module provides a tokenizer that emits edge labels at specific positions
//! matching token indices in the sentence. This enables position-aware queries
//! where we can find which TOKEN POSITION has a specific edge label.
//!
//! Example:
//! For a sentence with 5 tokens where:
//! - Token 1 has incoming edge "nsubj"
//! - Token 3 has incoming edges "dobj" and "prep"
//!
//! The edge labels are indexed at positions:
//! - Position 0: (no edge, skip)
//! - Position 1: "nsubj"
//! - Position 2: (no edge, skip)
//! - Position 3: "dobj", "prep" (same position)
//! - Position 4: (no edge, skip)

use tantivy::tokenizer::{Token, TokenStream, Tokenizer, BoxTokenStream};

/// A token stream that emits edge labels at specific positions
/// matching token indices in the sentence.
///
/// This is similar to Odinson's DependencyTokenStream.
pub struct PositionAwareEdgeTokenStream {
    /// Edge labels per position: edges[position] = vec of labels at that position
    edges: Vec<Vec<String>>,
    /// Current token position (sentence token index)
    token_index: usize,
    /// Current edge index within the current token
    edge_index: usize,
    /// Current token being emitted
    token: Token,
}

impl PositionAwareEdgeTokenStream {
    /// Create a new token stream from edge labels per position
    ///
    /// # Arguments
    /// * `edges` - Vector where edges[i] contains labels for token position i
    pub fn new(edges: Vec<Vec<String>>) -> Self {
        Self {
            edges,
            token_index: 0,
            edge_index: 0,
            token: Token::default(),
        }
    }
}

impl TokenStream for PositionAwareEdgeTokenStream {
    fn advance(&mut self) -> bool {
        loop {
            // Check if we've processed all positions
            if self.token_index >= self.edges.len() {
                return false;
            }

            let token_edges = &self.edges[self.token_index];

            // Check if we have more edges at this position
            if self.edge_index < token_edges.len() {
                let label = &token_edges[self.edge_index];

                // Set token text
                self.token.text.clear();
                self.token.text.push_str(label);

                // Set position to match token index
                // All edges at the same token position share the same position
                self.token.position = self.token_index;

                // Set offsets (character-level, but we use position-based)
                self.token.offset_from = self.token_index;
                self.token.offset_to = self.token_index + 1;

                // Move to next edge at this position
                self.edge_index += 1;

                return true;
            }

            // No more edges at this position, move to next token
            self.token_index += 1;
            self.edge_index = 0;
            // Continue to find next position with edges
        }
    }

    fn token(&self) -> &Token {
        &self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.token
    }
}

/// Tokenizer that creates position-aware edge token streams
#[derive(Clone)]
pub struct PositionAwareEdgeTokenizer;

impl Tokenizer for PositionAwareEdgeTokenizer {
    type TokenStream<'a> = PositionAwareEdgeTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        // Parse the special format: "pos0_label1,label2|pos1_label3|..."
        // Format: positions separated by |, labels at each position separated by ,
        // Empty positions are represented as empty strings
        //
        // Example: "|nsubj||dobj,prep|" means:
        // - Position 0: no edges
        // - Position 1: "nsubj"
        // - Position 2: no edges
        // - Position 3: "dobj" and "prep"
        // - Position 4: no edges

        let edges: Vec<Vec<String>> = text
            .split('|')
            .map(|pos_labels| {
                if pos_labels.is_empty() {
                    Vec::new()
                } else {
                    pos_labels.split(',').map(|s| s.to_string()).collect()
                }
            })
            .collect();

        PositionAwareEdgeTokenStream::new(edges)
    }
}

/// Helper function to encode edge labels into position-aware format
/// for indexing.
///
/// # Arguments
/// * `edges_per_position` - Vector where edges[i] contains labels for token position i
///
/// # Returns
/// A string in the format: "|nsubj||dobj,prep|" suitable for the tokenizer
pub fn encode_position_aware_edges(edges_per_position: &[Vec<String>]) -> String {
    edges_per_position
        .iter()
        .map(|labels| labels.join(","))
        .collect::<Vec<_>>()
        .join("|")
}

// ========================================================================
// Position-aware token field tokenizer (for word, lemma, pos, entity, etc.)
// ========================================================================

/// A token stream for regular token fields (word, lemma, pos, etc.)
/// that properly assigns positions to each token.
///
/// Format: "John|eats|pizza|with|chopsticks"
/// Each token separated by | gets an incrementing position.
pub struct PositionAwareTokenStream {
    tokens: Vec<String>,
    current_index: usize,
    token: Token,
}

impl PositionAwareTokenStream {
    pub fn new(tokens: Vec<String>) -> Self {
        Self {
            tokens,
            current_index: 0,
            token: Token::default(),
        }
    }
}

impl TokenStream for PositionAwareTokenStream {
    fn advance(&mut self) -> bool {
        if self.current_index >= self.tokens.len() {
            return false;
        }

        let token_text = &self.tokens[self.current_index];

        self.token.text.clear();
        self.token.text.push_str(token_text);
        self.token.position = self.current_index;
        self.token.offset_from = self.current_index;
        self.token.offset_to = self.current_index + 1;

        self.current_index += 1;
        true
    }

    fn token(&self) -> &Token {
        &self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.token
    }
}

/// Tokenizer for regular token fields that preserves position information.
///
/// Input format: "John|eats|pizza|with|chopsticks"
/// Each | separator represents a token boundary, and positions are assigned sequentially.
#[derive(Clone)]
pub struct PositionAwareTokenTokenizer;

impl Tokenizer for PositionAwareTokenTokenizer {
    type TokenStream<'a> = PositionAwareTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        let tokens: Vec<String> = text.split('|').map(|s| s.to_string()).collect();
        PositionAwareTokenStream::new(tokens)
    }
}

/// Encode tokens into position-aware format for indexing.
///
/// # Arguments
/// * `tokens` - Vector of token strings (e.g., ["John", "eats", "pizza"])
///
/// # Returns
/// A string in format "John|eats|pizza" for the tokenizer
pub fn encode_position_aware_tokens(tokens: &[String]) -> String {
    tokens.join("|")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_aware_token_stream() {
        let edges = vec![
            vec![],                                    // Position 0: no edges
            vec!["nsubj".to_string()],                 // Position 1: nsubj
            vec![],                                    // Position 2: no edges
            vec!["dobj".to_string(), "prep".to_string()], // Position 3: dobj, prep
            vec![],                                    // Position 4: no edges
        ];

        let mut stream = PositionAwareEdgeTokenStream::new(edges);

        // Should emit "nsubj" at position 1
        assert!(stream.advance());
        assert_eq!(stream.token().text, "nsubj");
        assert_eq!(stream.token().position, 1);

        // Should emit "dobj" at position 3
        assert!(stream.advance());
        assert_eq!(stream.token().text, "dobj");
        assert_eq!(stream.token().position, 3);

        // Should emit "prep" at position 3 (same position)
        assert!(stream.advance());
        assert_eq!(stream.token().text, "prep");
        assert_eq!(stream.token().position, 3);

        // No more tokens
        assert!(!stream.advance());
    }

    #[test]
    fn test_encode_position_aware_edges() {
        let edges = vec![
            vec![],
            vec!["nsubj".to_string()],
            vec![],
            vec!["dobj".to_string(), "prep".to_string()],
            vec![],
        ];

        let encoded = encode_position_aware_edges(&edges);
        assert_eq!(encoded, "|nsubj||dobj,prep|");
    }

    #[test]
    fn test_tokenizer_roundtrip() {
        let edges = vec![
            vec![],
            vec!["nsubj".to_string()],
            vec![],
            vec!["dobj".to_string(), "prep".to_string()],
        ];

        let encoded = encode_position_aware_edges(&edges);
        let mut tokenizer = PositionAwareEdgeTokenizer;
        let mut stream = tokenizer.token_stream(&encoded);

        // Verify tokens
        assert!(stream.advance());
        assert_eq!(stream.token().text, "nsubj");
        assert_eq!(stream.token().position, 1);

        assert!(stream.advance());
        assert_eq!(stream.token().text, "dobj");
        assert_eq!(stream.token().position, 3);

        assert!(stream.advance());
        assert_eq!(stream.token().text, "prep");
        assert_eq!(stream.token().position, 3);

        assert!(!stream.advance());
    }

    #[test]
    fn test_position_aware_token_stream_basic() {
        let tokens = vec![
            "John".to_string(),
            "eats".to_string(),
            "pizza".to_string(),
        ];

        let mut stream = PositionAwareTokenStream::new(tokens);

        assert!(stream.advance());
        assert_eq!(stream.token().text, "John");
        assert_eq!(stream.token().position, 0);

        assert!(stream.advance());
        assert_eq!(stream.token().text, "eats");
        assert_eq!(stream.token().position, 1);

        assert!(stream.advance());
        assert_eq!(stream.token().text, "pizza");
        assert_eq!(stream.token().position, 2);

        assert!(!stream.advance());
    }

    #[test]
    fn test_token_tokenizer_roundtrip() {
        let tokens = vec![
            "John".to_string(),
            "eats".to_string(),
            "pizza".to_string(),
        ];

        let encoded = encode_position_aware_tokens(&tokens);
        assert_eq!(encoded, "John|eats|pizza");

        let mut tokenizer = PositionAwareTokenTokenizer;
        let mut stream = tokenizer.token_stream(&encoded);

        assert!(stream.advance());
        assert_eq!(stream.token().text, "John");
        assert_eq!(stream.token().position, 0);

        assert!(stream.advance());
        assert_eq!(stream.token().text, "eats");
        assert_eq!(stream.token().position, 1);

        assert!(stream.advance());
        assert_eq!(stream.token().text, "pizza");
        assert_eq!(stream.token().position, 2);

        assert!(!stream.advance());
    }
}
