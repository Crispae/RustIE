use serde::{Deserialize, Serialize};

/// Represents a complete document with metadata and sentences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub metadata: Vec<String>,
    pub sentences: Vec<Sentence>,
}

/// Represents a single sentence with its fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sentence {
    pub numTokens: u32,
    pub fields: Vec<Field>,
}

/// Represents a field in a sentence
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "$type")]
pub enum Field {
    #[serde(rename = "ai.lum.odinson.TokensField")]
    TokensField {
        name: String,
        tokens: Vec<String>,
    },
    #[serde(rename = "ai.lum.odinson.GraphField")]
    GraphField {
        name: String,
        edges: Vec<(u32, u32, String)>, // (from, to, relation)
        roots: Vec<u32>,
    },
}

/// Represents a tokens field (word, lemma, tag, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokensField {
    pub name: String,
    pub tokens: Vec<String>,
}

/// Represents a dependency graph field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphField {
    pub name: String,
    pub edges: Vec<(u32, u32, String)>, // (from, to, relation)
    pub roots: Vec<u32>,
}

impl Document {
    /// Get a specific field by name from a sentence
    pub fn get_field(&self, sentence_idx: usize, field_name: &str) -> Option<&Field> {
        self.sentences.get(sentence_idx)?.fields.iter()
            .find(|field| match field {
                Field::TokensField { name, .. } => name == field_name,
                Field::GraphField { name, .. } => name == field_name,
            })
    }

    /// Get tokens from a specific field
    pub fn get_tokens(&self, sentence_idx: usize, field_name: &str) -> Option<&[String]> {
        match self.get_field(sentence_idx, field_name)? {
            Field::TokensField { tokens, .. } => Some(tokens),
            _ => None,
        }
    }

    /// Get dependencies from a sentence
    pub fn get_dependencies(&self, sentence_idx: usize) -> Option<GraphField> {
        match self.get_field(sentence_idx, "dependencies")? {
            Field::GraphField { edges, roots, .. } => {
                Some(GraphField {
                    name: "dependencies".to_string(),
                    edges: edges.clone(),
                    roots: roots.clone(),
                })
            }
            _ => None,
        }
    }

    /// Convert dependencies to string format for Tantivy indexing
    pub fn dependencies_to_string(&self, sentence_idx: usize) -> Option<String> {
        let deps = self.get_dependencies(sentence_idx)?;
        let edges: Vec<String> = deps.edges.iter()
            .map(|(from, to, rel)| format!("{}:{}:{}", from, to, rel))
            .collect();
        Some(edges.join(" "))
    }

    /// Get sentence length
    pub fn sentence_length(&self, sentence_idx: usize) -> Option<u32> {
        self.sentences.get(sentence_idx).map(|s| s.numTokens)
    }
}

impl TokensField {
    /// Convert tokens to a single string for Tantivy indexing
    pub fn to_string(&self) -> String {
        self.tokens.join(" ")
    }
}

impl GraphField {
    /// Convert edges to a string format for Tantivy indexing
    pub fn edges_to_string(&self) -> String {
        self.edges.iter()
            .map(|(from, to, rel)| format!("{}:{}:{}", from, to, rel))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get incoming edges for a token
    pub fn incoming_edges(&self, token_idx: u32) -> Vec<(u32, String)> {
        self.edges.iter()
            .filter(|(_, to, _)| *to == token_idx)
            .map(|(from, _, rel)| (*from, rel.to_string()))
            .collect()
    }

    /// Get outgoing edges for a token
    pub fn outgoing_edges(&self, token_idx: u32) -> Vec<(u32, String)> {
        self.edges.iter()
            .filter(|(from, _, _)| *from == token_idx)
            .map(|(_, to, rel)| (*to, rel.to_string()))
            .collect()
    }
} 

// Metadata field will be added in future