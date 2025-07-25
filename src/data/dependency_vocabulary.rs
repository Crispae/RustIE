use std::collections::HashMap;
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// Vocabulary for mapping dependency labels to integer IDs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyVocabulary {
    /// Mapping from label string to integer ID
    pub label_to_id: HashMap<String, u32>,
    /// Mapping from integer ID to label string
    pub id_to_label: HashMap<u32, String>,
    /// Next available ID
    pub next_id: u32,
}

impl DependencyVocabulary {
    /// Create a new empty vocabulary
    pub fn new() -> Self {
        Self {
            label_to_id: HashMap::new(),
            id_to_label: HashMap::new(),
            next_id: 0,
        }
    }

    /// Get or create an ID for a dependency label
    pub fn get_or_create_id(&mut self, label: &str) -> u32 {
        if let Some(&id) = self.label_to_id.get(label) {
            id
        } else {
            let id = self.next_id;
            self.label_to_id.insert(label.to_string(), id);
            self.id_to_label.insert(id, label.to_string());
            self.next_id += 1;
            id
        }
    }

    /// Get the label for a given ID
    pub fn get_label(&self, id: u32) -> Option<&String> {
        self.id_to_label.get(&id)
    }

    /// Get the ID for a given label
    pub fn get_id(&self, label: &str) -> Option<&u32> {
        self.label_to_id.get(label)
    }

    /// Convert a list of dependency edges to use integer IDs
    pub fn convert_edges(&mut self, edges: &[(u32, u32, String)]) -> Vec<(u32, u32, u32)> {
        edges.iter()
            .map(|(from, to, label)| {
                let label_id = self.get_or_create_id(label);
                (*from, *to, label_id)
            })
            .collect()
    }

    /// Convert integer edge IDs back to string labels
    pub fn convert_edges_back(&self, edges: &[(u32, u32, u32)]) -> Vec<(u32, u32, String)> {
        edges.iter()
            .map(|(from, to, label_id)| {
                let label = self.get_label(*label_id)
                    .unwrap_or(&format!("unknown_{}", label_id))
                    .clone();
                (*from, *to, label)
            })
            .collect()
    }

    /// Save vocabulary to a file
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load vocabulary from a file
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let content = fs::read_to_string(path)?;
        let vocab: Self = serde_json::from_str(&content)?;
        Ok(vocab)
    }

    /// Get the size of the vocabulary
    pub fn size(&self) -> usize {
        self.label_to_id.len()
    }

    /// Get all labels in the vocabulary
    pub fn labels(&self) -> Vec<&String> {
        self.label_to_id.keys().collect()
    }

    /// Check if vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.label_to_id.is_empty()
    }
}

impl Default for DependencyVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_basic() {
        let mut vocab = DependencyVocabulary::new();
        
        // Test getting IDs for new labels
        let id1 = vocab.get_or_create_id("nsubj");
        let id2 = vocab.get_or_create_id("dobj");
        let id3 = vocab.get_or_create_id("nsubj"); // Should return same ID
        
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 0); // Same as id1
        
        // Test getting labels
        assert_eq!(vocab.get_label(0), Some(&"nsubj".to_string()));
        assert_eq!(vocab.get_label(1), Some(&"dobj".to_string()));
        assert_eq!(vocab.get_label(2), None);
        
        // Test getting IDs
        assert_eq!(vocab.get_id("nsubj"), Some(&0));
        assert_eq!(vocab.get_id("dobj"), Some(&1));
        assert_eq!(vocab.get_id("unknown"), None);
    }

    #[test]
    fn test_convert_edges() {
        let mut vocab = DependencyVocabulary::new();
        
        let edges = vec![
            (0, 1, "nsubj".to_string()),
            (1, 2, "dobj".to_string()),
            (0, 3, "nsubj".to_string()), // Same label
        ];
        
        let converted = vocab.convert_edges(&edges);
        
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0], (0, 1, 0)); // nsubj -> 0
        assert_eq!(converted[1], (1, 2, 1)); // dobj -> 1
        assert_eq!(converted[2], (0, 3, 0)); // nsubj -> 0
        
        // Test converting back
        let converted_back = vocab.convert_edges_back(&converted);
        assert_eq!(converted_back, edges);
    }

    #[test]
    fn test_save_load() {
        let mut vocab = DependencyVocabulary::new();
        vocab.get_or_create_id("nsubj");
        vocab.get_or_create_id("dobj");
        
        let temp_path = std::env::temp_dir().join("test_vocab.json");
        
        // Save
        vocab.save(&temp_path).unwrap();
        
        // Load
        let loaded_vocab = DependencyVocabulary::load(&temp_path).unwrap();
        
        assert_eq!(loaded_vocab.size(), vocab.size());
        assert_eq!(loaded_vocab.get_label(0), vocab.get_label(0));
        assert_eq!(loaded_vocab.get_label(1), vocab.get_label(1));
        
        // Cleanup
        let _ = fs::remove_file(temp_path);
    }
} 