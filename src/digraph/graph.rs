use std::collections::HashMap;
use anyhow::{Result, anyhow};
use crate::data::dependency_vocabulary::DependencyVocabulary;

/// Vocabulary for dependency labels (matches Scala implementation)
#[derive(Debug, Clone)]
pub struct Vocabulary {
    id_to_term: Vec<String>,
    term_to_id: HashMap<String, usize>,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self {
            id_to_term: Vec::new(),
            term_to_id: HashMap::new(),
        }
    }

    /// Get or create ID for a term
    pub fn get_or_create_id(&mut self, term: &str) -> usize {
        if let Some(&id) = self.term_to_id.get(term) {
            id
        } else {
            let id = self.id_to_term.len();
            self.id_to_term.push(term.to_string());
            self.term_to_id.insert(term.to_string(), id);
            id
        }
    }

    /// Get ID for a term
    pub fn get_id(&self, term: &str) -> Option<usize> {
        self.term_to_id.get(term).copied()
    }

    /// Get term for an ID
    pub fn get_term(&self, id: usize) -> Option<&str> {
        self.id_to_term.get(id).map(|s| s.as_str())
    }

    /// Check if vocabulary contains a term
    pub fn contains(&self, term: &str) -> bool {
        self.term_to_id.contains_key(term)
    }

    /// Check if vocabulary contains an ID
    pub fn contains_id(&self, id: usize) -> bool {
        id < self.id_to_term.len()
    }

    /// Get the number of terms
    pub fn len(&self) -> usize {
        self.id_to_term.len()
    }

    /// Check if vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.id_to_term.is_empty()
    }

    /// Convert from DependencyVocabulary
    pub fn from_dependency_vocabulary(vocab: &DependencyVocabulary) -> Self {
        let mut new_vocab = Self::new();
        for (label, &id) in &vocab.label_to_id {
            new_vocab.id_to_term.push(label.clone());
            new_vocab.term_to_id.insert(label.clone(), id as usize);
        }
        new_vocab
    }

    /// Convert to DependencyVocabulary
    pub fn to_dependency_vocabulary(&self) -> DependencyVocabulary {
        let mut vocab = DependencyVocabulary::new();
        for (term, &id) in &self.term_to_id {
            vocab.label_to_id.insert(term.clone(), id as u32);
            vocab.id_to_label.insert(id as u32, term.clone());
        }
        vocab.next_id = self.id_to_term.len() as u32;
        vocab
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Label matcher for dependency edges (matches Scala implementation)
#[derive(Debug, Clone)]
pub enum LabelMatcher {
    Exact { string: String, id: usize },
    Regex { pattern: String, regex: std::sync::Arc<regex::Regex> },
    Fail,
}

impl LabelMatcher {
    /// Create an exact label matcher
    pub fn exact(string: String, id: usize) -> Self {
        Self::Exact { string, id }
    }

    /// Create a regex label matcher (pre-compiles the regex)
    pub fn regex(pattern: String) -> Self {
        let regex = std::sync::Arc::new(regex::Regex::new(&pattern).expect("Invalid regex pattern"));
        Self::Regex { pattern, regex }
    }

    /// Create a failing matcher
    pub fn fail() -> Self {
        Self::Fail
    }

    /// Check if a label ID matches this matcher
    pub fn matches(&self, label_id: usize, vocabulary: &Vocabulary) -> bool {
        match self {
            LabelMatcher::Exact { id, .. } => label_id == *id,
            LabelMatcher::Regex { regex, .. } => {
                if let Some(term) = vocabulary.get_term(label_id) {
                    regex.is_match(term)
                } else {
                    false
                }
            }
            LabelMatcher::Fail => false,
        }
    }
}

/// Directed graph matching Scala implementation structure
/// 
/// This matches the Scala DirectedGraph structure:
/// ```scala
/// case class DirectedGraph(
///     incoming: Array[Array[Int]],  // Flattened (node, label) pairs
///     outgoing: Array[Array[Int]],  // Flattened (node, label) pairs
///     roots: Array[Int]
/// )
/// ```
#[derive(Debug, Clone)]
pub struct DirectedGraph {
    /// Incoming edges for each node as flattened (source_node, label_id) pairs
    incoming: Vec<Vec<usize>>,
    /// Outgoing edges for each node as flattened (target_node, label_id) pairs
    outgoing: Vec<Vec<usize>>,
    /// Root nodes
    roots: Vec<usize>,
    /// Vocabulary for label IDs
    vocabulary: Vocabulary,
}

impl DirectedGraph {
    pub fn new() -> Self {
        Self {
            incoming: Vec::new(),
            outgoing: Vec::new(),
            roots: Vec::new(),
            vocabulary: Vocabulary::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node_id: usize) {
        // Ensure we have enough space for the node
        while self.incoming.len() <= node_id {
            self.incoming.push(Vec::new());
        }
        while self.outgoing.len() <= node_id {
            self.outgoing.push(Vec::new());
        }
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, from: usize, to: usize, label: &str) {
        // Ensure nodes exist
        self.add_node(from);
        self.add_node(to);
        
        // Get or create label ID
        let label_id = self.vocabulary.get_or_create_id(label);
        
        // Add to outgoing edges (from -> to with label_id)
        // Format: [target_node, label_id, target_node, label_id, ...]
        self.outgoing[from].push(to);
        self.outgoing[from].push(label_id);
        
        // Add to incoming edges (to <- from with label_id)
        // Format: [source_node, label_id, source_node, label_id, ...]
        self.incoming[to].push(from);
        self.incoming[to].push(label_id);
    }

    /// Get incoming edges for a node (matches Scala implementation)
    pub fn incoming(&self, node_id: usize) -> Option<&[usize]> {
        self.incoming.get(node_id).map(|v| v.as_slice())
    }

    /// Get outgoing edges for a node (matches Scala implementation)
    pub fn outgoing(&self, node_id: usize) -> Option<&[usize]> {
        self.outgoing.get(node_id).map(|v| v.as_slice())
    }

    /// Check if a node has incoming edges (matches Scala isDefinedAt)
    pub fn has_incoming(&self, node_id: usize) -> bool {
        node_id < self.incoming.len() && !self.incoming[node_id].is_empty()
    }

    /// Check if a node has outgoing edges (matches Scala isDefinedAt)
    pub fn has_outgoing(&self, node_id: usize) -> bool {
        node_id < self.outgoing.len() && !self.outgoing[node_id].is_empty()
    }

    /// Get the vocabulary
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// Set root nodes
    pub fn set_roots(&mut self, roots: Vec<usize>) {
        self.roots = roots;
    }

    /// Get root nodes
    pub fn roots(&self) -> &[usize] {
        &self.roots
    }

    /// Get the number of nodes
    pub fn node_count(&self) -> usize {
        self.incoming.len().max(self.outgoing.len())
    }

    /// Get the number of edges
    pub fn edge_count(&self) -> usize {
        let mut count = 0;
        for edges in &self.outgoing {
            count += edges.len() / 2; // Each edge is (node, label) pair
        }
        count
    }

    /// Create graph from edges (convenience method)
    pub fn from_edges(edges: &[(usize, usize, &str)]) -> Self {
        let mut graph = Self::new();
        for &(from, to, label) in edges {
            graph.add_edge(from, to, label);
        }
        graph
    }

    /// Create a graph from a list of edges with integer labels
    pub fn from_edges_with_ids(edges: &[(usize, usize, usize)], vocabulary: &Vocabulary) -> Self {
        let mut graph = Self::new();
        graph.vocabulary = vocabulary.clone();
        
        for &(from, to, label_id) in edges {
            // Ensure nodes exist
            graph.add_node(from);
            graph.add_node(to);
            
            // Add to outgoing edges (from -> to with label_id)
            graph.outgoing[from].push(to);
            graph.outgoing[from].push(label_id);
            
            // Add to incoming edges (to <- from with label_id)
            graph.incoming[to].push(from);
            graph.incoming[to].push(label_id);
        }
        
        // Find root nodes (nodes with no incoming edges)
        let mut roots = Vec::new();
        for node_id in 0..graph.node_count() {
            if !graph.has_incoming(node_id) {
                roots.push(node_id);
            }
        }
        graph.set_roots(roots);
        
        graph
    }

    /// Create a graph from a dependency vocabulary and integer edges
    pub fn from_dependency_vocabulary_and_edges(edges: &[(usize, usize, usize)], vocab: &DependencyVocabulary) -> Self {
        let vocabulary = Vocabulary::from_dependency_vocabulary(vocab);
        Self::from_edges_with_ids(edges, &vocabulary)
    }

    /// Serialize to bytes (matches Scala implementation)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        
        // Write node count
        buffer.extend_from_slice(&(self.node_count() as u32).to_le_bytes());
        
        // Write incoming edges
        for edges in &self.incoming {
            buffer.extend_from_slice(&(edges.len() as u32).to_le_bytes());
            for &x in edges {
                buffer.extend_from_slice(&(x as u32).to_le_bytes());
            }
        }
        
        // Write outgoing edges
        for edges in &self.outgoing {
            buffer.extend_from_slice(&(edges.len() as u32).to_le_bytes());
            for &x in edges {
                buffer.extend_from_slice(&(x as u32).to_le_bytes());
            }
        }
        
        // Write roots
        buffer.extend_from_slice(&(self.roots.len() as u32).to_le_bytes());
        for &x in &self.roots {
            buffer.extend_from_slice(&(x as u32).to_le_bytes());
        }
        
        // Write vocabulary
        let vocab_bytes = self.vocabulary.to_bytes()?;
        buffer.extend_from_slice(&(vocab_bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&vocab_bytes);
        
        Ok(buffer)
    }

    /// Deserialize from bytes (matches Scala implementation)
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut offset = 0;
        
        // Read node count
        if offset + 4 > bytes.len() {
            return Err(anyhow!("Insufficient bytes for node count"));
        }
        let node_count = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
        offset += 4;
        
        let mut graph = Self::new();
        
        // Read incoming edges
        for _ in 0..node_count {
            if offset + 4 > bytes.len() {
                return Err(anyhow!("Insufficient bytes for incoming edge count"));
            }
            let edge_count = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
            offset += 4;
            
            if offset + edge_count * 4 > bytes.len() {
                return Err(anyhow!("Insufficient bytes for incoming edges"));
            }
            let mut edges = Vec::with_capacity(edge_count);
            for _ in 0..edge_count {
                let value = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
                edges.push(value);
                offset += 4;
            }
            graph.incoming.push(edges);
        }
        
        // Read outgoing edges
        for _ in 0..node_count {
            if offset + 4 > bytes.len() {
                return Err(anyhow!("Insufficient bytes for outgoing edge count"));
            }
            let edge_count = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
            offset += 4;
            
            if offset + edge_count * 4 > bytes.len() {
                return Err(anyhow!("Insufficient bytes for outgoing edges"));
            }
            let mut edges = Vec::with_capacity(edge_count);
            for _ in 0..edge_count {
                let value = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
                edges.push(value);
                offset += 4;
            }
            graph.outgoing.push(edges);
        }
        
        // Read roots
        if offset + 4 > bytes.len() {
            return Err(anyhow!("Insufficient bytes for root count"));
        }
        let root_count = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
        offset += 4;
        
        if offset + root_count * 4 > bytes.len() {
            return Err(anyhow!("Insufficient bytes for roots"));
        }
        for _ in 0..root_count {
            let root = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
            graph.roots.push(root);
            offset += 4;
        }
        
        // Read vocabulary
        if offset + 4 <= bytes.len() {
            let vocab_len = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
            offset += 4;
            if offset + vocab_len <= bytes.len() {
                graph.vocabulary = Vocabulary::from_bytes(&bytes[offset..offset+vocab_len])?;
            }
        }
        
        Ok(graph)
    }
}

impl Vocabulary {
    /// Serialize vocabulary to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        
        // Write term count
        buffer.extend_from_slice(&(self.id_to_term.len() as u32).to_le_bytes());
        
        // Write terms
        for term in &self.id_to_term {
            let term_bytes = term.as_bytes();
            buffer.extend_from_slice(&(term_bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(term_bytes);
        }
        
        Ok(buffer)
    }

    /// Deserialize vocabulary from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut offset = 0;
        
        // Read term count
        if offset + 4 > bytes.len() {
            return Err(anyhow!("Insufficient bytes for term count"));
        }
        let term_count = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
        offset += 4;
        
        let mut vocabulary = Vocabulary::new();
        
        // Read terms
        for _ in 0..term_count {
            if offset + 4 > bytes.len() {
                return Err(anyhow!("Insufficient bytes for term length"));
            }
            let term_len = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
            offset += 4;
            
            if offset + term_len > bytes.len() {
                return Err(anyhow!("Insufficient bytes for term"));
            }
            
            let term = std::str::from_utf8(&bytes[offset..offset+term_len])
                .map_err(|e| anyhow!("Invalid UTF-8 in term: {}", e))?
                .to_string();
            
            vocabulary.id_to_term.push(term.clone());
            vocabulary.term_to_id.insert(term, vocabulary.id_to_term.len() - 1);
            
            offset += term_len;
        }
        
        Ok(vocabulary)
    }
}

impl Default for DirectedGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// GraphAccess trait implementation for DirectedGraph
// ============================================================================

use crate::digraph::graph_trait::GraphAccess;

/// Iterator over edges stored as flattened (node, label_id) pairs.
pub struct EdgePairIterator<'a> {
    edges: &'a [usize],
    pos: usize,
}

impl<'a> EdgePairIterator<'a> {
    fn new(edges: &'a [usize]) -> Self {
        Self { edges, pos: 0 }
    }
}

impl<'a> Iterator for EdgePairIterator<'a> {
    type Item = (usize, usize);
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + 1 < self.edges.len() {
            let target = self.edges[self.pos];
            let label_id = self.edges[self.pos + 1];
            self.pos += 2;
            Some((target, label_id))
        } else {
            None
        }
    }
    
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.edges.len() - self.pos) / 2;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for EdgePairIterator<'a> {}

impl GraphAccess for DirectedGraph {
    #[inline]
    fn node_count(&self) -> usize {
        self.incoming.len().max(self.outgoing.len())
    }
    
    #[inline]
    fn incoming(&self, node: usize) -> Option<impl Iterator<Item = (usize, usize)>> {
        self.incoming.get(node).map(|edges| EdgePairIterator::new(edges))
    }
    
    #[inline]
    fn outgoing(&self, node: usize) -> Option<impl Iterator<Item = (usize, usize)>> {
        self.outgoing.get(node).map(|edges| EdgePairIterator::new(edges))
    }
    
    #[inline]
    fn get_label(&self, label_id: usize) -> Option<&str> {
        self.vocabulary.get_term(label_id)
    }
    
    #[inline]
    fn get_label_id(&self, label: &str) -> Option<usize> {
        self.vocabulary.get_id(label)
    }
    
    #[inline]
    fn label_count(&self) -> usize {
        self.vocabulary.len()
    }
    
    #[inline]
    fn roots(&self) -> impl Iterator<Item = usize> {
        self.roots.iter().copied()
    }
    
    #[inline]
    fn root_count(&self) -> usize {
        self.roots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary() {
        let mut vocab = Vocabulary::new();
        
        // Test get_or_create_id
        assert_eq!(vocab.get_or_create_id("nsubj"), 0);
        assert_eq!(vocab.get_or_create_id("dobj"), 1);
        assert_eq!(vocab.get_or_create_id("nsubj"), 0); // Should return existing ID
        
        // Test get_id
        assert_eq!(vocab.get_id("nsubj"), Some(0));
        assert_eq!(vocab.get_id("dobj"), Some(1));
        assert_eq!(vocab.get_id("nonexistent"), None);
        
        // Test get_term
        assert_eq!(vocab.get_term(0), Some("nsubj"));
        assert_eq!(vocab.get_term(1), Some("dobj"));
        assert_eq!(vocab.get_term(2), None);
    }

    #[test]
    fn test_directed_graph() {
        let mut graph = DirectedGraph::new();
        
        // Add edges
        graph.add_edge(0, 1, "nsubj");
        graph.add_edge(1, 2, "dobj");
        graph.add_edge(0, 3, "amod");
        
        // Test outgoing edges
        let outgoing = graph.outgoing(0).unwrap();
        assert_eq!(outgoing.len(), 4); // 2 edges * 2 values per edge
        assert_eq!(outgoing[0], 1); // target node
        assert_eq!(outgoing[1], 0); // label ID for "nsubj"
        assert_eq!(outgoing[2], 3); // target node
        assert_eq!(outgoing[3], 2); // label ID for "amod"
        
        // Test incoming edges
        let incoming = graph.incoming(1).unwrap();
        assert_eq!(incoming.len(), 2); // 1 edge * 2 values per edge
        assert_eq!(incoming[0], 0); // source node
        assert_eq!(incoming[1], 0); // label ID for "nsubj"
    }

    #[test]
    fn test_label_matcher() {
        let mut vocab = Vocabulary::new();
        vocab.get_or_create_id("nsubj");
        vocab.get_or_create_id("dobj");
        
        let exact_matcher = LabelMatcher::exact("nsubj".to_string(), 0);
        let regex_matcher = LabelMatcher::regex("subj".to_string());
        let fail_matcher = LabelMatcher::fail();
        
        assert!(exact_matcher.matches(0, &vocab));
        assert!(!exact_matcher.matches(1, &vocab));
        
        assert!(regex_matcher.matches(0, &vocab)); // "nsubj" contains "subj"
        assert!(!regex_matcher.matches(1, &vocab)); // "dobj" doesn't contain "subj"
        
        assert!(!fail_matcher.matches(0, &vocab));
    }
} 