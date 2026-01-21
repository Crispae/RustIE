//! Writer for creating zero-copy graph binary format.
//!
//! This module provides `ZeroCopyGraphWriter` which converts a `DirectedGraph`
//! into the zero-copy binary format that can be efficiently read at query time.

use crate::digraph::graph::DirectedGraph;
use crate::digraph::graph_trait::GraphError;
use crate::digraph::zero_copy::{MAGIC, VERSION, HEADER_SIZE};

/// Writer for creating zero-copy graph binary format.
///
/// This struct builds the binary representation of a graph that can be
/// read without allocations by `ZeroCopyGraph`.
///
/// # Limits
/// - Maximum 65,535 nodes
/// - Maximum 65,535 labels
/// - Maximum 65,535 bytes of label data
///
/// If any limit is exceeded, conversion will fail with an appropriate error.
///
/// # Example
/// ```ignore
/// let graph = DirectedGraph::new();
/// // ... add edges ...
/// 
/// match ZeroCopyGraphWriter::from_directed_graph(&graph) {
///     Ok(writer) => {
///         let bytes = writer.serialize();
///         // Store bytes in index
///     }
///     Err(e) => {
///         // Fall back to legacy format
///         let bytes = graph.to_bytes()?;
///     }
/// }
/// ```
pub struct ZeroCopyGraphWriter {
    num_nodes: u16,
    /// Incoming edges per node: Vec of (source, label_id) pairs
    incoming_edges: Vec<Vec<(u16, u16)>>,
    /// Outgoing edges per node: Vec of (target, label_id) pairs
    outgoing_edges: Vec<Vec<(u16, u16)>>,
    /// Label strings in order by ID
    labels: Vec<String>,
    /// Root node IDs
    roots: Vec<u16>,
}

impl ZeroCopyGraphWriter {
    /// Create a new empty writer with the given number of nodes.
    pub fn new(num_nodes: u16) -> Self {
        Self {
            num_nodes,
            incoming_edges: vec![Vec::new(); num_nodes as usize],
            outgoing_edges: vec![Vec::new(); num_nodes as usize],
            labels: Vec::new(),
            roots: Vec::new(),
        }
    }
    
    /// Convert an existing `DirectedGraph` to a writer.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Graph has more than 65,535 nodes
    /// - Graph has more than 65,535 unique labels
    /// - Total label data exceeds 65,535 bytes
    pub fn from_directed_graph(graph: &DirectedGraph) -> Result<Self, GraphError> {
        let node_count = graph.node_count();
        
        // Check node limit
        if node_count > u16::MAX as usize {
            return Err(GraphError::TooManyNodes(node_count));
        }
        
        let num_nodes = node_count as u16;
        let vocabulary = graph.vocabulary();
        
        // Check label count limit
        let label_count = vocabulary.len();
        if label_count > u16::MAX as usize {
            return Err(GraphError::TooManyLabels(label_count));
        }
        
        // Build label list and check total size
        let mut labels = Vec::with_capacity(label_count);
        let mut total_label_bytes = 0usize;
        
        for i in 0..label_count {
            if let Some(label) = vocabulary.get_term(i) {
                total_label_bytes += label.len();
                if total_label_bytes > u16::MAX as usize {
                    return Err(GraphError::LabelDataTooLarge(total_label_bytes));
                }
                labels.push(label.to_string());
            }
        }
        
        // Build edge lists
        let mut incoming_edges = vec![Vec::new(); node_count];
        let mut outgoing_edges = vec![Vec::new(); node_count];
        
        for node in 0..node_count {
            // Process incoming edges
            if let Some(edges) = graph.incoming(node) {
                for pair in edges.chunks(2) {
                    if pair.len() == 2 {
                        let source = pair[0] as u16;
                        let label_id = pair[1] as u16;
                        incoming_edges[node].push((source, label_id));
                    }
                }
            }
            
            // Process outgoing edges
            if let Some(edges) = graph.outgoing(node) {
                for pair in edges.chunks(2) {
                    if pair.len() == 2 {
                        let target = pair[0] as u16;
                        let label_id = pair[1] as u16;
                        outgoing_edges[node].push((target, label_id));
                    }
                }
            }
        }
        
        // Build roots list
        let roots: Vec<u16> = graph.roots().iter().map(|&r| r as u16).collect();
        
        Ok(Self {
            num_nodes,
            incoming_edges,
            outgoing_edges,
            labels,
            roots,
        })
    }
    
    /// Serialize to binary format.
    ///
    /// The returned bytes can be stored and later read by `ZeroCopyGraph::from_bytes()`.
    pub fn serialize(&self) -> Vec<u8> {
        // Calculate sizes
        let num_incoming: u32 = self.incoming_edges.iter().map(|e| e.len() as u32).sum();
        let num_outgoing: u32 = self.outgoing_edges.iter().map(|e| e.len() as u32).sum();
        let num_labels = self.labels.len() as u16;
        let num_roots = self.roots.len() as u16;
        
        let incoming_offsets_size = (self.num_nodes as usize + 1) * 4;
        let outgoing_offsets_size = (self.num_nodes as usize + 1) * 4;
        let incoming_edges_size = num_incoming as usize * 4;
        let outgoing_edges_size = num_outgoing as usize * 4;
        let label_offsets_size = (num_labels as usize + 1) * 2;
        let label_data_size: usize = self.labels.iter().map(|l| l.len()).sum();
        let roots_size = num_roots as usize * 2;
        
        let total_size = HEADER_SIZE 
            + incoming_offsets_size 
            + outgoing_offsets_size
            + incoming_edges_size 
            + outgoing_edges_size
            + label_offsets_size
            + label_data_size
            + roots_size;
        
        let mut buf = Vec::with_capacity(total_size);
        
        // Write header (20 bytes)
        buf.extend_from_slice(&MAGIC.to_le_bytes());           // magic: u32
        buf.extend_from_slice(&VERSION.to_le_bytes());         // version: u16
        buf.extend_from_slice(&self.num_nodes.to_le_bytes());  // num_nodes: u16
        buf.extend_from_slice(&num_incoming.to_le_bytes());    // num_incoming: u32
        buf.extend_from_slice(&num_outgoing.to_le_bytes());    // num_outgoing: u32
        buf.extend_from_slice(&num_labels.to_le_bytes());      // num_labels: u16
        buf.extend_from_slice(&num_roots.to_le_bytes());       // num_roots: u16
        
        // Write incoming offsets
        let mut offset = 0u32;
        for edges in &self.incoming_edges {
            buf.extend_from_slice(&offset.to_le_bytes());
            offset += edges.len() as u32;
        }
        buf.extend_from_slice(&offset.to_le_bytes()); // Sentinel
        
        // Write outgoing offsets
        let mut offset = 0u32;
        for edges in &self.outgoing_edges {
            buf.extend_from_slice(&offset.to_le_bytes());
            offset += edges.len() as u32;
        }
        buf.extend_from_slice(&offset.to_le_bytes()); // Sentinel
        
        // Write incoming edges
        for edges in &self.incoming_edges {
            for &(target, label_id) in edges {
                buf.extend_from_slice(&target.to_le_bytes());
                buf.extend_from_slice(&label_id.to_le_bytes());
            }
        }
        
        // Write outgoing edges
        for edges in &self.outgoing_edges {
            for &(target, label_id) in edges {
                buf.extend_from_slice(&target.to_le_bytes());
                buf.extend_from_slice(&label_id.to_le_bytes());
            }
        }
        
        // Write label offsets
        let mut label_offset = 0u16;
        for label in &self.labels {
            buf.extend_from_slice(&label_offset.to_le_bytes());
            label_offset += label.len() as u16;
        }
        buf.extend_from_slice(&label_offset.to_le_bytes()); // Sentinel
        
        // Write label data
        for label in &self.labels {
            buf.extend_from_slice(label.as_bytes());
        }
        
        // Write roots
        for &root in &self.roots {
            buf.extend_from_slice(&root.to_le_bytes());
        }
        
        debug_assert_eq!(buf.len(), total_size, "Buffer size mismatch");
        buf
    }
    
    /// Add an edge to the graph (for testing/manual construction).
    pub fn add_edge(&mut self, from: u16, to: u16, label_id: u16) {
        if (from as usize) < self.outgoing_edges.len() {
            self.outgoing_edges[from as usize].push((to, label_id));
        }
        if (to as usize) < self.incoming_edges.len() {
            self.incoming_edges[to as usize].push((from, label_id));
        }
    }
    
    /// Add a label and return its ID (for testing/manual construction).
    pub fn add_label(&mut self, label: &str) -> u16 {
        let id = self.labels.len() as u16;
        self.labels.push(label.to_string());
        id
    }
    
    /// Set root nodes (for testing/manual construction).
    pub fn set_roots(&mut self, roots: Vec<u16>) {
        self.roots = roots;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digraph::zero_copy::ZeroCopyGraph;
    use crate::digraph::graph_trait::GraphAccess;
    
    #[test]
    fn test_empty_graph_roundtrip() {
        let writer = ZeroCopyGraphWriter::new(0);
        let bytes = writer.serialize();
        
        assert!(ZeroCopyGraph::is_valid_format(&bytes));
        let graph = ZeroCopyGraph::from_bytes(&bytes).unwrap();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.label_count(), 0);
        assert_eq!(graph.root_count(), 0);
    }
    
    #[test]
    fn test_simple_graph_roundtrip() {
        let mut writer = ZeroCopyGraphWriter::new(3);
        
        // Add labels
        let nsubj = writer.add_label("nsubj");
        let dobj = writer.add_label("dobj");
        
        // Add edges: 0 --nsubj--> 1, 0 --dobj--> 2
        writer.add_edge(0, 1, nsubj);
        writer.add_edge(0, 2, dobj);
        
        // Set root
        writer.set_roots(vec![0]);
        
        let bytes = writer.serialize();
        
        // Parse and verify
        let graph = ZeroCopyGraph::from_bytes(&bytes).unwrap();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.label_count(), 2);
        assert_eq!(graph.root_count(), 1);
        
        // Check labels
        assert_eq!(graph.get_label(0), Some("nsubj"));
        assert_eq!(graph.get_label(1), Some("dobj"));
        assert_eq!(graph.get_label_id("nsubj"), Some(0));
        assert_eq!(graph.get_label_id("dobj"), Some(1));
        
        // Check outgoing edges from node 0
        let edges: Vec<_> = graph.outgoing(0).unwrap().collect();
        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&(1, 0))); // (target=1, label=nsubj)
        assert!(edges.contains(&(2, 1))); // (target=2, label=dobj)
        
        // Check incoming edges to node 1
        let edges: Vec<_> = graph.incoming(1).unwrap().collect();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], (0, 0)); // (source=0, label=nsubj)
        
        // Check roots
        let roots: Vec<_> = graph.roots().collect();
        assert_eq!(roots, vec![0]);
    }
    
    #[test]
    fn test_from_directed_graph() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");
        graph.add_edge(0, 2, "dobj");
        graph.set_roots(vec![0]);
        
        let writer = ZeroCopyGraphWriter::from_directed_graph(&graph).unwrap();
        let bytes = writer.serialize();
        
        let zc_graph = ZeroCopyGraph::from_bytes(&bytes).unwrap();
        assert_eq!(zc_graph.node_count(), 3);
        
        // Verify edges match
        let edges: Vec<_> = zc_graph.outgoing(0).unwrap().collect();
        assert_eq!(edges.len(), 2);
    }
}
