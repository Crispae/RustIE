//! Graph access traits for unified interface over owned and zero-copy graphs.
//!
//! This module defines the `GraphAccess` trait which abstracts over different
//! graph storage formats, allowing the same traversal code to work with both
//! the owned `DirectedGraph` and zero-copy `ZeroCopyGraph` implementations.

use thiserror::Error;

/// Errors that can occur during graph operations
#[derive(Debug, Error)]
pub enum GraphError {
    #[error("Invalid magic number - expected zero-copy format")]
    InvalidMagic,
    
    #[error("Unsupported format version: {0}")]
    UnsupportedVersion(u16),
    
    #[error("Too many nodes: {0} (max 65535)")]
    TooManyNodes(usize),
    
    #[error("Too many labels: {0} (max 65535)")]
    TooManyLabels(usize),
    
    #[error("Label data too large: {0} bytes (max 65535)")]
    LabelDataTooLarge(usize),
    
    #[error("Buffer too small: need {needed} bytes, have {available}")]
    BufferTooSmall { needed: usize, available: usize },
    
    #[error("Invalid UTF-8 in label data")]
    InvalidUtf8,
    
    #[error("Node index out of bounds: {0}")]
    NodeOutOfBounds(usize),
    
    #[error("Alignment error in zero-copy data")]
    AlignmentError,
}

/// Unified graph access trait for both owned and zero-copy graphs.
///
/// This trait provides a common interface for accessing graph structure,
/// enabling the same traversal algorithms to work with different storage formats.
///
/// # Edge Format
/// Edges are returned as `(target_node, label_id)` pairs where:
/// - `target_node` is the index of the connected node
/// - `label_id` is an opaque identifier that can be resolved via `get_label()`
///
/// # Example
/// ```ignore
/// fn traverse<G: GraphAccess>(graph: &G, start: usize) {
///     if let Some(edges) = graph.outgoing(start) {
///         for (target, label_id) in edges {
///             if let Some(label) = graph.get_label(label_id) {
///                 println!("{} --{}-> {}", start, label, target);
///             }
///         }
///     }
/// }
/// ```
pub trait GraphAccess {
    /// Returns the number of nodes in the graph.
    fn node_count(&self) -> usize;
    
    /// Returns an iterator over incoming edges for the given node.
    ///
    /// Each edge is a `(source_node, label_id)` pair.
    /// Returns `None` if the node index is out of bounds.
    fn incoming(&self, node: usize) -> Option<impl Iterator<Item = (usize, usize)>>;
    
    /// Returns an iterator over outgoing edges for the given node.
    ///
    /// Each edge is a `(target_node, label_id)` pair.
    /// Returns `None` if the node index is out of bounds.
    fn outgoing(&self, node: usize) -> Option<impl Iterator<Item = (usize, usize)>>;
    
    /// Returns the label string for the given label ID.
    ///
    /// Returns `None` if the label ID is out of bounds.
    fn get_label(&self, label_id: usize) -> Option<&str>;
    
    /// Returns the label ID for the given label string.
    ///
    /// This is called rarely (labels are pre-resolved before traversal),
    /// so a linear scan implementation is acceptable.
    ///
    /// Returns `None` if the label is not found.
    fn get_label_id(&self, label: &str) -> Option<usize>;
    
    /// Returns the number of labels in the vocabulary.
    fn label_count(&self) -> usize;
    
    /// Returns an iterator over root node indices.
    ///
    /// Root nodes are typically nodes with no incoming edges.
    fn roots(&self) -> impl Iterator<Item = usize>;
    
    /// Returns the number of root nodes.
    fn root_count(&self) -> usize;
    
    /// Checks if the node has any incoming edges.
    #[inline]
    fn has_incoming(&self, node: usize) -> bool {
        self.incoming(node).map(|mut it| it.next().is_some()).unwrap_or(false)
    }
    
    /// Checks if the node has any outgoing edges.
    #[inline]
    fn has_outgoing(&self, node: usize) -> bool {
        self.outgoing(node).map(|mut it| it.next().is_some()).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Test that GraphError can be created and displayed
    #[test]
    fn test_graph_error_display() {
        let err = GraphError::TooManyNodes(70000);
        assert!(err.to_string().contains("70000"));
        
        let err = GraphError::BufferTooSmall { needed: 100, available: 50 };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));
    }
}
