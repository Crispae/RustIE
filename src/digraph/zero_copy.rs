//! Zero-copy graph implementation for efficient graph deserialization.
//!
//! This module provides `ZeroCopyGraph`, a graph structure that reads directly
//! from a byte slice without any allocations. This is significantly faster than
//! the owned `DirectedGraph` for query-time access.
//!
//! # Binary Format (v1)
//!
//! All multi-byte values are little-endian.
//!
//! ```text
//! HEADER (20 bytes):
//!   magic: u32 = 0x5A435047 ("ZCPG")
//!   version: u16 = 1
//!   num_nodes: u16
//!   num_incoming: u32
//!   num_outgoing: u32  
//!   num_labels: u16
//!   num_roots: u16
//!
//! INCOMING_OFFSETS: (num_nodes + 1) × u32
//! OUTGOING_OFFSETS: (num_nodes + 1) × u32
//! INCOMING_EDGES: num_incoming × PackedEdge (4 bytes each)
//! OUTGOING_EDGES: num_outgoing × PackedEdge (4 bytes each)
//! LABEL_OFFSETS: (num_labels + 1) × u16
//! LABEL_DATA: concatenated UTF-8 strings
//! ROOT_IDS: num_roots × u16
//! ```


/// # Lifetime Safety
/// 
/// `ZeroCopyGraph<'a>` borrows from the input byte slice with lifetime `'a`.
/// All iterators and string slices returned by this type also carry lifetime `'a`.
/// 
/// ## Safe Usage Pattern
/// ```
/// let binary_data: &[u8] = doc.get_bytes(...);  // Lives for scope
/// let graph = ZeroCopyGraph::from_bytes(binary_data)?;  // Borrows binary_data
/// let engine = GraphTraversal::new(graph);  // Owns graph (preserves 'a)
/// engine.query(...);  // All accesses valid
/// // binary_data dropped after all uses ✓
/// ```
/// 
/// ## Invariants
/// - `binary_data` must outlive all uses of `ZeroCopyGraph`
/// - Iterators from `outgoing()`/`incoming()` borrow from the graph
/// - No graph data may escape the function that owns `binary_data`

use crate::digraph::graph_trait::{GraphAccess, GraphError};
use zerocopy::{FromBytes, Immutable, KnownLayout, little_endian as le};

/// Magic number identifying zero-copy graph format: "ZCPG" in ASCII
pub const MAGIC: u32 = 0x5A435047;

/// Current format version
pub const VERSION: u16 = 1;

/// Header size in bytes
pub const HEADER_SIZE: usize = 20;

/// Packed edge structure (4 bytes, zerocopy-safe).
///
/// Stored as two little-endian u16 values for target node and label ID.
#[derive(FromBytes, Immutable, KnownLayout, Clone, Copy, Debug)]
#[repr(C)]
pub struct PackedEdge {
    pub target: le::U16,
    pub label_id: le::U16,
}

impl PackedEdge {
    /// Create a new packed edge
    #[inline]
    pub fn new(target: u16, label_id: u16) -> Self {
        Self {
            target: le::U16::new(target),
            label_id: le::U16::new(label_id),
        }
    }
    
    /// Get target node index
    #[inline]
    pub fn target(&self) -> usize {
        self.target.get() as usize
    }
    
    /// Get label ID
    #[inline]
    pub fn label_id(&self) -> usize {
        self.label_id.get() as usize
    }
}

/// Zero-copy graph header structure
#[derive(FromBytes, Immutable, KnownLayout, Clone, Copy, Debug)]
#[repr(C)]
pub struct Header {
    pub magic: le::U32,
    pub version: le::U16,
    pub num_nodes: le::U16,
    pub num_incoming: le::U32,
    pub num_outgoing: le::U32,
    pub num_labels: le::U16,
    pub num_roots: le::U16,
}

impl Header {
    /// Validate the header
    pub fn validate(&self) -> Result<(), GraphError> {
        if self.magic.get() != MAGIC {
            return Err(GraphError::InvalidMagic);
        }
        if self.version.get() != VERSION {
            return Err(GraphError::UnsupportedVersion(self.version.get()));
        }
        Ok(())
    }
}

/// Zero-copy graph that reads directly from a byte slice.
///
/// This struct holds references into the original byte buffer,
/// providing O(1) access to graph structure without allocations.
///
/// # Lifetime
/// The graph borrows the byte slice for its entire lifetime.
/// The slice must remain valid while the graph is in use.
///
/// # Example
/// ```ignore
/// let bytes: &[u8] = /* from Tantivy stored field */;
/// if ZeroCopyGraph::is_valid_format(bytes) {
///     let graph = ZeroCopyGraph::from_bytes(bytes)?;
///     for (target, label) in graph.outgoing(0).unwrap() {
///         println!("Edge to {} with label {}", target, graph.get_label(label).unwrap());
///     }
/// }
/// ```
pub struct ZeroCopyGraph<'a> {
    /// Number of nodes
    num_nodes: usize,
    /// Number of labels in vocabulary
    num_labels: usize,
    /// Incoming edge offsets: offset[i] is start index for node i's edges
    incoming_offsets: &'a [le::U32],
    /// Outgoing edge offsets: offset[i] is start index for node i's edges
    outgoing_offsets: &'a [le::U32],
    /// Packed incoming edges array
    incoming_edges: &'a [PackedEdge],
    /// Packed outgoing edges array
    outgoing_edges: &'a [PackedEdge],
    /// Label string offsets into label_data
    label_offsets: &'a [le::U16],
    /// Concatenated label strings (UTF-8)
    label_data: &'a [u8],
    /// Root node IDs
    roots: &'a [le::U16],
}

impl<'a> ZeroCopyGraph<'a> {
    /// Check if the given bytes start with the zero-copy format magic number.
    ///
    /// This is a quick check to determine which deserializer to use.
    #[inline]
    pub fn is_valid_format(data: &[u8]) -> bool {
        data.len() >= 4 && u32::from_le_bytes([data[0], data[1], data[2], data[3]]) == MAGIC
    }
    
    /// Parse bytes into a zero-copy graph.
    ///
    /// This operation is O(1) - it only validates the header and computes
    /// slice boundaries. No data is copied.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Magic number doesn't match
    /// - Version is unsupported
    /// - Buffer is too small for declared sizes
    /// - Data is misaligned (shouldn't happen with zerocopy)
    pub fn from_bytes(data: &'a [u8]) -> Result<Self, GraphError> {
        // Check minimum size for header
        if data.len() < HEADER_SIZE {
            return Err(GraphError::BufferTooSmall {
                needed: HEADER_SIZE,
                available: data.len(),
            });
        }
        
        // Parse header using zerocopy
        let (header, _) = Header::ref_from_prefix(data)
            .map_err(|_| GraphError::AlignmentError)?;
        header.validate()?;
        
        let num_nodes = header.num_nodes.get() as usize;
        let num_incoming = header.num_incoming.get() as usize;
        let num_outgoing = header.num_outgoing.get() as usize;
        let num_labels = header.num_labels.get() as usize;
        let num_roots = header.num_roots.get() as usize;
        
        // Calculate section sizes
        let incoming_offsets_size = (num_nodes + 1) * 4;
        let outgoing_offsets_size = (num_nodes + 1) * 4;
        let incoming_edges_size = num_incoming * 4;
        let outgoing_edges_size = num_outgoing * 4;
        let label_offsets_size = (num_labels + 1) * 2;
        
        // Calculate section starts
        let incoming_offsets_start = HEADER_SIZE;
        let outgoing_offsets_start = incoming_offsets_start + incoming_offsets_size;
        let incoming_edges_start = outgoing_offsets_start + outgoing_offsets_size;
        let outgoing_edges_start = incoming_edges_start + incoming_edges_size;
        let label_offsets_start = outgoing_edges_start + outgoing_edges_size;
        let label_data_start = label_offsets_start + label_offsets_size;
        
        // We need to read label_offsets to know label_data size
        let label_offsets_end = label_data_start;
        if label_offsets_end > data.len() {
            return Err(GraphError::BufferTooSmall {
                needed: label_offsets_end,
                available: data.len(),
            });
        }
        
        // Parse label offsets to get label data size
        let label_offsets_bytes = &data[label_offsets_start..label_offsets_end];
        let (label_offsets, _) = <[le::U16]>::ref_from_prefix_with_elems(label_offsets_bytes, num_labels + 1)
            .map_err(|_| GraphError::AlignmentError)?;
        
        // Last offset tells us total label data size
        let label_data_size = if num_labels > 0 {
            label_offsets[num_labels].get() as usize
        } else {
            0
        };
        
        let label_data_end = label_data_start + label_data_size;
        let roots_start = label_data_end;
        let roots_size = num_roots * 2;
        let total_size = roots_start + roots_size;
        
        // Final size check
        if total_size > data.len() {
            return Err(GraphError::BufferTooSmall {
                needed: total_size,
                available: data.len(),
            });
        }
        
        // Parse all sections using zerocopy
        let incoming_offsets_bytes = &data[incoming_offsets_start..outgoing_offsets_start];
        let (incoming_offsets, _) = <[le::U32]>::ref_from_prefix_with_elems(incoming_offsets_bytes, num_nodes + 1)
            .map_err(|_| GraphError::AlignmentError)?;
        
        let outgoing_offsets_bytes = &data[outgoing_offsets_start..incoming_edges_start];
        let (outgoing_offsets, _) = <[le::U32]>::ref_from_prefix_with_elems(outgoing_offsets_bytes, num_nodes + 1)
            .map_err(|_| GraphError::AlignmentError)?;
        
        let incoming_edges_bytes = &data[incoming_edges_start..outgoing_edges_start];
        let (incoming_edges, _) = <[PackedEdge]>::ref_from_prefix_with_elems(incoming_edges_bytes, num_incoming)
            .map_err(|_| GraphError::AlignmentError)?;
        
        let outgoing_edges_bytes = &data[outgoing_edges_start..label_offsets_start];
        let (outgoing_edges, _) = <[PackedEdge]>::ref_from_prefix_with_elems(outgoing_edges_bytes, num_outgoing)
            .map_err(|_| GraphError::AlignmentError)?;
        
        let label_data = &data[label_data_start..label_data_end];
        
        let roots_bytes = &data[roots_start..roots_start + roots_size];
        let (roots, _) = <[le::U16]>::ref_from_prefix_with_elems(roots_bytes, num_roots)
            .map_err(|_| GraphError::AlignmentError)?;
        
        Ok(Self {
            num_nodes,
            num_labels,
            incoming_offsets,
            outgoing_offsets,
            incoming_edges,
            outgoing_edges,
            label_offsets,
            label_data,
            roots,
        })
    }
    
    /// Get the raw incoming edges slice for a node (for advanced use).
    #[inline]
    pub fn incoming_edges_raw(&self, node: usize) -> Option<&'a [PackedEdge]> {
        if node >= self.num_nodes {
            return None;
        }
        let start = self.incoming_offsets[node].get() as usize;
        let end = self.incoming_offsets[node + 1].get() as usize;
        Some(&self.incoming_edges[start..end])
    }
    
    /// Get the raw outgoing edges slice for a node (for advanced use).
    #[inline]
    pub fn outgoing_edges_raw(&self, node: usize) -> Option<&'a [PackedEdge]> {
        if node >= self.num_nodes {
            return None;
        }
        let start = self.outgoing_offsets[node].get() as usize;
        let end = self.outgoing_offsets[node + 1].get() as usize;
        Some(&self.outgoing_edges[start..end])
    }
}

impl<'a> GraphAccess for ZeroCopyGraph<'a> {
    #[inline]
    fn node_count(&self) -> usize {
        self.num_nodes
    }
    
    #[inline]
    fn incoming(&self, node: usize) -> Option<impl Iterator<Item = (usize, usize)>> {
        self.incoming_edges_raw(node).map(|edges| {
            edges.iter().map(|e| (e.target(), e.label_id()))
        })
    }
    
    #[inline]
    fn outgoing(&self, node: usize) -> Option<impl Iterator<Item = (usize, usize)>> {
        self.outgoing_edges_raw(node).map(|edges| {
            edges.iter().map(|e| (e.target(), e.label_id()))
        })
    }
    
    fn get_label(&self, label_id: usize) -> Option<&str> {
        if label_id >= self.num_labels {
            return None;
        }
        let start = self.label_offsets[label_id].get() as usize;
        let end = self.label_offsets[label_id + 1].get() as usize;
        // Safety: Labels are validated as UTF-8 at index time
        std::str::from_utf8(&self.label_data[start..end]).ok()
    }
    
    fn get_label_id(&self, label: &str) -> Option<usize> {
        // Linear scan is acceptable - labels are pre-resolved before traversal
        for i in 0..self.num_labels {
            if self.get_label(i) == Some(label) {
                return Some(i);
            }
        }
        None
    }
    
    #[inline]
    fn label_count(&self) -> usize {
        self.num_labels
    }
    
    #[inline]
    fn roots(&self) -> impl Iterator<Item = usize> {
        self.roots.iter().map(|r| r.get() as usize)
    }
    
    #[inline]
    fn root_count(&self) -> usize {
        self.roots.len()
    }
}

impl<'a> ZeroCopyGraph<'a> {
    /// Convert this zero-copy graph to an owned DirectedGraph.
    /// 
    /// This is useful for compatibility with code that expects DirectedGraph,
    /// though it loses the zero-copy benefits. Use this as a migration path
    /// until all code is updated to use the GraphAccess trait.
    pub fn to_directed_graph(&self) -> crate::digraph::graph::DirectedGraph {
        use crate::digraph::graph::DirectedGraph;
        
        let mut graph = DirectedGraph::new();
        
        // Build edges - we need to add them so vocabulary gets built
        for node in 0..self.num_nodes {
            if let Some(edges) = self.outgoing_edges_raw(node) {
                for edge in edges {
                    let target = edge.target() as usize;
                    let label_id = edge.label_id();
                    if let Some(label) = self.get_label(label_id) {
                        graph.add_edge(node, target, label);
                    }
                }
            }
        }
        
        // Set roots
        let roots: Vec<usize> = self.roots().collect();
        graph.set_roots(roots);
        
        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_is_valid_format() {
        // Valid magic
        let valid = [0x47, 0x50, 0x43, 0x5A]; // "ZCPG" in little-endian
        assert!(ZeroCopyGraph::is_valid_format(&valid));
        
        // Invalid magic
        let invalid = [0x00, 0x00, 0x00, 0x00];
        assert!(!ZeroCopyGraph::is_valid_format(&invalid));
        
        // Too short
        let short = [0x47, 0x50];
        assert!(!ZeroCopyGraph::is_valid_format(&short));
    }
    
    #[test]
    fn test_packed_edge() {
        let edge = PackedEdge::new(42, 7);
        assert_eq!(edge.target(), 42);
        assert_eq!(edge.label_id(), 7);
    }
    
    #[test]
    fn test_to_directed_graph() {
        use crate::digraph::zero_copy_writer::ZeroCopyGraphWriter;
        
        // Create a zero-copy graph
        let mut writer = ZeroCopyGraphWriter::new(3);
        let nsubj = writer.add_label("nsubj");
        let dobj = writer.add_label("dobj");
        writer.add_edge(0, 1, nsubj);
        writer.add_edge(0, 2, dobj);
        writer.set_roots(vec![0]);
        
        let bytes = writer.serialize();
        let zc_graph = ZeroCopyGraph::from_bytes(&bytes).unwrap();
        
        // Convert to DirectedGraph
        let directed = zc_graph.to_directed_graph();
        
        // Verify the conversion using DirectedGraph's own methods
        assert_eq!(directed.node_count(), 3);
        
        // Check outgoing edges from node 0 (DirectedGraph returns raw &[usize] slices)
        let edges = directed.outgoing(0).unwrap();
        // edges are stored as flattened (target, label_id) pairs, so 2 edges = 4 elements
        assert_eq!(edges.len(), 4); // 2 edges * 2 elements per edge
        
        // Verify vocabulary was reconstructed
        assert!(directed.vocabulary().get_id("nsubj").is_some());
        assert!(directed.vocabulary().get_id("dobj").is_some());
    }
    
    #[test]
    fn test_header_validation() {
        let mut header = Header {
            magic: le::U32::new(MAGIC),
            version: le::U16::new(VERSION),
            num_nodes: le::U16::new(10),
            num_incoming: le::U32::new(20),
            num_outgoing: le::U32::new(20),
            num_labels: le::U16::new(5),
            num_roots: le::U16::new(1),
        };
        assert!(header.validate().is_ok());
        
        // Invalid magic
        header.magic = le::U32::new(0x12345678);
        assert!(matches!(header.validate(), Err(GraphError::InvalidMagic)));
        
        // Invalid version
        header.magic = le::U32::new(MAGIC);
        header.version = le::U16::new(99);
        assert!(matches!(header.validate(), Err(GraphError::UnsupportedVersion(99))));
    }
}
