//! Position-aware prefilter for graph traversal queries (Odinson-style)
//!
//! This module implements a prefilter that uses position postings from the inverted
//! index to quickly eliminate documents where constraints cannot possibly match
//! at positions connected by the required edge labels.
//!
//! The key insight from Odinson:
//! - Instead of just checking if a document contains "B-Gene" and "nsubj" somewhere
//! - Check if there's a position where BOTH the entity matches AND the edge exists
//!
//! This dramatically reduces false positives (from ~2000 to ~50-100 candidates).

use std::collections::{HashMap, HashSet};
use tantivy::schema::{Field, IndexRecordOption};
use tantivy::{DocId, SegmentReader, Term, TERMINATED};
use tantivy::DocSet;  // Import trait for docset iteration
use tantivy::postings::Postings;  // Import trait for postings positions
use log::debug;

/// Position information for a term in a document
#[derive(Debug, Clone)]
pub struct TermPositions {
    /// Document ID
    pub doc_id: DocId,
    /// Positions where the term appears
    pub positions: Vec<u32>,
}

/// Reads position postings from the inverted index for a given term
pub struct PositionPostingsReader<'a> {
    segment_reader: &'a SegmentReader,
}

impl<'a> PositionPostingsReader<'a> {
    pub fn new(segment_reader: &'a SegmentReader) -> Self {
        Self { segment_reader }
    }

    /// Get all documents and their positions for a given term
    ///
    /// Returns a map from doc_id to positions where the term appears
    pub fn get_term_positions(&self, field: Field, term_text: &str) -> HashMap<DocId, Vec<u32>> {
        let mut result = HashMap::new();

        let term = Term::from_field_text(field, term_text);
        let inverted_index = self.segment_reader.inverted_index(field);

        // Check if we can read this inverted index
        let Ok(inverted_index) = inverted_index else {
            debug!("Could not get inverted index for field");
            return result;
        };

        // Read postings with positions
        let postings_opt = inverted_index.read_postings(&term, IndexRecordOption::WithFreqsAndPositions);

        let Ok(Some(mut postings)) = postings_opt else {
            debug!("No postings found for term: {}", term_text);
            return result;
        };

        let mut positions_buffer = Vec::with_capacity(32);

        loop {
            let doc_id = postings.doc();
            if doc_id == TERMINATED {
                break;
            }

            // Skip deleted documents
            if !self.segment_reader.is_deleted(doc_id) {
                // Get positions for this document
                positions_buffer.clear();
                postings.positions(&mut positions_buffer);

                if !positions_buffer.is_empty() {
                    result.insert(doc_id, positions_buffer.clone());
                }
            }

            // Advance to next document
            postings.advance();
        }

        result
    }
}

/// Prefilter result containing candidate documents and their matching positions
#[derive(Debug)]
pub struct PrefilterResult {
    /// Documents that pass the prefilter
    /// Map from doc_id to (src_positions, dst_positions)
    /// where src and dst positions are positions that could potentially match the pattern
    pub candidates: HashMap<DocId, CandidatePositions>,
}

#[derive(Debug, Clone)]
pub struct CandidatePositions {
    /// Positions matching the source constraint (with required edge)
    pub src_positions: Vec<u32>,
    /// Positions matching the destination constraint (with required edge)
    pub dst_positions: Vec<u32>,
}

/// Position-aware prefilter for graph traversal
pub struct PositionPrefilter<'a> {
    segment_reader: &'a SegmentReader,
}

impl<'a> PositionPrefilter<'a> {
    pub fn new(segment_reader: &'a SegmentReader) -> Self {
        Self { segment_reader }
    }

    /// Prefilter documents based on position overlap between constraints and edges
    ///
    /// For a query like `[entity=/B-Gene/] <nsubj [tag=/V.*/]`:
    /// - src_constraint_terms: terms that match B-Gene in entity field
    /// - src_edge_label: "nsubj"
    /// - src_edge_field: incoming_edges (because <nsubj is incoming)
    ///
    /// Returns documents where there exists a position p such that:
    /// - entity[p] matches src_constraint
    /// - incoming_edges[p] contains src_edge_label
    pub fn prefilter_by_position_overlap(
        &self,
        constraint_field: Field,
        constraint_terms: &[String], // Terms to look for in constraint field
        edge_field: Field,
        edge_label: &str,
    ) -> HashSet<DocId> {
        let reader = PositionPostingsReader::new(self.segment_reader);

        // Get positions for edge label
        let edge_positions = reader.get_term_positions(edge_field, edge_label);

        if edge_positions.is_empty() {
            debug!("No documents have edge label: {}", edge_label);
            return HashSet::new();
        }

        let mut matching_docs = HashSet::new();

        // For each constraint term, find docs where constraint position == edge position
        for term in constraint_terms {
            let constraint_positions = reader.get_term_positions(constraint_field, term);

            for (doc_id, constraint_pos) in &constraint_positions {
                if let Some(edge_pos) = edge_positions.get(doc_id) {
                    // Check if any position overlaps
                    let constraint_set: HashSet<_> = constraint_pos.iter().collect();
                    let has_overlap = edge_pos.iter().any(|p| constraint_set.contains(p));

                    if has_overlap {
                        matching_docs.insert(*doc_id);
                    }
                }
            }
        }

        debug!(
            "Position prefilter: {} docs with edge '{}', {} pass overlap check",
            edge_positions.len(),
            edge_label,
            matching_docs.len()
        );

        matching_docs
    }

    /// Advanced prefilter for full graph pattern
    ///
    /// For query `[entity=/B-Gene/] <nsubj [tag=/V.*/] >dobj [entity=/B-Gene/]`:
    /// Checks:
    /// 1. Position p1 matches B-Gene AND has incoming nsubj
    /// 2. Position p2 matches V* AND has outgoing dobj
    /// 3. (Graph traversal will verify p1 and p2 are connected)
    pub fn prefilter_graph_pattern(
        &self,
        src_constraint_field: Field,
        src_constraint_terms: &[String],
        src_edge_field: Field,
        src_edge_label: &str,
        dst_constraint_field: Field,
        dst_constraint_terms: &[String],
        dst_edge_field: Field,
        dst_edge_label: &str,
    ) -> PrefilterResult {
        let reader = PositionPostingsReader::new(self.segment_reader);

        // Get positions for both edge labels
        let src_edge_positions = reader.get_term_positions(src_edge_field, src_edge_label);
        let dst_edge_positions = reader.get_term_positions(dst_edge_field, dst_edge_label);

        let mut candidates = HashMap::new();

        // For each document that has both edge labels
        for doc_id in src_edge_positions.keys() {
            if !dst_edge_positions.contains_key(doc_id) {
                continue;
            }

            let src_edge_pos = src_edge_positions.get(doc_id).unwrap();
            let dst_edge_pos = dst_edge_positions.get(doc_id).unwrap();

            // Find src positions: constraint matches AND edge exists
            let mut valid_src_positions = Vec::new();
            for term in src_constraint_terms {
                let constraint_positions = reader.get_term_positions(src_constraint_field, term);
                if let Some(positions) = constraint_positions.get(doc_id) {
                    for &pos in positions {
                        if src_edge_pos.contains(&pos) {
                            valid_src_positions.push(pos);
                        }
                    }
                }
            }

            // Find dst positions: constraint matches AND edge exists
            let mut valid_dst_positions = Vec::new();
            for term in dst_constraint_terms {
                let constraint_positions = reader.get_term_positions(dst_constraint_field, term);
                if let Some(positions) = constraint_positions.get(doc_id) {
                    for &pos in positions {
                        if dst_edge_pos.contains(&pos) {
                            valid_dst_positions.push(pos);
                        }
                    }
                }
            }

            // Only include document if both src and dst have valid positions
            if !valid_src_positions.is_empty() && !valid_dst_positions.is_empty() {
                candidates.insert(
                    *doc_id,
                    CandidatePositions {
                        src_positions: valid_src_positions,
                        dst_positions: valid_dst_positions,
                    },
                );
            }
        }

        debug!(
            "Graph pattern prefilter: {} candidates after position overlap check",
            candidates.len()
        );

        PrefilterResult { candidates }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests would require a Tantivy index with position data
    // These are placeholder tests for the data structures

    #[test]
    fn test_candidate_positions() {
        let candidate = CandidatePositions {
            src_positions: vec![1, 3, 5],
            dst_positions: vec![2, 4],
        };

        assert_eq!(candidate.src_positions.len(), 3);
        assert_eq!(candidate.dst_positions.len(), 2);
    }

    #[test]
    fn test_prefilter_result() {
        let mut candidates = HashMap::new();
        candidates.insert(
            42,
            CandidatePositions {
                src_positions: vec![1],
                dst_positions: vec![3],
            },
        );

        let result = PrefilterResult { candidates };
        assert_eq!(result.candidates.len(), 1);
        assert!(result.candidates.contains_key(&42));
    }
}
