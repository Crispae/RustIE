use crate::types::{Span, SpanWithCaptures};
use tantivy::{DocId, Score};
use tantivy::DocSet;
use tantivy::query::Scorer;
use std::collections::HashMap;
use std::fmt;

/// Trait for span-based scorers
pub trait SpanScorer: Scorer + DocSet {
    fn spans(&mut self) -> Vec<Span>; // Should return the spans, containf start, end and the name of the token
    fn spans_with_captures(&mut self) -> Vec<SpanWithCaptures>; // Capture are varaible assigned to constrains.
}

/// Basic span scorer implementation
pub struct BasicSpanScorer {
    doc_id: DocId,
    spans: Vec<Span>,
    captures: HashMap<String, Vec<Span>>,
}

impl BasicSpanScorer {
    pub fn new(doc_id: DocId) -> Self {
        Self {
            doc_id,
            spans: Vec::new(),
            captures: HashMap::new(),
        }
    }

    pub fn add_span(&mut self, span: Span) {
        self.spans.push(span);
    }

    pub fn add_capture(&mut self, name: String, span: Span) {
        self.captures.entry(name).or_insert_with(Vec::new).push(span);
    }
}

impl DocSet for BasicSpanScorer {
    fn advance(&mut self) -> u32 { tantivy::TERMINATED }
    fn doc(&self) -> DocId { self.doc_id }
    fn size_hint(&self) -> u32 { 0 }
}

impl Scorer for BasicSpanScorer {
    fn score(&mut self) -> Score { 1.0 }
}

impl SpanScorer for BasicSpanScorer {
    fn spans(&mut self) -> Vec<Span> {
        self.spans.clone()
    }

    fn spans_with_captures(&mut self) -> Vec<SpanWithCaptures> {
        // Only capture the span (start, end); captures are left empty for now
        self.spans.iter().map(|span| SpanWithCaptures::new(span.clone())).collect()
    }
}

/// Span conjunction scorer for combining multiple span scorers
pub struct ConjunctionSpans {
    scorers: Vec<Box<dyn SpanScorer>>,
    current_doc: Option<DocId>,
}

impl DocSet for ConjunctionSpans {
    fn advance(&mut self) -> u32 { tantivy::TERMINATED }
    fn doc(&self) -> DocId { self.current_doc.unwrap_or(DocId::MAX) }
    fn size_hint(&self) -> u32 { 0 }
}
impl fmt::Debug for ConjunctionSpans {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConjunctionSpans").finish()
    }
}

impl Scorer for ConjunctionSpans {
    fn score(&mut self) -> Score { 1.0 }
}

impl SpanScorer for ConjunctionSpans {
    fn spans(&mut self) -> Vec<Span> { Vec::new() }
    fn spans_with_captures(&mut self) -> Vec<SpanWithCaptures> { Vec::new() }
}

/// Span disjunction scorer for combining multiple span scorers
pub struct DisjunctionSpans {
    scorers: Vec<Box<dyn SpanScorer>>,
    current_doc: Option<DocId>,
}

impl DocSet for DisjunctionSpans {
    fn advance(&mut self) -> u32 { tantivy::TERMINATED }
    fn doc(&self) -> DocId { self.current_doc.unwrap_or(DocId::MAX) }
    fn size_hint(&self) -> u32 { 0 }
}
impl fmt::Debug for DisjunctionSpans {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DisjunctionSpans").finish()
    }
}

impl Scorer for DisjunctionSpans {
    fn score(&mut self) -> Score { 1.0 }
}

impl SpanScorer for DisjunctionSpans {
    fn spans(&mut self) -> Vec<Span> { Vec::new() }
    fn spans_with_captures(&mut self) -> Vec<SpanWithCaptures> { Vec::new() }
}

/// Helper: Find all matching sequences for a concatenated pattern and return the spans of each constraint
/// DEPRECATED: This function is no longer used. Pattern matching now uses the pattern's built-in
/// extract_matching_positions method directly, which is more efficient and consistent with Tantivy's approach.
/// 
/// The function was redundant because:
/// 1. Tantivy's query engine already found matching documents
/// 2. We were re-implementing pattern matching logic that Tantivy already did
/// 3. The pattern's extract_matching_positions method provides the same functionality more efficiently
pub fn find_constraint_spans_in_sequence(pattern: &crate::query::ast::Pattern, tokens: &[String]) -> Vec<Vec<SpanWithCaptures>> {
    // This function is deprecated and should not be used
    // Use pattern.extract_matching_positions() instead
    vec![]
} 

