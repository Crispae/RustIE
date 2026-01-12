use tantivy::{
    collector::{Collector, SegmentCollector},
    DocId, Score, SegmentReader,
    Result as TantivyResult,
};
use crate::results::rustie_results::{RustIeResult, RustieDoc};
use crate::types::Span;

/// Custom collector for Odinson queries that collects spans and captures
pub struct OdinsonCollector {
    limit: usize,
}

impl OdinsonCollector {
    pub fn new(limit: usize) -> Self {
        Self { limit }
    }
}

impl Collector for OdinsonCollector {
    type Fruit = RustIeResult;
    type Child = OdinsonSegmentCollector;

    fn for_segment(&self, segment_local_id: u32, _segment: &SegmentReader) -> TantivyResult<Self::Child> {
        Ok(OdinsonSegmentCollector::new(self.limit, segment_local_id))
    }

    fn requires_scoring(&self) -> bool {
        true
    }

    fn merge_fruits(&self, segment_fruits: Vec<Self::Fruit>) -> TantivyResult<Self::Fruit> {
        // Merge results from multiple segments
        let mut all_score_docs = Vec::new();
        let mut total_hits = 0;
        let mut max_score = None;

        for fruit in segment_fruits {
            total_hits += fruit.total_hits;
            all_score_docs.extend(fruit.score_docs);
            
            if let Some(score) = fruit.max_score {
                max_score = max_score.map(|s: Score| s.max(score)).or(Some(score));
            }
        }

        // Sort by score and take top N
        all_score_docs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_score_docs.truncate(self.limit);

        Ok(RustIeResult::new(total_hits, all_score_docs, max_score))
    }
}

/// Segment-level collector for Odinson queries
pub struct OdinsonSegmentCollector {
    limit: usize,
    segment_ord: u32,
    score_docs: Vec<RustieDoc>,
    max_score: Option<Score>,
}

impl OdinsonSegmentCollector {
    pub fn new(limit: usize, segment_ord: u32) -> Self {
        Self {
            limit,
            segment_ord,
            score_docs: Vec::new(),
            max_score: None,
        }
    }
}

impl SegmentCollector for OdinsonSegmentCollector {
    type Fruit = RustIeResult;

    fn collect(&mut self, doc: DocId, score: Score) {
        // Convert segment-local doc ID to global doc address using correct segment_ord
        let doc_address = tantivy::DocAddress::new(self.segment_ord, doc);

        let score_doc = RustieDoc::new(doc_address, score);
        self.score_docs.push(score_doc);

        self.max_score = self.max_score.map(|s| s.max(score)).or(Some(score));
    }

    fn harvest(self) -> Self::Fruit {
        // Sort by score and take top N
        let mut score_docs = self.score_docs;
        score_docs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        score_docs.truncate(self.limit);

        RustIeResult::new(
            score_docs.len(),
            score_docs,
            self.max_score,
        )
    }
}

/// Collector that also collects span information
pub struct OdinsonSpanCollector {
    limit: usize,
}

impl OdinsonSpanCollector {
    pub fn new(limit: usize) -> Self {
        Self { limit }
    }
}

impl Collector for OdinsonSpanCollector {
    type Fruit = RustIeResult;
    type Child = OdinsonSpanSegmentCollector;

    fn for_segment(&self, segment_local_id: u32, _segment: &SegmentReader) -> TantivyResult<Self::Child> {
        Ok(OdinsonSpanSegmentCollector::new(self.limit, segment_local_id))
    }

    fn requires_scoring(&self) -> bool {
        true
    }

    fn merge_fruits(&self, segment_fruits: Vec<Self::Fruit>) -> TantivyResult<Self::Fruit> {
        // Same merge logic as OdinsonCollector
        let mut all_score_docs = Vec::new();
        let mut total_hits = 0;
        let mut max_score = None;

        for fruit in segment_fruits {
            total_hits += fruit.total_hits;
            all_score_docs.extend(fruit.score_docs);
            
            if let Some(score) = fruit.max_score {
                max_score = max_score.map(|s: Score| s.max(score)).or(Some(score));
            }
        }

        all_score_docs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_score_docs.truncate(self.limit);

        Ok(RustIeResult::new(total_hits, all_score_docs, max_score))
    }
}

/// Segment-level collector that also collects spans
pub struct OdinsonSpanSegmentCollector {
    limit: usize,
    segment_ord: u32,
    score_docs: Vec<RustieDoc>,
    max_score: Option<Score>,
    spans: Vec<Span>,
    captures: Vec<(String, Span)>,
}

impl OdinsonSpanSegmentCollector {
    pub fn new(limit: usize, segment_ord: u32) -> Self {
        Self {
            limit,
            segment_ord,
            score_docs: Vec::new(),
            max_score: None,
            spans: Vec::new(),
            captures: Vec::new(),
        }
    }

    pub fn add_span(&mut self, span: Span) {
        self.spans.push(span);
    }

    pub fn add_capture(&mut self, name: String, span: Span) {
        self.captures.push((name, span));
    }
}

impl SegmentCollector for OdinsonSpanSegmentCollector {
    type Fruit = RustIeResult;

    fn collect(&mut self, doc: DocId, score: Score) {
        // Convert segment-local doc ID to global doc address using correct segment_ord
        let doc_address = tantivy::DocAddress::new(self.segment_ord, doc);
        let score_doc = RustieDoc::new(doc_address, score);
        self.score_docs.push(score_doc);

        self.max_score = self.max_score.map(|s| s.max(score)).or(Some(score));
    }

    fn harvest(self) -> Self::Fruit {
        let mut score_docs = self.score_docs;
        score_docs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        score_docs.truncate(self.limit);

        RustIeResult::new(
            score_docs.len(),
            score_docs,
            self.max_score,
        )
    }
} 