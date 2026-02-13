use tantivy::query::{Query, Weight, Scorer, EnableScoring};
use tantivy::{DocId, Score, SegmentReader, Result as TantivyResult, DocSet};
use tantivy::schema::{Field, Value};
use crate::query::ast::Pattern;

/// Query that wraps an inner query and tags its matches with a capture name
#[derive(Debug)]
pub struct RustieNamedCaptureQuery {
    pub inner_query: Box<dyn Query>,
    pub capture_name: String,
    pub pattern: Pattern,
    pub default_field: Field,
}

impl Clone for RustieNamedCaptureQuery {
    fn clone(&self) -> Self {
        RustieNamedCaptureQuery {
            inner_query: self.inner_query.box_clone(),
            capture_name: self.capture_name.clone(),
            pattern: self.pattern.clone(),
            default_field: self.default_field,
        }
    }
}

impl RustieNamedCaptureQuery {
    pub fn new(
        inner_query: Box<dyn Query>,
        capture_name: String,
        pattern: Pattern,
        default_field: Field,
    ) -> Self {
        Self {
            inner_query,
            capture_name,
            pattern,
            default_field,
        }
    }
}

impl Query for RustieNamedCaptureQuery {
    fn weight(&self, scoring: EnableScoring<'_>) -> TantivyResult<Box<dyn Weight>> {
        let inner_weight = self.inner_query.weight(scoring)?;
        
        Ok(Box::new(RustieNamedCaptureWeight {
            inner_weight,
            capture_name: self.capture_name.clone(),
            pattern: self.pattern.clone(),
            default_field: self.default_field,
        }))
    }
}

// QueryClone is automatically implemented by Tantivy for Clone types

struct RustieNamedCaptureWeight {
    inner_weight: Box<dyn Weight>,
    capture_name: String,
    pattern: Pattern,
    default_field: Field,
}

impl Weight for RustieNamedCaptureWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> TantivyResult<Box<dyn Scorer>> {
        let inner_scorer = self.inner_weight.scorer(reader, boost)?;
        
        let mut scorer = RustieNamedCaptureScorer {
            inner_scorer,
            capture_name: self.capture_name.clone(),
            pattern: self.pattern.clone(),
            default_field: self.default_field,
            reader: reader.clone(),
            current_doc_matches: Vec::new(),
            current_doc: None,
        };
        
        // Initialize by advancing inner scorer to first document
        if scorer.inner_scorer.doc() == tantivy::TERMINATED {
             // It might have already started or be empty, try advance if at 0?
             // Actually, usually we let the caller drive the advance, but our wrapper logic
             // needs to be in sync. 
             // IMPORTANT: inner_scorer starts at DOC_ID_Start (uninitialized), so we 
             // don't advance it immediately, we let our advance() call do it.
        }
        
        Ok(Box::new(scorer))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> TantivyResult<tantivy::query::Explanation> {
        self.inner_weight.explain(reader, doc)
    }
}

pub struct RustieNamedCaptureScorer {
    inner_scorer: Box<dyn Scorer>,
    capture_name: String,
    pattern: Pattern,
    default_field: Field,
    reader: SegmentReader,
    current_doc_matches: Vec<crate::types::SpanWithCaptures>,
    current_doc: Option<DocId>,
}

impl Scorer for RustieNamedCaptureScorer {
    fn score(&mut self) -> Score {
        self.inner_scorer.score()
    }
}

impl DocSet for RustieNamedCaptureScorer {
    fn advance(&mut self) -> DocId {
        let doc_id = self.inner_scorer.advance();
        if doc_id != tantivy::TERMINATED {
            self.current_doc = Some(doc_id);
            // Pre-calculate matches for this document
            self.calculate_matches(doc_id);
        } else {
            self.current_doc = None;
        }
        doc_id
    }

    fn doc(&self) -> DocId {
        self.inner_scorer.doc()
    }

    fn size_hint(&self) -> u32 {
        self.inner_scorer.size_hint()
    }
}

impl RustieNamedCaptureScorer {
    fn calculate_matches(&mut self, doc_id: DocId) {
        self.current_doc_matches.clear();
        
        // Strategy:
        // 1. If inner scorer is one of our custom scorers (concatenated, etc), 
        //    we can extract its matches and wrap them.
        // 2. If inner scorer is a basic primitive (term query), 
        //    we scan the document for the pattern.
        
        // Try downcasting to known custom scorers 
        // (This would require known types to be public/accessible)
        // For now, simpler robust approach: SCAN DOCUMENT
        // This ensures correct behavior regardless of inner query optimization
        
        // 1. Get tokens
        let store_reader = match self.reader.get_store_reader(1) {
            Ok(reader) => reader,
            Err(_) => return,
        };
        let doc = match store_reader.get::<tantivy::schema::TantivyDocument>(doc_id) {
            Ok(doc) => doc,
            Err(_) => return,
        };
        
        // 1. Get tokens for the default field
        let field_name = self.reader.schema().get_field_name(self.default_field);
        let tokens = crate::tantivy_integration::utils::extract_field_values(self.reader.schema(), &doc, field_name);

        if tokens.is_empty() {
             return;
        }

        // 2. Find matching positions
        // We use a field cache for consistency, even if it's just one field for now
        let mut field_cache = std::collections::HashMap::new();
        field_cache.insert(field_name.to_string(), tokens.clone());

        let match_positions = self.pattern.extract_matching_positions(field_name, &tokens);

        // 3. Create named captures
        for pos in match_positions {
            let span = crate::types::Span { start: pos, end: pos + 1 };
            
            // Create the capture with OUR name
            let capture = crate::types::NamedCapture::new(
                self.capture_name.clone(), 
                span.clone()
            );
            
            self.current_doc_matches.push(crate::types::SpanWithCaptures::with_captures(
                span, 
                vec![capture]
            ));
        }
    }

    pub fn get_current_doc_matches(&self) -> &[crate::types::SpanWithCaptures] {
        &self.current_doc_matches
    }
}
