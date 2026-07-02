use tantivy::query::{Query, Weight, EnableScoring, Scorer};
use tantivy::{DocId, Score, SegmentReader, Result as TantivyResult, DocSet};

#[derive(Debug)]
pub struct RustieOrQuery {
    pub sub_queries: Vec<Box<dyn Query>>,
}

impl Clone for RustieOrQuery {
    fn clone(&self) -> Self {
        RustieOrQuery {
            sub_queries: self.sub_queries.iter().map(|q| q.box_clone()).collect(),
        }
    }
}

impl Query for RustieOrQuery {
    fn weight(&self, scoring: EnableScoring<'_>) -> TantivyResult<Box<dyn Weight>> {
        let sub_weights: Vec<Box<dyn Weight>> = self.sub_queries
            .iter()
            .map(|q| q.weight(scoring.clone()))
            .collect::<TantivyResult<Vec<_>>>()?;
        Ok(Box::new(RustieOrWeight { sub_weights }))
    }
}

struct RustieOrWeight {
    sub_weights: Vec<Box<dyn Weight>>,
}

impl Weight for RustieOrWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> TantivyResult<Box<dyn Scorer>> {
        let sub_scorers: Vec<Box<dyn Scorer>> = self.sub_weights
            .iter()
            .map(|w| w.scorer(reader, boost))
            .collect::<TantivyResult<Vec<_>>>()?;
        // Tantivy collectors use doc()-first iteration: sub-scorers start on their
        // first match, so seed current_doc from the minimum sub-scorer doc.
        let mut current_doc = tantivy::TERMINATED;
        for scorer in &sub_scorers {
            let doc = scorer.doc();
            if doc != tantivy::TERMINATED
                && (current_doc == tantivy::TERMINATED || doc < current_doc)
            {
                current_doc = doc;
            }
        }
        Ok(Box::new(RustieOrScorer {
            sub_scorers,
            current_doc: if current_doc == tantivy::TERMINATED {
                None
            } else {
                Some(current_doc)
            },
        }))
    }

    fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> TantivyResult<tantivy::query::Explanation> {
        Ok(tantivy::query::Explanation::new("RustieOrQuery", Score::default()))
    }
}

struct RustieOrScorer {
    sub_scorers: Vec<Box<dyn Scorer>>,
    current_doc: Option<DocId>,
}

impl Scorer for RustieOrScorer {
    fn score(&mut self) -> Score {
        // Return the max score among all sub-scorers at the current doc
        let doc = self.doc();
        self.sub_scorers.iter_mut()
            .filter(|s| s.doc() == doc)
            .map(|s| s.score())
            .fold(Score::default(), |a, b| a.max(b))
    }
}

impl DocSet for RustieOrScorer {
    fn advance(&mut self) -> DocId {
        let current = self.current_doc.unwrap_or(tantivy::TERMINATED);
        if current == tantivy::TERMINATED {
            return tantivy::TERMINATED;
        }

        // Move past the current doc on every sub-scorer that still points at it.
        for scorer in &mut self.sub_scorers {
            if scorer.doc() == current {
                scorer.advance();
            }
        }

        let mut min_doc = tantivy::TERMINATED;
        for scorer in &self.sub_scorers {
            let doc = scorer.doc();
            if doc != tantivy::TERMINATED && (min_doc == tantivy::TERMINATED || doc < min_doc) {
                min_doc = doc;
            }
        }

        if min_doc == tantivy::TERMINATED {
            self.current_doc = None;
        } else {
            self.current_doc = Some(min_doc);
        }
        min_doc
    }

    fn doc(&self) -> DocId {
        self.current_doc.unwrap_or(tantivy::TERMINATED)
    }

    fn size_hint(&self) -> u32 {
        0
    }
}
