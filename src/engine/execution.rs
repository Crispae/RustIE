//! Query execution methods for ExtractorEngine

use crate::engine::constants::*;
use crate::engine::core::ExtractorEngine;
use crate::results::rustie_results::{RustIeResult, SentenceResult};
use crate::tantivy_integration::concat_query::{RustieConcatQuery, RustieConcatWeight};
use crate::tantivy_integration::named_capture_query::{
    RustieNamedCaptureQuery, RustieNamedCaptureScorer,
};
use crate::tantivy_integration::graph_traversal::{
    OptimizedGraphTraversalQuery, OptimizedGraphTraversalWeight,
};
use crate::query::ast::Pattern;
use crate::tantivy_integration::paging_collector::{
    PaginatedSearchResult, PagingCollector, ScoredDoc, SimpleCollector,
};
use crate::types::SearchCursor;
use anyhow::{anyhow, Result};
use log;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tantivy::{
    collector::TopDocs,
    query::{Query, Scorer},
    DocAddress, DocSet, Score,
};

/// Result of a paginated search: total hits, one page of sentence results, optional next cursor.
pub struct PaginatedResult {
    pub total_hits: usize,
    pub sentence_results: Vec<SentenceResult>,
    pub next_cursor: Option<SearchCursor>,
}

/// A sentence result with its DocAddress and score, sortable in the same global order
/// as ScoredDoc (score DESC, segment ASC, doc ASC). Used by execute_graph_traversal_paginated
/// to merge, dedup, skip, and page graph traversal results while preserving cursor compatibility.
#[derive(Clone)]
struct ScoredSentence {
    score: Score,
    address: DocAddress,
    sentence_result: SentenceResult,
}

impl ScoredSentence {
    fn new(score: Score, address: DocAddress, sentence_result: SentenceResult) -> Self {
        Self {
            score,
            address,
            sentence_result,
        }
    }

    fn as_scored_doc(&self) -> ScoredDoc {
        ScoredDoc {
            score: self.score,
            address: self.address,
        }
    }

    fn dedup_key(&self) -> (Arc<str>, Arc<str>) {
        (
            self.sentence_result.document_id.clone(),
            self.sentence_result.sentence_id.clone(),
        )
    }
}

impl PartialEq for ScoredSentence {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.address == other.address
    }
}

impl Eq for ScoredSentence {}

impl PartialOrd for ScoredSentence {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredSentence {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.address.segment_ord.cmp(&other.address.segment_ord))
            .then_with(|| self.address.doc_id.cmp(&other.address.doc_id))
    }
}

impl ExtractorEngine {
    
    /// Execute a query string and return results
    pub fn query(&self, query: &str) -> Result<RustIeResult> {
        self.query_with_limit(query, self.num_docs())
    }

    /// Execute a query string with a limit on results
    pub fn query_with_limit(&self, query: &str, limit: usize) -> Result<RustIeResult> {
        let pattern = self.parser().parse_query(query)?;
        let tantivy_query = self.compiler().compile(query)?;
        self.execute_query(tantivy_query.as_ref(), limit, &pattern)
    }

    /// Execute a query with Odinson-style cursor-based pagination.
    /// Supports simple constraints (collector-based) and graph traversal (best-effort merge-then-page).
    /// Returns an error for Concatenated and NamedCapture patterns.
    pub fn query_paginated(
        &self,
        query: &str,
        page_size: usize,
        after: Option<SearchCursor>,
    ) -> Result<PaginatedResult> {
        let pattern = self.parser().parse_query(query)?;
        let tantivy_query = self.compiler().compile(query)?;

        match &pattern {
            Pattern::GraphTraversal { .. } => {
                self.execute_graph_traversal_paginated(
                    tantivy_query.as_ref(),
                    page_size,
                    after,
                    &pattern,
                )
            }
            Pattern::Concatenated(..) => Err(anyhow!(
                "Paginated search does not support concatenated patterns; use POST /api/v1/query instead"
            )),
            Pattern::NamedCapture { .. } => Err(anyhow!(
                "Paginated search does not support named capture patterns; use POST /api/v1/query instead"
            )),
            _ => self.execute_paginated(
                tantivy_query.as_ref(),
                page_size,
                after,
                &pattern,
            ),
        }
    }

    /// Execute a compiled query with pagination: use paging collector, load only one page of docs.
    fn execute_paginated(
        &self,
        query: &dyn Query,
        page_size: usize,
        after: Option<SearchCursor>,
        pattern: &crate::query::ast::Pattern,
    ) -> Result<PaginatedResult> {
        let searcher = self.reader.searcher();
        let search_result: PaginatedSearchResult = match &after {
            None => searcher
                .search(query, &SimpleCollector::new(page_size))
                .map_err(anyhow::Error::from)?,
            Some(cursor) => searcher
                .search(query, &PagingCollector::new(page_size, cursor.clone()))
                .map_err(anyhow::Error::from)?,
        };

        let needs_word_positions = pattern.references_field(FIELD_WORD);
        let mut sentence_results = Vec::with_capacity(search_result.scored_docs.len());

        for scored_doc in &search_result.scored_docs {
            let doc = self.doc(scored_doc.address)?;
            let mut result = self.extract_sentence_result(&doc, scored_doc.score)?;
            if needs_word_positions {
                let tokens = self.extract_field_values(&doc, FIELD_WORD);
                let match_positions = pattern.extract_matching_positions(FIELD_WORD, &tokens);
                let mut matches = Vec::new();
                for (i, &start) in match_positions.iter().enumerate() {
                    let span = crate::types::Span {
                        start,
                        end: start + 1,
                    };
                    let capture = crate::types::NamedCapture::new(format!("c{}", i), span.clone());
                    matches.push(crate::types::SpanWithCaptures::with_captures(span, vec![capture]));
                }
                result.matches = matches;
            }
            sentence_results.push(result);
        }

        let next_cursor = if !sentence_results.is_empty()
            && sentence_results.len() == page_size
            && search_result.total_hits > page_size
        {
            search_result.scored_docs.last().map(|last| SearchCursor {
                segment_ord: last.address.segment_ord,
                doc_id: last.address.doc_id,
                score: last.score,
            })
        } else {
            None
        };

        Ok(PaginatedResult {
            total_hits: search_result.total_hits,
            sentence_results,
            next_cursor,
        })
    }

    /// Execute a compiled query with the original pattern for match extraction
    pub fn execute_query(
        &self,
        query: &dyn Query,
        limit: usize,
        pattern: &crate::query::ast::Pattern,
    ) -> Result<RustIeResult> {
        match pattern {
            crate::query::ast::Pattern::GraphTraversal { .. } => {
                self.execute_graph_traversal(query, limit, pattern)
            }
            crate::query::ast::Pattern::Concatenated { .. }
            | crate::query::ast::Pattern::Constraint { .. } => {
                self.execute_pattern_matching(query, limit, pattern)
            }
            _ => {
                self.execute_fallback(query, limit, pattern)
            }
        }
    }

    /// Execute graph traversal queries using dependency graph edges
    /// OPTIMIZED: Parallel segment processing + Single-pass collection
    fn execute_graph_traversal(
        &self,
        query: &dyn Query,
        limit: usize,
        _pattern: &crate::query::ast::Pattern,
    ) -> Result<RustIeResult> {

        let searcher = self.reader.searcher();
        let num_segments = searcher.segment_readers().len();

        let graph_query = match query.as_any().downcast_ref::<OptimizedGraphTraversalQuery>() {
            Some(gq) => gq,
            None => {
                return Ok(RustIeResult {
                    total_hits: 0,
                    score_docs: Vec::new(),
                    sentence_results: Vec::new(),
                    max_score: None,
                });
            }
        };

        // Create weight once and share across segments (avoids per-segment schema/prefilter/regex work).
        let weight: Arc<OptimizedGraphTraversalWeight> = Arc::new(
            graph_query
                .concrete_weight_with_cache(self.regex_cache.clone())
                .map_err(anyhow::Error::from)?,
        );

        // PARALLEL: Process all segments concurrently using Rayon
        let segment_results: Vec<(Vec<(SentenceResult, Score)>, u32)> = (0..num_segments)
            .into_par_iter()
            .filter_map(|segment_ord| {
                let segment_reader = searcher.segment_reader(segment_ord as u32);

                let mut scorer = match weight.concrete_scorer(segment_reader, 1.0) {
                    Ok(s) => s,
                    Err(e) => {
                        log::warn!("graph traversal: segment {segment_ord} scorer creation failed: {e}");
                        return None;
                    }
                };

                let mut segment_sentence_results = Vec::new();

                loop {
                    let doc_id = scorer.doc();
                    if doc_id == tantivy::TERMINATED {
                        break;
                    }

                    let score = scorer.score();
                    let doc_address = DocAddress::new(segment_ord as u32, doc_id);

                    let matches = scorer.take_current_doc_matches();

                    if let Ok(doc) = searcher.doc(doc_address) {
                        if let Ok(mut sentence_result) = self.extract_sentence_result(&doc, score) {
                            sentence_result.matches = matches;
                            segment_sentence_results.push((sentence_result, score));
                        }
                    }

                    if scorer.advance() == tantivy::TERMINATED {
                        break;
                    }
                }

                Some((segment_sentence_results, segment_ord as u32))
            })
            .collect();

        
            // MERGE: Combine results from all segments
        let mut all_results: Vec<(SentenceResult, Score)> = segment_results
            .into_iter()
            .flat_map(|(results, _)| results)
            .collect();

        Ok(Self::build_result_from_sentence_results(all_results, limit))
    }

    /// Best-effort paginated execution for graph traversal queries.
    /// Reuses the parallel segment loop from execute_graph_traversal, then sorts by
    /// (score DESC, segment_ord ASC, doc_id ASC), deduplicates by (document_id, sentence_id),
    /// records total_hits, skips past cursor, takes page_size, and builds next_cursor.
    fn execute_graph_traversal_paginated(
        &self,
        query: &dyn Query,
        page_size: usize,
        after: Option<SearchCursor>,
        _pattern: &Pattern,
    ) -> Result<PaginatedResult> {
        let searcher = self.reader.searcher();
        let num_segments = searcher.segment_readers().len();

        let graph_query = match query.as_any().downcast_ref::<OptimizedGraphTraversalQuery>() {
            Some(gq) => gq,
            None => {
                return Ok(PaginatedResult {
                    total_hits: 0,
                    sentence_results: Vec::new(),
                    next_cursor: None,
                });
            }
        };

        let weight: Arc<OptimizedGraphTraversalWeight> = Arc::new(
            graph_query
                .concrete_weight_with_cache(self.regex_cache.clone())
                .map_err(anyhow::Error::from)?,
        );

        let segment_results: Vec<Vec<ScoredSentence>> = (0..num_segments)
            .into_par_iter()
            .filter_map(|segment_ord| {
                let segment_reader = searcher.segment_reader(segment_ord as u32);
                let mut scorer = match weight.concrete_scorer(segment_reader, 1.0) {
                    Ok(s) => s,
                    Err(e) => {
                        log::warn!(
                            "graph traversal paginated: segment {segment_ord} scorer creation failed: {e}"
                        );
                        return None;
                    }
                };

                let mut segment_scored = Vec::new();

                loop {
                    let doc_id = scorer.doc();
                    if doc_id == tantivy::TERMINATED {
                        break;
                    }

                    let score = scorer.score();
                    let doc_address = DocAddress::new(segment_ord as u32, doc_id);
                    let matches = scorer.take_current_doc_matches();

                    if let Ok(doc) = searcher.doc(doc_address) {
                        if let Ok(mut sentence_result) =
                            self.extract_sentence_result(&doc, score)
                        {
                            sentence_result.matches = matches;
                            segment_scored.push(ScoredSentence::new(
                                score,
                                doc_address,
                                sentence_result,
                            ));
                        }
                    }

                    if scorer.advance() == tantivy::TERMINATED {
                        break;
                    }
                }

                Some(segment_scored)
            })
            .collect();

        let mut all_scored: Vec<ScoredSentence> = segment_results
            .into_iter()
            .flat_map(|v| v.into_iter())
            .collect();

        all_scored.sort();

        let mut seen: HashSet<(Arc<str>, Arc<str>)> = HashSet::new();
        all_scored.retain(|ss| seen.insert(ss.dedup_key()));

        let total_hits = all_scored.len();

        let after_index = match &after {
            Some(cursor) => all_scored
                .iter()
                .position(|ss| !ss.as_scored_doc().should_skip(cursor))
                .unwrap_or(all_scored.len()),
            None => 0,
        };

        let page: Vec<ScoredSentence> = all_scored
            .into_iter()
            .skip(after_index)
            .take(page_size)
            .collect();

        let next_cursor =
            if page.len() == page_size && after_index + page_size < total_hits {
                page.last().map(|last| {
                    SearchCursor::new(
                        last.address.segment_ord,
                        last.address.doc_id,
                        last.score,
                    )
                })
            } else {
                None
            };

        let sentence_results = page
            .into_iter()
            .map(|ss| ss.sentence_result)
            .collect();

        Ok(PaginatedResult {
            total_hits,
            sentence_results,
            next_cursor,
        })
    }

    /// Execute pattern matching queries using token sequence matching
    fn execute_pattern_matching(
        &self,
        query: &dyn Query,
        limit: usize,
        pattern: &crate::query::ast::Pattern,
    ) -> Result<RustIeResult> {
        if let Some(pattern_query) = query.as_any().downcast_ref::<RustieConcatQuery>() {
            return self.execute_optimized_pattern_matching(pattern_query, limit);
        }

        if let Some(named_query) = query.as_any().downcast_ref::<RustieNamedCaptureQuery>() {
            return self.execute_named_capture_matching(named_query, limit);
        }

        let searcher = self.reader.searcher();
        let top_docs = searcher
            .search(query, &TopDocs::with_limit(limit))
            .map_err(anyhow::Error::from)?;

        let mut sentence_results = Vec::new();

        for (score, doc_address) in top_docs {
            if let Ok(doc) = self.doc(doc_address) {
                let mut sentence_result = self.extract_sentence_result(&doc, score)?;
                let tokens = self.extract_field_values(&doc, FIELD_WORD);

                let match_positions = pattern.extract_matching_positions(FIELD_WORD, &tokens);

                let mut pattern_matches = Vec::new();
                for (i, &pos) in match_positions.iter().enumerate() {
                    let span = crate::types::Span {
                        start: pos,
                        end: pos + 1,
                    };
                    let capture =
                        crate::types::NamedCapture::new(format!("c{}", i), span.clone());
                    pattern_matches
                        .push(crate::types::SpanWithCaptures::with_captures(span, vec![capture]));
                }

                sentence_result.matches = pattern_matches;
                sentence_results.push(sentence_result);
            }
        }

        let results_with_scores: Vec<(SentenceResult, Score)> = sentence_results
            .into_iter()
            .map(|r| {
                let score = r.score;
                (r, score)
            })
            .collect();

        Ok(Self::build_result_from_sentence_results(results_with_scores, limit))
    }

    /// Execute optimized pattern matching queries using custom scorer
    /// OPTIMIZED: Parallel segment processing
    fn execute_optimized_pattern_matching(
        &self,
        pattern_query: &RustieConcatQuery,
        limit: usize,
    ) -> Result<RustIeResult> {

        let searcher = self.reader.searcher();
        let num_segments = searcher.segment_readers().len();

        let weight: Arc<RustieConcatWeight> = Arc::new(
            pattern_query.concrete_weight(&searcher).map_err(anyhow::Error::from)?,
        );

        let segment_results: Vec<Vec<(SentenceResult, Score)>> = (0..num_segments)
            .into_par_iter()
            .filter_map(|segment_ord| {
                let segment_reader = searcher.segment_reader(segment_ord as u32);

                let mut scorer = match weight.concrete_scorer(segment_reader, 1.0) {
                    Ok(s) => s,
                    Err(e) => {
                        log::warn!("pattern matching: segment {segment_ord} scorer creation failed: {e}");
                        return None;
                    }
                };

                let mut segment_sentence_results = Vec::new();

                loop {
                    let doc_id = scorer.advance();
                    if doc_id == tantivy::TERMINATED {
                        break;
                    }

                    let score = scorer.score();
                    let doc_address = DocAddress::new(segment_ord as u32, doc_id);

                    let matches = scorer.take_current_doc_matches();

                    if let Ok(doc) = searcher.doc(doc_address) {
                        if let Ok(mut sentence_result) = self.extract_sentence_result(&doc, score) {
                            sentence_result.matches = matches;
                            segment_sentence_results.push((sentence_result, score));
                        }
                    }
                }

                Some(segment_sentence_results)
            })
            .collect();

        let mut all_results: Vec<(SentenceResult, Score)> =
            segment_results.into_iter().flatten().collect();

        Ok(Self::build_result_from_sentence_results(all_results, limit))
    }

    /// Execute named capture pattern matching queries using custom scorer.
    /// Creates weight ONCE and caches scorers per segment (avoids O(N) weight+scorer creation).
    fn execute_named_capture_matching(
        &self,
        named_query: &RustieNamedCaptureQuery,
        limit: usize,
    ) -> Result<RustIeResult> {

        let searcher = self.reader.searcher();
        let top_docs = searcher
            .search(named_query, &TopDocs::with_limit(limit))
            .map_err(anyhow::Error::from)?;

        // Create concrete weight ONCE and cache scorers per segment
        let weight = named_query.concrete_weight(&searcher).map_err(anyhow::Error::from)?;
        let mut scorer_cache: HashMap<u32, RustieNamedCaptureScorer> = HashMap::new();

        let mut sentence_results = Vec::new();

        for (score, doc_address) in top_docs {
            if let Ok(doc) = self.doc(doc_address) {
                let mut sentence_result = self.extract_sentence_result(&doc, score)?;

                let segment_ord = doc_address.segment_ord;

                // Get or create concrete scorer for this segment
                let scorer = scorer_cache.entry(segment_ord).or_insert_with(|| {
                    let segment_reader = searcher.segment_reader(segment_ord);
                    weight.concrete_scorer(segment_reader, 1.0).expect("named capture scorer creation")
                });

                sentence_result.matches = scorer.take_current_doc_matches();

                sentence_results.push(sentence_result);
            }
        }

        let results_with_scores: Vec<(SentenceResult, Score)> = sentence_results
            .into_iter()
            .map(|r| {
                let score = r.score;
                (r, score)
            })
            .collect();

        Ok(Self::build_result_from_sentence_results(results_with_scores, limit))
    }

    /// Execute fallback for other pattern types (Assertion, Disjunctive, Repetition)
    fn execute_fallback(
        &self,
        query: &dyn Query,
        limit: usize,
        pattern: &crate::query::ast::Pattern,
    ) -> Result<RustIeResult> {

        let searcher = self.reader.searcher();
        let top_docs = searcher
            .search(query, &TopDocs::with_limit(limit))
            .map_err(anyhow::Error::from)?;

        let mut sentence_results = Vec::new();

        for (score, doc_address) in top_docs {
            if let Ok(doc) = self.doc(doc_address) {
                let mut sentence_result = self.extract_sentence_result(&doc, score)?;
                let tokens = self.extract_field_values(&doc, FIELD_WORD);

                let match_positions = pattern.extract_matching_positions(FIELD_WORD, &tokens);

                let mut fallback_matches = Vec::new();
                for (i, &start) in match_positions.iter().enumerate() {
                    let span = crate::types::Span {
                        start,
                        end: start + 1,
                    };
                    let capture = crate::types::NamedCapture::new(format!("c{}", i), span.clone());
                    fallback_matches
                        .push(crate::types::SpanWithCaptures::with_captures(span, vec![capture]));
                }

                sentence_result.matches = fallback_matches;
                sentence_results.push(sentence_result);
            }
        }

        let results_with_scores: Vec<(SentenceResult, Score)> = sentence_results
            .into_iter()
            .map(|r| {
                let score = r.score;
                (r, score)
            })
            .collect();

        Ok(Self::build_result_from_sentence_results(results_with_scores, limit))
    }

    /// Build result from sentence results with deduplication and max score computation
    fn build_result_from_sentence_results(
        results: Vec<(SentenceResult, Score)>,
        limit: usize,
    ) -> RustIeResult {
        let deduplicated = Self::deduplicate_results(results, limit);
        let max_score = deduplicated.first().map(|r| r.score);

        RustIeResult {
            total_hits: deduplicated.len(),
            score_docs: Vec::new(), // Always empty - sentence_results is the primary data structure
            sentence_results: deduplicated,
            max_score,
        }
    }

    /// Deduplicate results based on (document_id, sentence_id), keeping highest score.
    /// Uses tuple keys instead of format! to avoid heap allocation per result.
    fn deduplicate_results(
        results: Vec<(SentenceResult, Score)>,
        limit: usize,
    ) -> Vec<SentenceResult> {
        let mut seen: HashMap<(Arc<str>, Arc<str>), SentenceResult> = HashMap::new();

        for (result, score) in results {
            let key = (result.document_id.clone(), result.sentence_id.clone());
            match seen.get(&key) {
                Some(existing) => {
                    if score > existing.score {
                        seen.insert(key, result);
                    }
                }
                None => {
                    seen.insert(key, result);
                }
            }
        }

        let mut deduplicated: Vec<SentenceResult> = seen.into_values().collect();
        deduplicated.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        deduplicated.truncate(limit);
        deduplicated
    }
}

#[cfg(test)]
mod graph_pagination_tests {
    use super::*;
    use std::collections::HashMap;

    fn make_sentence_result(
        document_id: &str,
        sentence_id: &str,
        score: f32,
    ) -> SentenceResult {
        SentenceResult::new(
            Arc::from(document_id),
            Arc::from(sentence_id),
            score,
            vec![],
            HashMap::new(),
        )
    }

    fn scored_sentence(
        score: f32,
        seg: u32,
        doc: u32,
        doc_id: &str,
        sent_id: &str,
    ) -> ScoredSentence {
        ScoredSentence::new(
            score,
            DocAddress::new(seg, doc),
            make_sentence_result(doc_id, sent_id, score),
        )
    }

    #[test]
    fn scored_sentence_ordering_matches_scored_doc() {
        let a = scored_sentence(5.0, 0, 10, "d1", "s1");
        let b = scored_sentence(5.0, 0, 20, "d1", "s2");
        let c = scored_sentence(3.0, 1, 5, "d2", "s1");

        let mut items = vec![c.clone(), a.clone(), b.clone()];
        items.sort();

        assert_eq!(items[0].address, DocAddress::new(0, 10));
        assert_eq!(items[1].address, DocAddress::new(0, 20));
        assert_eq!(items[2].address, DocAddress::new(1, 5));
    }

    #[test]
    fn dedup_keeps_first_occurrence() {
        let items = vec![
            scored_sentence(5.0, 0, 10, "doc1", "sent1"),
            scored_sentence(3.0, 0, 20, "doc1", "sent1"),
            scored_sentence(4.0, 1, 5, "doc2", "sent1"),
        ];

        let mut sorted = items;
        sorted.sort();

        let mut seen = HashSet::new();
        sorted.retain(|ss| seen.insert(ss.dedup_key()));

        assert_eq!(sorted.len(), 2);
        assert_eq!(
            sorted[0].dedup_key(),
            (Arc::from("doc1"), Arc::from("sent1"))
        );
        assert!((sorted[0].score - 5.0).abs() < 1e-6);
    }

    #[test]
    fn skip_past_cursor_works() {
        let items = vec![
            scored_sentence(5.0, 0, 10, "d1", "s1"),
            scored_sentence(4.0, 0, 20, "d2", "s1"),
            scored_sentence(3.0, 1, 5, "d3", "s1"),
            scored_sentence(2.0, 1, 10, "d4", "s1"),
        ];

        let cursor = SearchCursor::new(0, 20, 4.0);

        let after_index = items
            .iter()
            .position(|ss| !ss.as_scored_doc().should_skip(&cursor))
            .unwrap_or(items.len());

        let page: Vec<_> = items.into_iter().skip(after_index).take(2).collect();

        assert_eq!(page.len(), 2);
        assert!((page[0].score - 3.0).abs() < 1e-6);
        assert!((page[1].score - 2.0).abs() < 1e-6);
    }

    #[test]
    fn full_graph_pagination_flow() {
        let mut all = vec![
            scored_sentence(5.0, 0, 1, "d1", "s1"),
            scored_sentence(4.5, 1, 1, "d2", "s1"),
            scored_sentence(4.0, 0, 2, "d3", "s1"),
            scored_sentence(3.0, 0, 3, "d4", "s1"),
            scored_sentence(2.0, 1, 2, "d5", "s1"),
        ];
        all.sort();
        let total_hits = all.len();

        let page1: Vec<_> = all.iter().take(2).cloned().collect();
        assert!((page1[0].score - 5.0).abs() < 1e-6);
        assert!((page1[1].score - 4.5).abs() < 1e-6);

        let cursor1 = SearchCursor::new(
            page1[1].address.segment_ord,
            page1[1].address.doc_id,
            page1[1].score,
        );

        let after_idx = all
            .iter()
            .position(|ss| !ss.as_scored_doc().should_skip(&cursor1))
            .unwrap_or(all.len());
        let page2: Vec<_> = all.iter().skip(after_idx).take(2).cloned().collect();
        assert!((page2[0].score - 4.0).abs() < 1e-6);
        assert!((page2[1].score - 3.0).abs() < 1e-6);

        let cursor2 = SearchCursor::new(
            page2[1].address.segment_ord,
            page2[1].address.doc_id,
            page2[1].score,
        );

        let after_idx2 = all
            .iter()
            .position(|ss| !ss.as_scored_doc().should_skip(&cursor2))
            .unwrap_or(all.len());
        let page3: Vec<_> = all.iter().skip(after_idx2).take(2).cloned().collect();
        assert_eq!(page3.len(), 1);
        assert!((page3[0].score - 2.0).abs() < 1e-6);

        let total_seen = page1.len() + page2.len() + page3.len();
        assert_eq!(total_seen, total_hits);
    }
}
