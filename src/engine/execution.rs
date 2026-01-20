//! Query execution methods for ExtractorEngine

use crate::engine::constants::*;
use crate::engine::core::ExtractorEngine;
use crate::query::parser::QueryParser;
use crate::results::rustie_results::{RustIeResult, RustieDoc, SentenceResult};
use crate::tantivy_integration::concat_query::RustieConcatQuery;
use crate::tantivy_integration::named_capture_query::RustieNamedCaptureQuery;
use crate::tantivy_integration::graph_traversal::{
    OptimizedGraphTraversalQuery, OptimizedGraphTraversalScorer,
};
use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use tantivy::{
    collector::TopDocs,
    query::Query,
    DocAddress, Score,
};

impl ExtractorEngine {
    
    /// Execute a query string and return results
    pub fn query(&self, query: &str) -> Result<RustIeResult> {
        self.query_with_limit(query, self.num_docs())
    }

    /// Execute a query string with a limit on results
    pub fn query_with_limit(&self, query: &str, limit: usize) -> Result<RustIeResult> {
        let parser = QueryParser::new(FIELD_WORD.to_string());
        let pattern = parser.parse_query(query)?;

        let tantivy_query = self.compiler().compile(query)?;

        let is_graph_query = tantivy_query
            .as_any()
            .downcast_ref::<OptimizedGraphTraversalQuery>()
            .is_some();

        self.execute_query(tantivy_query.as_ref(), limit, &pattern, is_graph_query)
    }

    /// Execute a compiled query with the original pattern for match extraction
    pub fn execute_query(
        &self,
        query: &dyn Query,
        limit: usize,
        pattern: &crate::query::ast::Pattern,
        is_graph_query: bool,
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

        // PARALLEL: Process all segments concurrently using Rayon
        let segment_results: Vec<(Vec<(SentenceResult, Score)>, u32)> = (0..num_segments)
            .into_par_iter()
            .filter_map(|segment_ord| {
                let segment_reader = searcher.segment_reader(segment_ord as u32);

                let weight = match graph_query.weight(tantivy::query::EnableScoring::Enabled {
                    searcher: &searcher,
                    statistics_provider: &searcher,
                }) {
                    Ok(w) => w,
                    Err(e) => {
                        return None;
                    }
                };

                let mut scorer = match weight.scorer(segment_reader, 1.0) {
                    Ok(s) => s,
                    Err(e) => {
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

                    let matches = if let Some(graph_scorer) =
                        scorer.as_any().downcast_ref::<OptimizedGraphTraversalScorer>()
                    {
                        graph_scorer.get_current_doc_matches().to_vec()
                    } else {
                        Vec::new()
                    };

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

        all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // DEDUPLICATE
        let deduplicated = Self::deduplicate_results(all_results, limit);
        let max_score = deduplicated.first().map(|r| r.score);


        Ok(RustIeResult {
            total_hits: deduplicated.len(),
            score_docs: Vec::new(),
            sentence_results: deduplicated,
            max_score,
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
        let mut score_docs = Vec::new();
        let mut max_score = None;

        for (score, doc_address) in top_docs {
            score_docs.push(RustieDoc::new(doc_address, score));

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

            max_score = max_score.map(|s: Score| s.max(score)).or(Some(score));
        }

        // DEDUPLICATE
        let results_with_scores: Vec<(SentenceResult, Score)> = sentence_results
            .into_iter()
            .map(|r| {
                let score = r.score;
                (r, score)
            })
            .collect();
        let deduplicated = Self::deduplicate_results(results_with_scores, limit);

        Ok(RustIeResult {
            total_hits: deduplicated.len(),
            score_docs,
            sentence_results: deduplicated,
            max_score,
        })
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

        let segment_results: Vec<Vec<(SentenceResult, Score)>> = (0..num_segments)
            .into_par_iter()
            .filter_map(|segment_ord| {
                let segment_reader = searcher.segment_reader(segment_ord as u32);

                let weight = match pattern_query.weight(tantivy::query::EnableScoring::Enabled {
                    searcher: &searcher,
                    statistics_provider: &searcher,
                }) {
                    Ok(w) => w,
                    Err(e) => {
                        return None;
                    }
                };

                let mut scorer = match weight.scorer(segment_reader, 1.0) {
                    Ok(s) => s,
                    Err(e) => {
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

                    let matches = if let Some(pattern_scorer) = scorer
                        .as_any()
                        .downcast_ref::<crate::tantivy_integration::concat_query::RustieConcatScorer>(
                        )
                    {
                        pattern_scorer.get_current_doc_matches().to_vec()
                    } else {
                        Vec::new()
                    };

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

        all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let deduplicated = Self::deduplicate_results(all_results, limit);
        let max_score = deduplicated.first().map(|r| r.score);


        Ok(RustIeResult {
            total_hits: deduplicated.len(),
            score_docs: Vec::new(),
            sentence_results: deduplicated,
            max_score,
        })
    }

    /// Execute named capture pattern matching queries using custom scorer
    fn execute_named_capture_matching(
        &self,
        named_query: &RustieNamedCaptureQuery,
        limit: usize,
    ) -> Result<RustIeResult> {

        let searcher = self.reader.searcher();
        let top_docs = searcher
            .search(named_query, &TopDocs::with_limit(limit))
            .map_err(anyhow::Error::from)?;

        let mut sentence_results = Vec::new();
        let mut max_score = None;

        for (score, doc_address) in top_docs {
            if let Ok(doc) = self.doc(doc_address) {
                let mut sentence_result = self.extract_sentence_result(&doc, score)?;

                let (segment_ord, _) = (doc_address.segment_ord, doc_address.doc_id);
                let segment_reader = searcher.segment_reader(segment_ord);

                let weight = named_query.weight(tantivy::query::EnableScoring::Enabled {
                    searcher: &searcher,
                    statistics_provider: &searcher,
                })?;

                let scorer = weight.scorer(segment_reader, 1.0)?;

                if let Some(named_scorer) = scorer
                    .as_any()
                    .downcast_ref::<crate::tantivy_integration::named_capture_query::RustieNamedCaptureScorer>()
                {
                    let matches = named_scorer.get_current_doc_matches();
                    sentence_result.matches = matches.to_vec();
                } else {
                    sentence_result.matches = Vec::new();
                }

                sentence_results.push(sentence_result);
            }

            max_score = max_score.map(|s: Score| s.max(score)).or(Some(score));
        }

        let results_with_scores: Vec<(SentenceResult, Score)> = sentence_results
            .into_iter()
            .map(|r| {
                let score = r.score;
                (r, score)
            })
            .collect();
        let deduplicated = Self::deduplicate_results(results_with_scores, limit);
        let max_score = deduplicated.first().map(|r| r.score).or(max_score);

        Ok(RustIeResult {
            total_hits: deduplicated.len(),
            score_docs: Vec::new(),
            sentence_results: deduplicated,
            max_score,
        })
    }

    /// Execute fallback for other pattern types
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
        let mut max_score = None;

        for (score, doc_address) in top_docs {
            if let Ok(doc) = self.doc(doc_address) {
                let mut sentence_result = self.extract_sentence_result(&doc, score)?;
                let tokens = self.extract_field_values(&doc, FIELD_WORD);

                let match_positions = pattern.extract_matching_positions(FIELD_WORD, &tokens);

                use rand::{distributions::Alphanumeric, Rng};
                let mut fallback_matches = Vec::new();
                for start in match_positions {
                    let span = crate::types::Span {
                        start,
                        end: start + 1,
                    };
                    let rand_name: String = rand::thread_rng()
                        .sample_iter(&Alphanumeric)
                        .take(8)
                        .map(char::from)
                        .collect();
                    let capture = crate::types::NamedCapture::new(rand_name, span.clone());
                    fallback_matches
                        .push(crate::types::SpanWithCaptures::with_captures(span, vec![capture]));
                }

                sentence_result.matches = fallback_matches;
                sentence_results.push(sentence_result);
            }

            max_score = max_score.map(|s: Score| s.max(score)).or(Some(score));
        }

        Ok(RustIeResult {
            total_hits: sentence_results.len(),
            score_docs: Vec::new(),
            sentence_results,
            max_score,
        })
    }

    /// Deduplicate results based on (document_id, sentence_id), keeping highest score
    fn deduplicate_results(
        results: Vec<(SentenceResult, Score)>,
        limit: usize,
    ) -> Vec<SentenceResult> {
        let mut seen: HashMap<String, SentenceResult> = HashMap::new();

        for (result, score) in results {
            let key = format!("{}:{}", result.document_id, result.sentence_id);
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
