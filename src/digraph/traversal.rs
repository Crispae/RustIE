use crate::compiler::ast::{Traversal, Matcher};
use crate::compiler::ast::FlatPatternStep;
use crate::digraph::graph::{DirectedGraph, Vocabulary, LabelMatcher};
use std::collections::{HashSet, VecDeque};

/// Result of a graph traversal (matches Scala implementation)
#[derive(Debug, Clone)]
pub enum TraversalResult {
    NoTraversal,
    FailTraversal,
    Success(Vec<usize>),
}

/// Graph traversal engine for dependency parsing
/// 
/// This implementation exactly matches the Scala GraphTraversal logic:
/// https://github.com/lum-ai/odinson/blob/master/core/src/main/scala/ai/lum/odinson/digraph/GraphTraversal.scala
pub struct GraphTraversal {
    graph: DirectedGraph,
}

impl GraphTraversal {
    pub fn new(graph: DirectedGraph) -> Self {
        Self { graph }
    }

    /// Execute a traversal pattern
    /// Mirrors the Scala GraphTraversal trait and its case classes.
    pub fn execute(&self, traversal: &Traversal, start_nodes: &[usize]) -> TraversalResult {
        log::debug!("Recursive traversal called (AST: {:?})", traversal);
        log::debug!("Traversal {:?} from nodes {:?}", traversal, start_nodes);

        match traversal {
            Traversal::NoTraversal => TraversalResult::NoTraversal,
            Traversal::OutgoingWildcard => self.outgoing_wildcard(start_nodes),
            Traversal::IncomingWildcard => self.incoming_wildcard(start_nodes),
            Traversal::Outgoing(matcher) => self.outgoing_traversal(start_nodes, matcher),
            Traversal::Incoming(matcher) => self.incoming_traversal(start_nodes, matcher),
            Traversal::Concatenated(traversals) => self.concatenated_traversal(start_nodes, traversals),
            Traversal::Disjunctive(traversals) => self.disjunctive_traversal(start_nodes, traversals),
            Traversal::Optional(traversal) => self.optional_traversal(start_nodes, traversal),
            Traversal::KleeneStar(traversal) => self.kleene_star_traversal(start_nodes, traversal),
        }
    }

    /// Traverse all outgoing edges
    /// Scala: OutgoingWildcard
    fn outgoing_wildcard(&self, start_nodes: &[usize]) -> TraversalResult {
        let mut result_nodes = Vec::new();
        for &start_node in start_nodes {
            if let Some(edges) = self.graph.outgoing(start_node) {
                // Iterate by 2 as in Scala: (node, label) pairs
                for i in (0..edges.len()).step_by(2) {
                    if i + 1 < edges.len() {
                        result_nodes.push(edges[i]); // Target node
                    }
                }
            }
        }
        // Deduplicate results as in Scala
        result_nodes.sort_unstable();
        result_nodes.dedup();
        if result_nodes.is_empty() {
            TraversalResult::FailTraversal
        } else {
            TraversalResult::Success(result_nodes)
        }
    }

    /// Traverse all incoming edges
    /// Scala: IncomingWildcard
    fn incoming_wildcard(&self, start_nodes: &[usize]) -> TraversalResult {
        let mut result_nodes = Vec::new();
        for &start_node in start_nodes {
            if let Some(edges) = self.graph.incoming(start_node) {
                // Iterate by 2 as in Scala: (node, label) pairs
                for i in (0..edges.len()).step_by(2) {
                    if i + 1 < edges.len() {
                        result_nodes.push(edges[i]); // Source node
                    }
                }
            }
        }
        // Deduplicate results as in Scala
        result_nodes.sort_unstable();
        result_nodes.dedup();
        if result_nodes.is_empty() {
            TraversalResult::FailTraversal
        } else {
            TraversalResult::Success(result_nodes)
        }
    }

    /// Traverse outgoing edges matching a specific label
    /// Scala: Outgoing(LabelMatcher)
    fn outgoing_traversal(&self, start_nodes: &[usize], matcher: &Matcher) -> TraversalResult {
        let mut result_nodes = Vec::new();
        let vocabulary = self.graph.vocabulary();
        
        for &start_node in start_nodes {
            if let Some(edges) = self.graph.outgoing(start_node) {
                // Iterate by 2 as in Scala: (node, label) pairs
                for i in (0..edges.len()).step_by(2) {
                    if i + 1 < edges.len() {
                        let target_node = edges[i];
                        let label_id = edges[i + 1];
                        
                        // Convert Rust Matcher to Scala-style LabelMatcher
                        let label_matcher = self.convert_matcher(matcher, vocabulary);
                        if label_matcher.matches(label_id, vocabulary) {
                            result_nodes.push(target_node);
                        }
                    }
                }
            }
        }
        // Deduplicate results as in Scala
        result_nodes.sort_unstable();
        result_nodes.dedup();
        if result_nodes.is_empty() {
            TraversalResult::FailTraversal
        } else {
            TraversalResult::Success(result_nodes)
        }
    }

    /// Traverse incoming edges matching a specific label
    /// Scala: Incoming(LabelMatcher)
    fn incoming_traversal(&self, start_nodes: &[usize], matcher: &Matcher) -> TraversalResult {
        let mut result_nodes = Vec::new();
        let vocabulary = self.graph.vocabulary();
        
        for &start_node in start_nodes {
            if let Some(edges) = self.graph.incoming(start_node) {
                // Iterate by 2 as in Scala: (node, label) pairs
                for i in (0..edges.len()).step_by(2) {
                    if i + 1 < edges.len() {
                        let source_node = edges[i];
                        let label_id = edges[i + 1];
                        
                        // Convert Rust Matcher to Scala-style LabelMatcher
                        let label_matcher = self.convert_matcher(matcher, vocabulary);
                        if label_matcher.matches(label_id, vocabulary) {
                            result_nodes.push(source_node);
                        }
                    }
                }
            }
        }
        // Deduplicate results as in Scala
        result_nodes.sort_unstable();
        result_nodes.dedup();
        if result_nodes.is_empty() {
            TraversalResult::FailTraversal
        } else {
            TraversalResult::Success(result_nodes)
        }
    }

    /// Concatenate multiple traversals (sequence)
    /// Scala: Concatenation(List[GraphTraversal])
    fn concatenated_traversal(&self, start_nodes: &[usize], traversals: &[Traversal]) -> TraversalResult {
        let mut current_nodes = start_nodes.to_vec();
        for (i, traversal) in traversals.iter().enumerate() {
            log::debug!("Step {}: Traversal {:?} from nodes {:?}", i, traversal, current_nodes);
            match self.execute(traversal, &current_nodes) {
                TraversalResult::Success(nodes) => {
                    current_nodes = nodes;
                }
                TraversalResult::FailTraversal => {
                    log::debug!("Step {} failed", i);
                    return TraversalResult::FailTraversal;
                }
                TraversalResult::NoTraversal => {
                    // Continue with current nodes (no change)
                }
            }
        }
        // Deduplicate results as in Scala
        current_nodes.sort_unstable();
        current_nodes.dedup();
        TraversalResult::Success(current_nodes)
    }

    /// Disjunctive traversal (union)
    /// Scala: Union(List[GraphTraversal])
    fn disjunctive_traversal(&self, start_nodes: &[usize], traversals: &[Traversal]) -> TraversalResult {
        let mut all_results = Vec::new();
        for traversal in traversals {
            match self.execute(traversal, start_nodes) {
                TraversalResult::Success(nodes) => {
                    all_results.extend(nodes);
                }
                TraversalResult::FailTraversal => {
                    // Continue trying other traversals
                }
                TraversalResult::NoTraversal => {
                    // Continue with current nodes
                    all_results.extend(start_nodes);
                }
            }
        }
        // Deduplicate results as in Scala
        if all_results.is_empty() {
            TraversalResult::FailTraversal
        } else {
            let mut unique_results: Vec<usize> = all_results.into_iter().collect();
            unique_results.sort_unstable();
            unique_results.dedup();
            TraversalResult::Success(unique_results)
        }
    }

    /// Optional traversal (0 or 1 occurrence)
    /// Scala: Optional(GraphTraversal)
    fn optional_traversal(&self, start_nodes: &[usize], traversal: &Traversal) -> TraversalResult {
        match self.execute(traversal, start_nodes) {
            TraversalResult::Success(mut nodes) => {
                // Add start_nodes to the result, deduplicate
                nodes.extend_from_slice(start_nodes);
                nodes.sort_unstable();
                nodes.dedup();
                TraversalResult::Success(nodes)
            },
            TraversalResult::FailTraversal | TraversalResult::NoTraversal => {
                let mut nodes = start_nodes.to_vec();
                nodes.sort_unstable();
                nodes.dedup();
                TraversalResult::Success(nodes)
            }
        }
    }

    /// Kleene star traversal (0 or more occurrences)
    /// Scala: KleeneStar(GraphTraversal)
    fn kleene_star_traversal(&self, start_nodes: &[usize], traversal: &Traversal) -> TraversalResult {
        let mut all_nodes: HashSet<usize> = start_nodes.iter().cloned().collect();
        let mut current_nodes = start_nodes.to_vec();
        let mut visited = HashSet::new();
        
        loop {
            let key = format!("{:?}", current_nodes);
            if visited.contains(&key) {
                break; // Avoid infinite loops
            }
            visited.insert(key);
            
            match self.execute(traversal, &current_nodes) {
                TraversalResult::Success(nodes) => {
                    let mut new_nodes = false;
                    for &node in &nodes {
                        if all_nodes.insert(node) {
                            new_nodes = true;
                        }
                    }
                    if !new_nodes {
                        break; // No new nodes found
                    }
                    current_nodes = nodes;
                }
                TraversalResult::FailTraversal | TraversalResult::NoTraversal => {
                    break;
                }
            }
        }
        // Return all unique nodes reached
        let mut result: Vec<usize> = all_nodes.into_iter().collect();
        result.sort_unstable();
        TraversalResult::Success(result)
    }

    /// Convert Rust Matcher to Scala-style LabelMatcher
    fn convert_matcher(&self, matcher: &Matcher, vocabulary: &Vocabulary) -> LabelMatcher {
        match matcher {
            Matcher::String(s) => {
                if let Some(id) = vocabulary.get_id(s) {
                    LabelMatcher::exact(s.clone(), id)
                } else {
                    LabelMatcher::fail()
                }
            }
            Matcher::Regex { pattern, regex } => {
                LabelMatcher::regex(pattern.clone())
            }
        }
    }

    /// Get the underlying graph
    pub fn graph(&self) -> &DirectedGraph {
        &self.graph
    }

    /// Find shortest path between two nodes
    pub fn shortest_path(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        if from == to {
            return Some(vec![from]);
        }
        
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent = std::collections::HashMap::new();
        
        queue.push_back(from);
        visited.insert(from);
        
        while let Some(current) = queue.pop_front() {
            if current == to {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = current;
                while node != from {
                    path.push(node);
                    node = parent[&node];
                }
                path.push(from);
                path.reverse();
                return Some(path);
            }
            
            // Explore outgoing edges
            if let Some(edges) = self.graph.outgoing(current) {
                for i in (0..edges.len()).step_by(2) {
                    if i + 1 < edges.len() {
                        let neighbor = edges[i];
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            parent.insert(neighbor, current);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }
        
        None // No path found
    }

    /// Check if there's a path between two nodes
    pub fn has_path(&self, from: usize, to: usize) -> bool {
        self.shortest_path(from, to).is_some()
    }

    /// Get all nodes reachable from a starting node
    pub fn reachable_nodes(&self, start: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        queue.push_back(start);
        visited.insert(start);
        
        while let Some(current) = queue.pop_front() {
            if let Some(edges) = self.graph.outgoing(current) {
                for i in (0..edges.len()).step_by(2) {
                    if i + 1 < edges.len() {
                        let neighbor = edges[i];
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }
        
        visited.into_iter().collect()
    }


}

impl GraphTraversal {

    
    /// Automaton-based traversal with early fail and memoization
    pub fn automaton_traverse(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        step_idx: usize,
        memo: &mut Vec<bool>,
        constraint_fields_and_tokens: &[(String, Vec<String>)],
    ) -> bool {
        let mut dummy_path = Vec::new();
        let mut dummy_results = Vec::new();
        self.automaton_traverse_paths(pattern, node, step_idx, memo, constraint_fields_and_tokens, &mut dummy_path, &mut dummy_results);
        !dummy_results.is_empty()
    }

    /// New: Traverse and collect all token index paths for matches
    pub fn automaton_traverse_paths(
        &self,
        pattern: &[FlatPatternStep],
        node: usize,
        step_idx: usize,
        memo: &mut Vec<bool>,
        constraint_fields_and_tokens: &[(String, Vec<String>)],
        path: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
    ) {
        log::debug!("Recursing - node: {}, step_idx: {}, path: {:?}", node, step_idx, path);
        let num_steps = pattern.len();
        let idx = node * num_steps + step_idx;
        if step_idx == pattern.len() {
            log::debug!("Match found! Final path: {:?}", path);
            results.push(path.clone());
            return;
        }
        // Cycle detection: skip if already visiting this node at this step in current path
        if memo[idx] {
            log::debug!("Cycle detected at node: {}, step_idx: {}, path: {:?}", node, step_idx, path);
            return;
        }
        memo[idx] = true; // Mark as visiting
        match &pattern[step_idx] {
            FlatPatternStep::Constraint(constraint_pat) => {
                // Use the correct tokens for this constraint step
                let count = constraint_fields_and_tokens.iter()
                    .take(step_idx + 1)
                    .filter(|(f, _)| !f.is_empty())
                    .count();
                if count == 0 {
                    log::debug!("No valid constraint fields at step {}", step_idx);
                    memo[idx] = false; // Backtrack
                    return;
                }
                let constraint_idx = count - 1;
                let (field_name, tokens) = &constraint_fields_and_tokens[constraint_idx];
                if let crate::compiler::ast::Pattern::Constraint(constraint) = constraint_pat {
                    if let Some(token) = tokens.get(node) {
                        let matches = constraint.matches(field_name, token);
                        if !matches {
                            log::debug!("Failed constraint at node: {}, step_idx: {}, path: {:?}", node, step_idx, path);
                            memo[idx] = false; // Backtrack
                            return;
                        }
                    } else {
                        log::debug!("No token at node: {}, step_idx: {}, path: {:?}", node, step_idx, path);
                        memo[idx] = false; // Backtrack
                        return;
                    }
                } else {
                    log::debug!("Not a constraint pattern at node: {}, step_idx: {}, path: {:?}", node, step_idx, path);
                    memo[idx] = false; // Backtrack
                    return;
                }
                path.push(node);
                log::debug!("Pushed node {} to path: {:?}", node, path);
                self.automaton_traverse_paths(pattern, node, step_idx + 1, memo, constraint_fields_and_tokens, path, results);
                path.pop();
                log::debug!("Popped node, path is now: {:?}", path);
            }
            FlatPatternStep::Traversal(traversal) => {
                match traversal {
                    crate::compiler::ast::Traversal::Optional(inner_traversal) => {
                        
                        // Try skipping the traversal
                        self.automaton_traverse_paths(pattern, node, step_idx + 1, memo, constraint_fields_and_tokens, path, results);
                        // Try taking the traversal
                        match &**inner_traversal {
                            crate::compiler::ast::Traversal::Outgoing(matcher) => {
                                if let Some(edges) = self.graph.outgoing(node) {
                                    let vocabulary = self.graph.vocabulary();
                                    let label_matcher = self.convert_matcher(matcher, vocabulary);
                                    for i in (0..edges.len()).step_by(2) {
                                        if i + 1 < edges.len() {
                                            let target_node = edges[i];
                                            let label_id = edges[i + 1];
                                            if label_matcher.matches(label_id, vocabulary) {
                                                self.automaton_traverse_paths(pattern, target_node, step_idx + 1, memo, constraint_fields_and_tokens, path, results);
                                            }
                                        }
                                    }
                                }
                            }
                            crate::compiler::ast::Traversal::Incoming(matcher) => {
                                if let Some(edges) = self.graph.incoming(node) {
                                    let vocabulary = self.graph.vocabulary();
                                    let label_matcher = self.convert_matcher(matcher, vocabulary);
                                    for i in (0..edges.len()).step_by(2) {
                                        if i + 1 < edges.len() {
                                            let source_node = edges[i];
                                            let label_id = edges[i + 1];
                                            if label_matcher.matches(label_id, vocabulary) {
                                                self.automaton_traverse_paths(pattern, source_node, step_idx + 1, memo, constraint_fields_and_tokens, path, results);
                                            }
                                        }
                                    }
                                }
                            }
                            crate::compiler::ast::Traversal::OutgoingWildcard => {
                                if let Some(edges) = self.graph.outgoing(node) {
                                    for i in (0..edges.len()).step_by(2) {
                                        if i + 1 < edges.len() {
                                            let target_node = edges[i];
                                            self.automaton_traverse_paths(pattern, target_node, step_idx + 1, memo, constraint_fields_and_tokens, path, results);
                                        }
                                    }
                                }
                            }
                            crate::compiler::ast::Traversal::IncomingWildcard => {
                                if let Some(edges) = self.graph.incoming(node) {
                                    for i in (0..edges.len()).step_by(2) {
                                        if i + 1 < edges.len() {
                                            let source_node = edges[i];
                                            self.automaton_traverse_paths(pattern, source_node, step_idx + 1, memo, constraint_fields_and_tokens, path, results);
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    crate::compiler::ast::Traversal::Outgoing(matcher) => {
                        if let Some(edges) = self.graph.outgoing(node) {
                            let vocabulary = self.graph.vocabulary();
                            let label_matcher = self.convert_matcher(matcher, vocabulary);
                            for i in (0..edges.len()).step_by(2) {
                                if i + 1 < edges.len() {
                                    let target_node = edges[i];
                                    let label_id = edges[i + 1];
                                    if label_matcher.matches(label_id, vocabulary) {
                                        self.automaton_traverse_paths(pattern, target_node, step_idx + 1, memo, constraint_fields_and_tokens, path, results);
                                    }
                                }
                            }
                        }
                    }
                    crate::compiler::ast::Traversal::Incoming(matcher) => {
                        if let Some(edges) = self.graph.incoming(node) {
                            let vocabulary = self.graph.vocabulary();
                            let label_matcher = self.convert_matcher(matcher, vocabulary);
                            for i in (0..edges.len()).step_by(2) {
                                if i + 1 < edges.len() {
                                    let source_node = edges[i];
                                    let label_id = edges[i + 1];
                                    if label_matcher.matches(label_id, vocabulary) {
                                        self.automaton_traverse_paths(pattern, source_node, step_idx + 1, memo, constraint_fields_and_tokens, path, results);
                                    }
                                }
                            }
                        }
                    }
                    crate::compiler::ast::Traversal::OutgoingWildcard => {
                        if let Some(edges) = self.graph.outgoing(node) {
                            for i in (0..edges.len()).step_by(2) {
                                if i + 1 < edges.len() {
                                    let target_node = edges[i];
                                    self.automaton_traverse_paths(pattern, target_node, step_idx + 1, memo, constraint_fields_and_tokens, path, results);
                                }
                            }
                        }
                    }
                    crate::compiler::ast::Traversal::IncomingWildcard => {
                        if let Some(edges) = self.graph.incoming(node) {
                            for i in (0..edges.len()).step_by(2) {
                                if i + 1 < edges.len() {
                                    let source_node = edges[i];
                                    self.automaton_traverse_paths(pattern, source_node, step_idx + 1, memo, constraint_fields_and_tokens, path, results);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        memo[idx] = false; // Backtrack: allow other paths through this node
    }

    /// Wrapper: Run automaton traversal for all start nodes matching the first constraint, collect all paths
    pub fn automaton_query_paths(
        &self,
        pattern: &[FlatPatternStep],
        candidate_nodes: &[usize],
        constraint_fields_and_tokens: &[(String, Vec<String>)],
    ) -> Vec<Vec<usize>> {
        let mut all_results = Vec::new();
        for &start_node in candidate_nodes {
            // Use the tokens for the first constraint step for memo size
            let tokens_len = if let Some((_, tokens)) = constraint_fields_and_tokens.get(0) { tokens.len() } else { 0 };
            let mut memo = vec![false; tokens_len * pattern.len()];
            let mut path = Vec::new();
            self.automaton_traverse_paths(pattern, start_node, 0, &mut memo, constraint_fields_and_tokens, &mut path, &mut all_results);
        }
        all_results
    }

    /// Wrapper: Run automaton traversal for all start nodes matching the first constraint
    pub fn automaton_query(
        &self,
        pattern: &[FlatPatternStep],
        candidate_nodes: &[usize],
        constraint_fields_and_tokens: &[(String, Vec<String>)],
    ) -> bool {
        for &start_node in candidate_nodes {
            // Use the tokens for the first constraint step for memo size
            let tokens_len = if let Some((_, tokens)) = constraint_fields_and_tokens.get(0) { tokens.len() } else { 0 };
            let mut memo = vec![false; tokens_len * pattern.len()];
            if self.automaton_traverse(pattern, start_node, 0, &mut memo, constraint_fields_and_tokens) {
                return true;
            }
        }
        false
    }
}






#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ast::{Traversal, Matcher};

    #[test]
    fn test_outgoing_wildcard() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");
        graph.add_edge(0, 2, "dobj");
        graph.add_edge(1, 3, "prep");
        
        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(&Traversal::OutgoingWildcard, &[0]);
        
        match result {
            TraversalResult::Success(nodes) => {
                assert!(nodes.contains(&1));
                assert!(nodes.contains(&2));
                assert_eq!(nodes.len(), 2);
            }
            _ => panic!("Expected success"),
        }
    }

    #[test]
    fn test_outgoing_traversal() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");
        graph.add_edge(0, 2, "dobj");
        graph.add_edge(1, 3, "prep");
        
        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(&Traversal::Outgoing(Matcher::String("nsubj".to_string())), &[0]);
        
        match result {
            TraversalResult::Success(nodes) => {
                assert!(nodes.contains(&1));
                assert!(!nodes.contains(&2));
                assert_eq!(nodes.len(), 1);
            }
            _ => panic!("Expected success"),
        }
    }

    #[test]
    fn test_concatenated_traversal() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");
        graph.add_edge(1, 2, "prep");
        graph.add_edge(2, 3, "nmod");
        
        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(
            &Traversal::Concatenated(vec![
                Traversal::Outgoing(Matcher::String("nsubj".to_string())),
                Traversal::Outgoing(Matcher::String("prep".to_string())),
            ]),
            &[0]
        );
        
        match result {
            TraversalResult::Success(nodes) => {
                assert!(nodes.contains(&2));
                assert_eq!(nodes.len(), 1);
            }
            _ => panic!("Expected success"),
        }
    }

    #[test]
    fn test_incoming_traversal() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");
        graph.add_edge(2, 1, "dobj");
        
        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(&Traversal::Incoming(Matcher::String("nsubj".to_string())), &[1]);
        
        match result {
            TraversalResult::Success(nodes) => {
                assert!(nodes.contains(&0));
                assert!(!nodes.contains(&2));
                assert_eq!(nodes.len(), 1);
            }
            _ => panic!("Expected success"),
        }
    }
} 