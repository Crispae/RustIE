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
                // Count the number of Constraint steps in pattern[0..step_idx] to get the correct index
                // into constraint_fields_and_tokens (which only contains Constraint entries, not Traversals)
                let constraint_idx = pattern[..step_idx].iter()
                    .filter(|s| matches!(s, FlatPatternStep::Constraint(_)))
                    .count();

                if constraint_idx >= constraint_fields_and_tokens.len() {
                    log::debug!("Constraint index {} out of bounds (len={})", constraint_idx, constraint_fields_and_tokens.len());
                    memo[idx] = false; // Backtrack
                    return;
                }
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
        // Use graph's node count for proper memo sizing (fixes potential out-of-bounds)
        let node_count = self.graph.node_count();
        if node_count == 0 || pattern.is_empty() {
            return all_results;
        }
        for &start_node in candidate_nodes {
            // Validate start_node is within bounds
            if start_node >= node_count {
                log::warn!("Start node {} exceeds graph node count {}", start_node, node_count);
                continue;
            }
            let mut memo = vec![false; node_count * pattern.len()];
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
        // Use graph's node count for proper memo sizing (fixes potential out-of-bounds)
        let node_count = self.graph.node_count();
        if node_count == 0 || pattern.is_empty() {
            return false;
        }
        for &start_node in candidate_nodes {
            // Validate start_node is within bounds
            if start_node >= node_count {
                log::warn!("Start node {} exceeds graph node count {}", start_node, node_count);
                continue;
            }
            let mut memo = vec![false; node_count * pattern.len()];
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

    // ==================== Incoming Wildcard Tests ====================

    #[test]
    fn test_incoming_wildcard() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 2, "nsubj");
        graph.add_edge(1, 2, "dobj");

        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(&Traversal::IncomingWildcard, &[2]);

        match result {
            TraversalResult::Success(nodes) => {
                assert!(nodes.contains(&0));
                assert!(nodes.contains(&1));
                assert_eq!(nodes.len(), 2);
            }
            _ => panic!("Expected success"),
        }
    }

    #[test]
    fn test_incoming_wildcard_no_edges() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");

        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(&Traversal::IncomingWildcard, &[0]);  // No incoming edges to node 0

        match result {
            TraversalResult::FailTraversal => {}
            _ => panic!("Expected FailTraversal"),
        }
    }

    // ==================== Disjunctive Traversal Tests ====================

    #[test]
    fn test_disjunctive_traversal() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");
        graph.add_edge(0, 2, "dobj");

        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(
            &Traversal::Disjunctive(vec![
                Traversal::Outgoing(Matcher::String("nsubj".to_string())),
                Traversal::Outgoing(Matcher::String("dobj".to_string())),
            ]),
            &[0]
        );

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
    fn test_disjunctive_traversal_partial_match() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");

        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(
            &Traversal::Disjunctive(vec![
                Traversal::Outgoing(Matcher::String("nsubj".to_string())),
                Traversal::Outgoing(Matcher::String("dobj".to_string())),  // No dobj edge
            ]),
            &[0]
        );

        match result {
            TraversalResult::Success(nodes) => {
                assert!(nodes.contains(&1));
                assert_eq!(nodes.len(), 1);
            }
            _ => panic!("Expected success"),
        }
    }

    // ==================== Optional Traversal Tests ====================

    #[test]
    fn test_optional_traversal_with_match() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");

        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(
            &Traversal::Optional(Box::new(Traversal::Outgoing(Matcher::String("nsubj".to_string())))),
            &[0]
        );

        match result {
            TraversalResult::Success(nodes) => {
                // Should contain both original node and traversed node
                assert!(nodes.contains(&0));
                assert!(nodes.contains(&1));
                assert_eq!(nodes.len(), 2);
            }
            _ => panic!("Expected success"),
        }
    }

    #[test]
    fn test_optional_traversal_without_match() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "dobj");  // No nsubj edge

        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(
            &Traversal::Optional(Box::new(Traversal::Outgoing(Matcher::String("nsubj".to_string())))),
            &[0]
        );

        match result {
            TraversalResult::Success(nodes) => {
                // Should only contain original node
                assert!(nodes.contains(&0));
                assert_eq!(nodes.len(), 1);
            }
            _ => panic!("Expected success"),
        }
    }

    // ==================== Kleene Star Traversal Tests ====================

    #[test]
    fn test_kleene_star_traversal() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");
        graph.add_edge(1, 2, "nsubj");
        graph.add_edge(2, 3, "nsubj");

        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(
            &Traversal::KleeneStar(Box::new(Traversal::Outgoing(Matcher::String("nsubj".to_string())))),
            &[0]
        );

        match result {
            TraversalResult::Success(nodes) => {
                // Should contain all reachable nodes via nsubj
                assert!(nodes.contains(&0));
                assert!(nodes.contains(&1));
                assert!(nodes.contains(&2));
                assert!(nodes.contains(&3));
                assert_eq!(nodes.len(), 4);
            }
            _ => panic!("Expected success"),
        }
    }

    #[test]
    fn test_kleene_star_no_edges() {
        let graph = DirectedGraph::new();

        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(
            &Traversal::KleeneStar(Box::new(Traversal::Outgoing(Matcher::String("nsubj".to_string())))),
            &[0]
        );

        match result {
            TraversalResult::Success(nodes) => {
                // Should still contain start node
                assert!(nodes.contains(&0));
                assert_eq!(nodes.len(), 1);
            }
            _ => panic!("Expected success"),
        }
    }

    // ==================== Shortest Path Tests ====================

    #[test]
    fn test_shortest_path_direct() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");

        let traversal = GraphTraversal::new(graph);
        let path = traversal.shortest_path(0, 1);

        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path, vec![0, 1]);
    }

    #[test]
    fn test_shortest_path_multi_hop() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");
        graph.add_edge(1, 2, "prep");
        graph.add_edge(2, 3, "nmod");

        let traversal = GraphTraversal::new(graph);
        let path = traversal.shortest_path(0, 3);

        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_shortest_path_no_path() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");
        graph.add_edge(2, 3, "dobj");  // Disconnected

        let traversal = GraphTraversal::new(graph);
        let path = traversal.shortest_path(0, 3);

        assert!(path.is_none());
    }

    #[test]
    fn test_shortest_path_same_node() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");

        let traversal = GraphTraversal::new(graph);
        let path = traversal.shortest_path(0, 0);

        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path, vec![0]);
    }

    // ==================== Has Path Tests ====================

    #[test]
    fn test_has_path_exists() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");
        graph.add_edge(1, 2, "prep");

        let traversal = GraphTraversal::new(graph);
        assert!(traversal.has_path(0, 2));
    }

    #[test]
    fn test_has_path_not_exists() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");
        graph.add_edge(3, 4, "dobj");  // Disconnected

        let traversal = GraphTraversal::new(graph);
        assert!(!traversal.has_path(0, 4));
    }

    // ==================== Reachable Nodes Tests ====================

    #[test]
    fn test_reachable_nodes() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");
        graph.add_edge(0, 2, "dobj");
        graph.add_edge(1, 3, "prep");
        graph.add_edge(4, 5, "amod");  // Disconnected from 0

        let traversal = GraphTraversal::new(graph);
        let mut reachable = traversal.reachable_nodes(0);
        reachable.sort();

        assert_eq!(reachable, vec![0, 1, 2, 3]);
        assert!(!reachable.contains(&4));
        assert!(!reachable.contains(&5));
    }

    #[test]
    fn test_reachable_nodes_isolated() {
        let graph = DirectedGraph::new();

        let traversal = GraphTraversal::new(graph);
        let reachable = traversal.reachable_nodes(0);

        assert_eq!(reachable, vec![0]);
    }

    // ==================== Automaton Query Tests ====================

    #[test]
    fn test_automaton_query_empty_pattern() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");

        let traversal = GraphTraversal::new(graph);
        let result = traversal.automaton_query(&[], &[0], &[]);

        assert!(!result);  // Empty pattern should return false
    }

    #[test]
    fn test_automaton_query_empty_graph() {
        let graph = DirectedGraph::new();

        let traversal = GraphTraversal::new(graph);
        let pattern = vec![
            FlatPatternStep::Constraint(crate::compiler::ast::Pattern::Constraint(
                crate::compiler::ast::Constraint::Wildcard
            ))
        ];
        let result = traversal.automaton_query(&pattern, &[0], &[("word".to_string(), vec!["test".to_string()])]);

        assert!(!result);  // Empty graph should return false
    }

    #[test]
    fn test_automaton_query_out_of_bounds_start_node() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");

        let traversal = GraphTraversal::new(graph);
        let pattern = vec![
            FlatPatternStep::Constraint(crate::compiler::ast::Pattern::Constraint(
                crate::compiler::ast::Constraint::Wildcard
            ))
        ];
        // Start node 100 is way out of bounds
        let result = traversal.automaton_query(&pattern, &[100], &[("word".to_string(), vec!["test".to_string()])]);

        assert!(!result);  // Out of bounds start node should be skipped
    }

    #[test]
    fn test_automaton_query_paths_empty_pattern() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");

        let traversal = GraphTraversal::new(graph);
        let paths = traversal.automaton_query_paths(&[], &[0], &[]);

        assert!(paths.is_empty());
    }

    #[test]
    fn test_automaton_query_paths_empty_graph() {
        let graph = DirectedGraph::new();

        let traversal = GraphTraversal::new(graph);
        let pattern = vec![
            FlatPatternStep::Constraint(crate::compiler::ast::Pattern::Constraint(
                crate::compiler::ast::Constraint::Wildcard
            ))
        ];
        let paths = traversal.automaton_query_paths(&pattern, &[0], &[("word".to_string(), vec!["test".to_string()])]);

        assert!(paths.is_empty());
    }

    // ==================== Regex Matcher Tests ====================

    #[test]
    fn test_outgoing_regex_traversal() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");
        graph.add_edge(0, 2, "nmod");
        graph.add_edge(0, 3, "dobj");

        let traversal = GraphTraversal::new(graph);
        // Match all edges starting with 'n'
        let result = traversal.execute(&Traversal::Outgoing(Matcher::regex("^n.*".to_string())), &[0]);

        match result {
            TraversalResult::Success(nodes) => {
                assert!(nodes.contains(&1));  // nsubj
                assert!(nodes.contains(&2));  // nmod
                assert!(!nodes.contains(&3)); // dobj doesn't match
                assert_eq!(nodes.len(), 2);
            }
            _ => panic!("Expected success"),
        }
    }

    // ==================== No Traversal Tests ====================

    #[test]
    fn test_no_traversal() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");

        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(&Traversal::NoTraversal, &[0]);

        match result {
            TraversalResult::NoTraversal => {}
            _ => panic!("Expected NoTraversal"),
        }
    }

    // ==================== Fail Traversal Tests ====================

    #[test]
    fn test_outgoing_traversal_no_match() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");

        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(&Traversal::Outgoing(Matcher::String("nonexistent".to_string())), &[0]);

        match result {
            TraversalResult::FailTraversal => {}
            _ => panic!("Expected FailTraversal"),
        }
    }

    #[test]
    fn test_concatenated_traversal_fail_midway() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");
        // No edge from 1 with label "prep"

        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(
            &Traversal::Concatenated(vec![
                Traversal::Outgoing(Matcher::String("nsubj".to_string())),
                Traversal::Outgoing(Matcher::String("prep".to_string())),
            ]),
            &[0]
        );

        match result {
            TraversalResult::FailTraversal => {}
            _ => panic!("Expected FailTraversal"),
        }
    }

    // ==================== Multiple Start Nodes Tests ====================

    #[test]
    fn test_multiple_start_nodes() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 10, "nsubj");
        graph.add_edge(1, 20, "nsubj");

        let traversal = GraphTraversal::new(graph);
        let result = traversal.execute(&Traversal::Outgoing(Matcher::String("nsubj".to_string())), &[0, 1]);

        match result {
            TraversalResult::Success(nodes) => {
                assert!(nodes.contains(&10));
                assert!(nodes.contains(&20));
                assert_eq!(nodes.len(), 2);
            }
            _ => panic!("Expected success"),
        }
    }

    // ==================== Graph Accessor Tests ====================

    #[test]
    fn test_graph_accessor() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "test");

        let traversal = GraphTraversal::new(graph);
        let graph_ref = traversal.graph();

        assert!(graph_ref.outgoing(0).is_some());
    }
} 