//! Graph query utilities for path finding and reachability.
//!
//! This module provides graph analysis functions like shortest path,
//! path existence checks, and reachability queries.

use std::collections::{HashSet, VecDeque, HashMap};

use crate::digraph::graph_trait::GraphAccess;
use super::GraphTraversal;

impl<G: GraphAccess> GraphTraversal<G> {
    /// Get the underlying graph.
    pub fn graph(&self) -> &G {
        &self.graph
    }

    /// Find shortest path between two nodes using BFS.
    pub fn shortest_path(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        if from == to {
            return Some(vec![from]);
        }

        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent = HashMap::new();

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
                for (neighbor, _label_id) in edges {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        parent.insert(neighbor, current);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        None // No path found
    }

    /// Check if there's a path between two nodes.
    pub fn has_path(&self, from: usize, to: usize) -> bool {
        self.shortest_path(from, to).is_some()
    }

    /// Get all nodes reachable from a starting node.
    pub fn reachable_nodes(&self, start: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            if let Some(edges) = self.graph.outgoing(current) {
                for (neighbor, _label_id) in edges {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        visited.into_iter().collect()
    }
}
