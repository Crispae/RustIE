//! Tests for graph traversal operations.

#[cfg(test)]
mod tests {
    use crate::digraph::traversal::{GraphTraversal, TraversalResult};
    use crate::query::ast::{Traversal, Matcher, FlatPatternStep};
    use crate::digraph::DirectedGraph;

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
        let mut get_token = |_constraint_idx: usize, _position: usize| -> Option<String> { None };
        let allowed_positions: Vec<Option<std::collections::HashSet<u32>>> = vec![];
        let constraint_exact_flags: Vec<bool> = vec![];
        let result = traversal.automaton_query(&[], &[0], &[], &mut get_token, &allowed_positions, &constraint_exact_flags);

        assert!(!result);  // Empty pattern should return false
    }

    #[test]
    fn test_automaton_query_empty_graph() {
        let graph = DirectedGraph::new();

        let traversal = GraphTraversal::new(graph);
        let pattern = vec![
            FlatPatternStep::Constraint(crate::query::ast::Pattern::Constraint(
                crate::query::ast::Constraint::Wildcard
            ))
        ];
        let field_names = vec!["word".to_string()];
        let tokens = vec![vec!["test".to_string()]];
        let mut get_token = |constraint_idx: usize, position: usize| -> Option<String> {
            tokens.get(constraint_idx)?.get(position).cloned()
        };
        let allowed_positions: Vec<Option<std::collections::HashSet<u32>>> = vec![None];
        let constraint_exact_flags: Vec<bool> = vec![false];
        let result = traversal.automaton_query(&pattern, &[0], &field_names, &mut get_token, &allowed_positions, &constraint_exact_flags);

        assert!(!result);  // Empty graph should return false
    }

    #[test]
    fn test_automaton_query_out_of_bounds_start_node() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");

        let traversal = GraphTraversal::new(graph);
        let pattern = vec![
            FlatPatternStep::Constraint(crate::query::ast::Pattern::Constraint(
                crate::query::ast::Constraint::Wildcard
            ))
        ];
        // Start node 100 is way out of bounds
        let field_names = vec!["word".to_string()];
        let tokens = vec![vec!["test".to_string()]];
        let mut get_token = |constraint_idx: usize, position: usize| -> Option<String> {
            tokens.get(constraint_idx)?.get(position).cloned()
        };
        let allowed_positions: Vec<Option<std::collections::HashSet<u32>>> = vec![None];
        let constraint_exact_flags: Vec<bool> = vec![false];
        let result = traversal.automaton_query(&pattern, &[100], &field_names, &mut get_token, &allowed_positions, &constraint_exact_flags);

        assert!(!result);  // Out of bounds start node should be skipped
    }

    #[test]
    fn test_automaton_query_paths_empty_pattern() {
        let mut graph = DirectedGraph::new();
        graph.add_edge(0, 1, "nsubj");

        let traversal = GraphTraversal::new(graph);
        let mut get_token = |_constraint_idx: usize, _position: usize| -> Option<String> { None };
        let allowed_positions: Vec<Option<std::collections::HashSet<u32>>> = vec![];
        let constraint_exact_flags: Vec<bool> = vec![];
        let paths = traversal.automaton_query_paths(&[], &[0], &[], &mut get_token, &allowed_positions, &constraint_exact_flags);

        assert!(paths.is_empty());
    }

    #[test]
    fn test_automaton_query_paths_empty_graph() {
        let graph = DirectedGraph::new();

        let traversal = GraphTraversal::new(graph);
        let pattern = vec![
            FlatPatternStep::Constraint(crate::query::ast::Pattern::Constraint(
                crate::query::ast::Constraint::Wildcard
            ))
        ];
        let field_names = vec!["word".to_string()];
        let tokens = vec![vec!["test".to_string()]];
        let mut get_token = |constraint_idx: usize, position: usize| -> Option<String> {
            tokens.get(constraint_idx)?.get(position).cloned()
        };
        let allowed_positions: Vec<Option<std::collections::HashSet<u32>>> = vec![None];
        let constraint_exact_flags: Vec<bool> = vec![false];
        let paths = traversal.automaton_query_paths(&pattern, &[0], &field_names, &mut get_token, &allowed_positions, &constraint_exact_flags);

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
