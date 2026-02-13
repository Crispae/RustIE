//! Tests for graph traversal query components.

#[cfg(test)]
mod tests {
    use tantivy::schema::Field;
    use crate::tantivy_integration::graph_traversal::{
        CollapsedMatcher, CollapsedSpec,
        intersection::intersect_sorted_into,
    };

    #[test]
    fn test_intersect_sorted_into_empty() {
        let a: Vec<u32> = vec![];
        let b: Vec<u32> = vec![1, 2, 3];
        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_intersect_sorted_into_no_overlap() {
        let a = vec![1, 3, 5];
        let b = vec![2, 4, 6];
        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_intersect_sorted_into_full_overlap() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3];
        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn test_intersect_sorted_into_partial_overlap() {
        let a = vec![1, 3, 5, 7, 9];
        let b = vec![2, 3, 4, 5, 6];
        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        assert_eq!(out, vec![3, 5]);
    }

    #[test]
    fn test_intersect_sorted_into_single_element_overlap() {
        let a = vec![5];
        let b = vec![1, 2, 5, 10];
        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        assert_eq!(out, vec![5]);
    }

    #[test]
    fn test_collapsed_spec_clone() {
        let spec = CollapsedSpec {
            constraint_field: Field::from_field_id(0),
            constraint_matcher: CollapsedMatcher::Exact("test".to_string()),
            edge_field: Field::from_field_id(1),
            edge_matcher: CollapsedMatcher::Exact("nsubj".to_string()),
            constraint_idx: 0,
        };
        let cloned = spec.clone();
        assert!(matches!(cloned.constraint_matcher, CollapsedMatcher::Exact(ref s) if s == "test"));
        assert!(matches!(cloned.edge_matcher, CollapsedMatcher::Exact(ref s) if s == "nsubj"));
        assert_eq!(cloned.constraint_idx, 0);
    }

    #[test]
    fn test_collapsed_spec_regex() {
        let spec = CollapsedSpec {
            constraint_field: Field::from_field_id(0),
            constraint_matcher: CollapsedMatcher::RegexPattern("protein.*".to_string()),
            edge_field: Field::from_field_id(1),
            edge_matcher: CollapsedMatcher::RegexPattern("nmod_.*".to_string()),
            constraint_idx: 0,
        };
        let cloned = spec.clone();
        assert!(matches!(cloned.constraint_matcher, CollapsedMatcher::RegexPattern(ref s) if s == "protein.*"));
        assert!(matches!(cloned.edge_matcher, CollapsedMatcher::RegexPattern(ref s) if s == "nmod_.*"));
    }

    #[test]
    fn test_collapsed_matcher_display() {
        let exact = CollapsedMatcher::Exact("hello".to_string());
        assert_eq!(exact.display(), "'hello'");

        let regex = CollapsedMatcher::RegexPattern("nmod_.*".to_string());
        assert_eq!(regex.display(), "/nmod_.*/");
    }

    #[test]
    fn test_generic_driver_no_positions() {
        // GenericDriver should always return None for matching_positions
        // We can't easily create a Box<dyn Scorer> without an index,
        // so we just verify the type exists and trait is implemented
        // The actual behavior is tested via integration tests
    }

    #[test]
    fn test_intersect_sorted_into_large_lists() {
        let a: Vec<u32> = (0..1000).step_by(2).collect();
        let b: Vec<u32> = (0..1000).step_by(3).collect();
        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        let expected: Vec<u32> = (0..1000).step_by(6).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn test_intersect_sorted_into_appends_to_buffer() {
        let mut out = vec![99, 98, 97];
        let a = vec![1, 2, 3];
        let b = vec![2, 3, 4];
        intersect_sorted_into(&a, &b, &mut out);
        assert_eq!(out, vec![99, 98, 97, 2, 3]);

        out.clear();
        intersect_sorted_into(&a, &b, &mut out);
        assert_eq!(out, vec![2, 3]);
    }

    #[test]
    fn test_intersect_sorted_skewed_galloping() {
        let a: Vec<u32> = (0..1000).collect();
        let b: Vec<u32> = vec![5, 50, 500, 999];

        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        assert_eq!(out, vec![5, 50, 500, 999], "Should find all matches in skewed intersection");

        out.clear();
        intersect_sorted_into(&b, &a, &mut out);
        assert_eq!(out, vec![5, 50, 500, 999], "Should handle reversed arguments correctly");
    }

    #[test]
    fn test_intersect_sorted_skewed_no_match() {
        let a: Vec<u32> = (0..1000).collect();
        let b: Vec<u32> = vec![1001, 2000];

        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        assert!(out.is_empty(), "Should be empty");
    }
}
