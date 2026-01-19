use crate::types::MatchWithMetadata;
use std::collections::HashMap;

/// Match selector implementing Odinson-style greedy/lazy disambiguation
/// 
/// This module implements the selection algorithm that chooses between
/// competing matches at the same position based on quantifier semantics:
/// - Greedy: prefer longer matches
/// - Lazy: prefer shorter matches
pub struct MatchSelector;

impl MatchSelector {
    /// Select matches from a collection, disambiguating competing matches
    /// at the same starting position.
    /// 
    /// Groups matches by starting position and applies selection algorithm
    /// to each group using fold-right approach (Odinson style).
    pub fn pick_matches(matches: Vec<MatchWithMetadata>) -> Vec<MatchWithMetadata> {
        if matches.is_empty() {
            return Vec::new();
        }

        // Group matches by starting position
        let mut grouped: HashMap<usize, Vec<MatchWithMetadata>> = HashMap::new();
        for m in matches {
            grouped.entry(m.span.start).or_insert_with(Vec::new).push(m);
        }

        let mut result = Vec::new();

        // Process each group of matches at the same starting position
        for (start_pos, mut group) in grouped {
            if group.len() == 1 {
                // No disambiguation needed
                result.push(group.pop().unwrap());
            } else {
                // Apply selection algorithm using fold-right
                // We reverse to simulate fold-right behavior
                group.reverse();
                let selected = group.into_iter().fold(None, |acc, m| {
                    match acc {
                        None => Some(m),
                        Some(current) => {
                            let picked = Self::pick_from_pair(m, current);
                            Some(picked.into_iter().next().unwrap())
                        }
                    }
                });
                if let Some(selected_match) = selected {
                    result.push(selected_match);
                }
            }
        }

        // Sort by start position for consistent output
        result.sort_by_key(|m| m.span.start);
        
        // Post-processing: Handle overlapping matches for greedy patterns
        // For greedy patterns: when two matches overlap, prefer the longer one
        // For lazy patterns: keep all matches even if they overlap
        let mut filtered_result: Vec<MatchWithMetadata> = Vec::new();
        for candidate in &result {
            let candidate_is_greedy = Self::is_greedy_match(candidate);
            let mut is_dominated = false;
            let mut indices_to_remove = Vec::new();
            
            // Check against existing matches
            for (idx, existing) in filtered_result.iter().enumerate() {
                let existing_is_greedy = Self::is_greedy_match(existing);
                
                // Check if matches overlap (not just containment)
                // Two matches overlap if: !(A.end <= B.start || A.start >= B.end)
                let overlaps = !(candidate.span.end <= existing.span.start 
                              || candidate.span.start >= existing.span.end);
                
                if overlaps {
                    if candidate_is_greedy && existing_is_greedy {
                        // Both greedy: prefer longer match
                        if existing.span.length() > candidate.span.length() {
                            // Existing is longer, candidate is dominated
                            is_dominated = true;
                            break;
                        } else if candidate.span.length() > existing.span.length() {
                            // Candidate is longer, mark existing for removal
                            indices_to_remove.push(idx);
                        }
                        // Equal length: keep both (they're at different positions)
                    } else if !candidate_is_greedy && !existing_is_greedy {
                        // Both lazy: keep both (lazy allows multiple short matches)
                        // No action needed
                    }
                    // Mixed greedy/lazy: keep both
                } else {
                    // No overlap: check containment for greedy patterns
                    if candidate_is_greedy || existing_is_greedy {
                        // Check if candidate is completely contained within existing
                        if existing.span.start <= candidate.span.start 
                            && candidate.span.end <= existing.span.end
                            && existing.span.length() > candidate.span.length()
                            && existing_is_greedy {
                            // Existing contains candidate and is greedy: candidate is dominated
                            is_dominated = true;
                            break;
                        }
                        // Check if existing is contained in candidate
                        if candidate.span.start <= existing.span.start
                            && existing.span.end <= candidate.span.end
                            && candidate.span.length() > existing.span.length()
                            && candidate_is_greedy {
                            // Candidate contains existing and is greedy: mark existing for removal
                            indices_to_remove.push(idx);
                        }
                    }
                }
            }
            
            // Remove dominated matches in reverse order to maintain correct indices
            for &idx in indices_to_remove.iter().rev() {
                filtered_result.remove(idx);
            }
            
            // Add candidate if not dominated
            if !is_dominated {
                filtered_result.push(candidate.clone());
            }
        }
        
        filtered_result
    }
    
    /// Check if a match contains greedy quantifiers
    fn is_greedy_match(m: &MatchWithMetadata) -> bool {
        match &m.kind {
            crate::types::MatchKind::Repetition { is_greedy, .. } => *is_greedy,
            crate::types::MatchKind::Optional { is_greedy, .. } => *is_greedy,
            crate::types::MatchKind::Sequence { sub_matches } => {
                // Check if any sub-match is greedy
                sub_matches.iter().any(|sub| Self::is_greedy_match(sub))
            }
            _ => false,
        }
    }

    /// Pick the better match from a pair of matches at the same position.
    /// 
    /// Traverses the match trees to compare matches based on their
    /// quantifier semantics (greedy vs lazy).
    fn pick_from_pair(
        lhs: MatchWithMetadata,
        rhs: MatchWithMetadata,
    ) -> Vec<MatchWithMetadata> {
        // Traverse match trees using stacks
        let mut l_stack = vec![&lhs];
        let mut r_stack = vec![&rhs];

        while let (Some(l), Some(r)) = (l_stack.pop(), r_stack.pop()) {
            match (&l.kind, &r.kind) {
                // Both are repetitions - compare based on greedy/lazy
                (
                    crate::types::MatchKind::Repetition {
                        is_greedy: l_greedy,
                        sub_matches: l_subs,
                    },
                    crate::types::MatchKind::Repetition {
                        is_greedy: r_greedy,
                        sub_matches: r_subs,
                    },
                ) => {
                    if *l_greedy && *r_greedy {
                        // Both greedy: prefer longer
                        if l.span.length() > r.span.length() {
                            return vec![lhs];
                        } else if l.span.length() < r.span.length() {
                            return vec![rhs];
                        }
                        // Equal length: continue traversing sub_matches
                        // Push sub_matches in reverse order for stack-based traversal
                        for sub in l_subs.iter().rev() {
                            l_stack.push(sub);
                        }
                        for sub in r_subs.iter().rev() {
                            r_stack.push(sub);
                        }
                    } else if !l_greedy && !r_greedy {
                        // Both lazy: prefer shorter
                        if l.span.length() < r.span.length() {
                            return vec![lhs];
                        } else if l.span.length() > r.span.length() {
                            return vec![rhs];
                        }
                        // Equal length: continue traversing sub_matches
                        for sub in l_subs.iter().rev() {
                            l_stack.push(sub);
                        }
                        for sub in r_subs.iter().rev() {
                            r_stack.push(sub);
                        }
                    } else {
                        // Mixed greedy/lazy - this shouldn't happen for the same pattern,
                        // but if it does, prefer the one that matches the pattern's quantifier
                        // For now, default to lhs
                        return vec![lhs];
                    }
                }

                // Both are optional matches
                (
                    crate::types::MatchKind::Optional {
                        is_greedy: l_greedy,
                        matched: l_matched,
                    },
                    crate::types::MatchKind::Optional {
                        is_greedy: r_greedy,
                        matched: r_matched,
                    },
                ) => {
                    if *l_greedy && *r_greedy {
                        // Both greedy: prefer matched over unmatched, then longer
                        if *l_matched && !*r_matched {
                            return vec![lhs];
                        } else if !*l_matched && *r_matched {
                            return vec![rhs];
                        } else if *l_matched && *r_matched {
                            // Both matched: prefer longer
                            if l.span.length() > r.span.length() {
                                return vec![lhs];
                            } else if l.span.length() < r.span.length() {
                                return vec![rhs];
                            }
                        }
                        // Equal: continue (but no sub_matches for Optional)
                        return vec![lhs];
                    } else if !l_greedy && !r_greedy {
                        // Both lazy: prefer unmatched over matched, then shorter
                        if !*l_matched && *r_matched {
                            return vec![lhs];
                        } else if *l_matched && !*r_matched {
                            return vec![rhs];
                        } else if *l_matched && *r_matched {
                            // Both matched: prefer shorter
                            if l.span.length() < r.span.length() {
                                return vec![lhs];
                            } else if l.span.length() > r.span.length() {
                                return vec![rhs];
                            }
                        }
                        // Equal: continue
                        return vec![lhs];
                    } else {
                        // Mixed: default to lhs
                        return vec![lhs];
                    }
                }

                // Both are sequences - compare by traversing sub_matches
                // Don't compare sequence lengths directly! Must traverse to find
                // Repetition/Optional nodes which contain the greedy/lazy flags
                (
                    crate::types::MatchKind::Sequence {
                        sub_matches: l_subs,
                    },
                    crate::types::MatchKind::Sequence {
                        sub_matches: r_subs,
                    },
                ) => {
                    // If sub_match counts differ, we can't compare element-by-element
                    if l_subs.len() != r_subs.len() {
                        // Different structure - prefer the one with more sub-matches (more specific)
                        if l_subs.len() > r_subs.len() {
                            return vec![lhs];
                        } else {
                            return vec![rhs];
                        }
                    }
                    
                    // Same structure - traverse sub_matches to find quantifiers
                    // This will eventually reach Repetition/Optional nodes where
                    // greedy/lazy flags determine the preference
                    for sub in l_subs.iter().rev() {
                        l_stack.push(sub);
                    }
                    for sub in r_subs.iter().rev() {
                        r_stack.push(sub);
                    }
                }

                // Both are atoms - atoms at same position are equivalent
                // (atoms are always length 1 for constraints)
                // DO NOT RETURN - continue to compare remaining sub-matches
                // This allows us to reach Repetition nodes where greedy/lazy flags matter
                (
                    crate::types::MatchKind::Atom,
                    crate::types::MatchKind::Atom,
                ) => {
                    // Atoms are equivalent, continue comparing remaining elements
                    // Empty block - loop naturally continues to next iteration
                }

                // Disjunction cases - prefer first clause (leftmost)
                (
                    crate::types::MatchKind::Disjunction { clause_index: l_idx },
                    crate::types::MatchKind::Disjunction { clause_index: r_idx },
                ) => {
                    if *l_idx < *r_idx {
                        return vec![lhs];
                    } else if *l_idx > *r_idx {
                        return vec![rhs];
                    }
                    // Same clause: compare by length
                    if l.span.length() > r.span.length() {
                        return vec![lhs];
                    } else {
                        return vec![rhs];
                    }
                }

                // Mixed types - default to lhs
                _ => {
                    return vec![lhs];
                }
            }
        }

        // Stacks exhausted - matches are equivalent, return first
        vec![lhs]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Span, MatchKind};

    #[test]
    fn test_pick_matches_single_match() {
        let match1 = MatchWithMetadata::atom(Span::new(0, 5), Vec::new());
        let matches = vec![match1.clone()];
        let selected = MatchSelector::pick_matches(matches);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].span, match1.span);
    }

    #[test]
    fn test_pick_matches_empty() {
        let selected = MatchSelector::pick_matches(Vec::new());
        assert!(selected.is_empty());
    }

    #[test]
    fn test_pick_matches_different_positions() {
        let match1 = MatchWithMetadata::atom(Span::new(0, 5), Vec::new());
        let match2 = MatchWithMetadata::atom(Span::new(10, 15), Vec::new());
        let matches = vec![match1.clone(), match2.clone()];
        let selected = MatchSelector::pick_matches(matches);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_pick_from_pair_greedy_repetition() {
        // Both greedy: prefer longer
        let short = MatchWithMetadata::repetition(
            Span::new(0, 5),
            Vec::new(),
            true, // greedy
            vec![MatchWithMetadata::atom(Span::new(0, 5), Vec::new())],
        );
        let long = MatchWithMetadata::repetition(
            Span::new(0, 10),
            Vec::new(),
            true, // greedy
            vec![MatchWithMetadata::atom(Span::new(0, 10), Vec::new())],
        );
        let selected = MatchSelector::pick_from_pair(short.clone(), long.clone());
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].span.length(), 10); // longer match selected
    }

    #[test]
    fn test_pick_from_pair_lazy_repetition() {
        // Both lazy: prefer shorter
        let short = MatchWithMetadata::repetition(
            Span::new(0, 5),
            Vec::new(),
            false, // lazy
            vec![MatchWithMetadata::atom(Span::new(0, 5), Vec::new())],
        );
        let long = MatchWithMetadata::repetition(
            Span::new(0, 10),
            Vec::new(),
            false, // lazy
            vec![MatchWithMetadata::atom(Span::new(0, 10), Vec::new())],
        );
        let selected = MatchSelector::pick_from_pair(short.clone(), long.clone());
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].span.length(), 5); // shorter match selected
    }

    #[test]
    fn test_pick_matches_same_position_greedy() {
        // Multiple greedy matches at same position - should pick longest
        let short = MatchWithMetadata::repetition(
            Span::new(0, 3),
            Vec::new(),
            true, // greedy
            vec![MatchWithMetadata::atom(Span::new(0, 3), Vec::new())],
        );
        let medium = MatchWithMetadata::repetition(
            Span::new(0, 6),
            Vec::new(),
            true, // greedy
            vec![MatchWithMetadata::atom(Span::new(0, 6), Vec::new())],
        );
        let long = MatchWithMetadata::repetition(
            Span::new(0, 9),
            Vec::new(),
            true, // greedy
            vec![MatchWithMetadata::atom(Span::new(0, 9), Vec::new())],
        );
        let matches = vec![short, medium, long.clone()];
        let selected = MatchSelector::pick_matches(matches);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].span.length(), 9); // longest selected
    }

    #[test]
    fn test_pick_matches_same_position_lazy() {
        // Multiple lazy matches at same position - should pick shortest
        let short = MatchWithMetadata::repetition(
            Span::new(0, 3),
            Vec::new(),
            false, // lazy
            vec![MatchWithMetadata::atom(Span::new(0, 3), Vec::new())],
        );
        let medium = MatchWithMetadata::repetition(
            Span::new(0, 6),
            Vec::new(),
            false, // lazy
            vec![MatchWithMetadata::atom(Span::new(0, 6), Vec::new())],
        );
        let long = MatchWithMetadata::repetition(
            Span::new(0, 9),
            Vec::new(),
            false, // lazy
            vec![MatchWithMetadata::atom(Span::new(0, 9), Vec::new())],
        );
        let matches = vec![short.clone(), medium, long];
        let selected = MatchSelector::pick_matches(matches);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].span.length(), 3); // shortest selected
    }
}
