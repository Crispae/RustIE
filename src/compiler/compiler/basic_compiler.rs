use tantivy::{
    query::{Query, BooleanQuery, Occur, TermQuery, RegexQuery},
    schema::{Term, Schema},
};
use crate::compiler::ast::{Pattern, Constraint, Matcher, Assertion, QuantifierKind};
use crate::tantivy_integration::concat_query::{RustieConcatQuery, ConcatPlan, ConcatStep};
use crate::tantivy_integration::boolean_query::RustieOrQuery;
use crate::tantivy_integration::assertion_query::LookaheadQuery;
use crate::tantivy_integration::named_capture_query::RustieNamedCaptureQuery;
use anyhow::{Result, anyhow};

/// Compiler for basic patterns (no graph traversal)
pub struct BasicCompiler {
    schema: Schema,
}

impl BasicCompiler {
    pub fn new(schema: Schema) -> Self {
        Self { schema }
    }

    pub fn compile_pattern(&self, pattern: &Pattern) -> Result<Box<dyn Query>> {
        match pattern {
            
            Pattern::Assertion(assertion) => self.compile_assertion(assertion),
            Pattern::Constraint(constraint) => self.compile_constraint(constraint),
            Pattern::Disjunctive(patterns) => self.compile_disjunctive(patterns),
            Pattern::Concatenated(patterns) => self.compile_concatenated(patterns),
            Pattern::NamedCapture { name, pattern } => {
                let inner_query = self.compile_pattern(pattern)?;
                let default_field = self.schema.get_field("word")
                    .map_err(|_| anyhow!("Default field 'word' not found within schema"))?;
                    
                Ok(Box::new(RustieNamedCaptureQuery::new(
                    inner_query,
                    name.clone(),
                    *pattern.clone(),
                    default_field,
                )))
            }
            Pattern::Mention { arg_name: _, label: _ } => {
                Err(anyhow!("Mention queries not yet implemented"))
            }
            Pattern::GraphTraversal { .. } => {
                Err(anyhow!("Graph traversal patterns should be handled by GraphCompiler"))
            }
            Pattern::Repetition { pattern, min, max, kind } => {
                self.compile_repetition(pattern, *min, *max, *kind)
            }
        }
    }

    fn compile_assertion(&self, assertion: &Assertion) -> Result<Box<dyn Query>> {
        // Get the default field for assertion queries
        let default_field = self.schema.get_field("word")
            .map_err(|_| anyhow!("Default field 'word' not found in schema"))?;

        match assertion {
            Assertion::PositiveLookahead(pattern) => {
                // Positive lookahead - next token must match pattern
                Ok(Box::new(LookaheadQuery::positive_lookahead(pattern.as_ref().clone(), default_field)))
            }
            Assertion::NegativeLookahead(pattern) => {
                // Negative lookahead - next token must NOT match pattern
                Ok(Box::new(LookaheadQuery::negative_lookahead(pattern.as_ref().clone(), default_field)))
            }
            Assertion::PositiveLookbehind(pattern) => {
                // Positive lookbehind - previous token must match pattern
                Ok(Box::new(LookaheadQuery::positive_lookbehind(pattern.as_ref().clone(), default_field)))
            }
            Assertion::NegativeLookbehind(pattern) => {
                // Negative lookbehind - previous token must NOT match pattern
                Ok(Box::new(LookaheadQuery::negative_lookbehind(pattern.as_ref().clone(), default_field)))
            }
        }
    }

    fn compile_constraint(&self, constraint: &Constraint) -> Result<Box<dyn Query>> {
        match constraint {
            Constraint::Wildcard => {
                let field = self.schema.get_field("word")
                    .map_err(|_| anyhow!("Default field 'word' not found in schema"))?;
                let regex_query = RegexQuery::from_pattern(".*", field)
                    .map_err(|e| anyhow!("Invalid wildcard regex pattern: {}", e))?;
                Ok(Box::new(regex_query))
            }
            Constraint::Field { name, matcher } => {
                self.compile_field_constraint(name, matcher)
            }
            Constraint::Fuzzy { name, matcher } => {
                // Get the field
                let field = self.schema.get_field(name)
                    .map_err(|_| anyhow!("Field '{}' not found in schema", name))?;
                
                // Tantivy doesn't have FuzzyTermQuery in the current version
                // We'll use a regex approximation for fuzzy matching
                // This is a simple approach - for better fuzzy matching, consider edit distance
                let fuzzy_pattern = format!(".*{}.*", regex::escape(matcher));
                let regex_query = RegexQuery::from_pattern(&fuzzy_pattern, field)
                    .map_err(|e| anyhow!("Invalid fuzzy pattern '{}': {}", matcher, e))?;
                Ok(Box::new(regex_query))
            }
            Constraint::Negated(constraint) => {
                // Compile inner constraint and wrap with NOT using BooleanQuery
                let inner_query = self.compile_constraint(constraint)?;
                let clauses = vec![(Occur::MustNot, inner_query)];
                Ok(Box::new(BooleanQuery::new(clauses)))
            }
            Constraint::Conjunctive(constraints) => {
                self.compile_conjunctive_constraints(constraints)
            }
            Constraint::Disjunctive(constraints) => {
                self.compile_disjunctive_constraints(constraints)
            }
        }
    }

    fn compile_field_constraint(&self, field_name: &str, matcher: &Matcher) -> Result<Box<dyn Query>> {
        let field = self.schema.get_field(field_name)
            .map_err(|_| anyhow!("Field '{}' not found in schema", field_name))?;

        match matcher {
            Matcher::String(s) => {
                // For position-aware tokenizer fields, we need to create the term correctly
                // The tokenizer splits on |, so when we index "DT|NN|VBZ", it creates terms "DT", "NN", "VBZ"
                // When querying, we need to match these exact terms
                // Term::from_field_text will tokenize the input, so "DT" becomes ["DT"] which should work
                // But let's also try creating the term directly to ensure it matches
                let term = Term::from_field_text(field, s);
                
                // Debug: Print the term to verify it's created correctly
                log::debug!("Created TermQuery for field '{}' with term text '{}', term={:?}", field_name, s, term);
                
                // Use WithFreqsAndPositions to match the indexing option
                // This ensures we can match terms that were indexed with positions
                Ok(Box::new(TermQuery::new(term, tantivy::schema::IndexRecordOption::WithFreqsAndPositions)))
            }
            Matcher::Regex { pattern, regex } => {
                // Strip /.../ delimiters before processing
                let clean_pattern = pattern.trim_start_matches('/').trim_end_matches('/');
                
                // Optimization: If the regex pattern is a simple literal (no special regex chars),
                // use TermQuery instead of RegexQuery for better performance and reliability
                // This handles cases like /ago/ which should match the literal "ago"
                if Self::is_simple_literal_regex(pattern) {
                    // Extract the literal string (remove leading/trailing /)
                    let literal = clean_pattern;
                    let term = Term::from_field_text(field, literal);
                    log::debug!("Optimizing regex pattern '{}' to TermQuery for literal '{}'", pattern, literal);
                    Ok(Box::new(TermQuery::new(term, tantivy::schema::IndexRecordOption::WithFreqsAndPositions)))
                } else {
                    let regex_query = RegexQuery::from_pattern(clean_pattern, field)
                        .map_err(|e| anyhow!("Invalid regex pattern '{}': {}", pattern, e))?;
                    Ok(Box::new(regex_query))
                }
            }
        }
    }

    fn compile_conjunctive_constraints(&self, constraints: &[Constraint]) -> Result<Box<dyn Query>> {
        let mut clauses = Vec::new();
        for constraint in constraints {
            let query = self.compile_constraint(constraint)?;
            clauses.push((Occur::Must, query));
        }
        Ok(Box::new(BooleanQuery::new(clauses)))
    }

    fn compile_disjunctive_constraints(&self, constraints: &[Constraint]) -> Result<Box<dyn Query>> {
        let mut clauses = Vec::new();
        for constraint in constraints {
            let query = self.compile_constraint(constraint)?;
            clauses.push((Occur::Should, query));
        }
        Ok(Box::new(BooleanQuery::new(clauses)))
    }

    fn compile_disjunctive(&self, patterns: &[Pattern]) -> Result<Box<dyn Query>> {
        let mut sub_queries = Vec::new();
        for pattern in patterns {
            let query = self.compile_pattern(pattern)?;
            sub_queries.push(query);
        }
        Ok(Box::new(RustieOrQuery { sub_queries }))
    }

    fn is_mandatory_pattern(&self, pattern: &Pattern) -> bool {
        match pattern {
            Pattern::Repetition { min, .. } => *min > 0,
            Pattern::Assertion(assertion) => {
                match assertion {
                    Assertion::NegativeLookahead(_) | Assertion::NegativeLookbehind(_) => false,
                    _ => true,
                }
            }
            Pattern::NamedCapture { pattern, .. } => self.is_mandatory_pattern(pattern),
            Pattern::Disjunctive(patterns) => {
                // Mandatory only if all alternatives are mandatory
                !patterns.is_empty() && patterns.iter().all(|p| self.is_mandatory_pattern(p))
            }
            Pattern::Concatenated(patterns) => {
                // Mandatory if any part is mandatory
                patterns.iter().any(|p| self.is_mandatory_pattern(p))
            }
            Pattern::Constraint(_) => true,
            _ => true,
        }
    }

    fn compile_concatenated(&self, patterns: &[Pattern]) -> Result<Box<dyn Query>> {
        // Build concat plan for postings-based execution
        let concat_plan = Self::build_concat_plan(patterns);
        
        let mut sub_queries = Vec::new();

        if concat_plan.is_some() {
            // Phase 1 optimization for concat queries:
            // only compile atom constraints as candidate generators; skip gap repetitions entirely.
            for pattern in patterns {
                // Only compile if it's actually a constraint or capture-wrapped constraint.
                // (Avoid compiling gap repetitions or other non-constraints.)
                if Self::as_constraint(pattern).is_some() {
                    // Reject wildcard as atom (shouldn't happen if build_concat_plan succeeded, but defensive)
                    if let Some(c) = Self::as_constraint(pattern) {
                        if !matches!(c, Constraint::Wildcard) {
                            sub_queries.push(self.compile_pattern(pattern)?);
                        }
                    }
                }
            }
        } else {
            // Original behavior for non-concat-plan patterns:
            // include mandatory patterns only (to avoid over-filtering).
            for pattern in patterns {
                if self.is_mandatory_pattern(pattern) {
                    sub_queries.push(self.compile_pattern(pattern)?);
                }
            }
        }
        
        // If no mandatory patterns found, we must add a query that matches all documents
        // that could potentially be a match. A simple "word:* " regex query works as a fallback.
        if sub_queries.is_empty() {
            let field = self.schema.get_field("word")
                .map_err(|_| anyhow!("Default field 'word' not found in schema"))?;
            let all_query = RegexQuery::from_pattern(".*", field)
                .map_err(|e| anyhow!("Invalid fallback regex: {}", e))?;
            sub_queries.push(Box::new(all_query));
        }
        
        // Create the concatenated pattern
        let concatenated_pattern = Pattern::Concatenated(patterns.to_vec());
        
        // Get the default field
        let default_field = self.schema.get_field("word")
            .map_err(|_| anyhow!("Default field 'word' not found in schema"))?;
        
        // Use RustieConcatQuery for concatenated patterns
        let mut pattern_query = RustieConcatQuery::new(
            default_field,
            concatenated_pattern,
            sub_queries,
        );
        
        // Set concat plan if detected
        if let Some(plan) = concat_plan {
            pattern_query.concat_plan = Some(plan);
        }
        
        Ok(Box::new(pattern_query))
    }
    
    /// Helper to extract constraint from pattern (supports NamedCapture wrapper)
    fn as_constraint(p: &Pattern) -> Option<&Constraint> {
        match p {
            Pattern::Constraint(c) => Some(c),
            Pattern::NamedCapture { pattern, .. } => {
                if let Pattern::Constraint(c) = pattern.as_ref() {
                    Some(c)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if a regex pattern is a simple literal (no special regex characters)
    fn is_simple_literal_regex(pattern: &str) -> bool {
        // Remove leading/trailing slashes
        let trimmed = pattern.trim_start_matches('/').trim_end_matches('/');
        // Check if it contains only alphanumeric characters and common word characters
        // (no regex special chars like ^, $, ., *, +, ?, |, [, ], {, }, (, ))
        trimmed.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    }

    /// Helper to detect gap repetition (Repetition(Wildcard))
    fn as_gap(p: &Pattern) -> Option<(usize, Option<usize>, bool)> {
        match p {
            Pattern::Repetition { pattern, min, max, kind } => {
                if matches!(pattern.as_ref(), Pattern::Constraint(Constraint::Wildcard)) {
                    Some((*min, *max, *kind == QuantifierKind::Lazy))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Build a ConcatPlan from patterns array.
    /// Returns None if the pattern cannot be represented as a postings plan
    /// (e.g., contains assertions, nested concats, or wildcard as atom).
    fn build_concat_plan(patterns: &[Pattern]) -> Option<ConcatPlan> {
        let mut steps = Vec::new();
        let mut next_cidx = 0usize;
        let mut prev_was_atom = false;

        for p in patterns {
            if let Some(c) = Self::as_constraint(p) {
                // Reject wildcard as atom for v1 (only allow in gaps)
                if matches!(c, Constraint::Wildcard) {
                    return None;
                }
                // Atom
                if prev_was_atom {
                    // Insert adjacency gap between consecutive atoms
                    steps.push(ConcatStep::Gap { min: 0, max: Some(0), lazy: false });
                }
                steps.push(ConcatStep::Atom { constraint_idx: next_cidx });
                next_cidx += 1;
                prev_was_atom = true;
                continue;
            }

            if let Some((min, max, lazy)) = Self::as_gap(p) {
                // Gap must come after an Atom, and must be followed by an Atom to be usable.
                // We can still record it now; validation later.
                steps.push(ConcatStep::Gap { min, max, lazy });
                prev_was_atom = false;
                continue;
            }

            // Anything else: assertions, nested concats, repetition of non-wildcard, etc.
            // Not supported in postings plan -> return None so you fall back.
            return None;
        }

        // Validate form: must start with Atom, end with Atom, and alternate Atom/Gaps properly
        if steps.is_empty() {
            return None;
        }
        if !matches!(steps.first().unwrap(), ConcatStep::Atom { .. }) {
            return None;
        }
        if !matches!(steps.last().unwrap(), ConcatStep::Atom { .. }) {
            return None;
        }

        // Also reject "Gap Gap" or "Atom Atom" sequences (after normalization)
        for w in steps.windows(2) {
            match (&w[0], &w[1]) {
                (ConcatStep::Atom { .. }, ConcatStep::Gap { .. }) => {}
                (ConcatStep::Gap { .. }, ConcatStep::Atom { .. }) => {}
                _ => {
                    return None;
                }
            }
        }

        Some(ConcatPlan { steps })
    }

    fn compile_repetition(&self, pattern: &Pattern, min: usize, max: Option<usize>, _kind: QuantifierKind) -> Result<Box<dyn Query>> {
        // Maximum expansion for unbounded patterns to prevent performance issues
        const MAX_EXPANSION: usize = 10;
        
        let effective_max = match max {
            Some(m) => m.min(min + MAX_EXPANSION),
            None => min.saturating_add(MAX_EXPANSION),
        };
        
        // Build alternatives for each length from min to effective_max
        let mut alternatives: Vec<Box<dyn Query>> = Vec::new();
        
        for n in min..=effective_max {
            if n == 0 {
                // Zero repetitions means "match empty" - we skip this for now
                // as Tantivy doesn't have a native empty match concept
                // The calling code can handle this separately when needed
                continue;
            }
            
            // Create a concatenation of n copies of the pattern
            let repeated: Vec<Pattern> = (0..n).map(|_| pattern.clone()).collect();
            
            if repeated.len() == 1 {
                // Single pattern - just compile it directly
                let query = self.compile_pattern(&repeated[0])?;
                alternatives.push(query);
            } else {
                // Multiple patterns - use concatenation
                let concat_query = self.compile_concatenated(&repeated)?;
                alternatives.push(concat_query);
            }
        }
        
        // Handle edge cases
        if alternatives.is_empty() {
            // This happens when min=0 and max=0 (match empty)
            // For now, return a wildcard query as a placeholder
            return self.compile_constraint(&Constraint::Wildcard);
        }
        
        if alternatives.len() == 1 {
            // Only one alternative - return it directly
            return Ok(alternatives.pop().unwrap());
        }
        
        // Multiple alternatives - create a disjunction
        Ok(Box::new(RustieOrQuery { sub_queries: alternatives }))
    }
} 