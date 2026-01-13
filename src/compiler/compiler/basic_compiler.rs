use tantivy::{
    query::{Query, BooleanQuery, Occur, TermQuery, RegexQuery},
    schema::{Term, Field, Schema},
};
use crate::compiler::ast::{Pattern, Constraint, Matcher, Assertion};
use crate::tantivy_integration::graph_traversal::OptimizedGraphTraversalQuery;
//use crate::tantivy_integration::queries::OptimizedPatternMatchingQuery;
use crate::tantivy_integration::concat_query::RustieConcatQuery;
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
            Pattern::Repetition { pattern, min, max } => {
                self.compile_repetition(pattern, *min, *max)
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
                // Use Term::from_field_text which will use the field's tokenizer
                // For position-aware tokenizer, this should work correctly:
                // - Input "The" gets tokenized by position-aware tokenizer
                // - Tokenizer splits on |, so "The" becomes ["The"]
                // - Term is created for "The" which should match indexed terms
                let term = Term::from_field_text(field, s);
                // Use WithFreqsAndPositions to match the indexing option
                // This ensures we can match terms that were indexed with positions
                Ok(Box::new(TermQuery::new(term, tantivy::schema::IndexRecordOption::WithFreqsAndPositions)))
            }
            Matcher::Regex { pattern, regex } => {
                let regex_query = RegexQuery::from_pattern(pattern, field)
                    .map_err(|e| anyhow!("Invalid regex pattern '{}': {}", pattern, e))?;
                Ok(Box::new(regex_query))
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

    fn compile_concatenated(&self, patterns: &[Pattern]) -> Result<Box<dyn Query>> {
        // Create sub-queries for each pattern
        let mut sub_queries = Vec::new();
        for pattern in patterns {
            let query = self.compile_pattern(pattern)?;
            sub_queries.push(query);
        }
        
        // Create the concatenated pattern
        let concatenated_pattern = Pattern::Concatenated(patterns.to_vec());
        
        // Get the default field
        let default_field = self.schema.get_field("word")
            .map_err(|_| anyhow!("Default field 'word' not found in schema"))?;
        
        // Use RustieConcatQuery for concatenated patterns
        let pattern_query = RustieConcatQuery::new(
            default_field,
            concatenated_pattern,
            sub_queries,
        );
        
        Ok(Box::new(pattern_query))
    }

    fn compile_repetition(&self, pattern: &Pattern, min: usize, max: Option<usize>) -> Result<Box<dyn Query>> {
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