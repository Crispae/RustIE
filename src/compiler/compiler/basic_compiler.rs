use tantivy::{
    query::{Query, BooleanQuery, Occur, TermQuery, RegexQuery},
    schema::{Term, Field, Schema},
};
use crate::compiler::ast::{Pattern, Constraint, Matcher, Assertion};
use crate::tantivy_integration::graph_traversal::OptimizedGraphTraversalQuery;
//use crate::tantivy_integration::queries::OptimizedPatternMatchingQuery;
use crate::tantivy_integration::concat_query::RustieConcatQuery;
use crate::tantivy_integration::boolean_query::RustieOrQuery;
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
            Pattern::NamedCapture { name: _, pattern } => self.compile_pattern(pattern),
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
        match assertion {
            Assertion::SentenceStart => {
                Err(anyhow!("Sentence start queries not yet implemented"))
            }
            Assertion::SentenceEnd => {
                Err(anyhow!("Sentence end queries not yet implemented"))
            }
            Assertion::PositiveLookahead(_pattern) => {
                Err(anyhow!("Lookahead assertions not yet implemented"))
            }
            Assertion::NegativeLookahead(_pattern) => {
                Err(anyhow!("Negative lookahead assertions not yet implemented"))
            }
            Assertion::PositiveLookbehind(_pattern) => {
                Err(anyhow!("Lookbehind assertions not yet implemented"))
            }
            Assertion::NegativeLookbehind(_pattern) => {
                Err(anyhow!("Negative lookbehind assertions not yet implemented"))
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
                let field = self.schema.get_field("word")
                    .map_err(|_| anyhow!("Default field 'word' not found in schema"))?;
                let term = Term::from_field_text(field, matcher);
                Ok(Box::new(TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic)))
            }
            Constraint::Negated(_constraint) => {
                Err(anyhow!("Negated constraints not yet implemented"))
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
                let term = Term::from_field_text(field, s);
                Ok(Box::new(TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic)))
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

    fn compile_repetition(&self, _pattern: &Pattern, _min: usize, _max: Option<usize>) -> Result<Box<dyn Query>> {
        Err(anyhow!("Repetition queries not yet implemented"))
    }
} 