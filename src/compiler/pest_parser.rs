use pest::Parser;
use pest_derive::Parser;
use crate::compiler::ast::{Pattern, Constraint, Matcher};

#[derive(Parser)]
#[grammar = "query.pest"]
pub struct QueryParser;

// Delegating AST builder
pub fn build_ast(pair: pest::iterators::Pair<Rule>) -> Pattern {
    match pair.as_rule() {
        Rule::query | Rule::traversal_seq => {
            let mut pairs = pair.into_inner();
            let mut src = build_ast(pairs.next().unwrap());
            let mut pairs = pairs.peekable();
            while let Some(trav_pair) = pairs.next() {
                if trav_pair.as_rule() == Rule::traversal {
                    let mut trav_inner = trav_pair.into_inner();
                    let traversal_op_pair = trav_inner.next().unwrap();
                    let traversal_kind_pair = traversal_op_pair.into_inner().next().unwrap();
                    let traversal = build_traversal(traversal_kind_pair);
                    let dst = build_ast(trav_inner.next().unwrap());
                    src = Pattern::GraphTraversal {
                        src: Box::new(src),
                        traversal,
                        dst: Box::new(dst),
                    };
                }
            }
            src
        }
        Rule::sequence => {
            let mut patterns = Vec::new();
            for inner in pair.into_inner() {
                match inner.as_rule() {
                    Rule::pattern | Rule::named_capture | Rule::sequence | Rule::constraint => {
                        patterns.push(build_ast(inner));
                    }
                    Rule::WHITESPACE => {}
                    _ => panic!("Unexpected rule in sequence: {:?}", inner.as_rule()),
                }
            }
            if patterns.len() == 1 {
                patterns.pop().unwrap()
            } else {
                Pattern::Concatenated(patterns)
            }
        }
        Rule::named_capture => {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str().to_string();
            let pattern = build_ast(inner.next().unwrap());
            Pattern::NamedCapture { name, pattern: Box::new(pattern) }
        }
        Rule::pattern | Rule::group => {
            let inner = pair.into_inner().next().unwrap();
            build_ast(inner)
        }
        Rule::constraint => {
            if let Some(body) = pair.into_inner().next() {
                Pattern::Constraint(build_constraint(body))
            } else {
                Pattern::Constraint(Constraint::Wildcard)
            }
        }
        _ => panic!("Unsupported pattern for now: {:?}", pair.as_rule()),
    }
}

fn build_constraint(pair: pest::iterators::Pair<Rule>) -> Constraint {
    match pair.as_rule() {
        Rule::constraint_body => {
            // Delegate to the inner node
            let inner = pair.into_inner().next().unwrap();
            build_constraint(inner)
        }
        Rule::constraint_expr | Rule::disjunction => {
            let mut children = pair.into_inner().map(build_constraint).collect::<Vec<_>>();
            if children.len() == 1 {
                children.pop().unwrap()
            } else {
                Constraint::Disjunctive(children)
            }
        }
        Rule::conjunction => {
            let mut children = pair.into_inner().map(build_constraint).collect::<Vec<_>>();
            if children.len() == 1 {
                children.pop().unwrap()
            } else {
                Constraint::Conjunctive(children)
            }
        }
        Rule::atom => {
            let inner = pair.into_inner().next().unwrap();
            build_constraint(inner)
        }
        Rule::field_constraint => {
            let mut inner = pair.into_inner();
            let field = inner.next().unwrap().as_str().to_string();
            let value_pair = inner.next().unwrap();
            let matcher = match value_pair.as_rule() {
                Rule::value => Matcher::String(value_pair.as_str().to_string()),
                Rule::regex_value => {
                    let pattern = &value_pair.as_str()[1..value_pair.as_str().len()-1];
                    Matcher::Regex {
                        pattern: pattern.to_string(),
                        regex: std::sync::Arc::new(regex::Regex::new(pattern).unwrap()),
                    }
                }
                _ => unreachable!(),
            };
            Constraint::Field {
                name: field,
                matcher,
            }
        }
        Rule::wildcard => Constraint::Wildcard,
        _ => panic!("Unsupported constraint: {:?}", pair.as_rule()),
    }
}

fn build_traversal(pair: pest::iterators::Pair<Rule>) -> crate::compiler::ast::Traversal {
    match pair.as_rule() {
        Rule::outgoing_wildcard => crate::compiler::ast::Traversal::OutgoingWildcard,
        Rule::incoming_wildcard => crate::compiler::ast::Traversal::IncomingWildcard,
        Rule::outgoing => {
            let mut inner = pair.into_inner();
            let label_pair = inner.next().unwrap();
            let quantifier_pair = inner.next(); // This might be None if no quantifier
            
            let base_traversal = match label_pair.as_rule() {
                Rule::label => crate::compiler::ast::Traversal::Outgoing(crate::compiler::ast::Matcher::String(label_pair.as_str().to_string())),
                Rule::traversal_regex => {
                    let pattern = &label_pair.as_str()[1..label_pair.as_str().len()-1];
                    crate::compiler::ast::Traversal::Outgoing(crate::compiler::ast::Matcher::Regex {
                        pattern: pattern.to_string(),
                        regex: std::sync::Arc::new(regex::Regex::new(pattern).unwrap()),
                    })
                }
                Rule::traversal_label => {
                    let mut inner = label_pair.into_inner();
                    let inner_pair = inner.next().unwrap();
                    match inner_pair.as_rule() {
                        Rule::label => crate::compiler::ast::Traversal::Outgoing(crate::compiler::ast::Matcher::String(inner_pair.as_str().to_string())),
                        Rule::traversal_regex => {
                            let pattern = &inner_pair.as_str()[1..inner_pair.as_str().len()-1];
                            crate::compiler::ast::Traversal::Outgoing(crate::compiler::ast::Matcher::Regex {
                                pattern: pattern.to_string(),
                                regex: std::sync::Arc::new(regex::Regex::new(pattern).unwrap()),
                            })
                        }
                        other => {
                            eprintln!("[DEBUG] Unexpected outgoing traversal_label inner rule: {:?}, text: {}", other, inner_pair.as_str());
                            unreachable!()
                        }
                    }
                }
                other => {
                    eprintln!("[DEBUG] Unexpected outgoing traversal label rule: {:?}, text: {}", other, label_pair.as_str());
                    unreachable!()
                }
            };
            
            // Check if there's a quantifier and wrap accordingly
            if quantifier_pair.is_some() {
                crate::compiler::ast::Traversal::Optional(Box::new(base_traversal))
            } else {
                base_traversal
            }
        }
        Rule::incoming => {
            let mut inner = pair.into_inner();
            let label_pair = inner.next().unwrap();
            let quantifier_pair = inner.next(); // This might be None if no quantifier
            
            let base_traversal = match label_pair.as_rule() {
                Rule::label => crate::compiler::ast::Traversal::Incoming(crate::compiler::ast::Matcher::String(label_pair.as_str().to_string())),
                Rule::traversal_regex => {
                    let pattern = &label_pair.as_str()[1..label_pair.as_str().len()-1];
                    crate::compiler::ast::Traversal::Incoming(crate::compiler::ast::Matcher::Regex {
                        pattern: pattern.to_string(),
                        regex: std::sync::Arc::new(regex::Regex::new(pattern).unwrap()),
                    })
                }
                Rule::traversal_label => {
                    let mut inner = label_pair.into_inner();
                    let inner_pair = inner.next().unwrap();
                    match inner_pair.as_rule() {
                        Rule::label => crate::compiler::ast::Traversal::Incoming(crate::compiler::ast::Matcher::String(inner_pair.as_str().to_string())),
                        Rule::traversal_regex => {
                            let pattern = &inner_pair.as_str()[1..inner_pair.as_str().len()-1];
                            crate::compiler::ast::Traversal::Incoming(crate::compiler::ast::Matcher::Regex {
                                pattern: pattern.to_string(),
                                regex: std::sync::Arc::new(regex::Regex::new(pattern).unwrap()),
                            })
                        }
                        other => {
                            eprintln!("[DEBUG] Unexpected incoming traversal_label inner rule: {:?}, text: {}", other, inner_pair.as_str());
                            unreachable!()
                        }
                    }
                }
                other => {
                    eprintln!("[DEBUG] Unexpected incoming traversal label rule: {:?}, text: {}", other, label_pair.as_str());
                    unreachable!()
                }
            };
            
            // Check if there's a quantifier and wrap accordingly
            if quantifier_pair.is_some() {
                crate::compiler::ast::Traversal::Optional(Box::new(base_traversal))
            } else {
                base_traversal
            }
        }
        other => {
            eprintln!("[DEBUG] Unexpected traversal rule: {:?}, text: {}", other, pair.as_str());
            unreachable!()
        }
    }
}