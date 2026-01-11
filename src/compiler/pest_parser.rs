use pest::Parser;
use pest_derive::Parser;
use crate::compiler::ast::{Pattern, Constraint, Matcher, Assertion};

#[derive(Parser)]
#[grammar = "query.pest"]
pub struct QueryParser;

// Delegating AST builder
pub fn build_ast(pair: pest::iterators::Pair<Rule>) -> Pattern {
    match pair.as_rule() {
        Rule::query => {
            let inner = pair.into_inner().next().unwrap();
            build_ast(inner)
        }
        Rule::default_field_query => {
            let inner = pair.into_inner().next().unwrap();
            match inner.as_rule() {
                Rule::default_string => {
                    let value = inner.as_str().to_string();
                    Pattern::Constraint(Constraint::Field {
                        name: "word".to_string(),
                        matcher: Matcher::String(value),
                    })
                }
                Rule::default_regex => {
                    let pattern = &inner.as_str()[1..inner.as_str().len()-1];
                    Pattern::Constraint(Constraint::Field {
                        name: "word".to_string(),
                        matcher: Matcher::Regex {
                            pattern: pattern.to_string(),
                            regex: std::sync::Arc::new(regex::Regex::new(pattern).unwrap()),
                        },
                    })
                }
                _ => unreachable!(),
            }
        }
        Rule::traversal_seq => {
            let mut pairs = pair.into_inner();
            let mut src = build_ast(pairs.next().unwrap());
            let mut pairs = pairs.peekable();
            while let Some(trav_pair) = pairs.next() {
                if trav_pair.as_rule() == Rule::traversal {
                    let mut trav_inner = trav_pair.into_inner();
                    let traversal_op_pair = trav_inner.next().unwrap();
                    // traversal_op_pair is traversal_op, which contains the actual traversal type
                    let inner_traversal = traversal_op_pair.into_inner().next().unwrap();
                    let traversal = build_traversal_op(inner_traversal);
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
                    Rule::quantified_pattern | Rule::named_capture | Rule::sequence => {
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
        Rule::quantified_pattern => {
            let mut pairs = pair.into_inner();
            let atomic = build_ast(pairs.next().unwrap());
            if let Some(quant_pair) = pairs.next() {
                // quant_pair is pattern_quantifier, which contains greedy_quantifier or lazy_quantifier
                let inner_quant = quant_pair.into_inner().next().unwrap();
                let (min, max, _is_lazy) = parse_quantifier(inner_quant);
                Pattern::Repetition {
                    pattern: Box::new(atomic),
                    min,
                    max,
                }
            } else {
                atomic
            }
        }
        Rule::atomic_pattern => {
            let inner = pair.into_inner().next().unwrap();
            build_ast(inner)
        }
        Rule::named_capture => {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str().to_string();
            let pattern = build_ast(inner.next().unwrap());
            Pattern::NamedCapture { name, pattern: Box::new(pattern) }
        }
        Rule::group => {
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
        Rule::assertion_pattern => {
            let inner = pair.into_inner().next().unwrap();
            Pattern::Assertion(build_assertion(inner))
        }
        _ => panic!("Unsupported pattern for now: {:?}", pair.as_rule()),
    }
}

fn parse_quantifier(pair: pest::iterators::Pair<Rule>) -> (usize, Option<usize>, bool) {
    let text = pair.as_str();
    match pair.as_rule() {
        Rule::greedy_quantifier => {
            // Check if it's a simple string literal first
            match text {
                "*" => return (0, None, false),
                "+" => return (1, None, false),
                "?" => return (0, Some(1), false),
                _ => {}
            }
            
            // Otherwise, it should be a range_quantifier
            let mut inner_pairs = pair.into_inner();
            if let Some(inner) = inner_pairs.next() {
                if inner.as_rule() == Rule::range_quantifier {
                    let mut pairs = inner.into_inner();
                    let min_str = pairs.next();
                    let max_str = pairs.next();
                    let min = min_str.map(|p| p.as_str().parse().unwrap()).unwrap_or(0);
                    let max = max_str.and_then(|p| p.as_str().parse::<usize>().ok());
                    (min, max, false)
                } else {
                    panic!("Unexpected inner rule in greedy_quantifier: {:?}, text: {}", inner.as_rule(), text)
                }
            } else {
                // Fallback: try to parse as range quantifier from text
                if text.starts_with('{') && text.ends_with('}') {
                    let content = &text[1..text.len()-1];
                    let parts: Vec<&str> = content.split(',').collect();
                    let min = parts.get(0).and_then(|s| s.trim().parse().ok()).unwrap_or(0);
                    let max = parts.get(1).and_then(|s| s.trim().parse::<usize>().ok());
                    (min, max, false)
                } else {
                    panic!("Unexpected greedy quantifier: {}", text)
                }
            }
        }
        Rule::lazy_quantifier => {
            // Check if it's a simple string literal first
            match text {
                "*?" => return (0, None, true),
                "+?" => return (1, None, true),
                "??" => return (0, Some(1), true),
                _ => {}
            }
            
            // Otherwise, it should be a lazy_range_quantifier
            let mut inner_pairs = pair.into_inner();
            if let Some(inner) = inner_pairs.next() {
                if inner.as_rule() == Rule::lazy_range_quantifier {
                    let mut pairs = inner.into_inner();
                    let min_str = pairs.next();
                    let max_str = pairs.next();
                    let min = min_str.map(|p| p.as_str().parse().unwrap()).unwrap_or(0);
                    let max = max_str.and_then(|p| p.as_str().parse::<usize>().ok());
                    (min, max, true)
                } else {
                    panic!("Unexpected inner rule in lazy_quantifier: {:?}, text: {}", inner.as_rule(), text)
                }
            } else {
                // Fallback: try to parse as lazy range quantifier from text
                if text.starts_with('{') && text.ends_with("}?") {
                    let content = &text[1..text.len()-2];
                    let parts: Vec<&str> = content.split(',').collect();
                    let min = parts.get(0).and_then(|s| s.trim().parse().ok()).unwrap_or(0);
                    let max = parts.get(1).and_then(|s| s.trim().parse::<usize>().ok());
                    (min, max, true)
                } else {
                    panic!("Unexpected lazy quantifier: {}", text)
                }
            }
        }
        _ => panic!("Unexpected quantifier rule: {:?}, text: {}", pair.as_rule(), text),
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
        Rule::negated_atom => {
            let mut inner = pair.into_inner();
            let first = inner.next().unwrap();
            if first.as_str() == "!" {
                // Negated atom
                let atom = inner.next().unwrap();
                Constraint::Negated(Box::new(build_constraint(atom)))
            } else {
                // Regular atom (no negation)
                build_constraint(first)
            }
        }
        Rule::atom => {
            let inner = pair.into_inner().next().unwrap();
            build_constraint(inner)
        }
        Rule::group_constraint => {
            let inner = pair.into_inner().next().unwrap();
            build_constraint(inner)
        }
        Rule::field_constraint => {
            let mut inner = pair.into_inner();
            let field = inner.next().unwrap().as_str().to_string();
            let op = inner.next().unwrap().as_str();
            let value_pair = inner.next().unwrap();
            let fuzzy_op = inner.next();
            
            let matcher = match value_pair.as_rule() {
                Rule::value => {
                    let value = value_pair.as_str().to_string();
                    if fuzzy_op.is_some() {
                        // Fuzzy matching - store as Fuzzy constraint
                        return Constraint::Fuzzy {
                            name: field,
                            matcher: value,
                        };
                    }
                    Matcher::String(value)
                }
                Rule::regex_value => {
                    let pattern = &value_pair.as_str()[1..value_pair.as_str().len()-1];
                    Matcher::Regex {
                        pattern: pattern.to_string(),
                        regex: std::sync::Arc::new(regex::Regex::new(pattern).unwrap()),
                    }
                }
                _ => unreachable!(),
            };
            
            if op == "!=" {
                Constraint::Negated(Box::new(Constraint::Field {
                    name: field,
                    matcher,
                }))
            } else {
                Constraint::Field {
                    name: field,
                    matcher,
                }
            }
        }
        Rule::wildcard => Constraint::Wildcard,
        _ => panic!("Unsupported constraint: {:?}", pair.as_rule()),
    }
}

fn build_assertion(pair: pest::iterators::Pair<Rule>) -> Assertion {
    match pair.as_rule() {
        Rule::lookahead_assertion => {
            let inner = pair.into_inner().next().unwrap();
            build_assertion(inner)
        }
        Rule::lookbehind_assertion => {
            let inner = pair.into_inner().next().unwrap();
            build_assertion(inner)
        }
        Rule::positive_lookahead => {
            let inner = pair.into_inner().next().unwrap();
            Assertion::PositiveLookahead(Box::new(build_ast(inner)))
        }
        Rule::negative_lookahead => {
            let inner = pair.into_inner().next().unwrap();
            Assertion::NegativeLookahead(Box::new(build_ast(inner)))
        }
        Rule::positive_lookbehind => {
            let inner = pair.into_inner().next().unwrap();
            Assertion::PositiveLookbehind(Box::new(build_ast(inner)))
        }
        Rule::negative_lookbehind => {
            let inner = pair.into_inner().next().unwrap();
            Assertion::NegativeLookbehind(Box::new(build_ast(inner)))
        }
        _ => panic!("Unsupported assertion: {:?}", pair.as_rule()),
    }
}

fn build_traversal_op(pair: pest::iterators::Pair<Rule>) -> crate::compiler::ast::Traversal {
    match pair.as_rule() {
        Rule::outgoing_wildcard => crate::compiler::ast::Traversal::OutgoingWildcard,
        Rule::incoming_wildcard => crate::compiler::ast::Traversal::IncomingWildcard,
        Rule::outgoing => {
            let mut inner = pair.into_inner();
            let label_pair = inner.next().unwrap();
            let quantifier_pair = inner.next(); // This might be None if no quantifier
            
            let base_traversal = build_traversal_label(label_pair, true);
            
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
            
            let base_traversal = build_traversal_label(label_pair, false);
            
            // Check if there's a quantifier and wrap accordingly
            if quantifier_pair.is_some() {
                crate::compiler::ast::Traversal::Optional(Box::new(base_traversal))
            } else {
                base_traversal
            }
        }
        Rule::disjunctive_traversal => {
            let mut labels = Vec::new();
            for label_pair in pair.into_inner() {
                if label_pair.as_rule() == Rule::traversal_label {
                    labels.push(build_traversal_label(label_pair, true));
                }
            }
            if labels.len() == 1 {
                labels.pop().unwrap()
            } else {
                crate::compiler::ast::Traversal::Disjunctive(labels)
            }
        }
        Rule::concatenated_traversal => {
            let mut labels = Vec::new();
            for label_pair in pair.into_inner() {
                if label_pair.as_rule() == Rule::traversal_label {
                    labels.push(build_traversal_label(label_pair, true));
                }
            }
            if labels.len() == 1 {
                labels.pop().unwrap()
            } else {
                crate::compiler::ast::Traversal::Concatenated(labels)
            }
        }
        other => {
            eprintln!("[DEBUG] Unexpected traversal op rule: {:?}, text: {}", other, pair.as_str());
            unreachable!()
        }
    }
}

fn build_traversal_label(pair: pest::iterators::Pair<Rule>, outgoing: bool) -> crate::compiler::ast::Traversal {
    match pair.as_rule() {
        Rule::label => {
            let matcher = crate::compiler::ast::Matcher::String(pair.as_str().to_string());
            if outgoing {
                crate::compiler::ast::Traversal::Outgoing(matcher)
            } else {
                crate::compiler::ast::Traversal::Incoming(matcher)
            }
        }
        Rule::traversal_regex => {
            let pattern = &pair.as_str()[1..pair.as_str().len()-1];
            let matcher = crate::compiler::ast::Matcher::Regex {
                pattern: pattern.to_string(),
                regex: std::sync::Arc::new(regex::Regex::new(pattern).unwrap()),
            };
            if outgoing {
                crate::compiler::ast::Traversal::Outgoing(matcher)
            } else {
                crate::compiler::ast::Traversal::Incoming(matcher)
            }
        }
        Rule::traversal_label => {
            let inner = pair.into_inner().next().unwrap();
            build_traversal_label(inner, outgoing)
        }
        _ => {
            eprintln!("[DEBUG] Unexpected traversal label rule: {:?}, text: {}", pair.as_rule(), pair.as_str());
            unreachable!()
        }
    }
}