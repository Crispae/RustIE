use pest::Parser;
use rustie::compiler::pest_parser::{QueryParser, Rule, build_ast};

fn main() {
    let queries = vec![
        "[word=john]",
        "[word=/john/]",
        "[lemma=/john|ram/]",
        "[word=dog & tag=NN]",
        "[word=/j.*/ | tag=NN]",
        "[*]",
        "[word=the] [word=dog]",
        "[word=the] [*] [word=dog]",
        "[word=dog] >nsubj [word=barks]",
        "[word=dog] >/nsubj.*/ [word=barks]",
        "[word=dog] >> [word=barks]",
        "[word=dog] <dobj [word=cat]",
        "[word=dog] </dobj.*/ [word=cat]",
        "[word=dog] << [word=cat]",
        "[word=dog] >nsubj [word=cat] <dobj [word=dog]",
        "[word=dog] >nsubj >dobj [word=dog]",
        // Test optional traversal
        "[word=TAZ] >amod? [word=transcriptional]"

    ];
    for query in queries {
    match QueryParser::parse(Rule::query, &query) {
        Ok(mut pairs) => {
            let ast = build_ast(pairs.next().unwrap());
            println!("Valid! AST: {:#?}", ast);
        },
        Err(e) => {
                println!("Invalid! Error: {}", e);
            }
        }
    }

} 

