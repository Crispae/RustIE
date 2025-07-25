use rustie::{ExtractorEngine, compiler::ast::{Pattern, Constraint, Matcher}};
use tantivy::{doc, Index, IndexWriter};
use tantivy::schema::{Schema, TEXT, STORED};
use std::path::Path;
use std::fs;
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== RustIE: Odinson Information Extraction with Tantivy ===");
    
    // Create or open index
    let index_path = Path::new("index");
    let schema_path = Path::new("configs/schema.yaml");
    
    let engine = if index_path.exists() {
        println!("Opening existing index...");
        ExtractorEngine::new(index_path, schema_path)?
    } else {
        println!("Creating new index...");
        // Ensure the index directory exists
        fs::create_dir_all(index_path)?;
        create_index(index_path, schema_path)?
    };

    println!("Index contains {} documents", engine.num_docs());

    // Example queries
    let queries = vec![
        "word:John",
        "pos:NNP",
        "lemma:be",
        "tag:VBZ",
    ];

    for query in queries {
        println!("\n--- Query: '{}' ---", query);
        match engine.query(query) {
            Ok(results) => {
                println!("Found {} matches", results.total_hits);
                if let Some(max_score) = results.max_score {
                    println!("Max score: {:.3}", max_score);
                }
                
                for (i, score_doc) in results.score_docs.iter().take(5).enumerate() {
                    println!("  {}. Document {} (score: {:.3})", 
                        i + 1, score_doc.doc.segment_ord, score_doc.score);
                }
            }
            Err(e) => {
                println!("Error executing query: {}", e);
            }
        }
    }

    // Example AST construction
    println!("\n--- Example AST Construction ---");
    
    // Create a simple pattern: [word=John] [pos=NNP]
    let john_constraint = Constraint::Field {
        name: "word".to_string(),
        matcher: Matcher::String("John".to_string()),
    };
    
    let nnp_constraint = Constraint::Field {
        name: "pos".to_string(),
        matcher: Matcher::String("NNP".to_string()),
    };
    
    let john_pattern = Pattern::Constraint(john_constraint);
    let nnp_pattern = Pattern::Constraint(nnp_constraint);
    
    let _concatenated = Pattern::Concatenated(vec![john_pattern, nnp_pattern]);
    
    println!("Created pattern: [word=John] [pos=NNP]");
    
    // Compile and execute the pattern
    let query_str = "[word=John] [pos=NNP]";
    println!("Compiling query: '{}'", query_str);
    
    match engine.query(query_str) {
        Ok(results) => {
            println!("Found {} matches for compiled pattern", results.total_hits);
        }
        Err(e) => {
            println!("Error executing compiled pattern: {}", e);
        }
    }

    Ok(())
}

fn create_index(index_path: &Path, schema_path: &Path) -> Result<ExtractorEngine> {
    // Create schema
    let mut schema_builder = Schema::builder();
    
    // Core fields
    let _word_field = schema_builder.add_text_field("word", TEXT | STORED);
    let _lemma_field = schema_builder.add_text_field("lemma", TEXT | STORED);
    let _pos_field = schema_builder.add_text_field("pos", TEXT | STORED);
    let _tag_field = schema_builder.add_text_field("tag", TEXT | STORED);
    
    // Document structure
    let _doc_id_field = schema_builder.add_text_field("doc_id", TEXT | STORED);
    let _sentence_id_field = schema_builder.add_text_field("sentence_id", TEXT | STORED);
    let _sentence_length_field = schema_builder.add_u64_field("sentence_length", STORED);
    
    // Binary dependencies field
    let _dependencies_binary_field = schema_builder.add_bytes_field("dependencies_binary", STORED);
    
    // Document type
    let _type_field = schema_builder.add_text_field("type", TEXT | STORED);
    
    let schema = schema_builder.build();
    
    // Create index
    let index = Index::create_in_dir(index_path, schema.clone())?;
    let mut index_writer = index.writer(50_000_000)?;
    
    // Add sample documents
    add_sample_documents(&mut index_writer, &schema)?;
    
    index_writer.commit()?;
    
    // Create engine
    ExtractorEngine::new(index_path, schema_path)
}

fn add_sample_documents(index_writer: &mut IndexWriter, schema: &Schema) -> Result<()> {
    let word_field = schema.get_field("word").unwrap();
    let lemma_field = schema.get_field("lemma").unwrap();
    let pos_field = schema.get_field("pos").unwrap();
    let tag_field = schema.get_field("tag").unwrap();
    let doc_id_field = schema.get_field("doc_id").unwrap();
    let sentence_id_field = schema.get_field("sentence_id").unwrap();
    let sentence_length_field = schema.get_field("sentence_length").unwrap();
    let dependencies_binary_field = schema.get_field("dependencies_binary").unwrap();
    let type_field = schema.get_field("type").unwrap();
    
    // Sample sentence: "John Smith works at Microsoft."
    let sentence = vec![
        ("John", "john", "NNP", "NNP", 0),
        ("Smith", "smith", "NNP", "NNP", 1),
        ("works", "work", "VBZ", "VBZ", 2),
        ("at", "at", "IN", "IN", 3),
        ("Microsoft", "microsoft", "NNP", "NNP", 4),
        (".", ".", ".", ".", 5),
    ];
    
    // Create a simple dependency graph for the sample sentence
    use rustie::digraph::DirectedGraph;
    let mut graph = DirectedGraph::new();
    
    // Add nodes
    for i in 0..sentence.len() {
        graph.add_node(i);
    }
    
    // Add edges (simple dependency structure)
    graph.add_edge(0, 2, "nsubj"); // John -> works
    graph.add_edge(1, 2, "compound"); // Smith -> works  
    graph.add_edge(2, 3, "prep"); // works -> at
    graph.add_edge(3, 4, "pobj"); // at -> Microsoft
    
    // Serialize graph to binary
    let graph_bytes = graph.to_bytes()?;
    
    for (token_idx, (word, lemma, pos, tag, _)) in sentence.iter().enumerate() {
        let doc = doc!(
            word_field => *word,
            lemma_field => *lemma,
            pos_field => *pos,
            tag_field => *tag,
            doc_id_field => "doc1",
            sentence_id_field => "sent1",
            sentence_length_field => sentence.len() as u64,
            dependencies_binary_field => graph_bytes.clone(),
            type_field => "sentence"
        );
        
        index_writer.add_document(doc)?;
    }
    
    // Another sample sentence: "The cat sat on the mat."
    let sentence2 = vec![
        ("The", "the", "DT", "DT", 0),
        ("cat", "cat", "NN", "NN", 1),
        ("sat", "sit", "VBD", "VBD", 2),
        ("on", "on", "IN", "IN", 3),
        ("the", "the", "DT", "DT", 4),
        ("mat", "mat", "NN", "NN", 5),
        (".", ".", ".", ".", 6),
    ];
    
    // Create dependency graph for second sentence
    let mut graph2 = DirectedGraph::new();
    
    // Add nodes
    for i in 0..sentence2.len() {
        graph2.add_node(i);
    }
    
    // Add edges
    graph2.add_edge(0, 1, "det"); // The -> cat
    graph2.add_edge(1, 2, "nsubj"); // cat -> sat
    graph2.add_edge(2, 3, "prep"); // sat -> on
    graph2.add_edge(3, 5, "pobj"); // on -> mat
    graph2.add_edge(4, 5, "det"); // the -> mat
    
    // Serialize graph to binary
    let graph2_bytes = graph2.to_bytes()?;
    
    for (token_idx, (word, lemma, pos, tag, _)) in sentence2.iter().enumerate() {
        let doc = doc!(
            word_field => *word,
            lemma_field => *lemma,
            pos_field => *pos,
            tag_field => *tag,
            doc_id_field => "doc2",
            sentence_id_field => "sent2",
            sentence_length_field => sentence2.len() as u64,
            dependencies_binary_field => graph2_bytes.clone(),
            type_field => "sentence"
        );
        
        index_writer.add_document(doc)?;
    }
    
    Ok(())
}