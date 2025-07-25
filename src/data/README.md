# Data Parsing Module

This module provides functionality to parse JSON documents in the Odinson format and convert them to Tantivy documents for indexing.

## Features

- Parse JSON documents with the Odinson schema
- Support for both regular JSON and gzipped JSON files
- Handle single documents, arrays of documents, and JSONL format
- Convert parsed documents to Tantivy documents for indexing
- Support for all Odinson field types (TokensField, GraphField)

## Data Structure

The module expects JSON documents in the following format:

```json
{
  "id": "document_id",
  "metadata": ["meta1", "meta2"],
  "sentences": [
    {
      "numTokens": 13,
      "fields": [
        {
          "name": "word",
          "$type": "ai.lum.odinson.TokensField",
          "tokens": ["The", "diffusivity", "is", "observed", "to", "be", "very", "directly", "linked", "to", "network", "imperfection", "."]
        },
        {
          "name": "lemma",
          "$type": "ai.lum.odinson.TokensField",
          "tokens": ["the", "diffusivity", "be", "observe", "to", "be", "very", "directly", "link", "to", "network", "imperfection", "."]
        },
        {
          "name": "dependencies",
          "$type": "ai.lum.odinson.GraphField",
          "edges": [[8, 11, "nmod"], [11, 9, "case"], [3, 12, "punct"]],
          "roots": [3]
        }
      ]
    }
  ]
}
```

## Field Types

### TokensField
Represents tokenized text fields like words, lemmas, POS tags, etc.

```rust
pub struct TokensField {
    pub name: String,
    pub tokens: Vec<String>,
}
```

### GraphField
Represents dependency graphs with edges and root nodes.

```rust
pub struct GraphField {
    pub name: String,
    pub edges: Vec<[u32; 3]>, // [from, to, relation]
    pub roots: Vec<u32>,
}
```

## Usage

### Basic Parsing

```rust
use rustie::data::{Document, DocumentParser};
use rustie::engine::ExtractorEngine;

// Create schema and parser
let schema = ExtractorEngine::create_schema_from_yaml(Some("configs/schema.yaml"));
let parser = DocumentParser::new(schema);

// Parse JSON string
let json_str = r#"{"id": "doc1", "metadata": [], "sentences": [...]}"#;
let documents = parser.parse_json(json_str)?;

// Parse from file (supports both .json and .json.gz)
let documents = parser.parse_file("documents.json")?;
let documents = parser.parse_file("documents.json.gz")?;
```

### Converting to Tantivy Documents

```rust
for document in documents {
    let tantivy_docs = parser.to_tantivy_document(&document)?;
    
    // Each sentence becomes a separate Tantivy document
    for tantivy_doc in tantivy_docs {
        // Add to index
        index_writer.add_document(tantivy_doc)?;
    }
}
```

### Accessing Document Data

```rust
let doc = &documents[0];

// Get tokens from a specific field
if let Some(words) = doc.get_tokens(0, "word") {
    println!("Words: {}", words.join(" "));
}

// Get dependencies
if let Some(deps) = doc.dependencies_to_string(0) {
    println!("Dependencies: {}", deps);
}

// Get sentence length
if let Some(length) = doc.sentence_length(0) {
    println!("Sentence length: {}", length);
}
```

## File Formats Supported

1. **Single Document JSON**: A single document object
2. **Array of Documents**: An array containing multiple document objects
3. **JSONL (JSON Lines)**: One JSON object per line
4. **Gzipped Files**: Any of the above formats compressed with gzip

## Field Mapping

The parser automatically maps Odinson fields to Tantivy schema fields:

- `word` → `word` field
- `lemma` → `lemma` field  
- `tag`/`pos` → `tag` field
- `entity` → `entity` field
- `chunk` → `chunk` field
- `norm` → `norm` field
- `raw` → `raw` field
- `dependencies` → `dependencies` field (with additional `incoming`/`outgoing` fields)

## Example Binary

Run the example binary to see the parser in action:

```bash
cargo run --bin data_parser
```

This will demonstrate parsing the example JSON structure and converting it to Tantivy documents.

## Error Handling

The parser uses `anyhow::Result` for comprehensive error handling:

- JSON parsing errors
- File I/O errors
- Schema field mapping errors
- Gzip decompression errors

All errors are wrapped in `anyhow::Error` for easy handling and debugging. 