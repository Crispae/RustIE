use rustie::{Document, ExtractorEngine};
use rustie::data::Field;
use std::path::{Path, PathBuf};
use std::fs;
use std::io::{BufReader, Read};
use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use serde_json::Value;
use log::{info, error};
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser, Debug)]
#[command(name = "document_indexer")]
#[command(about = "Index JSON documents for information extraction")]
#[command(version)]
struct Args {
    /// Input directory containing JSON documents
    #[arg(short, long)]
    input_dir: PathBuf,

    /// Output directory for the index
    #[arg(short, long, default_value = "./index")]
    output_dir: PathBuf,

    /// Index format
    #[arg(short, long, default_value = "tantivy")]
    format: IndexFormat,

    /// Schema configuration file (optional)
    #[arg(short, long)]
    schema: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// File pattern to match (default: *.json)
    #[arg(short, long, default_value = "*.json")]
    pattern: String,
}

#[derive(Debug, Clone, ValueEnum)]
enum IndexFormat {
    Tantivy,
    Lucene,
}

/// Document indexer for processing JSON files
pub struct DocumentIndexer {
    engine: ExtractorEngine,
    stats: IndexStats,
    progress_bar: Option<ProgressBar>,
    verbose: bool,
}

#[derive(Debug)]
struct IndexStats {
    total_files: usize,
    processed_files: usize,
    failed_files: usize,
    total_sentences: usize,
    total_tokens: usize,
    total_dependency_labels: usize,
    current_file: Option<String>,
    start_time: std::time::Instant,
}

impl Default for IndexStats {
    fn default() -> Self {
        Self {
            total_files: 0,
            processed_files: 0,
            failed_files: 0,
            total_sentences: 0,
            total_tokens: 0,
            total_dependency_labels: 0,
            current_file: None,
            start_time: std::time::Instant::now(),
        }
    }
}

impl DocumentIndexer {
    /// Create a new document indexer
    pub fn new(index_dir: &Path, schema_path: Option<&Path>, verbose: bool) -> Result<Self> {
        let engine = if let Some(schema) = schema_path {
            ExtractorEngine::new(index_dir, schema)?
        } else {
            // Use default schema if none provided
            let default_schema = Path::new("configs/schema.yaml");
            ExtractorEngine::new(index_dir, default_schema)?
        };

        Ok(Self {
            engine,
            stats: IndexStats::default(),
            progress_bar: None,
            verbose,
        })
    }

    /// Initialize progress bar
    fn init_progress_bar(&mut self, total_files: usize) {
        if !self.verbose {
            let pb = ProgressBar::new(total_files as u64);
            let style = ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-");
            
            pb.set_style(style);
            pb.set_message("Indexing documents...");
            self.progress_bar = Some(pb);
        }
    }

    /// Update progress bar
    fn update_progress(&mut self, current_file: &str) {
        if let Some(pb) = &mut self.progress_bar {
            pb.set_message(format!("Processing: {}", current_file));
            pb.inc(1);
        } else if self.verbose {
            info!("Processing: {}", current_file);
        }
    }

    /// Finish progress bar
    fn finish_progress(&mut self) {
        if let Some(pb) = &mut self.progress_bar {
            pb.finish_with_message("Indexing completed!");
        }
    }

    /// Index all documents in a directory
    pub fn index_directory(&mut self, input_dir: &Path, pattern: &str) -> Result<()> {
        if self.verbose {
            info!("Starting to index documents from: {}", input_dir.display());
        }

        if !input_dir.exists() {
            return Err(anyhow!("Input directory does not exist: {}", input_dir.display()));
        }

        if !input_dir.is_dir() {
            return Err(anyhow!("Input path is not a directory: {}", input_dir.display()));
        }

        let files = self.find_json_files(input_dir, pattern)?;
        
        if self.verbose {
            info!("Found {} JSON files to process", files.len());
        } else {
            println!("Found {} JSON files to process", files.len());
        }

        // Initialize progress bar
        self.init_progress_bar(files.len());

        for file_path in &files {
            self.process_file(file_path)?;
            self.update_progress(file_path.file_name().unwrap().to_str().unwrap_or("unknown"));
        }

        self.finish_progress();
        
        // Commit all changes to the index
        if self.verbose {
            info!("Committing changes to index...");
        }
        self.engine.commit()?;
        
        // Note: For binary dependency graphs, vocabulary size is not tracked in stats
        // The mapping is handled in-memory during traversal
        
        self.print_stats();
        Ok(())
    }

    /// Find all JSON files in the directory (including compressed)
    fn find_json_files(&self, dir: &Path, pattern: &str) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        
        if pattern == "*.json" {
            // Use glob pattern for JSON files (including .json.gz)
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_file() {
                    if let Some(extension) = path.extension() {
                        if extension == "json" {
                            files.push(path);
                        } else if extension == "gz" {
                            // Check if it's a .json.gz file
                            if let Some(stem) = path.file_stem() {
                                if let Some(stem_str) = stem.to_str() {
                                    if stem_str.ends_with(".json") {
                                        files.push(path);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Use custom pattern (simplified glob matching)
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_file() {
                    let file_name = path.file_name().unwrap().to_string_lossy();
                    if self.matches_pattern(&file_name, pattern) {
                        files.push(path);
                    }
                }
            }
        }

        files.sort();
        Ok(files)
    }

    /// Simple pattern matching (supports * wildcard and compressed files)
    fn matches_pattern(&self, filename: &str, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }
        
        if pattern.contains('*') {
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                let prefix = parts[0];
                let suffix = parts[1];
                return filename.starts_with(prefix) && filename.ends_with(suffix);
            }
        }
        
        // Special handling for .json.gz pattern
        if pattern == "*.json.gz" {
            return filename.ends_with(".json.gz");
        }
        
        filename == pattern
    }

    /// Process a single JSON file
    fn process_file(&mut self, file_path: &Path) -> Result<()> {
        self.stats.total_files += 1;
        
        if self.verbose {
            info!("Processing file: {}", file_path.display());
        }
        
        match self.parse_and_index_file(file_path) {
            Ok(_) => {
                self.stats.processed_files += 1;
                if self.verbose {
                    info!("Successfully processed: {}", file_path.display());
                }
            }
            Err(e) => {
                self.stats.failed_files += 1;
                error!("Failed to process {}: {}", file_path.display(), e);
            }
        }
        
        Ok(())
    }

    /// Parse and index a single file (supports both plain and gzipped JSON)
    fn parse_and_index_file(&mut self, file_path: &Path) -> Result<()> {
        let content = self.read_file_content(file_path)?;
        let json_value: Value = serde_json::from_str(&content)?;
        
        // Try to parse as a single document
        if let Ok(document) = serde_json::from_value::<Document>(json_value.clone()) {
            self.index_document(&document, file_path)?;
        } else {
            // Try to parse as an array of documents
            if let Ok(documents) = serde_json::from_value::<Vec<Document>>(json_value) {
                for (i, document) in documents.into_iter().enumerate() {
                    let doc_path = format!("{}#{}", file_path.display(), i);
                    self.index_document(&document, Path::new(&doc_path))?;
                }
            } else {
                return Err(anyhow!("Could not parse as Document or Vec<Document>"));
            }
        }
        
        Ok(())
    }

    /// Read file content, handling both plain and gzipped files
    fn read_file_content(&self, file_path: &Path) -> Result<String> {
        let file = fs::File::open(file_path)?;
        let mut reader = BufReader::new(file);
        
        // Check if file is gzipped by looking at the extension
        if let Some(extension) = file_path.extension() {
            if extension == "gz" {
                // Handle gzipped file
                let mut gz_decoder = GzDecoder::new(reader);
                let mut content = String::new();
                gz_decoder.read_to_string(&mut content)?;
                return Ok(content);
            }
        }
        
        // Handle plain text file
        let mut content = String::new();
        reader.read_to_string(&mut content)?;
        Ok(content)
    }

    /// Index a single document
    fn index_document(&mut self, document: &Document, source_path: &Path) -> Result<()> {
        // Update statistics
        self.stats.total_sentences += document.sentences.len();
        
        for (sentence_idx, _sentence) in document.sentences.iter().enumerate() {
            // Count tokens
            if let Some(tokens_field) = document.get_field(sentence_idx, "word") {
                if let Field::TokensField { tokens, .. } = tokens_field {
                    self.stats.total_tokens += tokens.len();
                }
            }
            
            // Count dependency labels (edges)
            if let Some(deps) = document.get_dependencies(sentence_idx) {
                self.stats.total_dependency_labels += deps.edges.len();
            }
        }

        // Add the document to the index
        self.engine.add_document(document)?;
        
        if self.verbose {
            info!("Document {} has {} sentences", document.id, document.sentences.len());
        }
        
        Ok(())
    }

    /// Print indexing statistics
    fn print_stats(&self) {
        let elapsed = self.stats.start_time.elapsed();
        
        println!("\n=== Indexing Statistics ===");
        println!("Total files found: {}", self.stats.total_files);
        println!("Successfully processed: {}", self.stats.processed_files);
        println!("Failed to process: {}", self.stats.failed_files);
        println!("Total sentences: {}", self.stats.total_sentences);
        println!("Total tokens: {}", self.stats.total_tokens);
        println!("Total dependency labels: {}", self.stats.total_dependency_labels);
        println!("Total time: {:.2?}", elapsed);
        
        if self.stats.total_files > 0 {
            let success_rate = (self.stats.processed_files as f64 / self.stats.total_files as f64) * 100.0;
            println!("Success rate: {:.1}%", success_rate);
            
            if elapsed.as_secs() > 0 {
                let files_per_second = self.stats.processed_files as f64 / elapsed.as_secs_f64();
                println!("Processing speed: {:.1} files/second", files_per_second);
            }
        }
        
        // Note: Vocabulary information is now handled in-memory during traversal
        println!("\n=== Dependency Vocabulary ===");
        println!("Note: Binary dependency graphs are used for efficient storage and traversal");
        println!("Vocabulary mapping is handled in-memory during graph traversal");
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        env_logger::init();
    }

    println!("Document Indexer for RustIE");
    println!("===========================");
    println!("Input directory: {}", args.input_dir.display());
    println!("Output directory: {}", args.output_dir.display());
    println!("File pattern: {}", args.pattern);
    println!();

    // Create output directory if it doesn't exist
    if !args.output_dir.exists() {
        fs::create_dir_all(&args.output_dir)?;
        info!("Created output directory: {}", args.output_dir.display());
    }

    // Initialize the indexer
    let mut indexer = DocumentIndexer::new(&args.output_dir, args.schema.as_deref(), args.verbose)?;

    // Index the documents
    indexer.index_directory(&args.input_dir, &args.pattern)?;

    println!("\nIndexing completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_pattern_matching() {
        // Use a temporary directory for the index
        let temp_dir = TempDir::new().unwrap();
        let indexer = DocumentIndexer::new(temp_dir.path(), None, false).unwrap();

        assert!(indexer.matches_pattern("test.json", "*.json"));
        assert!(indexer.matches_pattern("document.json", "*.json"));
        assert!(!indexer.matches_pattern("test.txt", "*.json"));
        assert!(indexer.matches_pattern("any_file", "*"));
    }

    #[test]
    fn test_find_json_files() {
        let temp_dir = TempDir::new().unwrap();
        let test_dir = temp_dir.path();

        // Create some test files
        fs::write(test_dir.join("test1.json"), "{}").unwrap();
        fs::write(test_dir.join("test2.json"), "[]").unwrap();
        fs::write(test_dir.join("ignore.txt"), "text").unwrap();

        // Use a temporary directory for the index
        let index_temp_dir = TempDir::new().unwrap();
        let indexer = DocumentIndexer::new(index_temp_dir.path(), None, false).unwrap();
        let files = indexer.find_json_files(test_dir, "*.json").unwrap();

        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|f| f.file_name().unwrap() == "test1.json"));
        assert!(files.iter().any(|f| f.file_name().unwrap() == "test2.json"));
    }
} 