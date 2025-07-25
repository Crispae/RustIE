use std::fs;
use std::io::Write;
use flate2::write::GzEncoder;
use flate2::Compression;
use anyhow::Result;

fn main() -> Result<()> {
    let input_file = "sample_documents/sample1.json";
    let output_file = "sample_documents/sample1.json.gz";
    
    // Read the input file
    let content = fs::read_to_string(input_file)?;
    
    // Create the output file
    let output = fs::File::create(output_file)?;
    let mut encoder = GzEncoder::new(output, Compression::default());
    
    // Write the compressed content
    encoder.write_all(content.as_bytes())?;
    encoder.finish()?;
    
    println!("Successfully compressed {} to {}", input_file, output_file);
    Ok(())
} 