use serde::{Deserialize, Serialize};

/// Represents a span of text with start and end positions
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn length(&self) -> usize {
        self.end - self.start
    }

    pub fn contains(&self, other: &Span) -> bool {
        self.start <= other.start && self.end >= other.end
    }

    pub fn overlaps(&self, other: &Span) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// Represents a named capture in a pattern match
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedCapture {
    pub name: String,
    pub span: Span,
}

impl NamedCapture {
    pub fn new(name: String, span: Span) -> Self {
        Self { name, span }
    }
}

/// A span with associated named captures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanWithCaptures {
    pub span: Span,
    pub captures: Vec<NamedCapture>,
}

impl SpanWithCaptures {
    pub fn new(span: Span) -> Self {
        Self {
            span,
            captures: Vec::new(),
        }
    }

    pub fn with_captures(span: Span, captures: Vec<NamedCapture>) -> Self {
        Self { span, captures }
    }

    pub fn add_capture(&mut self, name: String, span: Span) {
        self.captures.push(NamedCapture::new(name, span));
    }
} 