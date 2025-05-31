"""PDF Processing module for extracting text from medical documents."""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import fitz  # PyMuPDF
import pdfplumber
from PyPDF2 import PdfReader
from loguru import logger

from config import config


@dataclass
class TextChunk:
    """Represents a chunk of extracted text with metadata."""
    
    content: str
    page_number: int
    chunk_index: int
    source_file: str
    extraction_method: str
    confidence_score: float = 1.0


class PDFProcessor:
    """Advanced PDF text extraction with multiple fallback methods."""
    
    def __init__(self):
        self.extraction_methods = [
            self._extract_with_pymupdf,
            self._extract_with_pdfplumber,
            self._extract_with_pypdf2
        ]
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[TextChunk]:
        """
        Extract text from PDF using multiple methods with fallback.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of TextChunk objects containing extracted text
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        for i, method in enumerate(self.extraction_methods):
            try:
                chunks = method(pdf_path)
                if chunks and self._validate_extraction(chunks):
                    logger.info(f"Successfully extracted text using method {i+1}")
                    return chunks
                else:
                    logger.warning(f"Method {i+1} failed or produced poor results")
            except Exception as e:
                logger.warning(f"Method {i+1} failed with error: {e}")
                continue
        
        logger.error(f"All extraction methods failed for {pdf_path}")
        return []
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> List[TextChunk]:
        """Extract text using PyMuPDF (best for complex layouts)."""
        chunks = []
        
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    # Clean and process the text
                    cleaned_text = self._clean_text(text)
                    page_chunks = self._create_chunks(
                        cleaned_text, page_num, str(pdf_path), "PyMuPDF"
                    )
                    chunks.extend(page_chunks)
        
        return chunks
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> List[TextChunk]:
        """Extract text using pdfplumber (good for tables and structured data)."""
        chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    # Also extract tables if present
                    tables = page.extract_tables()
                    if tables:
                        table_text = self._tables_to_text(tables)
                        text += f"\n\nTables:\n{table_text}"
                    
                    cleaned_text = self._clean_text(text)
                    page_chunks = self._create_chunks(
                        cleaned_text, page_num, str(pdf_path), "pdfplumber"
                    )
                    chunks.extend(page_chunks)
        
        return chunks
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> List[TextChunk]:
        """Extract text using PyPDF2 (fallback method)."""
        chunks = []
        
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    cleaned_text = self._clean_text(text)
                    page_chunks = self._create_chunks(
                        cleaned_text, page_num, str(pdf_path), "PyPDF2"
                    )
                    chunks.extend(page_chunks)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\d+\s*$', '', text)  # Numbers at end of lines
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces
        text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)     # Missing spaces after periods
        
        # Remove extra newlines but preserve paragraph structure
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _tables_to_text(self, tables: List[List[List[str]]]) -> str:
        """Convert extracted tables to readable text format."""
        table_texts = []
        
        for table in tables:
            if not table:
                continue
                
            # Convert table to text representation
            table_text = []
            for row in table:
                if row:
                    row_text = " | ".join(str(cell) if cell else "" for cell in row)
                    table_text.append(row_text)
            
            if table_text:
                table_texts.append("\n".join(table_text))
        
        return "\n\n".join(table_texts)
    
    def _create_chunks(self, text: str, page_num: int, source_file: str, method: str) -> List[TextChunk]:
        """Split text into chunks for processing."""
        chunks = []
        
        # Split text into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > config.chunk_size:
                if current_chunk.strip():
                    chunks.append(TextChunk(
                        content=current_chunk.strip(),
                        page_number=page_num,
                        chunk_index=chunk_index,
                        source_file=source_file,
                        extraction_method=method,
                        confidence_score=self._calculate_confidence(current_chunk)
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap
                if config.chunk_overlap > 0:
                    overlap_sentences = sentences[max(0, len(sentences) - 2):]
                    current_chunk = " ".join(overlap_sentences) + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(TextChunk(
                content=current_chunk.strip(),
                page_number=page_num,
                chunk_index=chunk_index,
                source_file=source_file,
                extraction_method=method,
                confidence_score=self._calculate_confidence(current_chunk)
            ))
        
        return chunks
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score for extracted text quality."""
        if not text:
            return 0.0
        
        # Factors that indicate good extraction
        word_count = len(text.split())
        if word_count < 10:
            return 0.3
        
        # Check for readable sentences
        sentences = re.split(r'[.!?]+', text)
        readable_sentences = sum(1 for s in sentences if len(s.split()) > 3)
        sentence_ratio = readable_sentences / max(len(sentences), 1)
        
        # Check for excessive special characters (indicates poor extraction)
        special_char_ratio = len(re.findall(r'[^\w\s.,!?;:]', text)) / len(text)
        
        confidence = min(1.0, sentence_ratio * (1 - special_char_ratio))
        return max(0.1, confidence)
    
    def _validate_extraction(self, chunks: List[TextChunk]) -> bool:
        """Validate that the extraction produced usable results."""
        if not chunks:
            return False
        
        # Check average confidence
        avg_confidence = sum(chunk.confidence_score for chunk in chunks) / len(chunks)
        if avg_confidence < 0.3:
            return False
        
        # Check total content length
        total_content = sum(len(chunk.content) for chunk in chunks)
        if total_content < 100:  # Too little content
            return False
        
        return True
    
    def process_directory(self, input_dir: Path) -> Dict[str, List[TextChunk]]:
        """Process all PDF files in a directory."""
        pdf_files = list(input_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return {}
        
        results = {}
        
        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}")
            chunks = self.extract_text_from_pdf(pdf_file)
            
            if chunks:
                results[pdf_file.name] = chunks
                logger.info(f"Extracted {len(chunks)} chunks from {pdf_file.name}")
            else:
                logger.error(f"Failed to extract text from {pdf_file.name}")
        
        return results 