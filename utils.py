"""
AI Notes Summarizer - Utility Functions

This module provides utility functions for file handling and text extraction.
"""

import io
from typing import Optional
from PyPDF2 import PdfReader
import pdfplumber
from docx import Document


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF file.
    
    Args:
        file_bytes: PDF file content as bytes
        
    Returns:
        Extracted text as string
    """
    text_parts = []
    
    # Try PyPDF2 first
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    except Exception:
        pass
    
    # If PyPDF2 didn't work well, try pdfplumber
    if not text_parts or len(''.join(text_parts).strip()) < 100:
        text_parts = []
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    return '\n'.join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract text from DOCX file.
    
    Args:
        file_bytes: DOCX file content as bytes
        
    Returns:
        Extracted text as string
    """
    try:
        doc = Document(io.BytesIO(file_bytes))
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        return '\n'.join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}")


def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Extract text from TXT file.
    
    Args:
        file_bytes: TXT file content as bytes
        
    Returns:
        Extracted text as string
    """
    try:
        # Try UTF-8 first
        return file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            # Fallback to latin-1
            return file_bytes.decode('latin-1')
        except Exception as e:
            raise ValueError(f"Failed to decode text file: {str(e)}")


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from uploaded file based on extension.
    
    Args:
        file_bytes: File content as bytes
        filename: Name of the file (used to determine type)
        
    Returns:
        Extracted text as string
    """
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    elif filename_lower.endswith('.docx'):
        return extract_text_from_docx(file_bytes)
    elif filename_lower.endswith('.txt'):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {filename}. Supported types: PDF, DOCX, TXT")


def format_stats(stats: dict) -> str:
    """
    Format statistics dictionary into readable string.
    
    Args:
        stats: Dictionary containing text statistics
        
    Returns:
        Formatted string
    """
    return (
        f"📊 **Characters:** {stats.get('char_count', 0):,} | "
        f"**Words:** {stats.get('word_count', 0):,} | "
        f"**Sentences:** {stats.get('sentence_count', 0)}"
    )


def calculate_compression_ratio(original_words: int, summary_words: int) -> float:
    """
    Calculate compression ratio as percentage.
    
    Args:
        original_words: Word count of original text
        summary_words: Word count of summary
        
    Returns:
        Compression ratio as percentage
    """
    if original_words == 0:
        return 0.0
    return round((1 - summary_words / original_words) * 100, 2)
