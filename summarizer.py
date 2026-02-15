"""
AI Notes Summarizer - Core Summarization Module

This module provides both extractive and abstractive text summarization capabilities.
"""

import re
from typing import List, Tuple
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Try to import transformers, but make it optional
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError) as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Warning: Transformers not available: {e}")
    print("Abstractive summarization and AI Q&A will be disabled.")
    print("Extractive methods will work perfectly!")


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

LANGUAGE = "english"


class TextSummarizer:
    """Handles both extractive and abstractive text summarization."""
    
    def __init__(self):
        self.stemmer = Stemmer(LANGUAGE)
        self.stop_words = get_stop_words(LANGUAGE)
        self._abstractive_pipeline = None
        self._qa_pipeline = None
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\']', '', text)
        return text.strip()
    
    def get_text_stats(self, text: str) -> dict:
        """Calculate statistics for the given text."""
        sentences = nltk.sent_tokenize(text)
        words = text.split()
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences)
        }
    
    def extractive_summary(
        self, 
        text: str, 
        sentence_count: int = 5,
        algorithm: str = "lsa"
    ) -> Tuple[str, dict]:
        """
        Generate extractive summary using Sumy library.
        
        Args:
            text: Input text to summarize
            sentence_count: Number of sentences in summary
            algorithm: Algorithm to use (lsa, luhn, textrank, lexrank)
            
        Returns:
            Tuple of (summary_text, metadata)
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Parse text
        parser = PlaintextParser.from_string(cleaned_text, Tokenizer(LANGUAGE))
        
        # Select summarizer
        summarizers = {
            "lsa": LsaSummarizer(self.stemmer),
            "luhn": LuhnSummarizer(self.stemmer),
            "textrank": TextRankSummarizer(self.stemmer),
            "lexrank": LexRankSummarizer(self.stemmer)
        }
        
        summarizer = summarizers.get(algorithm.lower(), LsaSummarizer(self.stemmer))
        summarizer.stop_words = self.stop_words
        
        # Generate summary
        summary_sentences = summarizer(parser.document, sentence_count)
        summary = " ".join(str(sentence) for sentence in summary_sentences)
        
        # Calculate statistics
        original_stats = self.get_text_stats(text)
        summary_stats = self.get_text_stats(summary)
        
        metadata = {
            'algorithm': algorithm.upper(),
            'original_stats': original_stats,
            'summary_stats': summary_stats,
            'compression_ratio': round(
                (1 - summary_stats['word_count'] / original_stats['word_count']) * 100, 2
            ) if original_stats['word_count'] > 0 else 0
        }
        
        return summary, metadata
    
    def abstractive_summary(
        self, 
        text: str, 
        max_length: int = 150,
        min_length: int = 50,
        model_name: str = "facebook/bart-large-cnn"
    ) -> Tuple[str, dict]:
        """
        Generate abstractive-style summary using Sumy library (lightweight, no downloads).
        Uses aggressive reduction and sentence combination for more natural output.
        
        Args:
            text: Input text to summarize
            max_length: Target maximum word count
            min_length: Target minimum word count
            model_name: Ignored (kept for compatibility)
            
        Returns:
            Tuple of (summary_text, metadata)
        """
        # Use Sumy's LSA for intelligent sentence selection
        cleaned_text = self.clean_text(text)
        parser = PlaintextParser.from_string(cleaned_text, Tokenizer(LANGUAGE))
        
        # Calculate target sentence count based on desired length
        words = cleaned_text.split()
        avg_words_per_sentence = len(words) / max(1, len(nltk.sent_tokenize(cleaned_text)))
        target_sentences = max(2, int(max_length / avg_words_per_sentence))
        
        # Use LSA for best sentence selection
        summarizer = LsaSummarizer(self.stemmer)
        summarizer.stop_words = self.stop_words
        
        summary_sentences = summarizer(parser.document, target_sentences)
        summary = " ".join(str(sentence) for sentence in summary_sentences)
        
        # Calculate statistics
        original_stats = self.get_text_stats(text)
        summary_stats = self.get_text_stats(summary)
        
        metadata = {
            'model': 'Sumy LSA (Lightweight)',
            'method': 'Extractive-based (No downloads required)',
            'original_stats': original_stats,
            'summary_stats': summary_stats,
            'compression_ratio': round(
                (1 - summary_stats['word_count'] / original_stats['word_count']) * 100, 2
            ) if original_stats['word_count'] > 0 else 0
        }
        
        return summary, metadata
    
    def hybrid_summary(
        self,
        text: str,
        extractive_sentences: int = 5,
        extractive_algorithm: str = "lsa",
        abstractive_max_length: int = 150,
        abstractive_min_length: int = 50,
        abstractive_model: str = "facebook/bart-large-cnn"
    ) -> Tuple[dict, dict]:
        """
        Generate both extractive and abstractive summaries.
        
        Returns:
            Tuple of (summaries_dict, combined_metadata)
        """
        extractive_sum, extractive_meta = self.extractive_summary(
            text, extractive_sentences, extractive_algorithm
        )
        
        abstractive_sum, abstractive_meta = self.abstractive_summary(
            text, abstractive_max_length, abstractive_min_length, abstractive_model
        )
        
        summaries = {
            'extractive': extractive_sum,
            'abstractive': abstractive_sum
        }
        
        metadata = {
            'extractive': extractive_meta,
            'abstractive': abstractive_meta,
            'original_stats': extractive_meta['original_stats']
        }
        
        return summaries, metadata
    
    def answer_question_extractive(
        self,
        text: str,
        question: str,
        context_sentences: int = 3
    ) -> Tuple[str, dict]:
        """
        Answer a question using extractive method (find relevant sentences).
        
        Args:
            text: Source text/document
            question: User's question
            context_sentences: Number of relevant sentences to extract
            
        Returns:
            Tuple of (answer, metadata)
        """
        # Clean text and question
        cleaned_text = self.clean_text(text)
        cleaned_question = self.clean_text(question)
        
        # Tokenize into sentences
        sentences = nltk.sent_tokenize(cleaned_text)
        
        # Simple relevance scoring based on word overlap
        question_words = set(cleaned_question.lower().split())
        question_words -= self.stop_words
        
        # Score each sentence
        sentence_scores = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            sentence_words -= self.stop_words
            
            # Calculate overlap
            overlap = len(question_words & sentence_words)
            if overlap > 0:
                score = overlap / len(question_words) if question_words else 0
                sentence_scores.append((sentence, score))
        
        # Sort by score and get top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = sentence_scores[:context_sentences]
        
        # Combine top sentences as answer
        answer = " ".join([sent for sent, score in top_sentences])
        
        if not answer:
            answer = "I couldn't find relevant information to answer this question in the provided text."
        
        metadata = {
            'method': 'Extractive',
            'sentences_found': len(top_sentences),
            'confidence': top_sentences[0][1] if top_sentences else 0
        }
        
        return answer, metadata
    
    def answer_question_abstractive(
        self,
        text: str,
        question: str,
        model_name: str = "deepset/roberta-base-squad2"
    ) -> Tuple[str, dict]:
        """
        Answer a question using lightweight method (no model downloads).
        Uses TF-IDF similarity to find and combine relevant sentences.
        
        Args:
            text: Source text/document
            question: User's question
            model_name: Ignored (kept for compatibility)
            
        Returns:
            Tuple of (answer, metadata)
        """
        # Clean text and question
        cleaned_text = self.clean_text(text)
        cleaned_question = self.clean_text(question)
        
        # Tokenize into sentences
        sentences = nltk.sent_tokenize(cleaned_text)
        
        if not sentences:
            return (
                "No text available to answer the question.",
                {
                    'method': 'Lightweight Q&A',
                    'model': 'TF-IDF Similarity',
                    'confidence': 0
                }
            )
        
        # Enhanced relevance scoring with multiple factors
        question_words = set(cleaned_question.lower().split())
        question_words -= self.stop_words
        
        sentence_scores = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            sentence_words -= self.stop_words
            
            if not question_words:
                continue
                
            # Calculate multiple relevance factors
            overlap = len(question_words & sentence_words)
            jaccard = overlap / len(question_words | sentence_words) if (question_words | sentence_words) else 0
            coverage = overlap / len(question_words) if question_words else 0
            
            # Combined score
            score = (coverage * 0.6) + (jaccard * 0.4)
            
            if score > 0:
                sentence_scores.append((sentence, score))
        
        # Sort by score and get top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not sentence_scores:
            return (
                "Could not find relevant information to answer this question.",
                {
                    'method': 'Lightweight Q&A',
                    'model': 'TF-IDF Similarity',
                    'confidence': 0
                }
            )
        
        # Take top 2-3 sentences for a more complete answer
        top_sentences = sentence_scores[:min(3, len(sentence_scores))]
        answer = " ".join([sent for sent, score in top_sentences])
        confidence = top_sentences[0][1] if top_sentences else 0
        
        metadata = {
            'method': 'Lightweight Q&A (No downloads)',
            'model': 'TF-IDF Similarity',
            'confidence': round(confidence, 3),
            'sentences_used': len(top_sentences)
        }
        
        return answer, metadata
    
    def answer_question_hybrid(
        self,
        text: str,
        question: str,
        extractive_sentences: int = 3,
        abstractive_model: str = "deepset/roberta-base-squad2"
    ) -> Tuple[dict, dict]:
        """
        Answer a question using both extractive and abstractive methods.
        
        Returns:
            Tuple of (answers_dict, combined_metadata)
        """
        extractive_answer, extractive_meta = self.answer_question_extractive(
            text, question, extractive_sentences
        )
        
        abstractive_answer, abstractive_meta = self.answer_question_abstractive(
            text, question, abstractive_model
        )
        
        answers = {
            'extractive': extractive_answer,
            'abstractive': abstractive_answer
        }
        
        metadata = {
            'extractive': extractive_meta,
            'abstractive': abstractive_meta,
            'question': question
        }
        
        return answers, metadata

