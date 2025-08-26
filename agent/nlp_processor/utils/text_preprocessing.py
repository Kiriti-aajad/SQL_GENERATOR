"""
Text Preprocessing Utilities for Banking Domain NLP
Handles text cleaning, normalization, and banking-specific preprocessing
"""

import re
import string
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing utilities for banking domain queries
    Handles cleaning, normalization, and domain-specific text processing
    """
    
    def __init__(self):
        """Initialize text preprocessor with banking domain settings"""
        
        # Banking abbreviations to expand
        self.banking_abbreviations = {
            'npa': 'non performing asset',
            'crar': 'capital to risk weighted assets ratio',
            'casa': 'current account savings account',
            'ctpt': 'counterparty',
            'fac': 'facility',
            'app': 'application',
            'rbi': 'reserve bank of india',
            'emi': 'equated monthly installment',
            'fd': 'fixed deposit',
            'rd': 'recurring deposit',
            'kyc': 'know your customer',
            'aml': 'anti money laundering'
        }
        
        # Common contractions
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        # Currency patterns
        self.currency_patterns = [
            (r'\brs\.?\s*(\d+)', r'rupees \1'),
            (r'₹\s*(\d+)', r'rupees \1'),
            (r'\binr\s*(\d+)', r'rupees \1'),
        ]
        
        # Scale patterns
        self.scale_patterns = [
            (r'(\d+)\s*k\b', r'\1 thousand'),
            (r'(\d+)\s*m\b', r'\1 million'),
            (r'(\d+)\s*b\b', r'\1 billion'),
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning: lowercase, remove extra whitespace
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions in text
        
        Args:
            text: Input text with contractions
            
        Returns:
            Text with expanded contractions
        """
        if not text:
            return ""
        
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def expand_banking_abbreviations(self, text: str) -> str:
        """
        Expand banking-specific abbreviations
        
        Args:
            text: Input text with banking abbreviations
            
        Returns:
            Text with expanded abbreviations
        """
        if not text:
            return ""
        
        for abbr, expansion in self.banking_abbreviations.items():
            pattern = rf'\b{re.escape(abbr)}\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_currency(self, text: str) -> str:
        """
        Normalize currency representations
        
        Args:
            text: Input text with currency mentions
            
        Returns:
            Text with normalized currency
        """
        if not text:
            return ""
        
        for pattern, replacement in self.currency_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_scales(self, text: str) -> str:
        """
        Normalize numeric scales (k, m, b)
        
        Args:
            text: Input text with scales
            
        Returns:
            Text with normalized scales
        """
        if not text:
            return ""
        
        for pattern, replacement in self.scale_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def remove_punctuation(self, text: str, keep_chars: str = "") -> str:
        """
        Remove punctuation from text
        
        Args:
            text: Input text
            keep_chars: Characters to keep (not remove)
            
        Returns:
            Text without punctuation
        """
        if not text:
            return ""
        
        # Create translation table
        translator = str.maketrans('', '', string.punctuation)
        
        # Remove specified characters from translation
        for char in keep_chars:
            translator[ord(char)] = char # type: ignore
        
        return text.translate(translator)
    
    def remove_extra_whitespace(self, text: str) -> str:
        """
        Remove extra whitespace from text
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        if not text:
            return ""
        
        return re.sub(r'\s+', ' ', text).strip()
    
    def preprocess(self, text: str, 
                  expand_contractions: bool = True,
                  expand_abbreviations: bool = True,
                  normalize_currency: bool = True,
                  normalize_scales: bool = True,
                  remove_punctuation: bool = False,
                  clean_text: bool = True) -> str:
        """
        Complete text preprocessing pipeline
        
        Args:
            text: Input text to preprocess
            expand_contractions: Whether to expand contractions
            expand_abbreviations: Whether to expand banking abbreviations
            normalize_currency: Whether to normalize currency
            normalize_scales: Whether to normalize scales
            remove_punctuation: Whether to remove punctuation
            clean_text: Whether to apply basic cleaning
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        try:
            processed_text = text
            
            # Apply preprocessing steps
            if clean_text:
                processed_text = self.clean_text(processed_text)
            
            if expand_contractions:
                processed_text = self.expand_contractions(processed_text)
            
            if expand_abbreviations:
                processed_text = self.expand_banking_abbreviations(processed_text)
            
            if normalize_currency:
                processed_text = self.normalize_currency(processed_text)
            
            if normalize_scales:
                processed_text = self.normalize_scales(processed_text)
            
            if remove_punctuation:
                processed_text = self.remove_punctuation(processed_text)
            
            # Final whitespace cleanup
            processed_text = self.remove_extra_whitespace(processed_text)
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            return text  # Return original text on error
    
    def preprocess_batch(self, texts: List[str], **kwargs) -> List[str]:
        """
        Preprocess a batch of texts
        
        Args:
            texts: List of texts to preprocess
            **kwargs: Arguments for preprocess method
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text, **kwargs) for text in texts]


# Convenience function for quick access
def preprocess_text(text: str, **kwargs) -> str:
    """
    Convenience function for text preprocessing
    
    Args:
        text: Input text
        **kwargs: Arguments for TextPreprocessor.preprocess
        
    Returns:
        Preprocessed text
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess(text, **kwargs)


# Create global preprocessor instance
text_preprocessor = TextPreprocessor()
