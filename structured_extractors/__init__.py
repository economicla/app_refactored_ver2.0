"""
Structured Extractors — Markdown-oriented PDF extraction for Kredi İstihbarat Raporu.
Single entry: PDF → structured extraction → markdown-style text → chunking → Vector DB.
"""

from .credit_intelligence_pdf_extractor import CreditIntelligencePDFExtractor

__all__ = ["CreditIntelligencePDFExtractor"]
