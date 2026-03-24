"""
Structured Extractors — PDF extraction for Kredi İstihbarat Raporu.

Two strategies:
  1. Legacy regex: CreditIntelligencePDFExtractor (pdfplumber + regex, 3000+ satır)
  2. VLM-based:   VLMPDFExtractor (Qwen3-VL vision model, sayfa görüntüsünden markdown)
"""

from .credit_intelligence_pdf_extractor import CreditIntelligencePDFExtractor
from .vlm_pdf_extractor import VLMPDFExtractor

__all__ = ["CreditIntelligencePDFExtractor", "VLMPDFExtractor"]
