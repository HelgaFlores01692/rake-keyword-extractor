"""rake-keyword-extractor: Rapid Automatic Keyword Extraction.

A minimal, dependency-free implementation of the RAKE algorithm
described by Rose, Engel, Cramer & Cowley (2010). It extracts
candidate keyphrases and scores them using word co-occurrence
within the text.
"""

from .core import Rake, rake, extract_keywords, RakeError

__all__ = ["Rake", "rake", "extract_keywords", "RakeError"]
__version__ = "1.0.0"
