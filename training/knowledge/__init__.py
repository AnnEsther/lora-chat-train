"""training/knowledge/ — Knowledge extraction, synthesis, and validation.

This module handles:
- Extracting topics/factors from conversation pairs
- Normalizing to structured knowledge records
- Synthesizing canonical Q&A using the model server
- Validating synthesized content
- Merging with the knowledge corpus
"""

from training.knowledge.extractor import KnowledgeExtractor
from training.knowledge.normalizer import KnowledgeNormalizer
from training.knowledge.synthesizer import QASynthesizer
from training.knowledge.validator import QAValidator
from training.knowledge.corpus import CorpusManager

__all__ = [
    "KnowledgeExtractor",
    "KnowledgeNormalizer",
    "QASynthesizer",
    "QAValidator",
    "CorpusManager",
]
