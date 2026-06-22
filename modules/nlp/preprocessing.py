"""NLP preprocessing utilities for AI-Answer-Evaluation-System (Member 2)

Provides basic preprocessing functions: tokenization, stopword removal, lemmatization,
and simple feature extraction. Uses spaCy and NLTK.

Usage:
  - Install dependencies (see modules/nlp/requirements-nlp.txt)
  - Run `python -m spacy download en_core_web_sm`

Functions:
  - init_nlp(): load spaCy model
  - preprocess_text(text, ...): returns tokens, lemmas, cleaned_text and features

"""
from typing import Dict, List
import re

# lazy imports to avoid heavy imports at module import time
_nlp = None
_stopwords = None


def init_nlp(spacy_model: str = "en_core_web_sm"):
    """Load and cache spaCy model. Call this once at startup.

    Note: run `python -m spacy download en_core_web_sm` beforehand if not present.
    """
    global _nlp
    import spacy

    if _nlp is None:
        _nlp = spacy.load(spacy_model, disable=["parser"])  # we don't need parsing for basic preprocessing
    return _nlp


def _init_stopwords():
    global _stopwords
    if _stopwords is None:
        import nltk
        try:
            _stopwords = set(nltk.corpus.stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            _stopwords = set(nltk.corpus.stopwords.words("english"))
    return _stopwords


def clean_text_basic(text: str) -> str:
    """Basic text normalization: normalize whitespace and remove control chars."""
    if not isinstance(text, str):
        text = str(text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # remove non-printable/control characters
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    return text


def preprocess_text(text: str,
                    lowercase: bool = True,
                    remove_stopwords: bool = True,
                    lemmatize: bool = True) -> Dict:
    """Preprocess `text` and return a dict with tokens, lemmas, cleaned_text and simple features.

    Returns:
      {
        'raw': original_text,
        'cleaned': cleaned_text,
        'tokens': [...],
        'lemmas': [...],
        'length_tokens': int,
        'stopword_count': int
      }

    This function uses spaCy for tokenization and lemmatization (if enabled) and NLTK for stopwords.
    """
    if _nlp is None:
        init_nlp()
    if _stopwords is None:
        _init_stopwords()

    raw = text
    text = clean_text_basic(text)
    if lowercase:
        text_proc = text.lower()
    else:
        text_proc = text

    doc = _nlp(text_proc)

    tokens: List[str] = []
    lemmas: List[str] = []
    sw_count = 0

    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        token_text = tok.text.strip()
        if token_text == "":
            continue
        tokens.append(token_text)
        lemma = tok.lemma_.strip() if lemmatize else token_text
        lemmas.append(lemma)
        if token_text.lower() in _stopwords:
            sw_count += 1

    cleaned_text = " ".join(tokens)

    return {
        "raw": raw,
        "cleaned": cleaned_text,
        "tokens": tokens,
        "lemmas": lemmas,
        "length_tokens": len(tokens),
        "stopword_count": sw_count,
    }


def sentence_split(text: str) -> List[str]:
    """Simple sentence splitter using spaCy sentencizer if available."""
    if _nlp is None:
        init_nlp()
    try:
        sdoc = _nlp(text)
        return [sent.text.strip() for sent in sdoc.sents]
    except Exception:
        # fallback naive splitter
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
