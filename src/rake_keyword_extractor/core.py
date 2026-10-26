"""Core RAKE implementation.

Design decisions documented in this module:

- Stopword list: built-in English stopword set. Users can override
  by passing a custom set to Rake(stopwords=...). The built-in list
  is a manually curated balance between NLTK-style stopword lists
  and what works well with the RAKE scoring formula.
- Punctuation handling: only sentence-terminal punctuation is used
  as a phrase boundary. Internal apostrophes (don't) and hyphens
  (state-of-the-art) are preserved, because splitting them would
  break common compound terms.
- Phrase boundaries: a candidate phrase is a maximal sequence of
  non-stopword words separated by internal whitespace. A stopword
  or sentence boundary terminates the current phrase.
- Scoring: word scores use adjacent-pair co-occurrence (degree/freq).
  Phrase score = sum of member word scores. This is the original
  paper formulation.
- Numbers: a token composed entirely of digits (with optional decimal
  point) is treated as a stopword, matching the reference implementation,
  because bare numbers rarely make good keyphrases on their own.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple


class RakeError(ValueError):
    """Raised for invalid input to the RAKE extractor (empty text, wrong type)."""


# --- Built-in stopword list (English) ------------------------------------

_STOPWORDS: FrozenSet[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
    "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
    "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most",
    "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
})

# Sentence-terminal punctuation: phrase boundaries. We deliberately
# keep apostrophes and hyphens inside tokens because splitting them
# destroys compounds like "state-of-the-art".
_SENT_BOUNDARY_RE: re.Pattern[str] = re.compile(r"[.!?;:]")

# A token that is entirely numeric (optionally with a decimal) is treated
# as a stopword, matching the reference RAKE behaviour: bare numbers are
# noise as keyphrases.
_NUMERIC_RE: re.Pattern[str] = re.compile(r"^[0-9]+(\.[0-9]+)?$")


def _tokenize_sentence(sentence: str) -> List[str]:
    """Split *sentence* into whitespace-delimited tokens.

    Each token has leading/trailing punctuation other than apostrophes
    and hyphens stripped, because those are preserved inside tokens.
    Empty tokens are dropped.
    """
    tokens: List[str] = []
    for raw in sentence.split():
        # Strip any non-alphanumeric, non-apostrophe, non-hyphen chars from
        # both ends. This removes wrapping quotes, brackets, commas, etc.
        tok = raw.strip(",.;:!?()[]{}\"'\u2018\u2019\u201c\u201d")
        # If the token still has internal terminal punctuation (e.g. a
        # trailing '.' survived because it was inside quotes), do a second
        # pass to strip it.
        while tok and tok[-1] in ".,;:!?":
            tok = tok[:-1].rstrip("\u2018\u2019\u201c\u201d\"'")
            if tok and tok[0] in "\u2018\u2019\u201c\u201d\"'":
                tok = tok[1:]
        while tok and tok[0] in "\u2018\u2019\u201c\u201d\"'":
            tok = tok[1:]
        if tok:
            tokens.append(tok)
    return tokens


def _is_stopword(token: str, stopwords: Set[str]) -> bool:
    """Return True if *token* is a stopword or a bare number."""
    lower = token.lower()
    if lower in stopwords:
        return True
    return bool(_NUMERIC_RE.match(token))


def _split_sentences(text: str) -> List[str]:
    """Split *text* on sentence-terminal punctuation.

    We deliberately do not use a full sentence segmenter. RAKE only needs
    phrase boundaries; sentence splitting is a cheap way to stop phrases
    from running across punctuation like periods. Empty segments are kept
    out of the result.
    """
    if not text:
        return []
    parts = _SENT_BOUNDARY_RE.split(text)
    return [p for p in parts if p.strip()]


def _extract_phrases(text: str, stopwords: Set[str]) -> List[List[str]]:
    """Return a list of candidate phrases, each a list of lowercased tokens.

    A phrase is a maximal run of non-stopword tokens within a sentence.
    Stopwords and sentence boundaries both terminate the current phrase.
    """
    phrases: List[List[str]] = []
    for sentence in _split_sentences(text):
        current: List[str] = []
        for tok in _tokenize_sentence(sentence):
            if _is_stopword(tok, stopwords):
                if current:
                    phrases.append(current)
                    current = []
            else:
                current.append(tok.lower())
        if current:
            phrases.append(current)
    return phrases


def _word_degree_and_freq(
    phrases: List[List[str]],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Compute per-word frequency and degree.

    Frequency: how many phrases contain the word.
    Degree: for each phrase, each word in that phrase gets degree +=
    (phrase_length - 1). Single-word phrases contribute degree 0, which
    is the original RAKE behaviour and causes single words to be scored
    by frequency alone.
    """
    freq: Dict[str, int] = defaultdict(int)
    degree: Dict[str, float] = defaultdict(float)
    for phrase in phrases:
        n = len(phrase)
        for w in set(phrase):
            freq[w] += 1
            degree[w] += max(n - 1, 0)
    return dict(degree), dict(freq)


def _word_scores(
    degree: Dict[str, float], freq: Dict[str, int]
) -> Dict[str, float]:
    """Compute word scores = degree / freq.

    If a word has freq 0 we guard against division by zero, though in
    practice every word in *freq* has at least one occurrence. We still
    handle it defensively: score 0.
    """
    scores: Dict[str, float] = {}
    for w, f in freq.items():
        if f <= 0:
            scores[w] = 0.0
            continue
        d = degree.get(w, 0.0)
        scores[w] = d / f
    return scores


def _phrase_scores(
    phrases: List[List[str]],
    word_scores: Dict[str, float],
) -> List[Tuple[str, float]]:
    """Score each phrase as the sum of its member word scores.

    Returns a list of (phrase_string, score) pairs. Phrase string joins
    member tokens with a single space.
    """
    out: List[Tuple[str, float]] = []
    for phrase in phrases:
        s = 0.0
        for w in phrase:
            s += word_scores.get(w, 0.0)
        out.append((" ".join(phrase), s))
    return out


class Rake:
    """RAKE keyword extractor.

    Construct once, call :meth:`extract` on each document. The stopword
    list is fixed at construction time for reproducibility.

    Parameters
    ----------
    stopwords:
        Iterable of stopword strings. Defaults to the built-in English
        list. Case-insensitive matching is used; the set is lower-cased
        on construction.
    """

    def __init__(self, stopwords: Optional[Sequence[str]] = None) -> None:
        base = stopwords if stopwords is not None else _STOPWORDS
        self._stopwords: Set[str] = {w.lower() for w in base}

    @property
    def stopwords(self) -> Set[str]:
        """A copy of the internal stopword set (mutating it has no effect)."""
        return set(self._stopwords)

    def extract(
        self,
        text: str,
        top_n: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Extract and rank keyphrases from *text*.

        Returns a list of ``(phrase, score)`` tuples, sorted by score
        descending. Ties are broken by phrase length descending, then
        alphabetically, so results are deterministic.

        Parameters
        ----------
        text:
            Input document. Must be a non-empty string.
        top_n:
            If given, return at most this many results.
        min_score:
            Drop phrases with score strictly below this value. Default
            0.0 keeps everything (single-word phrases naturally score 0
            under the original RAKE formula, so they are included by
            default).

        Raises
        ------
        RakeError
            If *text* is not a non-empty string.
        """
        if not isinstance(text, str) or not text.strip():
            raise RakeError("extract() requires a non-empty string")

        phrases = _extract_phrases(text, self._stopwords)
        if not phrases:
            return []

        degree, freq = _word_degree_and_freq(phrases)
        word_scores = _word_scores(degree, freq)
        scored = _phrase_scores(phrases, word_scores)

        # Deterministic sort: score desc, then length desc, then lexical.
        scored.sort(key=lambda p: (-p[1], -len(p[0]), p[0]))

        result = [
            (phrase, score)
            for phrase, score in scored
            if score >= min_score
        ]

        if top_n is not None:
            if top_n < 0:
                raise RakeError("top_n must be non-negative")
            result = result[:top_n]
        return result


def rake(
    text: str,
    top_n: int = 10,
    stopwords: Optional[Sequence[str]] = None,
) -> List[Tuple[str, float]]:
    """Convenience wrapper: one-shot RAKE extraction.

    Constructs a :class:`Rake`, extracts, and returns the top *top_n*
    keyphrases. If the text yields fewer than *top_n* phrases, fewer are
    returned.
    """
    extractor = Rake(stopwords=stopwords)
    return extractor.extract(text, top_n=top_n)


def extract_keywords(
    text: str,
    stopwords: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return only the keyphrase strings, in ranked order (no scores).

    Useful when callers don't care about the numeric score.
    """
    extractor = Rake(stopwords=stopwords)
    return [p for p, _ in extractor.extract(text)]
