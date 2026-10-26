# rake-keyword-extractor

Extracts candidate keyphrases from English text using the Rapid Automatic Keyword Extraction (RAKE) algorithm. Pure standard library, no dependencies.

```python
from rake_keyword_extractor import Rake, rake, extract_keywords

extractor = Rake()
results = extractor.extract("Keyword extraction is the task of "
                            "automatically identifying keyphrases in text.")
for phrase, score in results:
    print(f"{score:.3f}  {phrase}")

# convenience helpers
top10 = rake("your text here", top_n=10)
just_phrases = extract_keywords("your text here")
```

## Why this exists

There are many RAKE implementations. Most pull in `nltk` or `numpy` for stopword lists or array math that this algorithm does not need. The original RAKE paper defines scoring purely in terms of word co-occurrence counts — arithmetic you can do with a `dict`. This library is for projects that want the algorithm without a transitive dependency tree.

The trade-off: the built-in stopword list is English-only and manually curated. If you need another language, pass your own stopword set to `Rake(stopwords=[...])`.

## Scoring

Word score = degree / frequency, where degree is the sum of (phrase_length − 1) over every phrase containing the word. Phrase score = sum of member word scores. Single-word phrases contribute degree 0, so their score is 0.0 — they survive only if you keep the default `min_score=0.0`. This matches the original paper.

## Edge cases you will hit

- **Numbers are stopwords.** Bare integers and decimals (`42`, `3.14`) are treated as stopwords by default, matching the reference behaviour. Override by subclassing if you need them as keyphrases.
- **Hyphens and apostrophes are preserved inside tokens.** `state-of-the-art` stays one token; `student's` stays one token. Only sentence-terminal punctuation (`. ! ? ; :`) acts as a phrase boundary.
- **Empty or all-stopword input** returns `[]` rather than raising. Non-string or empty-string input raises `RakeError`.
- **Tie-breaking is deterministic.** Equal scores are ordered by phrase length descending, then alphabetically, so the same input always produces the same output.

## API

- `Rake(stopwords=None)` — constructor. `stopwords` is an iterable of strings, matched case-insensitively.
- `Rake.extract(text, top_n=None, min_score=0.0)` → `list[tuple[str, float]]`, sorted by score descending.
- `rake(text, top_n=10, stopwords=None)` → `list[tuple[str, float]]`, one-shot wrapper.
- `extract_keywords(text, stopwords=None)` → `list[str]`, phrases only, no scores.

Run tests with `PYTHONPATH=src python -m unittest discover -s tests`.
