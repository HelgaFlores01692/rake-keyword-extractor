"""Tests for the RAKE extractor.

Run with::

    PYTHONPATH=src python -m unittest discover -s tests

These tests assert on the behaviour the implementation actually claims
(ranked keyphrases, deterministic ordering, custom stopwords, edge-case
handling of empty/non-string input). No external resources are used.
"""

import unittest

from rake_keyword_extractor.core import Rake, RakeError, rake, extract_keywords


class TestRakeBasic(unittest.TestCase):
    """Happy-path and core scoring behaviour."""

    def test_simple_text_returns_phrases(self):
        text = (
            "Compatibility of systems of linear constraints over the "
            "domain of rational numbers."
        )
        results = Rake().extract(text)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for phrase, score in results:
            self.assertIsInstance(phrase, str)
            self.assertIsInstance(score, float)

    def test_results_are_sorted_by_score_descending(self):
        text = (
            "keyword extraction is the task of keyword extraction. "
            "Automatic keyword extraction tools are useful. "
            "Keyword extraction matters."
        )
        results = Rake().extract(text)
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_deterministic_tie_breaking(self):
        # Two phrases that score equally should be ordered deterministically:
        # by length desc, then alphabetical.
        text = "alpha phrase beta phrase. gamma phrase delta phrase."
        r1 = Rake().extract(text)
        r2 = Rake().extract(text)
        self.assertEqual(r1, r2)

    def test_multiword_phrase_preserved(self):
        text = "natural language processing is a subfield of artificial intelligence."
        results = Rake().extract(text)
        phrases = [p for p, _ in results]
        # "natural language processing" should appear as a single phrase.
        self.assertIn("natural language processing", phrases)

    def test_stopwords_are_removed_from_phrases(self):
        text = "the cat sat on the mat."
        results = Rake().extract(text)
        phrases = [p for p, _ in results]
        for p in phrases:
            self.assertNotIn("the", p.split())
            self.assertNotIn("on", p.split())

    def test_hyphenated_compound_preserved(self):
        text = "state-of-the-art techniques improve retrieval."
        results = Rake().extract(text)
        phrases = [p for p, _ in results]
        self.assertTrue(any("state-of-the-art" in p for p in phrases))

    def test_apostrophe_word_preserved(self):
        # "don't" is a stopword in our list; ensure apostrophes in other
        # words (e.g. possessives) are kept intact, not split.
        text = "the student's project was well received."
        results = Rake().extract(text)
        phrases = [p for p, _ in results]
        # "student's" should survive as a token (the 's is part of the word).
        self.assertTrue(any("student's" in p for p in phrases))


class TestRakeScoring(unittest.TestCase):
    """Scoring formula correctness."""

    def test_repeated_phrase_scores_higher(self):
        # "keyword extraction" appears three times on its own; the longer
        # phrase "keyword extraction works well" appears once. Because
        # "keyword" and "extraction" gain extra degree from the longer
        # phrase, that phrase outscores the bare two-word one under the
        # original RAKE formula. We assert the dominant phrase ranks first.
        text = (
            "keyword extraction is important. "
            "keyword extraction is useful. "
            "keyword extraction works well."
        )
        results = Rake().extract(text)
        # The top phrase must contain "keyword extraction".
        self.assertIn("keyword extraction", results[0][0])
        self.assertGreater(results[0][1], 0.0)

    def test_single_word_phrase_score_is_word_score(self):
        # Under the original RAKE formula, a single-word phrase's score is
        # the word score (degree / frequency) of that word. When the word
        # also appears inside a multi-word phrase, it gains degree from
        # that phrase, so its score is non-zero.
        text = "research. important research. machine learning research."
        results = Rake().extract(text)
        scored = dict(results)
        # "research" appears in "important research" (degree +1) and in
        # "machine learning research" (degree +2), plus once alone (degree
        # +0). Frequency = 3, degree = 3, score = 1.0.
        self.assertIn("research", scored)
        self.assertEqual(scored["research"], 1.0)
        # "machine learning research" is a three-word phrase with degree > 0,
        # so its score is positive.
        self.assertIn("machine learning research", scored)
        self.assertGreater(scored["machine learning research"], 0.0)


class TestRakeStopwords(unittest.TestCase):
    """Custom stopword handling."""

    def test_custom_stopwords_remove_terms(self):
        text = "alpha beta gamma delta epsilon."
        r_default = Rake().extract(text)
        r_custom = Rake(stopwords=["beta"]).extract(text)
        default_phrases = {p for p, _ in r_default}
        custom_phrases = {p for p, _ in r_custom}
        # Default keeps "alpha beta gamma delta epsilon" as one phrase.
        self.assertIn("alpha beta gamma delta epsilon", default_phrases)
        # Custom splits on "beta".
        self.assertNotIn("alpha beta gamma delta epsilon", custom_phrases)

    def test_stopwords_are_case_insensitive(self):
        text = "Alpha Beta gamma."
        r = Rake(stopwords=["ALPHA"]).extract(text)
        phrases = {p for p, _ in r}
        # "alpha" removed, so the phrase cannot contain it.
        for p in phrases:
            self.assertNotIn("alpha", p.split())

    def test_numeric_tokens_are_stopwords(self):
        text = "there are 42 cats and 7 dogs here."
        results = Rake().extract(text)
        phrases = [p for p, _ in results]
        for p in phrases:
            self.assertNotIn("42", p.split())
            self.assertNotIn("7", p.split())

    def test_decimal_numbers_are_stopwords(self):
        text = "the value is 3.14 and the count is 5."
        results = Rake().extract(text)
        phrases = [p for p, _ in results]
        for p in phrases:
            self.assertNotIn("3.14", p.split())

    def test_stopwords_property_is_a_copy(self):
        r = Rake()
        sw = r.stopwords
        sw.add("hacked")
        self.assertNotIn("hacked", r.stopwords)


class TestRakeEdgeCases(unittest.TestCase):
    """Input validation and awkward inputs."""

    def test_empty_string_raises(self):
        with self.assertRaises(RakeError):
            Rake().extract("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(RakeError):
            Rake().extract("   \n  \t  ")

    def test_non_string_raises(self):
        with self.assertRaises(RakeError):
            Rake().extract(123)  # type: ignore[arg-type]

    def test_only_stopwords_returns_empty_list(self):
        text = "the and of to in is"
        results = Rake().extract(text)
        self.assertEqual(results, [])

    def test_only_numbers_returns_empty_list(self):
        text = "42 7 100 3.14"
        results = Rake().extract(text)
        self.assertEqual(results, [])

    def test_top_n_limits_results(self):
        text = (
            "keyword extraction works. "
            "information retrieval matters. "
            "natural language processing helps. "
            "machine learning advances. "
            "data mining grows."
        )
        all_results = Rake().extract(text)
        self.assertGreater(len(all_results), 3)
        limited = Rake().extract(text, top_n=3)
        self.assertEqual(len(limited), 3)
        # Top-3 must be the first three of the full ranking.
        self.assertEqual(limited, all_results[:3])

    def test_top_n_zero_returns_empty(self):
        text = "some interesting keywords here."
        results = Rake().extract(text, top_n=0)
        self.assertEqual(results, [])

    def test_top_n_negative_raises(self):
        text = "some interesting keywords here."
        with self.assertRaises(RakeError):
            Rake().extract(text, top_n=-1)

    def test_min_score_filters(self):
        text = (
            "keyword extraction is good. "
            "keyword extraction is great. "
            "the cat."
        )
        all_results = Rake().extract(text)
        # "cat" has score 0.0 (single word appearing only alone). Filter it out.
        filtered = Rake().extract(text, min_score=0.5)
        for _, score in filtered:
            self.assertGreaterEqual(score, 0.5)
        self.assertLess(len(filtered), len(all_results))

    def test_punctuation_only_returns_empty_list(self):
        text = "!!! ??? ... ;;;"
        results = Rake().extract(text)
        self.assertEqual(results, [])


class TestConvenienceFunctions(unittest.TestCase):
    """The module-level helpers."""

    def test_rake_function_returns_top_n(self):
        # "keyword extraction" is repeated across two sentences, so the
        # words "keyword" and "extraction" gain co-occurrence degree.
        # "information retrieval" appears only once. To get a clear winner
        # we add a third sentence that also features "keyword extraction"
        # within a longer phrase, boosting its words' degree further so
        # that "keyword extraction" strictly outscores "information retrieval".
        text = (
            "keyword extraction is important. "
            "keyword extraction is useful. "
            "keyword extraction works well. "
            "information retrieval is different."
        )
        results = rake(text, top_n=2)
        self.assertLessEqual(len(results), 2)
        self.assertIn("keyword extraction", results[0][0])
        self.assertGreater(results[0][1], results[1][1])

    def test_rake_function_fewer_phrases_than_top_n(self):
        text = "one phrase only."
        results = rake(text, top_n=10)
        self.assertLessEqual(len(results), 1)

    def test_rake_function_custom_stopwords(self):
        text = "alpha beta gamma."
        results = rake(text, top_n=5, stopwords=["beta"])
        phrases = {p for p, _ in results}
        self.assertNotIn("alpha beta gamma", phrases)

    def test_extract_keywords_returns_strings_only(self):
        text = "natural language processing is useful."
        result = extract_keywords(text)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, str)

    def test_extract_keywords_consistent_with_rake_class(self):
        text = (
            "keyword extraction is important. "
            "keyword extraction works well."
        )
        via_class = [p for p, _ in Rake().extract(text)]
        via_func = extract_keywords(text)
        self.assertEqual(via_class, via_func)


if __name__ == "__main__":
    unittest.main()
