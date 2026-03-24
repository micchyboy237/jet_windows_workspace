import unittest

from sentence_matcher_ja import FuzzyMatchResult, fuzzy_shortest_best_match


class TestFuzzyShortestBestMatch(unittest.TestCase):
    def test_perfect_match(self):
        result: FuzzyMatchResult = fuzzy_shortest_best_match(
            "潮ひ狩りはえ", ["去年の初めての潮ひ狩りはえあうん楽しかったよ"]
        )
        self.assertEqual(result["match"], "潮ひ狩りはえ")
        self.assertAlmostEqual(result["score"], 100.0, places=1)
        self.assertGreater(result["start"], -1)
        self.assertEqual(result["end"] - result["start"], len(result["match"]))
        self.assertIn("text", result)

    def test_with_typo(self):
        result: FuzzyMatchResult = fuzzy_shortest_best_match(
            "去る初めての消ひ狩りはえ",
            "去る初めての潮ひ狩りはえあうん楽しかったよそうケン君は？",
        )
        self.assertEqual(result["match"], "去る初めての潮ひ狩りはえ")
        self.assertGreaterEqual(result["score"], 90.0)

    def test_multiple_texts_returns_best(self):
        texts = [
            "昨日は雨だった",
            "去る初めての潮ひ狩りはえあうん楽しかったよ",
            "全く関係ない文章です",
        ]
        result: FuzzyMatchResult = fuzzy_shortest_best_match(
            "去る初めての消ひ狩りはえ", texts
        )
        self.assertEqual(result["match"], "去る初めての潮ひ狩りはえ")
        self.assertGreaterEqual(result["score"], 90.0)
        self.assertEqual(result["text"], texts[1])

    def test_shortest_on_score_tie(self):
        result: FuzzyMatchResult = fuzzy_shortest_best_match(
            "楽しかった", ["楽しかったよそう楽しかったね"]
        )
        self.assertEqual(result["match"], "楽しかった")

    def test_no_good_match(self):
        result: FuzzyMatchResult = fuzzy_shortest_best_match(
            "全く関係ない文字列です",
            ["去る初めての潮ひ狩りはえあうん楽しかったよ"],
            score_cutoff=30,
        )
        self.assertLess(result["score"], 30)
        self.assertEqual(result["start"], -1)

    def test_empty_inputs(self):
        empty: FuzzyMatchResult = fuzzy_shortest_best_match("", "abc")
        self.assertEqual(empty["match"], "")
        self.assertEqual(empty["score"], 0.0)
        self.assertEqual(empty["start"], -1)
        self.assertEqual(empty.get("text", ""), "")

    def test_japanese_punctuation(self):
        result: FuzzyMatchResult = fuzzy_shortest_best_match(
            "ケン君は？", ["楽しかったよそうケン君は？明日も行こうか"]
        )
        self.assertEqual(result["match"], "ケン君は？")
        self.assertGreaterEqual(result["score"], 95.0)


if __name__ == "__main__":
    unittest.main()
