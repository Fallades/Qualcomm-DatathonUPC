import unittest

from copilot.topics import TOPICS, classify_problem, classify_text


class TopicClassificationTests(unittest.TestCase):
    def test_geometry(self):
        result = classify_text("A square has a perimeter of 36 cm. What is its area?")
        self.assertEqual(result["topic"], "geometry")

    def test_catalan_combinatorics(self):
        result = classify_text("De quantes maneres podem ordenar cinc llibres?")
        self.assertEqual(result["topic"], "combinatorics")

    def test_manual_override_wins(self):
        problem = {"id": "p1", "language": "eng", "text": "A triangle has three sides."}
        tagged = classify_problem(problem, {"p1|eng": "geometry"})
        self.assertEqual(tagged["topic"], "geometry")
        self.assertEqual(tagged["topic_source"], "manual")

    def test_invalid_override_fails(self):
        problem = {"id": "p1", "language": "eng", "text": "A triangle."}
        with self.assertRaises(ValueError):
            classify_problem(problem, {"p1|eng": "not-a-topic"})

    def test_unknown_text_goes_to_review(self):
        result = classify_text("A strange statement with no mathematical keywords.")
        self.assertEqual(result["topic"], "other")
        self.assertEqual(result["topic_confidence"], "low")


if __name__ == "__main__":
    unittest.main()
