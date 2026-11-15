import json
import os
import tempfile
import unittest
from unittest.mock import patch

from copilot.topics import classify_problem
from copilot.tutor import AdaptiveTutor, TutorError


def sample_problems():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_problems.jsonl")
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    tagged = [classify_problem(row) for row in rows]
    # Give the smoke-test bank enough repeated items to exercise adaptation.
    for i in range(4):
        source = dict(tagged[2])
        source["id"] = f"geometry-extra-{i}"
        source["level"] = min(7, 2 + i)
        tagged.append(source)
    return tagged


class TutorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.temp.name, "students.db")
        self.tutor = AdaptiveTutor(sample_problems(), self.db)

    def tearDown(self):
        self.temp.cleanup()

    def test_start_answer_and_profile(self):
        problem = self.tutor.start("student-test", language="eng", start_level=0)
        self.assertIsNotNone(problem)
        self.assertNotIn("label", problem)
        result = self.tutor.submit_answer("student-test", self.tutor.by_key[
            self.tutor.key_of(problem)]["label"])
        self.assertTrue(result["correct"])
        self.assertEqual(result["profile"]["total"], 1)
        self.assertEqual(result["profile"]["correct"], 1)

    def test_hint_steps_and_three_hint_limit(self):
        problem = self.tutor.start("hint-student", language="cat", start_level=1)
        generated = [
            {"title": f"Pista {i}", "explanation": f"Explicació específica {i} amb dades de l'exercici.",
             "visual": f"DADES\n{3+i} → relació", "action": f"Prova el pas {i}", "kind": "ai"}
            for i in (1, 2, 3)
        ]
        with patch.object(self.tutor, "_generate_hint", side_effect=generated) as generator:
            for stage in (1, 2, 3):
                hint = self.tutor.hint("hint-student")
                self.assertEqual(hint["stage"], stage)
                self.assertEqual(hint["remaining"], 3 - stage)
                self.assertEqual(hint["kind"], "ai")
                self.assertIn("visual", hint)
            with self.assertRaises(TutorError):
                self.tutor.hint("hint-student")
        self.assertEqual(generator.call_count, 3)
        self.assertEqual(generator.call_args_list[0].args, (self.tutor.by_key[self.tutor.key_of(problem)], 1))
        self.assertEqual(generator.call_args_list[2].args[1], 3)

    def test_hint_uses_exercise_specific_fallback_when_model_is_down(self):
        problem = self.tutor.start("fallback-student", language="eng", start_level=1)
        with patch.object(self.tutor, "_generate_hint", side_effect=OSError("model unavailable")):
            hint = self.tutor.hint("fallback-student")
        self.assertEqual(hint["kind"], "fallback")
        self.assertTrue(hint["explanation"])
        self.assertTrue(hint["visual"])
        self.assertTrue(hint["action"])
        self.assertIn("model", hint["note"].lower())
        self.assertIn("?", hint["visual"])

    def test_fallback_hint_is_long_localised_and_exercise_specific(self):
        arithmetic = next(p for p in self.tutor.problems if p.get("id") == "sample-2")
        hint = self.tutor._fallback_hint(arithmetic, 1)
        self.assertGreaterEqual(len(hint["explanation"].split()), 45)
        self.assertIn("2 + 3 × 4", hint["visual"])
        self.assertIn("OPERATION ORDER", hint["visual"])
        self.assertNotIn("Aritmètica", hint["visual"])
        self.assertIn("local model", hint["note"].lower())

        geometry = next(p for p in self.tutor.problems if p.get("id") == "sample-3")
        geometry_hint = self.tutor._fallback_hint(geometry, 1)
        self.assertIn("36 cm", geometry_hint["visual"])
        self.assertIn("AREA = ?", geometry_hint["visual"])
        self.assertIn("figure", geometry_hint["action"].lower())

    def test_automatic_start_level_uses_bank_distribution(self):
        levels = sorted(p["level"] for p in self.tutor.problems if p["language"] == "eng")
        expected = levels[len(levels) // 2]
        self.assertEqual(self.tutor._automatic_start_level("eng"), expected)
        problem = self.tutor.start("automatic-level-student", language="eng")
        self.assertLessEqual(abs(problem["level"] - expected), 1)

    def test_diagnostic_phase_tries_another_topic(self):
        problem = self.tutor.start("diagnostic-student", language="eng", start_level=0)
        first_topic = problem["topic"]
        label = self.tutor.by_key[self.tutor.key_of(problem)]["label"]
        self.tutor.submit_answer("diagnostic-student", label)
        next_problem = self.tutor.next_problem("diagnostic-student", language="eng")
        available_topics = {p["topic"] for p in self.tutor.problems if p.get("language") == "eng"}
        if len(available_topics) > 1:
            self.assertNotEqual(next_problem["topic"], first_topic)

    def test_reject_invalid_answer(self):
        self.tutor.start("bad-student", language="eng", start_level=0)
        with self.assertRaises(TutorError):
            self.tutor.submit_answer("bad-student", "X")

    def test_profile_for_new_student(self):
        profile = self.tutor.profile("new-student")
        self.assertEqual(profile["total"], 0)
        self.assertIsNone(profile["accuracy"])


if __name__ == "__main__":
    unittest.main()
