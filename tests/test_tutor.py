import json
import os
import tempfile
import unittest

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

    def test_hint_steps(self):
        problem = self.tutor.start("hint-student", language="cat", start_level=1)
        hint = self.tutor.hint("hint-student")
        self.assertEqual(hint["stage"], 1)
        self.assertEqual(len(hint["steps"]), 3)

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
