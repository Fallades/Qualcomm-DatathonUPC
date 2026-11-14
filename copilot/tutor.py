"""Offline adaptive practice engine, deliberately separate from predict.py.

The tutor uses Python and SQLite for student profiles and exercise selection.
The optional embedding index only reorders candidate exercises; the tutor can
also run without the embedding service or any third-party Python packages.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from collections import defaultdict

from .topics import TOPICS, classify_problem

LETTERS = "ABCDE"

PROCEDURES = {
    "arithmetic": [
        "Identifica les quantitats i què et demana exactament l'enunciat.",
        "Decideix quina operació relaciona aquestes quantitats (suma, resta, producte o divisió).",
        "Fes el càlcul pas a pas i comprova si el resultat té sentit amb les unitats.",
    ],
    "algebra": [
        "Defineix què representa la incògnita (per exemple, x).",
        "Tradueix la informació de l'enunciat a una expressió o equació.",
        "Aïlla la incògnita fent la mateixa operació als dos costats i comprova el resultat.",
    ],
    "geometry": [
        "Identifica la figura i les mesures que coneixes. Fes un petit esquema si t'ajuda.",
        "Decideix quina propietat o fórmula necessites (àrea, perímetre, angle, etc.).",
        "Substitueix les mesures, calcula i comprova que les unitats siguin correctes.",
    ],
    "number_theory": [
        "Anota els nombres i fixa't en les seves propietats: xifres, divisors, múltiples o paritat.",
        "Prova els divisors o les condicions de manera ordenada, sense saltar cap cas.",
        "Comprova que el nombre compleix totes les condicions de l'enunciat.",
    ],
    "combinatorics": [
        "Decideix què comptaràs: objectes, posicions, grups o possibilitats.",
        "Divideix el problema en casos que no se solapin i fes una llista o una taula.",
        "Compta els casos vàlids una sola vegada i comprova que no en falta cap.",
    ],
    "probability": [
        "Defineix quin és l'esdeveniment que vols estudiar.",
        "Compta els casos favorables i tots els casos possibles, si són equiprobables.",
        "Calcula favorables / possibles i comprova que la probabilitat queda entre 0 i 1.",
    ],
    "statistics": [
        "Organitza les dades en una llista ordenada o una taula de freqüències.",
        "Identifica què et demanen: mitjana, mediana, moda o una freqüència.",
        "Aplica la definició corresponent i revisa el càlcul amb les dades originals.",
    ],
    "measurement": [
        "Anota les magnituds i les unitats. Converteix-les a la mateixa unitat si cal.",
        "Identifica la relació entre les magnituds (temps, distància, velocitat, proporció...).",
        "Calcula i comprova que la unitat final i l'ordre de magnitud siguin raonables.",
    ],
    "logic": [
        "Reescriu les condicions de l'enunciat en una llista curta.",
        "Prova els casos de manera ordenada i elimina els que contradiguin una condició.",
        "Revisa si el patró es manté en tots els casos, no només en els primers exemples.",
    ],
    "other": [
        "Subratlla les dades importants i explica amb les teves paraules què et demanen.",
        "Divideix el problema en passos petits i resol primer la part que tens més clara.",
        "Comprova el resultat substituint-lo a l'enunciat i revisant totes les condicions.",
    ],
}


SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    language TEXT NOT NULL DEFAULT 'cat',
    start_level INTEGER NOT NULL DEFAULT 1,
    current_problem_key TEXT,
    current_topic TEXT,
    current_level INTEGER,
    hint_stage INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    problem_key TEXT NOT NULL,
    topic TEXT NOT NULL,
    level INTEGER NOT NULL,
    choice TEXT NOT NULL,
    correct INTEGER NOT NULL,
    hints_used INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    FOREIGN KEY(student_id) REFERENCES students(student_id)
);
CREATE INDEX IF NOT EXISTS attempts_student_idx ON attempts(student_id, attempt_id);
CREATE INDEX IF NOT EXISTS attempts_topic_idx ON attempts(student_id, topic);
"""


class TutorError(Exception):
    """Expected error safe to return to the tutor UI."""


class AdaptiveTutor:
    def __init__(self, problems, db_path, index=None):
        self.problems = []
        self.by_key = {}
        for original in problems:
            problem = classify_problem(original)
            if not self._has_all_options(problem):
                continue
            key = self.key_of(problem)
            if key in self.by_key:
                continue
            self.problems.append(problem)
            self.by_key[key] = problem
        self.index = index
        self.db_path = db_path
        folder = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(folder, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)

    @staticmethod
    def key_of(problem):
        return f"{problem.get('id', '')}|{problem.get('language', '')}"

    @staticmethod
    def _has_all_options(problem):
        options = problem.get("options") or {}
        return all(letter in options and str(options[letter]).strip() for letter in LETTERS)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _check_student_id(student_id):
        student_id = str(student_id or "").strip()
        if not student_id or len(student_id) > 64:
            raise TutorError("Introdueix un identificador d'estudiant d'entre 1 i 64 caràcters.")
        return student_id

    def _ensure_student(self, db, student_id, language="cat", start_level=1):
        student_id = self._check_student_id(student_id)
        now = time.time()
        db.execute(
            "INSERT OR IGNORE INTO students(student_id, language, start_level, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (student_id, language, int(start_level), now, now),
        )
        db.execute("UPDATE students SET language=?, updated_at=? WHERE student_id=?",
                   (language, now, student_id))
        return db.execute("SELECT * FROM students WHERE student_id=?", (student_id,)).fetchone()

    def _attempt_history(self, db, student_id):
        return db.execute(
            "SELECT * FROM attempts WHERE student_id=? ORDER BY attempt_id", (student_id,)
        ).fetchall()

    def _weakest_topic(self, history):
        by_topic = defaultdict(list)
        for row in history:
            by_topic[row["topic"]].append(int(row["correct"]))
        if not by_topic:
            return None
        # Prefer topics with more evidence when accuracy is tied.
        return min(by_topic, key=lambda t: (sum(by_topic[t]) / len(by_topic[t]), -len(by_topic[t]), t))

    def profile(self, student_id):
        student_id = self._check_student_id(student_id)
        with self._connect() as db:
            student = db.execute("SELECT * FROM students WHERE student_id=?", (student_id,)).fetchone()
            if student is None:
                return {"student_id": student_id, "total": 0, "correct": 0, "accuracy": None,
                        "topics": [], "weakest_topic": None}
            history = self._attempt_history(db, student_id)
        total = len(history)
        correct = sum(int(r["correct"]) for r in history)
        groups = defaultdict(list)
        for row in history:
            groups[row["topic"]].append(int(row["correct"]))
        topics = []
        for topic, values in sorted(groups.items()):
            hits = sum(values)
            topics.append({"topic": topic, "label": TOPICS.get(topic, topic), "correct": hits,
                           "total": len(values), "accuracy": round(100 * hits / len(values), 1)})
        weak = self._weakest_topic(history)
        return {
            "student_id": student_id, "total": total, "correct": correct,
            "accuracy": round(100 * correct / total, 1) if total else None,
            "topics": topics, "weakest_topic": weak,
            "weakest_topic_label": TOPICS.get(weak) if weak else None,
        }

    def current(self, student_id):
        student_id = self._check_student_id(student_id)
        with self._connect() as db:
            row = db.execute("SELECT current_problem_key FROM students WHERE student_id=?",
                             (student_id,)).fetchone()
        if not row or not row["current_problem_key"]:
            return None
        problem = self.by_key.get(row["current_problem_key"])
        return self._public_problem(problem) if problem else None

    @staticmethod
    def _public_problem(problem):
        if not problem:
            return None
        return {k: v for k, v in problem.items() if k not in {"label", "topic_scores"}}

    def start(self, student_id, language="cat", topic="", start_level=1):
        if language not in {p.get("language") for p in self.problems}:
            available = sorted({p.get("language") for p in self.problems if p.get("language")})
            language = available[0] if available else "eng"
        start_level = max(0, min(7, int(start_level)))
        with self._connect() as db:
            student = self._ensure_student(db, student_id, language, start_level)
            if student["current_problem_key"] in self.by_key:
                return self.current(student_id)
        return self.next_problem(student_id, language=language, topic=topic, start_level=start_level)

    def next_problem(self, student_id, language=None, topic="", start_level=None):
        student_id = self._check_student_id(student_id)
        with self._connect() as db:
            student = db.execute("SELECT * FROM students WHERE student_id=?", (student_id,)).fetchone()
            if student is None:
                language = language or (self.problems[0].get("language", "eng") if self.problems else "eng")
                student = self._ensure_student(db, student_id, language, start_level or 1)
            history = self._attempt_history(db, student_id)
            attempted = {r["problem_key"] for r in history}
            language = language or student["language"]
            recent = history[-1] if history else None
            diagnostic_topics = set()
            if recent:
                target_level = int(recent["level"]) + (1 if int(recent["correct"]) else -1)
                target_level = max(0, min(7, target_level))
                reference_key = recent["problem_key"]
                if topic:
                    chosen_topic = topic
                elif len(history) < 5:
                    # First build a small diagnostic profile across different
                    # topics instead of drilling the first topic encountered.
                    seen_topics = {r["topic"] for r in history}
                    remaining_topics = {p.get("topic", "other") for p in self.problems
                                        if self.key_of(p) not in attempted}
                    diagnostic_topics = remaining_topics - seen_topics
                    chosen_topic = "" if diagnostic_topics else recent["topic"]
                else:
                    # After five diagnostic answers, target the weakest topic.
                    chosen_topic = self._weakest_topic(history) or recent["topic"]
            else:
                target_level = int(start_level if start_level is not None else student["start_level"])
                chosen_topic = topic or self._weakest_topic(history) or ""
                reference_key = ""
            target_level = max(0, min(7, target_level))

            candidates = [p for p in self.problems if self.key_of(p) not in attempted]
            if diagnostic_topics:
                unseen = [p for p in candidates if p.get("topic", "other") in diagnostic_topics]
                if unseen:
                    candidates = unseen
            if chosen_topic:
                same_topic = [p for p in candidates if p.get("topic") == chosen_topic]
                if same_topic:
                    candidates = same_topic
            if not candidates:
                # If a learner has exhausted a topic, continue with the next
                # weakest available topic rather than reusing an exercise.
                used_topics = {r["topic"] for r in history}
                alternatives = [p for p in self.problems if self.key_of(p) not in attempted
                                and p.get("topic") not in used_topics]
                candidates = alternatives or [p for p in self.problems if self.key_of(p) not in attempted]
            if not candidates:
                raise TutorError("Ja s'han completat tots els exercicis disponibles d'aquest banc.")

            # Keep the student's preferred language when possible.
            same_language = [p for p in candidates if p.get("language") == language]
            if same_language:
                candidates = same_language

            similarity = {}
            if self.index and reference_key and reference_key in self.by_key:
                try:
                    hits = self.index.search(self.by_key[reference_key], k=min(80, len(self.problems)),
                                             exclude_id=self.by_key[reference_key]["id"])
                    similarity = {self.key_of(p): score for score, p in hits}
                except Exception:
                    # Retrieval is an optional ranking signal; deterministic
                    # level/topic selection still works when embeddings are down.
                    similarity = {}

            candidates.sort(key=lambda p: (
                abs(int(p.get("level", 0)) - target_level),
                -similarity.get(self.key_of(p), -1.0),
                hashlib.sha256(f"{student_id}:{self.key_of(p)}".encode()).hexdigest(),
            ))
            selected = candidates[0]
            key = self.key_of(selected)
            db.execute(
                "UPDATE students SET current_problem_key=?, current_topic=?, current_level=?, "
                "hint_stage=0, language=?, updated_at=? WHERE student_id=?",
                (key, selected.get("topic", "other"), int(selected.get("level", 0)),
                 selected.get("language", language), time.time(), student_id),
            )
            return self._public_problem(selected)

    def submit_answer(self, student_id, choice):
        student_id = self._check_student_id(student_id)
        choice = str(choice or "").strip().upper()
        if choice not in LETTERS:
            raise TutorError("La resposta ha de ser una lletra entre A i E.")
        with self._connect() as db:
            student = db.execute("SELECT * FROM students WHERE student_id=?", (student_id,)).fetchone()
            if student is None or not student["current_problem_key"]:
                raise TutorError("No hi ha cap exercici actiu. Comença una sessió primer.")
            problem = self.by_key.get(student["current_problem_key"])
            if problem is None:
                raise TutorError("L'exercici actiu ja no és al banc de dades. Comença una sessió nova.")
            correct = choice == problem.get("label")
            now = time.time()
            db.execute(
                "INSERT INTO attempts(student_id, problem_key, topic, level, choice, correct, hints_used, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (student_id, student["current_problem_key"], problem.get("topic", "other"),
                 int(problem.get("level", 0)), choice, int(correct), int(student["hint_stage"]), now),
            )
            db.execute("UPDATE students SET current_problem_key=NULL, hint_stage=0, updated_at=? WHERE student_id=?",
                       (now, student_id))
        return {"correct": correct, "label": problem["label"], "topic": problem.get("topic", "other"),
                "topic_label": TOPICS.get(problem.get("topic", "other"), "Altres"),
                "level": problem.get("level", 0), "profile": self.profile(student_id)}

    def _easier_example(self, student_id, topic, level, language):
        if not self.index:
            return None
        current = self.current(student_id)
        if not current:
            return None
        try:
            hits = self.index.search(current, k=min(50, len(self.problems)), exclude_id=current["id"])
        except Exception:
            return None
        for _score, problem in hits:
            if (problem.get("topic") == topic and int(problem.get("level", 99)) < int(level)
                    and self._has_all_options(problem)):
                return {"text": problem.get("text", ""), "level": problem.get("level"),
                        "options": problem.get("options", {}), "answer": problem.get("label")}
        return None

    def hint(self, student_id):
        student_id = self._check_student_id(student_id)
        with self._connect() as db:
            student = db.execute("SELECT * FROM students WHERE student_id=?", (student_id,)).fetchone()
            if student is None or not student["current_problem_key"]:
                raise TutorError("No hi ha cap exercici actiu per donar-hi pistes.")
            problem = self.by_key.get(student["current_problem_key"])
            if problem is None:
                raise TutorError("No s'ha trobat l'exercici actiu.")
            stage = int(student["hint_stage"]) + 1
            db.execute("UPDATE students SET hint_stage=?, updated_at=? WHERE student_id=?",
                       (stage, time.time(), student_id))

        topic = problem.get("topic", "other")
        procedure = PROCEDURES.get(topic, PROCEDURES["other"])
        if stage == 1:
            return {"stage": stage, "kind": "procedure", "title": "Procediment per començar",
                    "text": procedure[0], "steps": procedure}
        if stage == 2:
            example = self._easier_example(student_id, topic, int(problem.get("level", 0)),
                                           problem.get("language", ""))
            if example:
                return {"stage": stage, "kind": "example", "title": "Exemple més senzill del mateix tema",
                        "text": "Fixa't en com s'organitzen les dades i quin mètode s'hi aplica.",
                        "example": example}
            return {"stage": stage, "kind": "procedure", "title": "Divideix-ho en passos",
                    "text": procedure[1], "steps": procedure}
        if stage == 3:
            return {"stage": stage, "kind": "procedure", "title": "Comprova el teu plantejament",
                    "text": procedure[2], "steps": procedure}
        return {"stage": stage, "kind": "reflection", "title": "Abans de veure més ajuda",
                "text": "Explica en veu alta què has provat i quin pas concret no et surt. Revisa les dades i les condicions abans de triar una resposta."}
