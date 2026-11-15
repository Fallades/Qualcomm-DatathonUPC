"""Offline adaptive practice engine, deliberately separate from predict.py.

The tutor uses Python and SQLite for student profiles and exercise selection.
The optional embedding index only reorders candidate exercises; the tutor can
also run without the embedding service or any third-party Python packages.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from collections import defaultdict

from .topics import TOPICS, classify_problem
from .config import CONFIG
from .client import post

LETTERS = "ABCDE"

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
CREATE TABLE IF NOT EXISTS hint_history (
    student_id TEXT NOT NULL,
    problem_key TEXT NOT NULL,
    stage INTEGER NOT NULL,
    hint_text TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(student_id, problem_key, stage)
);
CREATE TABLE IF NOT EXISTS hint_plans (
    student_id TEXT NOT NULL,
    problem_key TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(student_id, problem_key)
);
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

    def start(self, student_id, language="cat", topic="", start_level=None):
        if language not in {p.get("language") for p in self.problems}:
            available = sorted({p.get("language") for p in self.problems if p.get("language")})
            language = available[0] if available else "eng"
        if start_level is None:
            start_level = self._automatic_start_level(language, topic)
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
                initial_level = (start_level if start_level is not None
                                 else self._automatic_start_level(language))
                student = self._ensure_student(db, student_id, language, initial_level)
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
                 int(problem.get("level", 0)), choice, int(correct), min(3, max(0, int(student["hint_stage"]))), now),
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

    def _automatic_start_level(self, language, topic=""):
        """Choose a representative starting difficulty from the available bank.

        The initial level is deliberately inferred from the exercise distribution;
        the learner does not need to configure a value that the tutor immediately
        changes again after each answer.
        """
        candidates = [p for p in self.problems if p.get("language") == language]
        if topic:
            same_topic = [p for p in candidates if p.get("topic") == topic]
            if same_topic:
                candidates = same_topic
        levels = sorted(int(p.get("level", 0)) for p in candidates)
        if not levels:
            levels = sorted(int(p.get("level", 0)) for p in self.problems)
        if not levels:
            return 3
        return max(0, min(7, levels[len(levels) // 2]))

    @staticmethod
    def _hint_language(problem):
        language = str(problem.get("language", "eng")).lower()
        if language in {"cat", "ca", "catalan"}:
            return "Catalan"
        if language in {"esp", "es", "spa", "spanish"}:
            return "Spanish"
        return "English"

    @staticmethod
    def _hint_title(language, stage):
        titles = {
            "Catalan": {
                1: "Pista 1 · Entén què et demanen",
                2: "Pista 2 · Troba la relació clau",
                3: "Pista 3 · Organitza els passos",
            },
            "English": {
                1: "Hint 1 · Understand the goal",
                2: "Hint 2 · Find the key relationship",
                3: "Hint 3 · Organise the steps",
            },
            "Spanish": {
                1: "Pista 1 · Entiende el objetivo",
                2: "Pista 2 · Encuentra la relación clave",
                3: "Pista 3 · Ordena los pasos",
            },
        }
        return titles.get(language, titles["English"])[stage]

    @staticmethod
    def _parse_json_object(raw):
        raw = str(raw or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        left, right = raw.find("{"), raw.rfind("}")
        if left < 0 or right <= left:
            raise ValueError("El model no ha retornat un objecte JSON.")
        payload = json.loads(raw[left:right + 1])
        if not isinstance(payload, dict):
            raise ValueError("La resposta del model no és un objecte JSON.")
        return payload

    def _generate_hint(self, problem, stage):
        """Generate one detailed, exercise-specific hint, not a three-hint bundle.

        Generating only the requested step reduces latency on the UNO Q and avoids
        the old 420-token response that could exceed the 60-second HTTP timeout.
        """
        language = self._hint_language(problem)
        options = problem.get("options") or {}
        correct_letter = str(problem.get("label", "")).strip().upper()
        distractors = "\n".join(
            f"{letter}) {options[letter]}" for letter in LETTERS
            if letter in options and letter != correct_letter
        )
        exercise_text = str(problem.get("text", "")).strip()
        if not exercise_text:
            raise ValueError("L'exercici no té cap enunciat per analitzar.")

        stage_instructions = {
            1: "Guide the student to identify the exact goal, useful data and a suitable visual representation. Do not suggest the full method yet.",
            2: "Build on the previous idea: explain the specific mathematical relationship or operation needed here, using the actual quantities and constraints in the statement. Do not finish the calculation.",
            3: "Give an ordered scaffold of the first concrete steps for THIS exercise. Include intermediate setup or calculations when useful, but stop before the final result and do not identify an option.",
        }
        system = (
            f"You are a patient expert mathematics tutor. Write only in {language}. "
            "Use only facts present in the exercise. Be precise, student-friendly and specific. "
            "Never reveal the final answer, the correct option, or the correct option letter. "
            "Return one valid JSON object only."
        )
        user = f"""Create ONLY hint {stage} of 3 for the exact exercise below.

EXERCISE:
{exercise_text}

TOPIC TAG (may be imperfect): {TOPICS.get(problem.get('topic', 'other'), 'Other')}
DIFFICULTY LEVEL: {problem.get('level', 'unknown')}
INCORRECT OPTIONS (the correct option and its letter are deliberately hidden):
{distractors or 'No readable options'}

PEDAGOGICAL ROLE:
{stage_instructions[stage]}

Requirements:
- Make the explanation exercise-specific: refer to concrete quantities, objects, conditions or wording from THIS statement.
- Write 45–75 words in the explanation, in 2–4 clear sentences. Do not pad with generic encouragement.
- Add a compact visual aid suited to this problem: a small ASCII sketch, labelled relationship, equation layout, table, number line, sequence or decision tree. Use 2–6 lines. Never invent a diagram detail that the statement does not support.
- End with one short action the student can try now.
- Each stage must be more concrete than the previous stage, without repeating it.
- Do not state or calculate the final answer. Do not mention any option letter.

Return this JSON shape:
{{"explanation":"...","visual":"...","action":"..."}}"""
        response = post(
            f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions",
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 210,
                "temperature": 0.15,
                "seed": int(CONFIG.get("SEED", 42)),
                "cache_prompt": True,
            },
            # A single, bounded hint has a longer scaffold but far fewer tokens
            # than the former three-hint bundle. Fall back after 60 seconds.
            timeout=60,
        )
        raw = response["choices"][0]["message"]["content"]
        payload = self._parse_json_object(raw)
        explanation = str(payload.get("explanation", "")).strip()
        visual = str(payload.get("visual", "")).strip()
        action = str(payload.get("action", "")).strip()
        if len(explanation.split()) < 40 or len(visual.splitlines()) < 2 or not action:
            raise ValueError("La pista generada és massa curta o no inclou un suport visual desenvolupat.")

        # A small defensive filter against an accidental answer-label leak.
        combined = f"{explanation}\n{visual}\n{action}".casefold()
        if correct_letter and re.search(
            rf"(?:option|opci[oó]|answer|resposta|respuesta)\s*{re.escape(correct_letter)}\b|^{re.escape(correct_letter)}[\):]",
            combined,
            flags=re.MULTILINE,
        ):
            raise ValueError("La pista sembla revelar la lletra de l'opció correcta.")
        correct_text = str(options.get(correct_letter, "")).strip()
        if len(correct_text) >= 5 and correct_text.casefold() in combined:
            raise ValueError("La pista sembla revelar el valor de l'opció correcta.")

        return {
            "title": self._hint_title(language, stage),
            "explanation": explanation,
            "visual": visual,
            "action": action,
            "kind": "ai",
        }

    def _fallback_hint(self, problem, stage, reason=""):
        """Provide a useful local hint even when llama-server is unavailable.

        The wording and the visual anchor are built from the current exercise,
        its detected topic and its actual quantities; this is not a global
        'think carefully' message.
        """
        language = self._hint_language(problem)
        topic = problem.get("topic", "other")
        statement = str(problem.get("text", "")).strip()
        numbers = re.findall(r"(?<![A-Za-z])\d+(?:[.,]\d+)?(?:\s?(?:cm²|cm2|mm|cm|m|km|kg|g|litres?|liters?|hours?|minutes?|seconds?))?", statement, flags=re.I)
        numbers = list(dict.fromkeys(numbers))[:8]
        number_line = "  →  ".join(numbers) if numbers else "(no explicit numbers detected)"
        question_parts = re.findall(r"[^.!?]*\?", statement)
        goal = question_parts[-1].strip() if question_parts else statement[-150:]
        anchor = goal[:150].strip()

        # Templates deliberately use the exact statement's quantities and goal.
        if language == "Catalan":
            content = {
                "geometry": [
                    f"En aquest exercici, separa les dades de la figura de la magnitud que et demanen. El text acaba preguntant: «{anchor}». Abans de calcular, identifica quines longituds, àrees o relacions pertanyen a cada figura; no barregis la figura sencera amb una part ombrejada o superposada.",
                    "Fes un esquema propi i posa-hi només les dades que apareixen a l'enunciat. Busca quina propietat geomètrica connecta aquestes dades amb la zona o longitud demanada; sovint cal expressar una àrea o un perímetre a partir d'una altra figura més senzilla.",
                    "Segueix aquest ordre: 1) dibuixa o reconstrueix la figura; 2) etiqueta les mesures conegudes; 3) escriu la relació geomètrica que les connecta; 4) aïlla la magnitud demanada. Deixa indicada l'operació final i comprova que les unitats siguin coherents, sense saltar directament a una opció.",
                ],
                "combinatorics": [
                    f"La pregunta concreta és «{anchor}». Abans de comptar, escriu quins objectes o posicions intervenen i copia totes les restriccions de l'enunciat. El punt important és decidir si dues disposicions que només canvien d'ordre compten com a casos diferents.",
                    "Converteix les condicions en una llista de casos que no se solapin. Si hi ha adjacències o posicions prohibides, marca-les primer; després compta les opcions que queden en cada cas. Una taula o un arbre curt evita repetir una mateixa disposició.",
                    "1) Anomena els objectes o les posicions; 2) aplica les restriccions abans de comptar; 3) divideix les possibilitats en casos disjunts; 4) compta cada cas i suma'ls. Repassa al final que cap disposició s'hagi comptat dues vegades ni n'hagi quedat cap fora.",
                ],
                "probability": [
                    f"En aquesta pregunta («{anchor}»), primer cal aclarir quin és l'experiment i què es considera un resultat favorable. Escriu tots els resultats possibles de manera ordenada; no calculis una probabilitat abans de saber quin és el conjunt de casos.",
                    "Construeix una taula o un arbre amb les possibilitats i marca les que compleixen la condició demanada. Si tots els resultats són igualment probables, la relació que necessites és casos favorables dividit per casos possibles.",
                    "1) Escriu l'espai mostral; 2) marca els casos que compleixen la condició; 3) compta favorables i totals; 4) forma la fracció corresponent i simplifica-la si cal. Comprova que el nombre de casos favorables no superi el total.",
                ],
                "number_theory": [
                    f"Fixa't en la propietat numèrica que demana «{anchor}». Reescriu les condicions amb paraules matemàtiques —divisible, primer, parell/senar, múltiple o divisor— i separa cada condició en una línia. Això evita provar nombres sense un criteri clar.",
                    "Busca una representació que faci visibles les condicions: descomposició en factors primers, residus o una taula de divisors, segons el que demani l'enunciat. Comprova cada condició per separat abans de combinar-les.",
                    "1) Tradueix cada condició a una regla verificable; 2) aplica-la als nombres o factors implicats; 3) combina les restriccions; 4) només llavors compta o compara els candidats. No acceptis cap candidat que incompleixi una sola condició.",
                ],
                "algebra": [
                    f"La pregunta és «{anchor}». Decideix què representa la incògnita i localitza al text les condicions que la determinen. En lloc d'operar directament amb tots els nombres, assigna un símbol a la quantitat que encara no coneixes.",
                    "Tradueix cada frase rellevant a una igualtat o relació entre la incògnita i les dades. Mantén les operacions en el mateix ordre que descriu l'enunciat i comprova que cada terme representa una quantitat del problema.",
                    "1) Defineix la incògnita; 2) escriu una equació amb les condicions del text; 3) simplifica o aïlla la incògnita pas a pas; 4) substitueix el valor obtingut a l'equació original per comprovar-lo. Atura't abans d'escollir una opció.",
                ],
                "measurement": [
                    f"En aquest problema, la quantitat que es busca és «{anchor}». Marca les magnituds que apareixen —distància, temps, velocitat, massa, volum o litres— i anota les seves unitats. La relació entre les magnituds és més important que fer operacions a l'atzar.",
                    "Organitza les dades en una taula amb una fila per a cada situació descrita. Mantén les unitats consistents i observa quina magnitud es manté constant i quina canvia; això t'indicarà si has de comparar proporcions, diferències o velocitats.",
                    "1) Fes la taula de dades i unitats; 2) escriu la relació entre les magnituds; 3) substitueix-hi les quantitats conegudes; 4) aïlla la quantitat demanada. Abans d'acabar, comprova que la unitat del resultat és la que pregunta l'enunciat.",
                ],
                "statistics": [
                    f"La pregunta demana «{anchor}». Abans de calcular, identifica quines dades formen el conjunt i quina mesura estadística et demanen. No confonguis el valor central, el més freqüent i la mitjana: són idees diferents.",
                    "Ordena les dades en una taula i anota quantes vegades apareix cada valor. Amb aquesta representació podràs veure si cal sumar i dividir, trobar el valor central o localitzar la freqüència més alta.",
                    "1) Copia totes les dades; 2) ordena-les o compta'n les freqüències; 3) aplica la definició de la mesura demanada; 4) revisa el recompte i el nombre total de dades. No utilitzis una fórmula abans de saber què representa.",
                ],
                "logic": [
                    f"L'objectiu exacte és «{anchor}». Converteix cada frase de l'enunciat en una condició que puguis comprovar. Separa el que és segur del que només és una possibilitat; això és especialment important quan hi ha diverses restriccions alhora.",
                    "Construeix una taula de candidats i condicions, amb una columna per a cada regla del problema. Elimina només els casos que contradiguin una condició explícita; després torna a revisar els candidats que queden.",
                    "1) Escriu totes les condicions; 2) prepara una taula de casos; 3) descarta els que incompleixen alguna regla; 4) comprova els casos restants un per un contra l'enunciat complet. Així evites quedar-te amb una solució que només compleix una part.",
                ],
                "arithmetic": [
                    f"En aquest exercici, la pregunta és «{anchor}». Les quantitats que apareixen a l'enunciat són {number_line}. No les combinis totes de cop: identifica què representa cada nombre i si descriu una quantitat inicial, un canvi, una repetició o un total. L'ordre de les accions també pot ser important.",
                    "Escriu les operacions en el mateix ordre que descriu el text i marca els parèntesis quan una operació s'hagi de fer abans que una altra. Si hi ha grups iguals, representa primer un grup i després indica quantes vegades es repeteix; si es treu o s'afegeix una quantitat, mostra aquest canvi al final.",
                    "1) Etiqueta cada nombre amb el seu significat; 2) tradueix cada acció a una operació; 3) ordena les operacions i afegeix parèntesis si cal; 4) calcula només els passos intermedis i comprova si el resultat ha de ser més gran o més petit que la quantitat inicial. No triïs encara cap opció.",
                ],
                "other": [
                    f"Comença per la pregunta literal: «{anchor}». Torna a llegir l'enunciat i marca quines dades serveixen directament per respondre-la. Separa la informació imprescindible de la història que només dona context.",
                    "Reescriu el problema com una seqüència curta: dades conegudes, condició que les relaciona i objectiu desconegut. Fes una taula o un esquema amb les quantitats detectades i indica què representa cadascuna.",
                    "1) Resumeix el que saps; 2) escriu què has de trobar; 3) relaciona cada dada amb una condició de l'enunciat; 4) resol una part cada vegada i comprova que el resultat respongui exactament la pregunta. No saltis cap condició.",
                ],
            }
            visual_label = ("DADES", "RELACIÓ", "OBJECTIU")
            action = "Assenyala al text la dada o condició que utilitzaràs primer i explica per què."
        elif language == "Spanish":
            # Spanish uses the same mathematical scaffolds with local wording.
            content = {
                "geometry": [f"La pregunta exacta es «{anchor}». Separa las medidas dadas de la magnitud que se busca. Antes de calcular, identifica qué longitudes, áreas o relaciones pertenecen a cada figura; no mezcles la figura completa con una región sombreada o superpuesta.", "Haz un esquema propio y coloca solo los datos del enunciado. Busca la propiedad geométrica que relaciona esos datos con la zona o longitud pedida; puede ser necesario expresar una parte a partir de una figura más sencilla.", "1) Reconstruye el dibujo; 2) etiqueta las medidas conocidas; 3) escribe la relación geométrica; 4) aísla la magnitud pedida. Comprueba las unidades y deja la operación final planteada antes de elegir una opción."],
                "combinatorics": [f"La pregunta es «{anchor}». Escribe qué objetos o posiciones intervienen y copia todas las restricciones. Decide si dos disposiciones que solo cambian de orden cuentan como casos distintos.", "Divide las posibilidades en casos que no se solapen. Marca primero las posiciones o combinaciones prohibidas y cuenta después las que quedan; una tabla o árbol evita repetir casos.", "1) Nombra los objetos; 2) aplica las restricciones; 3) separa los casos; 4) cuenta cada caso y súmalos. Comprueba que no hayas contado una disposición dos veces ni olvidado ninguna."],
                "probability": [f"En «{anchor}», identifica el experimento y qué se considera un resultado favorable. Escribe ordenadamente todos los resultados posibles antes de calcular.", "Haz una tabla o árbol de posibilidades y marca las que cumplen la condición. Si los resultados son equiprobables, relaciona casos favorables con casos posibles.", "1) Escribe el espacio muestral; 2) marca los casos favorables; 3) cuenta favorables y totales; 4) forma y simplifica la fracción. Comprueba que los favorables no superen el total."],
                "number_theory": [f"Fíjate en la propiedad numérica de «{anchor}». Traduce cada condición a una regla comprobable —primo, divisible, par/impar, múltiplo o divisor— y escríbelas por separado.", "Usa una representación adecuada: factores primos, restos o una tabla de divisores, según el enunciado. Comprueba las condiciones una por una antes de combinarlas.", "1) Traduce cada condición; 2) aplica la regla a los números implicados; 3) combina las restricciones; 4) cuenta o compara candidatos. Descarta cualquier candidato que incumpla una sola condición."],
                "algebra": [f"La pregunta es «{anchor}». Decide qué representa la incógnita y localiza las condiciones que la determinan. Asigna un símbolo a la cantidad desconocida antes de operar.", "Traduce cada frase relevante a una igualdad o relación. Mantén el mismo orden de operaciones que describe el enunciado y comprueba qué representa cada término.", "1) Define la incógnita; 2) escribe una ecuación; 3) simplifica y despeja paso a paso; 4) sustituye el valor en la ecuación original para comprobarlo. Detente antes de elegir una opción."],
                "measurement": [f"La cantidad que se busca es «{anchor}». Marca las magnitudes y sus unidades. La relación entre ellas importa más que hacer operaciones sin un plan.", "Organiza los datos en una tabla, con una fila por situación. Mantén unidades coherentes y observa qué magnitud permanece constante y cuál cambia.", "1) Haz la tabla; 2) escribe la relación; 3) sustituye los datos; 4) despeja la cantidad pedida. Comprueba que la unidad final coincide con la pregunta."],
                "statistics": [f"La pregunta pide «{anchor}». Identifica el conjunto de datos y la medida estadística solicitada. No confundas media, mediana y moda.", "Ordena los datos en una tabla y cuenta las frecuencias. Así verás si debes sumar y dividir, buscar el valor central o localizar la frecuencia mayor.", "1) Copia los datos; 2) ordénalos o cuenta frecuencias; 3) aplica la definición de la medida; 4) revisa el recuento y el total."],
                "logic": [f"El objetivo exacto es «{anchor}». Convierte cada frase en una condición comprobable. Separa lo seguro de lo que solo es una posibilidad.", "Construye una tabla de candidatos y condiciones, con una columna por regla. Descarta solo los casos que contradigan una condición explícita.", "1) Escribe todas las condiciones; 2) prepara una tabla; 3) elimina los casos imposibles; 4) comprueba cada candidato restante contra todo el enunciado."],
                "arithmetic": [f"En este ejercicio se pregunta «{anchor}». Las cantidades que aparecen son {number_line}. No las combines todas de golpe: identifica qué representa cada número y si indica una cantidad inicial, un cambio, una repetición o un total. El orden de las acciones también puede ser importante.", "Escribe las operaciones en el mismo orden que describe el texto y añade paréntesis cuando una operación deba hacerse antes. Si hay grupos iguales, representa primero un grupo y después indica cuántas veces se repite; si se añade o se quita una cantidad, representa ese cambio por separado.", "1) Etiqueta cada número; 2) traduce cada acción a una operación; 3) ordena las operaciones y usa paréntesis si hace falta; 4) calcula los pasos intermedios y comprueba si el resultado debe ser mayor o menor que la cantidad inicial. No elijas aún una opción."],
                "other": [f"Empieza por la pregunta literal: «{anchor}». Marca qué datos sirven directamente para responderla y separa la información imprescindible del contexto.", "Reescribe el problema como datos conocidos, relación entre ellos y objetivo desconocido. Haz una tabla o esquema con las cantidades detectadas.", "1) Resume los datos; 2) escribe qué debes encontrar; 3) relaciona cada dato con una condición; 4) resuelve una parte cada vez y comprueba que respondes exactamente a la pregunta."],
            }
            visual_label = ("DATOS", "RELACIÓN", "OBJETIVO")
            action = "Subraya el dato o condición que usarás primero y explica por qué."
        else:
            content = {
                "geometry": [f"The exact question is: “{anchor}”. Separate the measurements given from the quantity being requested. Before calculating, identify which lengths, areas or relationships belong to each shape; do not mix the whole figure with a shaded or overlapping part.", "Make your own sketch and label only information stated in the problem. Look for the geometric property that connects those facts to the requested region or length; you may be able to express a complex part using a simpler shape.", "1) Rebuild the diagram; 2) label known measurements; 3) write the geometric relationship connecting them; 4) isolate the requested quantity. Check the units and leave the final operation set up before choosing an option."],
                "combinatorics": [f"The exact question is “{anchor}”. List the objects or positions involved and copy every restriction from the statement. Decide whether two arrangements that differ only in order count as different outcomes.", "Split the possibilities into cases that do not overlap. Mark forbidden positions or pairings first, then count what remains; a small table or tree helps prevent double-counting.", "1) Name the objects; 2) apply the restrictions; 3) split the valid arrangements into separate cases; 4) count each case and add them. Check that no arrangement was counted twice or missed."],
                "probability": [f"For “{anchor}”, identify the experiment and what counts as a favourable outcome. List the possible outcomes in an orderly way before calculating anything.", "Build a table or tree of the possibilities and mark those satisfying the condition. If all outcomes are equally likely, relate favourable cases to all possible cases.", "1) Write the sample space; 2) mark favourable outcomes; 3) count favourable and total cases; 4) form and simplify the fraction. Check that favourable cases do not exceed the total."],
                "number_theory": [f"Focus on the number property in “{anchor}”. Translate each condition into a checkable rule — prime, divisible, odd/even, multiple or divisor — and keep the rules separate.", "Choose a representation that exposes the restrictions: prime factorisation, remainders or a divisor table, depending on the statement. Check each condition before combining them.", "1) Translate every condition into a rule; 2) apply it to the relevant numbers; 3) combine the restrictions; 4) count or compare the candidates. Reject any candidate that fails even one condition."],
                "algebra": [f"The question is “{anchor}”. Decide what the unknown represents and locate the conditions that determine it. Give the unknown a symbol before manipulating the quantities.", "Translate each relevant sentence into an equation or relationship. Preserve the order of operations described in the statement and check what each term represents.", "1) Define the unknown; 2) write an equation from the conditions; 3) simplify and isolate the unknown step by step; 4) substitute it back into the original equation to check. Stop before selecting an option."],
                "measurement": [f"The quantity being requested is “{anchor}”. Mark each measurement and its unit — distance, time, speed, mass or volume. The relationship between quantities matters more than trying operations at random.", "Organise the information in a table, with one row for each situation described. Keep units consistent and identify which quantity stays constant and which one changes.", "1) Build the data table; 2) write the relationship between quantities; 3) substitute the known values; 4) isolate the requested quantity. Check that the final unit matches the question."],
                "statistics": [f"The question asks for “{anchor}”. Identify the data set and the statistical measure requested. Do not confuse the mean, median and mode; they describe different features.", "Arrange the values in a table and count their frequencies. This will show whether you need to add and divide, find the central value, or locate the most frequent value.", "1) Copy all data; 2) sort the values or count frequencies; 3) apply the definition of the requested measure; 4) verify the count and the number of data points."],
                "logic": [f"The exact goal is “{anchor}”. Turn each sentence in the statement into a condition you can check. Separate what is certain from what is only a possibility.", "Create a candidate-versus-condition table, with one column for each rule. Eliminate only the cases that contradict an explicit condition, then inspect the remaining candidates.", "1) Write every condition; 2) make a case table; 3) remove impossible cases; 4) check each remaining candidate against the full statement, not just one condition."],
                "arithmetic": [f"This exercise asks: “{anchor}”. The quantities appearing in the statement are {number_line}. Do not combine them immediately: identify what each number represents and whether it describes a starting amount, a change, a repeated group or a total. The order of the actions may matter too.", "Write the operations in the same order as the statement describes them, using brackets when one operation must happen before another. If equal groups are involved, show one group first and then how many times it is repeated; keep adding or removing quantities as a separate step.", "1) Label every number with its meaning; 2) translate each action into an operation; 3) order the operations and add brackets if needed; 4) calculate the intermediate steps and check whether the result should be larger or smaller than the starting amount. Do not choose an option yet."],
                "other": [f"Start with the exact question: “{anchor}”. Mark which facts are directly useful for answering it and separate essential information from story context.", "Rewrite the problem as known information, the relationship between the facts, and the unknown goal. A small table or sketch can make the quantities and constraints visible.", "1) Summarise the data; 2) state what must be found; 3) connect each fact to a condition; 4) solve one part at a time and check that it answers the exact question."],
            }
            visual_label = ("GIVEN DATA", "RELATIONSHIP", "GOAL")
            action = "Underline the first fact or condition you will use and explain why."

        explanation = content.get(topic, content["other"])[stage - 1]
        supplements = {
            "Catalan": [
                "Comença separant les dades que l'enunciat dona explícitament de les suposicions que fas; no inventis mesures, resultats possibles ni condicions que no hi apareixen.",
                "Fes visible la relació amb una taula, un dibuix, una llista o una equació etiquetada, i indica al costat de cada símbol quina quantitat representa.",
                "Després de cada pas intermedi, contrasta'l amb les condicions originals; revisa les unitats, el signe i que la quantitat calculada sigui exactament la que et demanen.",
            ],
            "English": [
                "Start by separating facts that the statement explicitly gives from assumptions you are making; never invent missing measurements, outcomes or conditions that are not written in the exercise.",
                "Make the relationship visible with a labelled table, diagram, list or equation, and write beside every symbol what quantity it represents and which condition justifies it.",
                "After each intermediate step, compare it with the original conditions; check units, sign and whether the quantity you are calculating is exactly the one requested.",
            ],
            "Spanish": [
                "Empieza separando los datos que el enunciado indica explícitamente de las suposiciones que haces; no inventes medidas, resultados posibles ni condiciones que no aparezcan.",
                "Haz visible la relación mediante una tabla, un dibujo, una lista o una ecuación etiquetada, e indica junto a cada símbolo qué cantidad representa.",
                "Después de cada paso intermedio, contrástalo con las condiciones originales; revisa las unidades, el signo y que la cantidad calculada sea exactamente la que se pide.",
            ],
        }
        explanation = explanation + " " + supplements.get(language, supplements["English"])[stage - 1]

        # Localise every label and tailor the visual scaffold to the mathematical
        # structure of the current exercise. In particular, do not inject the
        # Catalan topic taxonomy into an English or Spanish hint.
        labels = {
            "Catalan": {
                "data": "DADES", "goal": "OBJECTIU", "relation": "RELACIÓ",
                "expression": "EXPRESSIÓ", "order": "ORDRE DE LES OPERACIONS",
                "first": "PRIMER PAS", "next": "PAS SEGÜENT", "result": "RESULTAT A TROBAR",
                "measure": "MESURES DONADES", "diagram": "FIGURA ORIGINAL",
                "sample": "ESPAI MOSTRAL", "favourable": "CASOS FAVORABLES",
                "conditions": "CONDICIONS", "candidate": "CANDIDAT", "unknown": "INCÒGNITA",
                "equation": "EQUACIÓ", "units": "UNITATS", "data_set": "CONJUNT DE DADES",
                "cases": "CASOS POSSIBLES", "keep": "CONSERVA NOMÉS ELS CASOS QUE COMPLEIXEN TOTES LES CONDICIONS",
                "favourable_question": "quins casos compleixen la condició?", "count": "recompte",
                "case_a": "cas A", "case_b": "cas B", "case_c": "cas C", "restrictions": "restriccions de l'enunciat",
                "parity": "paritat", "divisibility": "divisibilitat", "factors": "factors",
                "statement_data": "dades de l'enunciat", "relationship_x": "relació amb x", "isolate_x": "aïlla x",
                "magnitude": "MAGNITUD", "value": "VALOR", "sort": "ORDENA", "frequencies": "COMPTA FREQÜÈNCIES",
                "measure": "MESURA", "mean": "mitjana", "median": "mediana", "mode": "moda",
                "condition1": "C1", "condition2": "C2", "condition3": "C3",
                "data_relation": "dades → condició → incògnita", "requested_value": "valor i unitat demanats",
                "diagram_instruction": "marca-hi les mesures i separa les parts de la figura",
            },
            "English": {
                "data": "GIVEN DATA", "goal": "GOAL", "relation": "RELATIONSHIP",
                "expression": "EXPRESSION", "order": "OPERATION ORDER",
                "first": "FIRST STEP", "next": "NEXT STEP", "result": "RESULT TO FIND",
                "measure": "GIVEN MEASUREMENTS", "diagram": "ORIGINAL FIGURE",
                "sample": "SAMPLE SPACE", "favourable": "FAVOURABLE CASES",
                "conditions": "CONDITIONS", "candidate": "CANDIDATE", "unknown": "UNKNOWN",
                "equation": "EQUATION", "units": "UNITS", "data_set": "DATA SET",
                "cases": "POSSIBLE CASES", "keep": "KEEP ONLY CASES THAT SATISFY EVERY CONDITION",
                "favourable_question": "which cases satisfy the condition?", "count": "count",
                "case_a": "case A", "case_b": "case B", "case_c": "case C", "restrictions": "restrictions from the statement",
                "parity": "parity", "divisibility": "divisibility", "factors": "factors",
                "statement_data": "known facts from the statement", "relationship_x": "relationship with x", "isolate_x": "isolate x",
                "magnitude": "QUANTITY", "value": "VALUE", "sort": "SORT", "frequencies": "COUNT FREQUENCIES",
                "measure": "MEASURE", "mean": "mean", "median": "median", "mode": "mode",
                "condition1": "C1", "condition2": "C2", "condition3": "C3",
                "data_relation": "data → condition → unknown", "requested_value": "requested value and unit",
                "diagram_instruction": "mark the measurements and separate the parts of the figure",
            },
            "Spanish": {
                "data": "DATOS", "goal": "OBJETIVO", "relation": "RELACIÓN",
                "expression": "EXPRESIÓN", "order": "ORDEN DE LAS OPERACIONES",
                "first": "PRIMER PASO", "next": "PASO SIGUIENTE", "result": "RESULTADO A ENCONTRAR",
                "measure": "MEDIDAS DADAS", "diagram": "FIGURA ORIGINAL",
                "sample": "ESPACIO MUESTRAL", "favourable": "CASOS FAVORABLES",
                "conditions": "CONDICIONES", "candidate": "CANDIDATO", "unknown": "INCÓGNITA",
                "equation": "ECUACIÓN", "units": "UNIDADES", "data_set": "CONJUNTO DE DATOS",
                "cases": "CASOS POSIBLES", "keep": "CONSERVA SOLO LOS CASOS QUE CUMPLEN TODAS LAS CONDICIONES",
                "favourable_question": "¿qué casos cumplen la condición?", "count": "recuento",
                "case_a": "caso A", "case_b": "caso B", "case_c": "caso C", "restrictions": "restricciones del enunciado",
                "parity": "paridad", "divisibility": "divisibilidad", "factors": "factores",
                "statement_data": "datos del enunciado", "relationship_x": "relación con x", "isolate_x": "despeja x",
                "magnitude": "MAGNITUD", "value": "VALOR", "sort": "ORDENA", "frequencies": "CUENTA FRECUENCIAS",
                "measure": "MEDIDA", "mean": "media", "median": "mediana", "mode": "moda",
                "condition1": "C1", "condition2": "C2", "condition3": "C3",
                "data_relation": "datos → condición → incógnita", "requested_value": "valor y unidad pedidos",
                "diagram_instruction": "marca las medidas y separa las partes de la figura",
            },
        }
        ui = labels.get(language, labels["English"])
        data_text = number_line if numbers else "—"
        goal_text = anchor[:100] if anchor else "?"

        # A short arithmetic expression is shown without evaluating it. That
        # makes the visual useful while keeping the answer hidden.
        expression_match = re.search(
            r"\d+(?:[.,]\d+)?(?:\s*[+\-×÷*/]\s*\d+(?:[.,]\d+)?){1,4}",
            statement,
        )
        expression = expression_match.group(0) if expression_match else ""

        if topic == "arithmetic" and expression:
            if language == "Catalan":
                order_text = "× i ÷ abans de + i −"
                first_text = "resol primer la multiplicació o divisió"
                next_text = "després combina el terme restant"
            elif language == "Spanish":
                order_text = "× y ÷ antes que + y −"
                first_text = "resuelve primero la multiplicación o división"
                next_text = "después combina el término restante"
            else:
                order_text = "× and ÷ before + and −"
                first_text = "do the multiplication or division first"
                next_text = "then combine the remaining term"
            visual = (
                f"{ui['expression']}\n{expression}\n\n"
                f"{ui['order']}\n{order_text}\n\n"
                f"{ui['first']}\n{first_text}\n"
                f"{ui['next']}\n{next_text}\n\n"
                f"{ui['result']}\n?"
            )
        elif topic == "arithmetic":
            if language == "Catalan":
                meaning, operations = "què representa cada nombre?", "quantitat inicial → canvi → total?"
            elif language == "Spanish":
                meaning, operations = "¿qué representa cada número?", "cantidad inicial → cambio → ¿total?"
            else:
                meaning, operations = "what does each number represent?", "starting amount → change → total?"
            visual = (
                f"{ui['data']}\n{data_text}\n\n"
                f"{ui['relation']}\n{meaning}\n{operations}\n\n"
                f"{ui['goal']}\n{goal_text}"
            )
        elif topic == "geometry":
            lowered = statement.casefold()
            if any(word in lowered for word in ("area", "àrea", "área")):
                target = "ÀREA = ? (unitats²)" if language == "Catalan" else ("ÁREA = ? (unidades²)" if language == "Spanish" else "AREA = ? (square units)")
            elif any(word in lowered for word in ("perimeter", "perímetre", "perímetro")):
                target = "PERÍMETRE = ?" if language == "Catalan" else ("PERÍMETRO = ?" if language == "Spanish" else "PERIMETER = ?")
            elif any(word in lowered for word in ("angle", "angle", "ángulo")):
                target = "ANGLE = ?" if language in {"Catalan", "English"} else "ÁNGULO = ?"
            else:
                target = f"{ui['goal']} = ?"
            visual = (
                f"{ui['measure']}\n{data_text}\n\n"
                f"{ui['diagram']}\n{ui['diagram_instruction']}\n\n"
                f"{ui['result']}\n{target}\n"
                f"{goal_text}"
            )
        elif topic == "probability":
            visual = (
                f"{ui['sample']}\n{ui['cases']} = ?\n\n"
                f"{ui['favourable']}\n{ui['favourable_question']}\n\n"
                f"P = {ui['favourable'].lower()} / total de casos"
            )
        elif topic == "combinatorics":
            visual = (
                f"{ui['cases']}\n{ui['case_a']}  |  {ui['case_b']}  |  {ui['case_c']}\n\n"
                f"{ui['conditions']}\n{ui['restrictions']}\n\n"
                f"TOTAL = {ui['count']}(A) + {ui['count']}(B) + {ui['count']}(C)"
            )
        elif topic == "number_theory":
            visual = (
                f"{ui['data']}\n{data_text}\n\n"
                f"{ui['conditions']}\n{ui['parity']}  ·  {ui['divisibility']}  ·  {ui['factors']}\n\n"
                f"{ui['keep']}"
            )
        elif topic == "algebra":
            visual = (
                f"{ui['unknown']}\nx = ?\n\n"
                f"{ui['conditions']}\n{ui['statement_data']}\n↓\n"
                f"{ui['equation']}\n{ui['relationship_x']}\n↓\n{ui['goal']} = {ui['isolate_x']}"
            )
        elif topic == "measurement":
            visual = (
                f"{ui['data']}\n{data_text}\n\n"
                f"{ui['magnitude']}  |  {ui['value']}  |  {ui['units']}\nA          |  ?     |  ?\n\n"
                f"{ui['goal']}\n{ui['requested_value']}"
            )
        elif topic == "statistics":
            visual = (
                f"{ui['data_set']}\n{data_text}\n\n"
                f"{ui['sort']} → {ui['frequencies']}\n\n"
                f"{ui['measure']}\n{ui['mean']}  |  {ui['median']}  |  {ui['mode']}\n"
                f"{ui['goal']} = ?"
            )
        elif topic == "logic":
            visual = (
                f"{ui['candidate']}  |  {ui['condition1']}  |  {ui['condition2']}  |  {ui['condition3']}\n"
                f"{ui['case_a']}      |  ?   |  ?   |  ?\n"
                f"{ui['case_b']}      |  ?   |  ?   |  ?\n"
                f"{ui['case_c']}      |  ?   |  ?   |  ?\n\n"
                f"{ui['keep']}"
            )
        else:
            visual = (
                f"{ui['data']}\n{data_text}\n\n"
                f"{ui['relation']}\n{ui['data_relation']}\n\n"
                f"{ui['goal']}\n{goal_text}"
            )

        actions_by_topic = {
            "arithmetic": {
                "Catalan": ["Escriu al costat de cada nombre què representa i identifica l'acció que descriu el text.", "Reescriu l'expressió amb parèntesis si cal i marca quina operació té prioritat.", "Calcula només el primer pas intermedi i comprova si el resultat té sentit abans de continuar."],
                "English": ["Label each number with what it represents and identify the action described in the statement.", "Rewrite the expression with brackets if needed and mark which operation has priority.", "Work out only the first intermediate step and check whether it makes sense before continuing."],
                "Spanish": ["Anota qué representa cada número e identifica la acción que describe el enunciado.", "Reescribe la expresión con paréntesis si hace falta y marca qué operación tiene prioridad.", "Calcula solo el primer paso intermedio y comprueba si tiene sentido antes de continuar."],
            },
            "geometry": {
                "Catalan": ["Localitza a la figura la mesura indicada i marca exactament quina part et demanen.", "Fes un esquema i etiqueta les longituds; separa la figura completa de qualsevol part interior.", "Escriu la fórmula o propietat geomètrica que relaciona les dades amb la magnitud demanada, sense acabar el càlcul."],
                "English": ["Find the given measurement in the figure and mark exactly which region or length the question asks for.", "Sketch the shape and label its lengths; separate the whole figure from any inner or shaded part.", "Write the geometric formula or property linking the data to the requested quantity, but do not finish the calculation."],
                "Spanish": ["Localiza en la figura la medida indicada y marca exactamente qué parte te piden.", "Haz un esquema y etiqueta las longitudes; separa la figura completa de cualquier parte interior.", "Escribe la fórmula o propiedad geométrica que relaciona los datos con la magnitud pedida, sin terminar el cálculo."],
            },
            "combinatorics": {
                "Catalan": ["Fes una llista dels objectes o posicions i copia al costat cada restricció de l'enunciat.", "Divideix les possibilitats en casos que no se solapin i assigna cada disposició a un únic cas.", "Compta els casos per separat i comprova que no hagis repetit ni omès cap disposició."],
                "English": ["List the objects or positions and write each restriction from the statement next to them.", "Split the possibilities into non-overlapping cases and assign every arrangement to exactly one case.", "Count the cases separately and check that no arrangement has been repeated or omitted."],
                "Spanish": ["Haz una lista de los objetos o posiciones y anota cada restricción del enunciado.", "Divide las posibilidades en casos que no se solapen y asigna cada disposición a un único caso.", "Cuenta los casos por separado y comprueba que no hayas repetido ni omitido ninguna disposición."],
            },
            "probability": {
                "Catalan": ["Descriu l'experiment i escriu què ha de passar perquè un resultat sigui favorable.", "Enumera els resultats possibles i marca només els que compleixen la condició demanada.", "Compara els casos favorables amb el total i comprova abans si tots els resultats són igualment probables."],
                "English": ["Describe the experiment and state exactly what makes an outcome favourable.", "List the possible outcomes and mark only those that satisfy the requested condition.", "Compare favourable outcomes with the total and first check whether all outcomes are equally likely."],
                "Spanish": ["Describe el experimento y concreta qué debe ocurrir para que un resultado sea favorable.", "Enumera los resultados posibles y marca solo los que cumplen la condición pedida.", "Compara los casos favorables con el total y comprueba antes si todos son igual de probables."],
            },
            "number_theory": {
                "Catalan": ["Subratlla la propietat numèrica que et demanen: divisibilitat, primalitat, paritat o factors.", "Prova cada nombre amb una regla cada vegada i anota el resultat de cada comprovació.", "Combina les comprovacions i descarta qualsevol candidat que incompleixi una sola condició."],
                "English": ["Underline the number property being tested: divisibility, primality, parity or factors.", "Test each number with one rule at a time and record the result of every check.", "Combine the checks and reject any candidate that fails even one condition."],
                "Spanish": ["Subraya la propiedad numérica que se busca: divisibilidad, primalidad, paridad o factores.", "Prueba cada número con una regla cada vez y anota el resultado de cada comprobación.", "Combina las comprobaciones y descarta cualquier candidato que incumpla una sola condición."],
            },
            "algebra": {
                "Catalan": ["Defineix amb una lletra la quantitat desconeguda i identifica quines dades la determinen.", "Converteix cada condició en una part de l'equació i mantén iguals els dos membres.", "Aïlla la incògnita amb operacions equivalents i substitueix el resultat per comprovar l'equació."],
                "English": ["Give the unknown quantity a symbol and identify which facts determine it.", "Translate each condition into part of an equation and keep both sides balanced.", "Isolate the unknown using equivalent operations, then substitute it back to check the equation."],
                "Spanish": ["Asigna una letra a la cantidad desconocida e identifica qué datos la determinan.", "Convierte cada condición en una parte de la ecuación y mantén equilibrados ambos miembros.", "Despeja la incógnita con operaciones equivalentes y sustituye el resultado para comprobar la ecuación."],
            },
            "measurement": {
                "Catalan": ["Anota cada magnitud amb el seu valor i la seva unitat, sense barrejar distàncies, temps o masses.", "Converteix les unitats a una mateixa escala i escriu la relació entre les magnituds.", "Aplica aquesta relació i revisa que la unitat final coincideixi amb la quantitat demanada."],
                "English": ["Write each quantity with its value and unit; do not mix distances, times or masses.", "Convert units to a common scale and write the relationship between the quantities.", "Apply that relationship and check that the final unit matches the quantity requested."],
                "Spanish": ["Anota cada magnitud con su valor y unidad, sin mezclar distancias, tiempos o masas.", "Convierte las unidades a una misma escala y escribe la relación entre las magnitudes.", "Aplica esa relación y revisa que la unidad final coincida con la cantidad solicitada."],
            },
            "statistics": {
                "Catalan": ["Copia totes les observacions i indica quantes dades hi ha en total.", "Ordena els valors o prepara una taula de freqüències segons la mesura que et demanen.", "Aplica la definició correcta de mitjana, mediana o moda i revisa el recompte."],
                "English": ["Copy every observation and record how many data points there are in total.", "Sort the values or prepare a frequency table, depending on the measure requested.", "Apply the correct definition of mean, median or mode and verify the count."],
                "Spanish": ["Copia todas las observaciones e indica cuántos datos hay en total.", "Ordena los valores o prepara una tabla de frecuencias según la medida solicitada.", "Aplica la definición correcta de media, mediana o moda y revisa el recuento."],
            },
            "logic": {
                "Catalan": ["Escriu cada condició en una línia i identifica quins candidats has de comprovar.", "Construeix una taula amb un candidat per fila i una columna per cada condició.", "Descarta només els candidats que contradiguin alguna regla i verifica tots els restants."],
                "English": ["Write each condition on a separate line and identify the candidates to test.", "Build a table with one candidate per row and one column for each condition.", "Eliminate only candidates that contradict a rule and verify every candidate left."],
                "Spanish": ["Escribe cada condición en una línea e identifica qué candidatos debes comprobar.", "Construye una tabla con un candidato por fila y una columna por cada condición.", "Descarta solo los candidatos que contradigan alguna regla y verifica todos los restantes."],
            },
        }
        default_actions = {
            "Catalan": ["Subratlla la condició de l'enunciat que determina què has de trobar i identifica les dades útils.", "Escriu la relació matemàtica entre les dades abans de substituir-hi nombres.", "Aplica el pla pas a pas i comprova cada resultat intermedi amb totes les condicions."],
            "English": ["Underline the condition that defines what must be found and identify the useful data.", "Write the mathematical relationship between the data before substituting numbers.", "Apply the plan step by step and check every intermediate result against all the conditions."],
            "Spanish": ["Subraya la condición que define qué debes encontrar e identifica los datos útiles.", "Escribe la relación matemática entre los datos antes de sustituir números.", "Aplica el plan paso a paso y comprueba cada resultado intermedio con todas las condiciones."],
        }
        action_set = actions_by_topic.get(topic, {}).get(language, default_actions.get(language, default_actions["English"]))
        note = {
            "Catalan": "El model local no ha respost; aquesta pista de suport utilitza l'enunciat, les quantitats i el tema detectat.",
            "English": "The local model did not respond; this support hint uses the statement, its quantities and the detected topic.",
            "Spanish": "El modelo local no ha respondido; esta pista de apoyo utiliza el enunciado, sus cantidades y el tema detectado.",
        }.get(language, "The local model did not respond; this support hint uses the exercise statement and detected topic.")
        return {
            "title": self._hint_title(language, stage),
            "explanation": explanation,
            "visual": visual,
            "action": action_set[stage - 1],
            "kind": "fallback",
            "note": note,
        }

    def hint(self, student_id):
        student_id = self._check_student_id(student_id)
        with self._connect() as db:
            student = db.execute(
                "SELECT current_problem_key, hint_stage FROM students WHERE student_id=?",
                (student_id,),
            ).fetchone()
            if student is None or not student["current_problem_key"]:
                raise TutorError("No hi ha cap exercici actiu per donar-hi pistes.")
            problem_key = student["current_problem_key"]
            problem = self.by_key.get(problem_key)
            if problem is None:
                raise TutorError("No s'ha trobat l'exercici actiu.")
            current_stage = int(student["hint_stage"])
            if current_stage >= 3:
                raise TutorError("Ja has utilitzat les tres pistes d'aquest exercici. Pots provar de respondre o passar al següent exercici.")
            stage = current_stage + 1
            cached = db.execute(
                "SELECT hint_text FROM hint_history WHERE student_id=? AND problem_key=? AND stage=?",
                (student_id, problem_key, stage),
            ).fetchone()

        if cached:
            try:
                hint_data = json.loads(cached["hint_text"])
                if not isinstance(hint_data, dict):
                    raise ValueError
            except (json.JSONDecodeError, ValueError, TypeError):
                # Compatibility with hint history created by older versions.
                hint_data = {
                    "title": self._hint_title(self._hint_language(problem), stage),
                    "explanation": str(cached["hint_text"]),
                    "visual": "",
                    "action": "Continua amb el pas següent de l'exercici.",
                    "kind": "ai",
                }
        else:
            try:
                hint_data = self._generate_hint(problem, stage)
            except Exception as exc:
                hint_data = self._fallback_hint(problem, stage, reason=str(exc))

        with self._connect() as db:
            latest = db.execute(
                "SELECT current_problem_key, hint_stage FROM students WHERE student_id=?",
                (student_id,),
            ).fetchone()
            if (latest is None or latest["current_problem_key"] != problem_key
                    or int(latest["hint_stage"]) != current_stage):
                raise TutorError("L'estat de l'exercici ha canviat. Torna a demanar la pista.")
            now = time.time()
            serialised = json.dumps(hint_data, ensure_ascii=False)
            db.execute(
                "INSERT OR IGNORE INTO hint_history(student_id, problem_key, stage, hint_text, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (student_id, problem_key, stage, serialised, now),
            )
            saved = db.execute(
                "SELECT hint_text FROM hint_history WHERE student_id=? AND problem_key=? AND stage=?",
                (student_id, problem_key, stage),
            ).fetchone()
            try:
                hint_data = json.loads(saved["hint_text"])
            except (json.JSONDecodeError, TypeError):
                pass
            db.execute(
                "UPDATE students SET hint_stage=?, updated_at=? WHERE student_id=?",
                (stage, now, student_id),
            )

        return {
            "stage": stage,
            "remaining": 3 - stage,
            "max_hints": 3,
            **hint_data,
        }

