#!/usr/bin/env python3
"""Serve the offline adaptive maths tutor.

Run after generating topic-tagged training data and (optionally) building the
embedding index:
  python3 tutor_server.py --input data/train_topics.jsonl --index data/index.db
Then open http://127.0.0.1:8001.
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from copilot.rag import Index  # noqa: E402
from copilot.topics import TOPICS  # noqa: E402
from copilot.tutor import AdaptiveTutor, TutorError  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def make_handler(tutor, languages):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write("  " + fmt % args + "\n")

        def send_json(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def send_file(self, path, content_type):
            with open(path, "rb") as stream:
                data = stream.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def body(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length > 16_384:
                raise TutorError("La petició és massa gran.")
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            student_id = params.get("student_id", [""])[0]
            try:
                if parsed.path == "/":
                    return self.send_file(os.path.join(ROOT, "explorer", "tutor.html"), "text/html; charset=utf-8")
                if parsed.path == "/api/meta":
                    return self.send_json(200, {"languages": languages, "topics": TOPICS,
                                                "problems": len(tutor.problems),
                                                "with_index": tutor.index is not None})
                if parsed.path == "/api/profile":
                    return self.send_json(200, tutor.profile(student_id))
                if parsed.path == "/api/current":
                    return self.send_json(200, {"problem": tutor.current(student_id)})
                return self.send_json(404, {"error": "No s'ha trobat la ruta."})
            except (TutorError, ValueError, KeyError) as error:
                return self.send_json(400, {"error": str(error)})

        def do_POST(self):
            try:
                payload = self.body()
                student_id = payload.get("student_id", "")
                if self.path == "/api/start":
                    problem = tutor.start(student_id, language=payload.get("language", "cat"),
                                          topic=payload.get("topic", ""),
                                          start_level=payload.get("start_level", 1))
                    return self.send_json(200, {"problem": problem, "profile": tutor.profile(student_id)})
                if self.path == "/api/next":
                    problem = tutor.next_problem(student_id, language=payload.get("language"),
                                                 topic=payload.get("topic", ""))
                    return self.send_json(200, {"problem": problem, "profile": tutor.profile(student_id)})
                if self.path == "/api/answer":
                    result = tutor.submit_answer(student_id, payload.get("choice"))
                    return self.send_json(200, result)
                if self.path == "/api/hint":
                    return self.send_json(200, tutor.hint(student_id))
                return self.send_json(404, {"error": "No s'ha trobat la ruta."})
            except (TutorError, ValueError, KeyError, json.JSONDecodeError) as error:
                return self.send_json(400, {"error": str(error)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/train_topics.jsonl")
    parser.add_argument("--index", default="data/index.db")
    parser.add_argument("--db", default="results/student_profiles.db")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        parser.error(f"{args.input} no existeix. Executa primer classify_topics.py.")
    problems = read_jsonl(args.input)
    index = Index(args.index) if os.path.exists(args.index) else None
    tutor = AdaptiveTutor(problems, args.db, index=index)
    languages = sorted({p.get("language") for p in tutor.problems if p.get("language")})
    if not tutor.problems:
        parser.error("No hi ha exercicis amb les cinc opcions OCR llegides.")
    print(f"Tutor preparat: {len(tutor.problems)} exercicis amb opcions completes; "
          f"{len(languages)} idiomes; índex {'disponible' if index else 'no disponible'}.")
    print(f"Obre http://{args.host}:{args.port} (Ctrl+C per aturar)", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(tutor, languages)).serve_forever()


if __name__ == "__main__":
    main()
