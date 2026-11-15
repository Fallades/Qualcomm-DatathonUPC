"""Ask the small language model for one letter, A to E."""
import re

from .config import CONFIG
from .client import post

URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"

# Keep the system prompt stable: llama-server caches the start of the prompt,
# so text that never changes is only read once.
SYSTEM = ("You solve multiple-choice math competition problems for school students. Read the "
          "question and all answer choices carefully. Solve the exact problem, use the choices "
          "as part of the reasoning, eliminate choices that contradict the rules or result, "
          "and verify that the selected letter matches its value. For counting questions, "
          "distinguish the number of visible objects from the number of valid arrangements, "
          "ways, or configurations requested. Reply with only one letter: A, B, C, D, or E.")
TEXT_INSTRUCTIONS = (
    "Solve the exact question before answering. Use the available choices to reason and "
    "discard choices that are impossible or contradict the problem. Verify that the final "
    "letter matches its option value. Do not confuse counting visible objects with counting "
    "valid ways or configurations. Reply with only one letter: A, B, C, D, or E."
)
VISION_SYSTEM = ("You solve a visual multiple-choice math problem. The question text and "
                 "answer choices in the user message are authoritative. Read the problem "
                 "carefully and use the labeled diagram and answer-options images as the "
                 "visual evidence needed to solve it. First solve the exact "
                 "question: apply its rules, relationships, order, paths, patterns, or "
                 "calculations. Then match the result or valid condition to the corresponding "
                 "choice letter. Do not select a letter from the diagram's position or from a "
                 "simple visual count; distinguish visible objects from valid ways or "
                 "configurations. Before replying, check that the letter matches the correct "
                 "choice. Reply with only the letter A, B, C, D, or E.")
VISION_INSTRUCTIONS = (
    "Use the image as evidence, but solve the exact question in the JSON text. If JSON choices "
    "are present, match your result to their exact letter. If they are missing, the image is "
    "the full problem: read the five options A), B), C), D), and E) at the bottom. Verify the "
    "choice, then reply with only one letter: A, B, C, D, or E. Do not confuse visible object "
    "counts with the number of valid ways or configurations requested."
)


def format_problem(problem, sees_image=False):
    lines = [problem["text"].strip()]
    options = problem.get("options") or {}
    for letter in "ABCDE":
        if letter in options:
            lines.append(f"{letter}) {options[letter]}")
    if not options:
        lines.append("(The five options are in the bottom of the picture.)" if sees_image else
                     "(The options are in a picture that you can't see. Pick the most likely letter.)")
    return "\n".join(lines)


def build_messages(problem, examples=(), image_b64=None):
    """image_b64 is a PNG as base64 text. Only a server started with --mmproj can see it."""
    messages = [{"role": "system", "content": VISION_SYSTEM if image_b64 else SYSTEM}]
    for ex in examples:  # solved problems retrieved with RAG, if any
        messages.append({"role": "user", "content": format_problem(ex) + "\nAnswer:"})
        messages.append({"role": "assistant", "content": ex["label"]})
    text = ("QUESTION TEXT AND ANSWER CHOICES (from JSON):\n" +
            format_problem(problem, sees_image=bool(image_b64)))
    instructions = VISION_INSTRUCTIONS if image_b64 else TEXT_INSTRUCTIONS
    text += "\n" + instructions
    if image_b64:
        images = image_b64 if isinstance(image_b64, list) else [image_b64]
        content = [{"type": "text", "text": text}]
        for index, image in enumerate(images):
            if len(images) == 1 and not problem.get("options"):
                label = "FULL PROBLEM IMAGE: diagram/table and five options at bottom"
            else:
                label = "DIAGRAM IMAGE" if index == 0 else "ANSWER OPTIONS IMAGE"
            mime = "image/jpeg" if image.startswith("/9j/") else "image/png"
            content.append({"type": "text", "text": label + ":"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64," + image}})
    else:
        content = text
    messages.append({"role": "user", "content": content})
    return messages


def answer_letter(problem, examples=(), timeout=60, image_b64=None, url=URL):
    """Return (letter, timings, raw_content), allowing concise vision reasoning."""
    payload = {
        "messages": build_messages(problem, examples, image_b64),
        "max_tokens": 1,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
        "grammar": "root ::= [A-E]",
    }
    response = post(url, payload, timeout=timeout)
    raw_content = response["choices"][0]["message"]["content"].strip()
    content = raw_content.upper()
    if image_b64:
        matches = re.findall(r"\bANSWER\s*:\s*([A-E])\b", content)
        if not matches:
            matches = re.findall(
                r"\b(?:ANSWER|CORRECT(?:\s+OPTION|\s+ANSWER)?)\s*(?:IS|:)\s*([A-E])\b",
                content)
        letter = matches[-1] if matches else (
            content if re.fullmatch(r"[A-E]", content)
            else CONFIG["FALLBACK_ANSWER"])
    else:
        matches = re.findall(r"\bANSWER\s*:\s*([A-E])\b", content)
        letter = matches[-1] if matches else (
            content if re.fullmatch(r"[A-E]", content)
            else CONFIG["FALLBACK_ANSWER"])
    return letter, response.get("timings", {}), raw_content
