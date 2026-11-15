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
VISION_SYSTEM = ("You solve a visual Kangaroo Math multiple-choice problem. The question text "
                 "and answer choices are authoritative. First inspect the whole image and "
                 "transcribe every relevant label, number, colour, shape, grid, table, path, "
                 "and arrow; keep the diagram separate from the answer options. Then solve "
                 "the exact question: calculate lengths, angles, perimeters, or areas; apply "
                 "rotations and other movements; compare shaded regions; follow paths or "
                 "patterns; or count objects and valid arrangements. Do not rely on apparent "
                 "scale, do not infer missing objects, and do not choose a letter from its "
                 "position. Use the diagram's rules and labels, match the result to the "
                 "corresponding choice, check it once, and reply only A, B, C, D, or E.")
VISION_INSTRUCTIONS = (
    "Inspect the whole image before solving. Use the diagram, labels, colours, arrows, and "
    "table/grid as evidence. If JSON choices are present, match the result to their exact "
    "letter. If they are missing, read the five options A), B), C), D), and E) at the bottom. "
    "For rotations, track positions before and after each turn; for areas, split or subtract "
    "shapes; for shaded regions, compare boundaries; for counting, count valid configurations "
    "rather than visible objects. Verify once, then reply only A, B, C, D, or E."
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
