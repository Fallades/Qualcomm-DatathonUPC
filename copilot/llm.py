"""Ask the small language model for one letter, A to E."""
import re

from .config import CONFIG
from .client import post

URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"

# Keep the system prompt stable: llama-server caches the start of the prompt,
# so text that never changes is only read once.
SYSTEM = ("You solve multiple-choice math competition problems for school students. "
          "Read the problem and the options, then reply with the letter of the correct option.")
VISION_SYSTEM = ("You solve a visual multiple-choice math problem. The question text and "
                 "answer choices in the user message are authoritative. Read the problem "
                 "carefully and use the labeled diagram and answer-options images as the "
                 "visual evidence needed to solve it. First solve the exact "
                 "question: apply its rules, relationships, order, paths, patterns, or "
                 "calculations. Then match the result or valid condition to the corresponding "
                 "choice letter. Do not select a letter from the diagram's position or from a "
                 "simple visual count. Before replying, check that the letter, choice value, "
                 "and explanation agree. Reply in exactly two lines: first `ANSWER: X` with "
                 "only the letter, then `REASON:` followed by at most 10 words.")


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
    text = ("QUESTION TEXT AND ANSWER CHOICES (use these as the problem statement):\n" +
            format_problem(problem, sees_image=bool(image_b64)))
    text += ("\nUse the diagram as evidence, but solve the exact question in the JSON text. "
             "If JSON choices are present, match your result to their exact letter. If they "
             "are missing, read all five choices in the ANSWER OPTIONS IMAGE. Verify that your "
             "chosen letter agrees with your reason. Reply first with `ANSWER: X` using only "
             "one letter, then write `REASON:` and at most 10 words. Do not write analysis."
             if image_b64 else "\nAnswer:")
    if image_b64:
        images = image_b64 if isinstance(image_b64, list) else [image_b64]
        content = [{"type": "text", "text": text}]
        for index, image in enumerate(images):
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
        "max_tokens": 32 if image_b64 else 1,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
    }
    if not image_b64:
        payload["grammar"] = "root ::= [A-E]"
    response = post(url, payload, timeout=timeout)
    raw_content = response["choices"][0]["message"]["content"].strip()
    content = raw_content.upper()
    if image_b64:
        matches = re.findall(r"\bANSWER\s*:\s*([A-E])\b", content)
        if not matches:
            matches = re.findall(
                r"\b(?:ANSWER|CORRECT(?:\s+OPTION|\s+ANSWER)?)\s*(?:IS|:)\s*([A-E])\b",
                content)
        letter = matches[-1] if matches else CONFIG["FALLBACK_ANSWER"]
    else:
        letter = content
    return letter, response.get("timings", {}), raw_content
