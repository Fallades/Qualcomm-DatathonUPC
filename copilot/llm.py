"""Ask the small language model for one letter, A to E."""
import re

from .config import CONFIG
from .client import post

URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"

# Keep the system prompt stable: llama-server caches the start of the prompt,
# so text that never changes is only read once.
SYSTEM = ("You solve multiple-choice math competition problems for school students. "
          "Read the problem and the options, then reply with the letter of the correct option.")
VISION_SYSTEM = ("You solve a visual multiple-choice math problem. Read the diagram, identify "
                 "the requested quantity, calculate it, and compare all choices. Reply in "
                 "exactly two lines: first `ANSWER: X`, then one short sentence explaining why. "
                 "Put the answer first and never guess from an option's position.")


def format_problem(problem, sees_image=False):
    lines = [problem["text"].strip()]
    options = problem.get("options") or {}
    for letter in "ABCDE":
        if letter in options:
            lines.append(f"{letter}) {options[letter]}")
    if not options:
        lines.append("(The options are in the picture.)" if sees_image else
                     "(The options are in a picture that you can't see. Pick the most likely letter.)")
    return "\n".join(lines)


def build_messages(problem, examples=(), image_b64=None):
    """image_b64 is a PNG as base64 text. Only a server started with --mmproj can see it."""
    messages = [{"role": "system", "content": VISION_SYSTEM if image_b64 else SYSTEM}]
    for ex in examples:  # solved problems retrieved with RAG, if any
        messages.append({"role": "user", "content": format_problem(ex) + "\nAnswer:"})
        messages.append({"role": "assistant", "content": ex["label"]})
    text = format_problem(problem, sees_image=bool(image_b64))
    text += ("\nRead the diagram and all visible choices. Reply first with `ANSWER: X`, "
             "then give one short reason. Do not write a long analysis."
             if image_b64 else "\nAnswer:")
    if image_b64:
        mime = "image/jpeg" if image_b64.startswith("/9j/") else "image/png"
        content = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64," + image_b64}},
                   {"type": "text", "text": text}]
    else:
        content = text
    messages.append({"role": "user", "content": content})
    return messages


def answer_letter(problem, examples=(), timeout=60, image_b64=None, url=URL):
    """Return (letter, timings, raw_content), allowing concise vision reasoning."""
    payload = {
        "messages": build_messages(problem, examples, image_b64),
        "max_tokens": 48 if image_b64 else 1,
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
