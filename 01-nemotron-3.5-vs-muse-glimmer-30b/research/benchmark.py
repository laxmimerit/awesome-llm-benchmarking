"""Simple-task benchmark: Nemotron 3.5 Lightning vs Muse Glimmer 30B.

Runs a suite of simple tasks against local Ollama models with a 64K context
window, records answers, speed stats, and auto-grades where possible.
Results are written to results.json (raw) for the blog write-up.

Only uses the Python standard library.
"""

import json
import re
import time
import urllib.request

OLLAMA = "http://localhost:11434"
NUM_CTX = 65536
MODELS = {
    "nemotron": "nemotron-3.5-lightning:latest",
    "glimmer": "muse-glimmer:latest",
}

# ---------------------------------------------------------------- tasks

def make_needle_prompt():
    """~30k words of filler with one hidden fact in the middle -> ~40K tokens."""
    filler_sentences = [
        "The city library opens at nine in the morning and closes at eight in the evening.",
        "Local farmers reported a good harvest of wheat and mustard this season.",
        "The community center is planning a new yoga class for senior citizens.",
        "Bus route 42 now stops near the railway station after the schedule change.",
        "The school science fair will be held in the main auditorium next month.",
        "A new bakery opened near the park and sells fresh bread every morning.",
        "The town council discussed repairing the old bridge across the river.",
        "Cricket practice for the junior team happens every Saturday at the ground.",
        "The weather station recorded mild temperatures and light rain this week.",
        "Volunteers planted two hundred trees along the highway last Sunday.",
    ]
    parts = []
    total_words = 0
    i = 0
    needle_inserted = False
    while total_words < 30000:
        s = filler_sentences[i % len(filler_sentences)]
        parts.append(f"Note {i + 1}: {s}")
        total_words += len(s.split()) + 2
        i += 1
        if total_words > 15000 and not needle_inserted:
            parts.append(
                "Note SPECIAL: The secret locker code for the gym is BLUE-742."
            )
            needle_inserted = True
    doc = "\n".join(parts)
    return (
        "Here is a long list of community notes.\n\n"
        + doc
        + "\n\nQuestion: What is the secret locker code for the gym? "
        "Answer with just the code."
    )


TASKS = [
    {
        "id": "math_arithmetic",
        "prompt": "What is 847 multiplied by 36? Give only the final number.",
        "check": lambda r: "30492" in r.replace(",", ""),
    },
    {
        "id": "math_word",
        "prompt": (
            "A shop sells pens at 12 rupees each. Ravi buys 7 pens and pays "
            "with a 100 rupee note. How much change does he get? Give only "
            "the number."
        ),
        "check": lambda r: "16" in r,
    },
    {
        "id": "logic",
        "prompt": (
            "Sara has 3 brothers and 2 sisters. How many sisters does each of "
            "Sara's brothers have? Give only the number."
        ),
        "check": lambda r: "3" in r,
    },
    {
        "id": "knowledge",
        "prompt": "What is the capital of Australia? One word answer.",
        "check": lambda r: "canberra" in r.lower(),
    },
    {
        "id": "coding_simple",
        "prompt": (
            "Write a Python function is_palindrome(s) that returns True if "
            "the string is a palindrome, ignoring case and spaces. Reply with "
            "only the code, no explanation."
        ),
        "check": "exec_palindrome",
    },
    {
        "id": "coding_bugfix",
        "prompt": (
            "This Python function should return the sum of the first n "
            "natural numbers but it is wrong. Fix it and reply with only the "
            "corrected code.\n\n"
            "def sum_n(n):\n"
            "    total = 0\n"
            "    for i in range(n):\n"
            "        total += i\n"
            "    return total"
        ),
        "check": "exec_sum_n",
    },
    {
        "id": "summarize",
        "prompt": (
            "Summarize this in one sentence:\n\n"
            "The James Webb Space Telescope, launched in December 2021, has "
            "changed how astronomers study the early universe. Its infrared "
            "instruments can see through cosmic dust and detect light from "
            "galaxies that formed more than 13 billion years ago. Scientists "
            "have used it to study the atmospheres of planets outside our "
            "solar system, finding water vapor and carbon dioxide. The "
            "telescope orbits the sun about 1.5 million kilometers from "
            "Earth, much farther than the Hubble Space Telescope."
        ),
        "check": None,  # graded by hand
    },
    {
        "id": "instruction_following",
        "prompt": (
            "List exactly three colors as bullet points. Each bullet must be "
            "a single word. Do not write anything else."
        ),
        "check": "check_bullets",
    },
    {
        "id": "json_extract",
        "prompt": (
            "Extract name, age, and city as a JSON object from this text. "
            "Output only the JSON.\n\n"
            "Anita is a 34-year-old teacher who lives in Pune and enjoys "
            "painting on weekends."
        ),
        "check": "check_json",
    },
    {
        "id": "long_context_needle",
        "prompt": make_needle_prompt(),
        "check": lambda r: "blue-742" in r.lower(),
    },
]

# ---------------------------------------------------------------- graders


def strip_code_fences(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def grade(task, response):
    check = task["check"]
    if check is None:
        return None
    if callable(check):
        return bool(check(response))
    code = strip_code_fences(response)
    try:
        if check == "exec_palindrome":
            ns = {}
            exec(code, ns)
            f = ns["is_palindrome"]
            return (
                f("Never odd or even") is True
                and f("hello") is False
                and f("A man a plan a canal Panama") is True
            )
        if check == "exec_sum_n":
            ns = {}
            exec(code, ns)
            f = ns["sum_n"]
            return f(5) == 15 and f(1) == 1 and f(100) == 5050
        if check == "check_bullets":
            lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
            bullets = [l for l in lines if re.match(r"^[-*\u2022]\s*\w+$", l)]
            return len(lines) == 3 and len(bullets) == 3
        if check == "check_json":
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if not m:
                return False
            obj = json.loads(m.group(0))
            return (
                str(obj.get("name", "")).lower() == "anita"
                and int(obj.get("age", 0)) == 34
                and str(obj.get("city", "")).lower() == "pune"
            )
    except Exception:
        return False
    return False


# ---------------------------------------------------------------- runner


def api(path, payload, timeout=1200):
    req = urllib.request.Request(
        OLLAMA + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read())


def unload(model):
    try:
        api("/api/generate", {"model": model, "keep_alive": 0})
    except Exception:
        pass


def run_model(key, model):
    results = []
    print(f"\n=== {model} ===", flush=True)
    for task in TASKS:
        t0 = time.time()
        r = api(
            "/api/chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": task["prompt"]}],
                "stream": False,
                "options": {"num_ctx": NUM_CTX, "temperature": 0.2},
            },
        )
        wall = time.time() - t0
        msg = r.get("message", {})
        content = msg.get("content", "")
        thinking = msg.get("thinking", "")
        eval_count = r.get("eval_count", 0)
        eval_dur = r.get("eval_duration", 1) / 1e9
        p_count = r.get("prompt_eval_count", 0)
        p_dur = r.get("prompt_eval_duration", 1) / 1e9
        ok = grade(task, content)
        results.append(
            {
                "task": task["id"],
                "correct": ok,
                "response": content,
                "thinking_tokens": len(thinking.split()) if thinking else 0,
                "gen_tokens": eval_count,
                "gen_tps": round(eval_count / eval_dur, 1) if eval_dur else 0,
                "prompt_tokens": p_count,
                "prompt_tps": round(p_count / p_dur, 1) if p_dur else 0,
                "wall_seconds": round(wall, 1),
            }
        )
        print(
            f"  {task['id']}: correct={ok} gen={eval_count}tok "
            f"@{results[-1]['gen_tps']}tps wall={wall:.1f}s",
            flush=True,
        )
    return results


def main():
    import sys

    path = (
        "D:/Claude AI/AI Model Performance/nemotron-3.5-vs-muse-glimmer/"
        "research/results.json"
    )
    try:
        with open(path, encoding="utf-8") as f:
            out = json.load(f)
    except Exception:
        out = {}
    keys = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS)
    for key in keys:
        model = MODELS[key]
        for other in MODELS.values():
            if other != model:
                unload(other)
        out[key] = run_model(key, model)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
