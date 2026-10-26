"""Prompt workloads for the Qwen3.8-27B sweep.

Three families, each isolating a different part of the pipeline:

  decode_*  short prompt, long generation  -> measures decode throughput,
            which is where speculative decoding (MTP) pays off
  prefill_* long prompt, short generation  -> measures prompt-processing
            throughput and time-to-first-token
  quality_* correctness probes, including needle-in-haystack at depth, to
            detect degradation caused by KV-cache quantisation at long context

Speculative decoding acceptance is content-dependent: predictable text (code,
structured output) drafts far better than open-ended prose. We therefore keep
both kinds so the report can show the spread rather than a single flattering
number.
"""

import re

# --------------------------------------------------------------- filler text

_FILLER = [
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

# Roughly 1.3 tokens per word for this kind of English prose.
_WORDS_PER_TOKEN = 0.75


def _filler_notes(target_tokens):
    """Build filler text of approximately target_tokens tokens."""
    target_words = int(target_tokens * _WORDS_PER_TOKEN)
    parts, words, i = [], 0, 0
    while words < target_words:
        s = _FILLER[i % len(_FILLER)]
        parts.append(f"Note {i + 1}: {s}")
        words += len(s.split()) + 2
        i += 1
    return parts


def needle_prompt(target_tokens, depth=0.5, code="BLUE-742"):
    """Filler of ~target_tokens with one fact buried at `depth` through it."""
    parts = _filler_notes(target_tokens)
    idx = max(1, min(len(parts) - 1, int(len(parts) * depth)))
    parts.insert(idx, f"Note SPECIAL: The secret locker code for the gym is {code}.")
    doc = "\n".join(parts)
    return (
        "Here is a long list of community notes.\n\n"
        + doc
        + "\n\nQuestion: What is the secret locker code for the gym? "
        "Answer with just the code and nothing else."
    )


# --------------------------------------------------------------- decode set

DECODE_WORKLOADS = [
    {
        "id": "decode_code",
        "kind": "decode",
        "max_tokens": 512,
        "prompt": (
            "Write a complete Python implementation of a thread-safe LRU cache "
            "class with get, put, and stats methods. Include full type hints "
            "and a short docstring on every method. Output only code."
        ),
    },
    {
        "id": "decode_prose",
        "kind": "decode",
        "max_tokens": 512,
        "prompt": (
            "Explain how a four-stroke petrol engine works, in plain English, "
            "for a first-year engineering student. Write about 400 words of "
            "flowing prose. Do not use bullet points or headings."
        ),
    },
    {
        "id": "decode_structured",
        "kind": "decode",
        "max_tokens": 512,
        "prompt": (
            "Output a JSON array of 25 objects. Each object must have the keys "
            "id (integer), name (string), city (string), and score (float). "
            "Output only valid JSON, no commentary."
        ),
    },
]

# --------------------------------------------------------------- prefill set
# Sized at run time against the context length actually being tested.

PREFILL_FRACTIONS = [0.25, 0.75]


def prefill_workloads(ctx_tokens):
    """Prompts that fill a meaningful fraction of the configured context."""
    out = []
    for frac in PREFILL_FRACTIONS:
        # Leave headroom for the generation and the chat template.
        target = int(ctx_tokens * frac) - 512
        if target < 1500:
            continue
        out.append(
            {
                "id": f"prefill_{int(frac * 100)}pct",
                "kind": "prefill",
                "max_tokens": 64,
                "target_prompt_tokens": target,
                "prompt": needle_prompt(target, depth=0.5),
                "check": lambda r: "blue-742" in r.lower(),
            }
        )
    return out


# --------------------------------------------------------------- quality set


def _strip_fences(text):
    m = re.search(r"```(?:python|json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def _check_palindrome(resp):
    ns = {}
    try:
        exec(_strip_fences(resp), ns)
        f = ns["is_palindrome"]
        return (
            f("Never odd or even") is True
            and f("hello") is False
            and f("A man a plan a canal Panama") is True
        )
    except Exception:
        return False


def _check_json(resp):
    m = re.search(r"\{.*\}", resp, re.DOTALL)
    if not m:
        return False
    try:
        import json

        obj = json.loads(m.group(0))
        return (
            str(obj.get("name", "")).lower() == "anita"
            and int(obj.get("age", 0)) == 34
            and str(obj.get("city", "")).lower() == "pune"
        )
    except Exception:
        return False


QUALITY_WORKLOADS = [
    {
        "id": "q_arithmetic",
        "kind": "quality",
        "max_tokens": 256,
        "prompt": "What is 847 multiplied by 36? Give only the final number.",
        "check": lambda r: "30492" in r.replace(",", ""),
    },
    {
        "id": "q_logic",
        "kind": "quality",
        "max_tokens": 256,
        "prompt": (
            "Sara has 3 brothers and 2 sisters. How many sisters does each of "
            "Sara's brothers have? Give only the number."
        ),
        "check": lambda r: re.search(r"\b3\b", r) is not None,
    },
    {
        "id": "q_code",
        "kind": "quality",
        "max_tokens": 400,
        "prompt": (
            "Write a Python function is_palindrome(s) that returns True if the "
            "string is a palindrome, ignoring case and spaces. Reply with only "
            "the code."
        ),
        "check": _check_palindrome,
    },
    {
        "id": "q_json",
        "kind": "quality",
        "max_tokens": 256,
        "prompt": (
            "Extract name, age, and city as a JSON object from this text. "
            "Output only the JSON.\n\n"
            "Anita is a 34-year-old teacher who lives in Pune and enjoys "
            "painting on weekends."
        ),
        "check": _check_json,
    },
]


def needle_suite(ctx_tokens, depths=(0.1, 0.5, 0.9)):
    """Needle-in-haystack at several depths, sized to ~85% of the context."""
    target = int(ctx_tokens * 0.85) - 512
    if target < 1500:
        return []
    out = []
    for d in depths:
        code = {0.1: "BLUE-742", 0.5: "GREEN-317", 0.9: "AMBER-905"}.get(d, "BLUE-742")
        out.append(
            {
                "id": f"needle_d{int(d * 100)}",
                "kind": "needle",
                "max_tokens": 32,
                "target_prompt_tokens": target,
                "prompt": needle_prompt(target, depth=d, code=code),
                "check": (lambda c: (lambda r: c.lower() in r.lower()))(code),
            }
        )
    return out
