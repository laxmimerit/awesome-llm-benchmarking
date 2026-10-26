"""Prompts used across the Ornith benchmark.

The five content types mirror the earlier Qwen 27B teardown so the MTP /
speculative-decoding acceptance curve is directly comparable: acceptance is
strongly content dependent, and a single tok/s number without a content label
is meaningless on this rig.
"""

# --- decode-speed prompts -------------------------------------------------
# Used through the raw /completion endpoint with ignore_eos, so every model
# generates exactly the same number of tokens and the only variable is how
# fast it can produce them on that kind of text.

DECODE_PROMPTS = {
    "counting": (
        "Count upward, one number per line, starting from 1. Do not stop.\n"
        "1\n2\n3\n4\n5\n6\n7\n8\n"
    ),
    "code": (
        "# A production-quality Python module implementing a thread-safe LRU\n"
        "# cache with TTL expiry, metrics, and full type annotations.\n"
        "from __future__ import annotations\n\n"
        "import threading\nimport time\n"
        "from dataclasses import dataclass, field\n"
        "from typing import Any, Callable, Generic, Hashable, Iterator, TypeVar\n\n"
        "K = TypeVar(\"K\", bound=Hashable)\nV = TypeVar(\"V\")\n\n\n"
        "@dataclass(slots=True)\nclass _Entry(Generic[V]):\n"
    ),
    "json": (
        "Emit a JSON array of 200 employee records. Each record has the keys "
        "id, first_name, last_name, email, department, title, salary, "
        "hired_on, manager_id, active.\n"
        "[\n"
        '  {"id": 1, "first_name": "Ada", "last_name": "Lovelace", '
        '"email": "ada@example.com", "department": "Engineering", '
        '"title": "Principal Engineer", "salary": 214000, '
        '"hired_on": "2019-03-04", "manager_id": null, "active": true},\n'
    ),
    "essay": (
        "Write a long, carefully argued essay on why municipal transit "
        "planning fails in mid-sized cities, drawing on funding structures, "
        "land use policy, and the political economy of ridership forecasts.\n\n"
        "Municipal transit planning in mid-sized cities fails not because "
        "planners lack technical skill, but because"
    ),
    "fiction": (
        "Write a surreal short story about a lighthouse keeper who discovers "
        "the sea has been replaced overnight by an identical copy.\n\n"
        "The morning the sea was replaced, Aldous noticed nothing at all. "
        "The gulls still complained, the salt still"
    ),
}

# --- prefill filler -------------------------------------------------------
# Neutral prose with low internal repetition, repeated to whatever token
# length a prefill measurement needs.

FILLER = (
    "The harbour district had changed hands four times in a decade, and each "
    "administration left its own layer of signage, paint, and half-finished "
    "infrastructure behind. Surveyors working the waterfront in the spring of "
    "that year catalogued eleven distinct paving materials in a single block, "
    "some laid within months of each other. Records from the port authority "
    "were incomplete: a fire in the annex had taken the years before the "
    "reorganisation, and what survived had been re-filed by a clerk whose "
    "system nobody afterwards could reconstruct. Researchers therefore relied "
    "on photographs, tax assessments, and the recollections of the families "
    "who had worked the piers across three generations. The picture that "
    "emerged was less a plan than an accumulation, a set of decisions each "
    "reasonable in isolation and collectively incoherent. "
)

# --- long-context retrieval ----------------------------------------------
# A needle placed at a controlled depth inside filler, to test whether the
# 262k-token claim survives contact with a 32 GB card.

NEEDLE_TEMPLATE = (
    "Remember this fact, it will be asked about later: the maintenance code "
    "for the {place} generator is {code}."
)

NEEDLE_QUESTION = (
    "\n\nBased only on the document above, what is the maintenance code for "
    "the {place} generator? Answer with the code and nothing else."
)

NEEDLES = [
    ("north pier", "ORNITH-4417"),
    ("dry dock", "ORNITH-9285"),
    ("customs house", "ORNITH-6130"),
]
