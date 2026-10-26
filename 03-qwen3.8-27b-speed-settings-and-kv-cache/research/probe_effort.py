"""Verify Qwen3.8's reasoning-effort levels actually change behaviour.

The chat template claims to accept only low / medium / xhigh. This confirms
that (a) each level is accepted and changes how much the model thinks, and
(b) an unsupported level is genuinely rejected rather than silently ignored.

Runs against an already-running llama-server on port 8099.
"""

import json
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8099/v1/chat/completions"
PROMPT = "A shop sells pens at 12 rupees each. Ravi buys 7 pens and pays with a 100 rupee note. How much change does he get?"


def ask(kwargs, label):
    payload = {
        "model": "qwen3.8",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 2000,
        "temperature": 0,
        "stream": False,
    }
    if kwargs:
        payload["chat_template_kwargs"] = kwargs
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as f:
            obj = json.loads(f.read())
    except urllib.error.HTTPError as e:
        # The server echoes the template's own exception text, which contains
        # non-cp1252 characters that crash printing on a Windows console.
        body = e.read().decode("utf-8", "replace")
        body = body.encode("ascii", "replace").decode("ascii")
        body = " ".join(body.split())
        print(f"{label:<28} REJECTED  HTTP {e.code}: {body[:200]}")
        return
    msg = obj["choices"][0]["message"]
    think = msg.get("reasoning_content") or ""
    ans = (msg.get("content") or "").strip().replace("\n", " ")
    u = obj.get("usage", {})
    print(
        f"{label:<28} total_tok={u.get('completion_tokens'):<6} "
        f"think_chars={len(think):<6} answer={ans[:48]!r}"
    )


print(f"prompt: {PROMPT}\n")
ask(None, "template default")
for lvl in ("xhigh", "medium", "low"):
    ask({"reasoning_effort": lvl}, f"reasoning_effort={lvl}")
ask({"enable_thinking": False}, "enable_thinking=False")
print("\n-- levels llama.cpp offers but Qwen3.8 should reject --")
for lvl in ("minimal", "high", "max"):
    ask({"reasoning_effort": lvl}, f"reasoning_effort={lvl}")
