"""Dump raw SSE chunks so we can see exactly which delta fields Qwen3.8 uses."""

import json
import urllib.request

import bench
import harness as H

c = bench.cfg(ctx=4096, kv="f16", spec="draft-mtp", n_max=2)
H._kill_stragglers("llama-server.exe")
rt = H.LlamaServer(c, bench.LOGS)
print(f"loaded in {rt.start()}s\n")

try:
    for label, extra in (
        ("DEFAULT", {}),
        ("NOTHINK", {"chat_template_kwargs": {"enable_thinking": False}}),
    ):
        payload = {
            "model": "qwen3.8",
            "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            "max_tokens": 200,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        payload.update(extra)
        req = urllib.request.Request(
            "http://127.0.0.1:8099/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        keys, n, first = set(), 0, []
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                obj = json.loads(body)
                n += 1
                for ch in obj.get("choices", []):
                    d = ch.get("delta") or {}
                    keys.update(d.keys())
                    if len(first) < 4:
                        first.append(d)
                if obj.get("usage"):
                    usage = obj["usage"]
        print(f"=== {label} ===")
        print(f"  chunks={n}  delta keys seen={sorted(keys)}")
        print(f"  first deltas: {json.dumps(first)[:300]}")
        print(f"  usage: {usage}\n")
finally:
    rt.stop()
