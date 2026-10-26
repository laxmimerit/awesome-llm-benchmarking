"""Is llama.cpp's 'runtime deficit' real, or an artefact of how it was timed?

With every performance flag matched to Ollama's, Ollama still measured ~22-29%
faster. Same binary, same flags -- that should not happen, which points at the
measurement rather than the runtime.

The two sides are not timed the same way:

    ollama    server-reported eval_count / eval_duration  (pure model eval)
    llamacpp  client-side SSE stream timing               (eval + streaming)

llama-server reports its own `timings` block on the same request. Reading both
numbers off one response separates the model's decode rate from the cost of
delivering it over HTTP. If the client-side figure is well below the
server-side one, the gap is transport, not the runtime.
"""

import json
import time
import urllib.request

import harness as H

CFG = {
    "id": "timing-parity", "ctx": 32768, "kv": "f16", "flash_attn": "on",
    "spec": "draft-mtp", "n_max": 4, "backend_sampling": True,
    "batch": 1024, "ubatch": 1024, "parallel": 1,
}
PROMPT = ("Write a detailed technical explanation of how speculative decoding "
          "works in transformer inference, including why it is lossless.")

import bench

srv = H.LlamaServer(CFG, bench.LOGS)
print("starting llama-server with Ollama-matched flags...")
print("  ", " ".join(srv.build_args()[1:]))
srv.start()
print(f"loaded in {srv.load_s}s\n")

url = srv.chat_url()

for label, stream in (("non-streaming", False), ("streaming", True)):
    payload = {
        "model": "qwen3.8",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 600, "temperature": 0, "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    first = None
    timings = None
    ntok = 0
    if not stream:
        with urllib.request.urlopen(req, timeout=600) as f:
            obj = json.loads(f.read())
        wall = time.time() - t0
        timings = obj.get("timings")
        ntok = obj.get("usage", {}).get("completion_tokens")
    else:
        with urllib.request.urlopen(req, timeout=600) as f:
            for raw in f:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                obj = json.loads(body)
                if obj.get("timings"):
                    timings = obj["timings"]
                if obj.get("usage"):
                    ntok = obj["usage"].get("completion_tokens") or ntok
                ch = obj.get("choices") or [{}]
                d = ch[0].get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    if first is None:
                        first = time.time()
                    ntok_stream = True
        wall = time.time() - t0

    print(f"--- {label} ---")
    print(f"  wall clock                {wall:.3f}s   tokens={ntok}")
    if ntok and wall:
        client_rate = ntok / wall
        print(f"  client-side  tok/s        {client_rate:.2f}   "
              f"(what the harness recorded for llama.cpp)")
    if timings:
        sr = timings.get("predicted_per_second")
        print(f"  server-side  tok/s        {sr:.2f}   "
              f"(predicted_n={timings.get('predicted_n')}, "
              f"predicted_ms={timings.get('predicted_ms'):.0f})")
        print(f"  prompt eval  tok/s        {timings.get('prompt_per_second', 0):.2f}")
        if ntok and wall and sr:
            print(f"  -> transport overhead     {(sr / (ntok / wall) - 1) * 100:+.1f}% "
                  f"(server faster than client-observed by this much)")
    else:
        print("  server timings: not present in response")
    print()

srv.stop()
print("server stopped")
