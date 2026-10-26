#!/usr/bin/env python3
"""
Benchmark Qwen 3.8 27B on Apple Silicon (MacBook Pro M5 Max / Metal backend).
Runs llama-server across draft depths (n=0 to n=4), measures prompt ingestion,
decode throughput (t/s), TTFT, and draft acceptance rates.
"""

import argparse
import json
import os
import subprocess
import time
import urllib.request
import uuid

PROMPTS = [
    {
        "id": "code_eval",
        "prompt": "Write a Python implementation of an LRU Cache with O(1) get and put operations using collections.OrderedDict. Include docstrings, error handling, and unit tests."
    }
]

DRAFT_LEVELS = [0, 1, 2, 3, 4]

def wait_for_server(url="http://127.0.0.1:8080/health", timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False

def query_server(prompt_text, max_tokens=256):
    req_id = str(uuid.uuid4())
    data = json.dumps({
        "prompt": prompt_text,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "id_slot": -1,
        "cache_prompt": False,
        "stream": False
    }).encode("utf-8")
    
    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": req_id
    }
    
    req = urllib.request.Request("http://127.0.0.1:8080/completion", data=data, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    t1 = time.time()
    
    tokens_predicted = result.get("tokens_predicted", 0)
    eval_duration_ms = result.get("timings", {}).get("predicted_ms", (t1 - t0) * 1000)
    tok_per_sec = (tokens_predicted / (eval_duration_ms / 1000.0)) if eval_duration_ms > 0 else 0
    
    return {
        "tokens_predicted": tokens_predicted,
        "wall_time_s": t1 - t0,
        "tok_per_sec": tok_per_sec,
        "timings": result.get("timings", {})
    }

def run_draft_sweep(llama_server_path, model_path, repeats=3):
    results = []
    for draft_n in DRAFT_LEVELS:
        print(f"=== Testing Draft Level n={draft_n} ===")
        cmd = [
            llama_server_path,
            "-m", model_path,
            "-c", "32768",
            "-fa", "1",
            "--port", "8080"
        ]
        if draft_n > 0:
            cmd.extend(["--draft-max", str(draft_n)])
        
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wait_for_server():
            print(f"Failed to start server for draft {draft_n}")
            proc.terminate()
            continue
        
        # Warmup
        query_server("Hello world", max_tokens=16)
        
        for r in range(repeats):
            for p in PROMPTS:
                res = query_server(p["prompt"], max_tokens=256)
                res["draft_depth"] = draft_n
                res["repeat"] = r
                results.append(res)
                print(f"  Draft {draft_n} | Rep {r+1} -> {res['tok_per_sec']:.2f} tok/s")
        
        proc.terminate()
        proc.wait()
        time.sleep(2)
        
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved sweep results to results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="llama-server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    run_draft_sweep(args.server, args.model, args.repeats)
