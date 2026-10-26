"""Fast end-to-end validation of the harness before the long sweep.

Runs one small config through the real code path with thinking on and off,
and prints every metric we intend to report, so bugs surface in ~2 minutes
instead of two hours into the sweep.
"""

import json

import bench
import harness as H
import workloads as W

H._kill_stragglers("llama-server.exe")
baseline = H.idle_vram()
print(f"idle VRAM baseline = {baseline:.0f} MB\n")

for think in (True, False):
    c = bench.cfg(ctx=4096, kv="f16", spec="draft-mtp", n_max=2, think=think)
    print(f"\n{'=' * 70}\nconfig: {c['id']}\n{'=' * 70}")

    rt = H.LlamaServer(c, bench.LOGS)
    try:
        print(f"loaded in {rt.start()}s")
        resident = H.nvidia_query(["memory.used"])[0] - baseline
        print(f"resident VRAM = {resident:.0f} MB\n")

        for task in (W.DECODE_WORKLOADS[0], W.QUALITY_WORKLOADS[0]):
            mark = len(rt.log_text())
            rec = bench.run_one_workload(rt.chat_url(), "qwen3.8", task, c, baseline)
            acc = H.parse_draft_acceptance_tail(rt.log_text(), mark)
            print(f"--- {task['id']} ---")
            for k in (
                "ttft_s", "ttfa_s", "decode_s", "wall_s",
                "prompt_tokens", "completion_tokens",
                "thinking_chunks", "thinking_chars", "stream_chunks",
                "decode_tps", "decode_tps_stream", "prefill_tps",
                "vram_peak_mib", "power_mean_w", "tokens_per_wh", "correct",
            ):
                print(f"    {k:<20} {rec.get(k)}")
            print(f"    accept_rate          {acc.get('accept_rate')} "
                  f"(mean len {acc.get('mean_accepted_len')})")
            print(f"    head                 {rec.get('response_head', '')[:70]!r}\n")

        print("--- aggregate draft stats ---")
        print(json.dumps(H.parse_draft_acceptance(rt.log_text(), rt.metrics()), indent=2))
        print(f"--- /metrics bytes: {len(rt.metrics())} ---")
    finally:
        rt.stop()
        H.wait_vram_release(baseline)
        print("GPU released.")
