"""Charts for the Qwen3.8-27B settings sweep (Plotly + Kaleido).

Palette: the validated categorical order used across this project --
slot 1 blue, slot 2 orange, slot 3 aqua, slot 4 yellow. Verified with the
dataviz validator: all checks pass, with a contrast WARN on the aqua/yellow
slots that is relieved by the direct value labels every chart carries.

One measure per axis throughout: speed, memory and quality never share a
y-axis, because a dual-scale chart invites false comparisons.

Only charts the stages that exist in results.json, so it can be run while the
sweep is still going.
"""

import json
import os

import plotly.graph_objects as go

import summarize as S

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "blog")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, BASELINE_C = "#898781", "#e1e0d9", "#c3c2b7"
FONT = 'system-ui, "Segoe UI", sans-serif'

SUB = "Qwen3.8-27B Q4_K_M · RTX 5090 32GB · llama.cpp b10448 · median of 3"


def layout(title, subtitle, height=460, **kw):
    lay = dict(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:13px;color:{INK2}'>{subtitle}</span>",
            font=dict(family=FONT, size=18, color=INK),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK2),
        width=900,
        height=height,
        margin=dict(l=80, r=50, t=95, b=65),
        showlegend=False,
        barcornerradius=4,
    )
    lay.update(kw)
    return lay


def axis(**kw):
    a = dict(
        gridcolor=GRID,
        linecolor=BASELINE_C,
        tickfont=dict(color=MUTED, size=12),
        title_font=dict(color=INK2, size=13),
        zeroline=False,
    )
    a.update(kw)
    return a


def save(fig, name):
    fig.write_image(os.path.join(OUT, name), scale=2)
    print("saved", name)


def pad(values, frac=0.18):
    hi = max(v for v in values if v is not None)
    return [0, hi * (1 + frac)]


# ------------------------------------------------------------------ load

data = S.load()
ROWS = [r for r in S.rows(data) if "FAILED" not in r]
BY_ID = {r["id"]: r for r in ROWS}
print(f"{len(ROWS)} usable configs")


# Every axis the sweep varies, with the value the baseline config uses. Any
# axis a caller does not mention is pinned to its baseline value rather than
# left free.
AXES = {
    "runtime": "llamacpp", "ctx_k": 32, "kv": "f16", "fa": "on",
    "spec": "none", "n_max": None, "backend_sampling": None,
    "batch": None, "ubatch": None, "think": True, "effort": None,
    "chat_template": None, "tag": None,
}


def find(**criteria):
    """The row matching `criteria`, with every unmentioned axis held at baseline.

    Matching only on the axes a chart happens to name is how this file has
    silently mixed configs three times: a filter for "n=3" also matched the
    reasoning-effort row, the q8_0 row, and later the backend-sampling and
    reverse-order control rows, and `find` returned whichever came first in
    results.json. Pinning the unmentioned axes makes each lookup describe
    exactly one config, so a new stage can never quietly rewrite an old chart.
    """
    unknown = set(criteria) - set(AXES)
    if unknown:
        raise KeyError(f"find() got non-axis criteria: {sorted(unknown)}")
    want = dict(AXES)
    want.update(criteria)
    hits = [r for r in ROWS if all(r.get(k) == v for k, v in want.items())]
    if len(hits) > 1:
        raise AssertionError(
            f"find({criteria}) is ambiguous: {[h['id'] for h in hits]}"
        )
    return hits[0] if hits else None


def have(rows):
    return [r for r in rows if r is not None]


# --------------------------------------------------- 1. MTP speculation ladder

ladder = have(
    [find(spec="none", ctx_k=32, kv="f16", think=True, effort=None)]
    + [
        find(spec="draft-mtp", n_max=n, ctx_k=32, kv="f16", think=True, effort=None)
        for n in (1, 2, 3, 4, 5)
    ]
)
if len(ladder) >= 3:
    labels = ["MTP off" if r["spec"] == "none" else f"n={r['n_max']}" for r in ladder]
    vals = [r["decode_tps"] for r in ladder]
    base = ladder[0]["decode_tps"]
    colors = [BASELINE_C] + [BLUE] * (len(ladder) - 1)
    text = [
        f"{v:.0f}" if i == 0 else f"{v:.0f}  ({v / base:.2f}x)"
        for i, v in enumerate(vals)
    ]
    fig = go.Figure()
    fig.add_bar(
        x=labels,
        y=vals,
        marker=dict(color=colors),
        text=text,
        textposition="outside",
        textfont=dict(color=INK, size=13),
    )
    fig.update_layout(
        **layout(
            "Speculative decoding: how many tokens to draft",
            f"Decode throughput at 32K context, f16 KV cache. {SUB}",
            height=430,
        ),
        bargap=0.35,
        xaxis=axis(title="--spec-draft-n-max"),
        yaxis=axis(title="decode tokens / second", range=pad(vals)),
    )
    save(fig, "chart-mtp-ladder.png")

    # Acceptance rate is the mechanism behind the ladder.
    acc = [r for r in ladder if r.get("accept_rate") is not None and r["spec"] != "none"]
    if len(acc) >= 3:
        av = [r["accept_rate"] * 100 for r in acc]
        fig = go.Figure()
        fig.add_bar(
            x=[f"n={r['n_max']}" for r in acc],
            y=av,
            marker=dict(color=ORANGE),
            text=[f"{v:.0f}%" for v in av],
            textposition="outside",
            textfont=dict(color=INK, size=13),
        )
        fig.update_layout(
            **layout(
                "Why deeper drafting stops paying",
                "Share of drafted tokens the full model accepted. "
                "The MTP head has one layer, so accuracy falls as depth grows.",
                height=430,
            ),
            bargap=0.35,
            xaxis=axis(title="--spec-draft-n-max"),
            yaxis=axis(title="draft acceptance rate (%)", range=[0, 109]),
        )
        save(fig, "chart-mtp-acceptance.png")

# --------------------------------------------------- 2. thinking on vs off

think_rows = have(
    [
        find(think=True, spec="none", ctx_k=32),
        find(think=False, spec="none", ctx_k=32),
        find(think=True, spec="draft-mtp", n_max=2, ctx_k=32),
        find(think=False, spec="draft-mtp", n_max=2, ctx_k=32),
    ]
)
if len(think_rows) == 4:
    names = ["MTP off", "MTP n=2"]
    on = [think_rows[0]["decode_tps"], think_rows[2]["decode_tps"]]
    off = [think_rows[1]["decode_tps"], think_rows[3]["decode_tps"]]
    fig = go.Figure()
    fig.add_bar(
        name="thinking on", x=names, y=on, marker=dict(color=BLUE),
        text=[f"{v:.0f}" for v in on], textposition="outside",
        textfont=dict(color=INK, size=12),
    )
    fig.add_bar(
        name="thinking off", x=names, y=off, marker=dict(color=ORANGE),
        text=[f"{v:.0f}" for v in off], textposition="outside",
        textfont=dict(color=INK, size=12),
    )
    fig.update_layout(
        **layout(
            "Reasoning mode costs more than any server flag",
            f"Decode throughput with Qwen3.8's thinking on vs off. {SUB}",
            height=440,
            showlegend=True,
            legend=dict(orientation="h", y=1.02, x=0.02, font=dict(size=12)),
        ),
        bargap=0.4,
        bargroupgap=0.12,
        xaxis=axis(),
        yaxis=axis(title="decode tokens / second", range=pad(on + off)),
    )
    save(fig, "chart-thinking-speed.png")

# --------------------------------------------------- 2b. reasoning effort

eff_rows = have(
    [find(effort=e, spec="draft-mtp", n_max=2, ctx_k=32, think=True)
     for e in ("xhigh", "medium", "low")]
    + [find(think=False, spec="draft-mtp", n_max=2, ctx_k=32)]
)
if len(eff_rows) == 4:
    labels = ["xhigh<br>(default)", "medium", "low", "thinking<br>off"]
    colors = [BLUE, BLUE, BLUE, ORANGE]

    # How much thinking each level actually spends. This is the claim under
    # test: the tweet says the default "burns your tokens".
    think_tok = [r["thinking_chunks"] or 0 for r in eff_rows]
    fig = go.Figure()
    fig.add_bar(
        x=labels, y=think_tok, marker=dict(color=colors),
        text=[f"{v:.0f}" for v in think_tok], textposition="outside",
        textfont=dict(color=INK, size=13),
    )
    fig.update_layout(
        **layout(
            "What each reasoning level actually costs",
            f"Median thinking tokens per request. {SUB}",
            height=440,
        ),
        bargap=0.42,
        xaxis=axis(title="reasoning_effort"),
        yaxis=axis(title="thinking tokens per request",
                   range=pad(think_tok) if any(think_tok) else [0, 1]),
    )
    save(fig, "chart-reasoning-tokens.png")

    # Latency the user actually feels: time until the answer starts, which
    # excludes nothing and includes all the thinking.
    ttfa = [r["ttfa_s"] for r in eff_rows if r.get("ttfa_s")]
    if len(ttfa) == 4:
        fig = go.Figure()
        fig.add_bar(
            x=labels, y=ttfa, marker=dict(color=colors),
            text=[f"{v:.2f}s" for v in ttfa], textposition="outside",
            textfont=dict(color=INK, size=13),
        )
        fig.update_layout(
            **layout(
                "How long before the answer starts",
                "Time to first answer token, including all thinking. "
                f"{SUB}",
                height=440,
            ),
            bargap=0.42,
            xaxis=axis(title="reasoning_effort"),
            yaxis=axis(title="seconds to first answer token", range=pad(ttfa)),
        )
        save(fig, "chart-reasoning-latency.png")

    # Speed is only worth having if the answers stay right.
    q = [r["quality_pct"] for r in eff_rows]
    if all(v is not None for v in q):
        fig = go.Figure()
        fig.add_bar(
            x=labels, y=q,
            marker=dict(color=[GOOD if v >= 95 else CRITICAL for v in q]),
            text=[f"{v:.0f}%" for v in q], textposition="outside",
            textfont=dict(color=INK, size=13),
        )
        fig.update_layout(
            **layout(
                "Does less thinking cost accuracy?",
                "Correctness across reasoning, coding, JSON and "
                "needle-in-haystack tasks",
                height=440,
            ),
            bargap=0.42,
            xaxis=axis(title="reasoning_effort"),
            yaxis=axis(title="tasks passed (%)", range=[0, 112]),
        )
        save(fig, "chart-reasoning-quality.png")

# --------------------------------------------------- 3. KV cache type

kv_rows = have([find(kv=k, spec="draft-mtp", n_max=2, ctx_k=32) for k in ("f16", "q8_0", "q4_0")])
if len(kv_rows) >= 2:
    ks = [r["kv"] for r in kv_rows]
    vram = [r["vram_proc_mib"] / 1024 for r in kv_rows]
    fig = go.Figure()
    fig.add_bar(
        x=ks, y=vram, marker=dict(color=AQUA),
        text=[f"{v:.1f} GB" for v in vram], textposition="outside",
        textfont=dict(color=INK, size=13),
    )
    fig.update_layout(
        **layout(
            "KV cache precision and memory",
            f"Peak VRAM at 32K context, idle baseline subtracted. {SUB}",
            height=420,
        ),
        bargap=0.45,
        xaxis=axis(title="--cache-type-k / -v"),
        yaxis=axis(title="VRAM (GB)", range=pad(vram)),
    )
    save(fig, "chart-kv-vram.png")

    q = [r["quality_pct"] for r in kv_rows]
    if all(v is not None for v in q):
        fig = go.Figure()
        fig.add_bar(
            x=ks, y=q,
            marker=dict(color=[GOOD if v >= 95 else CRITICAL for v in q]),
            text=[f"{v:.0f}%" for v in q], textposition="outside",
            textfont=dict(color=INK, size=13),
        )
        fig.update_layout(
            **layout(
                "Does a quantised KV cache cost accuracy?",
                "Correctness across reasoning, coding, JSON and needle-in-haystack tasks",
                height=420,
            ),
            bargap=0.45,
            xaxis=axis(title="--cache-type-k / -v"),
            yaxis=axis(title="tasks passed (%)", range=[0, 112]),
        )
        save(fig, "chart-kv-quality.png")

# --------------------------------------------------- 4. context scaling

# Must pin every axis except context, or the reasoning-effort configs (all
# at 32K) get swept in and pile up as overlapping points on one x value.
ctx_rows = sorted(
    [
        r for r in ROWS
        if r["spec"] == "draft-mtp"
        and r["n_max"] == 2
        and r["think"]
        and r["effort"] is None
        and r["runtime"] == "llamacpp"
        and r["fa"] == "on"
        and r["kv"] == ("f16" if r["ctx_k"] <= 64 else "q8_0")
        and r["decode_tps"]
    ],
    key=lambda r: r["ctx_k"],
)
# One row per context length, and only chart a genuine sweep.
_seen = set()
ctx_rows = [
    r for r in ctx_rows
    if not (r["ctx_k"] in _seen or _seen.add(r["ctx_k"]))
]
if len({r["ctx_k"] for r in ctx_rows}) >= 4:
    xs = [r["ctx_k"] for r in ctx_rows]
    ys = [r["decode_tps"] for r in ctx_rows]
    fig = go.Figure()
    fig.add_scatter(
        x=xs, y=ys, mode="lines+markers+text",
        line=dict(color=BLUE, width=2),
        marker=dict(size=9, color=BLUE, line=dict(width=2, color=SURFACE)),
        text=[f"{v:.0f}" for v in ys], textposition="top center",
        textfont=dict(color=INK, size=11),
    )
    fig.update_layout(
        **layout(
            "Raising the context limit barely costs decode speed",
            "Measured with short prompts, so this is the cost of configuring a "
            f"bigger window, not of filling it. {SUB}",
            height=440,
        ),
        xaxis=axis(title="context window (K tokens)", type="log",
                   tickvals=xs, ticktext=[f"{x}K" for x in xs]),
        yaxis=axis(title="decode tokens / second", range=pad(ys)),
    )
    save(fig, "chart-ctx-speed.png")

    vr = [r["vram_proc_mib"] / 1024 for r in ctx_rows]
    fig = go.Figure()
    fig.add_scatter(
        x=xs, y=vr, mode="lines+markers+text",
        line=dict(color=AQUA, width=2),
        marker=dict(size=9, color=AQUA, line=dict(width=2, color=SURFACE)),
        text=[f"{v:.1f}" for v in vr], textposition="top center",
        textfont=dict(color=INK, size=11),
    )
    fig.add_hline(
        y=32607 / 1024, line=dict(color=CRITICAL, width=2, dash="dot"),
        annotation_text="RTX 5090 32GB limit",
        annotation_position="bottom right",
        annotation_font=dict(color=CRITICAL, size=12),
    )
    fig.update_layout(
        **layout(
            "Memory cost of a longer context",
            "f16 KV cache up to 64K, q8_0 at 128K and 256K (f16 would not "
            f"fit). {SUB}",
            height=440,
        ),
        xaxis=axis(title="context window (K tokens)", type="log",
                   tickvals=xs, ticktext=[f"{x}K" for x in xs]),
        yaxis=axis(title="VRAM (GB)", range=[0, 34]),
    )
    save(fig, "chart-ctx-vram.png")

# ------------------------------------ 4b. what a FULL context actually costs

full = [r for r in ctx_rows if r.get("long_ttft_s") and r.get("long_prompt_tokens")]
if len(full) >= 4:
    xs = [r["ctx_k"] for r in full]
    ys = [r["long_ttft_s"] for r in full]
    fig = go.Figure()
    fig.add_scatter(
        x=xs, y=ys, mode="lines+markers+text",
        line=dict(color=ORANGE, width=2),
        marker=dict(size=9, color=ORANGE, line=dict(width=2, color=SURFACE)),
        text=[f"{v:.0f}s" if v >= 10 else f"{v:.1f}s" for v in ys],
        textposition="top center",
        textfont=dict(color=INK, size=11),
    )
    fig.update_layout(
        **layout(
            "The real cost of a long context is waiting, not memory",
            "Time to first token with the window ~85% full · "
            "Qwen3.8-27B Q4_K_M · RTX 5090 · MTP n=2",
            height=450,
        ),
        xaxis=axis(title="context window (K tokens)", type="log",
                   tickvals=xs, ticktext=[f"{x}K" for x in xs]),
        # Explicit decade ticks; Plotly's default log minor ticks read as
        # bare 2s and 5s and are easy to misparse as data values.
        yaxis=axis(
            title="seconds to first token (log scale)",
            type="log",
            tickvals=[1, 10, 100, 1000],
            ticktext=["1s", "10s", "100s", "1000s"],
        ),
    )
    save(fig, "chart-ctx-fullprompt-latency.png")

# --------------------------------------------------- 5. runtime comparison

# Built as a decomposition rather than a head-to-head. Ollama runs llama.cpp
# as a child process, so a two-bar "Ollama vs llama.cpp" chart cannot be
# measuring engines -- it is measuring the flags each one defaults to. Adding
# Ollama's flags to llama-server one group at a time shows where the gap
# actually comes from.
OLL = dict(spec="draft-mtp", n_max=4, backend_sampling=True,
           batch=1024, ubatch=1024)
rt_rows = have(
    [
        find(kv="f16"),
        find(kv="f16", **OLL),
        find(kv="f16", chat_template="chatml", **OLL),
        find(runtime="ollama", kv="f16"),
    ]
)
if len(rt_rows) == 4:
    labels = ["llama.cpp<br>stock", "+ Ollama's<br>MTP flags",
              "+ Ollama's<br>chatml template", "Ollama<br>(default)"]
    vals = [r["decode_tps"] for r in rt_rows]
    base = vals[0]
    fig = go.Figure()
    fig.add_bar(
        x=labels, y=vals,
        marker=dict(color=[BASELINE_C, BLUE, AQUA, ORANGE]),
        text=[f"{v:.0f}" + ("" if i == 0 else f"  ({v/base:.2f}x)")
              for i, v in enumerate(vals)],
        textposition="outside",
        textfont=dict(color=INK, size=13),
    )
    fig.update_layout(
        **layout(
            "The Ollama speed gap is defaults, not runtime",
            "Ollama runs llama-server as a child process. Adding its flags to "
            f"stock llama.cpp closes most of the gap. {SUB}",
            height=470,
        ),
        bargap=0.42,
        xaxis=axis(),
        yaxis=axis(title="decode tokens / second", range=pad(vals)),
    )
    save(fig, "chart-runtime.png")

print("charts done")
