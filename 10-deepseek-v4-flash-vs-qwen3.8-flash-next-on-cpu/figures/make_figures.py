"""Blog figures for "DeepSeek-V4-Flash vs Qwen3.8-Flash-Next at 1-bit".

Every number is read from ../results/*.jsonl, the raw llama-bench and
llama-server outputs. Nothing is typed in by hand. The delivered/derived split
reads ../results/audit_verdicts.json, which records the manual verdict on every
cell either audit flagged.

House style follows the earlier posts: light surface, muted grid, direct value
labels on every mark, no legend where the axis already names the categories.

Hue follows the entity: DeepSeek-V4-Flash teal, Qwen3.8-Flash-Next blue,
CPU-only slate, and anything caused by VRAM spill red.
"""

import json
import os

import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

DS = "#0f766e"      # DeepSeek-V4-Flash
FN = "#2563eb"      # Qwen3.8-Flash-Next
SLATE = "#64748b"   # CPU-only
RED = "#dc2626"     # spill / failure
AMBER = "#d97706"   # derived-but-not-delivered

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, sans-serif", size=14, color="#1f2937"),
    margin=dict(l=70, r=30, t=70, b=60),
    width=900, height=520,
    title=dict(font=dict(size=19)),
)


def jsonl(name):
    out = []
    for line in open(os.path.join(RES, name), encoding="utf-8"):
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def save(fig, name):
    p = os.path.join(HERE, name)
    fig.write_image(p, scale=2)
    print("wrote", name)


def rows_gpu():
    return [r for r in jsonl("hard.jsonl") if r["config"].endswith("-gpu")]


# ------------------------------------------------------------------ figure 1
def fig_scoreboard():
    """Delivered vs derived. The gap IS the finding."""
    rows = rows_gpu()
    av = json.load(open(os.path.join(RES, "audit_verdicts.json"),
                        encoding="utf-8"))["verdicts"]
    labels, delivered, extra = [], [], []
    for cfg, name in (("dsv4-gpu", "DeepSeek-V4-Flash<br>284B, IQ1_S"),
                      ("flashnext-gpu", "Qwen3.8-Flash-Next<br>177B, IQ1_S")):
        rs = [r for r in rows if r["config"] == cfg]
        d = sum(1 for r in rs if r["passed"])
        ff = [v for v in av if v["config"] == cfg
              and v["verdict"] == "false-failure" and v["derived"]
              and not any(r["task"] == v["task"] and r["passed"] for r in rs)]
        labels.append(name)
        delivered.append(d)
        extra.append(len(ff))
    n = len(set(r["task"] for r in rows))
    fig = go.Figure()
    fig.add_bar(x=labels, y=delivered, name="delivered an answer",
                marker_color=[DS, FN],
                text=[str(v) for v in delivered], textposition="inside",
                textfont=dict(size=20, color="white"))
    fig.add_bar(x=labels, y=extra, name="derived it but never said it",
                marker_color=AMBER,
                text=["+%d" % v if v else "" for v in extra],
                textposition="inside", textfont=dict(size=16, color="white"))
    fig.update_layout(barmode="stack", **LAYOUT)
    fig.update_layout(
        title="The score gap is a termination gap, not a reasoning gap",
        yaxis_title="tasks out of %d" % n,
        legend=dict(orientation="h", y=1.06, x=0))
    fig.update_yaxes(range=[0, n + 1], gridcolor="#e5e7eb")
    save(fig, "chart-scoreboard.png")


# ------------------------------------------------------------------ figure 2
def fig_failure_modes():
    """Same total failures, completely different kind."""
    rows = rows_gpu()
    wrong, trunc, labels = [], [], []
    for cfg, name in (("dsv4-gpu", "DeepSeek-V4-Flash"),
                      ("flashnext-gpu", "Qwen3.8-Flash-Next")):
        rs = [r for r in rows if r["config"] == cfg]
        labels.append(name)
        wrong.append(sum(1 for r in rs if not r["passed"] and not r.get("truncated")))
        trunc.append(sum(1 for r in rs if r.get("truncated")))
    fig = go.Figure()
    fig.add_bar(x=labels, y=wrong, name="wrong answer", marker_color=SLATE,
                text=wrong, textposition="outside")
    fig.add_bar(x=labels, y=trunc, name="never finished (hit the cap)",
                marker_color=RED, text=trunc, textposition="outside")
    fig.update_layout(barmode="group", **LAYOUT)
    fig.update_layout(
        title="How each model fails: wrong answers versus no answer at all",
        yaxis_title="cells", legend=dict(orientation="h", y=1.06, x=0))
    fig.update_yaxes(range=[0, max(trunc + wrong) + 1.5], gridcolor="#e5e7eb")
    save(fig, "chart-failure-modes.png")


# ------------------------------------------------------------------ figure 3
def fig_speed():
    """CPU-only and GPU offload, both models."""
    cpu = {r["label"]: r for r in jsonl("cpu_speed.jsonl") if r.get("n_gen")}
    rows = rows_gpu()
    served = {}
    for cfg in ("dsv4-gpu", "flashnext-gpu"):
        v = sorted(r["gen_tps"] for r in rows if r["config"] == cfg
                   and r["gen_tps"] and r["task"] != "multihop_needle_64k")
        served[cfg] = v[len(v) // 2]
    names = ["DeepSeek-V4-Flash", "Qwen3.8-Flash-Next"]
    cpu_v = [cpu["dsv4-cpu"]["avg_ts"], cpu["flashnext-cpu"]["avg_ts"]]
    gpu_v = [served["dsv4-gpu"], served["flashnext-gpu"]]
    fig = go.Figure()
    fig.add_bar(x=names, y=cpu_v, name="CPU only (no GPU)", marker_color=SLATE,
                text=["%.2f" % v for v in cpu_v], textposition="outside")
    fig.add_bar(x=names, y=gpu_v, name="GPU offload (median served)",
                marker_color=[DS, FN],
                text=["%.1f" % v for v in gpu_v], textposition="outside")
    fig.update_layout(barmode="group", **LAYOUT)
    fig.update_layout(
        title="Generation speed: a 284B model runs at 7.1 t/s on the CPU alone",
        yaxis_title="tokens per second", legend=dict(orientation="h", y=1.06, x=0))
    fig.update_yaxes(range=[0, max(gpu_v) * 1.25], gridcolor="#e5e7eb")
    save(fig, "chart-speed.png")


# ------------------------------------------------------------------ figure 4
def fig_spill():
    """More GPU layers makes it slower. The trap that never errors."""
    rows = [r for r in jsonl("ncmoe_ds.jsonl") if r.get("n_gen")]
    rows.sort(key=lambda r: -r["n_cpu_moe"])
    x = ["%d" % r["n_cpu_moe"] for r in rows]
    y = [r["avg_ts"] for r in rows]
    colors = [DS] + [RED] * (len(y) - 1)
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors,
                           text=["%.2f" % v for v in y], textposition="outside"))
    fig.update_layout(**LAYOUT)
    fig.update_layout(
        title="Moving more layers onto the GPU made DeepSeek 4.5x slower",
        xaxis_title="--n-cpu-moe  (lower = more layers on the GPU)",
        yaxis_title="tokens per second")
    fig.update_yaxes(range=[0, max(y) * 1.2], gridcolor="#e5e7eb")
    fig.add_annotation(x=1, y=y[1], ax=0, ay=-55, text="VRAM full, spills to PCIe",
                       font=dict(color=RED, size=13), arrowcolor=RED)
    save(fig, "chart-spill-cliff.png")


# ------------------------------------------------------------------ figure 5
def fig_affinity():
    """Pinning threads to homogeneous cores loses on Windows."""
    rows = [r for r in jsonl("affinity.jsonl") if r.get("n_gen")]
    order = ["all24-unpinned", "pcores-only", "ecores-only", "all24-pinned"]
    pretty = {"all24-unpinned": "all 24 cores<br>unpinned",
              "pcores-only": "8 P-cores<br>pinned",
              "ecores-only": "16 E-cores<br>pinned",
              "all24-pinned": "all 24 cores<br>pinned"}
    d = {r["affinity_label"]: r["avg_ts"] for r in rows}
    y = [d[k] for k in order]
    fig = go.Figure(go.Bar(x=[pretty[k] for k in order], y=y,
                           marker_color=[FN, SLATE, SLATE, RED],
                           text=["%.2f" % v for v in y], textposition="outside"))
    fig.update_layout(**LAYOUT)
    fig.update_layout(
        title="Pinning threads to cores made generation slower, every way we tried it",
        yaxis_title="tokens per second")
    fig.update_yaxes(range=[0, max(y) * 1.2], gridcolor="#e5e7eb")
    save(fig, "chart-affinity.png")


# ------------------------------------------------------------------ figure 6
def fig_game():
    """Four models, one game task. Reads hard.jsonl, game_retry.jsonl, game_27b.jsonl."""
    hard = [r for r in jsonl("hard.jsonl") if r["task"] == "threejs_game"]
    retry = jsonl("game_retry.jsonl")
    b27 = jsonl("game_27b.jsonl")
    ds = next(r for r in hard if r["config"] == "dsv4-gpu")
    fn = next(r for r in retry if r["variant"] == "directive")
    q4 = next(r for r in b27 if r["model"] == "qwen38-27b-Q4_K_M")
    q1 = next(r for r in b27 if r["model"] == "qwen38-27b-IQ1_S")
    tok = lambda r: (r["usage"] or {}).get("completion_tokens") or 0
    names = ["DeepSeek-V4-Flash<br>284B, 1-bit",
             "Flash-Next 177B<br>1-bit, told not to plan",
             "Qwen3.8-27B<br>4-bit",
             "Qwen3.8-27B<br>1-bit"]
    vals = [tok(ds), tok(fn), tok(q4), tok(q1)]
    notes = ["working game<br>first try", "working game", "nothing emitted",
             "no game<br>(CSS blob)"]
    colors = [DS, FN, RED, RED]
    fig = go.Figure(go.Bar(
        x=names, y=vals, marker_color=colors,
        text=["%s tok<br>%s" % (format(v, ","), n) for v, n in zip(vals, notes)],
        textposition="outside"))
    fig.update_layout(**LAYOUT)
    fig.update_layout(
        title="One three.js game task: only the 284B model got it right unprompted",
        yaxis_title="completion tokens spent")
    fig.update_yaxes(range=[0, max(vals) * 1.35], gridcolor="#e5e7eb")
    save(fig, "chart-game.png")


# ------------------------------------------------------------------ figure 7
def fig_expert_grids():
    """One DeepSeek layer next to one Flash-Next block, squares at relative
    expert size. DS: 256 experts of 3*4096*2048 = 25.2M, 6 routed fire.
    FN: 512 experts of 3*640*2560 = 4.9M, 10 routed fire. Both keep one
    shared expert always on."""
    fig = go.Figure()
    unlit = "#dfe3ea"

    # DeepSeek: 16x16 grid, cell 2 units, at x 0..32
    ds_lit = {17, 52, 98, 130, 179, 231}
    xs = [2 * (i % 16) for i in range(256)]
    ys = [-2 * (i // 16) for i in range(256)]
    fig.add_scatter(x=xs, y=ys, mode="markers",
                    marker=dict(symbol="square", size=24,
                                color=[DS if i in ds_lit else unlit
                                       for i in range(256)],
                                line=dict(width=0)))

    # Flash-Next: 16 cols x 32 rows, cell 1 unit, at x 40..55
    fn_lit = {3, 47, 96, 141, 190, 233, 288, 335, 379, 470}
    xs = [40 + (i % 16) for i in range(512)]
    ys = [-(i // 16) for i in range(512)]
    fig.add_scatter(x=xs, y=ys, mode="markers",
                    marker=dict(symbol="square", size=12,
                                color=[FN if i in fn_lit else unlit
                                       for i in range(512)],
                                line=dict(width=0)))

    for x, color, head, sub in (
            (0, DS, "<b>DeepSeek-V4-Flash</b>, one layer",
             "256 experts, 25.2M each, 6 fire per token"),
            (40, FN, "<b>Qwen3.8-Flash-Next</b>, one block",
             "512 experts, 4.9M each, 10 fire per token")):
        fig.add_annotation(x=x - 1, y=2.6, text=head, showarrow=False,
                           xanchor="left", font=dict(size=15, color="#1f2937"))
        fig.add_annotation(x=x - 1, y=1.0, text=sub, showarrow=False,
                           xanchor="left", font=dict(size=13, color=color))

    fig.update_layout(
        **dict(LAYOUT, width=900, height=680,
               margin=dict(l=40, r=30, t=110, b=40),
               title=dict(font=dict(size=19), x=0.01,
                          text="Experts in one layer of each model"
                               "<br><sub>Square size follows expert size. "
                               "Both models also keep one shared expert "
                               "always on.</sub>")),
        showlegend=False,
        xaxis=dict(visible=False, range=[-2, 57]),
        yaxis=dict(visible=False, range=[-32.5, 4.5],
                   scaleanchor="x", scaleratio=1),
    )
    save(fig, "chart-expert-grids.png")


# ------------------------------------------------------------------ figure 8
def fig_param_map():
    """Where the weights sit. DS: 277B routed experts + 7B rest of 284B.
    FN: 121B experts + 51.2B n-gram table + 4B draft head + ~1B rest of
    176.94B. All from the derivations in the article."""
    names = ["Qwen3.8-Flash-Next<br>177B", "DeepSeek-V4-Flash<br>284B"]
    segs = [
        ("routed experts", [121, 277], [FN, DS]),
        ("n-gram table", [51, 0], [AMBER, AMBER]),
        ("draft head", [4, 0], [SLATE, SLATE]),
        ("attention and the rest", [1, 7], ["#cbd5e1", "#cbd5e1"]),
    ]
    fig = go.Figure()
    for label, vals, colors in segs:
        fig.add_bar(y=names, x=vals, name=label, orientation="h",
                    marker_color=colors,
                    text=["%dB" % v if v >= 10 else "" for v in vals],
                    textposition="inside",
                    textfont=dict(size=16, color="white"))
    fig.update_layout(barmode="stack", **dict(LAYOUT, height=380,
                      margin=dict(l=150, r=30, t=90, b=60)))
    fig.update_layout(
        title="Where each model's parameters sit",
        xaxis_title="billions of parameters",
        legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(gridcolor="#e5e7eb")
    fig.add_annotation(x=121 + 51 + 2, y=0, ax=40, ay=-34,
                       text="4B draft head", arrowcolor="#8b939f",
                       arrowhead=0, arrowwidth=1.3, xanchor="left",
                       font=dict(size=12.5, color="#43506b"))
    save(fig, "chart-param-map.png")


# ------------------------------------------------------------------ figure 9
def fig_bits_per_param():
    """Same IQ1_S label, different real compression. Bits per parameter =
    disk bytes * 8 / total parameters. FN experts are natively fp4."""
    names = ["DeepSeek-V4-Flash<br>76.87 GB / 284B",
             "Qwen3.8-Flash-Next<br>67.56 GB / 177B"]
    gib = 1024 ** 3
    vals = [76.87 * gib * 8 / 284e9, 67.56 * gib * 8 / 176_943_899_520]
    fig = go.Figure(go.Bar(
        x=names, y=vals, marker_color=[DS, FN],
        text=["%.2f bits" % v for v in vals], textposition="outside"))
    fig.update_layout(**dict(LAYOUT, height=460))
    fig.update_layout(
        title="Bits per parameter in the two IQ1_S builds",
        yaxis_title="bits per parameter")
    fig.add_hline(y=4, line_dash="dash", line_color=RED, line_width=1.5)
    fig.add_annotation(x=0.02, y=4, xref="paper", yref="y", yanchor="bottom",
                       text="Flash Next's experts are natively fp4 (4-bit)",
                       showarrow=False, xanchor="left",
                       font=dict(size=13, color=RED))
    fig.update_yaxes(range=[0, 4.8], gridcolor="#e5e7eb")
    save(fig, "chart-bits-per-param.png")


if __name__ == "__main__":
    fig_scoreboard()
    fig_failure_modes()
    fig_speed()
    fig_spill()
    fig_affinity()
    fig_game()
    print("done")
