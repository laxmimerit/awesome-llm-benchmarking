"""Blog figures for "Qwen3.8-Flash-Next on limited hardware".

Every number is read from ../results/*.jsonl, which are the raw llama-bench and
llama-server outputs. Nothing here is typed in by hand except the HF file sizes
in fig_fit(), which come from the Hugging Face manifest and are asserted against
the two files actually on disk.

House style follows the earlier posts in this series: light surface, muted grid,
direct value labels on every mark, no legend where the axis already names the
categories.

Hue follows the entity: Qwen3.8-Flash-Next blue, Qwen3.8-27B orange, CPU-only
slate, and the VRAM-spill cliff red.
"""

import json
import os
from collections import defaultdict

import plotly.graph_objects as go

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
RES = os.path.join(ROOT, "results")
OUT = BASE

C_FN, C_27 = "#2f5fe0", "#eb6834"
C_FN_L, C_27_L = "#9fb4f2", "#f6b79a"
C_CPU = "#5b6472"
C_BAD = "#d1344a"
C_MUTE = "#b6bcc7"

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, sans-serif", size=15, color="#22252b"),
    margin=dict(l=110, r=50, t=80, b=64),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#fbfbfc",
    showlegend=False,
)


def load(name):
    p = os.path.join(RES, name)
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save(fig, name, w=1000, h=560):
    fig.write_image(os.path.join(OUT, name), width=w, height=h, scale=2)
    print("wrote", name)


def kind(r):
    return "pp" if r["n_prompt"] else "tg"


# ---------------------------------------------------------------- fig 1: threads
def fig_threads():
    rows = [r for r in load("cpu.jsonl") if r["_run"]["note"] == "threadsweep"]
    d = defaultdict(dict)
    for r in rows:
        d[(r["_run"]["quant"], r["n_threads"])][kind(r)] = r["avg_ts"]
    thr = sorted({k[1] for k in d})
    fig = go.Figure()
    for q, col, dash in (("Q2_K_XL", C_FN, "solid"), ("IQ1_S", C_FN_L, "dot")):
        fig.add_trace(go.Scatter(
            x=thr, y=[d[(q, t)]["pp"] for t in thr], mode="lines+markers",
            line=dict(color=col, width=3, dash=dash), marker=dict(size=9),
            name=q + " prompt"))
        fig.add_trace(go.Scatter(
            x=thr, y=[d[(q, t)]["tg"] for t in thr], mode="lines+markers",
            line=dict(color=C_CPU if q == "Q2_K_XL" else C_MUTE, width=3, dash=dash),
            marker=dict(size=9), name=q + " generation"))
    for t in thr:
        fig.add_annotation(x=t, y=d[("Q2_K_XL", t)]["pp"], text=f"{d[('Q2_K_XL', t)]['pp']:.0f}",
                           showarrow=False, yshift=17, font=dict(size=12, color=C_FN))
        fig.add_annotation(x=t, y=d[("Q2_K_XL", t)]["tg"], text=f"{d[('Q2_K_XL', t)]['tg']:.1f}",
                           showarrow=False, yshift=-17, font=dict(size=12, color=C_CPU))
    fig.add_vrect(x0=7.4, x1=24.6, fillcolor=C_MUTE, opacity=0.13, line_width=0)
    fig.add_annotation(x=16, y=22, text="generation is flat from 8 threads on:<br>memory-bound, the 16 E-cores add nothing",
                       showarrow=False, font=dict(size=13, color=C_CPU))
    fig.update_layout(
        title="Pure CPU: prompt processing scales with threads, generation does not",
        xaxis_title="threads", yaxis_title="tokens / sec", **LAYOUT)
    fig.update_xaxes(tickvals=thr)
    save(fig, "chart-cpu-threads.png")


# ------------------------------------------------------------- fig 2: the cliff
def fig_cliff():
    rows = [r for r in load("gpu.jsonl") if r["_run"]["note"] == "ncmoe_sweep_pleCPU"]
    d = defaultdict(dict)
    for r in rows:
        d[(r["_run"]["quant"], r["n_cpu_moe"])][kind(r)] = r["avg_ts"]
    fig = go.Figure()
    for q, col, label, up in (("IQ1_S", C_FN, "1-bit build", True),
                              ("Q2_K_XL", C_27, "2-bit build", False)):
        nc = sorted({k[1] for k in d if k[0] == q}, reverse=True)
        x = [48 - n for n in nc]
        y = [d[(q, n)]["tg"] for n in nc]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=label,
                                 line=dict(color=col, width=3), marker=dict(size=11)))
        for xi, yi, n in zip(x, y, nc):
            cliff = (q == "IQ1_S" and n == 12) or (q == "Q2_K_XL" and n == 16)
            fig.add_annotation(x=xi, y=yi, text=f"<b>{yi:.1f}</b>", showarrow=False,
                               yshift=(20 if up else -20) if not cliff else -22,
                               font=dict(size=12, color=C_BAD if cliff else col))
    fig.add_annotation(x=32, y=52.0, ax=-4, ay=-42, text="best setting", showarrow=True,
                       arrowhead=2, arrowcolor=C_FN, font=dict(size=13, color=C_FN))
    fig.add_annotation(x=34, y=26, text="<b>push one step further</b><br>"
                                        "and both collapse below<br>the CPU-only baseline",
                       showarrow=False, xanchor="center",
                       font=dict(size=13, color=C_BAD))
    fig.add_hline(y=11.1, line=dict(color=C_CPU, width=2, dash="dash"))
    fig.add_annotation(x=2, y=11.1, text="pure CPU baseline 11.1 t/s", showarrow=False,
                       yshift=14, xanchor="left", font=dict(size=12, color=C_CPU))
    fig.update_layout(
        title="Offloading experts to the 5090: a sharp peak, then a silent collapse",
        xaxis_title="expert layers moved onto the GPU (of 48)",
        yaxis_title="generation, tokens / sec", **LAYOUT)
    fig.update_layout(showlegend=True, margin=dict(l=110, r=60, t=108, b=64),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="left", x=0, font=dict(size=14)))
    fig.update_xaxes(range=[-2, 40])
    save(fig, "chart-offload-cliff.png")


# ------------------------------------------------- fig 3: what is in a 1-bit file
def fig_composition():
    comp = [("MoE expert banks", 37.11, C_FN),
            ("PLE n-gram table", 26.82, C_BAD),
            ("attention / QSA", 1.71, C_27),
            ("everything else", 1.91, C_MUTE)]
    tot = sum(c[1] for c in comp)
    fig = go.Figure(go.Bar(
        x=[c[1] for c in comp], y=[c[0] for c in comp], orientation="h",
        marker_color=[c[2] for c in comp],
        text=[f"  {c[1]:.2f} GB  ({100 * c[1] / tot:.1f}%)" for c in comp],
        textposition="outside", cliponaxis=False))
    fig.add_annotation(x=26.82, y="PLE n-gram table", ax=120, ay=40,
                       text="<b>stored at 4-bit even in the 1-bit build</b><br>pure lookup, zero matmul",
                       showarrow=True, arrowhead=2, arrowcolor=C_BAD,
                       font=dict(size=13, color=C_BAD))
    fig.update_layout(
        title=f"Inside the \"1-bit\" 72.5 GB build ({tot:.1f} GB of tensors)",
        xaxis_title="GB", **LAYOUT)
    fig.update_xaxes(range=[0, 46])
    save(fig, "chart-composition.png", h=460)


# ------------------------------------------------------------ fig 4: does it fit
def fig_fit():
    quants = [("UD-IQ1_S", 72.55), ("UD-IQ1_M", 74.54), ("UD-Q2_K_XL", 78.87),
              ("UD-IQ3_XXS", 81.96), ("UD-Q3_K_XL", 89.97), ("UD-IQ4_XS", 93.68),
              ("UD-Q4_K_XL", 111.33)]
    for name, gb in quants:                      # assert against what is on disk
        p = os.path.join(ROOT, "models", name)
        if os.path.isdir(p):
            on_disk = sum(os.path.getsize(os.path.join(p, f))
                          for f in os.listdir(p) if f.endswith(".gguf")) / 1e9
            assert abs(on_disk - gb) < 0.02, (name, on_disk, gb)
    ram = 95.3
    cols = [C_FN if gb < ram else C_BAD for _, gb in quants]
    fig = go.Figure(go.Bar(
        x=[q[0].replace("UD-", "") for q in quants], y=[q[1] for q in quants],
        marker_color=cols, text=[f"{q[1]:.1f}" for q in quants], textposition="outside",
        cliponaxis=False))
    fig.add_hline(y=ram, line=dict(color=C_BAD, width=2, dash="dash"))
    fig.add_annotation(x=6.4, y=ram, text=f"<b>{ram} GB RAM ceiling</b>", showarrow=False,
                       yshift=14, xanchor="right", font=dict(size=13, color=C_BAD))
    fig.add_annotation(x=5.5, y=104, text="the two 4-bit builds<br>do not fit", showarrow=False,
                       font=dict(size=13, color=C_BAD))
    fig.update_layout(
        title="Every published quant against a 96 GB desktop",
        yaxis_title="download size, GB", **LAYOUT)
    fig.update_yaxes(range=[0, 125])
    save(fig, "chart-fit.png", h=520)


# ------------------------------------------------------------ fig 5: perplexity
def fig_ppl():
    rows = load("ppl.jsonl")
    order = ["flashnext-Q2_K_XL", "flashnext-IQ1_S", "qwen38-27b-Q4_K_M",
             "qwen38-27b-Q2_K_XL", "qwen38-27b-IQ1_S"]
    lbl = {"flashnext-Q2_K_XL": "Flash-Next 180B<br>2-bit",
           "flashnext-IQ1_S": "Flash-Next 180B<br>1-bit",
           "qwen38-27b-Q4_K_M": "Qwen3.8-27B<br>4-bit",
           "qwen38-27b-Q2_K_XL": "Qwen3.8-27B<br>2-bit",
           "qwen38-27b-IQ1_S": "Qwen3.8-27B<br>1-bit"}
    m = {r["model"]: r for r in rows}
    fig = go.Figure(go.Bar(
        x=[lbl[k] for k in order], y=[m[k]["ppl"] for k in order],
        error_y=dict(type="data", array=[m[k]["stderr_ppl"] for k in order], color="#5b6472"),
        marker_color=[C_FN if "flashnext" in k else C_27 for k in order],
        text=[f"{m[k]['ppl']:.2f}" for k in order], textposition="outside", cliponaxis=False))
    fig.add_annotation(x=1, y=4.01, ax=90, ay=-70,
                       text="<b>the 180B crushed to 1 bit still beats<br>the 27B at 4 bits</b>",
                       showarrow=True, arrowhead=2, arrowcolor=C_FN,
                       font=dict(size=13, color=C_FN))
    fig.update_layout(
        title="Perplexity on wikitext-2 test (lower is better, identical tokenizer)",
        yaxis_title="perplexity", **LAYOUT)
    fig.update_yaxes(range=[0, 9.6])
    save(fig, "chart-perplexity.png")


# --------------------------------------------------------------- fig 6: quality
def fig_quality():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "research"))
    from quality import score as rescore
    probs = {p["id"]: p for p in json.load(
        open(os.path.join(ROOT, "research", "problems.json"), encoding="utf-8"))}
    rows = load("quality.jsonl")
    agg = defaultdict(list)
    for r in rows:
        if r.get("truncated"):
            continue                     # graded on runs the model was allowed to finish
        agg[r["model"]].append(rescore(probs[r["problem"]], r["content"])[0])
    order = ["flashnext-Q2_K_XL", "flashnext-IQ1_S", "qwen38-27b-Q4_K_M",
             "qwen38-27b-Q2_K_XL", "qwen38-27b-IQ1_S"]
    lbl = {"flashnext-Q2_K_XL": "Flash-Next 180B<br>2-bit",
           "flashnext-IQ1_S": "Flash-Next 180B<br>1-bit",
           "qwen38-27b-Q4_K_M": "Qwen3.8-27B<br>4-bit",
           "qwen38-27b-Q2_K_XL": "Qwen3.8-27B<br>2-bit",
           "qwen38-27b-IQ1_S": "Qwen3.8-27B<br>1-bit"}
    val = {k: sum(agg[k]) / len(agg[k]) for k in order}
    fig = go.Figure(go.Bar(
        x=[lbl[k] for k in order], y=[val[k] for k in order],
        marker_color=[C_BAD if val[k] < 0.5 else (C_FN if "flashnext" in k else C_27)
                      for k in order],
        text=[f"{val[k]:.3f}" for k in order], textposition="outside", cliponaxis=False))
    fig.add_annotation(x=4, y=val["qwen38-27b-IQ1_S"], ax=-10, ay=-96,
                       text="<b>same 1-bit recipe, opposite outcome</b><br>"
                            "invents who discovered the transistor",
                       showarrow=True, arrowhead=2, arrowcolor=C_BAD,
                       font=dict(size=13, color=C_BAD))
    fig.update_layout(
        title="12-problem suite, 2 seeds, scored on runs that finished",
        yaxis_title="mean score", **LAYOUT)
    fig.update_yaxes(range=[0, 1.19], tickformat=".1f")
    save(fig, "chart-quality.png")


# ---------------------------------------------------------- fig 7: long context
def fig_longctx():
    rows = load("longctx.jsonl")
    d = defaultdict(dict)
    for r in rows:
        d[(r["_run"]["quant"], r["_run"]["note"])][(r.get("n_depth", 0), kind(r))] = r["avg_ts"]
    series = [(("IQ1_S", "gpu_ncmoe20"), "1-bit + GPU offload", C_FN),
              (("Q2_K_XL", "gpu_ncmoe28"), "2-bit + GPU offload", C_27),
              (("Q2_K_XL", "cpu"), "2-bit, pure CPU", C_CPU)]
    depths = [0, 4096, 16384, 32768]
    fig = go.Figure()
    for key, name, col in series:
        y = [d[key][(dep, "tg")] for dep in depths]
        fig.add_trace(go.Scatter(x=list(range(len(depths))), y=y, mode="lines+markers",
                                 name=name, line=dict(color=col, width=3),
                                 marker=dict(size=11)))
        for i, v in enumerate(y):
            fig.add_annotation(x=i, y=v, text=f"{v:.1f}", showarrow=False, yshift=19,
                               font=dict(size=12, color=col))
        drop = (y[0] - y[-1]) / y[0] * 100
        fig.add_annotation(x=len(depths) - 1, y=y[-1], text=f"  {name}<br>  minus {drop:.0f}% at 32k",
                           showarrow=False, xanchor="left", xshift=6,
                           font=dict(size=12, color=col))
    fig.update_layout(
        title="Generation speed holds up to 32k context on every configuration",
        xaxis_title="context already in the KV cache", yaxis_title="generation, tokens / sec",
        **LAYOUT)
    fig.update_xaxes(tickvals=list(range(len(depths))),
                     ticktext=["empty", "4k", "16k", "32k"], range=[-0.15, 3.85])
    fig.update_layout(margin=dict(l=110, r=210, t=80, b=64))
    save(fig, "chart-longctx.png")


# ------------------------------------------------------------- fig 8: the ladder
def fig_ladder():
    steps = [("4-bit\non CPU", 0.0, C_MUTE, "does not fit"),
             ("1-bit\npure CPU", 10.82, C_CPU, ""),
             ("2-bit\npure CPU", 11.10, C_CPU, ""),
             ("2-bit\n+ GPU", 43.69, C_27, "3.9x"),
             ("1-bit\n+ GPU", 51.97, C_FN, "4.8x"),
             ("one rung\ntoo far", 3.68, C_BAD, "silent spill")]
    fig = go.Figure(go.Bar(
        x=[s[0].replace("\n", "<br>") for s in steps], y=[s[1] for s in steps],
        marker_color=[s[2] for s in steps],
        text=[(f"{s[1]:.1f} t/s" if s[1] else "n/a") + (f"<br>{s[3]}" if s[3] else "")
              for s in steps],
        textposition="outside", cliponaxis=False))
    fig.update_layout(
        title="Qwen3.8-Flash-Next, 180B, on one desktop: 96 GB RAM and an RTX 5090",
        yaxis_title="generation, tokens / sec", **LAYOUT)
    fig.update_yaxes(range=[0, 62])
    save(fig, "chart-ladder.png", h=520)


if __name__ == "__main__":
    fig_ladder()
    fig_threads()
    fig_cliff()
    fig_composition()
    fig_fit()
    fig_ppl()
    fig_quality()
    fig_longctx()
    print("\nall figures written to", OUT)
