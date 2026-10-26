"""Blog figures for the Granite 4.2 post, Plotly + kaleido.

House style follows the earlier posts in this series: light surface, muted
grid, direct value labels on every mark, no legend where the categories are
already named on the axis.

Hue follows the entity across every figure:
  Granite red, Gemma violet, Qwen orange, Ornith green.
"""

import json
import os
import statistics as st

import plotly.graph_objects as go

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(BASE), "results")
OUT = BASE

C_GR, C_GE, C_QW, C_OR = "#d1495b", "#7c5cd6", "#eb6834", "#1baf7a"

SHORT = {
    "granite4.2-8b": "Granite 4.2 8B",
    "gemma4-12b": "Gemma 4 12B",
    "ornith-9b": "Ornith 1.5 9B",
    "granite4.2-30b": "Granite 4.2 30B",
    "gemma4-31b": "Gemma 4 31B",
    "qwen3.8-27b": "Qwen 3.8 27B",
}
COLOR = {
    "granite4.2-8b": C_GR, "granite4.2-30b": C_GR,
    "gemma4-12b": C_GE, "gemma4-31b": C_GE,
    "qwen3.8-27b": C_QW, "ornith-9b": C_OR,
}
ORDER = list(SHORT)

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, sans-serif", size=15, color="#22252b"),
    margin=dict(l=90, r=40, t=70, b=60),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#fbfbfc",
    showlegend=False,
)


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def loadj(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.write_image(path, width=1000, height=560, scale=2)
    print("wrote", path)


def bar(keys, vals, title, xtitle, texts=None, suffix=""):
    fig = go.Figure(go.Bar(
        x=[SHORT[k] for k in keys],
        y=vals,
        marker_color=[COLOR[k] for k in keys],
        text=texts or [f"{v:g}{suffix}" for v in vals],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(title=title, yaxis_title=xtitle, **LAYOUT)
    fig.update_yaxes(gridcolor="#e8e9ee", zeroline=False,
                     range=[0, max(vals) * 1.18])
    return fig


# --------------------------------------------------------------------------

def fig_kv():
    arch = loadj("arch.json")
    keys = [k for k in ORDER if k in arch]
    vals = [arch[k]["kv_kib_per_token"] for k in keys]
    fig = bar(keys, vals,
              "KV cache cost per token of context (f16, measured)",
              "KB per token", suffix=" KB")
    save(fig, "chart-kv-per-token.png")


def fig_decode():
    rows = [r for r in load("speed_fwd.jsonl") + load("speed_rev.jsonl")
            if r.get("content") and r.get("decode_tps_server")]
    mtp = loadj("mtp_check.json")
    keys = [k for k in ORDER if any(r["model"] == k for r in rows)]
    base = [st.median([r["decode_tps_server"] for r in rows if r["model"] == k])
            for k in keys]

    fig = go.Figure()
    fig.add_bar(x=[SHORT[k] for k in keys], y=base,
                marker_color=[COLOR[k] for k in keys],
                text=[f"{v:.0f}" for v in base], textposition="outside",
                cliponaxis=False, name="MTP off")

    # Only two models in this line-up carry a speculative head; showing them at
    # MTP-off rates alone would understate what they actually do on this rig.
    best = {}
    for k, entry in (mtp or {}).items():
        runs = ((entry.get("on") or {}).get("runs") or {})
        vals = [r["decode_tps"] for r in runs.values() if r.get("decode_tps")]
        if vals:
            best[k] = (min(vals), max(vals))
    if best:
        xs, lo, hi = [], [], []
        for i, k in enumerate(keys):
            if k in best:
                xs.append(SHORT[k])
                lo.append(best[k][0])
                hi.append(best[k][1])
        fig.add_bar(x=xs, y=[h for h in hi],
                    marker_color="rgba(0,0,0,0)",
                    marker_line=dict(color="#22252b", width=2),
                    text=[f"MTP {l:.0f} to {h:.0f}" for l, h in zip(lo, hi)],
                    textposition="outside", cliponaxis=False, name="MTP on")
    fig.update_layout(
        title="Decode speed, single stream (MTP off, with MTP range where a head exists)",
        yaxis_title="tokens per second", barmode="overlay", **LAYOUT)
    fig.update_yaxes(gridcolor="#e8e9ee", zeroline=False)
    save(fig, "chart-decode.png")


def fig_ctx():
    rows = [r for r in load("usable_ctx.jsonl") if r.get("kind") == "usable_ctx"]
    keys = [k for k in ORDER if any(r["model"] == k for r in rows)]
    got = {r["model"]: r for r in rows}
    fig = go.Figure()
    fig.add_bar(x=[SHORT[k] for k in keys],
                y=[got[k]["nominal_ctx"] / 1024 for k in keys],
                marker_color="#dcdde4",
                text=[f"{got[k]['nominal_ctx'] // 1024}k advertised" for k in keys],
                textposition="outside", cliponaxis=False)
    fig.add_bar(x=[SHORT[k] for k in keys],
                y=[got[k]["usable_max_ctx"] / 1024 for k in keys],
                marker_color=[COLOR[k] for k in keys],
                text=[f"{got[k]['usable_max_ctx'] / 1024:.0f}k" for k in keys],
                textposition="inside", insidetextanchor="middle",
                cliponaxis=False)
    fig.update_layout(
        title="Usable context on one 32 GB card vs the number on the box",
        yaxis_title="thousands of tokens", barmode="overlay", **LAYOUT)
    fig.update_yaxes(gridcolor="#e8e9ee", zeroline=False)
    save(fig, "chart-usable-context.png")


def fig_budget():
    rows = [r for r in (load("quality_hard_rescored.jsonl")
                        or load("quality_hard.jsonl"))
            if r.get("dept") in ("math", "trap")]
    budgets = [2048, 4096, 8192, 16384, 32768]
    keys = [k for k in ORDER if any(r["model"] == k for r in rows)]
    fig = go.Figure()
    for k in keys:
        rs = [r for r in rows if r["model"] == k]
        ys = [sum(1 for r in rs if r.get("correct")
                  and (r.get("completion_tokens") or 10 ** 9) <= b)
              for b in budgets]
        fig.add_scatter(x=[b / 1024 for b in budgets], y=ys, mode="lines+markers",
                        name=SHORT[k], line=dict(color=COLOR[k], width=3,
                                                 dash="dot" if "8b" in k or "12b" in k or "9b" in k else "solid"),
                        marker=dict(size=9))
    lay = dict(LAYOUT)
    lay["showlegend"] = True
    fig.update_layout(
        title="Problems solved within an output-token budget (20 items)",
        xaxis_title="output tokens allowed (thousands)",
        yaxis_title="problems solved", **lay)
    fig.update_xaxes(gridcolor="#e8e9ee", type="log",
                     tickvals=[2, 4, 8, 16, 32],
                     ticktext=["2k", "4k", "8k", "16k", "32k"])
    fig.update_yaxes(gridcolor="#e8e9ee", zeroline=False, range=[0, 21])
    save(fig, "chart-token-budget.png")


def fig_layers():
    arch = loadj("arch.json")
    keys = [k for k in ORDER if k in arch]
    fig = go.Figure()
    fig.add_bar(x=[SHORT[k] for k in keys],
                y=[arch[k]["n_full_attn_blocks"] for k in keys],
                marker_color=[COLOR[k] for k in keys],
                text=[f"{arch[k]['n_full_attn_blocks']} full attn" for k in keys],
                textposition="inside", insidetextanchor="middle",
                cliponaxis=False, name="full attention")
    fig.add_bar(x=[SHORT[k] for k in keys],
                y=[arch[k]["n_blocks"] - arch[k]["n_full_attn_blocks"]
                   for k in keys],
                marker_color="#c9ccd6",
                text=[(f"{arch[k]['n_ssm_blocks']} SSM"
                       if arch[k]["n_ssm_blocks"] else
                       (f"{arch[k]['n_blocks'] - arch[k]['n_full_attn_blocks']} sliding"
                        if arch[k]["n_blocks"] > arch[k]["n_full_attn_blocks"] else ""))
                      for k in keys],
                textposition="inside", insidetextanchor="middle",
                cliponaxis=False, name="cheap layers")
    fig.update_layout(
        title="Where each model spends its layers",
        yaxis_title="blocks", barmode="stack", **LAYOUT)
    fig.update_yaxes(gridcolor="#e8e9ee", zeroline=False)
    save(fig, "chart-layer-mix.png")


if __name__ == "__main__":
    fig_layers()
    fig_kv()
    fig_decode()
    fig_ctx()
    fig_budget()
