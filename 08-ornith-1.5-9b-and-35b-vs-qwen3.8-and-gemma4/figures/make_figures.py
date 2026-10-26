"""Blog figures for the Ornith 1.5 post, Plotly + kaleido.

House style matches the earlier posts in this series (qwen-teardown/make_figures.py):
light surface, muted grid, direct value labels on every mark, no legend where the
categories are already named on the axis.

Hue follows the entity across every figure:
  Ornith 9B aqua, Ornith 35B-A3B blue, Qwen 3.8 orange, Gemma 4 violet, Muse gold.
"""

import json
import os

import plotly.graph_objects as go

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(BASE), "results")
OUT = BASE

C_O9, C_O35 = "#1baf7a", "#2a78d6"
C_QW, C_GE, C_MU = "#eb6834", "#7c5cd6", "#c9971f"

SHORT = {
    "ornith-35b-a3b": "Ornith 35B-A3B",
    "gemma4-26b": "Gemma 4-26B-A4B",
    "ornith-9b": "Ornith 9B",
    "qwen3.8-27b": "Qwen 3.8 27B",
}
COLOR = {
    "ornith-9b": C_O9, "ornith-35b-a3b": C_O35, "qwen3.8-27b": C_QW,
    "gemma4-26b": C_GE,
}
ORDER = list(SHORT)

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def load(name):
    p = os.path.join(RESULTS, name)
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def axis(**kw):
    a = dict(gridcolor=GRID, griddash="solid", linecolor=BASELINE,
             tickfont=dict(color=MUTED, size=12),
             title_font=dict(color=INK2, size=13), zeroline=False)
    a.update(kw)
    return a


def layout(title, subtitle, height=460, **kw):
    lay = dict(
        title=dict(
            text=(f"<b>{title}</b><br>"
                  f"<span style='font-size:13px;color:{INK2}'>{subtitle}</span>"),
            font=dict(family=FONT, size=19, color=INK),
            x=0.02, xanchor="left",
        ),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK2),
        width=900, height=height,
        margin=dict(l=150, r=70, t=100, b=60),
        showlegend=False, barcornerradius=4,
    )
    lay.update(kw)
    return lay


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.write_image(path, scale=2)
    print("wrote", path)


# --------------------------------------------------------------------------
# 1. decode throughput
# --------------------------------------------------------------------------
def fig_decode():
    rows = [r for r in load("speed.jsonl") if r.get("content")]
    means = {}
    for m in ORDER:
        v = [r["decode_tps_server"] for r in rows if r["model"] == m]
        means[m] = sum(v) / len(v)
    order = sorted(ORDER, key=lambda m: means[m])
    fig = go.Figure(go.Bar(
        x=[means[m] for m in order],
        y=[SHORT[m] for m in order],
        orientation="h",
        marker_color=[COLOR[m] for m in order],
        text=[f"{means[m]:.1f}" for m in order],
        textposition="outside",
        textfont=dict(size=13, color=INK),
    ))
    fig.update_layout(**layout(
        "The mixture-of-experts model is the fastest of the five",
        "Decode speed, tokens per second, 512 tokens, speculative decoding off",
        height=420,
        xaxis=axis(title="tokens per second", range=[0, 305]),
        yaxis=axis(showgrid=False),
    ))
    save(fig, "chart-decode.png")


# --------------------------------------------------------------------------
# 2. active parameters against speed
# --------------------------------------------------------------------------
def fig_active():
    active = {"ornith-9b": 9.2, "ornith-35b-a3b": 3.0, "qwen3.8-27b": 27.3,
              "gemma4-26b": 4.0}
    rows = [r for r in load("speed.jsonl") if r.get("content")]
    means = {m: sum(r["decode_tps_server"] for r in rows if r["model"] == m)
                / len([r for r in rows if r["model"] == m]) for m in ORDER}
    # The three dense 27-31B models land almost on top of each other, so their
    # labels get leaders into open space instead of sitting on the markers.
    OFFSET = {}
    fig = go.Figure()
    for m in ORDER:
        fig.add_trace(go.Scatter(
            x=[active[m]], y=[means[m]], mode="markers",
            marker=dict(size=20, color=COLOR[m],
                        line=dict(width=2, color=SURFACE)),
        ))
        if m in OFFSET:
            ax, ay = OFFSET[m]
            fig.add_annotation(
                x=active[m], y=means[m], ax=ax, ay=ay, text=SHORT[m],
                arrowhead=0, arrowsize=1, arrowwidth=1.2, arrowcolor=COLOR[m],
                font=dict(size=12, color=COLOR[m]),
            )
        else:
            fig.add_annotation(
                x=active[m], y=means[m], text=SHORT[m],
                showarrow=False, xanchor="left", yanchor="middle", xshift=16,
                font=dict(size=12, color=COLOR[m]),
            )
    fig.update_layout(**layout(
        "Speed tracks active parameters, not model size",
        "Ornith 35B-A3B touches about 3B parameters per token and outruns every dense model here",
        height=470,
        xaxis=axis(title="active parameters per token (billions)", range=[0, 42]),
        yaxis=axis(title="tokens per second", range=[0, 310]),
        margin=dict(l=90, r=70, t=100, b=60),
    ))
    save(fig, "chart-active-params.png")


# --------------------------------------------------------------------------
# 3. the context cliff
# --------------------------------------------------------------------------
def fig_cliff():
    rows = [r for r in load("usable_ctx.jsonl") if r.get("ok") and r.get("ctx")]
    fig = go.Figure()
    for m in ORDER:
        pts = sorted((r["ctx"], r["decode_tps"]) for r in rows if r["model"] == m)
        fig.add_trace(go.Scatter(
            x=[p[0] / 1024 for p in pts], y=[p[1] for p in pts],
            mode="lines+markers", name=SHORT[m],
            line=dict(color=COLOR[m], width=2.5),
            marker=dict(size=7, color=COLOR[m]),
        ))
        yshift = 0
        last = pts[-1]
        fig.add_annotation(
            x=last[0] / 1024, y=last[1], text=f"  {SHORT[m]}",
            showarrow=False, xanchor="left", yanchor="middle", yshift=yshift,
            font=dict(size=11.5, color=COLOR[m]),
        )
    # Text only, no arrows. The band between 90 and 185 tok/s is empty across
    # the full width, and the cliff shape needs no pointer to be read; long
    # leader lines just cut across every series.
    fig.add_annotation(
        x=6, y=150, text="Qwen 3.8 holds 78 tok/s to 245k, then falls to 7.3",
        showarrow=False, xanchor="left", yanchor="middle",
        font=dict(size=12, color=C_QW),
    )
    fig.add_annotation(
        x=6, y=120, text="Both mixtures and the Ornith 9B run flat to the full 262k window",
        showarrow=False, xanchor="left", yanchor="middle",
        font=dict(size=12, color=INK2),
    )
    fig.update_layout(**layout(
        "Context ceilings are cliffs, not slopes",
        "Decode speed against allocated context. Past the edge the KV cache spills to system memory",
        height=520,
        xaxis=axis(title="allocated context (thousands of tokens)",
                   range=[0, 300], dtick=64),
        yaxis=axis(title="tokens per second", range=[0, 300]),
        margin=dict(l=90, r=150, t=100, b=60),
    ))
    save(fig, "chart-context-cliff.png")


# --------------------------------------------------------------------------
# 4. KV cache cost per token
# --------------------------------------------------------------------------
def fig_kv():
    rows = load("vram.jsonl")
    kv = {r["model"]: r["kv_bytes_per_token"] / 1024
          for r in rows if r.get("kind") == "kv_per_token"}
    order = sorted(ORDER, key=lambda m: kv[m])
    fig = go.Figure(go.Bar(
        x=[kv[m] for m in order], y=[SHORT[m] for m in order],
        orientation="h", marker_color=[COLOR[m] for m in order],
        text=[f"{kv[m]:.0f} KB   ({kv[m]*262144/1048576:.1f} GB at 262k)"
              for m in order],
        textposition="outside", textfont=dict(size=12, color=INK),
    ))
    fig.update_layout(**layout(
        "KV cache cost separates these architectures more than size does",
        "Kilobytes of key-value cache per token, and what a full 262k window would cost",
        height=420,
        xaxis=axis(title="KB per token", range=[0, 125]),
        yaxis=axis(showgrid=False),
    ))
    save(fig, "chart-kv-cache.png")


# --------------------------------------------------------------------------
# 5. prefill by depth
# --------------------------------------------------------------------------
def fig_prefill():
    rows = [r for r in load("prefill.jsonl") if r.get("depth_target")]
    deep = [r for r in load("longctx.jsonl") if r.get("target_tokens") and r.get("prefill_tps")]
    # Evenly spaced index positions rather than a log axis. The depths are
    # already a doubling ladder, so even spacing reads the same, and it avoids
    # log-axis annotations, whose coordinates are log10 and silently wreck the
    # range if passed in data units.
    depths = [512, 2048, 8192, 32768, 65536, 131072]
    ticks = ["512", "2k", "8k", "32k", "64k", "128k"]
    fig = go.Figure()
    for m in ORDER:
        vals = {}
        for r in rows:
            if r["model"] == m:
                vals[r["depth_target"]] = r["prompt_tps_server"]
        for d in (65536, 131072):
            v = [r["prefill_tps"] for r in deep
                 if r["model"] == m and r["target_tokens"] == d]
            if v:
                vals[d] = sum(v) / len(v)
        xs = [i for i, d in enumerate(depths) if d in vals]
        ys = [vals[depths[i]] for i in xs]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=COLOR[m], width=2.5),
            marker=dict(size=6, color=COLOR[m]),
        ))
        fig.add_annotation(
            x=xs[-1], y=ys[-1], text=f"  {SHORT[m]}",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=11.5, color=COLOR[m]),
        )
    fig.update_layout(**layout(
        "Prompt reading speed against prompt length",
        "Tokens per second while filling context, measured from 512 tokens to 128k",
        height=480,
        xaxis=axis(title="prompt length", tickvals=list(range(len(depths))),
                   ticktext=ticks, range=[-0.15, 6.6]),
        yaxis=axis(title="tokens per second", range=[0, 11500]),
        margin=dict(l=90, r=90, t=100, b=60),
    ))
    save(fig, "chart-prefill.png")


# --------------------------------------------------------------------------
# 6. MTP speedup by content type
# --------------------------------------------------------------------------
def fig_mtp():
    with open(os.path.join(RESULTS, "mtp_check.json"), encoding="utf-8") as f:
        cur = json.load(f)
    with open(os.path.join(os.path.dirname(BASE), "results_prep", "mtp_check.json"),
              encoding="utf-8") as f:
        prep = json.load(f)
    data = {**prep, **cur}
    contents = ["counting", "code", "json", "essay", "fiction"]
    models = ["qwen3.8-27b", "ornith-9b", "ornith-35b-a3b"]
    fig = go.Figure()
    for m in models:
        sp = data[m]["speedup"]
        fig.add_trace(go.Bar(
            name=SHORT[m], x=contents, y=[sp[c] for c in contents],
            marker_color=COLOR[m],
            text=[f"{sp[c]:.2f}x" for c in contents],
            textposition="outside", textfont=dict(size=11, color=INK2),
        ))
    fig.add_hline(y=1.0, line=dict(color=BASELINE, width=1.5, dash="dash"))
    fig.add_annotation(x=4.42, y=1.0, text="no gain", showarrow=False,
                       yanchor="bottom", xanchor="left",
                       font=dict(size=11, color=MUTED))
    fig.update_layout(**layout(
        "Multi-token prediction pays on structured text and loses on prose",
        "Speedup against the same model with speculative decoding off. Below the dashed line it is a loss",
        height=490, showlegend=True, barmode="group",
        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom",
                    font=dict(size=12)),
        xaxis=axis(showgrid=False),
        yaxis=axis(title="speed relative to MTP off", range=[0, 2.75]),
        margin=dict(l=90, r=70, t=125, b=60),
    ))
    save(fig, "chart-mtp-speedup.png")


# --------------------------------------------------------------------------
# 7. concurrency
# --------------------------------------------------------------------------
def fig_concurrency():
    rows = [r for r in load("concurrency.jsonl") if r.get("ok")]
    levels = [1, 4, 8]
    fig = go.Figure()
    for m in ORDER:
        d = {r["parallel"]: r["aggregate_tps"] for r in rows if r["model"] == m}
        fig.add_trace(go.Scatter(
            x=levels, y=[d.get(l) for l in levels], mode="lines+markers",
            line=dict(color=COLOR[m], width=2.5),
            marker=dict(size=8, color=COLOR[m]),
        ))
        fig.add_annotation(
            x=8, y=d.get(8), text=f"  {SHORT[m]}  {d.get(8):.0f}",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=11.5, color=COLOR[m]),
        )
    fig.update_layout(**layout(
        "Throughput under parallel requests",
        "Total tokens per second across all slots. Gemma 4-31B goes backwards past four",
        height=470,
        xaxis=axis(title="parallel requests", tickvals=levels, range=[0.6, 11.5]),
        yaxis=axis(title="aggregate tokens per second", range=[0, 730]),
        margin=dict(l=90, r=200, t=100, b=60),
    ))
    save(fig, "chart-concurrency.png")


# --------------------------------------------------------------------------
# 8. time to a correct hard answer
# --------------------------------------------------------------------------
def fig_answer():
    import statistics as st
    rows = [r for r in load("quality.jsonl") if r.get("tier") == "hard"]
    secs, toks = {}, {}
    for m in ORDER:
        mine = [r for r in rows if r["model"] == m]
        secs[m] = st.median([r["wall_s"] for r in mine])
        toks[m] = st.median([r["completion_tokens"] for r in mine])
    order = sorted(ORDER, key=lambda m: secs[m])
    fig = go.Figure(go.Bar(
        x=[secs[m] for m in order], y=[SHORT[m] for m in order],
        orientation="h", marker_color=[COLOR[m] for m in order],
        text=[f"{secs[m]:.1f} s   ({toks[m]:,.0f} tokens)" for m in order],
        textposition="outside", textfont=dict(size=12, color=INK),
    ))
    fig.update_layout(**layout(
        "Every model got the hard problems right, at very different cost",
        "Median seconds to a correct answer, with the median tokens spent getting there",
        height=420,
        xaxis=axis(title="seconds", range=[0, 78]),
        yaxis=axis(showgrid=False),
    ))
    save(fig, "chart-answer-time.png")


# --------------------------------------------------------------------------
# 9. long context answer latency
# --------------------------------------------------------------------------
def fig_longctx():
    import statistics as st
    rows = [r for r in load("longctx.jsonl") if r.get("target_tokens") == 131072]
    secs = {m: st.mean([r["wall_s"] for r in rows if r["model"] == m])
            for m in ORDER}
    order = sorted(ORDER, key=lambda m: secs[m])
    fig = go.Figure(go.Bar(
        x=[secs[m] for m in order], y=[SHORT[m] for m in order],
        orientation="h", marker_color=[COLOR[m] for m in order],
        text=[f"{secs[m]:.1f} s" for m in order],
        textposition="outside", textfont=dict(size=13, color=INK),
    ))
    fig.update_layout(**layout(
        "Answering a question about a 128k token document",
        "Seconds from prompt to answer. All five models found the fact every time",
        height=420,
        xaxis=axis(title="seconds", range=[0, 235]),
        yaxis=axis(showgrid=False),
    ))
    save(fig, "chart-longctx-latency.png")


if __name__ == "__main__":
    fig_decode()
    fig_active()
    fig_cliff()
    fig_kv()
    fig_prefill()
    fig_mtp()
    fig_concurrency()
    fig_answer()
    fig_longctx()
