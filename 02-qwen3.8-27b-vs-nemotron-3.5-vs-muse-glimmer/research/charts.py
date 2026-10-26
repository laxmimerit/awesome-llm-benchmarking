"""Comparison figures for Qwen 3.8 27B vs Nemotron 3.5 Lightning vs Muse Glimmer.

Palette: validated categorical slots 1-3 from the reference instance. Both the
light and dark all-pairs runs PASS (worst CVD dE 9.2 light / 9.4 dark, worst
normal-vision dE 24.0 light / 20.9 dark). The light run WARNs that aqua sits at
2.74:1 on the light surface, so the relief rule applies: every mark that carries
a value is directly labelled, and results-summary.md is the table-view twin.
Hue follows the entity and matches the previous post in this series, so a
returning reader keeps "Nemotron is blue".

Pass/fail state uses the reserved status colours as a light tint plus a
saturated glyph, never a saturated block, and distinguishes three states:
solved, wrong answer, and out of generation budget.
"""

import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE = "D:/Claude AI/AI Model Performance/qwen3.8-vs-nemotron-vs-glimmer"

COLORS = {
    "nemotron": "#2a78d6",   # slot 1 blue
    "glimmer": "#eb6834",    # slot 2 orange
    "qwen38": "#1baf7a",     # slot 3 aqua
}
NAMES = {
    "nemotron": "Nemotron 3.5 Lightning",
    "glimmer": "Muse Glimmer",
    "qwen38": "Qwen 3.8 27B",
}
SHORT = {
    "nemotron": "Nemotron 3.5",
    "glimmer": "Muse Glimmer",
    "qwen38": "Qwen 3.8",
}
PARAMS = {"nemotron": "32.9B MoE", "glimmer": "27.9B", "qwen38": "27.3B"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

CHECK = "\u2713"
CROSS = "\u2717"
HOURGLASS = "\u29d7"
DOT = "\u00b7"

with open(BASE + "/research/results-hard2.json", encoding="utf-8") as f:
    R = json.load(f)

MODELS = [k for k in ["nemotron", "glimmer", "qwen38"] if k in R]
TASKS = [t["task"] for t in R[MODELS[0]]["tasks"]]
LABEL = {t["task"]: t["label"] for t in R[MODELS[0]]["tasks"]}
CAT = {t["task"]: t["cat"] for t in R[MODELS[0]]["tasks"]}
N_TASKS = len(TASKS)


def tasks_of(k):
    return {t["task"]: t for t in R[k]["tasks"]}


def score(k):
    return sum(1 for t in R[k]["tasks"] if t["correct"])


def avg_tps(k):
    v = [t["gen_tps"] for t in R[k]["tasks"] if t.get("gen_tps")]
    return sum(v) / len(v) if v else 0


def total_min(k):
    return sum(t.get("wall_seconds", 0) for t in R[k]["tasks"]) / 60


def total_thinking(k):
    return sum(t.get("thinking_words", 0) for t in R[k]["tasks"])


def vram(k):
    return R[k].get("vram_gb") or 0


def layout(title, subtitle, height=420, **kw):
    lay = dict(
        title=dict(
            text=(f"<b>{title}</b><br>"
                  f"<span style='font-size:13px;color:{INK2}'>{subtitle}</span>"),
            font=dict(family=FONT, size=19, color=INK),
            x=0.02, xanchor="left",
        ),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK2),
        width=900, height=height,
        margin=dict(l=180, r=60, t=95, b=60),
        showlegend=False,
        barcornerradius=4,
    )
    lay.update(kw)
    return lay


def axis(**kw):
    a = dict(gridcolor=GRID, griddash="solid", linecolor=BASELINE,
             tickfont=dict(color=MUTED, size=12),
             title_font=dict(color=INK2, size=13), zeroline=False)
    a.update(kw)
    return a


def save(fig, name):
    fig.write_image(f"{BASE}/blog/{name}", scale=2)
    print("saved", name)


def ranked(metric, reverse=True):
    return sorted(MODELS, key=metric, reverse=reverse)


def hbar(order, values, labels, title, subtitle, xtitle, height=380,
         left=170, right=90):
    fig = go.Figure()
    fig.add_bar(
        y=[SHORT[k] for k in order],
        x=[values(k) for k in order],
        orientation="h",
        marker=dict(color=[COLORS[k] for k in order]),
        text=[labels(k) for k in order],
        textposition="outside",
        textfont=dict(color=INK, size=14, family=FONT),
        width=0.42,
    )
    fig.update_layout(
        **layout(title, subtitle, height=height,
                 margin=dict(l=left, r=right, t=95, b=60)),
        bargap=0.5,
        xaxis=axis(title=xtitle,
                   range=[0, max(values(k) for k in MODELS) * 1.26],
                   showgrid=True),
        yaxis=axis(showgrid=False, autorange="reversed",
                   tickfont=dict(color=INK, size=13)),
    )
    return fig


# 1 ------------------------------------------------------------ scoreboard
order = ranked(score)
fig = go.Figure()
fig.add_bar(
    y=[f"{SHORT[k]}<br><span style='font-size:11px;color:{MUTED}'>"
       f"{PARAMS[k]}</span>" for k in order],
    x=[score(k) for k in order],
    orientation="h",
    marker=dict(color=[COLORS[k] for k in order]),
    text=[f"{score(k)}/{N_TASKS}" for k in order],
    textposition="outside",
    textfont=dict(color=INK, size=15, family=FONT),
    width=0.42,
)
fig.update_layout(
    **layout("Hard-problem scoreboard",
             f"Correct answers out of {N_TASKS} hard tasks &nbsp;{DOT}&nbsp; "
             f"RTX 5090 &nbsp;{DOT}&nbsp; Ollama &nbsp;{DOT}&nbsp; 64K context "
             f"&nbsp;{DOT}&nbsp; Q4_K_M",
             height=380),
    bargap=0.5,
    xaxis=axis(title="tasks solved", range=[0, N_TASKS + 2], dtick=2,
               showgrid=True),
    yaxis=axis(showgrid=False, autorange="reversed",
               tickfont=dict(color=INK, size=13)),
)
save(fig, "chart-scoreboard.png")

# 2 --------------------------------------------------------- pass/fail matrix
TINT = {"ok": "#e8f5e8", "wrong": "#fbeaea", "cap": "#fdf1de"}
GLYPH_INK = {"ok": "#0a7d0a", "wrong": "#c23434", "cap": "#9a6410"}
GLYPH = {"ok": CHECK, "wrong": CROSS, "cap": HOURGLASS}
ZVAL = {"ok": 0, "wrong": 1, "cap": 2}


def state_of(rec):
    if rec["correct"]:
        return "ok"
    return "cap" if rec.get("capped") else "wrong"


z, states = [], []
for t in TASKS:
    rz, rs = [], []
    for m in MODELS:
        s = state_of(tasks_of(m)[t])
        rs.append(s)
        rz.append(ZVAL[s])
    z.append(rz)
    states.append(rs)

ylabels = [f"{LABEL[t]}&nbsp;&nbsp;"
           f"<span style='font-size:10px;color:{MUTED}'>{CAT[t]}</span>"
           for t in TASKS]
xlabels = [SHORT[m] for m in MODELS]

fig = go.Figure(go.Heatmap(
    z=z, x=xlabels, y=ylabels,
    colorscale=[[0.0, TINT["ok"]], [0.33, TINT["ok"]],
                [0.34, TINT["wrong"]], [0.66, TINT["wrong"]],
                [0.67, TINT["cap"]], [1.0, TINT["cap"]]],
    zmin=0, zmax=2, showscale=False, xgap=3, ygap=3, hoverinfo="skip",
))
for yi in range(len(TASKS)):
    for xi in range(len(MODELS)):
        s = states[yi][xi]
        fig.add_annotation(
            x=xlabels[xi], y=ylabels[yi], text=f"<b>{GLYPH[s]}</b>",
            showarrow=False,
            font=dict(size=19, color=GLYPH_INK[s], family=FONT),
        )
fig.update_layout(
    **layout("Who solved what",
             f"{CHECK} solved &nbsp;{DOT}&nbsp; {CROSS} wrong answer "
             f"&nbsp;{DOT}&nbsp; {HOURGLASS} ran out of generation budget "
             "(hit the 32,768-token ceiling mid-reasoning)",
             height=min(180 + 34 * N_TASKS, 780),
             margin=dict(l=235, r=50, t=125, b=40)),
    xaxis=axis(showgrid=False, side="top",
               tickfont=dict(color=INK, size=13)),
    yaxis=axis(showgrid=False, autorange="reversed",
               tickfont=dict(color=INK2, size=12)),
)
save(fig, "chart-matrix.png")

# 3 ------------------------------------------------------ category breakdown
CATS = ["Math", "Code", "Reasoning"]
cat_totals = {c: sum(1 for t in TASKS if CAT[t] == c) for c in CATS}
fig = go.Figure()
for k in MODELS:
    tk = tasks_of(k)
    vals = [sum(1 for t in TASKS if CAT[t] == c and tk[t]["correct"])
            for c in CATS]
    fig.add_bar(
        name=SHORT[k],
        x=[f"{c}<br><span style='font-size:11px;color:{MUTED}'>"
           f"{cat_totals[c]} tasks</span>" for c in CATS],
        y=vals,
        marker=dict(color=COLORS[k], line=dict(width=2, color=SURFACE)),
        text=vals, textposition="outside",
        textfont=dict(color=INK, size=13, family=FONT),
        width=0.24,
    )
fig.update_layout(
    **layout("Category breakdown",
             "Tasks solved per category. The three models separate on Math, "
             "not on Code or Reasoning.",
             height=440, showlegend=True,
             margin=dict(l=70, r=50, t=125, b=70),
             legend=dict(orientation="h", y=1.06, x=0, xanchor="left",
                         font=dict(color=INK2, size=12),
                         bgcolor="rgba(0,0,0,0)")),
    bargap=0.42, bargroupgap=0.06,
    xaxis=axis(showgrid=False, tickfont=dict(color=INK, size=13)),
    yaxis=axis(title="tasks solved",
               range=[0, max(cat_totals.values()) + 1.4], dtick=1,
               showgrid=True),
)
save(fig, "chart-category.png")

# 4 ----------------------------------------------------------------- speed
save(hbar(ranked(avg_tps), avg_tps, lambda k: f"{avg_tps(k):.0f} tok/s",
          "Generation speed on hard problems",
          f"Tokens per second, averaged over the {N_TASKS} tasks "
          "(higher is better)",
          "tokens / second"), "chart-speed.png")

# 5 ------------------------------------------------------------ total time
save(hbar(ranked(total_min, reverse=False), total_min,
          lambda k: f"{total_min(k):.1f} min",
          "Time to finish the whole suite",
          f"Total wall-clock minutes for all {N_TASKS} tasks, including "
          "thinking time (lower is better)",
          "minutes"), "chart-time.png")

# 6 ------------------------------------------------------------------ vram
save(hbar(ranked(vram, reverse=False), vram, lambda k: f"{vram(k):.1f} GB",
          "VRAM footprint at 64K context",
          "Resident GPU memory while running. The RTX 5090 has 32 GB "
          "(lower is better)",
          "GB of VRAM"), "chart-vram.png")

# 7 --------------------------------------------------------- thinking volume
if any(total_thinking(k) for k in MODELS):
    save(hbar(ranked(total_thinking, reverse=False), total_thinking,
              lambda k: f"{total_thinking(k):,} words",
              "How much each model thought",
              f"Total words of internal reasoning across {N_TASKS} tasks. "
              "Glimmer does not expose a separate thinking field, so its "
              "count is not like-for-like.",
              "words of thinking", right=120), "chart-thinking.png")

# 8 ------------------------------------------------------------- efficiency
# Labels are placed per point so none collides with a neighbouring marker.
TEXTPOS = {"qwen38": "top center", "nemotron": "bottom center",
           "glimmer": "top center"}
fig = go.Figure()
for k in MODELS:
    fig.add_scatter(
        x=[total_min(k)], y=[score(k)],
        mode="markers+text",
        marker=dict(size=20, color=COLORS[k],
                    line=dict(width=2, color=SURFACE)),
        text=[SHORT[k]], textposition=TEXTPOS[k],
        textfont=dict(color=INK, size=13, family=FONT),
        name=SHORT[k],
    )
lo = min(score(k) for k in MODELS) - 1
hi = max(score(k) for k in MODELS) + 1
fig.update_layout(
    **layout("Accuracy against time spent",
             "Up and to the left is better: more solved, less time. "
             "The y-axis starts at "
             f"{int(lo)} to show the spread, so bar-style zero comparison "
             "does not apply.",
             height=470, margin=dict(l=90, r=90, t=95, b=70)),
    xaxis=axis(title="total wall-clock minutes (lower is better)",
               range=[0, max(total_min(k) for k in MODELS) * 1.25],
               showgrid=True),
    yaxis=axis(title="tasks solved (higher is better)",
               range=[lo, hi], dtick=1, showgrid=True),
)
save(fig, "chart-efficiency.png")

# 9 ----------------------------------------------------- at-a-glance summary
METRICS = [
    ("Tasks solved", score, lambda k: f"{score(k)}/{N_TASKS}", False),
    ("Avg generation speed", avg_tps, lambda k: f"{avg_tps(k):.0f} tok/s",
     False),
    ("Total time", total_min, lambda k: f"{total_min(k):.1f} min", True),
    ("VRAM at 64K", vram, lambda k: f"{vram(k):.1f} GB", True),
]
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[m[0] for m in METRICS],
    horizontal_spacing=0.22, vertical_spacing=0.20,
)
for i, (title, fn, lab, lower_better) in enumerate(METRICS):
    r, c = i // 2 + 1, i % 2 + 1
    o = ranked(fn, reverse=not lower_better)
    fig.add_bar(
        y=[SHORT[k] for k in o], x=[fn(k) for k in o],
        orientation="h",
        marker=dict(color=[COLORS[k] for k in o]),
        text=[lab(k) for k in o], textposition="outside",
        textfont=dict(color=INK, size=12, family=FONT),
        width=0.42, showlegend=False, row=r, col=c,
    )
    fig.update_xaxes(**axis(range=[0, max(fn(k) for k in MODELS) * 1.42],
                            showgrid=True, showticklabels=False),
                     row=r, col=c)
    fig.update_yaxes(**axis(showgrid=False, autorange="reversed",
                            tickfont=dict(color=INK, size=12)),
                     row=r, col=c)
fig.update_layout(
    title=dict(
        text=("<b>Three 27B-class models on 16 hard problems</b><br>"
              f"<span style='font-size:13px;color:{INK2}'>"
              f"RTX 5090 &nbsp;{DOT}&nbsp; Ollama &nbsp;{DOT}&nbsp; "
              f"64K context &nbsp;{DOT}&nbsp; Q4_K_M &nbsp;{DOT}&nbsp; "
              "each model run alone on the GPU</span>"),
        font=dict(family=FONT, size=20, color=INK), x=0.02, xanchor="left",
    ),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(family=FONT, size=12, color=INK2),
    width=1000, height=620, margin=dict(l=130, r=60, t=120, b=50),
    showlegend=False, barcornerradius=4, bargap=0.5,
)
# keep each subplot title over its own panel; only restyle the font
for ann in fig.layout.annotations[:len(METRICS)]:
    ann.font = dict(family=FONT, size=13, color=INK)
save(fig, "chart-summary.png")

print("\nall charts done")
