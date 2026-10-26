"""Qwen-forward hero figure: the four metrics Qwen 3.8 leads outright.

Uses the emphasis pattern rather than three competing hues - Qwen keeps its
series aqua, the two rivals recede to a neutral so the eye lands on the
subject of the post. Identity is still carried by the axis label and the
direct value label on every bar, so nothing depends on colour.

Only metrics Qwen actually leads are shown, and the panel titles say which
direction is better. VRAM is deliberately excluded: Muse Glimmer is lower
there (16.9 GB vs 17.5 GB), so including it under "where Qwen leads" would
be false.
"""

import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE = "D:/Claude AI/AI Model Performance/qwen3.8-vs-nemotron-vs-glimmer"

QWEN = "#1baf7a"
RECESSIVE = "#c9c7be"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
DOT = "\u00b7"

SHORT = {"qwen38": "Qwen 3.8", "nemotron": "Nemotron 3.5",
         "glimmer": "Muse Glimmer"}

with open(BASE + "/research/results-hard2.json", encoding="utf-8") as f:
    R = json.load(f)

M = ["qwen38", "nemotron", "glimmer"]


def total_min(k):
    return sum(t.get("wall_seconds", 0) for t in R[k]["tasks"]) / 60


def gen_tokens(k):
    return sum(t.get("gen_tokens", 0) for t in R[k]["tasks"])


def thinking(k):
    return sum(t.get("thinking_words", 0) for t in R[k]["tasks"])


def capped(k):
    return sum(1 for t in R[k]["tasks"] if t.get("capped"))


PANELS = [
    ("Total suite time  (lower is better)", total_min,
     lambda k: f"{total_min(k):.1f} min"),
    ("Tokens generated  (lower is better)", gen_tokens,
     lambda k: f"{gen_tokens(k):,}"),
    ("Words of thinking  (lower is better)", thinking,
     lambda k: f"{thinking(k):,}"),
    ("Times it ran out of budget  (lower is better)", capped,
     lambda k: f"{capped(k)}"),
]


def axis(**kw):
    a = dict(gridcolor=GRID, griddash="solid", linecolor=BASELINE,
             tickfont=dict(color=MUTED, size=11),
             title_font=dict(color=INK2, size=12), zeroline=False)
    a.update(kw)
    return a


fig = make_subplots(rows=2, cols=2,
                    subplot_titles=[p[0] for p in PANELS],
                    horizontal_spacing=0.24, vertical_spacing=0.22)

for i, (title, fn, lab) in enumerate(PANELS):
    r, c = i // 2 + 1, i % 2 + 1
    order = sorted(M, key=fn)                   # best (lowest) first
    fig.add_bar(
        y=[SHORT[k] for k in order], x=[fn(k) for k in order],
        orientation="h",
        marker=dict(color=[QWEN if k == "qwen38" else RECESSIVE
                           for k in order]),
        text=[lab(k) for k in order], textposition="outside",
        textfont=dict(color=INK, size=12, family=FONT),
        width=0.42, showlegend=False, row=r, col=c,
    )
    top = max(fn(k) for k in M)
    fig.update_xaxes(**axis(range=[0, top * 1.5 if top else 1],
                            showgrid=True, showticklabels=False),
                     row=r, col=c)
    fig.update_yaxes(**axis(showgrid=False, autorange="reversed",
                            tickfont=dict(color=INK, size=12)),
                     row=r, col=c)

fig.update_layout(
    title=dict(
        text=("<b>Qwen 3.8 27B is the efficiency pick</b><br>"
              f"<span style='font-size:13px;color:{INK2}'>"
              "Four measures where Qwen 3.8 leads outright, on 16 hard "
              f"problems &nbsp;{DOT}&nbsp; RTX 5090 &nbsp;{DOT}&nbsp; 64K "
              f"context &nbsp;{DOT}&nbsp; Q4_K_M<br>It is not the most "
              "accurate model here: Muse Glimmer scored 15/16 to Qwen's "
              "14/16.</span>"),
        font=dict(family=FONT, size=20, color=INK), x=0.02, xanchor="left",
    ),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(family=FONT, size=12, color=INK2),
    width=1000, height=560, margin=dict(l=140, r=70, t=135, b=50),
    showlegend=False, barcornerradius=4, bargap=0.5,
)
for ann in fig.layout.annotations[:len(PANELS)]:
    ann.font = dict(family=FONT, size=13, color=INK)

fig.write_image(f"{BASE}/blog/chart-qwen-hero.png", scale=2)
print("saved chart-qwen-hero.png")
