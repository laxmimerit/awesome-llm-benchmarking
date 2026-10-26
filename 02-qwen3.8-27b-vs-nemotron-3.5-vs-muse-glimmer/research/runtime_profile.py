"""One figure replacing four: the whole runtime profile of the three models.

Generation speed, total suite time, VRAM and thinking volume were four
separate charts in the draft. Individually each is a single-number bar chart,
which is a weak use of a whole figure; together they are one story about what
each model costs to run. Full entity hues here (not the emphasis treatment)
because this panel is a neutral three-way comparison, not a Qwen argument.

Every bar is directly labelled, which also covers the relief rule for aqua on
the light surface.
"""

import json

from plotly.subplots import make_subplots

BASE = "D:/Claude AI/AI Model Performance/qwen3.8-vs-nemotron-vs-glimmer"

COLORS = {"qwen38": "#1baf7a", "nemotron": "#2a78d6", "glimmer": "#eb6834"}
SHORT = {"qwen38": "Qwen 3.8", "nemotron": "Nemotron 3.5",
         "glimmer": "Muse Glimmer"}
VRAM = {"qwen38": 17.5, "nemotron": 25.4, "glimmer": 16.9}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
DOT = "\u00b7"

with open(BASE + "/research/results-hard2.json", encoding="utf-8") as f:
    R = json.load(f)

M = ["qwen38", "nemotron", "glimmer"]


def avg_tps(k):
    v = [t["gen_tps"] for t in R[k]["tasks"] if t.get("gen_tps")]
    return sum(v) / len(v)


def total_min(k):
    return sum(t.get("wall_seconds", 0) for t in R[k]["tasks"]) / 60


def thinking(k):
    return sum(t.get("thinking_words", 0) for t in R[k]["tasks"])


PANELS = [
    ("Generation speed  (higher is better)", avg_tps,
     lambda k: f"{avg_tps(k):.0f} tok/s", True),
    ("Total suite time  (lower is better)", total_min,
     lambda k: f"{total_min(k):.1f} min", False),
    ("VRAM at 64K  (lower is better)", lambda k: VRAM[k],
     lambda k: f"{VRAM[k]} GB", False),
    ("Words of thinking  (lower is better)", thinking,
     lambda k: f"{thinking(k):,}", False),
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

for i, (title, fn, lab, higher_better) in enumerate(PANELS):
    r, c = i // 2 + 1, i % 2 + 1
    order = sorted(M, key=fn, reverse=higher_better)
    fig.add_bar(
        y=[SHORT[k] for k in order], x=[fn(k) for k in order],
        orientation="h",
        marker=dict(color=[COLORS[k] for k in order]),
        text=[lab(k) for k in order], textposition="outside",
        textfont=dict(color=INK, size=12, family=FONT),
        width=0.42, showlegend=False, row=r, col=c,
    )
    fig.update_xaxes(**axis(range=[0, max(fn(k) for k in M) * 1.45],
                            showgrid=True, showticklabels=False),
                     row=r, col=c)
    fig.update_yaxes(**axis(showgrid=False, autorange="reversed",
                            tickfont=dict(color=INK, size=12)),
                     row=r, col=c)

fig.update_layout(
    title=dict(
        text=("<b>What each model costs to run</b><br>"
              f"<span style='font-size:13px;color:{INK2}'>"
              f"All 16 tasks &nbsp;{DOT}&nbsp; RTX 5090 &nbsp;{DOT}&nbsp; "
              f"64K context &nbsp;{DOT}&nbsp; Q4_K_M &nbsp;{DOT}&nbsp; each "
              "model alone on the GPU<br>Muse Glimmer has no separate "
              "thinking field, so its word count is not measured the same "
              "way as the other two.</span>"),
        font=dict(family=FONT, size=20, color=INK), x=0.02, xanchor="left",
    ),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(family=FONT, size=12, color=INK2),
    width=1000, height=560, margin=dict(l=140, r=80, t=135, b=50),
    showlegend=False, barcornerradius=4, bargap=0.5,
)
for ann in fig.layout.annotations[:len(PANELS)]:
    ann.font = dict(family=FONT, size=13, color=INK)

fig.write_image(f"{BASE}/blog/chart-runtime-profile.png", scale=2)
print("saved chart-runtime-profile.png")
