"""Architecture comparison: three different bets on the same size class.

Every number here comes from `ollama show` on the installed tag (see
model-cards.md), except VRAM, which is measured from /api/ps during the
benchmark run at 64K context.

Qwen is placed first in every panel because this figure accompanies a
Qwen-focused post; the bars are still drawn to true scale and the ranking is
readable from the values, so ordering is presentation, not distortion.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE = "D:/Claude AI/AI Model Performance/qwen3.8-vs-nemotron-vs-glimmer"

COLORS = {"qwen38": "#1baf7a", "nemotron": "#2a78d6", "glimmer": "#eb6834"}
SHORT = {"qwen38": "Qwen 3.8", "nemotron": "Nemotron 3.5",
         "glimmer": "Muse Glimmer"}
ORDER = ["qwen38", "nemotron", "glimmer"]

SPEC = {
    "qwen38": dict(params=27.3, ctx=262144, emb=5120, vram=17.5,
                   arch="qwen35", moe=False, vision=True, thinking=True),
    "nemotron": dict(params=32.9, ctx=1048576, emb=2688, vram=25.4,
                     arch="nemotron_h_moe", moe=True, vision=False,
                     thinking=True),
    "glimmer": dict(params=27.9, ctx=131072, emb=6656, vram=16.9,
                    arch="muse-glimmer", moe=False, vision=True,
                    thinking=False),
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
CHECK = "\u2713"
CROSS = "\u2717"
DOT = "\u00b7"

TINT_YES, TINT_NO = "#e8f5e8", "#f2f1ec"
INK_YES, INK_NO = "#0a7d0a", "#8a8880"


def axis(**kw):
    a = dict(gridcolor=GRID, griddash="solid", linecolor=BASELINE,
             tickfont=dict(color=MUTED, size=11),
             title_font=dict(color=INK2, size=12), zeroline=False)
    a.update(kw)
    return a


PANELS = [
    ("Parameters", lambda k: SPEC[k]["params"],
     lambda k: f"{SPEC[k]['params']}B" + (" MoE" if SPEC[k]["moe"] else "")),
    ("Context window", lambda k: SPEC[k]["ctx"] / 1024,
     lambda k: f"{SPEC[k]['ctx'] // 1024}K"),
    ("Embedding width", lambda k: SPEC[k]["emb"],
     lambda k: f"{SPEC[k]['emb']:,}"),
    ("Measured VRAM at 64K", lambda k: SPEC[k]["vram"],
     lambda k: f"{SPEC[k]['vram']} GB"),
]

fig = make_subplots(
    rows=3, cols=2,
    specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None]],
    # no title on the capability row: its own column headers sit on top and
    # a subplot title would land in exactly the same place
    subplot_titles=[p[0] for p in PANELS] + [""],
    horizontal_spacing=0.20, vertical_spacing=0.10,
    row_heights=[0.27, 0.27, 0.46],
)

for i, (title, fn, lab) in enumerate(PANELS):
    r, c = i // 2 + 1, i % 2 + 1
    fig.add_bar(
        y=[SHORT[k] for k in ORDER], x=[fn(k) for k in ORDER],
        orientation="h",
        marker=dict(color=[COLORS[k] for k in ORDER]),
        text=[lab(k) for k in ORDER], textposition="outside",
        textfont=dict(color=INK, size=11, family=FONT),
        width=0.42, showlegend=False, row=r, col=c,
    )
    fig.update_xaxes(**axis(range=[0, max(fn(k) for k in ORDER) * 1.45],
                            showgrid=True, showticklabels=False),
                     row=r, col=c)
    fig.update_yaxes(**axis(showgrid=False, autorange="reversed",
                            tickfont=dict(color=INK, size=11)),
                     row=r, col=c)

# ---- capability matrix (row 3)
CAPS = [
    ("Mixture of experts", lambda k: SPEC[k]["moe"]),
    ("Vision input", lambda k: SPEC[k]["vision"]),
    ("Separate thinking field", lambda k: SPEC[k]["thinking"]),
]
xlabels = [SHORT[k] for k in ORDER]
ylabels = [c[0] for c in CAPS]
z = [[1 if fn(k) else 0 for k in ORDER] for _, fn in CAPS]

fig.add_trace(
    go.Heatmap(
        z=z, x=xlabels, y=ylabels,
        colorscale=[[0, TINT_NO], [1, TINT_YES]],
        zmin=0, zmax=1, showscale=False, xgap=3, ygap=3, hoverinfo="skip",
    ),
    row=3, col=1,
)
for yi, (_, fn) in enumerate(CAPS):
    for xi, k in enumerate(ORDER):
        yes = bool(fn(k))
        fig.add_annotation(
            x=xlabels[xi], y=ylabels[yi],
            text=f"<b>{CHECK if yes else CROSS}</b>",
            showarrow=False, xref="x5", yref="y5",
            font=dict(size=17, color=INK_YES if yes else INK_NO, family=FONT),
        )
fig.update_xaxes(**axis(showgrid=False, side="top",
                        tickfont=dict(color=INK, size=12)), row=3, col=1)
fig.update_yaxes(**axis(showgrid=False, autorange="reversed",
                        tickfont=dict(color=INK2, size=11)), row=3, col=1)

fig.update_layout(
    title=dict(
        text=("<b>Three different bets on the same size class</b><br>"
              f"<span style='font-size:13px;color:{INK2}'>"
              "All three are Q4_K_M on one RTX 5090. Qwen 3.8 is a dense "
              f"27.3B with vision; Nemotron is a 32.9B sparse MoE with a 1M "
              f"context and no vision;<br>Muse Glimmer is a dense 27.9B with "
              "the widest embedding and no separate thinking channel."
              "</span>"),
        font=dict(family=FONT, size=20, color=INK), x=0.02, xanchor="left",
    ),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(family=FONT, size=12, color=INK2),
    width=1000, height=780, margin=dict(l=140, r=60, t=140, b=50),
    showlegend=False, barcornerradius=4, bargap=0.5,
)
for ann in fig.layout.annotations[:len(PANELS)]:
    ann.font = dict(family=FONT, size=13, color=INK)

fig.write_image(f"{BASE}/blog/chart-architecture.png", scale=2)
print("saved chart-architecture.png")
