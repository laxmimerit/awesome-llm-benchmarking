"""Five 1:1 cards (1080x1080) explaining the benchmark for social.

Each card pairs a headline with one real chart, so the text carries the claim
and the graphic carries the evidence. One idea per card, because a square in a
feed is read in about two seconds.

Light surface and the validated light categorical slots, so the set matches
the article thumbnail. Numbers are read from results-hard2.json wherever
possible rather than retyped.
"""

import json

import plotly.graph_objects as go

BASE = "D:/Claude AI/AI Model Performance/qwen3.8-vs-nemotron-vs-glimmer"
OUT = BASE + "/blog/social"

COLORS = {"qwen38": "#1baf7a", "nemotron": "#2a78d6", "glimmer": "#eb6834"}
SHORT = {"qwen38": "Qwen 3.8", "nemotron": "Nemotron 3.5",
         "glimmer": "Muse Glimmer"}
VRAM = {"qwen38": 17.5, "nemotron": 25.4, "glimmer": 16.9}
CTX = {"qwen38": 256, "nemotron": 1024, "glimmer": 128}
PARAMS = {"qwen38": "27.3B dense", "nemotron": "32.9B MoE",
          "glimmer": "27.9B dense"}
MEDIAN = {"qwen38": 5.9, "nemotron": 10.6, "glimmer": 14.5}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
GOOD = "#0a7d0a"
CRIT = "#c23434"
WARN = "#9a6410"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
DOT = "\u00b7"

with open(BASE + "/research/results-hard2.json", encoding="utf-8") as f:
    R = json.load(f)

M = ["qwen38", "nemotron", "glimmer"]


def score(k):
    return sum(1 for t in R[k]["tasks"] if t["correct"])


def capped(k):
    return sum(1 for t in R[k]["tasks"] if t.get("capped"))


def wrong(k):
    return sum(1 for t in R[k]["tasks"]
               if not t["correct"] and not t.get("capped"))


def total_min(k):
    return sum(t.get("wall_seconds", 0) for t in R[k]["tasks"]) / 60


def card(eyebrow, headline, footer, chart_domain=(0.20, 0.60)):
    """Square canvas with eyebrow, headline, a chart band, and a footer."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        width=540, height=540,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False, barcornerradius=4, bargap=0.45,
        font=dict(family=FONT, size=12, color=INK2),
        xaxis=dict(domain=[0.30, 0.93], visible=False),
        yaxis=dict(domain=list(chart_domain), showgrid=False,
                   tickfont=dict(color=INK2, size=12, family=FONT),
                   ticksuffix="  ", linecolor=SURFACE),
        annotations=[
            dict(x=0.07, y=0.955, xref="paper", yref="paper", text=eyebrow,
                 font=dict(family=FONT, size=12, color=MUTED),
                 showarrow=False, xanchor="left", yanchor="top"),
            dict(x=0.07, y=0.915, xref="paper", yref="paper",
                 text=f"<b>{headline}</b>",
                 font=dict(family=FONT, size=25, color=INK),
                 showarrow=False, xanchor="left", yanchor="top",
                 align="left"),
            dict(x=0.07, y=0.165, xref="paper", yref="paper", text=footer,
                 font=dict(family=FONT, size=13.5, color=INK2),
                 showarrow=False, xanchor="left", yanchor="top",
                 align="left"),
            dict(x=0.07, y=0.035, xref="paper", yref="paper",
                 text=f"kgptalkie.com &nbsp;{DOT}&nbsp; RTX 5090 "
                      f"&nbsp;{DOT}&nbsp; Ollama &nbsp;{DOT}&nbsp; 64K "
                      f"&nbsp;{DOT}&nbsp; Q4_K_M",
                 font=dict(family=FONT, size=10.5, color=MUTED),
                 showarrow=False, xanchor="left", yanchor="bottom"),
        ],
    )
    return fig


def save(fig, n):
    fig.write_image(f"{OUT}/card-{n}.png", scale=2)
    print(f"saved card-{n}.png")


def hbar(fig, values, labels, order=None, colors=None, xmax_mult=1.55):
    o = order or M
    fig.add_bar(
        y=[SHORT[k] for k in o][::-1],
        x=[values(k) for k in o][::-1],
        orientation="h",
        marker=dict(color=[(colors or COLORS)[k] for k in o][::-1]),
        text=[f"  {labels(k)}" for k in o][::-1],
        textposition="outside",
        textfont=dict(color=INK, size=14, family=FONT),
        width=0.5,
    )
    fig.update_xaxes(range=[0, max(values(k) for k in o) * xmax_mult])
    return fig


# ---------------------------------------------------------------- card 1
fig = card(
    "LOCAL LLM BENCHMARK",
    "3 models.<br>16 hard problems.<br>1 RTX 5090.",
    "Every answer is checked by running code, not by reading it.<br>"
    "Each model ran alone on the GPU, with all others unloaded.",
    chart_domain=(0.22, 0.55),
)
# model spec strip instead of a chart
shapes, anns = [], list(fig.layout.annotations)
for i, k in enumerate(M):
    y1 = 0.68 - i * 0.135
    y0 = y1 - 0.10
    shapes.append(dict(type="rect", xref="paper", yref="paper",
                       x0=0.07, x1=0.093, y0=y0, y1=y1,
                       fillcolor=COLORS[k], line=dict(width=0)))
    anns.append(dict(x=0.115, y=(y0 + y1) / 2 + 0.018, xref="paper",
                     yref="paper", text=f"<b>{SHORT[k]}</b>",
                     font=dict(family=FONT, size=16, color=INK),
                     showarrow=False, xanchor="left", yanchor="middle"))
    anns.append(dict(x=0.115, y=(y0 + y1) / 2 - 0.025, xref="paper",
                     yref="paper", text=PARAMS[k],
                     font=dict(family=FONT, size=12, color=MUTED),
                     showarrow=False, xanchor="left", yanchor="middle"))
fig.update_layout(shapes=shapes, annotations=anns,
                  xaxis=dict(visible=False), yaxis=dict(visible=False))
save(fig, 1)

# ---------------------------------------------------------------- card 2
fig = card(
    "ARCHITECTURE",
    "Same size class.<br>Very different inside.",
    "Nemotron is a sparse mixture of experts with a 1M context, and it<br>"
    "pays for that in memory. On a 32GB card, 25.4GB leaves little room.",
    chart_domain=(0.27, 0.64),
)
hbar(fig, lambda k: VRAM[k], lambda k: f"{VRAM[k]} GB",
     order=sorted(M, key=lambda k: VRAM[k]))
anns = list(fig.layout.annotations)
anns.append(dict(x=0.07, y=0.66, xref="paper", yref="paper",
                 text="VRAM USED AT 64K CONTEXT",
                 font=dict(family=FONT, size=11, color=MUTED),
                 showarrow=False, xanchor="left", yanchor="bottom"))
fig.update_layout(annotations=anns)
save(fig, 2)

# ---------------------------------------------------------------- card 3
fig = card(
    "THE RESULT",
    "Two models tied.<br>They failed in opposite ways.",
    "3 of the 5 total failures were not wrong reasoning. The model was<br>"
    "still working when the 32,768 token budget ran out.",
    chart_domain=(0.27, 0.66),
)
order = sorted(M, key=lambda k: -score(k))
# score sits INSIDE the bar so the whole right margin is free for the
# failure-mode note; outside labels would collide with it
fig.add_bar(
    y=[SHORT[k] for k in order][::-1],
    x=[score(k) for k in order][::-1],
    orientation="h",
    marker=dict(color=[COLORS[k] for k in order][::-1]),
    text=[f"{score(k)}/16  " for k in order][::-1],
    textposition="inside", insidetextanchor="end",
    textfont=dict(color="#ffffff", size=14, family=FONT),
    width=0.5,
)
fig.update_xaxes(range=[0, 16], domain=[0.30, 0.62])
anns = list(fig.layout.annotations)
anns.append(dict(x=0.07, y=0.68, xref="paper", yref="paper",
                 text="TASKS SOLVED OUT OF 16",
                 font=dict(family=FONT, size=11, color=MUTED),
                 showarrow=False, xanchor="left", yanchor="bottom"))
for i, k in enumerate(order):
    if wrong(k):
        txt, col = f"{wrong(k)} wrong answers", CRIT
    elif capped(k):
        txt, col = f"{capped(k)} ran out of budget", WARN
    else:
        txt, col = "", MUTED
    # bar centres for a 3-category axis spanning the chart domain
    anns.append(dict(x=0.955, y=0.595 - i * 0.13, xref="paper", yref="paper",
                     text=txt, font=dict(family=FONT, size=11.5, color=col),
                     showarrow=False, xanchor="right", yanchor="middle"))
fig.update_layout(annotations=anns)
save(fig, 3)

# ---------------------------------------------------------------- card 4
fig = card(
    "SPEED",
    "Qwen 3.8 finishes first.",
    "Median task: 5.9s for Qwen, 10.6s for Nemotron, 14.5s for Glimmer.<br>"
    "Qwen was quickest on 8 of the 13 tasks all three solved.",
    chart_domain=(0.27, 0.66),
)
hbar(fig, total_min, lambda k: f"{total_min(k):.1f} min",
     order=sorted(M, key=total_min))
anns = list(fig.layout.annotations)
anns.append(dict(x=0.07, y=0.68, xref="paper", yref="paper",
                 text="TIME TO FINISH ALL 16 PROBLEMS",
                 font=dict(family=FONT, size=11, color=MUTED),
                 showarrow=False, xanchor="left", yanchor="bottom"))
fig.update_layout(annotations=anns)
save(fig, 4)

# ---------------------------------------------------------------- card 5
fig = card(
    "THE CATCH",
    "Fast, and confidently wrong.",
    "A model that runs out of tokens tells you it failed.<br>"
    "A model that invents a citation does not. Check the fast answers.",
    chart_domain=(0.30, 0.60),
)
anns = list(fig.layout.annotations)
shapes = []
anns.append(dict(
    x=0.07, y=0.68, xref="paper", yref="paper",
    text="Qwen 3.8 on the 4x10 domino tiling problem:",
    font=dict(family=FONT, size=12.5, color=INK2),
    showarrow=False, xanchor="left", yanchor="bottom"))
boxes = [
    (0.07, 0.50, "#fbeaea", CRIT, "17305", "Qwen's answer, with the note<br>"
     "\"this matches OEIS A005178\""),
    (0.53, 0.96, "#e8f5e8", GOOD, "18061", "What A005178 actually says,<br>"
     "confirmed by our own script"),
]
for x0, x1, tint, ink, num, cap in boxes:
    shapes.append(dict(type="rect", xref="paper", yref="paper",
                       x0=x0, x1=x1, y0=0.36, y1=0.64,
                       fillcolor=tint, line=dict(width=0)))
    anns.append(dict(x=(x0 + x1) / 2, y=0.555, xref="paper", yref="paper",
                     text=f"<b>{num}</b>",
                     font=dict(family=FONT, size=34, color=ink),
                     showarrow=False, xanchor="center", yanchor="middle"))
    anns.append(dict(x=(x0 + x1) / 2, y=0.425, xref="paper", yref="paper",
                     text=cap, font=dict(family=FONT, size=10.5, color=INK2),
                     showarrow=False, xanchor="center", yanchor="middle",
                     align="center"))
fig.update_layout(shapes=shapes, annotations=anns,
                  xaxis=dict(visible=False), yaxis=dict(visible=False))
save(fig, 5)

print("\nall 5 cards done (1080x1080)")
