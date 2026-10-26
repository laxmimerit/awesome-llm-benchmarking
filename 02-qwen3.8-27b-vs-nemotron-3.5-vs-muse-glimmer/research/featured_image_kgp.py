"""KGP Talkie thumbnail: 1920x1080 JPG, light background.

Site convention is 1920x1080 JPG under public/images/thumbs/, so this renders
960x540 at scale 2. Light surface per request.

The mini chart is total suite time rather than the scoreboard: Muse Glimmer
wins the scoreboard, so a scoreboard on a Qwen-led thumbnail would contradict
the headline. Total time is what the article argues on and Qwen leads it.

Light steps of the validated reference palette; the three-slot light set passes
the all-pairs validator and every bar is directly labelled, which also covers
the relief rule for aqua on a light surface.
"""

import json

import plotly.graph_objects as go

BASE = "D:/Claude AI/AI Model Performance/qwen3.8-vs-nemotron-vs-glimmer"
OUT = ("D:/Packages/kgptalkie/public/images/thumbs/"
       "qwen-3-8-27b-vs-nemotron-3-5-vs-muse-glimmer.jpg")

COLORS = {"qwen38": "#1baf7a", "nemotron": "#2a78d6", "glimmer": "#eb6834"}
NAMES = {"qwen38": "Qwen 3.8", "nemotron": "Nemotron 3.5",
         "glimmer": "Muse Glimmer"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
FONT = 'system-ui, "Segoe UI", sans-serif'

with open(BASE + "/research/results-hard2.json", encoding="utf-8") as f:
    R = json.load(f)


def total_min(k):
    return sum(t.get("wall_seconds", 0) for t in R[k]["tasks"]) / 60


models = sorted(["qwen38", "nemotron", "glimmer"], key=total_min)

fig = go.Figure()
fig.add_bar(
    y=[NAMES[k] for k in models][::-1],
    x=[total_min(k) for k in models][::-1],
    orientation="h",
    marker=dict(color=[COLORS[k] for k in models][::-1]),
    text=[f"  {total_min(k):.1f} min" for k in models][::-1],
    textposition="outside",
    textfont=dict(color=INK, size=19, family=FONT),
    width=0.5,
    showlegend=False,
)

fig.update_layout(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    width=960,
    height=540,
    margin=dict(l=0, r=40, t=70, b=60),
    barcornerradius=5,
    xaxis=dict(domain=[0.60, 1.0],
               range=[0, max(total_min(k) for k in models) * 1.45],
               visible=False),
    yaxis=dict(domain=[0.12, 0.76],
               tickfont=dict(color=INK2, size=16, family=FONT),
               showgrid=False, ticksuffix="  "),
    annotations=[
        dict(
            x=0.04, y=0.93, xref="paper", yref="paper",
            text="<b>3 models.<br>16 hard problems.<br>1 RTX 5090.</b>",
            font=dict(family=FONT, size=40, color=INK),
            showarrow=False, align="left", xanchor="left", yanchor="top",
        ),
        dict(
            x=0.04, y=0.35, xref="paper", yref="paper",
            text=("Qwen 3.8 27B finished the whole suite in 6.3<br>"
                  "minutes and was the only model that never<br>"
                  "ran out of generation budget."),
            font=dict(family=FONT, size=17, color=INK2),
            showarrow=False, align="left", xanchor="left", yanchor="top",
        ),
        dict(
            x=0.04, y=0.09, xref="paper", yref="paper",
            text="RTX 5090  \u00b7  64K context  \u00b7  Ollama  \u00b7  Q4_K_M",
            font=dict(family=FONT, size=14, color=MUTED),
            showarrow=False, align="left", xanchor="left", yanchor="top",
        ),
        dict(
            x=0.80, y=0.96, xref="paper", yref="paper",
            text="<b>TIME TO FINISH ALL 16</b>",
            font=dict(family=FONT, size=14, color=MUTED),
            showarrow=False, xanchor="center", yanchor="top",
        ),
    ],
)

fig.write_image(OUT, scale=2)
print("saved", OUT)
