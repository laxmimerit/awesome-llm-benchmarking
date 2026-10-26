"""Featured image for the Qwen 3.8 post (1500x750, dark-mode palette).

The mini chart is total suite time, not the scoreboard. Muse Glimmer wins the
scoreboard, so a scoreboard here would fight the headline; total time is the
measure this post actually argues on, and Qwen leads it honestly.

Dark steps of the validated reference palette. The three-slot dark set passes
the all-pairs validator (worst CVD dE 9.4, worst normal-vision dE 20.9), and
every bar is directly labelled.
"""

import json

import plotly.graph_objects as go

BASE = "D:/Claude AI/AI Model Performance/qwen3.8-vs-nemotron-vs-glimmer"

# dark-mode steps of the validated reference palette
COLORS = {"qwen38": "#199e70", "nemotron": "#3987e5", "glimmer": "#d95926"}
NAMES = {"qwen38": "Qwen 3.8", "nemotron": "Nemotron 3.5",
         "glimmer": "Muse Glimmer"}
SURFACE = "#1a1a19"
INK = "#ffffff"
INK2 = "#c3c2b7"
MUTED = "#898781"
FONT = 'system-ui, "Segoe UI", sans-serif'

with open(BASE + "/research/results-hard2.json", encoding="utf-8") as f:
    R = json.load(f)


def total_min(k):
    return sum(t.get("wall_seconds", 0) for t in R[k]["tasks"]) / 60


# fastest first; reversed on the way in so the fastest sits at the top
models = sorted(["qwen38", "nemotron", "glimmer"], key=total_min)

fig = go.Figure()
fig.add_bar(
    y=[NAMES[k] for k in models][::-1],
    x=[total_min(k) for k in models][::-1],
    orientation="h",
    marker=dict(color=[COLORS[k] for k in models][::-1]),
    text=[f"  {total_min(k):.1f} min" for k in models][::-1],
    textposition="outside",
    textfont=dict(color=INK, size=26, family=FONT),
    width=0.55,
    showlegend=False,
)

fig.update_layout(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    width=1500,
    height=750,
    margin=dict(l=0, r=60, t=110, b=90),
    barcornerradius=6,
    xaxis=dict(domain=[0.63, 1.0],
               range=[0, max(total_min(k) for k in models) * 1.42],
               visible=False),
    yaxis=dict(domain=[0.10, 0.80],
               tickfont=dict(color=INK2, size=22, family=FONT),
               showgrid=False, ticksuffix="  "),
    annotations=[
        dict(
            x=0.03, y=0.94, xref="paper", yref="paper",
            text="<b>3 models.<br>16 hard problems.<br>1 RTX 5090.</b>",
            font=dict(family=FONT, size=58, color=INK),
            showarrow=False, align="left", xanchor="left", yanchor="top",
        ),
        dict(
            x=0.03, y=0.33, xref="paper", yref="paper",
            text=(
                "Qwen 3.8 27B finished the whole suite<br>"
                "in 6.3 minutes and was the only model<br>"
                "that never ran out of generation budget."
            ),
            font=dict(family=FONT, size=25, color=INK2),
            showarrow=False, align="left", xanchor="left", yanchor="top",
        ),
        dict(
            x=0.03, y=0.05, xref="paper", yref="paper",
            text="RTX 5090  \u00b7  64K context  \u00b7  Ollama  \u00b7  Q4_K_M",
            font=dict(family=FONT, size=21, color=MUTED),
            showarrow=False, align="left", xanchor="left", yanchor="top",
        ),
        dict(
            x=0.815, y=0.97, xref="paper", yref="paper",
            text="<b>TIME TO FINISH ALL 16</b>",
            font=dict(family=FONT, size=20, color=MUTED),
            showarrow=False, xanchor="center", yanchor="top",
        ),
    ],
)

fig.write_image(BASE + "/blog/featured-image.png", scale=2)
print("saved featured-image.png (3000x1500)")
