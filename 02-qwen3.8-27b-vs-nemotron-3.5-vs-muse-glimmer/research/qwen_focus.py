"""Where Qwen 3.8 27B comes out ahead.

Two honest views:

  1. Per-task speed advantage over the fastest rival, on the 13 tasks all
     three models solved. Comparing time on a task somebody got wrong would
     be meaningless, so unsolved-by-anyone tasks are excluded and the excluded
     set is named in the subtitle rather than silently dropped.
  2. The head-to-head "fastest model" record over those same 13 tasks.

Bar colour is the entity that won the task, matching the series palette used
throughout (Nemotron blue, Glimmer orange, Qwen aqua), so hue keeps meaning
the same model it means everywhere else. A zero rule marks the break between
"Qwen faster" and "Qwen slower", and every bar is directly labelled, which
also satisfies the relief rule for aqua on the light surface.
"""

import json

import plotly.graph_objects as go

BASE = "D:/Claude AI/AI Model Performance/qwen3.8-vs-nemotron-vs-glimmer"

COLORS = {"nemotron": "#2a78d6", "glimmer": "#eb6834", "qwen38": "#1baf7a"}
SHORT = {"nemotron": "Nemotron 3.5", "glimmer": "Muse Glimmer",
         "qwen38": "Qwen 3.8"}
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
T = {k: {t["task"]: t for t in R[k]["tasks"]} for k in M}
TASKS = [t["task"] for t in R["qwen38"]["tasks"]]
LAB = {t["task"]: t["label"] for t in R["qwen38"]["tasks"]}

# only tasks every model solved -> a time comparison is meaningful
common = [t for t in TASKS if all(T[k][t]["correct"] for k in M)]
excluded = [t for t in TASKS if t not in common]


def axis(**kw):
    a = dict(gridcolor=GRID, griddash="solid", linecolor=BASELINE,
             tickfont=dict(color=MUTED, size=12),
             title_font=dict(color=INK2, size=13), zeroline=False)
    a.update(kw)
    return a


def save(fig, name):
    fig.write_image(f"{BASE}/blog/{name}", scale=2)
    print("saved", name)


# ---------------------------------------------------- 1. speed advantage
rows = []
for t in common:
    qt = T["qwen38"][t]["wall_seconds"]
    rivals = {k: T[k][t]["wall_seconds"] for k in M if k != "qwen38"}
    best_rival = min(rivals, key=rivals.get)
    advantage = rivals[best_rival] - qt          # >0 means Qwen was faster
    winner = "qwen38" if advantage > 0 else best_rival
    rows.append((t, advantage, winner, qt, rivals[best_rival], best_rival))

rows.sort(key=lambda r: r[1])                     # worst at bottom of list
ylabels = [LAB[r[0]] for r in rows]
vals = [r[1] for r in rows]
colors = [COLORS[r[2]] for r in rows]
texts = [(f"  {r[1]:+.1f}s  " if r[1] > 0 else f"  {r[1]:+.1f}s  ")
         for r in rows]

fig = go.Figure()
fig.add_bar(
    y=ylabels, x=vals, orientation="h",
    marker=dict(color=colors),
    text=texts,
    textposition="outside",
    textfont=dict(color=INK, size=12, family=FONT),
    width=0.5,
    hoverinfo="skip",
)
span = max(max(vals), abs(min(vals)))
fig.add_shape(type="line", x0=0, x1=0, y0=-0.6, y1=len(rows) - 0.4,
              line=dict(color=BASELINE, width=1.5))
fig.add_annotation(x=span * 0.62, y=len(rows) - 0.25, text="Qwen 3.8 faster",
                   showarrow=False,
                   font=dict(color=COLORS["qwen38"], size=12, family=FONT))
fig.add_annotation(x=-span * 0.62, y=len(rows) - 0.25,
                   text="Nemotron 3.5 faster", showarrow=False,
                   font=dict(color=COLORS["nemotron"], size=12, family=FONT))
fig.update_layout(
    title=dict(
        text=("<b>Where Qwen 3.8 is faster</b><br>"
              f"<span style='font-size:13px;color:{INK2}'>"
              "Seconds saved against the quickest rival, on the 13 tasks all "
              f"three models solved.<br>Qwen 3.8 is quickest on 8 of 13 "
              f"&nbsp;{DOT}&nbsp; the 3 tasks somebody got wrong are excluded, "
              "because timing a wrong answer means nothing.</span>"),
        font=dict(family=FONT, size=19, color=INK), x=0.02, xanchor="left",
    ),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(family=FONT, size=13, color=INK2),
    width=940, height=580, margin=dict(l=230, r=70, t=130, b=60),
    showlegend=False, barcornerradius=4, bargap=0.42,
    xaxis=axis(title="seconds faster than the best rival",
               range=[-span * 1.35, span * 1.35], showgrid=True),
    yaxis=axis(showgrid=False, tickfont=dict(color=INK2, size=12)),
)
save(fig, "chart-qwen-advantage.png")

# ------------------------------------------------------ 2. fastest record
record = {k: 0 for k in M}
for t in common:
    w = {k: T[k][t]["wall_seconds"] for k in M}
    record[min(w, key=w.get)] += 1
order = sorted(M, key=lambda k: -record[k])

fig = go.Figure()
fig.add_bar(
    y=[SHORT[k] for k in order], x=[record[k] for k in order],
    orientation="h",
    marker=dict(color=[COLORS[k] for k in order]),
    text=[f"{record[k]} of {len(common)}" for k in order],
    textposition="outside",
    textfont=dict(color=INK, size=14, family=FONT),
    width=0.42,
)
fig.update_layout(
    title=dict(
        text=("<b>Fastest model, task by task</b><br>"
              f"<span style='font-size:13px;color:{INK2}'>"
              f"How often each model finished first across the {len(common)} "
              "tasks all three solved</span>"),
        font=dict(family=FONT, size=19, color=INK), x=0.02, xanchor="left",
    ),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(family=FONT, size=13, color=INK2),
    width=900, height=360, margin=dict(l=170, r=90, t=100, b=60),
    showlegend=False, barcornerradius=4, bargap=0.5,
    xaxis=axis(title="tasks finished first", range=[0, len(common)],
               dtick=2, showgrid=True),
    yaxis=axis(showgrid=False, autorange="reversed",
               tickfont=dict(color=INK, size=13)),
)
save(fig, "chart-qwen-winrecord.png")

# ------------------------------------------------------------- console
print(f"\ncommon tasks: {len(common)}   excluded: "
      f"{[LAB[t] for t in excluded]}")
print("fastest record:", {SHORT[k]: v for k, v in record.items()})
print("\nper-task advantage (positive = Qwen faster):")
for t, adv, win, qt, rt, rk in sorted(rows, key=lambda r: -r[1]):
    print(f"  {LAB[t]:28} {adv:+7.1f}s   qwen {qt:6.1f}s vs "
          f"{SHORT[rk]} {rt:6.1f}s")
