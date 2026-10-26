"""Blog charts: Nemotron 3.5 Lightning vs Muse Glimmer 30B (Plotly + Kaleido).

Colors and chrome follow the validated reference dataviz palette (light mode):
series-1 blue #2a78d6 (Nemotron), series-2 orange #eb6834 (Glimmer).
"""

import json

import plotly.graph_objects as go

BASE = "D:/Claude AI/AI Model Performance/nemotron-3.5-vs-muse-glimmer"
OUT = BASE + "/blog"

BLUE = "#2a78d6"   # Nemotron
ORANGE = "#eb6834"  # Muse Glimmer
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
FONT = 'system-ui, "Segoe UI", sans-serif'

NEMO = "Nemotron 3.5 Lightning"
GLIM = "Muse Glimmer 30B"

with open(BASE + "/research/results.json", encoding="utf-8") as f:
    R = json.load(f)

TASK_LABELS = {
    "math_arithmetic": "Arithmetic",
    "math_word": "Word problem",
    "logic": "Logic riddle",
    "knowledge": "Knowledge",
    "coding_simple": "Write function",
    "coding_bugfix": "Fix bug",
    "summarize": "Summarize",
    "instruction_following": "Follow format",
    "json_extract": "JSON extract",
    "long_context_needle": "64K needle",
}


def layout(title, subtitle, height=460, **kw):
    lay = dict(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:13px;color:{INK2}'>{subtitle}</span>",
            font=dict(family=FONT, size=18, color=INK),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK2),
        width=900,
        height=height,
        margin=dict(l=70, r=40, t=90, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="right",
            x=1,
            font=dict(size=13, color=INK2),
        ),
        barcornerradius=4,
    )
    lay.update(kw)
    return lay


def axis(**kw):
    a = dict(
        gridcolor=GRID,
        gridwidth=1,
        linecolor=BASELINE,
        tickfont=dict(color=MUTED, size=12),
        title_font=dict(color=INK2, size=13),
        zeroline=False,
    )
    a.update(kw)
    return a


def save(fig, name):
    fig.write_image(f"{OUT}/{name}", scale=2)
    print("saved", name)


# 1 ---- per-task generation speed (grouped horizontal bars)
tasks = [t for t in TASK_LABELS if t != "long_context_needle"]
labels = [TASK_LABELS[t] for t in tasks]
nemo_tps = [next(r["gen_tps"] for r in R["nemotron"] if r["task"] == t) for t in tasks]
glim_tps = [next(r["gen_tps"] for r in R["glimmer"] if r["task"] == t) for t in tasks]

fig = go.Figure()
fig.add_bar(
    y=labels, x=nemo_tps, name=NEMO, orientation="h",
    marker=dict(color=BLUE),
    text=[f"{v:.0f}" for v in nemo_tps], textposition="outside",
    textfont=dict(color=INK2, size=12),
)
fig.add_bar(
    y=labels, x=glim_tps, name=GLIM, orientation="h",
    marker=dict(color=ORANGE),
    text=[f"{v:.0f}" for v in glim_tps], textposition="outside",
    textfont=dict(color=INK2, size=12),
)
fig.update_layout(
    **layout(
        "Generation speed on simple tasks",
        "Tokens per second on an RTX 5090, 64K context window (higher is better)",
        height=560,
    ),
    barmode="group",
    bargap=0.3,
    bargroupgap=0.12,
    xaxis=axis(title="tokens / second", range=[0, 165]),
    yaxis=axis(autorange="reversed", showgrid=False),
)
save(fig, "chart-gen-speed.png")

# 2 ---- speed vs context length (the KV cache story)
short_nemo = sum(nemo_tps) / len(nemo_tps)
short_glim = sum(glim_tps) / len(glim_tps)
long_nemo = next(r["gen_tps"] for r in R["nemotron"] if r["task"] == "long_context_needle")
long_glim = next(r["gen_tps"] for r in R["glimmer"] if r["task"] == "long_context_needle")

cats = ["Short prompt<br>(under 200 tokens)", "Long prompt<br>(~40,000 tokens)"]
fig = go.Figure()
fig.add_bar(
    x=cats, y=[short_nemo, long_nemo], name=NEMO,
    marker=dict(color=BLUE),
    text=[f"{short_nemo:.0f}", f"{long_nemo:.0f}"], textposition="outside",
    textfont=dict(color=INK2, size=13),
)
fig.add_bar(
    x=cats, y=[short_glim, long_glim], name=GLIM,
    marker=dict(color=ORANGE),
    text=[f"{short_glim:.0f}", f"{long_glim:.0f}"], textposition="outside",
    textfont=dict(color=INK2, size=13),
)
fig.update_layout(
    **layout(
        "What a full context window does to speed",
        "Generation tokens/sec with a short prompt vs a ~40K-token prompt loaded",
        height=480,
    ),
    barmode="group",
    bargap=0.35,
    bargroupgap=0.12,
    xaxis=axis(showgrid=False),
    yaxis=axis(title="tokens / second", range=[0, 155]),
)
save(fig, "chart-context-speed.png")

# 3 ---- GPU memory at 64K context
fig = go.Figure()
fig.add_bar(
    y=[NEMO, GLIM], x=[30.9, 17.0], orientation="h",
    marker=dict(color=[BLUE, ORANGE]),
    text=["30.9 GB", "17.0 GB"], textposition="outside",
    textfont=dict(color=INK2, size=13),
    showlegend=False,
)
fig.add_vline(
    x=32.6, line_width=2, line_dash="dot", line_color=MUTED,
    annotation_text="RTX 5090 limit (32.6 GB)",
    annotation_font=dict(color=INK2, size=12),
    annotation_position="top left",
)
fig.update_layout(
    **layout(
        "GPU memory needed with a 64K context",
        "Total VRAM in use while the model is loaded (RTX 5090 has 32.6 GB)",
        height=340,
    ),
    bargap=0.45,
    xaxis=axis(title="GB of VRAM", range=[0, 36]),
    yaxis=axis(showgrid=False),
)
save(fig, "chart-vram.png")

# 4 ---- reasoning verbosity (thinking length per task)
nemo_think = [next(r["thinking_tokens"] for r in R["nemotron"] if r["task"] == t) for t in tasks]
glim_think = [next(r["thinking_tokens"] for r in R["glimmer"] if r["task"] == t) for t in tasks]

fig = go.Figure()
fig.add_bar(
    y=labels, x=nemo_think, name=NEMO, orientation="h",
    marker=dict(color=BLUE),
)
fig.add_bar(
    y=labels, x=glim_think, name=GLIM, orientation="h",
    marker=dict(color=ORANGE),
)
fig.update_layout(
    **layout(
        "How much each model 'thinks' before answering",
        "Length of hidden reasoning per task (words) — both are reasoning models",
        height=560,
    ),
    barmode="group",
    bargap=0.3,
    bargroupgap=0.12,
    xaxis=axis(title="reasoning length (words)"),
    yaxis=axis(autorange="reversed", showgrid=False),
)
save(fig, "chart-thinking.png")

print("all charts done")
