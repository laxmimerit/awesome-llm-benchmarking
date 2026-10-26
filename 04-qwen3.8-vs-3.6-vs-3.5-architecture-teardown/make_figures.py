"""Report figures, Plotly + kaleido, matching the house style of the earlier
posts in this series (charts.py / arch_chart.py in qwen3.8-vs-nemotron-vs-glimmer).

Palette: validated categorical slots 1-3 from the reference instance (light
surface run passes; aqua's 2.74:1 contrast WARN is relieved by direct labels on
every value-carrying mark, and the blog's tables are the table-view twin).
Hue follows the entity across every figure: 3.5 blue, 3.6 orange, 3.8 aqua.
"""

import json, re, os, statistics as st
from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "blog")
os.makedirs(OUT, exist_ok=True)

C35, C36, C38 = "#2a78d6", "#eb6834", "#1baf7a"
T35, T36, T38 = "rgba(42,120,214,0.22)", "rgba(235,104,52,0.22)", "rgba(27,175,122,0.22)"
MODELS = ["qwen3.5:27b", "qwen3.6:27b", "qwen3.8:latest"]
COLOR = dict(zip(MODELS, [C35, C36, C38]))
TINT = dict(zip(MODELS, [T35, T36, T38]))
SHORT = dict(zip(MODELS, ["Qwen 3.5", "Qwen 3.6", "Qwen 3.8"]))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
TINT_SAME, TINT_DIFF = "#e8f5e8", "#fdeee6"


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
        margin=dict(l=110, r=60, t=100, b=60),
        showlegend=False, barcornerradius=4,
    )
    lay.update(kw)
    return lay


def save(fig, name):
    fig.write_image(os.path.join(OUT, name), scale=2)
    print("saved", name)


# ---------------- data ----------------
def load():
    best = {}
    for l in open(os.path.join(BASE, "results.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        k = (r["key"], r.get("draft"))
        if k not in best or (best[k].get("error") and not r.get("error")):
            best[k] = r
    return [r for r in best.values() if not r.get("error")]


rows = load()


def tps(r):
    if r.get("eval_count") and r.get("eval_duration_ns"):
        return r["eval_count"] / (r["eval_duration_ns"] / 1e9)


def acc_rate(r):
    m = re.search(r"draft acceptance = ([\d.]+)", r.get("mtp_acceptance_tail") or "")
    return float(m.group(1)) if m else None


# ================ chart-architecture ================
SPECS = [
    ("Layers",          ["64", "64", "64 + 1 MTP layer"],          [1, 1, 0]),
    ("Hidden size",     ["5,120", "5,120", "5,120"],               [1, 1, 1]),
    ("Attention heads", ["24 Q / 4 KV", "24 Q / 4 KV", "24 Q / 4 KV"], [1, 1, 1]),
    ("Linear : full attention", ["3 : 1", "3 : 1", "3 : 1"],       [1, 1, 1]),
    ("Vocabulary",      ["248,320", "248,320", "248,320"],         [1, 1, 1]),
    ("Parameters",      ["27,781,427,952", "27,781,427,952", "27.3B + 461M vision"], [1, 1, 0]),
    ("Tensor list",     ["1,307 tensors", "identical to 3.5", "866, vision split out"], [1, 1, 0]),
    ("MTP draft head",  ["present, unused", "present, unused", "present, ACTIVE"], [1, 1, 0]),
]
ylabels = [s[0] for s in SPECS]
xlabels = [SHORT[m] for m in MODELS]
z = [s[2] for s in SPECS]
fig = go.Figure(go.Heatmap(
    z=z, x=xlabels, y=ylabels,
    colorscale=[[0, TINT_DIFF], [1, TINT_SAME]],
    zmin=0, zmax=1, showscale=False, xgap=3, ygap=3, hoverinfo="skip",
))
for yi, (label, vals, same) in enumerate(SPECS):
    for xi, v in enumerate(vals):
        fig.add_annotation(
            x=xlabels[xi], y=label, text=(v if same[xi] else f"<b>{v}</b>"),
            showarrow=False,
            font=dict(size=12, color=INK, family=FONT),
        )
fig.update_xaxes(**axis(showgrid=False, side="top",
                        tickfont=dict(color=INK, size=13)))
fig.update_yaxes(**axis(showgrid=False, autorange="reversed",
                        tickfont=dict(color=INK2, size=12)))
fig.update_layout(**layout(
    "Three releases, one machine",
    "Read straight from the GGUF headers of the installed Ollama blobs. Green = identical to Qwen 3.5; "
    "orange = the only<br>things that changed. 3.5 and 3.6 match tensor-for-tensor; 3.8 keeps the same "
    "backbone and activates the draft head.",
    height=590, width=940, margin=dict(l=190, r=40, t=155, b=30),
))
save(fig, "chart-architecture.png")

# ================ chart-thinking ================
base_t = {m: st.mean(r["eval_count"] for r in rows if r["pass_name"].startswith("A") and r["model"] == m and r["think"] is True) for m in MODELS}
hard_t = {m: st.mean(r["eval_count"] for r in rows if r["pass_name"].startswith("G") and r["model"] == m and str(r["think"]) == "True") for m in MODELS}
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.16,
                    subplot_titles=["Base problems (8, both orders)", "Hard problems (4, both orders)"])
for c, data in [(1, base_t), (2, hard_t)]:
    fig.add_bar(
        y=[SHORT[m] for m in MODELS], x=[data[m] for m in MODELS],
        orientation="h", marker=dict(color=[COLOR[m] for m in MODELS]),
        text=[f"{data[m]:,.0f}" for m in MODELS], textposition="outside",
        textfont=dict(color=INK, size=13, family=FONT), width=0.5,
        showlegend=False, row=1, col=c,
    )
    fig.update_xaxes(**axis(range=[0, max(data.values()) * 1.3], showgrid=True,
                            showticklabels=False), row=1, col=c)
    fig.update_yaxes(**axis(showgrid=False, autorange="reversed",
                            tickfont=dict(color=INK, size=13)), row=1, col=c)
fig.update_layout(**layout(
    "Same accuracy, very different token bills",
    "Average output tokens per solved problem, thinking on, identical sampling, MTP off, forward + reverse "
    "order.<br>Accuracy at these bills: 3.5 went 24/24, 3.6 went 22/24 (both misses were budget cut-offs), 3.8 went 24/24.",
    height=420, width=980, margin=dict(l=90, r=60, t=145, b=40), bargap=0.35,
))
for ann in fig.layout.annotations[:2]:
    ann.font = dict(family=FONT, size=13, color=INK)
save(fig, "chart-thinking.png")

# ================ chart-rumination ================
disc = {"qwen3.5:27b": 12, "qwen3.6:27b": 16, "qwen3.8:latest": 49}
fig = go.Figure()
fig.add_bar(y=[SHORT[m] for m in MODELS], x=[disc[m] for m in MODELS],
            orientation="h", marker=dict(color=[COLOR[m] for m in MODELS]),
            width=0.55)
fig.add_bar(y=[SHORT[m] for m in MODELS], x=[100 - disc[m] for m in MODELS],
            orientation="h", marker=dict(color=[TINT[m] for m in MODELS]),
            width=0.55)
for m in MODELS:
    fig.add_annotation(x=disc[m] + 2, y=SHORT[m],
                       text=f"<b>answer found at {disc[m]}% of the trace</b>",
                       showarrow=False, xanchor="left",
                       font=dict(color=INK, size=13, family=FONT))
fig.update_xaxes(**axis(range=[0, 100], showgrid=True, ticksuffix="%",
                        title_text="position in the thinking trace"))
fig.update_yaxes(**axis(showgrid=False, autorange="reversed",
                        tickfont=dict(color=INK, size=13)))
fig.update_layout(**layout(
    "Where the correct answer first shows up in the thinking",
    "Solid = still searching. Faded = re-checking an answer it already had. "
    "3.5 spends 88% of its trace ruminating;<br>3.8 finds the answer midway "
    "through a trace that is 7x shorter, verifies once, and stops.",
    height=360, barmode="stack", margin=dict(l=90, r=40, t=100, b=60),
))
save(fig, "chart-rumination.png")

# ================ chart-mtp-curve (hero) ================
h = [r for r in rows if r["pass_name"].startswith("H")]
labels = {"H_count": "counting", "H_code": "code", "H_json": "JSON",
          "H_essay": "essay", "H_surreal": "surreal fiction",
          "H_essay_pp": "essay + penalty", "H_code_t02": "code @ temp 0.2"}
pts = defaultdict(list)
for r in h:
    if r["draft"] == 4 and tps(r) and acc_rate(r):
        pts[(r["model"], r["problem_id"])].append((acc_rate(r), tps(r)))
fig = go.Figure()
fig.add_hline(y=79, line=dict(color=MUTED, dash="dash", width=1.5))
fig.add_annotation(x=1.04, y=84, xanchor="right", showarrow=False,
                   text="MTP off: every model sits near 79 tok/s regardless of content",
                   font=dict(color=INK2, size=12, family=FONT))
for m in MODELS:
    xs, ys = [], []
    for (mm, pid), v in pts.items():
        if mm == m:
            xs.append(st.mean(x[0] for x in v)); ys.append(st.mean(x[1] for x in v))
    fig.add_scatter(x=xs, y=ys, mode="markers", name=SHORT[m],
                    marker=dict(color=COLOR[m], size=15,
                                line=dict(color=SURFACE, width=2)))
ANN = [
    ("qwen3.8:latest", "H_count", "counting", 8, 10, "left"),
    ("qwen3.8:latest", "H_code_t02", "code (temp 0.2 and 1.0)", 6, 16, "right"),
    ("qwen3.8:latest", "H_json", "JSON", -10, -6, "right"),
    ("qwen3.8:latest", "H_essay", "essay", -12, 6, "right"),
    ("qwen3.8:latest", "H_surreal", "surreal fiction", 12, 14, "left"),
    ("qwen3.8:latest", "H_essay_pp", "essay + penalty", 12, -12, "left"),
    ("qwen3.5:27b", "H_count", "3.5 / 3.6 counting", -14, 8, "right"),
    ("qwen3.6:27b", "H_code", "3.6 code", 12, 0, "left"),
    ("qwen3.5:27b", "H_code", "3.5 code", 12, 0, "left"),
    ("qwen3.6:27b", "H_essay", "3.5 / 3.6 essays: slower than no MTP at all", 14, -2, "left"),
]
for m, pid, txt, dx, dy, anch in ANN:
    v = pts[(m, pid)]
    a = st.mean(x[0] for x in v); t = st.mean(x[1] for x in v)
    fig.add_annotation(x=a, y=t, text=txt, showarrow=False,
                       xanchor=anch, xshift=dx, yshift=dy,
                       font=dict(color=INK2, size=12, family=FONT))
fig.update_xaxes(**axis(range=[0, 1.06], showgrid=True, tickformat=".1f",
                        title_text="draft acceptance rate (how often the model's own draft guesses are right)"))
fig.update_yaxes(**axis(range=[40, 220], showgrid=True, title_text="decode speed (tok/s)"))
fig.update_layout(**layout(
    "One curve explains every speed number",
    "Each point is one model writing one kind of content, speculative decoding on, 32K context, paired "
    "forward + reverse runs.<br>Speed follows draft acceptance, and acceptance follows how predictable the "
    "writing is. 3.8's retrained head sits right of its siblings everywhere.",
    height=560, width=980, showlegend=True,
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)",
                font=dict(color=INK, size=13)),
    margin=dict(l=80, r=50, t=135, b=70),
))
save(fig, "chart-mtp-curve.png")

# ================ chart-content-speed ================
order = ["H_count", "H_code_t02", "H_code", "H_json", "H_essay", "H_essay_pp", "H_surreal"]
on38, off38 = [], []
for pid in order:
    on38.append(st.mean(tps(r) for r in h if r["model"] == "qwen3.8:latest" and r["problem_id"] == pid and r["draft"] == 4 and tps(r)))
    off38.append(st.mean(tps(r) for r in h if r["model"] == "qwen3.8:latest" and r["problem_id"] == pid and r["draft"] == 0 and tps(r)))
fig = go.Figure()
fig.add_bar(x=[labels[p] for p in order], y=on38, name="MTP on",
            marker=dict(color=C38), text=[f"<b>{v:.0f}</b>" for v in on38],
            textposition="outside", textfont=dict(color=INK, size=13, family=FONT))
fig.add_bar(x=[labels[p] for p in order], y=off38, name="MTP off",
            marker=dict(color=T38), text=[f"{v:.0f}" for v in off38],
            textposition="outside", textfont=dict(color=INK2, size=12, family=FONT))
fig.update_xaxes(**axis(showgrid=False, tickfont=dict(color=INK, size=12)))
fig.update_yaxes(**axis(range=[0, 235], showgrid=True, title_text="decode speed (tok/s)"))
fig.update_layout(**layout(
    "Qwen 3.8 runs 88 to 205 tok/s depending on what it writes",
    "Same model, same GPU, same settings except the content. The speculative decoder profits exactly as "
    "much as the output is predictable.",
    height=450, width=980, showlegend=True, barmode="group", bargap=0.32,
    legend=dict(x=0.85, y=0.98, font=dict(color=INK, size=13)),
    margin=dict(l=80, r=40, t=95, b=70),
))
save(fig, "chart-content-speed.png")

# ================ chart-mtp-353638 ================
cts = ["H_count", "H_code", "H_essay"]
fig = go.Figure()
for m in MODELS:
    vals = [st.mean(tps(r) for r in h if r["model"] == m and r["problem_id"] == pid and r["draft"] == 4 and tps(r)) for pid in cts]
    fig.add_bar(x=["counting", "code", "essay"], y=vals, name=SHORT[m],
                marker=dict(color=COLOR[m]),
                text=[f"<b>{v:.0f}</b>" for v in vals], textposition="outside",
                textfont=dict(color=INK, size=12, family=FONT))
fig.add_hline(y=79, line=dict(color=MUTED, dash="dash", width=1.5))
fig.add_annotation(x=2.35, y=87, showarrow=False, xanchor="right",
                   text="MTP off baseline (~79)",
                   font=dict(color=INK2, size=12, family=FONT))
fig.update_xaxes(**axis(showgrid=False, tickfont=dict(color=INK, size=13)))
fig.update_yaxes(**axis(range=[0, 235], showgrid=True, title_text="decode speed with MTP on (tok/s)"))
fig.update_layout(**layout(
    "All three carry the draft head; only 3.8's earns its keep everywhere",
    "Speculative decoding forced on for every model. On essays, 3.5 and 3.6 fall below their own no-MTP "
    "baseline;<br>3.8 never loses on any content tested.",
    height=440, width=920, showlegend=True, barmode="group",
    legend=dict(x=0.02, y=0.98, font=dict(color=INK, size=13)),
    margin=dict(l=80, r=40, t=130, b=50),
))
save(fig, "chart-mtp-353638.png")

# ================ chart-context ================
d = [r for r in rows if r["pass_name"].startswith("D")]
fills = [16384, 49152, 81920]
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14,
                    subplot_titles=["Decode speed", "Prompt-reading speed"])
for m in MODELS:
    dec, pre = [], []
    for f in fills:
        rs = [r for r in d if r["model"] == m and r["fill"] == f]
        dec.append(st.mean(tps(r) for r in rs if tps(r)))
        pre.append(st.mean(r["prompt_eval_count"] / (r["prompt_eval_duration_ns"] / 1e9) for r in rs if r.get("prompt_eval_duration_ns")))
    fig.add_scatter(x=[16, 48, 80], y=dec, mode="lines+markers", name=SHORT[m],
                    line=dict(color=COLOR[m], width=2.5), marker=dict(size=9),
                    row=1, col=1)
    fig.add_scatter(x=[16, 48, 80], y=pre, mode="lines+markers", name=SHORT[m],
                    line=dict(color=COLOR[m], width=2.5), marker=dict(size=9),
                    showlegend=False, row=1, col=2)
fig.update_xaxes(**axis(tickvals=[16, 48, 80], ticksuffix="K", showgrid=False,
                        title_text="context filled (thousands of tokens)"), row=1, col=1)
fig.update_xaxes(**axis(tickvals=[16, 48, 80], ticksuffix="K", showgrid=False,
                        title_text="context filled (thousands of tokens)"), row=1, col=2)
fig.update_yaxes(**axis(range=[50, 78], showgrid=True, title_text="tok/s"), row=1, col=1)
fig.update_yaxes(**axis(range=[2000, 3600], showgrid=True), row=1, col=2)
fig.update_layout(**layout(
    "Three lines, one curve: identical machines decay identically",
    "Speed vs how full the context window is, thinking on, MTP off. Accuracy stayed 12/12 for every model "
    "at every depth,<br>so the only cost of long context here is time, and it is the same time for all three.",
    height=420, width=980, showlegend=True,
    legend=dict(x=0.02, y=0.05, font=dict(color=INK, size=12)),
    margin=dict(l=80, r=50, t=140, b=60),
))
for ann in fig.layout.annotations[:2]:
    ann.font = dict(family=FONT, size=13, color=INK)
save(fig, "chart-context.png")

# ================ chart-effort ================
e = [r for r in rows if r["pass_name"].startswith("E")]
lv = ["low", "medium", "high", "max"]
vals = [st.mean(r["eval_count"] for r in e if r["think"] == lvl) for lvl in lv]
xcats = ['think:"low"', 'think:"medium"', 'think:"high"', 'think:"max"', "official template, xhigh"]
fig = go.Figure()
fig.add_bar(x=xcats[:4], y=vals, marker=dict(color=C38),
            text=[f"<b>{v:,.0f}</b>" for v in vals], textposition="outside",
            textfont=dict(color=INK, size=13, family=FONT), width=0.55)
fig.add_bar(x=[xcats[4]], y=[22276], marker=dict(color=BASELINE),
            text=["<b>22,276</b>"], textposition="outside",
            textfont=dict(color=INK2, size=13, family=FONT), width=0.55)
fig.add_annotation(x=xcats[4], y=11000, showarrow=False,
                   text="LM Studio, xhigh default<br>(Simon Willison's<br>pelican SVG test)",
                   font=dict(color=INK2, size=11, family=FONT))
fig.update_xaxes(**axis(showgrid=False, tickfont=dict(color=INK, size=12)))
fig.update_yaxes(**axis(showgrid=True, range=[0, 25500],
                        title_text="thinking + answer tokens"))
fig.update_layout(**layout(
    "Ollama's effort dial barely moves, and the real deep-thinking mode is unreachable",
    "Green: Ollama's four think levels, 8 problems each, all answers correct. Ollama injects plain-text "
    "nudges instead of the official<br>template's reasoning_effort variable (verified in its source), so "
    "\"max\" actually shortens thinking. Grey: the regime Ollama cannot reach.",
    height=490, width=940, margin=dict(l=90, r=40, t=135, b=60),
))
save(fig, "chart-effort.png")

# ================ chart-reconciliation ================
hist = [
    ("<b>171</b> tok/s", "Aug 14 report<br>code-heavy suite<br>temp 0.2, MTP n=4", 171, C38),
    ("<b>125.5</b> tok/s", "Aug 16 report<br>llama.cpp ladder<br>MTP n=2", 125.5, C38),
    ("<b>104</b> tok/s", "this report<br>essay prose<br>MTP n=4", 104, C38),
    ("<b>79</b> tok/s", "this report<br>MTP off", 79, BASELINE),
]
fig = go.Figure()
fig.add_bar(x=[hh[1] for hh in hist], y=[hh[2] for hh in hist],
            marker=dict(color=[hh[3] for hh in hist]),
            text=[hh[0] for hh in hist], textposition="outside",
            textfont=dict(color=INK, size=14, family=FONT), width=0.5)
fig.update_xaxes(**axis(showgrid=False, tickfont=dict(color=INK2, size=12)))
fig.update_yaxes(**axis(showgrid=True, range=[0, 200],
                        title_text="Qwen 3.8 decode speed (tok/s)"))
fig.update_layout(**layout(
    "Four sessions, four numbers, zero contradictions",
    "The same model on the same GPU legitimately measures 79 to 190+ tok/s. Every historical figure is one "
    "operating point of one<br>mechanism: base speed multiplied by draft acceptance, which depends on content, "
    "temperature, and settings.",
    height=460, width=920, margin=dict(l=90, r=40, t=130, b=90),
))
save(fig, "chart-reconciliation.png")

# ================ chart-evolution (official card numbers) ================
EVO = [
    ("GPQA Diamond (science reasoning)", [85.5, 87.8, 89.2], "#4a3aa7", ["top center", "top center", "bottom center"], -10),
    ("LiveCodeBench v6 (coding)",        [80.7, 83.9, 90.3], "#2a78d6", ["bottom center", "bottom center", "top center"], 12),
    ("SWE-bench Pro (real repo fixes)",  [51.2, 53.5, 61.7], "#eda100", "top center", 0),
    ("Terminal-Bench (agentic terminal)", [41.6, 59.3, 73.0], "#e34948", "top center", 0),
]
xs = ["Qwen 3.5<br>Feb 2026", "Qwen 3.6<br>Apr 2026", "Qwen 3.8<br>Aug 2026"]
fig = go.Figure()
for name, ys, col, tpos, lshift in EVO:
    fig.add_scatter(x=xs, y=ys, mode="lines+markers+text",
                    line=dict(color=col, width=2.5), marker=dict(size=9),
                    text=[f"{v:g}" for v in ys], textposition=tpos,
                    textfont=dict(color=INK, size=11, family=FONT),
                    name=name, showlegend=False)
    fig.add_annotation(x=2.02, y=ys[-1], text=name, showarrow=False,
                       xanchor="left", yshift=lshift, xshift=26,
                       font=dict(color=col, size=12, family=FONT))
fig.update_xaxes(**axis(showgrid=False, tickfont=dict(color=INK, size=13)))
fig.update_yaxes(**axis(showgrid=True, range=[35, 100], ticksuffix=""))
fig.update_layout(**layout(
    "The official scoreboard: knowledge crawls, agency soars",
    "Scores from the Qwen model cards themselves (their conditions, their template, not this benchmark). "
    "Science reasoning gained<br>3.7 points in six months; agentic terminal work gained 31. The gains "
    "concentrate exactly where thinking efficiency matters most.",
    height=480, width=980, margin=dict(l=70, r=270, t=130, b=60),
))
save(fig, "chart-evolution.png")

print("all figures done")
