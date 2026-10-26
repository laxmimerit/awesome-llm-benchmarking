"""Figures for the abliteration teardown post, Plotly + kaleido.

House style carried over from qwen-teardown/make_figures.py: light surface,
900-980px wide, direct labels on every value-carrying mark, no legend where a
direct label will do.

Hue follows the entity: base (stock Qwen 3.8) keeps the series' aqua; the
abliterated build gets violet, a hue not yet spent in this series. Refusal
semantics use their own separate ramp so they never read as an entity colour.
"""

import json, os, sys
import statistics as st
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "blog")
os.makedirs(OUT, exist_ok=True)

# entity hues
CB, CU = "#1baf7a", "#7b3fb5"           # base aqua, uncensored violet
TB, TU = "rgba(27,175,122,0.20)", "rgba(123,63,181,0.20)"

# refusal semantics (deliberately outside the entity ramp)
C_REF, C_COMP, C_TRUNC = "#c0392b", "#4a7fb5", "#e0a030"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
TINT_SAME, TINT_DIFF = "#e8f5e8", "#f3e8fb"


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


# ---------------------------------------------------------------- data
R = [json.loads(l) for l in open(os.path.join(BASE, "results.jsonl"), encoding="utf-8")]
D = [json.loads(l) for l in open(os.path.join(BASE, "diff.jsonl"), encoding="utf-8")]
BYNAME = {r["name"]: r for r in D}

stage = defaultdict(list)
for r in R:
    stage[r["stage"]].append(r)

RESID = ["attn_output", "ssm_out", "ffn_down"]
INPUT_FAMS = {"attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate",
              "ffn_gate", "ffn_up"}
FLOOR = float(np.median([BYNAME[n]["rel_ab"] for n in BYNAME
                         if BYNAME[n].get("fam") in INPUT_FAMS
                         and BYNAME[n].get("rel_ab")]))
THRESH = 8 * FLOOR


# ============================================ 1. architecture identity
SPECS = [
    ("GGUF tensor count",      ["866", "866"],                        [1, 1]),
    ("Tensor names",           ["identical set", "identical set"],    [1, 1]),
    ("Tensor shapes",          ["0 mismatches", "0 mismatches"],      [1, 1]),
    ("Quantization type",      ["Q4_K_M", "Q4_K_M"],                  [1, 1]),
    ("Blocks",                 ["65", "65"],                          [1, 1]),
    ("  full attention",       ["17 (layers 3,7,…,63)", "17 (layers 3,7,…,63)"], [1, 1]),
    ("  Gated DeltaNet",       ["48", "48"],                          [1, 1]),
    ("  MTP draft head",       ["1 (nextn)", "1 (nextn)"],            [1, 1]),
    ("Hidden size",            ["5,120", "5,120"],                    [1, 1]),
    ("Total tensor bytes",     ["identical", "identical"],            [1, 1]),
    ("general.name",           ["Qwen3.8 27B", "Qwen3.8 27B Abliterated"], [1, 0]),
    ("File size",              ["16,810,714,464 B", "16,810,714,496 B"],  [1, 0]),
    ("Weight values",          ["unchanged", "131 tensors rewritten"], [1, 0]),
]
xlabels = ["Stock Qwen 3.8", "Uncensored build"]
ylabels = [s[0] for s in SPECS]
z = [s[2] for s in SPECS]
fig = go.Figure(go.Heatmap(
    z=z, x=xlabels, y=ylabels,
    colorscale=[[0, TINT_DIFF], [1, TINT_SAME]],
    zmin=0, zmax=1, showscale=False, xgap=3, ygap=3, hoverinfo="skip",
))
for yi, (label, vals, same) in enumerate(SPECS):
    for xi, v in enumerate(vals):
        fig.add_annotation(x=xlabels[xi], y=label,
                           text=(v if same[xi] else f"<b>{v}</b>"),
                           showarrow=False,
                           font=dict(size=12, color=INK, family=FONT))
fig.update_xaxes(**axis(showgrid=False, side="top",
                        tickfont=dict(color=INK, size=13)))
fig.update_yaxes(**axis(showgrid=False, autorange="reversed",
                        tickfont=dict(color=INK2, size=12)))
fig.update_layout(**layout(
    "GGUF Header Comparison: Stock vs Uncensored Qwen 3.8 27B",
    "Green marks fields that are identical between the two files; violet marks the three that differ.",
    height=660, width=940, margin=dict(l=215, r=40, t=105, b=30),
))
save(fig, "chart-architecture.png")


# ============================================ 2. which tensors moved
GROUPS = [
    ("attn_output",      "attn_output",   2),
    ("ssm_out",          "ssm_out",       2),
    ("ffn_down",         "ffn_down",      2),
    ("token_embd",       "token_embd.weight", 1),
    ("ffn_gate / ffn_up", ("ffn_gate", "ffn_up"), 0),
    ("attn_q / k / v",   ("attn_q", "attn_k", "attn_v"), 0),
    ("attn_qkv",         "attn_qkv",      0),
    ("ssm_alpha / beta", ("ssm_alpha", "ssm_beta"), 0),
    ("output head",      "output.weight", 0),
    ("all norms, ssm_a,<br>ssm_conv1d, biases", "__zero__", 0),
]


def famvals(sel, key):
    if sel == "__zero__":
        return [0.0]
    sel = (sel,) if isinstance(sel, str) else sel
    return [r[key] for r in D if r["fam"] in sel and r.get(key) is not None]


labels, v_ab, v_ac, is_res = [], [], [], []
for lbl, sel, res in GROUPS:
    a = famvals(sel, "rel_ab")
    c = famvals(sel, "rel_ac")
    labels.append(lbl)
    v_ab.append(float(np.median(a)) if a else 0.0)
    v_ac.append(float(np.median(c)) if c else 0.0)
    is_res.append(res)

fig = go.Figure()
fig.add_bar(y=labels, x=v_ac, orientation="h", width=0.36, offset=-0.38,
            marker=dict(color="#cfcec6"),
            text=[f"{v:.3f}" for v in v_ac], textposition="outside",
            textfont=dict(color=MUTED, size=11, family=FONT))
fig.add_bar(y=labels, x=v_ab, orientation="h", width=0.36, offset=0.02,
            marker=dict(color=[{2: CU, 1: "#b98fdb", 0: "#b9b8b0"}[r]
                               for r in is_res]),
            text=[f"<b>{v:.3f}</b>" if r else f"{v:.3f}"
                  for v, r in zip(v_ab, is_res)],
            textposition="outside",
            textfont=dict(color=INK, size=11, family=FONT))
fig.add_vline(x=FLOOR, line=dict(color=BASELINE, width=1.5, dash="dot"))
# rows 8 and 9 carry no visible bars, so the callout sits in real whitespace
fig.add_annotation(x=0.020, y=8.55, xanchor="left", yanchor="middle",
                   ax=FLOOR, ay=8.55, axref="x", ayref="y",
                   text=f"dotted line: same-quantizer floor {FLOOR:.4f}",
                   showarrow=True, arrowhead=0, arrowwidth=1,
                   arrowcolor=BASELINE,
                   font=dict(color=MUTED, size=11, family=FONT))
fig.update_xaxes(**axis(range=[0, 0.135], showgrid=True,
                        title_text="relative change  ‖ΔW‖ / ‖W‖"))
fig.update_yaxes(**axis(showgrid=False, autorange="reversed",
                        tickfont=dict(color=INK, size=12)))
fig.update_layout(**layout(
    "Relative Weight Change by Tensor Family",
    "Violet is the abliteration delta; grey is a different-quantizer control on the same base weights.",
    height=560, width=980, margin=dict(l=190, r=90, t=95, b=60), bargap=0.28,
))
save(fig, "chart-weight-diff.png")


# ============================================ 3. layer profile
COLS = {"attn_output": "#2a78d6", "ssm_out": CU, "ffn_down": "#eb6834"}
TITLES = {
    "attn_output": "attn_output   ·   17 full-attention blocks",
    "ssm_out": "ssm_out   ·   48 Gated DeltaNet blocks",
    "ffn_down": "ffn_down   ·   all 65 MLP blocks",
}
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.085,
                    subplot_titles=[TITLES[f] for f in RESID])
for i, f in enumerate(RESID, start=1):
    rows_f = sorted([r for r in D if r["fam"] == f], key=lambda r: r["layer"])
    fig.add_scatter(x=[r["layer"] for r in rows_f], y=[r["rel_ab"] for r in rows_f],
                    mode="markers", marker=dict(color=COLS[f], size=7,
                                                line=dict(width=0)),
                    row=i, col=1)
    fig.add_hline(y=FLOOR, line=dict(color=BASELINE, width=1.2, dash="dot"),
                  row=i, col=1)
    fig.update_yaxes(**axis(showgrid=True, range=[0, 0.075],
                            tickvals=[0, 0.02, 0.04, 0.06],
                            tickfont=dict(color=MUTED, size=11)), row=i, col=1)
fig.add_annotation(x=65.5, y=FLOOR, xanchor="left", yanchor="bottom",
                   text=f"untouched-tensor<br>floor {FLOOR:.4f}", showarrow=False,
                   align="left",
                   font=dict(color=MUTED, size=10.5, family=FONT), row=1, col=1)
fig.add_annotation(x=31, y=0.0685, xanchor="center",
                   text="<b>blocks 8 to 54, two of every three</b>", showarrow=False,
                   font=dict(color="#a8461a", size=11, family=FONT), row=3, col=1)
fig.add_annotation(x=3.5, y=0.0115, xanchor="center",
                   text="first 8", showarrow=False,
                   font=dict(color=MUTED, size=10.5, family=FONT), row=3, col=1)
fig.add_annotation(x=59.5, y=0.0115, xanchor="center",
                   text="last 10", showarrow=False,
                   font=dict(color=MUTED, size=10.5, family=FONT), row=3, col=1)
fig.update_xaxes(**axis(showgrid=True, range=[-1.5, 72],
                        title_text="block index (0 to 64)"), row=3, col=1)
for i in (1, 2):
    fig.update_xaxes(**axis(showgrid=True, range=[-1.5, 72],
                            showticklabels=False), row=i, col=1)
fig.update_layout(**layout(
    "Relative Weight Change by Block Index, per Residual-Writing Family",
    "One dot per tensor across the 65 blocks; ffn_down alternates between two cut strengths.",
    height=640, width=980, margin=dict(l=90, r=175, t=118, b=55),
))
for ann in fig.layout.annotations[:3]:
    ann.font = dict(family=FONT, size=12.5, color=INK)
    ann.x = 0.0
    ann.xanchor = "left"
save(fig, "chart-layer-profile.png")


# ============================================ 4. the direction (decisive)
dirs = np.load(os.path.join(BASE, "directions.npz"))
r_cons = np.load(os.path.join(BASE, "refusal_direction.npy"))


def unitvecs(names):
    return np.stack([dirs[n] / np.linalg.norm(dirs[n]) for n in names])


# Grouped by ROLE, not by delta size: a residual writer can carry the ablation,
# an input projection cannot. (A magnitude threshold wrongly drops the 33
# ffn_down tensors whose edit is small but aligns at |cos| 0.9999.)
resid = sorted([n for n in dirs.files
                if BYNAME.get(n, {}).get("fam") in set(RESID)
                and np.linalg.norm(dirs[n]) > 0],
               key=lambda n: (BYNAME[n]["fam"], BYNAME[n]["layer"]))
edited = resid + ["token_embd.weight"]
control = sorted([n for n in dirs.files
                  if BYNAME.get(n, {}).get("fam") in INPUT_FAMS
                  and np.linalg.norm(dirs[n]) > 0],
                 key=lambda n: (BYNAME[n]["fam"], BYNAME[n]["layer"]))

cos_e = np.abs(unitvecs(edited) @ r_cons)
cos_c = np.abs(unitvecs(control) @ r_cons)
RAND = float(np.sqrt(2 / (np.pi * 5120)))

rng = np.random.default_rng(7)
fig = go.Figure()
fig.add_scatter(x=cos_c, y=1 + rng.uniform(-0.24, 0.24, len(cos_c)),
                mode="markers",
                marker=dict(color="#b9b8b0", size=6, opacity=0.5,
                            line=dict(width=0)))
fig.add_scatter(x=cos_e, y=0 + rng.uniform(-0.24, 0.24, len(cos_e)),
                mode="markers",
                marker=dict(color=CU, size=8, opacity=0.8,
                            line=dict(width=0)))
fig.add_vline(x=RAND, line=dict(color=BASELINE, width=1.5, dash="dot"))
fig.add_annotation(x=RAND, y=0.62, xanchor="left", xshift=9, yanchor="middle",
                   text=f"random chance for a 5,120-dim space = {RAND:.4f}",
                   showarrow=False,
                   font=dict(color=MUTED, size=11, family=FONT))
fig.add_annotation(x=1.0, y=-0.46, xanchor="right",
                   text=f"<b>{len(edited)} tensors that write into the residual stream</b><br>"
                        f"mean |cos| {cos_e.mean():.4f}    worst case {cos_e.min():.4f}",
                   showarrow=False, align="right",
                   font=dict(color=CU, size=12.5, family=FONT))
fig.add_annotation(x=0.075, y=1.45, xanchor="left", yanchor="middle",
                   text=f"<b>{len(control)} input projections</b>  mean |cos| {cos_c.mean():.4f}"
                        f"  =  {cos_c.mean()/RAND:.2f}× chance",
                   showarrow=False,
                   font=dict(color=INK2, size=12.5, family=FONT))
fig.update_xaxes(**axis(range=[-0.03, 1.08], showgrid=True,
                        title_text="|cosine| between this tensor's leading change vector and the consensus direction"))
fig.update_yaxes(**axis(showgrid=False, range=[-0.85, 1.75],
                        tickmode="array", tickvals=[0, 1],
                        ticktext=["writes to<br>residual stream", "input<br>projection"],
                        tickfont=dict(color=INK, size=12)))
fig.update_layout(**layout(
    "Cosine Alignment of Each Tensor Change Vector to the Consensus Direction",
    "One dot per tensor, grouped by whether it writes into the residual stream.",
    height=430, width=980, margin=dict(l=150, r=60, t=95, b=70),
))
save(fig, "chart-direction.png")


# ============================================ 5. per-family alignment
fam_align = defaultdict(list)
for n, a_ in zip(edited, cos_e):
    fam_align[BYNAME[n]["fam"]].append(a_)
rows = [("attn_output", "output projection of the 17 full-attention blocks"),
        ("ssm_out", "output projection of the 48 Gated DeltaNet blocks"),
        ("ffn_down", "down projection of all 65 MLP blocks"),
        ("token_embd.weight", "the input embedding matrix")]
ys = [f"{f.replace('.weight','')}<br><span style='font-size:11px;color:{MUTED}'>{d}</span>"
      for f, d in rows]
vals = [float(np.mean(fam_align[f])) for f, _ in rows]
ns = [len(fam_align[f]) for f, _ in rows]
ys.append(f"input projections<br><span style='font-size:11px;color:{MUTED}'>"
          f"q, k, v, qkv, gate, ffn_gate, ffn_up: cannot carry an ablation</span>")
vals.append(float(cos_c.mean()))
ns.append(len(control))

fig = go.Figure()
fig.add_bar(y=ys, x=vals, orientation="h", width=0.52,
            marker=dict(color=[CU, CU, CU, "#b98fdb", "#cfcec6"]),
            text=[f"<b>{v:.4f}</b>   (n={n})" if i < 4 else f"{v:.4f}   (n={n})"
                  for i, (v, n) in enumerate(zip(vals, ns))],
            textposition="outside",
            textfont=dict(color=INK, size=13, family=FONT))
fig.add_vline(x=RAND, line=dict(color=BASELINE, width=1.5, dash="dot"))
# sits in the clear band below the last bar, above the axis line
fig.add_annotation(x=0.02, y=4.44, xanchor="left", yanchor="middle",
                   text=f"dotted line: random chance {RAND:.4f}", showarrow=False,
                   font=dict(color=MUTED, size=11, family=FONT))
fig.update_xaxes(**axis(range=[0, 1.25], showgrid=True,
                        title_text="mean |cos| to the consensus refusal direction"))
fig.update_yaxes(**axis(showgrid=False, autorange="reversed",
                        tickfont=dict(color=INK, size=13)))
fig.update_layout(**layout(
    "Mean Cosine Alignment to the Refusal Direction by Tensor Family",
    "Four unrelated tensor types share one direction; input projections stay at random chance.",
    height=480, width=980, margin=dict(l=330, r=100, t=95, b=70), bargap=0.35,
))
save(fig, "chart-family-alignment.png")


# ============================================ 6. the truncation trap
trap = {
    ("base", "reported"): 6, ("base", "real"): 6,
    ("unc", "reported"): 11, ("unc", "real"): 0,
}
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.18, subplot_titles=[
    "What the 2,048-token run reported", "What the transcripts actually contained"])
for c, key in [(1, "reported"), (2, "real")]:
    vals = [trap[("base", key)], trap[("unc", key)]]
    fig.add_bar(y=["stock Qwen 3.8", "Uncensored build"], x=vals, orientation="h",
                marker=dict(color=[CB, CU]), width=0.5,
                text=[f"<b>{v}/15</b>" for v in vals], textposition="outside",
                textfont=dict(color=INK, size=14, family=FONT),
                row=1, col=c)
    fig.update_xaxes(**axis(range=[0, 15], showgrid=True, showticklabels=False),
                     row=1, col=c)
    fig.update_yaxes(**axis(showgrid=False, autorange="reversed",
                            tickfont=dict(color=INK, size=13)), row=1, col=c)
fig.add_annotation(x=0.5, y=-0.20, xref="x domain", yref="y domain",
                   text="<i>\"the uncensored model refuses almost twice as often\"</i>",
                   showarrow=False, font=dict(color=C_REF, size=12, family=FONT),
                   row=1, col=1)
fig.add_annotation(x=0.5, y=-0.20, xref="x2 domain", yref="y2 domain",
                   text="<i>all 11 were empty answers: the model ran out of tokens mid-recipe</i>",
                   showarrow=False, font=dict(color=INK2, size=12, family=FONT),
                   row=1, col=2)
fig.update_layout(**layout(
    "Harmful-Prompt Refusals: Classifier Output vs Transcript Contents",
    "Out of 15 harmful prompts, at a 2,048-token budget that truncated the uncensored answers.",
    height=420, width=980, margin=dict(l=140, r=60, t=120, b=80), bargap=0.4,
))
for ann in fig.layout.annotations[:2]:
    ann.font = dict(family=FONT, size=13, color=INK)
save(fig, "chart-truncation-trap.png")


# ============================================ 7. rumination cliff
fig = go.Figure()
for arm, col in [("base", CB), ("unc", CU)]:
    it = [r for r in stage["safety_nothink"]
          if r["arm"] == arm and r["kind"] == "harmful"]
    ys = sorted(r.get("think_chars", 0) for r in it)
    xs = list(range(1, len(ys) + 1))
    fig.add_scatter(x=xs, y=ys, mode="lines+markers", name=arm,
                    line=dict(color=col, width=2.5),
                    marker=dict(color=col, size=8))
fig.add_hrect(y0=4400, y1=8200, fillcolor="rgba(224,160,48,0.13)",
              line_width=0)
fig.add_annotation(x=1.2, y=7700, xanchor="left",
                   text="<b>above ≈4,400 chars the 1,536-token budget runs out</b><br>"
                        "answer comes back empty and gets logged as truncated",
                   showarrow=False, font=dict(color="#8a6410", size=11, family=FONT))
fig.add_annotation(x=15, y=7172, text="<b>Uncensored</b>", showarrow=False,
                   xanchor="right", yshift=18,
                   font=dict(color=CU, size=13, family=FONT))
fig.add_annotation(x=13.2, y=1430, text="<b>stock Qwen 3.8</b>", showarrow=False,
                   xanchor="right", yshift=-24,
                   font=dict(color=CB, size=13, family=FONT))
fig.update_xaxes(**axis(showgrid=True, title_text="the 15 harmful prompts, sorted by how long the model thought",
                        tickvals=list(range(1, 16))))
fig.update_yaxes(**axis(showgrid=True, title_text="thinking characters", range=[0, 8200]))
fig.update_layout(**layout(
    "Thinking Length per Harmful Prompt With the /no_think Suffix",
    "The 15 harmful prompts sorted by thinking length; neither model honours the suffix.",
    height=520, width=980, margin=dict(l=100, r=60, t=100, b=60),
))
save(fig, "chart-rumination.png")


# ============================================ 8. refusal outcome
def counts(stg, arm, kind):
    """Count by the three-way `verdict`, never by the boolean flags.

    `truncated` is set whenever finish == length, which also fires on a
    substantive answer that was merely clipped at the end. Three such rows
    carry complied=True AND truncated=True, so summing the flags overcounts.
    `verdict` is the single authoritative label and always sums to n.
    """
    it = [r for r in stage[stg] if r["arm"] == arm and r["kind"] == kind]
    v = [r["verdict"] for r in it]
    return (v.count("refuse"), v.count("comply"), v.count("truncated"))


ROWS = [
    ("harmful  ·  stock Qwen 3.8", "safety_nothink", "base", "harmful"),
    ("harmful  ·  Uncensored",     "safety_nothink", "unc",  "harmful"),
    ("benign  ·  stock Qwen 3.8",  "safety_nothink", "base", "benign"),
    ("benign  ·  Uncensored",      "safety_nothink", "unc",  "benign"),
]
ys = [r[0] for r in ROWS]
refs = [counts(r[1], r[2], r[3])[0] for r in ROWS]
comps = [counts(r[1], r[2], r[3])[1] for r in ROWS]
truncs = [counts(r[1], r[2], r[3])[2] for r in ROWS]

fig = go.Figure()
for vals, col, nm in [(refs, C_REF, "refused"),
                      (comps, C_COMP, "complied"),
                      (truncs, C_TRUNC, "truncated")]:
    fig.add_bar(y=ys, x=vals, orientation="h", marker=dict(color=col),
                width=0.52, name=nm,
                text=[f"<b>{v}</b>" if v else "" for v in vals],
                textposition="inside", insidetextanchor="middle",
                textfont=dict(color="#ffffff", size=13, family=FONT))
fig.update_layout(barmode="stack")
for i, (r, c, t) in enumerate(zip(refs, comps, truncs)):
    fig.add_annotation(x=r + c + t + 0.45, y=ys[i], xanchor="left",
                       text=f"{r} refused · {c} complied" + (f" · {t} truncated" if t else ""),
                       showarrow=False,
                       font=dict(color=INK2, size=11, family=FONT))
for lx, ltxt, lcol in [(0.0, "refused", C_REF), (0.11, "complied", C_COMP),
                       (0.235, "truncated (inconclusive)", "#8a6410")]:
    fig.add_annotation(x=lx, y=-0.13, xref="paper", yref="paper", xanchor="left",
                       text=f"<b>{ltxt}</b>", showarrow=False,
                       font=dict(color=lcol, size=12, family=FONT))
fig.update_xaxes(**axis(range=[0, 25], showgrid=True, showticklabels=False))
fig.update_yaxes(**axis(showgrid=False, autorange="reversed",
                        tickfont=dict(color=INK, size=12)))
fig.update_layout(**layout(
    "Refused, Complied and Truncated Outcomes per Prompt Set",
    "15 prompts per set, counted by final verdict on the answer the user receives.",
    height=460, width=980, margin=dict(l=210, r=190, t=100, b=85),
    showlegend=False,
))
save(fig, "chart-refusal.png")


# ============================================ 9. cost of the edit
q = {a: [r for r in stage["quality"] if r["arm"] == a] for a in ("base", "unc")}
ppl = {r["arm"]: r for r in stage["ppl"]}
sp = lambda a, mtp, c: st.mean(
    x["client_tps"] for x in stage["speed"]
    if x["arm"] == a and x["mtp"] == mtp and x["content"] == c)

fig = make_subplots(
    rows=1, cols=3, horizontal_spacing=0.10,
    subplot_titles=["Auto-graded tasks solved", "Perplexity, wikitext-2",
                    "Decode speed, tok/s"])

# panel 1: quality
for i, a in enumerate(("base", "unc")):
    ok = sum(x["correct"] for x in q[a])
    fig.add_bar(x=[["Stock", "Uncensored"][i]], y=[ok],
                marker=dict(color=[CB, CU][i]), width=0.5,
                text=[f"<b>{ok}/20</b>"], textposition="outside",
                textfont=dict(color=INK, size=14, family=FONT), row=1, col=1)
fig.update_yaxes(**axis(range=[0, 25], showgrid=True, showticklabels=False), row=1, col=1)

# panel 2: ppl
for i, a in enumerate(("base", "unc")):
    v = ppl[a]["ppl"]
    fig.add_bar(x=[["Stock", "Uncensored"][i]], y=[v],
                marker=dict(color=[CB, CU][i]), width=0.5,
                error_y=dict(type="data", array=[ppl[a]["ppl_err"]],
                             color=INK2, thickness=1.4, width=6),
                textfont=dict(color=INK, size=14, family=FONT), row=1, col=2)
    fig.add_annotation(x=["Stock", "Uncensored"][i], y=v + ppl[a]["ppl_err"] + 0.022,
                       text=f"<b>{v:.3f}</b>", showarrow=False, yanchor="bottom",
                       font=dict(color=INK, size=14, family=FONT), row=1, col=2)
fig.update_yaxes(**axis(range=[6.5, 7.15], showgrid=True,
                        tickfont=dict(color=MUTED, size=11)), row=1, col=2)

# panel 3: speed
cfgs = [("MTP off", False), ("MTP on", True)]
for i, a in enumerate(("base", "unc")):
    vals = [sp(a, m, "code") for _, m in cfgs]
    fig.add_bar(x=[c for c, _ in cfgs], y=vals, name=a,
                marker=dict(color=[CB, CU][i]), width=0.32,
                offset=[-0.34, 0.02][i],
                text=[f"{v:.0f}" for v in vals], textposition="outside",
                textfont=dict(color=INK, size=12, family=FONT), row=1, col=3)
fig.update_yaxes(**axis(range=[0, 140], showgrid=True,
                        tickfont=dict(color=MUTED, size=11)), row=1, col=3)

for c in (1, 2, 3):
    fig.update_xaxes(**axis(showgrid=False, tickfont=dict(color=INK, size=12)),
                     row=1, col=c)
fig.update_layout(**layout(
    "Task Accuracy, Perplexity and Decode Speed: Stock vs Uncensored",
    "Aqua is the stock model, violet the uncensored build; bars show the mean of 3 runs.",
    height=450, width=980, margin=dict(l=70, r=50, t=125, b=50), bargap=0.35,
))
for ann in fig.layout.annotations[:3]:
    ann.font = dict(family=FONT, size=13, color=INK)
save(fig, "chart-cost.png")

print("\nall figures written to", OUT)
