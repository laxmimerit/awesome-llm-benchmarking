"""Blog figures for the Qwen 3.8 27B vs Qwen3.8-Flash-Next architecture post.

Everything here is derived from two files pulled off the Hugging Face repos,
saved under ../data/:

  qwen3.8-27b.config.json          / qwen3.8-27b.index.json
  qwen3.8-flash-next.config.json   / qwen3.8-flash-next.index.json

The index files are the safetensors weight maps, so the block census is a count
of real tensor names, not a reading of the model card.

House style follows the earlier posts in this series: light surface, muted
grid, direct value labels on every mark, no legend where the categories are
already named on the axis.

Hue follows the entity: Qwen 3.8 27B orange, Qwen3.8-Flash-Next blue.
"""

import json
import os
import re
from collections import Counter

import plotly.graph_objects as go

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(BASE), "data")
OUT = BASE

C_27, C_FN = "#eb6834", "#2f5fe0"
C_27_L, C_FN_L = "#f6b79a", "#9fb4f2"
C_MUTE = "#b6bcc7"

N_27, N_FN = "Qwen 3.8 27B", "Qwen3.8-Flash-Next"

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, sans-serif", size=15, color="#22252b"),
    margin=dict(l=110, r=50, t=76, b=60),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#fbfbfc",
    showlegend=False,
)


def loadj(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name, w=1000, h=560):
    fig.write_image(os.path.join(OUT, name), width=w, height=h, scale=2)
    print("wrote", name)


CFG_27 = loadj("qwen3.8-27b.config.json")["text_config"]
CFG_FN = loadj("qwen3.8-flash-next.config.json")["text_config"]
IDX_27 = loadj("qwen3.8-27b.index.json")["weight_map"]
IDX_FN = loadj("qwen3.8-flash-next.index.json")["weight_map"]


def census(idx):
    """Count real blocks by the tensor families that only that block type has."""
    gen = Counter(re.sub(r"\.\d+\.", ".N.", k) for k in idx)
    body = {k: v for k, v in gen.items() if k.startswith("model.language_model.layers")}
    linear = max((v for k, v in body.items() if ".linear_attn." in k), default=0)
    full = max((v for k, v in body.items() if ".self_attn.q_proj" in k), default=0)
    mtp = 1 if any(k.startswith("mtp.layers") for k in idx) else 0
    return linear, full, mtp


LIN_27, FULL_27, MTP_27 = census(IDX_27)
LIN_FN, FULL_FN, MTP_FN = census(IDX_FN)

# KV cache per token, f16, from the config head counts.
KV_27 = FULL_27 * 2 * CFG_27["num_key_value_heads"] * CFG_27["head_dim"] * 2
KV_FN = FULL_FN * 2 * CFG_FN["num_key_value_heads"] * CFG_FN["head_dim"] * 2
# QSA also keeps one compressed indexer key per micro-block of 4 tokens.
IDXR_FN = FULL_FN * CFG_FN["indexer_kv_heads"] * CFG_FN["indexer_head_dim"] * 2 \
    / CFG_FN["indexer_compress_ratio"]

# Recurrent Gated DeltaNet state, fp32, constant in context length.
def gdn_state(cfg, n_lin):
    return n_lin * cfg["linear_num_value_heads"] * cfg["linear_key_head_dim"] \
        * cfg["linear_value_head_dim"] * 4


ST_27 = gdn_state(CFG_27, LIN_27)
ST_FN = gdn_state(CFG_FN, LIN_FN)

BUDGET = CFG_FN["indexer_budget"]


def attn_read(ctx, which):
    """KV bytes an attention stack touches to decode one token at this context."""
    if which == "27":
        return ctx * KV_27
    return min(ctx, BUDGET) * KV_FN + ctx * IDXR_FN


# ---------------------------------------------------------------- figure 2
def fig_params():
    fig = go.Figure()
    fig.add_bar(
        x=["core weights", "n-gram table", "MTP head"], y=[125, 51, 4],
        marker_color=[C_FN, C_FN_L, "#5a6270"],
        text=["125B", "51B", "4B"], textposition="outside",
        textfont=dict(size=15),
    )
    fig.add_bar(x=["core weights"], y=[0], marker_color=C_27)
    fig.add_hline(y=27, line=dict(color=C_27, width=2, dash="dot"))
    fig.add_annotation(x=2.42, y=28.4, text=f"<b>{N_27}</b>, all 27B of it live on every token",
                       showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(size=13, color=C_27))
    fig.add_hline(y=6, line=dict(color="#1baf7a", width=2, dash="dot"))
    fig.add_annotation(x=-0.42, y=7.4, text="<b>6B</b> of Flash-Next fires per token",
                       showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=13, color="#1baf7a"))
    fig.update_layout(
        **LAYOUT,
        title=dict(text="Parameters stored against parameters used per token"
                        "<br><sub>Flash-Next stores 180B and runs 6B of it. Qwen 3.8 27B is "
                        "dense, so for it those are the same number.</sub>", x=0.01),
        yaxis=dict(title="parameters (billions)", showgrid=True, gridcolor="#e9ecf1",
                   range=[0, 142]),
        xaxis=dict(title=""),
    )
    save(fig, "chart-params.png", h=520)


# ---------------------------------------------------------------- figure 3
def fig_kv():
    a, b, c = KV_27 / 1024, KV_FN / 1024, IDXR_FN / 1024
    fig = go.Figure()
    fig.add_bar(x=[N_27, N_FN], y=[a, b], marker_color=[C_27, C_FN],
                text=[f"{a:.0f} KB", f"{b:.0f} KB"], textposition="inside",
                insidetextanchor="middle", textfont=dict(color="#fff", size=15))
    fig.add_bar(x=[N_27, N_FN], y=[0, c], marker_color=[C_FN_L, C_FN_L],
                text=["", f"+{c:.2f} KB indexer"], textposition="outside",
                textfont=dict(size=13, color="#43506b"))
    fig.update_layout(
        **LAYOUT, barmode="stack",
        title=dict(text="KV cache stored per token of context"
                        f"<br><sub>Flash-Next charges {(a/(b+c)):.1f}x less: "
                        f"{FULL_FN} attention blocks with {CFG_FN['num_key_value_heads']} "
                        f"KV heads against {FULL_27} blocks with {CFG_27['num_key_value_heads']}. "
                        f"At 262k that is {262144*(KV_FN+IDXR_FN)/2**30:.1f} GB against "
                        f"{262144*KV_27/2**30:.1f} GB.</sub>", x=0.01),
        yaxis=dict(title="KV cache, KB per token (f16)", showgrid=True,
                   gridcolor="#e9ecf1", range=[0, 74]),
        xaxis=dict(title=""),
    )
    save(fig, "chart-kv-per-token.png", h=500)


# ---------------------------------------------------------------- figure 4
def fig_attn_read():
    ctxs = [2 ** k for k in range(10, 21)]
    y27 = [attn_read(c, "27") / 2 ** 20 for c in ctxs]
    yfn = [attn_read(c, "fn") / 2 ** 20 for c in ctxs]
    fig = go.Figure()
    fig.add_scatter(x=ctxs, y=y27, mode="lines+markers", line=dict(color=C_27, width=3),
                    marker=dict(size=7))
    fig.add_scatter(x=ctxs, y=yfn, mode="lines+markers", line=dict(color=C_FN, width=3),
                    marker=dict(size=7))
    fig.add_annotation(x=5.3, y=4.24, text=f"<b>{N_27}</b>", showarrow=False,
                       xanchor="right", font=dict(color=C_27, size=14))
    fig.add_annotation(x=5.3, y=2.42, text=f"<b>{N_FN}</b>", showarrow=False,
                       xanchor="right", font=dict(color=C_FN, size=14))
    r = attn_read(262144, "27") / attn_read(262144, "fn")
    fig.add_annotation(
        x=5.42, y=1.1, xanchor="right",
        text=f"at 262k context:<br>{attn_read(262144,'27')/2**30:.1f} GB against "
             f"{attn_read(262144,'fn')/2**20:.0f} MB<br><b>{r:.0f}x less</b>",
        showarrow=False, align="right", font=dict(size=13, color="#43506b"),
        bgcolor="rgba(255,255,255,0.85)")
    fig.update_layout(
        **LAYOUT,
        title=dict(text="KV cache read to decode one token, by context length"
                        "<br><sub>Flash-Next reads a fixed 2048-token budget plus a cheap "
                        "block index. Qwen 3.8 27B reads the whole cache, every token.</sub>",
                   x=0.01),
        xaxis=dict(title="context length (tokens)", type="log", showgrid=True,
                   gridcolor="#e9ecf1",
                   tickvals=[1024, 8192, 65536, 262144, 1048576],
                   ticktext=["1k", "8k", "64k", "262k", "1M"]),
        yaxis=dict(title="KV bytes read per decoded token (MB)", type="log",
                   showgrid=True, gridcolor="#e9ecf1",
                   tickvals=[1, 10, 100, 1000, 10000, 20000],
                   ticktext=["1 MB", "10", "100", "1 GB", "10 GB", ""]),
    )
    save(fig, "chart-attention-read.png", h=540)


# ---------------------------------------------------------------- figure 5
def fig_traffic():
    """Memory traffic to decode one token at 262k context, both at bf16."""
    ctx = 262144
    w27, wfn = 27e9 * 2, 6e9 * 2
    bars = [
        ("weights read", w27 / 2 ** 30, wfn / 2 ** 30, C_27, C_FN),
        ("recurrent state", ST_27 / 2 ** 30, ST_FN / 2 ** 30, C_27_L, C_FN_L),
        ("attention / KV", attn_read(ctx, "27") / 2 ** 30, attn_read(ctx, "fn") / 2 ** 30,
         "#8a3d1c", "#1b3585"),
    ]
    t27 = sum(b[1] for b in bars)
    tfn = sum(b[2] for b in bars)
    fig = go.Figure()
    for label, v27, vfn, c27, cfn in bars:
        fig.add_bar(y=[N_27, N_FN], x=[v27, vfn], orientation="h",
                    marker_color=[c27, cfn], name=label,
                    text=[f"{label} {v27:.1f}" if v27 > 4 else "",
                          f"{label} {vfn:.1f}" if vfn > 4 else ""],
                    textposition="inside", insidetextanchor="middle",
                    textfont=dict(color="#fff", size=13))
    fig.update_layout(
        **LAYOUT, barmode="stack",
        title=dict(text="Memory traffic to decode one token at 262k context"
                        f"<br><sub>Both at bf16. Flash-Next needs {tfn:.1f} GB per token against "
                        f"{t27:.1f} GB, or {t27/tfn:.1f}x less.</sub>",
                   x=0.01),
        xaxis=dict(title="GB read per decoded token", showgrid=True, gridcolor="#e9ecf1"),
        yaxis=dict(title=""),
    )
    for name, tot in [(N_27, t27), (N_FN, tfn)]:
        fig.add_annotation(x=tot + 1.2, y=name, text=f"<b>{tot:.1f} GB</b>",
                           showarrow=False, xanchor="left",
                           font=dict(size=14, color="#33383f"))
    fig.update_xaxes(range=[0, 78])
    save(fig, "chart-token-traffic.png", h=420)


# ---------------------------------------------------------------- figure 6
BENCH = [
    ("JobBench", "agentic", 55.7, 33.4),
    ("DeepSWE 1.1", "agentic", 58.7, 42.2),
    ("SWE-bench Multilingual", "agentic", 81.0, 73.8),
    ("Agents' Last Exam", "agentic", 51.2, 42.9),
    ("Toolathlon Verified", "agentic", 73.5, 67.1),
    ("NL2Repo-Bench", "agentic", 48.1, 42.3),
    ("HLE", "static", 35.9, 30.8),
    ("CoWorkBench", "agentic", 73.9, 70.7),
    ("GPQA Diamond", "static", 91.7, 89.2),
    ("IFBench", "static", 81.3, 79.5),
    ("LiveCodeBench v6", "static", 91.9, 90.3),
    ("SWE-bench Pro", "agentic", 62.5, 61.7),
]


def fig_bench():
    rows = sorted(BENCH, key=lambda r: r[2] - r[3])
    fig = go.Figure()
    fig.add_bar(
        y=[f"{r[0]}  <span style='color:#9aa1ac'>{r[3]} → {r[2]}</span>" for r in rows],
        x=[r[2] - r[3] for r in rows], orientation="h",
        marker_color=[C_FN if r[1] == "agentic" else C_MUTE for r in rows],
        text=[f"+{r[2]-r[3]:.1f}" for r in rows], textposition="outside",
        textfont=dict(size=13),
    )
    lay = dict(LAYOUT, margin=dict(l=290, r=50, t=76, b=60))
    fig.update_layout(
        **lay,
        title=dict(text="Benchmark points Flash-Next gains over Qwen 3.8 27B"
                        "<br><sub>Qwen's own published numbers. Blue is multi-turn or "
                        "long-horizon work, grey is single-shot.</sub>",
                   x=0.01),
        xaxis=dict(title="points over Qwen 3.8 27B", showgrid=True, gridcolor="#e9ecf1",
                   range=[0, 25]),
        yaxis=dict(title=""),
    )
    save(fig, "chart-benchmark-delta.png", h=620)


# ---------------------------------------------------------------- figure 7
def fig_stack():
    """Draw both stacks block by block, so the interleave is visible."""
    from plotly.subplots import make_subplots

    def strip(n_lin, n_full, interval, accent):
        colors = []
        for i in range(n_lin + n_full):
            full = (i % interval) == interval - 1
            colors.append(accent if full else C_MUTE)
        return colors + ["#5a6270"]

    c27 = strip(LIN_27, FULL_27, CFG_27["full_attention_interval"], C_27)
    cfn = strip(LIN_FN, FULL_FN, CFG_FN["full_attention_interval"], C_FN)

    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.44,
        subplot_titles=(
            N_FN + "   48 blocks, hidden 2560",
            N_27 + "   64 blocks, hidden 5120"))
    for row, colors in [(1, cfn), (2, c27)]:
        fig.add_bar(x=list(range(len(colors))), y=[1] * len(colors),
                    marker_color=colors, marker_line_width=0, width=0.84,
                    row=row, col=1)

    ple = CFG_FN["ple_layer_ids"][0]
    fig.add_annotation(x=ple, y=1.05, row=1, col=1, text="51B n-gram table",
                       showarrow=True, arrowhead=0, arrowwidth=1.3, ax=44, ay=-6,
                       arrowcolor="#8b939f", xanchor="left",
                       font=dict(size=12, color="#43506b"))
    for row, n in [(1, len(cfn)), (2, len(c27))]:
        fig.add_annotation(x=n - 1, y=1.05, row=row, col=1, text="MTP head",
                           showarrow=True, arrowhead=0, arrowwidth=1.3, ax=-6, ay=-6,
                           arrowcolor="#8b939f", xanchor="right",
                           font=dict(size=12, color="#43506b"))

    fig.update_layout(
        **dict(LAYOUT, margin=dict(l=60, r=50, t=112, b=64)),
        title=dict(text="Block layout: what kind of block sits at each position"
                        "<br><sub>Grey blocks hold a fixed-size recurrent state. "
                        "Coloured blocks are full attention with a KV cache.</sub>",
                   x=0.01),
        bargap=0.16,
    )
    fig.update_yaxes(visible=False, range=[0, 1.55])
    fig.update_xaxes(title="block index", showgrid=False, zeroline=False,
                     tickvals=[0, 8, 16, 24, 32, 40, 48, 56, 63])
    for a in fig.layout.annotations[:2]:
        a.update(font=dict(size=14, color="#33383f"), x=0, xanchor="left")
    save(fig, "chart-stack-map.png", h=480)


# ---------------------------------------------------------------- figure 8
def fig_experts():
    """512 experts per block, 11 of them fire."""
    n = CFG_FN["num_experts"]
    cols = 32
    lit = CFG_FN["num_experts_per_tok"]
    xs = [i % cols for i in range(n)]
    ys = [-(i // cols) for i in range(n)]
    routed = {3, 47, 96, 141, 190, 233, 288, 335, 379, 470}
    colors = [C_FN if i in routed else "#dfe3ea" for i in range(n)]
    fig = go.Figure()
    fig.add_scatter(x=xs, y=ys, mode="markers",
                    marker=dict(symbol="square", size=14, color=colors,
                                line=dict(width=0)))
    fig.add_scatter(x=[cols + 2.0], y=[-9.0], mode="markers",
                    marker=dict(symbol="square", size=14, color="#1baf7a"))
    fig.add_annotation(x=cols + 3.0, y=-9.0, text="1 shared expert,<br>always on",
                       showarrow=False, xanchor="left", align="left",
                       font=dict(size=12.5, color="#43506b"))
    fig.add_annotation(x=cols + 3.0, y=-5.6,
                       text="<b>" + str(lit) + " routed experts</b><br>chosen per token",
                       showarrow=False, xanchor="left", align="left",
                       font=dict(size=12.5, color=C_FN))
    fig.update_layout(
        **dict(LAYOUT, margin=dict(l=60, r=50, t=90, b=40)),
        title=dict(text="Experts in one Flash-Next block: " + str(n) + " stored, "
                        + str(lit + 1) + " used per token"
                        "<br><sub>Each expert is only "
                        + str(CFG_FN["moe_intermediate_size"])
                        + " wide against a " + str(CFG_FN["hidden_size"])
                        + " hidden dimension. 2.52B of weights sit in this grid, "
                          "54M of them run.</sub>", x=0.01),
        xaxis=dict(visible=False, range=[-1.5, cols + 12]),
        yaxis=dict(visible=False, range=[-(n // cols) - 0.5, 1],
                   scaleanchor="x", scaleratio=1),
    )
    save(fig, "chart-expert-grid.png", h=430)


# ---------------------------------------------------------------- figure 9
def fig_storage():
    rows = [
        ("bf16, as released", 360.0),
        ("FP8, shipped by Qwen", 180.0),
        ("Unsloth UD-Q4_K_XL", 111.0),
        ("Unsloth UD-IQ1_S,<br>the smallest build that exists", 72.5),
    ][::-1]
    fig = go.Figure()
    fig.add_bar(y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h",
                marker_color=[C_FN, C_FN, C_FN_L, C_FN_L],
                text=["  " + ("%g" % r[1]) + " GB   " + ("%.1f" % (r[1] / 32)) + "x over"
                      for r in rows],
                textposition="outside", textfont=dict(size=14), width=0.62)
    fig.add_vline(x=32, line=dict(color=C_27, width=2.5))
    fig.add_annotation(x=32, y=3.55, text="  32 GB, one RTX 5090", showarrow=False,
                       xanchor="left", font=dict(size=13, color=C_27))
    fig.update_layout(
        **dict(LAYOUT, margin=dict(l=250, r=50, t=90, b=60)),
        title=dict(text="Flash-Next file size at four precisions, against 32 GB of VRAM"
                        "<br><sub>Even the smallest build that exists, a dynamic 1-bit quant, "
                        "is 2.3x too big for one RTX 5090.</sub>", x=0.01),
        xaxis=dict(title="size on disk (GB)", showgrid=True, gridcolor="#e9ecf1",
                   range=[0, 510]),
        yaxis=dict(title=""),
    )
    save(fig, "chart-storage.png", h=440)


# ---------------------------------------------------------------- figure 10
def fig_mtp():
    """Measured on my own rig: what the 27B's MTP head is worth."""
    path = os.path.join(os.path.dirname(os.path.dirname(BASE)),
                        "ornith-benchmark", "results", "mtp_check.json")
    d = json.load(open(path, encoding="utf-8"))["qwen3.8-27b"]
    order = ["counting", "json", "code", "fiction", "essay"]
    off = [d["off"]["runs"][k]["decode_tps"] for k in order]
    on = [d["on"]["runs"][k]["decode_tps"] for k in order]
    acc = [d["on"]["runs"][k]["accept_rate"] for k in order]
    fig = go.Figure()
    fig.add_bar(x=order, y=off, marker_color="#d9dee6",
                text=["%.0f" % v for v in off], textposition="outside",
                textfont=dict(size=13, color="#5f6874"))
    fig.add_bar(x=order, y=on, marker_color=C_27,
                text=["<b>%.0f</b><br><span style='font-size:11px'>%.2f accepted</span>"
                      % (v, a) for v, a in zip(on, acc)],
                textposition="outside", textfont=dict(size=13, color="#8a3d1c"))
    fig.update_layout(
        **dict(LAYOUT, margin=dict(l=90, r=50, t=94, b=60)),
        barmode="group", bargap=0.34, bargroupgap=0.06,
        title=dict(text="Qwen 3.8 27B decode speed with the MTP head off and on"
                        "<br><sub>Our own runs: RTX 5090, Q4_K_M, 512 tokens. Grey is off, "
                        "orange is on. The gain tracks the acceptance rate printed under "
                        "each bar.</sub>", x=0.01),
        yaxis=dict(title="decode, tok/s", showgrid=True, gridcolor="#e9ecf1",
                   range=[0, 235]),
        xaxis=dict(title="content type"),
    )
    save(fig, "chart-mtp-speedup.png", h=510)


if __name__ == "__main__":
    print(f"27B   : {LIN_27} linear + {FULL_27} full + {MTP_27} mtp, "
          f"KV {KV_27/1024:.0f} KB/tok, state {ST_27/2**20:.0f} MB")
    print(f"Flash : {LIN_FN} linear + {FULL_FN} full + {MTP_FN} mtp, "
          f"KV {KV_FN/1024:.0f} + {IDXR_FN/1024:.2f} KB/tok, state {ST_FN/2**20:.0f} MB")
    fig_stack()
    fig_params()
    fig_experts()
    fig_kv()
    fig_attn_read()
    fig_traffic()
    fig_bench()
    fig_storage()
    fig_mtp()
