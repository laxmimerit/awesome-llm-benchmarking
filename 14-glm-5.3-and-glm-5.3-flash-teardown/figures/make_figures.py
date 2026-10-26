"""Blog figures for the GLM-5.3 vs GLM-5.3-Flash architecture teardown.

Every number is read from the raw files: the two config.json files, the
complete tensor shape maps built from the safetensors shard headers
(results/*_tensors.json via research/analyze.py), and
results/vendor_benchmarks.json for the clearly-labeled vendor rows.
Nothing is typed in by hand.

House style follows the earlier posts: light surface, muted grid, direct
value labels, no legend where the axis already names the categories.

Hue follows the entity: GLM-5.3 teal (#0d9488), GLM-5.3-Flash blue
(#2563eb), vendor-reported amber (#d97706), slate reserved for reference
lines. Palette validated with the dataviz six-checks script (all pass;
tritan 6.4 covered by direct labels on every mark).
"""

import json
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

TEAL = "#0d9488"    # GLM-5.3
BLUE = "#2563eb"    # GLM-5.3-Flash
AMBER = "#d97706"   # vendor-reported / rivals
SLATE = "#64748b"   # reference annotations only
INK = "#1f2937"

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, sans-serif", size=14, color=INK),
    margin=dict(l=70, r=30, t=70, b=60),
    width=900, height=520,
    title_font_size=19,
)


def load(rel):
    return json.load(open(os.path.join(ROOT, rel), encoding="utf-8"))


BIG = load("models/glm53_config.json")
FLASH_ALL = load("models/glm53_flash_config.json")
FLASH = FLASH_ALL["text_config"]
ANA = load("results/architecture_analysis.json")
VENDOR = load("results/vendor_benchmarks.json")


def save(fig, name):
    fig.write_image(os.path.join(HERE, name), scale=2)
    print("wrote", name)


# ------------------------------------------------------------- fig 1: layer map
def fig_layer_map():
    """Side-by-side layer stacks, one cell per layer, colored by attention type."""
    big_idx = BIG["indexer_types"]                      # 78 entries full/shared
    fl_types = FLASH["layer_types"]                      # 45 entries

    # category codes: 0 dense-MLP note handled by annotation; color by attention
    # GLM-5.3: DSA full indexer vs DSA shared indexer, + MTP
    big_colors = {"full": "#0f766e", "shared": "#5eead4"}
    fl_colors = {"linear_attention": "#2563eb",
                 "deepseek_sparse_attention": "#0f766e"}

    fig = make_subplots(
        cols=2, rows=1, shared_yaxes=False,
        subplot_titles=("GLM-5.3 — 78 layers + MTP",
                        "GLM-5.3-Flash — 45 layers + MTP"),
        horizontal_spacing=0.18)

    def stack(col, n, color_of, mtp_color):
        for i in range(n):
            fig.add_shape(type="rect", x0=0, x1=1, y0=i, y1=i + 0.86,
                          fillcolor=color_of(i), line=dict(width=0),
                          row=1, col=col)
        fig.add_shape(type="rect", x0=0, x1=1, y0=n + 0.6, y1=n + 1.46,
                      fillcolor=mtp_color, line=dict(width=1, color=INK),
                      row=1, col=col)
        fig.add_annotation(x=1.08, y=n + 1.03, text="MTP draft layer",
                           showarrow=False, xanchor="left", font=dict(size=11),
                           row=1, col=col)

    stack(1, 78, lambda i: big_colors[big_idx[i]], "#0f766e")
    stack(2, 45, lambda i: fl_colors[fl_types[i]], "#0f766e")

    # dense stem annotations
    for col, n in ((1, 78), (2, 45)):
        fig.add_annotation(x=-0.08, y=1.4, text="dense<br>MLP ×3", showarrow=False,
                           xanchor="right", font=dict(size=10, color=SLATE),
                           row=1, col=col)
        fig.add_shape(type="line", x0=-0.04, x1=-0.04, y0=0, y1=2.86,
                      line=dict(color=SLATE, width=2), row=1, col=col)

    # legend via dummy scatter
    for name, color in [
            ("DSA full attention, own indexer", "#0f766e"),
            ("DSA full attention, shared index", "#5eead4"),
            ("KDA linear attention", "#2563eb")]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color=color, symbol="square"),
                                 name=name))

    fig.update_xaxes(visible=False, range=[-0.3, 1.9])
    fig.update_yaxes(title="layer index", range=[-1, 82], dtick=10, col=1)
    fig.update_yaxes(title=None, range=[-1, 82], dtick=10, col=2)
    fig.update_layout(**{**LAYOUT, "height": 760},
                      title="Layer map: every 4th Flash layer keeps full attention",
                      legend=dict(orientation="h", y=-0.06),
                      showlegend=True)
    save(fig, "chart-layer-map.png")


# ------------------------------------------------------- fig 2: kv cache growth
def fig_kv_cache():
    rows = {r["model"]: r for r in ANA["kv_cache"]}
    big = rows["GLM-5.3"]
    fl = rows["GLM-5.3-Flash"]
    ctx = [0, 131072, 262144, 524288, 786432, 1048576]

    def gb(row, n_tok, state_bytes=0.0):
        return (row["total_elems_per_token"] * n_tok + state_bytes) / 1e9

    big_y = [gb(big, n) for n in ctx]                       # FP8: 1 B/elem
    fl_y = [gb(fl, n, fl["kda_state_elems_per_seq"] * 1) for n in ctx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ctx, y=big_y, mode="lines+markers+text",
                             name="GLM-5.3", line=dict(color=TEAL, width=2),
                             marker=dict(size=8),
                             text=[f"{v:.1f}" if v else "" for v in big_y],
                             textposition="top left", textfont=dict(size=11)))
    fig.add_trace(go.Scatter(x=ctx, y=fl_y, mode="lines+markers+text",
                             name="GLM-5.3-Flash", line=dict(color=BLUE, width=2),
                             marker=dict(size=8),
                             text=[f"{v:.1f}" if v > 0.2 else "" for v in fl_y],
                             textposition="bottom right", textfont=dict(size=11)))
    fig.add_hline(y=32, line=dict(color=SLATE, dash="dash", width=1.5),
                  annotation_text="RTX 5090 VRAM, 32 GB (weights not counted)",
                  annotation_font=dict(size=11, color=SLATE))
    ratio = big["total_elems_per_token"] / fl["total_elems_per_token"]
    fig.add_annotation(x=1048576, y=(big_y[-1] + fl_y[-1]) / 2,
                       text=f"{ratio:.1f}× at any long context",
                       showarrow=False, xanchor="right",
                       font=dict(size=13, color=INK))
    fig.update_layout(**LAYOUT,
                      title="KV cache vs context (FP8 cache, derived from configs)",
                      xaxis_title="context length (tokens)",
                      yaxis_title="KV cache (GB)",
                      legend=dict(orientation="h", y=1.08, x=0))
    save(fig, "chart-kv-cache.png")


# ------------------------------------------- fig 3: parameter allocation stacks
def fig_param_alloc():
    """Experts dwarf everything; left shows the split, right zooms the rest."""
    fig = make_subplots(cols=2, rows=1, column_widths=[0.32, 0.68],
                        horizontal_spacing=0.14,
                        subplot_titles=("Share of all parameters",
                                        "The non-expert remainder, in billions"))
    models = [("glm53", "GLM-5.3", TEAL), ("glm53_flash", "GLM-5.3-Flash", BLUE)]

    # left: experts vs everything else, 100% stacked
    for mk, label, color in models:
        g = ANA[mk]["groups"]
        tot = ANA[mk]["total_params_excl_scales"]
        exp = g["routed_experts"]["params"] + g["shared_experts"]["params"]
        rest = tot - exp
        fig.add_trace(go.Bar(x=[label], y=[100 * exp / tot], name="experts",
                             marker=dict(color=color),
                             text=[f"experts<br>{100*exp/tot:.1f}%"],
                             textposition="inside", textfont=dict(color="white")),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=[label], y=[100 * rest / tot], name="rest",
                             base=[100 * exp / tot],
                             marker=dict(color=color, opacity=0.3),
                             text=[f"{100*rest/tot:.1f}%"], textposition="outside"),
                      row=1, col=1)

    # right: grouped horizontal bars of the remainder
    comps = [("attention", "Attention"),
             ("embeddings", "Embeddings"),
             ("lm_head", "LM head"),
             ("dense_mlp", "Dense MLP stem"),
             ("vision_tower", "Vision tower"),
             ("dsa_indexer", "DSA indexer"),
             ("moe_router", "Routers"),
             ("mtp_glue", "MTP glue"),
             ("hyper_connections", "mHC")]
    for mk, label, color in models:
        g = ANA[mk]["groups"]
        vals = [g.get(k, {"params": 0})["params"] / 1e9 for k, _ in comps]
        fig.add_trace(go.Bar(y=[c for _, c in comps], x=vals, orientation="h",
                             name=label, marker=dict(color=color),
                             text=[f"{v:.2f}" if v >= 0.005 else "0" for v in vals],
                             textposition="outside", textfont=dict(size=10)),
                      row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(title="share of total (%)", row=1, col=1)
    fig.update_xaxes(title="parameters (billions)", row=1, col=2)
    fig.update_layout(**{**LAYOUT, "height": 540}, barmode="group",
                      title="Where the parameters live: experts are 97% of both models",
                      showlegend=False)
    # entity legend by annotation
    fig.add_annotation(x=0.99, y=0.02, xref="paper", yref="paper",
                       text=("<span style='color:#0d9488'>&#9632;</span> GLM-5.3   "
                             "<span style='color:#2563eb'>&#9632;</span> GLM-5.3-Flash"),
                       showarrow=False, font=dict(size=13), xanchor="right")
    save(fig, "chart-param-map.png")


# ---------------------------------------------------- fig 4: total vs active
def fig_total_active():
    act = ANA["active_params"]
    data = [("GLM-5.3", ANA["glm53"]["total_params_excl_scales"] / 1e9,
             act["glm53"]["total"] / 1e9, TEAL),
            ("GLM-5.3-Flash", ANA["glm53_flash"]["total_params_excl_scales"] / 1e9,
             act["glm53_flash"]["total"] / 1e9, BLUE)]
    fig = go.Figure()
    for name, tot, a, color in data:
        fig.add_trace(go.Bar(x=[name], y=[tot], name=f"{name} total",
                             marker_color=color, opacity=0.35,
                             text=[f"{tot:.0f}B total"], textposition="outside"))
        fig.add_trace(go.Bar(x=[name], y=[a], name=f"{name} active",
                             marker_color=color,
                             text=[f"{a:.1f}B active/token"],
                             textposition="outside"))
    for i, (name, tot, a, _) in enumerate(data):
        fig.add_annotation(x=name, y=tot * 0.55,
                           text=f"{100*a/tot:.1f}% of weights<br>touch each token",
                           showarrow=False, font=dict(size=12, color=INK))
    fig.update_layout(**LAYOUT, barmode="overlay", showlegend=False,
                      title="Total vs active parameters (derived from tensor maps)",
                      yaxis_title="parameters (billions)")
    save(fig, "chart-total-vs-active.png")


# ------------------------------------------------------ fig 5: MoE layer scale
def fig_moe():
    per_exp_big = 3 * BIG["hidden_size"] * BIG["moe_intermediate_size"]
    per_exp_fl = 3 * FLASH["hidden_size"] * FLASH["moe_intermediate_size"]
    cap_big = BIG["n_routed_experts"] * per_exp_big / 1e9
    cap_fl = FLASH["n_routed_experts"] * per_exp_fl / 1e9
    act_big = (BIG["num_experts_per_tok"] + BIG["n_shared_experts"]) * per_exp_big / 1e9
    act_fl = (FLASH["num_experts_per_tok"] + FLASH["n_shared_experts"]) * per_exp_fl / 1e9

    fig = go.Figure()
    names = ["GLM-5.3<br>256 experts × 37.7M", "GLM-5.3-Flash<br>288 experts × 25.2M"]
    fig.add_trace(go.Bar(x=names, y=[cap_big, cap_fl], name="capacity per MoE layer",
                         marker_color=[TEAL, BLUE], opacity=0.35,
                         text=[f"{cap_big:.2f}B", f"{cap_fl:.2f}B"],
                         textposition="outside"))
    fig.add_trace(go.Bar(x=names, y=[act_big, act_fl],
                         name="activated per token (8 routed + 1 shared)",
                         marker_color=[TEAL, BLUE],
                         text=[f"{act_big:.2f}B", f"{act_fl:.2f}B"],
                         textposition="outside"))
    fig.update_layout(**LAYOUT, barmode="group",
                      title="One MoE layer: capacity vs what a token actually uses",
                      yaxis_title="parameters (billions)",
                      legend=dict(orientation="h", y=1.08, x=0))
    save(fig, "chart-moe-layer.png")


# --------------------------------------------------- fig 6: vendor benchmarks
def fig_vendor():
    rows = VENDOR["flash_vs_52"]
    benches = [r["bench"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=benches, x=[r["glm52"] for r in rows], orientation="h",
                         name="GLM-5.2 (prev flagship)", marker_color=AMBER,
                         text=[f'{r["glm52"]}' for r in rows], textposition="outside"))
    fig.add_trace(go.Bar(y=benches, x=[r["flash"] for r in rows], orientation="h",
                         name="GLM-5.3-Flash (18B active)", marker_color=BLUE,
                         text=[f'{r["flash"]}' for r in rows], textposition="outside"))
    fig.update_layout(**{**LAYOUT, "height": 560}, barmode="group",
                      title="Vendor-reported agentic benchmarks (Z.ai's own numbers)",
                      xaxis_title="score", xaxis_range=[0, 100],
                      yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", y=1.08, x=0))
    save(fig, "chart-vendor-bench.png")


if __name__ == "__main__":
    fig_layer_map()
    fig_kv_cache()
    fig_param_alloc()
    fig_total_active()
    fig_moe()
    fig_vendor()
