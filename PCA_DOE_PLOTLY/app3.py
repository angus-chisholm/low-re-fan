"""
app.py  —  PCA + GP Surrogate · DoE Compressor Tool  (web edition)
===================================================================
Plotly Dash replacement for the matplotlib GUI.

Run:
    python app.py
Then open  http://localhost:8050

Directory layout expected (same as original):
    doe_params.csv
    stl_files/DOE_*.stl
    data/doe_data/*.csv
    audio/doe_data/*.wav   (optional — used by future audio tab)
"""

import io, os, csv, warnings
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, ALL

warnings.filterwarnings("ignore")

# ─── Try to import core; fall back gracefully if data files are missing ───────
try:
    from pca_core import (
        load_data, remove_outliers_from_smoothing, remove_outliers_redo_PCA, calculate_oaspl,
        PCAModel, GPSurrogates,
        PARAM_BOUNDS, PARAM_NAMES, PARAM_KEYS, PARAM_LO, PARAM_HI, PARAM_DEFAULTS,
        PHI_COMMON, N_PHI, CENTRE_FREQS
    )
    _HAVE_CORE = True
except Exception as e:
    print(f"[warn] pca_core import: {e}")
    _HAVE_CORE = False

# ── Fallback stubs for UI preview without data ────────────────────────────────
if not _HAVE_CORE:
    PARAM_BOUNDS = {
        "Mass Flow Rate":    ("mdot",           0.025, 0.075, 0.050),
        "De Haller Midspan": ("DH_mid",         0.75,  0.90,  0.825),
        "Incidence Angle":   ("incidence",     -10.0,  5.0,  -2.5),
        "Vortex Exponent":   ("Vexp",          -2.0,   0.0,  -1.0),
        "Compound Lean Max": ("lean_compound",  0.0,   10.0,   5.0),
        "Straight Lean":     ("lean_straight",  0.0,   90.0,  45.0),
        "Tip Clearance %":   ("tip_clearance",  1.0,    5.0,   3.0),
        "Blade Count":       ("n_blade",        5,      9,     7),
    }
    PARAM_NAMES    = list(PARAM_BOUNDS.keys())
    PARAM_KEYS     = [v[0] for v in PARAM_BOUNDS.values()]
    PARAM_LO       = np.array([v[1] for v in PARAM_BOUNDS.values()], dtype=float)
    PARAM_HI       = np.array([v[2] for v in PARAM_BOUNDS.values()], dtype=float)
    PARAM_DEFAULTS = np.array([v[3] for v in PARAM_BOUNDS.values()], dtype=float)
    N_PHI          = 100
    PHI_COMMON     = np.linspace(0.05, 0.25, N_PHI)

# ─────────────────────────────────────────────────────────────────────────────
#  Light design tokens
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":          "#f4f6f9",
    "surface":     "#ffffff",
    "surface2":    "#f0f4f8",
    "border":      "#d0dae8",
    "border2":     "#b8c8db",
    "text":        "#1a2332",
    "text2":       "#4a5d72",
    "text3":       "#8099b3",
    "accent_psi":  "#d4183a",
    "accent_eta":  "#6d28d9",
    "cyan":        "#0369a1",
    "amber":       "#b45309",
    "green":       "#047857",
    "red":         "#b91c1c",
    "pink":        "#be185d",
    "grey_line":   "#94a3b8",
    "shadow":      "rgba(15,30,60,0.07)",
    "band_psi":    "rgba(212,24,58,0.10)",
    "band_psi2":   "rgba(212,24,58,0.22)",
    "band_eta":    "rgba(109,40,217,0.10)",
    "band_eta2":   "rgba(109,40,217,0.22)",
    "accent_audio":"#0e7490",
    "band_audio":  "rgba(14,116,144,0.10)",
    "band_audio2": "rgba(14,116,144,0.22)",
}

MCOLORS = [C["cyan"], C["amber"], C["green"], C["pink"], C["red"], C["accent_eta"]]

PLOT_LAYOUT_BASE = dict(
    paper_bgcolor=C["surface"],
    plot_bgcolor=C["surface2"],
    font=dict(family="'IBM Plex Mono', 'Fira Code', 'Courier New', monospace",
              size=11, color=C["text"]),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=C["border"],
                borderwidth=1, font=dict(size=10)),
    xaxis=dict(gridcolor=C["border"], linecolor=C["border2"],
               tickcolor=C["border2"], zerolinecolor=C["border"]),
    yaxis=dict(gridcolor=C["border"], linecolor=C["border2"],
               tickcolor=C["border2"], zerolinecolor=C["border"]),
)

PLOT_LAYOUT_LOG = dict(
    paper_bgcolor=C["surface"],
    plot_bgcolor=C["surface2"],
    font=dict(family="'IBM Plex Mono', 'Fira Code', 'Courier New', monospace",
              size=11, color=C["text"]),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=C["border"],
                borderwidth=1, font=dict(size=10)),
    yaxis=dict(gridcolor=C["border"], linecolor=C["border2"],
               tickcolor=C["border2"], zerolinecolor=C["border"]),
)

# PLOT_LAYOUT_LOG = PLOT_LAYOUT_BASE.pop("xaxis")
# print(PLOT_LAYOUT_LOG, PLOT_LAYOUT_BASE)

# ─────────────────────────────────────────────────────────────────────────────
#  Load all data at startup
# ─────────────────────────────────────────────────────────────────────────────
RUNS      = []
PSI_PCA   = None; PSI_GP   = None; PSI_LOO   = None
ETA_PCA   = None; ETA_GP   = None; ETA_LOO   = None
AUDIO_PCA = None; AUDIO_GP = None; AUDIO_LOO = None


def _try_load(base_dir="."):
    global RUNS, PSI_PCA, PSI_GP, PSI_LOO, ETA_PCA, ETA_GP, ETA_LOO, AUDIO_PCA, AUDIO_GP, AUDIO_LOO
    if not _HAVE_CORE:
        print("[demo] pca_core unavailable — UI preview mode.")
        return
    try:
        print("Loading DoE dataset …")
        RUNS, r2s_psi, r2s_eta = load_data(base_dir=base_dir)
        print(f"  → {len(RUNS)} runs loaded.")

        print("Building ψ PCA + GP …")
        PSI_PCA = remove_outliers_redo_PCA(PCAModel(remove_outliers_from_smoothing(RUNS, r2s_psi), curve_key="psi", n_modes=5))
        PSI_GP  = GPSurrogates(PSI_PCA)
        PSI_LOO = PSI_GP.loo_rms()
        print(f"  ψ LOO RMS = {PSI_LOO:.5f}")

        eta_runs = [r for r in RUNS if not np.all(np.isnan(r["eta"]))]
        print(f"Building η PCA + GP ({len(eta_runs)} valid runs) …")
        ETA_PCA = remove_outliers_redo_PCA(PCAModel(remove_outliers_from_smoothing(eta_runs, r2s_eta), curve_key="eta", n_modes=3))
        ETA_GP  = GPSurrogates(ETA_PCA)
        ETA_LOO = ETA_GP.loo_rms()
        print(f"  η LOO RMS = {ETA_LOO:.5f}")
        
        print("Building AUDIO PCA + GP …")
        AUDIO_PCA = remove_outliers_redo_PCA(PCAModel(RUNS, curve_key="audio_amplitude", n_modes=6))
        AUDIO_GP  = GPSurrogates(AUDIO_PCA)
        AUDIO_LOO = AUDIO_GP.loo_rms()
        print(f"  AUDIO LOO RMS = {AUDIO_LOO:.5f}")
        
    except Exception as exc:
        print(f"[warn] Data load failed: {exc}\n       Running in UI preview mode.")


_try_load()
N_MODES_INIT = 5


# ─────────────────────────────────────────────────────────────────────────────
#  Layout helpers
# ─────────────────────────────────────────────────────────────────────────────
def card(children, extra_style=None):
    style = {
        "background": C["surface"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "10px",
        "padding": "14px 16px",
        "marginBottom": "10px",
        "boxShadow": f"0 2px 8px {C['shadow']}",
    }
    if extra_style:
        style.update(extra_style)
    return html.Div(children, style=style)


def section_label(text):
    return html.P(text, style={
        "fontSize": "10px",
        "fontFamily": "'IBM Plex Mono', monospace",
        "color": C["text3"],
        "fontWeight": "700",
        "letterSpacing": "0.1em",
        "textTransform": "uppercase",
        "margin": "0 0 8px 0",
        "borderBottom": f"1px solid {C['border']}",
        "paddingBottom": "4px",
    })


def make_sliders(pfx):
    rows = []
    for i, (label, (key, lo, hi, init)) in enumerate(PARAM_BOUNDS.items()):
        is_int = key == "n_blade"
        rows.append(html.Div([
            html.Div([
                html.Span(label, style={
                    "fontSize": "11px", "fontFamily": "'IBM Plex Mono', monospace",
                    "color": C["text2"], "fontWeight": "600",
                }),
                html.Span(id={"type": f"{pfx}-val", "index": i},
                          style={"fontSize": "11px", "fontFamily": "'IBM Plex Mono', monospace",
                                 "color": C["text"], "fontWeight": "700", "float": "right"}),
            ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "2px"}),
            dcc.Slider(
                id={"type": f"{pfx}-sl", "index": i},
                min=lo, max=hi,
                step=1 if is_int else (hi - lo) / 50,
                value=init,
                marks={lo: {"label": f"{lo:g}", "style": {"fontSize": "9px", "color": C["text3"]}},
                       hi: {"label": f"{hi:g}", "style": {"fontSize": "9px", "color": C["text3"]}}},
                tooltip={"always_visible": False},
                updatemode="drag",
            ),
        ], style={"marginBottom": "8px"}))
    return rows


def make_sweep_radio(pfx):
    options = [{"label": "None", "value": -1}] + [
        {"label": n, "value": i} for i, n in enumerate(PARAM_NAMES)
    ]
    return dcc.RadioItems(
        id=f"{pfx}-sweep", options=options, value=-1,
        labelStyle={"display": "block", "fontSize": "11px",
                    "fontFamily": "'IBM Plex Mono', monospace",
                    "color": C["text2"], "cursor": "pointer", "padding": "2px 0"},
        inputStyle={"marginRight": "6px"},
    )


# def make_modes_radio(pfx):
#     return dcc.RadioItems(
#         id=f"{pfx}-nmodes",
#         options=[{"label": f" {k}", "value": k} for k in range(1, 7)],
#         value=N_MODES_INIT, inline=True,
#         labelStyle={"fontSize": "11px", "fontFamily": "'IBM Plex Mono', monospace",
#                     "color": C["text2"], "cursor": "pointer", "marginRight": "10px"},
#         inputStyle={"marginRight": "3px"},
#     )


def control_col(pfx):
    accent = C["accent_psi"] if pfx == "psi" else (C["accent_eta"] if pfx == "eta" else C["accent_audio"])
    return html.Div([
        card([section_label("Parameters"), *make_sliders(pfx)]),
        card([section_label("Sweep parameter"), make_sweep_radio(pfx)]),
        # card([section_label("PCA modes  k"), make_modes_radio(pfx)]),
        card([
            section_label("Model info"),
            html.Pre(id=f"{pfx}-info", style={
                "fontSize": "10px", "fontFamily": "'IBM Plex Mono', monospace",
                "color": C["text2"], "margin": "0", "whiteSpace": "pre-wrap",
                "lineHeight": "1.6",
            }),
        ]),
        html.Button("⬇  Export CSV", id=f"{pfx}-export-btn", n_clicks=0, style={
            "width": "100%", "padding": "10px 0",
            "background": accent, "color": "#fff", "border": "none",
            "borderRadius": "8px", "fontFamily": "'IBM Plex Mono', monospace",
            "fontSize": "12px", "fontWeight": "700", "cursor": "pointer",
            "letterSpacing": "0.05em",
        }),
        dcc.Download(id=f"{pfx}-dl"),
        html.Div(id=f"{pfx}-export-msg", style={
            "fontSize": "10px", "color": C["green"],
            "fontFamily": "'IBM Plex Mono', monospace",
            "marginTop": "5px", "textAlign": "center",
        }),
    ], style={"width": "265px", "flexShrink": "0",
              "overflowY": "auto", "maxHeight": "calc(100vh - 110px)",
              "paddingRight": "4px"})


def chart_col(pfx):
    cfg = {"displayModeBar": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
    cfg_small = {"displayModeBar": False}
    return html.Div([
        dcc.Graph(id=f"{pfx}-main",  config=cfg,       style={"height": "420px", "marginBottom": "10px"}),
        html.Div([
            dcc.Graph(id=f"{pfx}-scree", config=cfg_small, style={"height": "280px", "flex": "1"}),
            dcc.Graph(id=f"{pfx}-modes", config=cfg_small, style={"height": "280px", "flex": "1"}),
        ], style={"display": "flex", "gap": "10px"}),
    ], style={"flex": "1", "minWidth": "0"})


def tab_panel(pfx):
    return html.Div([
        control_col(pfx),
        chart_col(pfx),
    ], style={"display": "flex", "gap": "14px", "alignItems": "flex-start"})


def audio_tab():
    return html.Div([
        control_col("audio"),
        html.Div([
            # OASPL badge
            html.Div([
                html.Span("OASPL  ", style={
                    "fontFamily": "'IBM Plex Mono', monospace",
                    "fontSize": "12px", "fontWeight": "600",
                    "color": C["text3"], "letterSpacing": "0.08em",
                }),
                html.Span(id="audio-oaspl", children="—", style={
                    "fontFamily": "'IBM Plex Mono', monospace",
                    "fontSize": "20px", "fontWeight": "700",
                    "color": C["accent_audio"],
                }),
                html.Span("  dBFS", style={
                    "fontFamily": "'IBM Plex Mono', monospace",
                    "fontSize": "12px", "color": C["text3"],
                }),
            ], style={
                "background": C["surface"],
                "border": f"1px solid {C['border']}",
                "borderRadius": "8px", "padding": "8px 18px",
                "display": "inline-flex", "alignItems": "baseline",
                "gap": "2px", "marginBottom": "10px",
                "boxShadow": f"0 1px 4px {C['shadow']}",
            }),
            dcc.Graph(id="audio-main",
                      config={"displayModeBar": True, "displaylogo": False,
                              "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                      style={"height": "420px", "marginBottom": "10px"}),
            html.Div([
                dcc.Graph(id="audio-scree", config={"displayModeBar": False},
                          style={"height": "280px", "flex": "1"}),
                dcc.Graph(id="audio-modes", config={"displayModeBar": False},
                          style={"height": "280px", "flex": "1"}),
            ], style={"display": "flex", "gap": "10px"}),
        ], style={"flex": "1", "minWidth": "0"}),
    ], style={"display": "flex", "gap": "14px", "alignItems": "flex-start"})


def _feature_row(icon, text, color):
    return html.Div([
        html.Span(icon, style={"color": color, "marginRight": "10px", "fontWeight": "700"}),
        html.Span(text, style={"color": C["text2"]}),
    ], style={
        "fontFamily": "'IBM Plex Mono', monospace",
        "fontSize": "12px", "lineHeight": "2.2",
    })


# ─────────────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;600&display=swap",
    ],
    title="PCA + GP Surrogate · DoE Compressor",
)
server = app.server

# Global CSS injection for slider accent colours
app.index_string = """
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #f4f6f9; }
  /* Plotly slider thumb */
  .rc-slider-handle { border-color: #0369a1 !important; }
  .rc-slider-track  { background: #0369a1 !important; }
  /* Dash core slider */
  .dash-slider .rc-slider-handle { border-color: #1a2332 !important; }
  /* Tab styling */
  .custom-tabs .tab { 
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #4a5d72 !important;
    border-top: 3px solid transparent !important;
    background: #f0f4f8 !important;
    padding: 10px 20px !important;
  }
  .custom-tabs .tab--selected {
    color: #1a2332 !important;
    background: #ffffff !important;
    border-top: 3px solid #0369a1 !important;
    border-bottom: 2px solid #ffffff !important;
  }
  .custom-tabs .tab:hover:not(.tab--selected) {
    background: #e8edf4 !important;
    color: #1a2332 !important;
  }
  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #f0f4f8; }
  ::-webkit-scrollbar-thumb { background: #b8c8db; border-radius: 3px; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
"""

app.layout = html.Div([

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("PCA + GP Surrogate", style={
                "fontFamily": "'IBM Plex Mono', monospace",
                "fontWeight": "700", "fontSize": "15px", "color": C["text"],
            }),
            html.Span(" · DoE Compressor Tool", style={
                "fontFamily": "'IBM Plex Sans', sans-serif",
                "fontSize": "13px", "color": C["text2"], "marginLeft": "4px",
            }),
        ]),
        html.Span(
            f"{len(RUNS)} runs loaded" if RUNS else "⚠ demo mode — no data",
            style={
                "background": C["surface2"],
                "border": f"1px solid {C['border']}",
                "borderRadius": "20px", "padding": "3px 12px",
                "fontSize": "11px", "fontFamily": "'IBM Plex Mono', monospace",
                "color": C["green"] if RUNS else C["amber"],
            },
        ),
    ], style={
        "height": "52px",
        "background": C["surface"],
        "borderBottom": f"2px solid {C['border']}",
        "display": "flex", "alignItems": "center",
        "justifyContent": "space-between",
        "padding": "0 22px",
        "position": "sticky", "top": "0", "zIndex": "200",
        "boxShadow": f"0 2px 8px {C['shadow']}",
    }),

    # ── Tabs ──────────────────────────────────────────────────────────────────
    dcc.Tabs(
        id="tabs", value="psi",
        className="custom-tabs",
        children=[
            dcc.Tab(label="Tab 1 — ψ(φ)  Pressure Rise",      value="psi",   className="tab"),
            dcc.Tab(label="Tab 2 — η(φ)  Efficiency",          value="eta",   className="tab"),
            dcc.Tab(label="Tab 3 — 🎵  Audio  SPL",            value="audio", className="tab"),
        ],
        style={"borderBottom": f"2px solid {C['border']}"},
        colors={"border": C["border"], "primary": C["cyan"], "background": C["surface2"]},
    ),

    # ── Content — all three panels pre-rendered, toggled via CSS ─────────────
    html.Div([
        html.Div(id="panel-psi",   children=tab_panel("psi"),
                 style={"display": "flex",  "padding": "16px 20px"}),
        html.Div(id="panel-eta",   children=tab_panel("eta"),
                 style={"display": "none",  "padding": "16px 20px"}),
        html.Div(id="panel-audio", children=audio_tab(),
                 style={"display": "none",  "padding": "16px 20px"}),
    ], style={
        "background": C["bg"],
        "minHeight": "calc(100vh - 52px - 42px)",
    }),

], style={"background": C["bg"], "minHeight": "100vh"})


# ─────────────────────────────────────────────────────────────────────────────
#  Tab router — only toggles CSS display, never re-mounts components
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("panel-psi",   "style"),
    Output("panel-eta",   "style"),
    Output("panel-audio", "style"),
    Input("tabs", "value"),
)
def show_tab(tab):
    base = "16px 20px"
    show = {"display": "flex",  "padding": base}
    hide = {"display": "none",  "padding": base}
    return (
        show if tab == "psi"   else hide,
        show if tab == "eta"   else hide,
        show if tab == "audio" else hide,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Slider label display (pattern-match callbacks)
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output({"type": "psi-val", "index": ALL}, "children"),
    Input({"type": "psi-sl",   "index": ALL}, "value"),
)
def _psi_labels(vals):
    return _fmt_labels(vals)


@app.callback(
    Output({"type": "eta-val", "index": ALL}, "children"),
    Input({"type": "eta-sl",   "index": ALL}, "value"),
)
def _eta_labels(vals):
    return _fmt_labels(vals)


@app.callback(
    Output({"type": "audio-val", "index": ALL}, "children"),
    Input({"type": "audio-sl",   "index": ALL}, "value"),
)
def _audio_labels(vals):
    return _fmt_labels(vals)


def _fmt_labels(vals):
    out = []
    for i, v in enumerate(vals):
        k = list(PARAM_BOUNDS.values())[i][0]
        out.append(f"{int(v)}" if k == "n_blade" else f"{v:.4g}")
    return out

# ─────────────────────────────────────────────────────────────────────────────
#  Cross-tab slider sync  — any slider change propagates to the other two tabs
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output({"type": "eta-sl",   "index": ALL}, "value", allow_duplicate=True),
    Output({"type": "audio-sl", "index": ALL}, "value", allow_duplicate=True),
    Input({"type": "psi-sl",    "index": ALL}, "value"),
    prevent_initial_call=True,
)
def _sync_from_psi(vals):
    return vals, vals


@app.callback(
    Output({"type": "psi-sl",   "index": ALL}, "value", allow_duplicate=True),
    Output({"type": "audio-sl", "index": ALL}, "value", allow_duplicate=True),
    Input({"type": "eta-sl",    "index": ALL}, "value"),
    prevent_initial_call=True,
)
def _sync_from_eta(vals):
    return vals, vals


@app.callback(
    Output({"type": "psi-sl",   "index": ALL}, "value", allow_duplicate=True),
    Output({"type": "eta-sl",   "index": ALL}, "value", allow_duplicate=True),
    Input({"type": "audio-sl",  "index": ALL}, "value"),
    prevent_initial_call=True,
)
def _sync_from_audio(vals):
    return vals, vals



# ─────────────────────────────────────────────────────────────────────────────
#  Core plot builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_plots(pfx, slider_vals, sweep_idx):
    is_psi    = pfx == "psi"
    accent    = C["accent_psi"] if is_psi else C["accent_eta"]
    band1     = C["band_psi"]   if is_psi else C["band_eta"]
    band2     = C["band_psi2"]  if is_psi else C["band_eta2"]
    curve_key = "psi" if is_psi else "eta"
    y_label   = "ψ" if is_psi else "η"

    # Access / rebuild model globals
    global PSI_PCA, PSI_GP, PSI_LOO, ETA_PCA, ETA_GP, ETA_LOO
    pca_obj = PSI_PCA if is_psi else ETA_PCA
    gp_obj  = PSI_GP  if is_psi else ETA_GP
    loo_err = PSI_LOO if is_psi else ETA_LOO
    try:
        n_modes = pca_obj.n_modes
    except AttributeError:
    # if pca_obj is None or pca_obj.n_modes != n_modes:
        if not RUNS:
            return _empty(), _empty(), _empty(), "No data loaded."
        runs_src = RUNS if is_psi else [r for r in RUNS if not np.all(np.isnan(r["eta"]))]
        pca_obj  = PCAModel(runs_src, curve_key=curve_key, n_modes=n_modes)
        gp_obj   = GPSurrogates(pca_obj)
        loo_err  = gp_obj.loo_rms()
        if is_psi: PSI_PCA, PSI_GP, PSI_LOO = pca_obj, gp_obj, loo_err
        else:      ETA_PCA, ETA_GP, ETA_LOO = pca_obj, gp_obj, loo_err

    param_vals = list(slider_vals)
    scores_m, scores_s, phi_s, _, phi_e, _ = gp_obj.predict(param_vals)
    pred_full = pca_obj.reconstruct(scores_m)
    mask      = pca_obj.return_mask(phi_s, phi_e)
    phi_m     = PHI_COMMON[mask]
    pred      = pred_full[mask]
    var       = sum((pca_obj.modes[k] * scores_s[k]) ** 2 for k in range(n_modes))
    std       = np.sqrt(var)[mask]

    # ── Main figure ──────────────────────────────────────────────────────────
    fig = go.Figure()

    # Background DoE runs (faint)
    runs_src = RUNS if is_psi else [r for r in RUNS if not np.all(np.isnan(r["eta"]))]
    for r in runs_src:
        fig.add_trace(go.Scatter(
            x=PHI_COMMON, y=r[curve_key], mode="lines",
            line=dict(color="#94a3b8", width=0.7), opacity=0.2,
            showlegend=False, hoverinfo="skip",
        ))

    # Mean curve
    fig.add_trace(go.Scatter(
        x=PHI_COMMON, y=pca_obj.mean_curve, mode="lines",
        line=dict(color=C["text3"], width=1.4, dash="dot"),
        name="Mean curve", opacity=0.6,
    ))

    # ±2σ band
    fig.add_trace(go.Scatter(
        x=np.concatenate([phi_m, phi_m[::-1]]),
        y=np.concatenate([pred + 2*std, (pred - 2*std)[::-1]]),
        fill="toself", fillcolor=band1,
        line=dict(color="rgba(0,0,0,0)"),
        name="GP ±2σ", hoverinfo="skip",
    ))

    # ±1σ band
    fig.add_trace(go.Scatter(
        x=np.concatenate([phi_m, phi_m[::-1]]),
        y=np.concatenate([pred + std, (pred - std)[::-1]]),
        fill="toself", fillcolor=band2,
        line=dict(color="rgba(0,0,0,0)"),
        name="GP ±1σ", hoverinfo="skip",
    ))

    # GP prediction
    fig.add_trace(go.Scatter(
        x=phi_m, y=pred, mode="lines",
        line=dict(color=accent, width=2.8),
        name=f"GP prediction  (k={n_modes})",
    ))

    # η-only peak marker
    if not is_psi:
        pi = int(np.nanargmax(pred))
        fig.add_trace(go.Scatter(
            x=[phi_m[pi]], y=[pred[pi]], mode="markers",
            marker=dict(color=C["red"], size=10, symbol="circle",
                        line=dict(color="white", width=1.5)),
            name=f"η_peak = {pred[pi]:.3f}  (φ={phi_m[pi]:.3f})",
        ))
        fig.add_vline(x=phi_m[pi], line_color=C["red"],
                      line_dash="dot", line_width=1, opacity=0.55)

    # Parameter sweep
    if sweep_idx >= 0:
        meta    = list(PARAM_BOUNDS.values())[sweep_idx]
        n_steps = 5 if sweep_idx == 7 else 8
        s_vals  = np.linspace(meta[1], meta[2], n_steps)
        for i, val in enumerate(s_vals):
            pv = list(param_vals); pv[sweep_idx] = val
            sm, _, ps, _, pe, _ = gp_obj.predict(pv)
            yy = pca_obj.reconstruct(sm)
            mk = pca_obj.return_mask(ps, pe)
            hue = int(200 + i * 15)
            lum = int(30 + i * 5)
            col = f"hsl({hue}, 72%, {lum}%)"
            fig.add_trace(go.Scatter(
                x=PHI_COMMON[mk], y=yy[mk], mode="lines",
                line=dict(color=col, width=1.5), opacity=0.75,
                name=f"{PARAM_KEYS[sweep_idx]} = {val:.3g}",
            ))

    param_title = "  ·  ".join(f"{k}={v:.3g}" for k, v in zip(PARAM_KEYS, param_vals))
    y_axis_label = ("Pressure-rise coefficient  ψ  [—]"
                    if is_psi else "Efficiency  η  [—]")

    fig.update_layout(
        **PLOT_LAYOUT_BASE,
        margin=dict(l=58, r=16, t=44, b=46),
        title=dict(text=param_title, font=dict(size=9, color=C["text3"]), x=0, xref="paper"),
        xaxis_title="Flow coefficient  φ  [—]",
        yaxis_title=y_axis_label,
        xaxis_range=[0.04, 0.26],
        yaxis_range=[0.1, 0.5] if is_psi else [0.0, 0.22],
    )

    # ── Scree ─────────────────────────────────────────────────────────────────
    k_max   = min(len(pca_obj.singular_values), 10)
    k_range = list(range(1, k_max + 1))
    ind_var = pca_obj.var_explained[:k_max].tolist()
    cum_var = pca_obj.cum_var[:k_max].tolist()

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Bar(
        x=k_range, y=ind_var,
        marker_color=C["cyan"], opacity=0.6,
        name="Individual %",
    ))
    fig_sc.add_trace(go.Scatter(
        x=k_range, y=cum_var, mode="lines+markers",
        line=dict(color=C["amber"], width=2),
        marker=dict(size=6, color=C["amber"]),
        name="Cumulative %",
    ))
    fig_sc.add_hline(y=95, line_color=C["red"], line_dash="dash",
                     line_width=1, opacity=0.65,
                     annotation_text="95%", annotation_font_size=9,
                     annotation_font_color=C["red"])
    fig_sc.add_vline(x=n_modes, line_color=C["green"], line_dash="dot",
                     line_width=1.5, opacity=0.8)
    fig_sc.update_layout(
        **PLOT_LAYOUT_BASE,
        margin=dict(l=48, r=12, t=36, b=42),
        title=dict(text="Scree plot", font=dict(size=11), x=0.5, xref="paper"),
        xaxis_title="Mode k",
        yaxis_title="Variance [%]",
        # xaxis=dict(**PLOT_LAYOUT_BASE["xaxis"], tickvals=k_range, range=[0, k_max + 1]),
    )

    # ── Modes ─────────────────────────────────────────────────────────────────
    fig_md = go.Figure()
    fig_md.add_hline(y=0, line_color=C["grey_line"], line_dash="dash",
                     line_width=0.8, opacity=0.4)
    fig_md.add_trace(go.Scatter(
        x=PHI_COMMON, y=pca_obj.mean_curve, mode="lines",
        line=dict(color=C["text3"], width=1.2, dash="dot"),
        name=f"Mean {y_label}", opacity=0.65,
    ))
    for k in range(n_modes):
        fig_md.add_trace(go.Scatter(
            x=PHI_COMMON, y=pca_obj.modes[k], mode="lines",
            line=dict(color=MCOLORS[k % len(MCOLORS)], width=1.8),
            name=f"Mode {k+1}  ({pca_obj.var_explained[k]:.1f}%)",
        ))
    fig_md.update_layout(
        **PLOT_LAYOUT_BASE,
        margin=dict(l=48, r=12, t=36, b=42),
        title=dict(text="Principal mode shapes", font=dict(size=11), x=0.5, xref="paper"),
        xaxis_title="Flow coefficient  φ  [—]",
        yaxis_title="Mode amplitude  [—]",
    )

    # ── Info text ─────────────────────────────────────────────────────────────
    lines = [
        f"DoE runs  : {len(RUNS)}",
        f"φ points  : {N_PHI}",
        f"Modes k   : {n_modes}",
        f"Var capt. : {pca_obj.cum_var[n_modes-1]:.1f}%",
        "",
        "── GP predicted scores ──",
    ]
    for k in range(n_modes):
        lines.append(f"  s{k+1}: {scores_m[k]:+.4f}  ±{scores_s[k]:.4f}")
    if loo_err is not None:
        lines += ["", f"LOO RMS   : {loo_err:.5f}"]

    return fig, fig_sc, fig_md, "\n".join(lines)


def _empty(msg="No data loaded"):
    fig = go.Figure()
    fig.update_layout(
        **PLOT_LAYOUT_BASE,
        margin=dict(l=48, r=16, t=40, b=44),
        annotations=[dict(
            text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=13, color=C["text3"]),
        )],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  ψ callbacks
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("psi-main",  "figure"),
    Output("psi-scree", "figure"),
    Output("psi-modes", "figure"),
    Output("psi-info",  "children"),
    Input({"type": "psi-sl", "index": ALL}, "value"),
    Input("psi-sweep",  "value"),
    # Input("psi-nmodes", "value"),
    prevent_initial_call=False,
)
def update_psi(slider_vals, sweep):
    if not slider_vals:
        e = _empty()
        return e, e, e, "Waiting for sliders…"
    try:
        return _build_plots("psi", slider_vals, sweep)
    except Exception as exc:
        e = _empty(f"Error: {exc}")
        return e, e, e, str(exc)


@app.callback(
    Output("psi-dl",         "data"),
    Output("psi-export-msg", "children"),
    Input("psi-export-btn",  "n_clicks"),
    State({"type": "psi-sl", "index": ALL}, "value"),
    # State("psi-nmodes",      "value"),
    prevent_initial_call=True,
)
def export_psi(_, vals):
    return _do_export("psi", vals)


# ─────────────────────────────────────────────────────────────────────────────
#  η callbacks
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("eta-main",  "figure"),
    Output("eta-scree", "figure"),
    Output("eta-modes", "figure"),
    Output("eta-info",  "children"),
    Input({"type": "eta-sl", "index": ALL}, "value"),
    Input("eta-sweep",  "value"),
    # Input("eta-nmodes", "value"),
    prevent_initial_call=True,
)
def update_eta(slider_vals, sweep):
    if not slider_vals:
        e = _empty()
        return e, e, e, "Waiting for sliders…"
    try:
        return _build_plots("eta", slider_vals, sweep,)
    except Exception as exc:
        e = _empty(f"Error: {exc}")
        return e, e, e, str(exc)


@app.callback(
    Output("eta-dl",         "data"),
    Output("eta-export-msg", "children"),
    Input("eta-export-btn",  "n_clicks"),
    State({"type": "eta-sl", "index": ALL}, "value"),
    # State("eta-nmodes",      "value"),
    prevent_initial_call=True,
)
def export_eta(_, vals):
    return _do_export("eta", vals)


# ─────────────────────────────────────────────────────────────────────────────
#  Audio plot builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_audio_plots(slider_vals, sweep_idx):
    global AUDIO_PCA, AUDIO_GP, AUDIO_LOO

    try:
        n_modes = AUDIO_PCA.n_modes
    except AttributeError:
        if not RUNS:
            return _empty(), _empty(), _empty(), "—", "No data loaded."
        AUDIO_PCA = PCAModel(RUNS, curve_key="audio_amplitude", n_modes=n_modes)
        AUDIO_GP  = GPSurrogates(AUDIO_PCA)
        AUDIO_LOO = AUDIO_GP.loo_rms()

    param_vals = list(slider_vals)
    # Audio GP returns only (mean_curve, std_curve) — not 6 values like psi/eta
    scores_m, scores_s = AUDIO_GP.predict(param_vals)
    pred = AUDIO_PCA.reconstruct(scores_m)
    var       = sum((AUDIO_PCA.modes[k] * scores_s[k]) ** 2 for k in range(n_modes))
    std       = np.sqrt(var)

    oaspl_val = calculate_oaspl(pred)
    oaspl_str = f"{oaspl_val:.1f}"

    freqs = np.array(CENTRE_FREQS)

    # ── Main SPL figure (log x-axis) ─────────────────────────────────────────
    fig = go.Figure()

    for r in RUNS:
        fig.add_trace(go.Scatter(
            x=freqs, y=r["audio_amplitude"], mode="lines",
            line=dict(color="#94a3b8", width=0.7), opacity=0.18,
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=freqs, y=AUDIO_PCA.mean_curve, mode="lines",
        line=dict(color=C["text3"], width=1.4, dash="dot"),
        name="Mean SPL", opacity=0.6,
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([freqs, freqs[::-1]]),
        y=np.concatenate([pred + 2*std, (pred - 2*std)[::-1]]),
        fill="toself", fillcolor=C["band_audio"],
        line=dict(color="rgba(0,0,0,0)"),
        name="GP ±2σ", hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([freqs, freqs[::-1]]),
        y=np.concatenate([pred + std, (pred - std)[::-1]]),
        fill="toself", fillcolor=C["band_audio2"],
        line=dict(color="rgba(0,0,0,0)"),
        name="GP ±1σ", hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=freqs, y=pred, mode="lines",
        line=dict(color=C["accent_audio"], width=2.8),
        name=f"GP prediction  (k={n_modes})",
    ))

    if sweep_idx >= 0:
        print("SWEEP")
        print(sweep_idx)
        meta    = list(PARAM_BOUNDS.values())[sweep_idx]
        n_steps = 5 if sweep_idx == 7 else 8
        s_vals  = np.linspace(meta[1], meta[2], n_steps)
        for i, val in enumerate(s_vals):
            pv = list(param_vals); pv[sweep_idx] = val
            sm, _ = AUDIO_GP.predict(pv)
            yy = AUDIO_PCA.reconstruct(sm)
            # print(yy)
            hue = int(170 + i * 16)
            lum = int(28 + i * 6)
            fig.add_trace(go.Scatter(
                x=freqs, y=yy, mode="lines",
                line=dict(color=f"hsl({hue}, 68%, {lum}%)", width=1.5),
                opacity=0.75,
                name=f"{PARAM_KEYS[sweep_idx]} = {val:.3g}",
            ))

    param_title = "  ·  ".join(f"{k}={v:.3g}" for k, v in zip(PARAM_KEYS, param_vals))
    log_xaxis = dict(
        type="log",
        gridcolor=C["border"], linecolor=C["border2"],
        tickcolor=C["border2"], zerolinecolor=C["border"],
        title="Frequency  [Hz]", tickformat=".0f",
    )
    
    fig.update_layout(
        **PLOT_LAYOUT_LOG,
        margin=dict(l=58, r=16, t=44, b=46),
        title=dict(text=param_title, font=dict(size=9, color=C["text3"]), x=0, xref="paper"),
        xaxis=log_xaxis,
        yaxis_title="SPL  [dBFS]",
        # range=[np.log10(10), np.log10(2000)]
    )
    fig.update_xaxes(
        range=[np.log10(40), np.log10(2000)]
    )

    # ── Scree ─────────────────────────────────────────────────────────────────
    k_max   = min(len(AUDIO_PCA.singular_values), 10)
    k_range = list(range(1, k_max + 1))
    fig_sc  = go.Figure()
    fig_sc.add_trace(go.Bar(
        x=k_range, y=AUDIO_PCA.var_explained[:k_max].tolist(),
        marker_color=C["accent_audio"], opacity=0.55, name="Individual %",
    ))
    fig_sc.add_trace(go.Scatter(
        x=k_range, y=AUDIO_PCA.cum_var[:k_max].tolist(), mode="lines+markers",
        line=dict(color=C["amber"], width=2),
        marker=dict(size=6, color=C["amber"]),
        name="Cumulative %",
    ))
    fig_sc.add_hline(y=95, line_color=C["red"], line_dash="dash", line_width=1,
                     opacity=0.65, annotation_text="95%", annotation_font_size=9,
                     annotation_font_color=C["red"])
    fig_sc.add_vline(x=n_modes, line_color=C["green"], line_dash="dot",
                     line_width=1.5, opacity=0.8)
    fig_sc.update_layout(
        **PLOT_LAYOUT_BASE,
        margin=dict(l=48, r=12, t=36, b=42),
        title=dict(text="Scree plot", font=dict(size=11), x=0.5, xref="paper"),
        xaxis_title="Mode k", yaxis_title="Variance [%]",
    )

    # ── Mode shapes (log x) ───────────────────────────────────────────────────
    fig_md = go.Figure()
    fig_md.add_hline(y=0, line_color=C["grey_line"], line_dash="dash",
                     line_width=0.8, opacity=0.4)
    # fig_md.add_trace(go.Scatter(
    #     x=freqs, y=AUDIO_PCA.mean_curve, mode="lines",
    #     line=dict(color=C["text3"], width=1.2, dash="dot"),
    #     name="Mean SPL", opacity=0.65,
    # ))
    for k in range(n_modes):
        fig_md.add_trace(go.Scatter(
            x=freqs, y=AUDIO_PCA.modes[k], mode="lines",
            line=dict(color=MCOLORS[k % len(MCOLORS)], width=1.8),
            name=f"Mode {k+1}  ({AUDIO_PCA.var_explained[k]:.1f}%)",
        ))
    fig_md.update_layout(
        **PLOT_LAYOUT_LOG,
        margin=dict(l=48, r=12, t=36, b=42),
        title=dict(text="Principal mode shapes", font=dict(size=11), x=0.5, xref="paper"),
        xaxis=log_xaxis,
        yaxis_title="Mode amplitude  [—]",
    )

    # ── Info text ─────────────────────────────────────────────────────────────
    lines = [
        f"DoE runs  : {len(RUNS)}",
        f"Freq bins : {len(freqs)}",
        f"Modes k   : {n_modes}",
        f"Var capt. : {AUDIO_PCA.cum_var[n_modes-1]:.1f}%",
        f"OASPL     : {oaspl_val:.2f} dBFS",
    ]
    if AUDIO_LOO is not None:
        lines += ["", f"LOO RMS   : {AUDIO_LOO:.5f}"]

    return fig, fig_sc, fig_md, oaspl_str, "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Audio callbacks
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("audio-main",  "figure"),
    Output("audio-scree", "figure"),
    Output("audio-modes", "figure"),
    Output("audio-oaspl", "children"),
    Output("audio-info",  "children"),
    Input({"type": "audio-sl", "index": ALL}, "value"),
    Input("audio-sweep",  "value"),
    # Input("audio-nmodes", "value"),
    prevent_initial_call=True,
)
def update_audio(slider_vals, sweep):
    if not slider_vals:
        e = _empty()
        return e, e, e, "—", "Waiting for sliders…"
    try:
        return _build_audio_plots(slider_vals, sweep or -1)
    except Exception as exc:
        e = _empty(f"Error: {exc}")
        return e, e, e, "—", str(exc)


@app.callback(
    Output("audio-dl",         "data"),
    Output("audio-export-msg", "children"),
    Input("audio-export-btn",  "n_clicks"),
    State({"type": "audio-sl", "index": ALL}, "value"),
    # State("audio-nmodes",      "value"),
    prevent_initial_call=True,
)
def export_audio(_, vals):
    if AUDIO_PCA is None or AUDIO_GP is None:
        return None, "⚠ No model — load data first."
    n_modes   = AUDIO_PCA.n_modes
    freqs     = np.array(CENTRE_FREQS)
    pred, _   = AUDIO_GP.predict(list(vals))
    oaspl_val = calculate_oaspl(pred)
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["freq_hz", "spl_predicted_dBFS", "spl_mean_dBFS"] +
               [f"mode{k+1}" for k in range(n_modes)] + ["oaspl_dBFS"])
    for i, f in enumerate(freqs):
        row = [f, pred[i], AUDIO_PCA.mean_curve[i]]
        row += [AUDIO_PCA.modes[k][i] for k in range(n_modes)]
        row += [f"{oaspl_val:.4f}" if i == 0 else ""]
        w.writerow(row)
    fname = ("audio_spl_" +
             "_".join(f"{k}{v:.2f}" for k, v in zip(PARAM_KEYS, vals)) + ".csv")
    return dict(content=buf.getvalue(), filename=fname, type="text/csv"), f"✓  {fname}"


# ─────────────────────────────────────────────────────────────────────────────
#  CSV export (ψ and η)
# ─────────────────────────────────────────────────────────────────────────────
def _do_export(pfx, slider_vals):
    is_psi    = pfx == "psi"
    pca_obj   = PSI_PCA if is_psi else ETA_PCA
    gp_obj    = PSI_GP  if is_psi else ETA_GP
    curve_key = "psi" if is_psi else "eta"
    if pca_obj is None or gp_obj is None:
        return None, "⚠ No model — load data first."
    n_modes = pca_obj.n_modes
    scores_m, _, phi_s, _, phi_e, _ = gp_obj.predict(list(slider_vals))
    pred = pca_obj.reconstruct(scores_m)
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["phi", f"{curve_key}_predicted", f"{curve_key}_mean"] +
               [f"mode{k+1}" for k in range(n_modes)])
    for i, phi in enumerate(PHI_COMMON):
        row = [phi, pred[i], pca_obj.mean_curve[i]] + [pca_obj.modes[k][i] for k in range(n_modes)]
        w.writerow([f"{v:.6f}" for v in row])
    fname = (f"{curve_key}_gp_" +
             "_".join(f"{k}{v:.2f}" for k, v in zip(PARAM_KEYS, slider_vals)) + ".csv")
    return dict(content=buf.getvalue(), filename=fname, type="text/csv"), f"✓  {fname}"


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import threading, webbrowser, time

    def _open():
        time.sleep(1.3)
        webbrowser.open("http://localhost:8050")

    threading.Thread(target=_open, daemon=True).start()
    print("\n" + "═" * 58)
    print("  PCA + GP Surrogate  ·  DoE Compressor Tool  (web)")
    print("  Open:  http://localhost:8050")
    print("═" * 58 + "\n")
    app.run(debug=False, port=8050)
