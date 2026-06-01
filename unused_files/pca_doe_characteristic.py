"""
PCA + Gaussian Process Regression for DoE Compressor/Fan Characteristics
=========================================================================
Workflow:
  2. Interpolates all runs onto a common φ grid → matrix X [N_runs × N_phi].
  3. Mean-centres X and performs SVD to extract principal mode shapes.
  4. Fits an independent GP surrogate for each retained mode score.
  5. Interactive Tkinter/Matplotlib GUI:
       - Sliders for design parameters (+ optional: manual mode score override)
       - Live reconstructed characteristic with ±2σ GP uncertainty band
       - Scree plot, mode shapes panel, score space scatter
       - Export current prediction to CSV
"""

import numpy as np
import pandas as pd
import matplotlib
import re
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import matplotlib.cm as cm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.interpolate import interp1d
import warnings, csv, os
warnings.filterwarnings("ignore")

# ─── Colour palette ──────────────────────────────────────────────────────────
BG      = "#0b0f1e"
PANEL   = "#111827"
BORDER  = "#1e3a5f"
CYAN    = "#00d4ff"
AMBER   = "#ffb347"
GREEN   = "#5fffaa"
RED     = "#ff5a5a"
PINK    = "#ff79c6"
GREY    = "#445566"
WHITE   = "#d0e8ff"
PURPLE  = "#921dd6"
LIME  = "#97e60f"
MCOLORS = [CYAN, AMBER, GREEN, PINK, RED, "#bd93f9"]
COLOUR_LIST = [CYAN, AMBER, GREEN, RED, PINK, WHITE, PURPLE, LIME]

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   WHITE,
    "xtick.color":       WHITE,
    "ytick.color":       WHITE,
    "text.color":        WHITE,
    "grid.color":        BORDER,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "font.family":       "monospace",
    "font.size":         8,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  BORDER,
    "legend.fontsize":   7,
})

# ─────────────────────────────────────────────────────────────────────────────
#  1.  Define Parameters
# ─────────────────────────────────────────────────────────────────────────────

PARAM_BOUNDS = {
    "Mass Flow Rate":    ("mdot",          0.025, 0.075, 0.050),
    "De Haller Midspan": ("DH_mid",        0.75,  0.9,   0.825),
    "Incidence Angle":   ("incidence",    -10.0,  5.0,  -2.5),
    "Vortex Exponent":   ("Vexp",         -2.0,   0.0,  -1.0),
    "Compound Lean Max": ("lean_compound", 0.0,   10.0,   5.0),
    "Straight Lean":     ("lean_straight", 0.0,   90.0,  45.0),
    "Tip Clearance %":   ("tip_clearance", 1.0,    5.0,   3.0),
    "Blade Count":       ("n_blade",       5,      9,     7),
}

PARAM_NAMES    = [k for k in PARAM_BOUNDS.keys()]
PARAM_KEYS     = [v[0] for v in PARAM_BOUNDS.values()]   # ["mdot", "DH_mid", ...]
PARAM_LO       = np.array([v[1] for v in PARAM_BOUNDS.values()], dtype=float)
PARAM_HI       = np.array([v[2] for v in PARAM_BOUNDS.values()], dtype=float)
PARAM_DEFAULTS = np.array([v[3] for v in PARAM_BOUNDS.values()], dtype=float)


N_PHI  = 100                                   # resolution of common φ grid
PHI_COMMON = np.linspace(0.05, 0.25, N_PHI)     # universal φ grid (modify!!)

# ─────────────────────────────────────────────────────────────────────────────
#  2.  Import test data
# ─────────────────────────────────────────────────────────────────────────────

# Helper fn to put spline over data
def load_with_gp_smooth(phi_raw, psi_raw, phi_common):
    """
    Fits a GP to one run's raw (phi, psi) scatter.
    Handles duplicates and noise naturally.
    Returns mean curve and std on common grid.
    """
    phi_raw = np.array(phi_raw).reshape(-1, 1)
    psi_raw = np.array(psi_raw)
    
    kernel = Matern(length_scale=0.05, length_scale_bounds=(1e-3, 1.0), nu=2.5) \
           + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                  normalize_y=True)
    try:
        gp.fit(phi_raw, psi_raw)
    except ValueError:
        print("Value Error!")
        print(psi_raw)
    
    
    mean, std = gp.predict(phi_common.reshape(-1, 1), return_std=True)
    return mean, std   # std gives per-point interpolation uncertainty

def load_data():
    # Load params table
    params_df = pd.read_csv("doe_params.csv", index_col=0)   # columns: run_id, mdot, dh_mid, ...

    runs = []
    for row in params_df.itertuples():
        # Extract data for filename
        index = row.Index
        n_blades = row.n_blade
        mdot = row.mdot
        DHmid = row.DH_mid
        INC = row.incidence
        Vexp = row.Vexp
        lean_max = row.lean_compound
        lean_weighting = int(1.0)
        lean_straight = row.lean_straight
        tip_clearance_percent = np.round(row.tip_clearance,3)
        # filename = f"DOE_{index}_{n_blades}_{mdot}_{DHmid}_{INC}_{Vexp}_{lean_max}_{lean_weighting}_{lean_straight}_{tip_clearance_percent}"
        
        for file in os.listdir("stl_files"):
            match = re.search(f"DOE_{index}", file)
            if match:
                parts = file.split(sep=".")
                # print(parts)
                if parts[-1] == "stl":
                    stl_file = file.rstrip(".stl")
        if stl_file is None:
            raise ValueError("No file found")
        filename = f'data/doe_data/{stl_file}.csv'
        
        
        try: 
            # print(os.listdir("data/doe_data"))
            # print(f"data/doe_data/{filename}blade.csv")
            curve_df = pd.read_csv(filename)
            # print(f"{index} found!!!")
            
        except FileNotFoundError:
            print(f'{index} file not found!!!')
            continue
        
        # Remove negative pressure reading vals
        for i, row2 in curve_df.iterrows():
            if row2['dp_venturi_mean'] < 0:
                curve_df.drop(i, inplace=True)
        
        # GP regression from existing chic
        mean_psi, _ = load_with_gp_smooth(curve_df["flow_coefficient_mean"], curve_df["pressure_rise_coefficient_mean"], PHI_COMMON)
        
        # Save params and psi values in list
        data_as_dict = row._asdict()
        data_as_dict["psi"] = mean_psi
        runs.append(data_as_dict)
        
    return runs

# ─────────────────────────────────────────────────────────────────────────────
#  3.  PCA  (SVD on mean-centred curve matrix)
# ─────────────────────────────────────────────────────────────────────────────

class PCAModel:
    def __init__(self, runs: list[dict], n_modes: int = 5):
        self.runs    = runs
        self.n_modes = n_modes
        self._fit()

    def _fit(self):
        X = np.vstack([r["psi"] for r in self.runs])
        self.mean_curve = X.mean(axis=0)
        Xc = X - self.mean_curve

        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.singular_values = S
        total_var = (S**2).sum()
        self.var_explained = (S**2) / total_var * 100
        self.cum_var = np.cumsum(self.var_explained)

        k = self.n_modes
        self.modes  = Vt[:k]
        self.scores = Xc @ Vt[:k].T

        # ← use PARAM_KEYS so this always matches GPSurrogates exactly
        self.params = np.array(
            [[float(r[key]) for key in PARAM_KEYS] for r in self.runs]
        )

    def reconstruct(self, mode_scores: np.ndarray) -> np.ndarray:
        return self.mean_curve + (mode_scores @ self.modes)


# ─────────────────────────────────────────────────────────────────────────────
#  4.  GP SURROGATES  (one GP per mode score)
# ─────────────────────────────────────────────────────────────────────────────

class GPSurrogates:
    def __init__(self, pca: PCAModel):
        self.pca = pca
        self.lo  = PARAM_LO
        self.hi  = PARAM_HI
        self.gps = []
        self._fit()

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.lo) / (self.hi - self.lo)

    def _fit(self):
        Xn = self._normalise(self.pca.params)   # [N × 8]  ✓
        self.gps = []
        for k in range(self.pca.n_modes):
            y = self.pca.scores[:, k]
            kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
                      Matern(length_scale=np.ones(len(PARAM_KEYS)),
                             length_scale_bounds=(1e-2, 10.0), nu=2.5) +
                      WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1)))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                          normalize_y=True, alpha=0.0)
            gp.fit(Xn, y)
            self.gps.append(gp)
        print(f"[GP] Fitted {self.pca.n_modes} GP surrogates on {len(self.pca.runs)} runs.")

    def predict(self, param_vals: np.ndarray):
        pn = self._normalise(np.array(param_vals, dtype=float).reshape(1, -1))
        means, stds = [], []
        for gp in self.gps:
            m, s = gp.predict(pn, return_std=True)
            means.append(float(m[0]))
            stds.append(float(s[0]))
        return np.array(means), np.array(stds)

    def loo_rms(self) -> float:
        errors = []
        for r in self.pca.runs:
            param_vals = np.array([float(r[k]) for k in PARAM_KEYS])
            scores_m, _ = self.predict(param_vals)
            pred  = self.pca.reconstruct(scores_m)
            truth = r["psi"]
            errors.append(np.sqrt(np.mean((pred - truth)**2)))
        return float(np.mean(errors))


# ─────────────────────────────────────────────────────────────────────────────
#  5.  INTERACTIVE GUI
# ─────────────────────────────────────────────────────────────────────────────

class PCADoEApp:
    N_MODES_MAX = 6
    
    def __init__(self):#, run_data):
        print("Loading DoE dataset …")
        self.runs    = load_data()
        self.n_modes = 4
        print(f"[DoE] {len(self.runs)} runs loaded.")
        ### DEBUG
        # # Verify every run has all expected keys
        # for i, r in enumerate(self.runs):
        #     missing = [k for k in PARAM_KEYS if k not in r]
        #     extra   = [k for k in r if k not in PARAM_KEYS and k != "psi"]
        #     if missing: print(f"Run {i}: MISSING keys {missing}")
        #     if extra:   print(f"Run {i}: UNEXPECTED keys {extra}")

        # # Verify shape consistency
        # params_check = np.array([[float(r[k]) for k in PARAM_KEYS] for r in self.runs])
        # print(f"params matrix shape: {params_check.shape}")   # should be (N_runs, 8)
        # print(f"PARAM_LO shape:      {PARAM_LO.shape}")       # should be (8,)
        print("Running PCA …")
        self.pca     = PCAModel(self.runs, n_modes=self.n_modes)
        print("Training GP surrogates …")
        self.gp      = GPSurrogates(self.pca)
        self.loo_err = self.gp.loo_rms()
        print(f"[GP] LOO RMS ψ error = {self.loo_err:.5f}")
        self._build_gui()

    # ── Layout ─────────────────────────────────────────────────────────────

    def _build_gui(self):
        self.fig = plt.figure(figsize=(20, 14), facecolor=BG)
        self.fig.canvas.manager.set_window_title(
            "PCA + GP Surrogate — DoE Compressor Characteristic Tool")

        # Outer grid: left controls | right plots
        outer = gridspec.GridSpec(1, 3, figure=self.fig,
                                  width_ratios=[1, 0.4, 2.6], wspace=0.2,
                                  left=0.08, right=0.98, top=0.95, bottom=0.04)

        # Left: controls
        left_gs = gridspec.GridSpecFromSubplotSpec(
            30, 1, subplot_spec=outer[0], hspace=0.3)
        
        middle_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[1], hspace=0.3)

        # Right: 2×2 plot grid
        right_gs = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[2], hspace=0.38, wspace=0.32)

        self.ax_main   = self.fig.add_subplot(right_gs[0, :])   # full top row
        self.ax_scree  = self.fig.add_subplot(right_gs[1, 0])
        self.ax_modes  = self.fig.add_subplot(right_gs[1, 1])

        # ── Sliders (design params) ──
        slider_axes = []
        for i in range(len(PARAM_BOUNDS)):
            ax = self.fig.add_subplot(left_gs[i * 2 + 1])
            slider_axes.append(ax)

        self.sliders = {}
        param_items  = list(PARAM_BOUNDS.items())
        for i, (label, (key, lo, hi, init)) in enumerate(param_items):
            ax = slider_axes[i]
            ax.set_facecolor(PANEL)
            if key == "n_blade":
                sl = Slider(ax, label, lo, hi, valinit=init, color=CYAN,
                            track_color=BORDER, valstep=1)
            else:
                sl = Slider(ax, label, lo, hi, valinit=init, color=CYAN,
                            track_color=BORDER)
            sl.label.set_color(WHITE)
            sl.label.set_fontsize(7.5)
            sl.valtext.set_color(CYAN)
            sl.valtext.set_fontsize(7.5)
            sl.on_changed(self._on_slider)
            self.sliders[key] = sl
            
        # ── Tick boxes (compare design params) ──
        param_items  = list(PARAM_BOUNDS.items())
        ax = self.fig.add_subplot(middle_gs[0])
        ax.set_facecolor(PANEL)
        self.ticks = CheckButtons(ax, PARAM_NAMES, actives = [False]*len(PARAM_NAMES),
                        useblit=[False]*len(PARAM_NAMES),
                        frame_props={
                            'edgecolor': 'black',  # box border colour
                            'facecolor': 'white',                    # box fill colour
                            's': 200,                                # box size (points²)
                            'linewidth': 2,                          # border thickness
                        },
                        check_props={
                            # 'facecolor': ['red', 'green', 'blue'],   # tick/check colour
                            'color': 'orange',                       # alternative: single colour for all
                            's': 200,                                # check mark size (must match frame)
                            'linewidth': 2,
                        },
                        )
        self.ticks.on_clicked(self._on_tick)

        # ── n_modes radio ──
        ax_radio = self.fig.add_subplot(left_gs[2*len(PARAM_BOUNDS)+1:2*len(PARAM_BOUNDS)+4])
        ax_radio.set_facecolor(PANEL)
        self.radio_modes = RadioButtons(
            ax_radio, [str(i) for i in range(1, self.N_MODES_MAX + 1)],
            active=self.n_modes - 1,
            activecolor=CYAN)
        for lbl in self.radio_modes.labels:
            lbl.set_color(WHITE); lbl.set_fontsize(7)
        ax_radio.set_title("Modes k", color=AMBER, fontsize=7, pad=2)
        self.radio_modes.on_clicked(self._on_mode_change)

        # ── Info text box ──
        ax_info = self.fig.add_subplot(left_gs[2*len(PARAM_BOUNDS)+5:2*len(PARAM_BOUNDS)+8])
        ax_info.set_facecolor(PANEL)
        ax_info.axis("off")
        self.info_text = ax_info.text(
            0.05, 0.95, "", transform=ax_info.transAxes,
            va="top", ha="left", fontsize=7, color=WHITE,
            fontfamily="monospace")

        # ── Export button ──
        ax_btn = self.fig.add_subplot(left_gs[2*len(PARAM_BOUNDS)+13])
        ax_btn.set_facecolor(PANEL)
        self.btn_export = Button(ax_btn, "⬇  Export CSV",
                                  color=BORDER, hovercolor=CYAN)
        self.btn_export.label.set_color(WHITE)
        self.btn_export.label.set_fontsize(8)
        self.btn_export.on_clicked(self._export_csv)

        # ── Title ──
        self.fig.text(0.01, 0.975,
                      "PCA + GP SURROGATE  ·  DoE CHARACTERISTIC RECONSTRUCTION",
                      fontsize=9, color=CYAN, fontweight="bold",
                      fontfamily="monospace")

        self._draw_all()
        plt.show()

    # ── Draw / update ───────────────────────────────────────────────────────

    def _get_params(self):
        return ([self.sliders[key].val for key in PARAM_KEYS])
    
    def _get_tick(self):
        checked_box_label = list(self.ticks.get_checked_labels())
        box_number = None
        if len(checked_box_label)>1:
            self.ticks.clear()
        elif len(checked_box_label)==1:
            for i, label in enumerate(PARAM_NAMES):
                if checked_box_label[0] == label:
                    box_number = i
                    break
            return box_number
        else:
            return 

    def _draw_all(self):
        self._draw_main()
        self._draw_scree()
        self._draw_modes()
        self._update_info()
        self.fig.canvas.draw_idle()

    def _draw_main(self):
        ax = self.ax_main
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.grid(True)

        param_vals  = self._get_params()
        scores_m, scores_s = self.gp.predict(param_vals)
        
        checked_box = self._get_tick()
        if checked_box != None:
            param = list(PARAM_BOUNDS.values())[checked_box]
            if checked_box != 7:
                plot_vals = np.linspace(param[1],param[2],8)
            else:
                plot_vals = np.linspace(param[1],param[2],5)

        # Mean prediction
        psi_pred  = self.pca.reconstruct(scores_m)

        # Uncertainty bands: propagate GP score uncertainty through linear reconstruction
        # var(ψ_j) = Σ_k  (mode_k_j)^2 * var(s_k)
        psi_var   = np.zeros(N_PHI)
        for k in range(self.n_modes):
            psi_var += (self.pca.modes[k] * scores_s[k])**2
        psi_std  = np.sqrt(psi_var)

        # # Ground truth
        # psi_true = true_characteristic(sol, cam, tip)

        # DoE runs (faded)
        for r in self.runs:
            ax.plot(PHI_COMMON, r["psi"], color=GREY, alpha=0.12, lw=0.6)

        # Mean curve
        ax.plot(PHI_COMMON, self.pca.mean_curve,
                color=WHITE, lw=1.2, ls="--", alpha=0.2, label="Mean curve")

        # GP ±1σ, ±2σ bands
        ax.fill_between(PHI_COMMON, psi_pred - 2*psi_std, psi_pred + 2*psi_std,
                        color=CYAN, alpha=0.10, label="GP ±2σ")
        ax.fill_between(PHI_COMMON, psi_pred - psi_std, psi_pred + psi_std,
                        color=CYAN, alpha=0.22, label="GP ±1σ")

        # Predicted
        ax.plot(PHI_COMMON, psi_pred,  color=CYAN,  lw=2.5,
                label=f"GP prediction ({self.n_modes} modes)", zorder=7)
        
        if checked_box != None:
            n = len(plot_vals)
            cmap = cm.coolwarm
            for i,val in enumerate(plot_vals):
                param_vals[checked_box] = val
                scores_m, scores_s = self.gp.predict(param_vals)
                # Mean prediction
                psi_pred  = self.pca.reconstruct(scores_m)
                
                ax.plot(PHI_COMMON, psi_pred,  color=cmap(i / (n - 1)),  lw=1.5, alpha=0.6,
                        label=f"{PARAM_KEYS[checked_box]}={val:.2f}", zorder=6)
        
        

        # rms = np.sqrt(np.mean((psi_pred - psi_true)**2))
        title = ""
        for i, param in enumerate(PARAM_KEYS):
            title += f"{param}={param_vals[i]}, "
        ax.set_title(
            title,#    RMS error = {rms:.5f}",
            color=AMBER, fontsize=8, pad=4)
        ax.set_xlabel("Flow coefficient  φ  [—]")
        ax.set_ylabel("Pressure rise coeff  ψ  [—]")
        ax.legend(loc="upper right", framealpha=0.7)
        ax.set_ylim(0.1,0.5)
        ax.set_xlim(0.04,0.26)

        # self._rms = rms
        self._scores_m = scores_m
        self._scores_s = scores_s
        self._psi_pred = psi_pred

    def _draw_scree(self):
        ax = self.ax_scree
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.grid(True)

        k_range = np.arange(1, min(len(self.pca.singular_values), 10) + 1)
        ind_var  = self.pca.var_explained[:len(k_range)]
        cum_var  = self.pca.cum_var[:len(k_range)]

        ax.bar(k_range, ind_var, color=CYAN, alpha=0.6, label="Individual %")
        ax.plot(k_range, cum_var, "o-", color=AMBER, lw=1.5, label="Cumulative %")
        ax.axhline(95, color=RED, ls="--", lw=0.8, alpha=0.7, label="95% threshold")
        ax.axvline(self.n_modes, color=GREEN, ls=":", lw=1.2, alpha=0.8,
                   label=f"k = {self.n_modes}")
        ax.set_xlabel("Mode index k")
        ax.set_ylabel("Variance explained [%]")
        ax.set_xlim(0,self.N_MODES_MAX+2)
        ax.set_title("Scree plot", color=AMBER, fontsize=8)
        ax.legend(fontsize=6)
        ax.set_xticks(k_range)

    def _draw_modes(self):
        ax = self.ax_modes
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.grid(True)
        ax.axhline(0, color=GREY, lw=0.7, ls="--")

        ax.plot(PHI_COMMON, self.pca.mean_curve,
                color=GREY, lw=1.2, ls="--", alpha=0.7, label="Mean ψ(φ)")
        for k in range(self.n_modes):
            c = MCOLORS[k % len(MCOLORS)]
            ax.plot(PHI_COMMON, self.pca.modes[k], color=c, lw=1.5,
                    label=f"Mode {k+1}  ({self.pca.var_explained[k]:.1f}%)")

        ax.set_xlabel("Flow coefficient  φ  [—]")
        ax.set_ylabel("Mode amplitude  [—]")
        ax.set_title("Principal mode shapes", color=AMBER, fontsize=8)
        ax.legend(fontsize=6)

    def _update_info(self):
        param_vals = self._get_params()
        lines = [
            f"DoE runs  : {len(self.runs)}",
            f"φ points  : {N_PHI}",
            f"Modes k   : {self.n_modes}",
            f"Var capt. : {self.pca.cum_var[self.n_modes-1]:.1f}%",
            "",
            "── GP predicted scores ──",
        ]
        for k in range(self.n_modes):
            m = self._scores_m[k]
            s = self._scores_s[k]
            lines.append(f"  s{k+1}: {m:+.4f}  ±{s:.4f}")
        lines += [
            "",
            # f"RMS ψ error  : {self._rms:.5f}",
            f"LOO RMS avg  : {self.loo_err:.5f}",
            "",
            "── Kernel (Mode 1) ──",
        ]
        k_str = str(self.gp.gps[0].kernel_)
        # wrap long kernel string
        for chunk in [k_str[i:i+32] for i in range(0, min(len(k_str), 96), 32)]:
            lines.append("  " + chunk)

        self.info_text.set_text("\n".join(lines))

    # ── Callbacks ───────────────────────────────────────────────────────────

    def _on_slider(self, _val):
        self._draw_main()
        self._update_info()
        self.fig.canvas.draw_idle()
    
    def _on_tick(self, _label):
        self._draw_main()
        self._update_info()
        self.fig.canvas.draw_idle()

    def _on_mode_change(self, label: str):
        k = int(label)
        if k == self.n_modes:
            return
        self.n_modes = k
        print(f"[PCA] Refitting with k = {k} modes …")
        self.pca = PCAModel(self.runs, n_modes=k)
        print("[GP] Retraining GP surrogates …")
        self.gp  = GPSurrogates(self.pca)
        self.loo_err = self.gp.loo_rms()
        self._draw_all()

    def _export_csv(self, _event):
        sol, cam, tip = self._get_params()
        fname = (f"prediction_s{sol:.2f}_c{self.sliders['camber'].val:.2f}"
                 f"_t{self.sliders['tip_clearance'].val:.2f}.csv")
        path = os.path.join(os.path.expanduser("~"), fname)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phi", "psi_predicted", "psi_truth",
                        "psi_mean_curve"] +
                       [f"mode{k+1}_shape" for k in range(self.n_modes)])
            truth = true_characteristic(sol, cam, tip)
            for i, phi in enumerate(PHI_COMMON):
                row = [phi, self._psi_pred[i], truth[i], self.pca.mean_curve[i]]
                for k in range(self.n_modes):
                    row.append(self.pca.modes[k][i])
                w.writerow([f"{v:.6f}" for v in row])
        print(f"[Export] Saved → {path}")
        self.ax_main.set_title(
            f"Exported → {fname}", color=GREEN, fontsize=7, pad=2)
        self.fig.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
#  6.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PCADoEApp()
