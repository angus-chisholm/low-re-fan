"""
pca_core.py
===========
Shared constants, palette, data-loading, PCA, and GP surrogate classes.
Imported by both the ψ-tab and η-tab GUI modules.
"""

import numpy as np
import pandas as pd
import matplotlib
import re
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import librosa
import scipy.signal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import warnings, os
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ─── Colour palette ──────────────────────────────────────────────────────────
BG      = "#ffffffef"
BLACK   = "#000000"
PANEL   = "#FFFFFF"
BORDER  = "#0d1d31"
CYAN    = "#00d4ff"
AMBER   = "#ffb347"
GREEN   = "#5fffaa"
RED     = "#ff5a5a"
PINK    = "#ff79c6"
GREY    = "#445566"
WHITE   = "#d0e8ff"
PURPLE  = "#921dd6"
LIME    = "#97e60f"
COLOUR_CHIC = "#f0293a"
COLOUR_ETA  = "#a958df"
MCOLORS = [CYAN, AMBER, GREEN, PINK, RED, "#bd93f9"]
COLOUR_LIST = [CYAN, AMBER, GREEN, RED, PINK, WHITE, PURPLE, LIME]
COLOURMAP = cm.winter

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   BLACK,
    "xtick.color":       BLACK,
    "ytick.color":       BLACK,
    "text.color":        BLACK,
    "grid.color":        GREY,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "font.family":       "monospace",
    "font.size":         8,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  BORDER,
    "legend.fontsize":   7,
})

# ─────────────────────────────────────────────────────────────────────────────
#  Parameters
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
PARAM_KEYS     = [v[0] for v in PARAM_BOUNDS.values()]
PARAM_LO       = np.array([v[1] for v in PARAM_BOUNDS.values()], dtype=float)
PARAM_HI       = np.array([v[2] for v in PARAM_BOUNDS.values()], dtype=float)
PARAM_DEFAULTS = np.array([v[3] for v in PARAM_BOUNDS.values()], dtype=float)

N_PHI      = 100
PHI_COMMON = np.linspace(0.05, 0.25, N_PHI)

# ─────────────────────────────────────────────────────────────────────────────
#  GP smoother helper
# ─────────────────────────────────────────────────────────────────────────────
def load_with_gp_smooth(phi_raw, y_raw, phi_common):
    phi_raw = np.array(phi_raw).reshape(-1, 1)
    y_raw   = np.array(y_raw)
    kernel  = (Matern(length_scale=0.05, length_scale_bounds=(1e-3, 1.0), nu=2.5)
               + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                  normalize_y=True)
    try:
        gp.fit(phi_raw, y_raw)
    except ValueError as e:
        print(f"GP smooth error: {e}")
    mean, std = gp.predict(phi_common.reshape(-1, 1), return_std=True)
    return mean, std

# ─────────────────────────────────────────────────────────────────────────────
#  Data loader  — returns runs with "psi" AND "eta" on PHI_COMMON
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    params_df = pd.read_csv("doe_params.csv", index_col=0)
    runs = []
    with tqdm(total=len(params_df)) as pbar:
        for row in params_df.itertuples():
            index              = row.Index

            stl_file = None
            for file in os.listdir("stl_files"):
                if re.search(f"DOE_{index}", file):
                    if file.split(".")[-1] == "stl":
                        stl_file = file.rstrip(".stl")
            if stl_file is None:
                raise ValueError(f"No STL file found for run {index}")

            filename = f"data/doe_data/{stl_file}.csv"
            try:
                curve_df = pd.read_csv(filename)
            except FileNotFoundError:
                print(f"{index} file not found!")
                continue

            # Drop negative venturi readings
            curve_df = curve_df[curve_df["dp_venturi_mean"] >= 0].copy()
            start_phi = float(np.min(curve_df["flow_coefficient_mean"]))
            end_phi = float(np.max(curve_df["flow_coefficient_mean"]))

            # ψ characteristic
            mean_psi, _ = load_with_gp_smooth(
                curve_df["flow_coefficient_mean"],
                curve_df["pressure_rise_coefficient_mean"],
                PHI_COMMON,
            )

            # Efficiency characteristic
            mean_eta, _ = load_with_gp_smooth(
                curve_df["flow_coefficient_mean"],
                curve_df["efficiency_mean" ],
                PHI_COMMON,
            )  
            
            # Import audio
            audio_filename = f"audio/doe_data/{stl_file}.wav"
            audio_data,sample_rate = librosa.load(audio_filename, sr=None)

            data_as_dict        = row._asdict()
            data_as_dict["psi"] = mean_psi
            data_as_dict["eta"] = mean_eta
            data_as_dict["phi_start"] = start_phi
            data_as_dict["phi_end"] = end_phi
            runs.append(data_as_dict)
            pbar.update(1)

    return runs


# ─────────────────────────────────────────────────────────────────────────────
#  PCA model
# ─────────────────────────────────────────────────────────────────────────────
class PCAModel:
    def __init__(self, runs: list, curve_key: str = "psi", n_modes: int = 5):
        self.runs       = runs
        self.curve_key  = curve_key
        self.n_modes    = n_modes
        self._fit()

    def _fit(self):
        X = np.vstack([r[self.curve_key] for r in self.runs])
        self.mean_curve = X.mean(axis=0)
        Xc = X - self.mean_curve

        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.singular_values = S
        total_var            = (S ** 2).sum()
        self.var_explained   = (S ** 2) / total_var * 100
        self.cum_var         = np.cumsum(self.var_explained)

        k             = self.n_modes
        self.modes    = Vt[:k]
        self.scores   = Xc @ Vt[:k].T
        self.params   = np.array(
            [[float(r[key]) for key in PARAM_KEYS] for r in self.runs]
        )

    def reconstruct(self, mode_scores: np.ndarray) -> np.ndarray:
        return self.mean_curve + (mode_scores @ self.modes)
    
    def return_mask(self, phi_start: float, phi_end: float) -> np.ndarray:
        mask = (phi_start <= PHI_COMMON) & (PHI_COMMON <= phi_end)
        return mask

# ─────────────────────────────────────────────────────────────────────────────
#  GP surrogates
# ─────────────────────────────────────────────────────────────────────────────
class GPSurrogates:
    def __init__(self, pca: PCAModel):
        self.pca = pca
        self.lo  = PARAM_LO
        self.hi  = PARAM_HI
        self.gps = []
        self.gp_phi_ends = {}
        self._fit()

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.lo) / (self.hi - self.lo)

    def _fit(self):
        Xn = self._normalise(self.pca.params)
        # GP for mode shapes
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
            
        # GP for start/end values
        phi_ends = {}
        for lab in ["phi_start","phi_end"]:
            phi_ends[lab] = np.array([r[lab] for r in self.pca.runs])
            kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
                        Matern(length_scale=np.ones(len(PARAM_KEYS)),
                                length_scale_bounds=(1e-2, 10.0), nu=2.5) +
                        WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1)))
            gp = GaussianProcessRegressor(kernel=kernel,
                                                n_restarts_optimizer=5,
                                                normalize_y=True, alpha=0.0)
            gp.fit(Xn, phi_ends[lab])
            self.gp_phi_ends[lab] = gp
        
        print(f"[GP] Fitted {self.pca.n_modes} GP surrogates on {len(self.pca.runs)} runs.")
        print(f"Phi starts from: {phi_ends["phi_start"].min():.3f}-{phi_ends["phi_start"].max():.3f}")
        print(f"Phi ends from: {phi_ends["phi_end"].min():.3f}-{phi_ends["phi_end"].max():.3f}")

    def predict(self, param_vals: np.ndarray):
        pn = self._normalise(np.array(param_vals, dtype=float).reshape(1, -1))
        means, stds = [], []
        for gp in self.gps:
            m, s = gp.predict(pn, return_std=True)
            means.append(float(m[0]))
            stds.append(float(s[0]))
            
        # Prediction for start/end of phi
        phi_start_m, phi_start_s = self.gp_phi_ends["phi_start"].predict(pn, return_std = True)
        phi_end_m, phi_end_s = self.gp_phi_ends["phi_end"].predict(pn, return_std = True)
    
        return np.array(means), np.array(stds), float(phi_start_m[0]), float(phi_start_s[0]), float(phi_end_m[0]), float(phi_end_s[0])

    def loo_rms(self) -> float:
        errors = []
        for r in self.pca.runs:
            param_vals = np.array([float(r[k]) for k in PARAM_KEYS])
            scores_m, _, phi_s_m, _, phi_e_m, _ = self.predict(param_vals)
            pred  = self.pca.reconstruct(scores_m)
            mask = self.pca.return_mask(phi_s_m, phi_e_m)
            pred = pred[mask]
            truth = r[self.pca.curve_key][mask]
            errors.append(np.sqrt(np.mean((pred - truth) ** 2)))
        return float(np.mean(errors))
