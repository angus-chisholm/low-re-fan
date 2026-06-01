"""
pca_core.py  (web version)
==========================
Shared constants, data-loading, PCA, and GP surrogate classes.
Identical logic to the original; matplotlib removed, librosa kept for the
future audio tab.
"""

import os, re, warnings
import numpy as np
import pandas as pd
import librosa
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import welch, medfilt
from tqdm import tqdm
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Parameters
# ─────────────────────────────────────────────────────────────────────────────
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
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
    try:
        gp.fit(phi_raw, y_raw)
        mask = (phi_raw >= 0.05) & (phi_raw <= 0.25)
        y_pred = gp.predict(phi_raw[mask].reshape(-1,1))
        mask = mask.flatten()
        r2 = r2_score(y_raw[mask], y_pred)
        
        
        # scores = cross_val_score(gp, phi_raw, y_raw, cv=5, scoring='neg_mean_squared_error')
        # # scores = np.array([0])
        # rmse = np.sqrt(-scores.mean())
        # r2_scores = cross_val_score(gp, phi_raw, y_raw, cv=LeaveOneOut(), scoring='r2')
        # r2_mean = np.mean(r2_scores)
        # print(r2_scores)
        # # print(rmse)
    except ValueError as e:
        print(f"GP smooth error: {e}")
    mean, std = gp.predict(phi_common.reshape(-1, 1), return_std=True)
    return mean, std, r2

# ─────────────────────────────────────────────────────────────────────────────
#  Helper function for extrapolation to max phi
# ─────────────────────────────────────────────────────────────────────────────

def axes_crossing(x,y):
        # Assuming all curves are in 1st quadrant and slope at either end of curve will intersect axes
        n_points = 20
        m, c, r, p, se = stats.linregress(x[:n_points], y[:n_points])
        y_crossing = c
        
        m, c, r, p, se = stats.linregress(x[-n_points:], y[-n_points:])
        x_crossing = -c/m
        
        return y_crossing, x_crossing

# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions for 1/n octave a-weighted amplitudes (dBFS)
# ─────────────────────────────────────────────────────────────────────────────

def third_octave_spectrum_and_a_weight(frequencies, amplitude_db):
    """Compute 1/3 octave band levels from a narrow-band spectrum."""    
    
    centre_freqs = np.array([
        10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80,
        100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
        1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
        10000, 12500, 16000, 20000
    ])
    
    a_weighting_db = np.array([
        -70.4, -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5,
        -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8,
         0.0,   0.6,   1.0,   1.2,   1.3,   1.2,   1.0,   0.5,  -0.1,  -1.1,
        -2.5,  -4.3,  -6.6,  -9.3
    ])
    
    # 1/3 octave band edges: fc * 2^(-1/6) to fc * 2^(+1/6)
    lower_edges = centre_freqs * 2**(-1/6)
    upper_edges = centre_freqs * 2**(+1/6)
    
    # Convert amplitude from dB to linear power
    power_linear = 10 ** (amplitude_db / 10)
    
    band_levels = []
    for i, (f_low, f_high) in enumerate(zip(lower_edges, upper_edges)):
        mask = (frequencies >= f_low) & (frequencies < f_high)
        if mask.any():
            band_power = power_linear[mask].sum()
            band_levels.append(10 * np.log10(band_power) + a_weighting_db[i])
        else:
            band_levels.append(np.nan)  # no bins in this band
    
    return centre_freqs, np.array(band_levels)


def get_sixth_octave_centres(f_min=20, f_max=20000):
    centres = []
    # 1/6 octave: fc = 1000 * 2^(n/6)
    n = -30  # start well below f_min
    while True:
        fc = 1000 * 2**(n/6)
        if fc > f_max:
            break
        if fc >= f_min:
            centres.append(fc)
        n += 1
    return np.array(centres)

CENTRE_FREQS = get_sixth_octave_centres(f_min=20, f_max=20000)

def a_weight(frequencies):
    """Compute A-weighting correction (dB) at arbitrary frequencies using the analytic formula."""
    f = np.asarray(frequencies, dtype=float)
    f2 = f**2
    
    numerator   = 12194**2 * f2**2
    denominator = ((f2 + 20.6**2) * 
                   np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * 
                   (f2 + 12194**2))
    
    RA = numerator / denominator
    a_db = 20 * np.log10(RA) + 2.00  # +2.00 normalises to 0dB at 1kHz
    return a_db


def sixth_octave_spectrum_and_a_weight(frequencies, amplitude_db):
    """Compute 1/6 octave band levels from a narrow-band spectrum."""    
    
    centre_freqs = get_sixth_octave_centres(f_min=20, f_max=20000)
    
    a_weighting_db = a_weight(centre_freqs)
    
    # 1/6 octave band edges:
    lower_edges = centre_freqs * 2**(-1/12)
    upper_edges = centre_freqs * 2**(+1/12)
    
    # Convert amplitude from dB to linear power
    power_linear = 10 ** (amplitude_db / 10)
    
    band_levels = []
    for i, (f_low, f_high) in enumerate(zip(lower_edges, upper_edges)):
        mask = (frequencies >= f_low) & (frequencies < f_high)
        if mask.any():
            band_power = power_linear[mask].sum()
            band_levels.append(10 * np.log10(band_power) + a_weighting_db[i])
        else:
            band_levels.append(np.nan)  # no bins in this band
    
    return centre_freqs, np.array(band_levels)

def calculate_oaspl(amplitudes):
    return 10*np.log10(sum(np.power(10,amplitudes/10)))


# ─────────────────────────────────────────────────────────────────────────────
#  Data loader  — returns runs with "psi", "eta", "audio" on PHI_COMMON
# ─────────────────────────────────────────────────────────────────────────────
def load_data(base_dir: str = "."):
    params_df = pd.read_csv(os.path.join(base_dir, "doe_params.csv"), index_col=0)
    runs = []
    r2s_psi = []
    r2s_eta = []
    with tqdm(total=len(params_df), desc="Loading runs") as pbar:
        for row in params_df.itertuples():
            index = row.Index

            # Find matching STL stem
            stl_file = None
            stl_dir  = os.path.join(base_dir, "stl_files")
            for f in os.listdir(stl_dir):
                if re.search(rf"DOE_{index}", f) and f.endswith(".stl"):
                    stl_file = f[:-4]   # strip .stl
            if stl_file is None:
                pbar.update(1)
                raise ValueError(f"No STL file found for run {index}")

            # Aero curve CSV
            csv_path = os.path.join(base_dir, "data", "doe_data", f"{stl_file}.csv")
            try:
                curve_df = pd.read_csv(csv_path)
            except FileNotFoundError:
                print(f"  [skip] {index} — CSV not found")
                pbar.update(1)
                continue

            curve_df = curve_df[curve_df["dp_venturi_mean"] >= 0].copy()
            start_phi = float(curve_df["flow_coefficient_mean"].min())
            end_phi   = float(curve_df["flow_coefficient_mean"].max())

            mean_psi, _, r2_psi = load_with_gp_smooth(
                curve_df["flow_coefficient_mean"],
                curve_df["pressure_rise_coefficient_mean"],
                PHI_COMMON,
            )
            mean_eta, _, r2_eta = load_with_gp_smooth(
                curve_df["flow_coefficient_mean"],
                curve_df["efficiency_mean"],
                PHI_COMMON,
            )
            r2s_psi.append(r2_psi)
            r2s_eta.append(r2_eta)
            
            # ── Audio  (kept for future audio tab) ──────────────────────────
            audio_path = os.path.join(base_dir, "audio", "doe_data_2", f"{stl_file}.wav")
            audio_data, sample_rate = None, None
            if os.path.exists(audio_path):
                try:
                    audio_data, sample_rate = librosa.load(audio_path, sr=None)
                except Exception as e:
                    print(f"  [audio] {stl_file}: {e}")
                    
            nperseg = len(audio_data)//8
            noverlap = nperseg//2  
            
            ## Plot ##
            frequencies, power = welch(
                audio_data,
                fs=sample_rate,
                window='blackman',
                nperseg=nperseg,
                noverlap=noverlap,
                scaling = 'spectrum'
                )

            amplitude = np.sqrt(power)
            amplitude_db = 20 * np.log10(amplitude + 1e-12)  
            
            amplitude_db = medfilt(amplitude_db, kernel_size=19)  # kernel must be odd
            # amplitude_db = np.array(pd.DataFrame(amplitude_db).rolling(window=9, center=True, min_periods=1).mean()).flatten()
            octave_freqs, a_weighted_octave_bands = sixth_octave_spectrum_and_a_weight(frequencies, amplitude_db)

            entry = row._asdict()
            entry["psi"]         = mean_psi
            entry["eta"]         = mean_eta
            entry["phi_start"]   = start_phi
            entry["phi_end"]     = end_phi
            entry["audio_frequencies"]  = octave_freqs
            entry["audio_amplitude"] = a_weighted_octave_bands
            entry["sample_rate"] = sample_rate
            entry["stl_stem"]    = stl_file
            entry["fan_index"]   = index
            runs.append(entry)
            pbar.update(1)
            
    avg_r2_psi, std_r2_psi, min_r2_psi = np.mean(r2s_psi), np.std(r2s_psi), np.min(r2s_psi)
    avg_r2_eta, std_r2_eta, min_r2_eta = np.mean(r2s_eta), np.std(r2s_eta), np.min(r2s_eta)
    if avg_r2_psi == 0 or avg_r2_eta == 0:
        pass
    else:
        print(f"[AVG R2]: psi={avg_r2_psi}, eta={avg_r2_eta}")
        print(f"[STD R2]: psi={std_r2_psi}, eta={std_r2_eta}")
        print(f"[MIN R2]: psi={min_r2_psi}, eta={min_r2_eta}")
    return runs, r2s_psi, r2s_eta

# ─────────────────────────────────────────────────────────────────────────────
#  Outlier removal functions
# ─────────────────────────────────────────────────────────────────────────────

def remove_outliers_from_smoothing(runs, r2s):
    threshold = 0.9
    mask = np.array(r2s)>=threshold
    modified_runs = [r for r, m in zip(runs, mask) if m]
    return modified_runs

def get_nrmse_and_stats_PCA(pca):
    X = np.vstack([r[pca.curve_key] for r in pca.runs])
    X_reconstructed = pca.reconstruct(pca.scores)
    mins = [np.min(x) for x in X]
    maxs = [np.max(x) for x in X]
    
    rmse_per_run = np.array([
        np.sqrt(mean_squared_error(X[i], X_reconstructed[i]))/(maxs[i]-mins[i]) for i in range(len(pca.runs))
    ])*100
    
    return rmse_per_run, np.mean(rmse_per_run), np.median(rmse_per_run)

def remove_outliers_redo_PCA(pca, n_stds=3):
    rmse_per_run, mean_rmse, median_rmse = get_nrmse_and_stats_PCA(pca)
    threshold = mean_rmse + n_stds*np.std(rmse_per_run)
    mask = rmse_per_run<threshold
    # print([i for i, b in enumerate(mask) if b==False])
    runs = [r for r, m in zip(pca.runs, mask) if m]
    pca_new = PCAModel(runs, pca.curve_key, pca.n_modes)
    return pca_new




# ─────────────────────────────────────────────────────────────────────────────
#  PCA model
# ─────────────────────────────────────────────────────────────────────────────
class PCAModel:
    def __init__(self, runs: list, curve_key: str = "psi", n_modes: int = 5):
        self.runs      = runs
        self.curve_key = curve_key
        self.n_modes   = n_modes
        self._fit()

    def _fit(self):
        X = np.vstack([r[self.curve_key] for r in self.runs])
        self.mean_curve = X.mean(axis=0)
        Xc = X - self.mean_curve

        U, S, Vt            = np.linalg.svd(Xc, full_matrices=False)
        self.singular_values = S
        total_var            = (S ** 2).sum()
        self.var_explained   = (S ** 2) / total_var * 100
        self.cum_var         = np.cumsum(self.var_explained)

        k           = self.n_modes
        self.modes  = Vt[:k]
        self.scores = Xc @ Vt[:k].T
        self.params = np.array(
            [[float(r[key]) for key in PARAM_KEYS] for r in self.runs]
        )
        self.fan_indices = np.array([int(r["fan_index"]) for r in self.runs])

    def reconstruct(self, mode_scores: np.ndarray) -> np.ndarray:
        return self.mean_curve + (mode_scores @ self.modes)

    def return_mask(self, phi_start: float, phi_end: float) -> np.ndarray:
        return (phi_start <= PHI_COMMON) & (PHI_COMMON <= phi_end)


# ─────────────────────────────────────────────────────────────────────────────
#  GP surrogates
# ─────────────────────────────────────────────────────────────────────────────
class GPSurrogates:
    def __init__(self, pca: PCAModel):
        self.pca        = pca
        self.lo         = PARAM_LO
        self.hi         = PARAM_HI
        self.gps        = []
        self.gp_phi_ends = {}
        self._fit()

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.lo) / (self.hi - self.lo)

    def _fit(self):
        Xn = self._normalise(self.pca.params)

        for k in range(self.pca.n_modes):
            y = self.pca.scores[:, k]
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3)) *
                Matern(length_scale=np.ones(len(PARAM_KEYS)),
                       length_scale_bounds=(1e-2, 10.0), nu=2.5) +
                WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
            )
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                          normalize_y=True, alpha=0.0)
            gp.fit(Xn, y)
            self.gps.append(gp)

        if self.pca.curve_key == "audio_amplitude":
            pass
        else:
            phi_ends = {}
            for lab in ["phi_start", "phi_end"]:
                phi_ends[lab] = np.array([r[lab] for r in self.pca.runs])
                kernel = (
                    ConstantKernel(1.0, (1e-3, 1e3)) *
                    Matern(length_scale=np.ones(len(PARAM_KEYS)),
                        length_scale_bounds=(1e-2, 10.0), nu=2.5) +
                    WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
                )
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                            normalize_y=True, alpha=0.0)
                gp.fit(Xn, phi_ends[lab])
                self.gp_phi_ends[lab] = gp

        print(f"[GP] Fitted {self.pca.n_modes} surrogates on {len(self.pca.runs)} runs.")

    def predict(self, param_vals: np.ndarray):
        pn = self._normalise(np.array(param_vals, dtype=float).reshape(1, -1))
        means, stds = [], []
        for gp in self.gps:
            m, s = gp.predict(pn, return_std=True)
            means.append(float(m[0]))
            stds.append(float(s[0]))
            
        if self.pca.curve_key == "audio_amplitude":
            return (
                np.array(means), np.array(stds)
            )
        else:
            phi_s_m, phi_s_s = self.gp_phi_ends["phi_start"].predict(pn, return_std=True)
            phi_e_m, phi_e_s = self.gp_phi_ends["phi_end"].predict(pn, return_std=True)
            return (
                np.array(means), np.array(stds),
                float(phi_s_m[0]), float(phi_s_s[0]),
                float(phi_e_m[0]), float(phi_e_s[0]),
            )

    def loo_rms(self) -> float:
        errors = []
        for r in self.pca.runs:
            pv            = np.array([float(r[k]) for k in PARAM_KEYS])
            if self.pca.curve_key == "audio_amplitude":
                scores_m, _, = self.predict(pv)
                pred          = self.pca.reconstruct(scores_m)
                errors.append(np.sqrt(np.mean((pred - r[self.pca.curve_key]) ** 2)))
            else:
                scores_m, _, phi_s_m, _, phi_e_m, _ = self.predict(pv)
                pred          = self.pca.reconstruct(scores_m)
                mask          = self.pca.return_mask(phi_s_m, phi_e_m)
                errors.append(np.sqrt(np.mean((pred[mask] - r[self.pca.curve_key][mask]) ** 2)))
        return float(np.mean(errors))
    
    
