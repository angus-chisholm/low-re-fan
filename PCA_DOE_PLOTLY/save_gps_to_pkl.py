import numpy as np
import joblib
from pca_core import (
        load_data, remove_outliers_from_smoothing, remove_outliers_redo_PCA, calculate_oaspl,
        PCAModel, GPSurrogates,
        PARAM_BOUNDS, PARAM_NAMES, PARAM_KEYS, PARAM_LO, PARAM_HI, PARAM_DEFAULTS,
        PHI_COMMON, N_PHI, CENTRE_FREQS
    )

def _try_load(base_dir="."):
    # global RUNS, PSI_PCA, PSI_GP, PSI_LOO, ETA_PCA, ETA_GP, ETA_LOO, AUDIO_PCA, AUDIO_GP, AUDIO_LOO
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
        holdup = input("press enter to save")
        
        return PSI_PCA, ETA_PCA, AUDIO_PCA, PSI_GP, ETA_GP, AUDIO_GP
        
    except Exception as exc:
        print(f"[warn] Data load failed: {exc}")
        
if __name__ == "__main__":
    PSI_PCA, ETA_PCA, AUDIO_PCA, PSI_GP, ETA_GP, AUDIO_GP = _try_load(base_dir='.')
    joblib.dump(PSI_PCA,   "GPS_PKL/pca_pressure.pkl")
    joblib.dump(ETA_PCA, "GPS_PKL/pca_efficiency.pkl")
    joblib.dump(AUDIO_PCA,    "GPS_PKL/pca_noise.pkl")
    joblib.dump(PSI_GP,   "GPS_PKL/gp_pressure.pkl")
    joblib.dump(ETA_GP, "GPS_PKL/gp_efficiency.pkl")
    joblib.dump(AUDIO_GP,    "GPS_PKL/gp_noise.pkl")