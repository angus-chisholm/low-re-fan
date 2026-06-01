"""
Fan Inverse Design — Multi-Objective Optimiser
================================================
Uses NSGA-II to find Pareto-optimal fan geometries that:
  1. Satisfy fan curve constraints (must pass through target Q–ΔP points)
  2. Maximise total-to-static efficiency
  3. Minimise noise

Surrogate models: scikit-learn GaussianProcessRegressor (one per output).

Usage
-----
1. Train your three GPs (pressure, efficiency, noise) on your existing data.
2. Plug them in at the bottom of this file (see "PLUG IN YOUR GPs HERE").
3. Set your TARGET_CURVE_POINTS and run.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from pca_core import (
    calculate_oaspl
)

# ---------------------------------------------------------------------------
# Parameter definitions  (name, internal key, lower bound, upper bound, default)
# ---------------------------------------------------------------------------
PARAM_BOUNDS = {
    "Mass Flow Rate":    ("mdot",           0.025, 0.075),
    "De Haller Midspan": ("DH_mid",         0.75,  0.90),
    "Incidence Angle":   ("incidence",     -10.0,  5.0),
    "Vortex Exponent":   ("Vexp",          -2.0,   0.0),
    "Compound Lean Max": ("lean_compound",  0.0,   10.0),
    "Straight Lean":     ("lean_straight",  0.0,   90.0),
    "Tip Clearance %":   ("tip_clearance",  1.0,    5.0),
    "Blade Count":       ("n_blade",        5,      9),
}

PARAM_KEYS  = [v[0] for v in PARAM_BOUNDS.values()]
LOWER       = np.array([v[1] for v in PARAM_BOUNDS.values()], dtype=float)
UPPER       = np.array([v[2] for v in PARAM_BOUNDS.values()], dtype=float)
N_PARAMS    = len(PARAM_KEYS)

N_PHI      = 100
PHI_COMMON = np.linspace(0.05, 0.25, N_PHI)

# ---------------------------------------------------------------------------
# Target fan points
# ---------------------------------------------------------------------------
p_rise_max_target = 25 # Pa
q_des_target = 40 # m^3/hr
RPM = 3000
r_mean = 0.0275
U = RPM*np.pi/30 * 0.0275
rho = 1.21
area = np.pi*(0.04**2-0.015**2)
psi_target = p_rise_max_target/(rho*U**2)
phi_target = (q_des_target/area)/U
TARGET_CURVE_POINT = (psi_target, phi_target)

# Penalty weight for curve constraint violation (Pa² per point)
CURVE_PENALTY_WEIGHT = 1e-4   # tune this if constraint dominates or is ignored

# ---------------------------------------------------------------------------
# GP surrogate interface
# ---------------------------------------------------------------------------

@dataclass
class FanSurrogates:
    """
    Holds the three trained GP models.
    Each GP must implement .predict(X, return_std=True) like sklearn GaussianProcessRegressor.
    X shape: (n_samples, n_params)
    """
    gp_pressure:   object  # predicts ΔP_ts [Pa] given geometry x
    gp_efficiency: object  # predicts η_ts [0–1]
    gp_noise:      object  # predicts sound power level [dB]

    def predict(self, X: np.ndarray):
        """
        Returns (pressure, efficiency, noise) as arrays of shape (n,)
        Also returns std arrays for optional uncertainty filtering.
        """
        X = np.atleast_2d(X)
        p,  p_std, _,_,phi_end_m,_  = self.gp_pressure.predict(X,   return_std=True)
        eta, e_std, _,_,_,_ = self.gp_efficiency.predict(X, return_std=True)
        spl, s_std = self.gp_noise.predict(X,      return_std=True)
        return (
            np.array(p).ravel(),
            np.array(eta).ravel(),
            np.array(spl).ravel(),
            np.array(p_std).ravel(),
            np.array(e_std).ravel(),
            np.array(s_std).ravel(),
            phi_end_m
        )
        


# ---------------------------------------------------------------------------
# Curve constraint
# ---------------------------------------------------------------------------

def curve_constraint_penalty(x: np.ndarray, surrogates: FanSurrogates) -> float:
    """
    For each target (Q_target, dP_target), substitute Q_target into x
    (replacing mdot) and query the pressure GP. Returns sum of squared
    normalised errors.

    NOTE: This assumes your pressure GP takes the full geometry vector x
    and that mdot is index 0. Adjust MDOT_IDX if needed.
    """
    penalty = 0.0
    (p_target,flow_target) = TARGET_CURVE_POINT
    
    # Two points - 1 at max p rise, 2 at max efficiency flow rate (i.e. design flow rate)
    x_query = x.copy()
    x_q = np.atleast_2d(x_query)
    dP_pred, eta_pred, _, _, _, _, phi_end_m = surrogates.predict(x_q)
    
    mask = PHI_COMMON < phi_end_m
    
    p_max_pred = np.max(dP_pred[mask])
    q_des_pred = PHI_COMMON[np.argmax(eta_pred[mask])]
    err_p = (p_max_pred - p_target) / abs(p_target)
    err_q = (q_des_pred - flow_target) / abs(flow_target)
    
    penalty += err_p ** 2 + err_q ** 2
    
    return penalty


# ---------------------------------------------------------------------------
# Objective evaluation  (returns values to MINIMISE)
# ---------------------------------------------------------------------------

def evaluate(x: np.ndarray, surrogates: FanSurrogates):
    """
    Returns:
        obj1 : –η_ts         (minimise → maximise efficiency)
        obj2 :  noise [dB]   (minimise directly)
        constraint_viol : sum of squared curve errors (0 = perfectly satisfied)
    """
    x_2d = np.atleast_2d(x)
    _, eta, noise, _, _, _, phi_end_m = surrogates.predict(x_2d)
    mask = PHI_COMMON<phi_end_m
    obj1 = -np.max(eta[mask])
    obj2 =  calculate_oaspl(noise)
    cv   =  curve_constraint_penalty(x, surrogates)
    return obj1, obj2, cv



class FanDesignProblem(Problem):
    def __init__(self, surrogates):
        super().__init__(
            n_var=8,            # your 8 geometry parameters
            n_obj=2,            # minimise –η and noise
            n_ieq_constr=1,     # curve constraint ≤ 0
            xl=LOWER,
            xu=UPPER,
        )
        self.surrogates = surrogates

    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X)
        obj1  = np.zeros(n)   # –η
        obj2  = np.zeros(n)   # noise
        cv    = np.zeros(n)   # curve constraint violation

        for i, x in enumerate(X):
            _, eta, noise, _, _, _ = self.surrogates.predict(x[None, :])
            obj1[i] = -eta[0]
            obj2[i] =  noise[0]
            cv[i]   =  curve_constraint_penalty(x, self.surrogates)

        out["F"] = np.column_stack([obj1, obj2])
        out["G"] = cv - CONSTRAINT_TOL   # pymoo expects G ≤ 0
        

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def print_pareto_front(results: dict, top_n: int = 10):
    """Print the Pareto front sorted by efficiency."""
    x     = results["pareto_x"]
    eta   = results["pareto_eta"]
    noise = results["pareto_noise"]
    dp    = results["pareto_dp"]
    cv    = results["pareto_cv"]
    keys  = results["param_keys"]

    order = np.argsort(-eta)[:top_n]

    print("\n" + "="*80)
    print(f"  PARETO FRONT  —  top {min(top_n, len(order))} designs (sorted by efficiency)")
    print("="*80)
    header = f"{'#':>3}  {'η_ts':>7}  {'Noise(dB)':>10}  {'ΔP(Pa)':>8}  {'CurveErr':>9}  " + \
             "  ".join(f"{k:>12}" for k in keys)
    print(header)
    print("-"*80)
    for rank, i in enumerate(order):
        row = (f"{rank+1:>3}  {eta[i]:>7.4f}  {noise[i]:>10.2f}  "
               f"{dp[i]:>8.1f}  {cv[i]:>9.4f}  " +
               "  ".join(f"{x[i,j]:>12.4f}" for j in range(len(keys))))
        print(row)
    print("="*80)


def plot_pareto_front(results: dict):
    """Plot the Pareto front (efficiency vs noise)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    eta   = results["pareto_eta"]
    noise = results["pareto_noise"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(noise, eta * 100, c=eta * 100, cmap="viridis",
                    s=60, edgecolors="k", linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="η_ts [%]")
    ax.set_xlabel("Sound Power Level [dB]", fontsize=12)
    ax.set_ylabel("Total-to-Static Efficiency η_ts [%]", fontsize=12)
    ax.set_title("Pareto Front — Efficiency vs Noise", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pareto_front.png", dpi=150)
    plt.show()
    print("  Plot saved: pareto_front.png")


# ---------------------------------------------------------------------------
# ============================================================
# PLUG IN YOUR GPs HERE and run
# ============================================================
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # STEP 1: Load or train your GP models
    # Replace the dummy GPs below with your actual trained models.
    # Each must follow the sklearn GaussianProcessRegressor interface:
    #   gp.predict(X, return_std=True) → (mean, std)
    #   X shape: (n_samples, 8)  — one column per PARAM_KEYS entry
    # ------------------------------------------------------------------

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

    # --- REPLACE BELOW with your real trained GPs ---
    # Example: gp_pressure = joblib.load("gp_pressure.pkl")

    # Dummy GPs for demonstration (trained on random data)
    _rng = np.random.default_rng(0)
    _X_dummy = _rng.uniform(LOWER, UPPER, size=(50, N_PARAMS))

    _kernel = C(1.0) * RBF(length_scale=np.ones(N_PARAMS))

    gp_pressure = GaussianProcessRegressor(kernel=_kernel, n_restarts_optimizer=2)
    gp_pressure.fit(_X_dummy, 200 + 100 * _rng.standard_normal(50))

    gp_efficiency = GaussianProcessRegressor(kernel=_kernel, n_restarts_optimizer=2)
    gp_efficiency.fit(_X_dummy, 0.75 + 0.1 * _rng.standard_normal(50))

    gp_noise = GaussianProcessRegressor(kernel=_kernel, n_restarts_optimizer=2)
    gp_noise.fit(_X_dummy, 60 + 5 * _rng.standard_normal(50))
    # --- END REPLACE ---

    surrogates = FanSurrogates(
        gp_pressure   = gp_pressure,
        gp_efficiency = gp_efficiency,
        gp_noise      = gp_noise,
    )

    # ------------------------------------------------------------------
    # STEP 2: Set your target operating points
    # (edit TARGET_CURVE_POINTS at the top of this file)
    # ------------------------------------------------------------------

    print("Target curve points:")
    for Q, dP in TARGET_CURVE_POINTS:
        print(f"  Q = {Q:.3f}  →  ΔP_ts = {dP:.1f} Pa")

    # ------------------------------------------------------------------
    # STEP 3: Run the optimiser
    # ------------------------------------------------------------------

    print("\nRunning NSGA-II...")
    results = nsga2_fan(
        surrogates     = surrogates,
        pop_size       = 200,
        n_gen          = 150,
        constraint_tol = 0.05,   # curve must be within ~5% of targets
        verbose        = True,
    )

    # ------------------------------------------------------------------
    # STEP 4: Inspect results
    # ------------------------------------------------------------------

    print_pareto_front(results, top_n=15)
    plot_pareto_front(results)

    # Access the full Pareto front programmatically:
    # results["pareto_x"]     → shape (n_pareto, 8) geometry arrays
    # results["pareto_eta"]   → efficiency values
    # results["pareto_noise"] → noise values
    # results["pareto_dp"]    → predicted ΔP at design point
