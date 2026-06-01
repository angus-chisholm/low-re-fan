"""
Fan Inverse Design — Multi-Objective Optimiser (pymoo)
=======================================================
Uses NSGA-II (via pymoo) to find Pareto-optimal fan geometries that:
  1. Satisfy fan curve constraints (curve must pass through target Q–ΔP points)
  2. Maximise total-to-static efficiency
  3. Minimise noise

Surrogate models: scikit-learn GaussianProcessRegressor (one per output).

Install dependency:
    pip install pymoo

Usage
-----
1. Train your three GPs (pressure, efficiency, noise) on your existing data.
2. Plug them in at the bottom under "PLUG IN YOUR GPs HERE".
3. Set TARGET_CURVE_POINTS to your required operating points.
4. Run:  python fan_inverse_design_optimiser.py
"""

import numpy as np
import warnings
import joblib
warnings.filterwarnings("ignore")

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from pca_core import (
        load_data, remove_outliers_from_smoothing, remove_outliers_redo_PCA, calculate_oaspl,
        PCAModel, GPSurrogates,
        PARAM_BOUNDS, PARAM_NAMES, PARAM_KEYS, PARAM_LO, PARAM_HI, PARAM_DEFAULTS,
        PHI_COMMON, N_PHI, CENTRE_FREQS
    )

# ---------------------------------------------------------------------------
# Parameter space
# (label, internal_key, lower_bound, upper_bound)
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

PARAM_KEYS = [v[0] for v in PARAM_BOUNDS.values()]
LOWER      = np.array([v[1] for v in PARAM_BOUNDS.values()], dtype=float)
UPPER      = np.array([v[2] for v in PARAM_BOUNDS.values()], dtype=float)

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
phi_target = ((q_des_target/3600)/area)/U
TARGET_CURVE_POINT = (psi_target, phi_target)

# Constraint tolerance: sum of squared normalised curve errors must be below this
# 0.05 ≈ each point within ~22% of target on average; tighten to 0.01 for stricter fit
CONSTRAINT_TOL = 0.05


# ---------------------------------------------------------------------------
# Curve constraint helper
# ---------------------------------------------------------------------------

def curve_constraint_penalty(x: np.ndarray, p_curve, e_curve, phi_end_m) -> float:
    penalty = 0.0
    (p_target,flow_target) = TARGET_CURVE_POINT
    
    # Two points - 1 at max p rise, 2 at max efficiency flow rate (i.e. design flow rate)
    
    mask = PHI_COMMON < phi_end_m
    
    p_max_pred = np.max(p_curve[mask])
    q_des_pred = PHI_COMMON[np.argmax(e_curve[mask])]
    err_p = (p_max_pred - p_target) / abs(p_target)
    err_q = (q_des_pred - flow_target) / abs(flow_target)
    
    penalty += err_p ** 2 + err_q ** 2
    
    return penalty


# ---------------------------------------------------------------------------
# pymoo Problem definition
# ---------------------------------------------------------------------------

class FanDesignProblem(Problem):
    """
    Two objectives (minimise):
        F[:, 0] = -η_ts       (negate so pymoo minimises → we maximise η)
        F[:, 1] =  noise [dB]

    One inequality constraint (pymoo requires G ≤ 0):
        G[:, 0] = curve_penalty - CONSTRAINT_TOL
    """

    def __init__(self, pca_pressure, pca_efficiency, pca_noise,
            gp_pressure, gp_efficiency, gp_noise):
        super().__init__(
            n_var       = len(PARAM_KEYS),
            n_obj       = 2,
            n_ieq_constr= 1,
            xl          = LOWER,
            xu          = UPPER,
        )
        self.pca_pressure   = pca_pressure
        self.pca_efficiency = pca_efficiency
        self.pca_noise      = pca_noise
        self.gp_pressure   = gp_pressure
        self.gp_efficiency = gp_efficiency
        self.gp_noise      = gp_noise

    def _evaluate(self, X, out, *args, **kwargs):
        # Ensure blade number is an integer (lazy rounding but it's a bit late to be doing proper mixed variable oh well)
        X = X.copy()
        X[:, 7] = np.round(X[:, 7]).astype(int) 

        n = len(X)

        max_etas = []
        oaspls = []
        cv = []
        for i in range(n):
            p_curve = self.pca_pressure.reconstruct(self.gp_pressure.predict(X[i])[0])
            e_curve = self.pca_efficiency.reconstruct(self.gp_efficiency.predict(X[i])[0])
            n_curve = self.pca_noise.reconstruct(self.gp_noise.predict(X[i])[0])
            phi_end_m = self.gp_pressure.predict(X[i])[4]
            max_etas.append(np.max(e_curve))
            oaspls.append(calculate_oaspl(n_curve))
            cv.append(curve_constraint_penalty(X[i], p_curve, e_curve, phi_end_m))

        max_etas = np.array(max_etas)
        oaspls = np.array(oaspls)
        cv = np.array(cv)


        out["F"] = np.column_stack([-max_etas, oaspls])[:, None]     # objectives
        out["G"] = (cv - CONSTRAINT_TOL)[:, None]               # constraint ≤ 0


# ---------------------------------------------------------------------------
# Run optimisation
# ---------------------------------------------------------------------------

def run_optimisation(
    pca_pressure, 
    pca_efficiency,
    pca_noise,
    gp_pressure,
    gp_efficiency,
    gp_noise,
    pop_size: int = 200,
    n_gen:    int = 150,
    seed:     int = 42,
    verbose:  bool = True,
):
    """
    Run NSGA-II and return a results dict.

    Parameters
    ----------
    gp_pressure, gp_efficiency, gp_noise
        Trained sklearn GaussianProcessRegressor objects.
    pop_size : population size (more = better Pareto coverage, slower)
    n_gen    : number of generations
    seed     : random seed for reproducibility
    verbose  : print pymoo progress

    Returns
    -------
    dict with keys:
        pareto_x     : geometry arrays on Pareto front  (n, n_params)
        pareto_eta   : efficiency values
        pareto_noise : noise values
        pareto_dp    : predicted ΔP at each design's own mdot value
        pymoo_result : raw pymoo Result object for further inspection
    """
    problem = FanDesignProblem(pca_pressure, pca_efficiency, pca_noise,
                                gp_pressure, gp_efficiency, gp_noise)

    algorithm = NSGA2(pop_size=pop_size)

    result = minimize(
        problem,
        algorithm,
        termination = get_termination("n_gen", n_gen),
        seed        = seed,
        verbose     = verbose,
    )

    if result.X is None:
        raise RuntimeError(
            "No feasible solutions found. Try:\n"
            "  • Relaxing CONSTRAINT_TOL (increase above 0.05)\n"
            "  • Checking TARGET_CURVE_POINTS are achievable by your GP\n"
            "  • Increasing pop_size or n_gen"
        )

    pareto_x     = result.X
    pareto_eta   = -result.F[:, 0]   # un-negate
    pareto_noise =  result.F[:, 1]

    # Predict ΔP at each design's own mdot (not the substituted curve points)
    # pareto_dp = gp_pressure.predict(pareto_x)

    return {
        "pareto_x":     pareto_x,
        "pareto_max_eta":   pareto_eta,
        "pareto_oaspl": pareto_noise,
        "pymoo_result": result,
        "param_keys":   PARAM_KEYS,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_pareto_front(results: dict, top_n: int = 15):
    """Print Pareto front sorted by efficiency descending."""
    x     = results["pareto_x"]
    eta   = results["pareto_max_eta"]
    noise = results["pareto_oaspl"]
    # dp    = results["pareto_dp"]
    keys  = results["param_keys"]

    order = np.argsort(-eta)[:top_n]
    n_show = min(top_n, len(order))

    col_w = 13
    header = (f"{'#':>3}  {'η_ts':>7}  {'Noise(dB)':>10}  {'ΔP(Pa)':>8}  "
              + "  ".join(f"{k:>{col_w}}" for k in keys))

    print("\n" + "=" * len(header))
    print(f"  PARETO FRONT  —  top {n_show} designs (sorted by efficiency)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for rank, i in enumerate(order):
        row = (f"{rank+1:>3}  {eta[i]:>7.4f}  {noise[i]:>10.2f}  "
               + "  ".join(f"{x[i, j]:>{col_w}.4f}" for j in range(len(keys))))
        print(row)

    print("=" * len(header))
    print(f"\n  Total Pareto solutions: {len(eta)}")


def plot_pareto_front(results: dict, save_path: str = "pareto_front.png"):
    """Scatter plot of Pareto front: noise vs efficiency."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    eta   = results["pareto_max_eta"]
    noise = results["pareto_oaspl"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        noise, eta * 100,
        c=eta * 100, cmap="viridis",
        s=70, edgecolors="k", linewidths=0.5, zorder=3
    )
    plt.colorbar(sc, ax=ax, label="η_ts [%]")
    ax.set_xlabel("Sound Power Level [dB]", fontsize=12)
    ax.set_ylabel("Total-to-Static Efficiency η_ts [%]", fontsize=12)
    ax.set_title("Pareto Front — Efficiency vs Noise", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Plot saved: {save_path}")


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # STEP 1: Load trained GP models
    # ------------------------------------------------------------------

    #
    # Each GP must be a fitted sklearn GaussianProcessRegressor with:
    #   gp.predict(X)              → mean predictions, shape (n,)
    #   gp.predict(X, return_std=True) → (mean, std)
    # where X has shape (n, 8) matching PARAM_KEYS order.
    
    pca_pressure   = joblib.load("GPS_PKL/pca_pressure.pkl")
    pca_efficiency = joblib.load("GPS_PKL/pca_efficiency.pkl")
    pca_noise      = joblib.load("GPS_PKL/pca_noise.pkl")
    gp_pressure   = joblib.load("GPS_PKL/gp_pressure.pkl")
    gp_efficiency = joblib.load("GPS_PKL/gp_efficiency.pkl")
    gp_noise      = joblib.load("GPS_PKL/gp_noise.pkl")

    # ------------------------------------------------------------------
    # STEP 2: Set target curve points (edit at top of file or here)
    # ------------------------------------------------------------------
    print(f"Target: Max psi = {TARGET_CURVE_POINT[0]}; Design phi = {TARGET_CURVE_POINT[1]}")

    # ------------------------------------------------------------------
    # STEP 3: Run
    # ------------------------------------------------------------------
    print(f"\nRunning NSGA-II  (pop={200}, gen={150}) ...")

    results = run_optimisation(
        pca_pressure, 
        pca_efficiency,
        pca_noise,
        gp_pressure   = gp_pressure,
        gp_efficiency = gp_efficiency,
        gp_noise      = gp_noise,
        pop_size      = 200,
        n_gen         = 150,
        seed          = 42,
        verbose       = True,
    )

    # ------------------------------------------------------------------
    # STEP 4: Inspect results
    # ------------------------------------------------------------------
    print_pareto_front(results, top_n=15)
    plot_pareto_front(results)

    # Programmatic access:
    # results["pareto_x"]      → (n, 8) geometry arrays
    # results["pareto_eta"]    → efficiency for each Pareto solution
    # results["pareto_noise"]  → noise for each Pareto solution
    # results["pymoo_result"]  → full pymoo Result object
