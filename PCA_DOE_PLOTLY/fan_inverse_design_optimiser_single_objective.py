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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import scienceplots
import pandas as pd
warnings.filterwarnings("ignore")

# from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from pymoo.core.variable import Real, Integer
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2

from pca_core import (
        load_data, remove_outliers_from_smoothing, remove_outliers_redo_PCA, calculate_oaspl,
        axes_crossing,
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
input_params = pd.read_csv("fan_inverse_design_parameters.csv", index_col=None, usecols=["Label", "Max airflow (m3/hr)", "Max static (mmh2o)", "Max RPM", "Max Noise", "U", "phi_target", "psi_target"]) # Label | Max airflow (m^3/hr) | Max static pressure rise (mmh2o) | Max RPM | Max Noise
print(input_params.head())
# rho = 1.21 # kg/m^3
# r_mean = 0.0275
# area = np.pi*(0.04**2-0.015**2)
# U = input_params["Max RPM"]*np.pi/30 * 0.0275
# input_params["psi_target"] = input_params["Max static"]*9.81 / (rho * U**2)
# input_params["phi_target"] = ((input_params["Max airflow"]/3600)/area)/U

TARGET_CURVE_POINTS = list(zip(input_params["psi_target"].to_numpy(), input_params["phi_target"].to_numpy()))

# Constraint tolerance: sum of squared normalised curve errors must be below this
# 0.05 ≈ each point within ~22% of target on average; tighten to 0.01 for stricter fit
CONSTRAINT_TOL = 0.05

VARS = {
    "mdot":          Real(bounds=(0.025, 0.075)),
    "DH_mid":        Real(bounds=(0.75,  0.90)),
    "incidence":     Real(bounds=(-10.0,  5.0)),
    "Vexp":          Real(bounds=(-2.0,   0.0)),
    "lean_compound": Real(bounds=(0.0,   10.0)),
    "lean_straight": Real(bounds=(0.0,   90.0)),
    "tip_clearance": Real(bounds=(1.0,    5.0)),
    "n_blade":       Integer(bounds=(5, 9)),
}

plt.style.use('science')
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'  # looks identical to LaTeX
mpl.rcParams['font.size'] = 14        # default for everything
mpl.rcParams['axes.labelsize'] = 14   # axis labels
mpl.rcParams['legend.fontsize'] = 12  # legend
mpl.rcParams['xtick.labelsize'] = 12  # tick labels
mpl.rcParams['ytick.labelsize'] = 12

pca_pressure, pca_efficiency, pca_noise, gp_pressure, gp_efficiency, gp_noise = [None] * 6  # placeholders for global variables to be loaded in main()


# ---------------------------------------------------------------------------
# Curve constraint helper
# ---------------------------------------------------------------------------

def curve_constraint_penalty(x: np.ndarray, p_curve, mask, target_points) -> float:
    penalty = 0.0
    (p_target,flow_target) = target_points
    
    # Two points - 1 at max p rise, 2 at predicted max flow rate (from linear extrapolation)
    
    phi_zero, phi_max = axes_crossing(PHI_COMMON[mask], p_curve)
    
    p_max_pred = np.max(p_curve)
    q_max_pred = phi_max
    err_p = (p_max_pred - p_target) / abs(p_target)
    err_q = (q_max_pred - flow_target) / abs(flow_target)
    
    penalty += err_p ** 2 + err_q ** 2
    
    return penalty


# ---------------------------------------------------------------------------
# pymoo Problem definition
# ---------------------------------------------------------------------------

class FanDesignProblem(Problem):
    """
    Choice of two objectives (minimise):
        Max efficiency (min negative efficiency) or min noise

    One inequality constraint (pymoo requires G ≤ 0):
        G[:, 0] = curve_penalty - CONSTRAINT_TOL
    """

    def __init__(self, target_points, objective = "eta"):
        super().__init__(vars=VARS, n_obj=1, n_ieq_constr=1)
        self.pca_pressure   = pca_pressure
        self.pca_efficiency = pca_efficiency
        self.pca_noise      = pca_noise
        self.gp_pressure   = gp_pressure
        self.gp_efficiency = gp_efficiency
        self.gp_noise      = gp_noise
        self.objective      = objective
        self.target_points = target_points

    def _evaluate(self, X, out, *args, **kwargs):
        # Convert dict-of-columns → 2D numpy array
        # arr = np.column_stack([X[i] for i in range(len(X))])

        n = len(X)
        max_etas = []
        oaspls   = []
        cv       = []

        for i in range(n):
            x = np.array(list(X[i].values()))
            p_curve = self.pca_pressure.reconstruct(self.gp_pressure.predict(x)[0])
            e_curve = self.pca_efficiency.reconstruct(self.gp_efficiency.predict(x)[0])
            n_curve = self.pca_noise.reconstruct(self.gp_noise.predict(x)[0])
            phi_end_m = self.gp_pressure.predict(x)[4]
            mask = PHI_COMMON < phi_end_m
            max_etas.append(np.max(e_curve[mask]))
            oaspls.append(calculate_oaspl(n_curve))
            cv.append(curve_constraint_penalty(x, p_curve[mask], mask, self.target_points))

        max_etas = np.array(max_etas)
        oaspls   = np.array(oaspls)
        cv       = np.array(cv)

        f = -max_etas if self.objective == "eta" else oaspls

        out["F"] = f[:, None]
        out["G"] = (cv - CONSTRAINT_TOL)[:, None]            


# ---------------------------------------------------------------------------
# Run optimisation
# ---------------------------------------------------------------------------

def run_optimisation(
    obj,
    target_points,
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
    problem = FanDesignProblem(target_points, obj)

    algorithm = GA(
        pop_size = pop_size,
        sampling = MixedVariableSampling(),
        mating               = MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        eliminate_duplicates = MixedVariableDuplicateElimination(),
    )

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

    best_x     = result.X
    best_eta   = -result.F[0] if obj=="eta" else None  # un-negate
    best_noise =  result.F[0] if obj=="noise" else None

    print("\n" + "=" * 50)
    print(f"  BEST DESIGN  —  optimised for: {obj}")
    print("=" * 50)
    for key, val in best_x.items():
        print(f"  {key} : {val:.4f}")
    label = "η" if obj == "eta" else "Noise [dBFS]"
    print(f"\n  {label} = {result.F[0]:.4f}")
    print("=" * 50)
    

    return {
        "best_x" : best_x,
        "best_eta" : best_eta,   
        "best_noise" : best_noise,
        "pymoo_result": result,
        "param_keys":   PARAM_KEYS,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def save_output(results):
    # Save best_x (params) to csv for later use
    # results_dict = {
    #     results["param_keys"][i]: results["best_x"][i] for i in range(len(results["param_keys"]))
    # }
    n = len(results)
    results_dict = []
    for i in range(n):
        res = results[i]
        params = {key: res["best_x"][key] for key in res["param_keys"]}
        params["best_eta"] = res["best_eta"]
        params["best_noise"] = res["best_noise"]
        results_dict.append(params)
    df = pd.DataFrame(results_dict)
    df.to_csv("optimisation_results_2.csv", index=False)
    print("Best design parameters saved to optimisation_results_2.csv")

def plot_output(results):
    
    n = len(results)
    
    centre_freqs = CENTRE_FREQS
    
    n_points = 10
    cmap = cm.get_cmap('tab10')  # or 'viridis', 'plasma', 'hsv', etc.
    colours = [cmap(i / n_points) for i in range(n_points)]
    
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=False)
    
    for i in range(n):
        res = results[i]
        params = np.array(list(res["best_x"].values()))
    
        # Predict curves with design params
        p_curve = pca_pressure.reconstruct(gp_pressure.predict(params)[0])
        e_curve = pca_efficiency.reconstruct(gp_efficiency.predict(params)[0])
        n_curve = pca_noise.reconstruct(gp_noise.predict(params)[0])
        
        m_psi, s_psi, s_m_psi, s_s_psi, e_m_psi, e_s_psi = gp_pressure.predict(params)
        m_eta, s_eta, s_m_eta, s_s_eta, e_m_eta, e_s_eta = gp_efficiency.predict(params)
        m_noise, s_noise = gp_noise.predict(params)
        
        phi_end_m = gp_pressure.predict(params)[4]
        mask = PHI_COMMON < phi_end_m
        
        p_curve = p_curve[mask]
        e_curve = e_curve[mask]
        PHI_PLOT = PHI_COMMON[mask]
        
        var       = sum((pca_efficiency.modes[k] * s_eta[k]) ** 2 for k in range(pca_efficiency.n_modes))
        std       = np.sqrt(var)[mask]

        ax[0].plot(PHI_PLOT, e_curve, color='k', label='Predicted')
        ax[0].fill_between(PHI_PLOT, pca_efficiency.reconstruct(m_eta)[mask]-std, pca_efficiency.reconstruct(m_eta)[mask]+std, color=colours[6], label=r'GP predicted $\pm1\sigma$', alpha=0.5, zorder=0)

        # PSI
        var       = sum((pca_pressure.modes[k] * s_psi[k]) ** 2 for k in range(pca_pressure.n_modes))
        std       = np.sqrt(var)[mask]

        ax[1].plot(PHI_PLOT, p_curve, color='k', label='Predicted')
        ax[1].fill_between(PHI_PLOT, pca_pressure.reconstruct(m_psi)[mask]-std, pca_pressure.reconstruct(m_psi)[mask]+std, color=colours[6], alpha=0.5, zorder=0)
        
        
        # Noise
        var       = sum((pca_noise.modes[k] * s_noise[k]) ** 2 for k in range(pca_noise.n_modes))
        std       = np.sqrt(var)

        ax[2].semilogx(centre_freqs, n_curve, color='k', label='Predicted')
        ax[2].fill_between(centre_freqs, pca_noise.reconstruct(m_noise)-std, pca_noise.reconstruct(m_noise)+std, color=colours[6], alpha=0.5, zorder=0)
        
        
    ax[0].grid()
    ax[0].set_xlabel(r'$\phi$')
    ax[0].set_ylabel(r'$\eta$')
    
    ax[1].grid()
    ax[1].set_xlabel(r'$\phi$')
    ax[1].set_ylabel(r'$\psi_{ts}$')
    
    ax[2].grid()
    ax[2].set_xlabel('Frequency (Hz)')
    ax[2].set_ylabel('SPL (dBFS)')
    
    fig.legend(bbox_to_anchor=(1.01, 0.6), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(r'C:\Users\angus\OneDrive\Documents\Cambridge\Year 4\Project\diagrams for final report\predicted_fan_curves.svg', bbox_inches='tight')
    plt.show()
    
    return


def main(target_points):
    global pca_pressure, pca_efficiency, pca_noise, gp_pressure, gp_efficiency, gp_noise
    # ------------------------------------------------------------------
    # Load trained GP models
    # ------------------------------------------------------------------
    pca_pressure   = joblib.load("GPS_PKL/pca_pressure.pkl")
    pca_efficiency = joblib.load("GPS_PKL/pca_efficiency.pkl")
    pca_noise      = joblib.load("GPS_PKL/pca_noise.pkl")
    gp_pressure   = joblib.load("GPS_PKL/gp_pressure.pkl")
    gp_efficiency = joblib.load("GPS_PKL/gp_efficiency.pkl")
    gp_noise      = joblib.load("GPS_PKL/gp_noise.pkl")

    # ------------------------------------------------------------------
    # Print target curve points
    # ------------------------------------------------------------------
    print(f"Target: Max psi = {target_points[0]}; Max phi = {target_points[1]}")
    # stopping = input("Press enter to continue with this target")

    # ------------------------------------------------------------------
    # Run Optimisation
    # ------------------------------------------------------------------
    print(f"\nRunning GA (pop={200}, gen={10}) ...")

    results = run_optimisation(
        obj          = "noise",  # "eta" or "noise"
        target_points = target_points,
        pop_size      = 200,
        n_gen         = 30,
        seed          = 42,
        verbose       = True,
    )
    return results

   

if __name__ == "__main__":
    print(TARGET_CURVE_POINTS)
    full_results = []
    for i, (p, q) in enumerate(TARGET_CURVE_POINTS):
        print(f"Target point {i+1}: psi={p:.4f}, phi={q:.4f}")
        try:
            result = main(target_points = (p,q))
            full_results.append(result)
        except RuntimeError as e:
            save_output(full_results)
            print(f"Error occurred for target point {i+1}: {e}")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    save_output(full_results)
    plot_output(full_results)