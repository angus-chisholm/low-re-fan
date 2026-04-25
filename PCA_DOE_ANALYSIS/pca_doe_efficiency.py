"""
pca_doe_efficiency.py  (η tab)
================================
Interactive PCA + GP surrogate panel for the efficiency η(φ).
Mirrors pca_doe_characteristic.py (ψ tab) exactly in structure and controls.
Embedded as Tab 2 by pca_doe_main.py.
"""

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import csv, os

from pca_core import (
    PARAM_BOUNDS, PARAM_NAMES, PARAM_KEYS,
    PHI_COMMON, N_PHI,
    BG, PANEL, BORDER, CYAN, AMBER, GREEN, RED, PINK, GREY, WHITE,
    MCOLORS,
    PCAModel, GPSurrogates,
)
from pca_panel_mixin import ControlPanelMixin

# Distinct accent colour for the η tab so the two tabs feel visually separate
ETA_ACCENT = GREEN      # main prediction line colour
ETA_BAND   = GREEN      # uncertainty band fill colour


class EtaTab(ControlPanelMixin):
    """
    η(φ) PCA surrogate tab.

    Parameters
    ----------
    fig   : matplotlib Figure  (created externally by the main launcher)
    outer : gridspec.SubplotSpec  — the region assigned to this tab
    runs  : list of run dicts (output of load_data(), shared with PsiTab)
    """

    _curve_key    = "eta"
    _ylabel_short = "η"
    N_MODES_MAX   = 6

    def __init__(self, fig, outer, runs: list, n_modes: int = 4):
        self.fig     = fig
        self.runs    = runs
        self.n_modes = n_modes

        # Filter out runs where η is all-NaN (e.g. missing column)
        valid_runs = [r for r in runs if not np.all(np.isnan(r["eta"]))]
        if len(valid_runs) < len(runs):
            print(f"[η-tab] {len(runs) - len(valid_runs)} runs skipped (no η data).")
        if not valid_runs:
            raise RuntimeError("[η-tab] No runs have efficiency data — check column name.")
        self.runs = valid_runs

        print("[η-tab] Running PCA …")
        self.pca     = PCAModel(self.runs, curve_key=self._curve_key, n_modes=n_modes)
        print("[η-tab] Training GP surrogates …")
        self.gp      = GPSurrogates(self.pca)
        self.loo_err = self.gp.loo_rms()
        print(f"[η-tab] LOO RMS = {self.loo_err:.5f}")

        self._scores_m = np.zeros(n_modes)
        self._scores_s = np.zeros(n_modes)
        self._y_pred   = self.pca.mean_curve.copy()

        self._build_layout(outer)
        self._draw_all()

    # ── Layout ───────────────────────────────────────────────────────────────
    def _build_layout(self, outer):
        outer_gs = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer,
            width_ratios=[1, 0.4, 2.6], wspace=0.2)

        right_gs = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer_gs[2], hspace=0.38, wspace=0.32)

        self.ax_main  = self.fig.add_subplot(right_gs[0, :])
        self.ax_scree = self.fig.add_subplot(right_gs[1, 0])
        self.ax_modes = self.fig.add_subplot(right_gs[1, 1])

        self._build_controls(self.fig, outer_gs[0], outer_gs[1])

    # ── Main plot ─────────────────────────────────────────────────────────────
    def _draw_main(self):
        ax = self.ax_main
        ax.cla(); ax.set_facecolor(PANEL); ax.grid(True)

        param_vals         = self._get_params()
        scores_m, scores_s, phi_s_m, phi_s_s, phi_e_m, phi_e_s = self.gp.predict(param_vals)

        checked_box = self._get_tick()
        plot_vals   = None
        if checked_box is not None:
            param   = list(PARAM_BOUNDS.values())[checked_box]
            n_steps = 5 if checked_box == 7 else 8
            plot_vals = np.linspace(param[1], param[2], n_steps)

        eta_pred = self.pca.reconstruct(scores_m)
        mask = self.pca.return_mask(phi_s_m, phi_e_m)
        eta_pred = eta_pred[mask]

        # Uncertainty: propagate GP score std through linear modes
        eta_var = np.zeros(N_PHI)
        for k in range(self.n_modes):
            eta_var += (self.pca.modes[k] * scores_s[k]) ** 2
        eta_std = np.sqrt(eta_var)[mask]

        # DoE runs (faded background)
        for r in self.runs:
            ax.plot(PHI_COMMON, r["eta"], color=GREY, alpha=0.12, lw=0.6)

        # Mean curve
        ax.plot(PHI_COMMON, self.pca.mean_curve,
                color=WHITE, lw=1.2, ls="--", alpha=0.2, label="Mean curve")

        # GP uncertainty bands
        ax.fill_between(PHI_COMMON[mask],
                        eta_pred - 2*eta_std, eta_pred + 2*eta_std,
                        color=ETA_BAND, alpha=0.10, label="GP ±2σ")
        ax.fill_between(PHI_COMMON[mask],
                        eta_pred - eta_std,   eta_pred + eta_std,
                        color=ETA_BAND, alpha=0.22, label="GP ±1σ")

        # GP mean prediction
        ax.plot(PHI_COMMON[mask], eta_pred, color=ETA_ACCENT, lw=2.5,
                label=f"GP prediction ({self.n_modes} modes)", zorder=7)

        # Peak-efficiency marker
        peak_idx = int(np.nanargmax(eta_pred))
        ax.axvline(PHI_COMMON[mask][peak_idx], color=AMBER, lw=1.0, ls=":", alpha=0.7)
        ax.scatter([PHI_COMMON[mask][peak_idx]], [eta_pred[peak_idx]],
                   color=AMBER, s=60, zorder=9,
                   label=f"η_peak={eta_pred[peak_idx]:.3f}  φ={PHI_COMMON[mask][peak_idx]:.3f}")

        # Parameter sweep
        if plot_vals is not None:
            n    = len(plot_vals)
            cmap = cm.coolwarm
            for i, val in enumerate(plot_vals):
                pv = list(param_vals); pv[checked_box] = val
                sm, _, phi_s_m, _, phi_e_m, _ = self.gp.predict(pv)
                y = self.pca.reconstruct(sm)
                mask = self.pca.return_mask(phi_s_m, phi_e_m)
                ax.plot(PHI_COMMON[mask], y[mask], color=cmap(i / (n - 1)), lw=1.5, alpha=0.6,
                        label=f"{PARAM_KEYS[checked_box]}={val:.2f}", zorder=6)

        title = "  ".join(f"{k}={v:.3g}" for k, v in zip(PARAM_KEYS, param_vals))
        ax.set_title(title, color=AMBER, fontsize=7, pad=4)
        ax.set_xlabel("Flow coefficient  φ  [—]")
        ax.set_ylabel("Efficiency [aero work/electrical work]  η  [—]")
        ax.legend(loc="upper left", framealpha=0.7)

        # Auto-scale y with a small margin, clamped to [0, 1]
        y_lo = max(0.0,  np.nanmin(eta_pred) - 0.05)
        y_hi = min(1.0,  np.nanmax(eta_pred) + 0.05)
        ax.set_ylim(0, 0.22)
        ax.set_xlim(0.04, 0.26)

        self._scores_m = scores_m
        self._scores_s = scores_s
        self._y_pred   = eta_pred

    # ── Export ────────────────────────────────────────────────────────────────
    def _export_csv(self, _event):
        if not self.is_active:
            return
        param_vals = self._get_params()
        fname = ("eta_prediction_" +
                 "_".join(f"{k}{v:.2f}" for k, v in zip(PARAM_KEYS, param_vals)) + ".csv")
        path = os.path.join(os.path.expanduser("~"), fname)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phi", "eta_predicted", "eta_mean_curve"] +
                       [f"mode{k+1}_shape" for k in range(self.n_modes)])
            for i, phi in enumerate(PHI_COMMON):
                row = [phi, self._y_pred[i], self.pca.mean_curve[i]]
                row += [self.pca.modes[k][i] for k in range(self.n_modes)]
                w.writerow([f"{v:.6f}" for v in row])
        print(f"[η Export] Saved → {path}")
        self.ax_main.set_title(f"Exported → {fname}", color=GREEN, fontsize=7, pad=2)
        self.fig.canvas.draw_idle()


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from pca_core import load_data

    runs = load_data()
    fig  = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.canvas.manager.set_window_title("η tab — standalone")
    outer_gs = gridspec.GridSpec(1, 1, figure=fig, left=0.02, right=0.98,
                                 top=0.95, bottom=0.04)
    EtaTab(fig, outer_gs[0], runs)
    plt.show()
