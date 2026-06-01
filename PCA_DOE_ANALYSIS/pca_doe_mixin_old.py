"""
pca_panel_mixin.py
==================
Mixin that builds the left-side control panel (sliders, checkbuttons, radio,
info text) and the middle checkbutton column.  Both the ψ-tab and η-tab
inherit from this to avoid code duplication.

Sub-classes must implement:
    _draw_main(self)
    _draw_scree(self)
    _draw_modes(self)
    _update_info(self)
    _export_csv(self, event)

They also receive:
    self.sliders      : dict  key→Slider
    self.ticks        : CheckButtons
    self.radio_modes  : RadioButtons
    self.info_text    : Text artist
"""

import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import matplotlib.cm as cm

from pca_core import (
    PARAM_BOUNDS, PARAM_NAMES, PARAM_KEYS, PARAM_LO, PARAM_HI,
    PHI_COMMON, N_PHI,
    BG, PANEL, BORDER, CYAN, AMBER, GREEN, RED, PINK, GREY, WHITE,
    MCOLORS, COLOUR_LIST,
    PCAModel, GPSurrogates,
)


class ControlPanelMixin:
    """
    Call  self._build_controls(fig, outer_spec)  from the subclass __init__
    after creating `self.fig` and the outer GridSpec column reserved for
    controls.

    `outer_spec` should be the three-element tuple:
        (left_outer_col, middle_outer_col, right_outer_col)
    from  gridspec.GridSpec(1, 3, ...).
    """

    N_MODES_MAX: int = 6

    # ── public entry point ───────────────────────────────────────────────────
    def _build_controls(self, fig, left_col, middle_col):
        self.is_active = True  # Flag to prevent callbacks on inactive tabs
        left_gs   = gridspec.GridSpecFromSubplotSpec(30, 1, subplot_spec=left_col,   hspace=0.3)
        middle_gs = gridspec.GridSpecFromSubplotSpec(2,  1, subplot_spec=middle_col, hspace=0.3)

        self._build_sliders(fig, left_gs)
        self._build_checkbuttons(fig, middle_gs)
        self._build_radio(fig, left_gs)
        self._build_info(fig, left_gs)
        self._build_export_button(fig, left_gs)

    # ── sliders ──────────────────────────────────────────────────────────────
    def _build_sliders(self, fig, left_gs):
        self.sliders = {}
        self._slider_observer_cids = {}  # Store all observer connection IDs
        for i, (label, (key, lo, hi, init)) in enumerate(PARAM_BOUNDS.items()):
            ax = fig.add_subplot(left_gs[i * 2 + 1])
            ax.set_facecolor(PANEL)
            kw = dict(color=CYAN, track_color=BORDER)
            if key == "n_blade":
                kw["valstep"] = 1
            sl = Slider(ax, label, lo, hi, valinit=init, **kw)
            sl.label.set_color(WHITE);  sl.label.set_fontsize(7.5)
            sl.valtext.set_color(CYAN); sl.valtext.set_fontsize(7.5)
            cid = sl.on_changed(self._on_slider)
            self._slider_observer_cids[key] = cid
            self.sliders[key] = sl

    # ── checkbuttons ─────────────────────────────────────────────────────────
    def _build_checkbuttons(self, fig, middle_gs):
        ax = fig.add_subplot(middle_gs[0])
        ax.set_facecolor(PANEL)
        self.ticks = CheckButtons(
            ax, PARAM_NAMES,
            actives=[False] * len(PARAM_NAMES),
            useblit=[False] * len(PARAM_NAMES),
            frame_props={
                "edgecolor": BORDER,
                "facecolor": "white",
                "s":         225,
                "linewidth": 1.5,
            },
            check_props={
                "color":     CYAN,
                "s":         225,
                "linewidth": 2.5,
            },
        )
        for lbl in self.ticks.labels:
            lbl.set_color(WHITE)
            lbl.set_fontsize(7.5)
        ax.set_title("Sweep param", color=AMBER, fontsize=7, pad=2)
        self.ticks.on_clicked(self._on_tick)

    # ── mode radio ───────────────────────────────────────────────────────────
    def _build_radio(self, fig, left_gs):
        n = len(PARAM_BOUNDS)
        ax_radio = fig.add_subplot(left_gs[2*n+1 : 2*n+4])
        ax_radio.set_facecolor(PANEL)
        self.radio_modes = RadioButtons(
            ax_radio, [str(i) for i in range(1, self.N_MODES_MAX + 1)],
            active=self.n_modes - 1, activecolor=CYAN)
        for lbl in self.radio_modes.labels:
            lbl.set_color(WHITE); lbl.set_fontsize(7)
        ax_radio.set_title("Modes k", color=AMBER, fontsize=7, pad=2)
        self.radio_modes.on_clicked(self._on_mode_change)

    # ── info text ────────────────────────────────────────────────────────────
    def _build_info(self, fig, left_gs):
        n = len(PARAM_BOUNDS)
        ax_info = fig.add_subplot(left_gs[2*n+5 : 2*n+8])
        ax_info.set_facecolor(PANEL)
        ax_info.axis("off")
        self.info_text = ax_info.text(
            0.05, 0.95, "", transform=ax_info.transAxes,
            va="top", ha="left", fontsize=7, color=WHITE, fontfamily="monospace")

    # ── export button ────────────────────────────────────────────────────────
    def _build_export_button(self, fig, left_gs):
        n = len(PARAM_BOUNDS)
        ax_btn = fig.add_subplot(left_gs[2*n+13])
        ax_btn.set_facecolor(PANEL)
        self.btn_export = Button(ax_btn, "⬇  Export CSV", color=BORDER, hovercolor=CYAN)
        self.btn_export.label.set_color(WHITE)
        self.btn_export.label.set_fontsize(8)
        self.btn_export.on_clicked(self._export_csv)

    # ── shared helpers ───────────────────────────────────────────────────────
    def _get_params(self):
        return [self.sliders[key].val for key in PARAM_KEYS]

    def _get_tick(self):
        """Return the index of the single checked box, or None."""
        checked = list(self.ticks.get_checked_labels())
        if len(checked) > 1:
            self.ticks.clear()
            return None
        if len(checked) == 1:
            for i, label in enumerate(PARAM_NAMES):
                if checked[0] == label:
                    return i
        return None

    def set_sliders_active(self, active: bool):
        """Enable or disable slider observers."""
        for key, cid in self._slider_observer_cids.items():
            if active:
                # Reconnect observer
                self._slider_observer_cids[key] = self.sliders[key].on_changed(self._on_slider)
            else:
                # Disconnect observer
                self.sliders[key].disconnect(cid)

    # ── shared callbacks ─────────────────────────────────────────────────────
    def _on_slider(self, _val):
        if not self.is_active:
            return
        self._draw_main()
        self._update_info()
        self.fig.canvas.draw_idle()

    def _on_tick(self, _label):
        if not self.is_active:
            return
        self._draw_main()
        self._update_info()
        self.fig.canvas.draw_idle()

    def _on_mode_change(self, label: str):
        if not self.is_active:
            return
        k = int(label)
        if k == self.n_modes:
            return
        self.n_modes = k
        print(f"[PCA] Refitting with k={k} modes …")
        self.pca     = PCAModel(self.runs, curve_key=self._curve_key, n_modes=k)
        print("[GP]  Retraining GP surrogates …")
        self.gp      = GPSurrogates(self.pca)
        self.loo_err = self.gp.loo_rms()
        self._draw_all()

    # ── shared draw dispatch ─────────────────────────────────────────────────
    def _draw_all(self):
        self._draw_main()
        self._draw_scree()
        self._draw_modes()
        self._update_info()
        self.fig.canvas.draw_idle()

    # ── shared scree & modes plots (identical for both tabs) ─────────────────
    def _draw_scree(self):
        ax = self.ax_scree
        ax.cla(); ax.set_facecolor(PANEL); ax.grid(True)
        k_range = np.arange(1, min(len(self.pca.singular_values), 10) + 1)
        ind_var = self.pca.var_explained[:len(k_range)]
        cum_var = self.pca.cum_var[:len(k_range)]
        ax.bar(k_range, ind_var, color=CYAN, alpha=0.6, label="Individual %")
        ax.plot(k_range, cum_var, "o-", color=AMBER, lw=1.5, label="Cumulative %")
        ax.axhline(95, color=RED, ls="--", lw=0.8, alpha=0.7, label="95% threshold")
        ax.axvline(self.n_modes, color=GREEN, ls=":", lw=1.2, alpha=0.8,
                   label=f"k = {self.n_modes}")
        ax.set_xlabel("Mode index k")
        ax.set_ylabel("Variance explained [%]")
        ax.set_xlim(0, self.N_MODES_MAX + 2)
        ax.set_title("Scree plot", color=AMBER, fontsize=8)
        ax.legend(fontsize=6)
        ax.set_xticks(k_range)

    def _draw_modes(self):
        ax = self.ax_modes
        ax.cla(); ax.set_facecolor(PANEL); ax.grid(True)
        ax.axhline(0, color=GREY, lw=0.7, ls="--")
        ax.plot(PHI_COMMON, self.pca.mean_curve,
                color=GREY, lw=1.2, ls="--", alpha=0.7,
                label=f"Mean {self._ylabel_short}")
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
            f"LOO RMS avg  : {self.loo_err:.5f}",
            "",
            "── Kernel (Mode 1) ──",
        ]
        k_str = str(self.gp.gps[0].kernel_)
        for chunk in [k_str[i:i+32] for i in range(0, min(len(k_str), 96), 32)]:
            lines.append("  " + chunk)
        self.info_text.set_text("\n".join(lines))
