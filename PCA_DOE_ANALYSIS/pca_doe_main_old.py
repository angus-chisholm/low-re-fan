"""
pca_doe_main.py
===============
Two-tab matplotlib GUI launcher.

  Tab 1 — ψ(φ)  pressure-rise coefficient                   (pca_doe_characteristic.py)
  Tab 2 — η(φ)  efficiency [aero work/electrical work]      (pca_doe_efficiency.py)

Tab switching is done with two matplotlib Button widgets at the top of the
figure; the inactive tab's axes are hidden/shown as needed.

Run:
    python pca_doe_main.py
"""

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

from pca_core import (
    load_data,
    BG, PANEL, BORDER, CYAN, AMBER, GREEN, WHITE,
)
from unused_files.pca_doe_characteristic import PsiTab
from pca_doe_efficiency      import EtaTab


# ─── Tab-button colours ───────────────────────────────────────────────────────
TAB_ACTIVE   = CYAN
TAB_INACTIVE = BORDER
TAB_TXT_ACT  = "#0b0f1e"
TAB_TXT_INACT= WHITE

TAB_HEIGHT   = 0.04   # fraction of figure height reserved for tab buttons


class TabbedApp:
    def __init__(self):
        print("═" * 60)
        print("  PCA + GP Surrogate  ·  DoE Compressor Tool")
        print("═" * 60)
        print("Loading DoE dataset …")
        self.runs = load_data()
        print(f"[DoE] {len(self.runs)} runs loaded.\n")

        # ── Figure ────────────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(22, 14), facecolor=BG)
        self.fig.canvas.manager.set_window_title(
            "PCA + GP Surrogate  ·  DoE Compressor Tool")

        # ── Tab button axes (top strip) ────────────────────────────────────
        # Two buttons side by side, full width, at the very top
        self.ax_tab_psi = self.fig.add_axes([0.01, 1 - TAB_HEIGHT, 0.18, TAB_HEIGHT - 0.005])
        self.ax_tab_eta = self.fig.add_axes([0.20, 1 - TAB_HEIGHT, 0.18, TAB_HEIGHT - 0.005])

        self.btn_psi = Button(self.ax_tab_psi, "Tab 1 — ψ(φ)  Pressure Rise",
                              color=TAB_ACTIVE, hovercolor=CYAN)
        self.btn_eta = Button(self.ax_tab_eta, "Tab 2 — η(φ)  Efficiency",
                              color=TAB_INACTIVE, hovercolor=CYAN)

        for btn, act in [(self.btn_psi, True), (self.btn_eta, False)]:
            btn.label.set_fontsize(9)
            btn.label.set_fontfamily("monospace")
            btn.label.set_color(TAB_TXT_ACT if act else TAB_TXT_INACT)

        self.btn_psi.on_clicked(self._show_psi)
        self.btn_eta.on_clicked(self._show_eta)

        # ── Content area (below tab strip) ────────────────────────────────
        top    = 1 - TAB_HEIGHT - 0.005
        bottom = 0.03

        # One SubplotSpec for each tab, occupying the same area
        outer_gs = gridspec.GridSpec(
            1, 1, figure=self.fig,
            left=0.05, right=0.95,
            top=top, bottom=bottom)

        self.outer_psi = outer_gs[0]
        self.outer_eta = outer_gs[0]   # same region; we hide/show via axes visibility

        # ── Build both tabs (axes created but only ψ visible initially) ────
        print("Building ψ tab …")
        self.psi_tab = PsiTab(self.fig, self.outer_psi, self.runs, n_modes=4)

        print("\nBuilding η tab …")
        self.eta_tab = EtaTab(self.fig, self.outer_eta, self.runs, n_modes=4)

        # Hide eta tab axes initially
        self._set_tab_visible(self.eta_tab, False)
        self._current = "psi"

        plt.show()

    # ── Visibility helpers ────────────────────────────────────────────────────
    @staticmethod
    def _get_tab_axes(tab):
        """Return all axes that belong to a tab (plots + widget axes)."""
        axes = [tab.ax_main, tab.ax_scree, tab.ax_modes]
        # Widget axes are stored on the Slider / CheckButtons objects
        for sl in tab.sliders.values():
            axes.append(sl.ax)
        axes.append(tab.ticks.ax)
        axes.append(tab.radio_modes.ax)
        axes.append(tab.info_text.axes)
        axes.append(tab.btn_export.ax)
        return axes

    def _set_tab_visible(self, tab, visible: bool):
        tab.is_active = visible
        tab.set_sliders_active(visible)  # Disable/enable slider observers
        for ax in self._get_tab_axes(tab):
            ax.set_visible(visible)

    def _sync_sliders(self, source_tab, dest_tab):
        """Copy slider values from source_tab to dest_tab without triggering callbacks or mouse grab."""
        dest_tab.is_active = False  # Suppress callbacks during sync
        for key, slider_dest in dest_tab.sliders.items():
            val = source_tab.sliders[key].val
            # Directly set value and update display, avoiding mouse grab on hidden axes
            slider_dest.val = val
            # slider_dest.valtext.set_text(slider_dest.valfmt % val)
        dest_tab.is_active = True   # Re-enable for visibility change

    # ── Tab switch callbacks ──────────────────────────────────────────────────
    def _show_psi(self, _event=None):
        if self._current == "psi":
            return
        self._sync_sliders(self.eta_tab, self.psi_tab)
        self._set_tab_visible(self.eta_tab, False)
        self._set_tab_visible(self.psi_tab, True)
        self._current = "psi"
        self._update_tab_buttons(active="psi")
        self.fig.canvas.draw_idle()

    def _show_eta(self, _event=None):
        if self._current == "eta":
            return
        self._sync_sliders(self.psi_tab, self.eta_tab)
        self._set_tab_visible(self.psi_tab, False)
        self._set_tab_visible(self.eta_tab, True)
        self._current = "eta"
        self._update_tab_buttons(active="eta")
        self.fig.canvas.draw_idle()

    def _update_tab_buttons(self, active: str):
        self.btn_psi.color          = TAB_ACTIVE   if active == "psi" else TAB_INACTIVE
        self.btn_psi.hovercolor     = CYAN
        self.btn_psi.label.set_color(TAB_TXT_ACT  if active == "psi" else TAB_TXT_INACT)

        self.btn_eta.color          = TAB_ACTIVE   if active == "eta" else TAB_INACTIVE
        self.btn_eta.hovercolor     = CYAN
        self.btn_eta.label.set_color(TAB_TXT_ACT  if active == "eta" else TAB_TXT_INACT)

        # Force button face colour to update immediately
        self.ax_tab_psi.set_facecolor(self.btn_psi.color)
        self.ax_tab_eta.set_facecolor(self.btn_eta.color)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TabbedApp()
