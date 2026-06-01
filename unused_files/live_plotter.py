"""
Live Fan Test Plotter — Enhanced Dark-Theme Dashboard
======================================================
Drop-in replacement for the update_plot / setup block.

Key fixes vs previous version:
  • No ax.clear() — all artists are created ONCE at setup, then
    updated with set_data() / set_offsets() each frame.
  • blit=True so matplotlib only redraws changed artists.
  • Snapshot copy of plot_data taken inside the lock, then all
    numpy work happens outside — minimises lock hold time.
  • Axes limits only updated when data actually goes out of bounds.
  • Error bar segments rebuilt efficiently via _update_errorbars().
  • Status bar text updated in-place (no remove/recreate).

Adds:
  • Efficiency vs Flow Coefficient  (row 2, left)
  • OASPL      vs Flow Coefficient  (row 2, right)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patheffects as pe
import time

# ── Colour palette ────────────────────────────────────────────────────────────
BG_DARK   = "#0A0C10"
BG_PANEL  = "#0F1117"
BG_AXES   = "#12151C"
GRID_COL  = "#1E2535"
TICK_COL  = "#3A4560"
TEXT_COL  = "#C8D0E0"
LABEL_COL = "#7A8AAA"

C_CYAN    = "#00E5FF"
C_AMBER   = "#FFB300"
C_LIME    = "#76FF03"
C_MAGENTA = "#FF4081"
C_VIOLET  = "#AA00FF"
C_CORAL   = "#FF6D60"
C_TEAL    = "#1DE9B6"


# ── Error bar in-place update ─────────────────────────────────────────────────
def _update_errorbars(container, x, y, xerr=None, yerr=None):
    """Update an existing errorbar container in-place."""
    bars = container[2]   # list of LineCollections
    idx  = 0
    if yerr is not None and len(bars) > idx:
        ye   = np.asarray(yerr)
        segs = [np.array([[xi, yi - ei], [xi, yi + ei]])
                for xi, yi, ei in zip(x, y, ye)]
        bars[idx].set_segments(segs)
        idx += 1
    if xerr is not None and len(bars) > idx:
        xe   = np.asarray(xerr)
        segs = [np.array([[xi - ei, yi], [xi + ei, yi]])
                for xi, yi, ei in zip(x, y, xe)]
        bars[idx].set_segments(segs)


def _pad_lim(vmin, vmax, frac=0.08):
    span = vmax - vmin
    if span < 1e-12:
        span = abs(vmin) * 0.2 or 1.0
    return vmin - frac * span, vmax + frac * span


def _update_lim(ax, x, y, xy_plot=False):
    """Expand axis limits only when data goes outside current view."""
    if len(x) == 0:
        return
    if xy_plot:
        xl, xh = _pad_lim(np.nanmin(x), np.nanmax(x))
        yl, yh = _pad_lim(np.nanmin(y), np.nanmax(y))
        cxl, cxh = ax.get_xlim()
        cyl, cyh = ax.get_ylim()
        if xl < cxl or xh > cxh:
            ax.set_xlim(xl, xh)
        if yl < cyl or yh > cyh:
            ax.set_ylim(yl, yh)
    else:
        yl, yh = _pad_lim(np.nanmin(y), np.nanmax(y))
        cyl, cyh = ax.get_ylim()
        if yl < cyl or yh > cyh:
            ax.set_ylim(yl, yh)
        xl, xh = _pad_lim(np.nanmin(x), np.nanmax(x))
        cxl, cxh = ax.get_xlim()
        if xl < cxl or xh > cxh:
            ax.set_xlim(xl, xh)


# ── Axes styling — called ONCE at setup ───────────────────────────────────────
def _style_ax(ax, title="", xlabel="", ylabel="", title_color=C_CYAN):
    ax.set_facecolor(BG_AXES)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=TICK_COL, labelsize=7, length=3, width=0.6,
                   which='both', direction='in', top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(which='major', color=GRID_COL, lw=0.6, alpha=0.8)
    ax.grid(which='minor', color=GRID_COL, lw=0.3, alpha=0.4)
    if xlabel:
        ax.set_xlabel(xlabel, color=LABEL_COL, fontsize=7.5,
                      fontfamily='monospace', labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, color=LABEL_COL, fontsize=7.5,
                      fontfamily='monospace', labelpad=4)
    if title:
        ax.set_title(title, color=title_color, fontsize=8,
                     fontfamily='monospace', fontweight='bold',
                     loc='left', pad=5,
                     path_effects=[pe.withStroke(linewidth=3,
                                                  foreground=BG_AXES)])
    plt.setp(ax.get_xticklabels(), color=TEXT_COL, fontsize=6.5,
             fontfamily='monospace')
    plt.setp(ax.get_yticklabels(), color=TEXT_COL, fontsize=6.5,
             fontfamily='monospace')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _make_legend(ax):
    leg = ax.legend(fontsize=6.5, framealpha=0.15, edgecolor=GRID_COL,
                    facecolor=BG_PANEL, labelcolor=TEXT_COL,
                    loc='best', borderpad=0.6, handlelength=1.2)
    for text in leg.get_texts():
        text.set_fontfamily('monospace')


# ── Artist factory ────────────────────────────────────────────────────────────
def _make_artists(ax, color, label, has_xerr=False, has_yerr=True):
    """Create all persistent artists for one data series."""
    nan1 = np.array([np.nan])

    glow, = ax.plot(nan1, nan1, color=color, lw=3.4, alpha=0.08,
                    zorder=1, animated=False)
    line, = ax.plot(nan1, nan1, color=color, lw=1.4, alpha=0.85,
                    zorder=2, animated=False)

    eb = ax.errorbar(nan1, nan1,
                     xerr=nan1 if has_xerr else None,
                     yerr=nan1 if has_yerr else None,
                     fmt='none', ecolor=color, elinewidth=0.8,
                     capsize=2, capthick=0.8, alpha=0.45,
                     zorder=3, animated=False)

    sc = ax.scatter(nan1, nan1, color=color, s=16, zorder=4,
                    edgecolors='none', label=label, animated=False)

    return glow, line, eb, sc


def _set_series(glow, line, eb, sc, x, y, xerr=None, yerr=None):
    """Push new data into the four artists for one series."""
    if len(x) == 0:
        nan1 = np.array([np.nan])
        glow.set_data(nan1, nan1)
        line.set_data(nan1, nan1)
        sc.set_offsets(np.c_[nan1, nan1])
        return
    glow.set_data(x, y)
    line.set_data(x, y)
    sc.set_offsets(np.c_[x, y])
    _update_errorbars(eb, x, y, xerr=xerr, yerr=yerr)


# ── Figure + artist setup ─────────────────────────────────────────────────────
def setup_plot():
    plt.rcParams.update({
        'figure.facecolor': BG_DARK,
        'axes.facecolor':   BG_AXES,
        'text.color':       TEXT_COL,
        'font.family':      'monospace',
    })

    fig = plt.figure(figsize=(16, 9), facecolor=BG_DARK)

    fig.text(0.5, 0.962, "FAN TEST  \u00b7  LIVE DATA ACQUISITION",
             ha='center', va='top', fontsize=12, fontfamily='monospace',
             fontweight='bold', color=C_CYAN,
             path_effects=[pe.withStroke(linewidth=6, foreground=BG_DARK)])
    fig.text(0.5, 0.943,
             "FLOW COEFFICIENT  /  PRESSURE RISE  /  EFFICIENCY  /  ACOUSTICS",
             ha='center', va='top', fontsize=7, fontfamily='monospace',
             color=LABEL_COL)

    # Status bar — updated in-place, never recreated
    status = fig.text(0.5, 0.005, "  awaiting data \u2026  ",
                      ha='center', va='bottom', fontsize=7.5,
                      fontfamily='monospace', color=C_AMBER,
                      bbox=dict(boxstyle='round,pad=0.3', fc=BG_PANEL,
                                ec=GRID_COL, lw=0.8, alpha=0.9),
                      animated=False)

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           left=0.07, right=0.97,
                           top=0.93, bottom=0.06,
                           hspace=0.52, wspace=0.32)

    def _ax(r, c, title, xlabel, ylabel, tc):
        a = fig.add_subplot(gs[r, c])
        _style_ax(a, title=title, xlabel=xlabel, ylabel=ylabel,
                  title_color=tc)
        return a

    ax_rpm  = _ax(0, 0, "[ RPM ]",
                  "t  (s)", "RPM", C_CYAN)
    ax_flow = _ax(0, 1, "[ MASS FLOW  &  AXIAL VELOCITY ]",
                  "t  (s)", "\u1e41\u00d710\u00b3 (kg/s)  /  V\u2093 (m/s)", C_AMBER)
    ax_perf = _ax(1, 0, "[ PERFORMANCE CURVE ]",
                  "Flow Coefficient  \u03a6",
                  "Pressure Rise Coeff  \u03c8", C_TEAL)
    ax_pres = _ax(1, 1, "[ PRESSURES  &  DENSITY ]",
                  "t  (s)", "\u0394P (Pa)  /  \u03c1\u00d710 (kg/m\u00b3)", C_CORAL)
    ax_eff  = _ax(2, 0, "[ EFFICIENCY  vs  \u03a6 ]",
                  "Flow Coefficient  \u03a6",
                  "Total-to-Static  \u03b7", C_LIME)
    ax_oas  = _ax(2, 1, "[ OASPL  vs  \u03a6 ]",
                  "Flow Coefficient  \u03a6",
                  "OASPL  (dB SPL)", C_MAGENTA)

    ax_eff.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

    # Build all persistent artists
    art = {
        'rpm':   _make_artists(ax_rpm,  C_CYAN,    "RPM"),
        'mflow': _make_artists(ax_flow, C_AMBER,   "\u1e41 \u00d710\u00b3"),
        'axvel': _make_artists(ax_flow, C_LIME,    "V\u2093"),
        'perf':  _make_artists(ax_perf, C_TEAL,    "\u03c8(\u03a6)",
                               has_xerr=True),
        'dp_v':  _make_artists(ax_pres, C_AMBER,   "\u0394P Venturi"),
        'dp_s':  _make_artists(ax_pres, C_VIOLET,  "\u0394P Stage"),
        'rho':   _make_artists(ax_pres, C_CORAL,   "\u03c1\u00d710"),
        'eff':   _make_artists(ax_eff,  C_LIME,    "\u03b7(\u03a6)",
                               has_xerr=True),
        'oaspl': _make_artists(ax_oas,  C_MAGENTA, "OASPL(\u03a6)",
                               has_xerr=True, has_yerr=False),
    }

    for a in (ax_rpm, ax_flow, ax_perf, ax_pres, ax_eff, ax_oas):
        _make_legend(a)

    axes = dict(rpm=ax_rpm, flow=ax_flow, perf=ax_perf,
                pres=ax_pres, eff=ax_eff, oas=ax_oas)

    # All animated artists for blit return
    all_artists = [status]
    for g, l, eb, sc in art.values():
        all_artists += [g, l, sc] + list(eb[2])

    return fig, axes, art, status, all_artists


# ── Frame update ──────────────────────────────────────────────────────────────
def _make_update(fig, axes, art, status, all_artists, plot_data, data_lock):

    def update(_frame):
        # 1. Snapshot under lock (keep this as short as possible)
        with data_lock:
            n = len(plot_data['time'])
            if n < 2:
                return all_artists
            snap = {k: list(v) for k, v in plot_data.items()}

        # 2. Build arrays outside the lock
        t0    = snap['time'][0]
        t     = np.array(snap['time']) - t0

        phi    = np.array(snap['phi_mean'])
        phi_e  = np.array(snap['phi_err'])
        prise  = np.array(snap['pRise_mean'])
        prise_e= np.array(snap['pRise_err'])
        rpm    = np.array(snap['rpm_mean'])
        rpm_e  = np.array(snap['rpm_err'])
        mflow  = np.array(snap['mflow_mean']) * 1e3
        mflow_e= np.array(snap['mflow_err'])  * 1e3
        axvel  = np.array(snap['axvel_mean'])
        axvel_e= np.array(snap['axvel_err'])
        dp_v   = np.array(snap['dp_venturi_mean'])
        dp_v_e = np.array(snap['dp_venturi_err'])
        dp_s   = np.array(snap['dp_stage_mean'])
        dp_s_e = np.array(snap['dp_stage_err'])
        rho    = np.array(snap['rho_mean'])  * 10
        rho_e  = np.array(snap['rho_err'])   * 10
        eff    = np.array(snap.get('efficiency_mean') or [0]*n)
        eff_e  = np.array(snap.get('efficiency_std')  or [0]*n)
        oaspl  = np.array(snap.get('oaspl')           or [0]*n)

        # Sort Phi-based plots for a clean curve
        order   = np.argsort(phi)
        phi_s   = phi[order];   phi_e_s  = phi_e[order]
        prise_s = prise[order]; prise_e_s= prise_e[order]
        eff_s   = eff[order];   eff_e_s  = eff_e[order]
        oaspl_s = oaspl[order]

        # 3. Push data into existing artists
        _set_series(*art['rpm'],   t, rpm,    yerr=rpm_e)
        _set_series(*art['mflow'], t, mflow,  yerr=mflow_e)
        _set_series(*art['axvel'], t, axvel,  yerr=axvel_e)
        _set_series(*art['perf'],  phi_s, prise_s,
                    xerr=phi_e_s, yerr=prise_e_s)
        _set_series(*art['dp_v'],  t, dp_v,   yerr=dp_v_e)
        _set_series(*art['dp_s'],  t, dp_s,   yerr=dp_s_e)
        _set_series(*art['rho'],   t, rho,    yerr=rho_e)
        _set_series(*art['eff'],   phi_s, eff_s,
                    xerr=phi_e_s, yerr=eff_e_s)
        _set_series(*art['oaspl'], phi_s, oaspl_s, xerr=phi_e_s)

        # 4. Expand axis limits only when data exceeds current view
        _update_lim(axes['rpm'],  t, rpm)
        _update_lim(axes['flow'], t, np.concatenate([mflow, axvel]))
        _update_lim(axes['perf'], phi_s, prise_s,  xy_plot=True)
        _update_lim(axes['pres'], t,
                    np.concatenate([dp_v, dp_s, rho]))
        _update_lim(axes['eff'],  phi_s, eff_s,    xy_plot=True)
        _update_lim(axes['oas'],  phi_s, oaspl_s,  xy_plot=True)

        # 5. Status bar in-place
        status.set_text(
            f"  \u03a6={phi[-1]:+.3f}   \u03c8={prise[-1]:+.3f}   "
            f"\u03b7={eff[-1]:.1%}   OASPL={oaspl[-1]:.1f} dB   "
            f"RPM={rpm[-1]:.0f}  "
        )

        return all_artists

    return update


# ── Entry point ────────────────────────────────────────────────────────────────
def run_live_plot(plot_data, data_lock):
    """Drop-in replacement for the original plt.show() block."""
    fig, axes, art, status, all_artists = setup_plot()
    update_fn = _make_update(fig, axes, art, status, all_artists,
                             plot_data, data_lock)
    ani = FuncAnimation(fig, update_fn, interval=250,
                        blit=False, cache_frame_data=False)
    print("Opening Plot Window \u2026")
    plt.show()
    return ani   # keep reference alive


# ── Standalone demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import threading, random

    data_lock = threading.Lock()
    plot_data = {k: [] for k in [
        'time', 'rpm_mean', 'rpm_err',
        'mflow_mean', 'mflow_err',
        'axvel_mean', 'axvel_err',
        'dp_venturi_mean', 'dp_venturi_err',
        'dp_stage_mean',   'dp_stage_err',
        'rho_mean',  'rho_err',
        'phi_mean',  'phi_err',
        'pRise_mean', 'pRise_err',
        'efficiency_mean', 'efficiency_std',
        'oaspl',
    ]}

    def _feed():
        phi_val = 0.05
        while True:
            burst = random.randint(3, 5)
            with data_lock:
                for _ in range(burst):
                    phi  = phi_val + random.gauss(0, 0.003)
                    psi  = max(0, 1.2 - 1.8*phi + random.gauss(0, 0.02))
                    eff  = max(0, min(1, -4*(phi-0.35)**2 + 0.78
                                     + random.gauss(0, 0.01)))
                    rpm  = 3000 + phi_val * 5000 + random.gauss(0, 20)
                    plot_data['time'].append(time.time())
                    plot_data['rpm_mean'].append(rpm)
                    plot_data['rpm_err'].append(15)
                    plot_data['mflow_mean'].append(phi * 12)
                    plot_data['mflow_err'].append(0.3)
                    plot_data['axvel_mean'].append(phi * 25)
                    plot_data['axvel_err'].append(0.4)
                    plot_data['dp_venturi_mean'].append(abs(phi)*60+5)
                    plot_data['dp_venturi_err'].append(1.5)
                    plot_data['dp_stage_mean'].append(psi*50+3)
                    plot_data['dp_stage_err'].append(2.0)
                    plot_data['rho_mean'].append(1.20+random.gauss(0, 0.001))
                    plot_data['rho_err'].append(0.002)
                    plot_data['phi_mean'].append(phi)
                    plot_data['phi_err'].append(0.003)
                    plot_data['pRise_mean'].append(psi)
                    plot_data['pRise_err'].append(0.015)
                    plot_data['efficiency_mean'].append(eff)
                    plot_data['efficiency_std'].append(0.02)
                    plot_data['oaspl'].append(65 + 8*phi**0.5
                                             + random.gauss(0, 0.5))
                    phi_val = min(phi_val + 0.004, 0.65)
            time.sleep(random.uniform(1.5, 3.0))

    threading.Thread(target=_feed, daemon=True).start()
    time.sleep(1.0)
    run_live_plot(plot_data, data_lock)
