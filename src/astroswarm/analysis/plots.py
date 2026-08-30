
from __future__ import annotations

from itertools import cycle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  

_FIG_DIR = Path("results/figures")


_CYAN = "#20C5D8"
_PURPLE = "#7B61C9"
_PINK = "#E66AA5"
_BLUE = "#3478C5"
_PALETTE = (_CYAN, _PURPLE, _PINK, _BLUE)
_LINE_STYLES = ("-", "--", "-.", ":")
_LINE_WIDTH = 2.4


def _style_axes(ax):
    
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.1,
                   length=5)
    ax.tick_params(axis="both", which="minor", labelsize=10, width=1.0,
                   length=3)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(14)
    ax.title.set_weight("semibold")
    ax.grid(True, alpha=0.22, linewidth=0.9)


def _style_colorbar(colorbar):
    colorbar.ax.tick_params(labelsize=10.5)
    colorbar.set_label(colorbar.ax.get_ylabel(), fontsize=12)


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_coverage_curve(metrics, out_path: str | Path | None = None):
    
    arr = metrics.as_arrays()
    out_path = Path(out_path) if out_path else _FIG_DIR / "coverage_curve.png"
    _ensure_dir(out_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(arr["time"], arr["coverage"], color=_CYAN, lw=_LINE_WIDTH,
            label="Coverage fraction")
    ax.plot(arr["time"], arr["quality"], color=_PURPLE, lw=_LINE_WIDTH,
            label="Mean observation quality", linestyle="--")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("fraction of surface")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Asteroid mapping progress: coverage and observation quality")
    _style_axes(ax)
    ax.legend(fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_coverage_map(asteroid, surface_map, out_path: str | Path | None = None):
    
    out_path = Path(out_path) if out_path else _FIG_DIR / "coverage_map.png"
    _ensure_dir(out_path)

    cells = asteroid.cells  
    lat = np.degrees(np.arcsin(np.clip(cells[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(cells[:, 1], cells[:, 0]))

    fig, ax = plt.subplots(figsize=(8, 4))
    sc = ax.scatter(lon, lat, c=surface_map.coverage, cmap="cool",
                    s=8, vmin=0.0, vmax=1.0)
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    ax.set_title("Mapped surface coverage by body-frame location")
    _style_axes(ax)
    colorbar = fig.colorbar(sc, ax=ax, label="Coverage fraction")
    _style_colorbar(colorbar)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_coverage_comparison(runs, out_path=None, target=None,
                             title="Strategy coverage performance over time"):
    
    out_path = Path(out_path) if out_path else _FIG_DIR / "comparison_coverage.png"
    _ensure_dir(out_path)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    styles = cycle(_LINE_STYLES)
    for i, (label, metrics) in enumerate(runs.items()):
        arr = metrics.as_arrays()
        line, = ax.plot(arr["time"], arr["coverage"], label=label,
                        color=_PALETTE[i % len(_PALETTE)], ls=next(styles),
                        lw=_LINE_WIDTH)
        tt = metrics.summary().get("time_to_target")
        if tt is not None:
            level = target if target is not None else metrics.target_coverage
            ax.plot([tt], [level], marker="o", color=line.get_color(),
                    markersize=7, markeredgecolor="white", zorder=5)
    if target is not None:
        ax.axhline(target, color=_PINK, lw=1.8, ls=":",
                   label=f"Target coverage ({target:.2f})")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("coverage fraction")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    _style_axes(ax)
    ax.legend(fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_redundancy_comparison(rows, out_path=None,
                               title="Strategy coordination trade-off: redundancy and delta-v"):
    
    out_path = Path(out_path) if out_path else _FIG_DIR / "redundancy_comparison.png"
    _ensure_dir(out_path)

    labels = [r["strategy"] for r in rows]
    redundancy = [float(r.get("final_redundancy") or 0.0) for r in rows]
    delta_v = [float(r.get("aggregate_dv") or 0.0) for r in rows]

    fig, (ax_r, ax_d) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    bar_colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]
    ax_r.bar(labels, redundancy, color=bar_colors)
    ax_r.set_ylabel("mean re-observations / cell")
    ax_r.set_title("Redundant re-mapping (lower = better)")
    _style_axes(ax_r)

    ax_d.bar(labels, delta_v, color=bar_colors)
    ax_d.set_ylabel("aggregate Δv [m/s]")
    ax_d.set_title("Swarm cost (lower = better)")
    _style_axes(ax_d)

    fig.suptitle(title, fontsize=15, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_baseline_pareto(aggregates, out_path=None,
                         title="Strategy trade-off: time to target vs. fuel cost"):
    
    out_path = Path(out_path) if out_path else _FIG_DIR / "baseline_pareto.png"
    _ensure_dir(out_path)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    for i, (label, agg) in enumerate(aggregates.items()):
        dv = agg.get("aggregate_dv", {})
        tt = agg.get("time_to_target", {})
        
        x = (dv.get("mean") or 0.0) / 1000.0
        xerr = (dv.get("std") or 0.0) / 1000.0
        y = (tt.get("mean") or 0.0) / 1000.0
        yerr = (tt.get("std") or 0.0) / 1000.0
        color = _PALETTE[i % len(_PALETTE)]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", markersize=10,
                   color=color, markeredgecolor="white", markeredgewidth=1.2,
                   ecolor=color, elinewidth=1.8, capsize=4, capthick=1.8,
                   label=label, zorder=5)
    ax.set_xlabel(r"aggregate $\Delta v$ [km/s]")
    ax.set_ylabel("time to target [ks]")
    ax.set_title(title)
    _style_axes(ax)
    ax.legend(fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_parameter_sweep(values, means, stds, param_name, metric_name,
                         out_path=None, logx=False):
    
    out_path = Path(out_path) if out_path else _FIG_DIR / f"sweep_{metric_name}.png"
    _ensure_dir(out_path)

    v = np.asarray(values, dtype=float)
    m = np.asarray(means, dtype=float)
    e = np.asarray(stds, dtype=float)
    ok = ~np.isnan(m)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(v[ok], m[ok], yerr=e[ok], marker="o", markersize=7,
                color=_BLUE, ecolor=_PINK, capsize=4, capthick=1.8,
                lw=_LINE_WIDTH)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(param_name)
    ax.set_ylabel(f"{metric_name} (mean ± std)")
    ax.set_title(f"Parameter sensitivity: {metric_name} vs. {param_name}")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_orbit_bands(time, altitudes, out_path=None, labels=None):
    
    out_path = Path(out_path) if out_path else _FIG_DIR / "orbit_bands.png"
    _ensure_dir(out_path)

    t = np.asarray(time, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    styles = cycle(_LINE_STYLES)
    for i, alt in enumerate(altitudes):
        lbl = labels[i] if labels else f"agent {i}"
        ax.plot(t, np.asarray(alt, dtype=float) / 1000.0,
                color=_PALETTE[i % len(_PALETTE)], ls=next(styles),
                lw=_LINE_WIDTH, label=lbl)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("altitude [km]")
    ax.set_title("Swarm orbital separation: altitude bands by agent")
    _style_axes(ax)
    if len(altitudes) <= 10:
        ax.legend(ncol=2, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pheromone_map(asteroid, pheromone_values, out_path=None):
    
    out_path = Path(out_path) if out_path else _FIG_DIR / "pheromone_map.png"
    _ensure_dir(out_path)

    cells = asteroid.cells
    lat = np.degrees(np.arcsin(np.clip(cells[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(cells[:, 1], cells[:, 0]))

    fig, ax = plt.subplots(figsize=(8, 4))
    sc = ax.scatter(lon, lat, c=np.asarray(pheromone_values, dtype=float),
                    cmap="cool", s=8)
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    ax.set_title("Final pheromone concentration by body-frame location")
    _style_axes(ax)
    colorbar = fig.colorbar(sc, ax=ax, label="Pheromone level")
    _style_colorbar(colorbar)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_evolution_history(history, out_path=None,
                           title="Evolution progress: best fitness by generation"):
    
    out_path = Path(out_path) if out_path else _FIG_DIR / "evolution_history.png"
    _ensure_dir(out_path)
    h = np.asarray(history, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(h.size), h, marker="o", markersize=6,
            color=_PURPLE, lw=_LINE_WIDTH)
    ax.set_xlabel("generation")
    ax.set_ylabel("best fitness so far")
    ax.set_title(title)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_evolved_vs_manual(benchmark, out_path=None,
                           metrics=("time_to_target", "aggregate_dv", "final_redundancy"),
                           title="Evolved vs. hand-tuned policy performance on held-out seeds"):
    
    out_path = Path(out_path) if out_path else _FIG_DIR / "evolved_vs_manual.png"
    _ensure_dir(out_path)

    man, evo = benchmark["manual"], benchmark["evolved"]
    panels = list(metrics) + ["fitness"]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 4.0))
    if len(panels) == 1:
        axes = [axes]
    for ax, metric in zip(axes, panels):
        if metric == "fitness":
            vals = [man["fitness"], evo["fitness"]]
            errs = [0.0, 0.0]
        else:
            vals = [man["aggregate"].get(metric, {}).get("mean") or 0.0,
                    evo["aggregate"].get(metric, {}).get("mean") or 0.0]
            errs = [man["aggregate"].get(metric, {}).get("std") or 0.0,
                    evo["aggregate"].get(metric, {}).get("std") or 0.0]
        ax.bar(["manual", "evolved"], vals, yerr=errs, capsize=4,
               color=[_BLUE, _PINK])
        ax.set_title(metric.replace("_", " ").title())
        _style_axes(ax)
    fig.suptitle(title, fontsize=15, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
