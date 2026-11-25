"""The three figures: the layer curve, risk-coverage, and the reliability plot.

Rendered on an explicit light surface so they stay legible whichever theme
GitHub is showing, and backed by the committed CSVs in `results/` so every line
can be read as a table instead.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from answer_probe.evaluate import SPLIT_LABELS, signal_label  # noqa: E402
from answer_probe.splits import SHIFT, TEST  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8982"
GRID = "#e4e3dd"

# Categorical slots 1-3 of the reference palette, validated all-pairs for both
# colour-vision deficiency and normal-vision separation on this surface.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")

LINE_WIDTH = 2.0
MARKER_SIZE = 8.0


def _style_axes(ax, *, xlabel: str, ylabel: str, title: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, color=GRID)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)


def _figure_legend(fig, ax, *, n_col: int) -> None:
    """One legend under the panels.

    Inside the axes it collides with whatever the data happens to do -- the
    baseline line sits at a different height in every run -- so it goes below
    the figure, where nothing can run into it.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=n_col,
        frameon=False,
        fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def layer_curve(
    scan: pd.DataFrame,
    selection: dict,
    *,
    split: str = TEST,
    baseline_auroc: float | None = None,
    baseline_label: str = "best confidence baseline",
    path: str | Path,
) -> Path:
    """AUROC against layer, one panel per probe, both positions on each panel.

    The vertical marker is the layer chosen on the validation split; the curve
    itself is drawn on the test split, so the gap between the marker and the
    curve's peak shows how much the selection gave away.
    """
    column = f"auroc_{split}"
    kinds = [("linear", "Linear probe"), ("diffmeans", "Difference of means")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, (kind, kind_label) in zip(axes, kinds, strict=True):
        frame = scan[scan["probe"] == kind]
        for i, position in enumerate(("before", "after")):
            rows = frame[frame["position"] == position].sort_values("layer")
            if rows.empty:
                continue
            ax.plot(
                rows["layer"],
                rows[column],
                color=SERIES[i],
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=MARKER_SIZE / 2.4,
                label=f"{position} answering",
                zorder=3,
            )
            chosen = selection["per_probe"].get(f"{position}_{kind}")
            if chosen:
                ax.axvline(
                    chosen["layer"],
                    color=SERIES[i],
                    linewidth=1.0,
                    linestyle=":",
                    alpha=0.8,
                    zorder=2,
                )
        if baseline_auroc is not None:
            ax.axhline(
                baseline_auroc,
                color=INK_SECONDARY,
                linewidth=1.4,
                linestyle="--",
                zorder=2,
            )
            ax.annotate(
                baseline_label,
                xy=(0.98, baseline_auroc),
                xycoords=("axes fraction", "data"),
                xytext=(0, 4),
                textcoords="offset points",
                color=INK_SECONDARY,
                fontsize=8.5,
                ha="right",
            )
        ax.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=(0, (1, 3)), zorder=1)
        ax.annotate(
            "chance",
            xy=(0.98, 0.5),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            color=INK_MUTED,
            fontsize=8.5,
            ha="right",
        )
        _style_axes(ax, xlabel="layer (0 = embeddings, last = final norm)",
                    ylabel="AUROC for predicting a wrong answer", title=kind_label)

    axes[1].set_ylabel("")
    _figure_legend(fig, axes[0], n_col=2)
    fig.suptitle(
        f"Predicting a wrong answer from the residual stream ({SPLIT_LABELS[split]})",
        color=INK,
        fontsize=12,
        x=0.012,
        ha="left",
    )
    fig.text(
        0.012,
        -0.12,
        "Dotted vertical lines mark the layer chosen on the validation split.",
        color=INK_MUTED,
        fontsize=8.5,
        ha="left",
    )
    return _save(fig, path)


def risk_coverage_plot(
    table: pd.DataFrame, columns: list[str], *, path: str | Path
) -> Path:
    """Accuracy among answered questions as the system abstains more."""
    splits = [s for s in (TEST, SHIFT) if (table["split"] == s).any()]
    fig, axes = plt.subplots(1, len(splits), figsize=(5.5 * len(splits), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)

    for ax, split in zip(axes, splits, strict=True):
        frame = table[table["split"] == split]
        for i, column in enumerate(columns[: len(SERIES)]):
            rows = frame[frame["signal"] == column].sort_values("coverage")
            if rows.empty:
                continue
            ax.plot(
                rows["coverage"],
                rows["accuracy"],
                color=SERIES[i],
                linewidth=LINE_WIDTH,
                label=signal_label(column),
                zorder=3,
            )
        _style_axes(
            ax,
            xlabel="coverage (share of questions answered)",
            ylabel="accuracy on answered questions",
            title=SPLIT_LABELS[split],
        )
        ax.set_xlim(0.05, 1.02)
    for ax in axes[1:]:
        ax.set_ylabel("")
    _figure_legend(fig, axes[0], n_col=min(3, len(columns)))
    fig.suptitle(
        "Selective answering: abstain on the questions a signal flags as wrong",
        color=INK,
        fontsize=12,
        x=0.012,
        ha="left",
    )
    return _save(fig, path)


def calibration_plot(bins: pd.DataFrame, summary: pd.DataFrame, *, path: str | Path) -> Path:
    """Predicted probability of a wrong answer against the observed frequency."""
    splits = [s for s in (TEST, SHIFT) if (bins["split"] == s).any()]
    fig, axes = plt.subplots(1, len(splits), figsize=(4.8 * len(splits), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)

    ece = dict(zip(summary["split"], summary["ece"], strict=True)) if len(summary) else {}
    for ax, split in zip(axes, splits, strict=True):
        frame = bins[(bins["split"] == split) & (bins["count"] > 0)]
        ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.0, linestyle=(0, (1, 3)), zorder=1)
        ax.annotate(
            "perfect calibration",
            xy=(0.62, 0.62),
            xytext=(2, -11),
            textcoords="offset points",
            color=INK_MUTED,
            fontsize=8.5,
            rotation=38,
            rotation_mode="anchor",
        )
        sizes = np.clip(frame["count"] / max(frame["count"].max(), 1) * 160, MARKER_SIZE, 180)
        ax.scatter(
            frame["mean_predicted"],
            frame["observed"],
            s=sizes,
            color=SERIES[0],
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
        )
        ax.plot(
            frame["mean_predicted"],
            frame["observed"],
            color=SERIES[0],
            linewidth=LINE_WIDTH,
            zorder=2,
        )
        title = SPLIT_LABELS[split]
        if split in ece:
            title += f"  ·  ECE {ece[split]:.3f}"
        _style_axes(
            ax,
            xlabel="predicted P(answer is wrong)",
            ylabel="observed share wrong",
            title=title,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.suptitle(
        "Calibration of the headline probe (marker area is the number of questions)",
        color=INK,
        fontsize=12,
        x=0.012,
        ha="left",
    )
    return _save(fig, path)
