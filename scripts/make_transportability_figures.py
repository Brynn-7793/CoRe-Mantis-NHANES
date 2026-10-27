"""Create submission figures for the two-cycle transportability study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from audit_panel_alignment import require_matplotlib_panel_alignment


COLORS = {
    "ink": "#1F2937",
    "muted": "#6B7280",
    "light": "#D1D5DB",
    "development": "#486F9C",
    "temporal": "#D5854F",
    "mantis_v2": "#486F9C",
    "transparent_wearable": "#D5854F",
    "reference": "#8B95A1",
    "positive": "#3F7D69",
}

DISPLAY_NAMES = {
    "clinical": "Clinical covariates",
    "activity_concepts": "Activity concepts",
    "classic_rar": "Classic RAR",
    "mantis_v2": "MantisV2",
    "transparent_wearable": "Transparent wearable",
}

MODEL_ORDER = [
    "clinical",
    "mantis_v2",
    "transparent_wearable",
    "activity_concepts",
    "classic_rar",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def rounded_box(
    axis: plt.Axes,
    x_position: float,
    y_position: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    text_color: str = COLORS["ink"],
    fontweight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x_position, y_position),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    axis.add_patch(patch)
    axis.text(
        x_position + width / 2,
        y_position + height / 2,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=7.2,
        fontweight=fontweight,
        linespacing=1.25,
    )


def arrow(axis: plt.Axes, x_position: float, y_start: float, y_end: float) -> None:
    axis.annotate(
        "",
        xy=(x_position, y_end),
        xytext=(x_position, y_start),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["muted"], "lw": 0.8},
    )


def make_study_design(flow: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.7))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    cycle_specs = [
        (
            "NHANES 2013–2014",
            "2013-2014",
            "Development cycle",
            0.08,
            COLORS["development"],
        ),
        (
            "NHANES 2011–2012",
            "2011-2012",
            "Temporal-evaluation cycle",
            0.56,
            COLORS["temporal"],
        ),
    ]
    stage_labels = [
        "Public demographic and\ncovariate records",
        "Depression-screener\nrecord",
        "Complete PHQ-9\noutcome",
        "Matched analytic\ncohort",
    ]
    y_positions = [0.74, 0.58, 0.42, 0.26]
    box_width = 0.36
    box_height = 0.105

    for cycle_name, cycle_key, subtitle, x_position, color in cycle_specs:
        cycle_rows = flow.loc[flow["cycle"].str.contains(cycle_key, regex=False)]
        if len(cycle_rows) != 5:
            raise ValueError(
                f"Expected five flow rows for {cycle_key}; found {len(cycle_rows)}."
            )
        analytic_rows = cycle_rows.iloc[:4]
        event_count = int(cycle_rows.iloc[4]["n"])
        axis.text(
            x_position + box_width / 2,
            0.94,
            cycle_name,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=color,
        )
        axis.text(
            x_position + box_width / 2,
            0.89,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.5,
            color=COLORS["muted"],
        )
        for index, (_, row) in enumerate(analytic_rows.iterrows()):
            label = f"{stage_labels[index]}\nN = {int(row['n']):,}"
            facecolor = "#F8FAFC" if index < 3 else f"{color}1A"
            edgecolor = COLORS["light"] if index < 3 else color
            rounded_box(
                axis,
                x_position,
                y_positions[index],
                box_width,
                box_height,
                label,
                facecolor,
                edgecolor,
                fontweight="bold" if index == 3 else "normal",
            )
            if index < len(y_positions) - 1:
                arrow(
                    axis,
                    x_position + box_width / 2,
                    y_positions[index] - 0.01,
                    y_positions[index + 1] + box_height + 0.01,
                )
        total = int(analytic_rows.iloc[-1]["n"])
        percentage = 100 * event_count / total
        rounded_box(
            axis,
            x_position + 0.04,
            0.10,
            box_width - 0.08,
            0.095,
            f"PHQ-9 ≥10: {event_count:,} ({percentage:.1f}%)",
            "#FFFFFF",
            color,
            text_color=color,
            fontweight="bold",
        )
        arrow(
            axis,
            x_position + box_width / 2,
            y_positions[-1] - 0.01,
            0.205,
        )

    axis.plot([0.5, 0.5], [0.08, 0.96], color=COLORS["light"], lw=0.8)
    axis.text(
        0.5,
        0.035,
        "Common representation models  →  out-of-fold development estimates  →  "
        "cross-cycle transport gaps\nSurvey-design sensitivity analysis: "
        "weights + strata + primary sampling units",
        ha="center",
        va="center",
        fontsize=6.8,
        color=COLORS["ink"],
        linespacing=1.35,
    )

    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def model_color(model: str) -> str:
    if model in {"mantis_v2", "transparent_wearable"}:
        return COLORS[model]
    return COLORS["reference"]


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-25, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=COLORS["ink"],
    )


def make_transportability(
    performance: pd.DataFrame,
    gaps: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.55),
        gridspec_kw={"wspace": 0.34},
    )
    absolute_axis, gap_axis = axes
    y_positions = np.arange(len(MODEL_ORDER))[::-1]

    auroc = performance.loc[
        (performance["metric"] == "auroc")
        & performance["model"].isin(MODEL_ORDER)
    ].copy()
    for y_position, model in zip(y_positions, MODEL_ORDER):
        rows = auroc.loc[auroc["model"] == model].set_index("cycle")
        development = rows.loc["development"]
        temporal = rows.loc["temporal"]
        absolute_axis.plot(
            [development["estimate"], temporal["estimate"]],
            [y_position, y_position],
            color=COLORS["light"],
            lw=1.2,
            zorder=1,
        )
        for cycle, marker, color in (
            ("development", "o", COLORS["development"]),
            ("temporal", "s", COLORS["temporal"]),
        ):
            row = rows.loc[cycle]
            absolute_axis.errorbar(
                row["estimate"],
                y_position,
                xerr=np.array(
                    [
                        [row["estimate"] - row["ci_lower"]],
                        [row["ci_upper"] - row["estimate"]],
                    ]
                ),
                fmt=marker,
                ms=4.5,
                color=color,
                ecolor=color,
                elinewidth=1,
                capsize=2,
                zorder=3,
            )

    absolute_axis.set_yticks(y_positions)
    absolute_axis.set_yticklabels([DISPLAY_NAMES[model] for model in MODEL_ORDER])
    absolute_axis.set_xlim(0.50, 0.88)
    absolute_axis.set_xticks([0.50, 0.60, 0.70, 0.80])
    absolute_axis.set_xlabel("Survey-weighted AUROC (95% CI)")
    absolute_axis.grid(axis="x", color="#E5E7EB", lw=0.6)
    absolute_axis.tick_params(axis="y", length=0)
    absolute_axis.text(
        0.39,
        1.045,
        "● Development",
        transform=absolute_axis.transAxes,
        ha="left",
        va="center",
        color=COLORS["development"],
    )
    absolute_axis.text(
        0.76,
        1.045,
        "■ Temporal",
        transform=absolute_axis.transAxes,
        ha="left",
        va="center",
        color=COLORS["temporal"],
    )
    add_panel_label(absolute_axis, "a")

    gap_rows = gaps.loc[
        (gaps["contrast"] == "temporal_minus_development")
        & (gaps["metric"] == "auroc")
    ].set_index("model")
    for y_position, model in zip(y_positions, MODEL_ORDER):
        row = gap_rows.loc[model]
        color = model_color(model)
        gap_axis.errorbar(
            row["difference"],
            y_position,
            xerr=np.array(
                [
                    [row["difference"] - row["ci_lower"]],
                    [row["ci_upper"] - row["difference"]],
                ]
            ),
            fmt="o",
            ms=5 if model in {"mantis_v2", "transparent_wearable"} else 4,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=2.2,
            zorder=3,
        )
    gap_axis.axvline(0, color=COLORS["ink"], lw=0.8, ls="--")
    gap_axis.set_yticks(y_positions)
    gap_axis.set_yticklabels([DISPLAY_NAMES[model] for model in MODEL_ORDER])
    gap_axis.set_xlim(-0.145, 0.105)
    gap_axis.set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10])
    gap_axis.set_xlabel("Transport gap in AUROC (95% CI)")
    gap_axis.grid(axis="x", color="#E5E7EB", lw=0.6)
    gap_axis.tick_params(axis="y", length=0)
    gap_axis.set_title(
        "MantisV2 − transparent difference-in-gaps: "
        "−0.086 [−0.128, −0.038]",
        loc="left",
        pad=8,
        fontsize=7.2,
        color=COLORS["ink"],
    )
    add_panel_label(gap_axis, "b")

    figure.canvas.draw()
    require_matplotlib_panel_alignment(
        figure,
        axes=axes,
        row_groups=[["a", "b"]],
        json_out=output.with_suffix(".alignment.json"),
        overlay_svg=output.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    flow = pd.read_csv(results_dir / "cohort_flow_two_cycle.csv")
    performance = pd.read_csv(
        results_dir / "survey_design_transport_performance.csv"
    )
    gaps = pd.read_csv(results_dir / "survey_design_transport_gaps.csv")

    flow.to_csv(output_dir / "figure_1_source_data.csv", index=False)
    performance.loc[
        (performance["metric"] == "auroc")
        & performance["model"].isin(MODEL_ORDER)
    ].to_csv(output_dir / "figure_2_absolute_source_data.csv", index=False)
    gaps.loc[
        gaps["metric"].eq("auroc")
    ].to_csv(output_dir / "figure_2_gap_source_data.csv", index=False)

    make_study_design(flow, output_dir / "figure_1_study_design")
    make_transportability(
        performance,
        gaps,
        output_dir / "figure_2_weighted_transportability",
    )


if __name__ == "__main__":
    main()
