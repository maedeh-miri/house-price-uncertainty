from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
FIGURES_DIR = ROOT / "reports" / "figures"

# Restrained visual language for project figures.
TEXT = "#172033"
MUTED = "#667085"
GRID = "#D8DEE8"
PRIMARY = "#2F5D8A"
ACCENT = "#C47A3A"
SECONDARY = "#7C8DA6"
DANGER = "#A44A4A"
LIGHT = "#EEF3F8"


def load_json(filename: str) -> dict:
    path = RESULTS_DIR / filename
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(filename: str) -> list[dict[str, str]]:
    path = RESULTS_DIR / filename
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
        }
    )


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


def add_heading(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(
        0.09,
        0.955,
        title,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="semibold",
        color=TEXT,
    )
    fig.text(
        0.09,
        0.915,
        subtitle,
        ha="left",
        va="top",
        fontsize=10.5,
        color=MUTED,
    )


def add_source_note(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.09,
        0.025,
        text,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES_DIR / f"{stem}.png",
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.25,
    )
    fig.savefig(
        FIGURES_DIR / f"{stem}.svg",
        bbox_inches="tight",
        pad_inches=0.25,
    )
    plt.close(fig)


def dollars(value: float, _position: float | None = None) -> str:
    return f"${value:,.0f}"


def percent(value: float, _position: float | None = None) -> str:
    return f"{value:.0f}%"


def load_model_comparison() -> list[dict[str, float | str]]:
    median = load_json("median_baseline_cv_summary.json")
    ridge = load_json("ridge_alpha_search_summary.json")
    elasticnet = load_json("elasticnet_search_summary.json")
    random_forest = load_json("random_forest_search_summary.json")

    return [
        {
            "model": "Median baseline",
            "mae": float(median["oof_mae"]),
            "rmse": float(median["oof_rmse"]),
        },
        {
            "model": "Ridge",
            "mae": float(ridge["best_oof_mae"]),
            "rmse": float(ridge["best_oof_rmse"]),
        },
        {
            "model": "ElasticNet",
            "mae": float(elasticnet["best_oof_mae"]),
            "rmse": float(elasticnet["best_oof_rmse"]),
        },
        {
            "model": "Random Forest",
            "mae": float(random_forest["best_oof_mae"]),
            "rmse": float(random_forest["best_oof_rmse"]),
        },
    ]


def plot_model_metric(
    rows: list[dict[str, float | str]],
    *,
    metric: str,
    title: str,
    subtitle: str,
    highlight_model: str,
    stem: str,
) -> None:
    configure_style()

    labels = [str(row["model"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    y_positions = list(range(len(rows)))

    colors = [PRIMARY if label == highlight_model else SECONDARY for label in labels]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    fig.subplots_adjust(top=0.80, left=0.24, right=0.94, bottom=0.16)
    add_heading(fig, title, subtitle)

    bars = ax.barh(y_positions, values, height=0.52, color=colors, alpha=0.94)
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(dollars))
    ax.set_xlabel(f"OOF {metric.upper()} (USD; lower is better)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    clean_axes(ax)

    max_value = max(values)
    ax.set_xlim(0, max_value * 1.20)

    for bar, label, value in zip(bars, labels, values, strict=True):
        ax.text(
            value + max_value * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"${value:,.0f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="semibold" if label == highlight_model else "normal",
            color=TEXT,
        )


    add_source_note(
        fig,
        "Source: training-only deterministic 5-fold OOF evaluation artifacts.",
    )
    save_figure(fig, stem)


def plot_conformal_sensitivity() -> None:
    configure_style()
    summary = load_json("conformal_sensitivity_summary.json")
    results = summary["results"]

    nominal = [float(row["nominal_coverage"]) * 100 for row in results]
    empirical = [float(row["empirical_coverage"]) * 100 for row in results]
    widths = [float(row["mean_interval_width"]) for row in results]

    fig, ax = plt.subplots(figsize=(8.2, 9.6))
    fig.subplots_adjust(top=0.82, left=0.13, right=0.94, bottom=0.16)
    add_heading(
        fig,
        "Conformal coverage sensitivity",
        "Frozen primary test set; labels report mean prediction-interval width.",
    )

    ax.plot(
        nominal,
        nominal,
        linestyle="--",
        linewidth=1.5,
        color=MUTED,
        label="Nominal = empirical",
        zorder=1,
    )
    ax.plot(
        nominal,
        empirical,
        marker="o",
        markersize=8,
        linewidth=2.5,
        color=PRIMARY,
        label="Empirical coverage",
        zorder=3,
    )

    primary_index = nominal.index(90.0)
    ax.scatter(
        [nominal[primary_index]],
        [empirical[primary_index]],
        s=230,
        facecolor="white",
        edgecolor=ACCENT,
        linewidth=2.4,
        zorder=4,
    )

    # Use point-based offsets instead of data-coordinate offsets so labels
    # remain stable when the figure is resized or embedded in Markdown.
    label_offsets = {
        80.0: (16, 14),
        90.0: (-18, 34),
        95.0: (18, 8),
    }
    label_alignments = {
        80.0: ("left", "bottom"),
        90.0: ("right", "bottom"),
        95.0: ("left", "center"),
    }

    for nominal_value, empirical_value, width in zip(
        nominal,
        empirical,
        widths,
        strict=True,
    ):
        dx, dy = label_offsets[nominal_value]
        ha, va = label_alignments[nominal_value]
        ax.annotate(
            f"{empirical_value:.2f}%\nmean width ${width:,.0f}",
            xy=(nominal_value, empirical_value),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=9.4,
            color=TEXT,
            bbox={
                "boxstyle": "round,pad=0.42",
                "facecolor": "white",
                "edgecolor": GRID,
                "linewidth": 0.9,
            },
            zorder=5,
        )

    # The orange ring alone marks the pre-specified primary operating point.

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xticks(nominal)
    ax.xaxis.set_major_formatter(FuncFormatter(percent))
    ax.yaxis.set_major_formatter(FuncFormatter(percent))
    ax.set_xlim(77.5, 97.5)
    ax.set_ylim(74.5, 100)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    clean_axes(ax)
    ax.legend(loc="lower right")

    add_source_note(
        fig,
        "Primary model: ElasticNet (alpha=0.1, l1_ratio=0.9) • primary test n=586.",
    )
    save_figure(fig, "conformal_sensitivity")


def plot_neighborhood_coverage() -> None:
    configure_style()
    summary = load_json("final_primary_test_summary.json")
    rows = [
        row
        for row in summary["subgroup_results"]
        if row["interpretation"] in {"primary", "exploratory"}
    ]
    rows.sort(key=lambda row: float(row["empirical_coverage"]))

    fig, ax = plt.subplots(figsize=(9.2, 8.1))
    fig.subplots_adjust(top=0.82, left=0.23, right=0.94, bottom=0.14)
    add_heading(
        fig,
        "Neighborhood coverage diagnostics",
        "Only primary (n≥50) and exploratory (20≤n<50) neighborhoods are shown.",
    )

    y_positions = list(range(len(rows)))
    labels = [str(row["neighborhood"]) for row in rows]

    for interpretation, marker, color in [
        ("exploratory", "o", SECONDARY),
        ("primary", "s", PRIMARY),
    ]:
        subset = [
            (index, row)
            for index, row in enumerate(rows)
            if row["interpretation"] == interpretation
        ]
        if not subset:
            continue
        xs = [float(row["empirical_coverage"]) * 100 for _, row in subset]
        ys = [index for index, _ in subset]
        sizes = [65 + float(row["n"]) * 2.2 for _, row in subset]
        ax.scatter(
            xs,
            ys,
            s=sizes,
            marker=marker,
            color=color,
            alpha=0.92,
            label="Primary" if interpretation == "primary" else "Exploratory",
            zorder=3,
        )

    ax.axvline(90, linestyle="--", linewidth=1.4, color=ACCENT, zorder=1)
    ax.text(
        90.4,
        len(rows) - 0.35,
        "90% nominal target",
        color=ACCENT,
        fontsize=9,
        va="top",
    )

    for index, row in enumerate(rows):
        coverage = float(row["empirical_coverage"]) * 100
        ax.text(
            coverage + 0.8,
            index,
            f"{coverage:.1f}%  n={int(row['n'])}",
            va="center",
            ha="left",
            fontsize=8.8,
            color=TEXT,
        )

    nridght_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row["neighborhood"] == "NridgHt"
        ),
        None,
    )
    if nridght_index is not None:
        row = rows[nridght_index]
        coverage = float(row["empirical_coverage"]) * 100
        ax.annotate(
            "Exploratory undercoverage signal",
            xy=(coverage, nridght_index),
            xytext=(18, 34),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color=DANGER,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": GRID,
                "linewidth": 0.7,
            },
            arrowprops={
                "arrowstyle": "->",
                "color": DANGER,
                "linewidth": 1.0,
                "shrinkA": 4,
                "shrinkB": 4,
            },
            zorder=5,
        )

    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("Empirical coverage")
    ax.xaxis.set_major_formatter(FuncFormatter(percent))
    ax.set_xlim(64, 104)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    clean_axes(ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=PRIMARY,
            markeredgecolor=PRIMARY,
            markersize=8,
            label="Primary (n≥50)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=SECONDARY,
            markeredgecolor=SECONDARY,
            markersize=8,
            label="Exploratory (20≤n<50)",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    add_source_note(
        fig,
        "Split conformal targets marginal coverage; subgroup results are diagnostic, not conditional guarantees.",
    )
    save_figure(fig, "neighborhood_coverage")


def plot_top_errors() -> None:
    configure_style()
    rows = load_csv("posthoc_top_errors.csv")
    primary = load_json("final_primary_test_summary.json")
    q_hat = float(primary["conformal"]["radius"])

    parsed = [
        {
            "label": f"Order {row['Order']} · {row['Neighborhood']}",
            "absolute_error": float(row["absolute_error"]),
        }
        for row in rows
    ]
    parsed.sort(key=lambda row: float(row["absolute_error"]))

    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    fig.subplots_adjust(top=0.81, left=0.31, right=0.94, bottom=0.15)
    add_heading(
        fig,
        "Largest held-out prediction errors",
        "Post-hoc diagnostic only; these observations were not used to tune the frozen system.",
    )

    labels = [str(row["label"]) for row in parsed]
    values = [float(row["absolute_error"]) for row in parsed]
    y_positions = list(range(len(parsed)))
    colors = [SECONDARY] * len(parsed)
    colors[-1] = DANGER

    bars = ax.barh(y_positions, values, height=0.56, color=colors, alpha=0.94)
    ax.set_yticks(y_positions, labels)
    ax.xaxis.set_major_formatter(FuncFormatter(dollars))
    ax.set_xlabel("Absolute error (USD)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    clean_axes(ax)

    ax.axvline(q_hat, linestyle="--", linewidth=1.4, color=ACCENT, zorder=2)
    ax.annotate(
        f"Primary conformal radius  ${q_hat:,.0f}",
        xy=(q_hat, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(8, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.8,
        color=ACCENT,
        annotation_clip=False,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": GRID,
            "linewidth": 0.7,
        },
        zorder=5,
    )

    max_value = max(values)
    ax.set_xlim(0, max_value * 1.16)

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + max_value * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"${value:,.0f}",
            va="center",
            ha="left",
            fontsize=9,
            color=TEXT,
        )

    add_source_note(
        fig,
        "The largest absolute residual is approximately $595k; tail errors strongly influence RMSE.",
    )
    save_figure(fig, "posthoc_top_errors")


def plot_protocol_coverage() -> None:
    configure_style()
    primary = load_json("final_primary_test_summary.json")
    temporal = load_json("temporal_stress_summary.json")

    labels = ["Primary random", "Temporal stress"]
    coverages = [
        float(primary["interval_metrics"]["empirical_coverage"]) * 100,
        float(temporal["metrics"]["empirical_coverage"]) * 100,
    ]
    widths = [
        float(primary["interval_metrics"]["mean_interval_width"]),
        float(temporal["metrics"]["mean_interval_width"]),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    fig.subplots_adjust(top=0.80, left=0.14, right=0.94, bottom=0.18)
    add_heading(
        fig,
        "Empirical coverage across evaluation protocols",
        "Temporal results are a stress test under shift, not an exchangeability-based guarantee.",
    )

    bars = ax.bar(labels, coverages, width=0.52, color=[PRIMARY, SECONDARY], alpha=0.95)
    ax.axhline(90, linestyle="--", linewidth=1.4, color=ACCENT)
    ax.text(1.47, 90.3, "90% nominal", ha="right", va="bottom", fontsize=9, color=ACCENT)

    ax.set_ylabel("Empirical coverage")
    ax.yaxis.set_major_formatter(FuncFormatter(percent))
    ax.set_ylim(84, 96)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    clean_axes(ax)

    for bar, coverage, width in zip(bars, coverages, widths, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            coverage + 0.35,
            f"{coverage:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color=TEXT,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            85.0,
            f"Mean width\n${width:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            color=TEXT,
            bbox={
                "boxstyle": "round,pad=0.30",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )

    add_source_note(
        fig,
        "Primary test n=586 • temporal test n=341. Point-error metrics are not directly ranked across different partitions.",
    )
    save_figure(fig, "protocol_coverage_comparison")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model_rows = load_model_comparison()
    plot_model_metric(
        model_rows,
        metric="mae",
        title="Point-model selection: OOF MAE",
        subtitle="Training-only deterministic 5-fold evaluation; MAE was the pre-specified selection metric.",
        highlight_model="ElasticNet",
        stem="model_selection_oof_mae",
    )
    plot_model_metric(
        model_rows,
        metric="rmse",
        title="Point-model comparison: OOF RMSE",
        subtitle="Random Forest achieved the lowest OOF RMSE, while MAE remained the primary selection criterion.",
        highlight_model="Random Forest",
        stem="model_selection_oof_rmse",
    )
    plot_conformal_sensitivity()
    plot_neighborhood_coverage()
    plot_top_errors()
    plot_protocol_coverage()

    generated = sorted(path.name for path in FIGURES_DIR.iterdir() if path.suffix in {".png", ".svg"})
    print("Generated report figures:")
    for name in generated:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
