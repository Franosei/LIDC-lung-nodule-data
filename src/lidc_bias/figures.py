"""Publication figures assembled from versioned analysis outputs."""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .errors import SourceDataError
from .provenance import base_manifest, sha256_file, write_json

plt.switch_backend("Agg")

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#666666"

REFERENCE_LABELS = {
    "median_exclude_3": "Median, exclude 3",
    "median_3_benign": "Median, 3 benign",
    "median_3_malignant": "Median, 3 malignant",
    "any_reader_4_or_5": "Any reader 4 or 5",
    "random_reader": "Random reader",
    "latent_class": "Latent class",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.3,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.16,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def _point_interval(
    axis: plt.Axes,
    y: np.ndarray,
    estimate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    colour: str = BLUE,
) -> None:
    axis.errorbar(
        estimate,
        y,
        xerr=np.vstack([estimate - lower, upper - estimate]),
        fmt="o",
        color=colour,
        ecolor=colour,
        markersize=3.5,
        capsize=2,
        elinewidth=1,
    )


def create_publication_figures(config: AnalysisConfig) -> dict[str, Any]:
    """Create the seven-panel main figure and its source-data table."""
    inputs = {
        "rq1": config.paths.results_dir / "rq1" / "summary.csv",
        "rq2": config.paths.results_dir / "rq2" / "fixed_reference_comparison.csv",
        "rq2_performance": config.paths.results_dir / "rq2" / "performance.csv",
        "rq3": config.paths.results_dir / "rq3" / "auc_comparisons.csv",
        "rq4_performance": config.paths.results_dir / "rq4" / "performance.csv",
        "rq4_panel": config.paths.results_dir / "rq4" / "panel_summary.csv",
        "rq5": config.paths.results_dir / "rq5" / "auc_summary.csv",
        "rq6_classes": config.paths.results_dir / "rq6" / "class_summary.csv",
        "rq6_ratings": config.paths.results_dir / "rq6" / "rating_probabilities.csv",
        "rq7": config.paths.results_dir / "rq7" / "auc_summary.csv",
    }
    if missing := [str(path) for path in inputs.values() if not path.is_file()]:
        raise SourceDataError(f"Figure inputs are missing: {missing}")
    rq1 = pd.read_csv(inputs["rq1"])
    rq2 = pd.read_csv(inputs["rq2"])
    rq2_performance = pd.read_csv(inputs["rq2_performance"])
    rq3 = pd.read_csv(inputs["rq3"])
    rq4 = pd.read_csv(inputs["rq4_performance"])
    rq4_panel = pd.read_csv(inputs["rq4_panel"])
    rq5 = pd.read_csv(inputs["rq5"])
    rq6_classes = pd.read_csv(inputs["rq6_classes"])
    rq6_ratings = pd.read_csv(inputs["rq6_ratings"])
    rq7 = pd.read_csv(inputs["rq7"])

    _style()
    figure = plt.figure(figsize=(7.2, 9.2), constrained_layout=True)
    grid = figure.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.08])
    axes = [figure.add_subplot(grid[row, column]) for row in range(4) for column in range(2)]

    axis = axes[0]
    rq1_plot = rq1.iloc[[0, 1, 4, 5]].copy()
    rq1_plot["label"] = [
        "Ordinal alpha",
        "Binary alpha",
        "All readers agree",
        "Benign-malignant discordance",
    ]
    y = np.arange(len(rq1_plot))[::-1]
    _point_interval(
        axis,
        y,
        rq1_plot["estimate"].to_numpy(),
        rq1_plot["ci_lower"].to_numpy(),
        rq1_plot["ci_upper"].to_numpy(),
    )
    axis.set_yticks(y, rq1_plot["label"])
    axis.set_xlim(0, 0.55)
    axis.set_xlabel("Estimate (95% CI)")
    axis.set_title("RQ1: reader disagreement", loc="left")
    _panel_label(axis, "A")

    axis = axes[1]
    rq2_plot = rq2.loc[rq2["metric"] == "auc"].copy()
    rq2_plot["label"] = rq2_plot["evaluation_reference"].map(REFERENCE_LABELS)
    y = np.arange(len(rq2_plot))[::-1]
    _point_interval(
        axis,
        y,
        rq2_plot["estimate"].to_numpy(),
        rq2_plot["ci_lower"].to_numpy(),
        rq2_plot["ci_upper"].to_numpy(),
        ORANGE,
    )
    axis.set_yticks(y, rq2_plot["label"])
    axis.set_xlim(0.62, 1.01)
    axis.set_xlabel("AUC of unchanged predictions (95% CI)")
    axis.set_title("RQ2: reference choice", loc="left")
    _panel_label(axis, "B")

    axis = axes[2]
    rq3_inflation = rq3.loc[rq3["comparison"] == "same_reader_inflation"].copy()
    rq3_inflation["label"] = rq3_inflation["stratum"].map(
        {
            "geometry_logistic": "Geometry inflation",
            "semantic_logistic": "Semantic inflation",
        }
    )
    rq3_reversal = rq3.loc[
        (rq3["comparison"] == "semantic_minus_geometry")
        & (rq3["evaluation_reader"] == "reader_c")
    ].copy()
    rq3_reversal["label"] = rq3_reversal["stratum"].map(
        {
            "same_reader": "Semantic minus geometry\n(same-reader features, reader C)",
            "other_reader": "Semantic minus geometry\n(other-reader features, reader C)",
        }
    )
    rq3_plot = pd.concat([rq3_inflation, rq3_reversal], ignore_index=True)
    y = np.arange(len(rq3_plot))[::-1]
    _point_interval(
        axis,
        y,
        rq3_plot["estimate"].to_numpy(),
        rq3_plot["ci_lower"].to_numpy(),
        rq3_plot["ci_upper"].to_numpy(),
        GREEN,
    )
    axis.axvline(0, color="#BBBBBB", linewidth=0.8)
    axis.set_yticks(y, rq3_plot["label"])
    axis.set_xlim(-0.14, 0.115)
    axis.set_xlabel("Paired AUC difference (95% CI)")
    axis.set_title("RQ3: shared-rater error", loc="left")
    _panel_label(axis, "C")

    axis = axes[3]
    rq4_plot = rq4.loc[
        (rq4["panel"] == "simulated_cautious_panel")
        & rq4["metric"].isin(["accuracy", "sensitivity", "auc"])
    ]
    for metric, colour, label in (
        ("accuracy", BLUE, "Accuracy"),
        ("sensitivity", RED, "Sensitivity"),
        ("auc", GREEN, "AUC"),
    ):
        values = rq4_plot.loc[rq4_plot["metric"] == metric]
        axis.plot(
            values["relabelled_percentage"],
            values["estimate"],
            "o-",
            color=colour,
            label=label,
            markersize=2.8,
        )
    axis.set_ylim(0.25, 0.9)
    axis.set_xlabel("Indeterminate nodules relabelled malignant (%)")
    axis.set_ylabel("Frozen-model performance")
    diameter_axis = axis.twinx()
    diameter_axis.plot(
        rq4_panel["relabelled_percentage"],
        rq4_panel["panel_implied_diameter_boundary_mm"],
        linestyle="--",
        color=GREY,
        label="Diameter boundary",
    )
    diameter_axis.set_ylabel("Panel diameter boundary (mm)", color=GREY)
    diameter_axis.tick_params(axis="y", colors=GREY)
    handles, labels = axis.get_legend_handles_labels()
    extra_handles, extra_labels = diameter_axis.get_legend_handles_labels()
    axis.legend(
        handles + extra_handles,
        labels + extra_labels,
        frameon=False,
        ncol=2,
        loc="lower left",
    )
    axis.set_title("RQ4: panel relabelling", loc="left")
    _panel_label(axis, "D")

    axis = axes[4]
    rq5_plot = rq5.loc[rq5["model"] == "combined_logistic"]
    for reference, colour, label in (
        ("consensus", BLUE, "Consensus"),
        ("latent_class", GREEN, "Latent class"),
        ("confirmed_diagnosis", RED, "Diagnosis"),
    ):
        values = rq5_plot.loc[rq5_plot["evaluation_reference"] == reference]
        axis.plot(
            values["noisy_fraction"] * 100,
            values["auc"],
            "o-",
            color=colour,
            label=label,
            markersize=3,
        )
    axis.set_ylim(0.6, 1.0)
    axis.set_xlabel("Lower-reader-count nodules added (%)")
    axis.set_ylabel("AUC")
    axis.legend(frameon=False)
    axis.set_title("RQ5: quantity versus label quality", loc="left")
    _panel_label(axis, "E")

    axis = axes[5]
    class_colours = [BLUE, ORANGE, RED, PURPLE, GREEN]
    for class_value, group in rq6_ratings.groupby("class", sort=True):
        class_row = rq6_classes.loc[rq6_classes["class"] == class_value].iloc[0]
        label = f"Class {class_value} ({class_row['mixing_proportion']:.0%})"
        axis.plot(
            group["rating"],
            group["probability"],
            "o-",
            color=class_colours[int(class_value) - 1],
            label=label,
            markersize=3,
        )
    axis.set_xticks(range(1, 6))
    axis.set_ylim(0, 0.95)
    axis.set_xlabel("Radiologist malignancy rating")
    axis.set_ylabel("Class-conditional probability")
    axis.legend(frameon=False)
    axis.set_title("RQ6: latent rating profiles", loc="left")
    _panel_label(axis, "F")

    axis = axes[6]
    rq7_radiologist = rq7.loc[rq7["analysis"] == "radiologist_reference"].copy()
    rq7_diagnosis = rq7.loc[
        (rq7["analysis"] == "confirmed_diagnosis")
        & (rq7["subset"] == "all_confirmed")
        & (rq7["score"] == "combined_logistic")
    ].copy()
    rq7_plot = pd.concat([rq7_radiologist, rq7_diagnosis], ignore_index=True)
    rq7_plot["label"] = rq7_plot["reference"].map(
        {**REFERENCE_LABELS, "diagnosis": "Confirmed diagnosis"}
    )
    y = np.arange(len(rq7_plot))[::-1]
    colours = np.array([ORANGE] * len(rq7_plot), dtype=object)
    colours[rq7_plot["reference"].eq("diagnosis").to_numpy()] = RED
    for row_index, (_, row) in enumerate(rq7_plot.iterrows()):
        _point_interval(
            axis,
            np.array([y[row_index]]),
            np.array([row["auc"]]),
            np.array([row["ci_lower"]]),
            np.array([row["ci_upper"]]),
            str(colours[row_index]),
        )
    axis.set_yticks(y, rq7_plot["label"])
    axis.set_xlim(0.4, 1.02)
    axis.set_xlabel("AUC on diagnosis-linked nodules (95% CI)")
    axis.set_title("RQ7: agreement versus diagnosis", loc="left")
    _panel_label(axis, "G")

    axes[7].axis("off")
    axes[7].text(
        0,
        0.95,
        "Central finding",
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    axes[7].text(
        0,
        0.82,
        "The apparent performance of the same clinical AI system depends on "
        "who or what is treated as truth.\n\n"
        "Shared label and feature sources inflate performance. Relabelling changes "
        "apparent performance without changing the model. More noisily labelled "
        "data does not correct the problem. Agreement with radiologists does not "
        "imply agreement with confirmed diagnosis.",
        fontsize=7.5,
        va="top",
        linespacing=1.45,
        wrap=True,
    )

    output_dir = config.paths.results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for output_format in config.figures.formats:
        path = output_dir / f"figure_1_main.{output_format}"
        save_options: dict[str, Any] = {"dpi": config.figures.dpi, "facecolor": "white"}
        if output_format == "tiff":
            save_options["pil_kwargs"] = {"compression": "tiff_lzw"}
        figure.savefig(path, **save_options)
        output_paths[output_format] = path
    plt.close(figure)

    model_labels = {
        "geometry_logistic": "Geometry logistic",
        "semantic_logistic": "Semantic logistic",
        "combined_logistic": "Combined logistic",
        "combined_random_forest": "Combined random forest",
    }
    reference_order = list(REFERENCE_LABELS)
    model_order = list(model_labels)
    heatmap_data = (
        rq2_performance.loc[rq2_performance["metric"] == "auc"]
        .pivot(index="model", columns="reference", values="estimate")
        .reindex(index=model_order, columns=reference_order)
    )
    supplementary_one, axis = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
    image = axis.imshow(heatmap_data.to_numpy(), cmap="viridis", vmin=0.7, vmax=1.0, aspect="auto")
    for row in range(len(model_order)):
        for column in range(len(reference_order)):
            value = heatmap_data.iloc[row, column]
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < 0.86 else "black",
                fontsize=7,
            )
    axis.set_xticks(np.arange(len(reference_order)), [REFERENCE_LABELS[x] for x in reference_order])
    axis.tick_params(axis="x", rotation=30)
    axis.set_yticks(np.arange(len(model_order)), [model_labels[x] for x in model_order])
    axis.set_title("Supplementary Figure 1. AUC by model and reference standard", loc="left")
    colourbar = supplementary_one.colorbar(image, ax=axis, fraction=0.03, pad=0.03)
    colourbar.set_label("AUC")
    for output_format in config.figures.formats:
        path = output_dir / f"figure_s1_rq2_matrix.{output_format}"
        options: dict[str, Any] = {"dpi": config.figures.dpi, "facecolor": "white"}
        if output_format == "tiff":
            options["pil_kwargs"] = {"compression": "tiff_lzw"}
        supplementary_one.savefig(path, **options)
        output_paths[f"s1_{output_format}"] = path
    plt.close(supplementary_one)

    supplementary_two, supplement_axes = plt.subplots(
        1, 3, figsize=(7.2, 2.8), constrained_layout=True, sharey=True
    )
    model_colours = [BLUE, ORANGE, GREEN, RED]
    for axis, reference in zip(
        supplement_axes,
        ("consensus", "latent_class", "confirmed_diagnosis"),
        strict=True,
    ):
        for model_name, colour in zip(model_order, model_colours, strict=True):
            values = rq5.loc[
                (rq5["model"] == model_name)
                & (rq5["evaluation_reference"] == reference)
            ]
            axis.plot(
                values["noisy_fraction"] * 100,
                values["auc"],
                "o-",
                color=colour,
                markersize=2.8,
                label=model_labels[model_name],
            )
        axis.set_title(reference.replace("_", " ").title())
        axis.set_xlabel("Noisy-data addition (%)")
        axis.set_ylim(0.6, 1.0)
    supplement_axes[0].set_ylabel("AUC")
    supplement_axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    supplementary_two.suptitle(
        "Supplementary Figure 2. Performance as lower-reader-count data are added",
        x=0,
        ha="left",
        fontsize=8,
    )
    for output_format in config.figures.formats:
        path = output_dir / f"figure_s2_rq5_curves.{output_format}"
        options = {"dpi": config.figures.dpi, "facecolor": "white"}
        if output_format == "tiff":
            options["pil_kwargs"] = {"compression": "tiff_lzw"}
        supplementary_two.savefig(path, **options)
        output_paths[f"s2_{output_format}"] = path
    plt.close(supplementary_two)

    source_frames = []
    for panel, frame in (
        ("A", rq1_plot),
        ("B", rq2_plot),
        ("C", rq3_plot),
        ("D_metrics", rq4_plot),
        ("D_diameter", rq4_panel),
        ("E", rq5_plot),
        ("F_classes", rq6_classes),
        ("F_ratings", rq6_ratings),
        ("G", rq7_plot),
    ):
        copy = frame.copy()
        copy.insert(0, "figure_panel", panel)
        source_frames.append(copy.astype(object))
    source_data = pd.concat(source_frames, ignore_index=True, sort=False)
    source_path = output_dir / "figure_1_source_data.csv"
    source_data.to_csv(source_path, index=False, lineterminator="\n")
    output_paths["source_data"] = source_path
    supplementary_source = pd.concat(
        [
            rq2_performance.loc[rq2_performance["metric"] == "auc"].assign(
                figure="S1"
            ),
            rq5.assign(figure="S2"),
        ],
        ignore_index=True,
        sort=False,
    )
    supplementary_source_path = output_dir / "supplementary_figure_source_data.csv"
    supplementary_source.to_csv(
        supplementary_source_path, index=False, lineterminator="\n"
    )
    output_paths["supplementary_source_data"] = supplementary_source_path

    caption_path = output_dir / "figure_1_caption.md"
    caption_path.write_text(
        "# Figure 1. Reference-label misclassification changes apparent clinical AI "
        "performance\n\n"
        "(A) Inter-reader agreement and disagreement. (B) AUC when unchanged "
        "predictions are scored against six reference definitions. (C) Same-reader "
        "AUC inflation. (D) Frozen-model performance as locally indeterminate nodules "
        "are progressively relabelled malignant; the dashed line is the panel-implied "
        "diameter boundary. (E) Combined logistic-regression AUC after adding "
        "lower-reader-count nodules with single-reader labels. (F) Rating profiles of "
        "the BIC-selected exchangeable-rater latent classes. (G) AUC on the "
        "diagnosis-linked subset against radiologist-derived references and confirmed "
        "diagnosis. Points are estimates and horizontal intervals are patient-bootstrap "
        "95% confidence intervals where shown.\n",
        encoding="utf-8",
    )
    output_paths["caption"] = caption_path
    supplementary_caption_path = output_dir / "supplementary_figure_captions.md"
    supplementary_caption_path.write_text(
        "# Supplementary figure captions\n\n"
        "## Supplementary Figure 1\n\n"
        "AUC for each model and reference-standard combination. All estimates use "
        "repeated patient-grouped out-of-fold predictions.\n\n"
        "## Supplementary Figure 2\n\n"
        "AUC for all four model specifications as progressively more "
        "lower-reader-count nodules are added to training. Test populations remain "
        "fixed within each evaluation reference.\n",
        encoding="utf-8",
    )
    output_paths["supplementary_captions"] = supplementary_caption_path

    manifest = base_manifest(config.project_root, config.config_path, "figures")
    manifest.update(
        {
            "figure": "figure_1_main",
            "dpi": config.figures.dpi,
            "formats": list(config.figures.formats),
            "inputs": {
                name: {
                    "path": str(path.relative_to(config.project_root)),
                    "sha256": sha256_file(path),
                }
                for name, path in inputs.items()
            },
            "outputs": {
                name: {
                    "path": str(path.relative_to(config.project_root)),
                    "sha256": sha256_file(path),
                }
                for name, path in output_paths.items()
            },
        }
    )
    write_json(output_dir / "manifest.json", manifest)
    return manifest
