"""Create only the figure and two compact tables specified by the proposal."""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import AnalysisConfig
from .errors import SourceDataError
from .provenance import base_manifest, sha256_file, write_json


def _effect_label(row: pd.Series) -> str:
    if row["comparison"] == "label_assignment":
        return "Label assignment\n(3 malignant - 3 benign)"
    reference = str(row.get("reference", "")).replace("median_3_", "")
    return f"Case exclusion\n(3 {reference} rule)"


def create_publication_outputs(config: AnalysisConfig) -> dict[str, Any]:
    """Build the prespecified effect figure and compact result tables."""
    inputs = {
        "effects": config.paths.results_dir / "label_case_effects" / "effects.csv",
        "references": config.paths.results_dir
        / "label_case_effects"
        / "reference_definitions.csv",
        "shared": config.paths.results_dir / "shared_assessment" / "effect_summary.csv",
    }
    if missing := [str(path) for path in inputs.values() if not path.is_file()]:
        raise SourceDataError(f"Publication inputs are missing: {missing}")
    effects = pd.read_csv(inputs["effects"])
    references = pd.read_csv(inputs["references"])
    shared = pd.read_csv(inputs["shared"])

    order = ["label_assignment", "case_exclusion", "case_exclusion"]
    reference_order = ["", "median_3_benign", "median_3_malignant"]
    ordered_rows = []
    for comparison, reference in zip(order, reference_order, strict=True):
        subset = effects.loc[effects["comparison"] == comparison]
        if reference:
            subset = subset.loc[subset["reference"] == reference]
        ordered_rows.append(subset)
    plot_data = pd.concat(ordered_rows, ignore_index=True)
    plot_data["label"] = plot_data.apply(_effect_label, axis=1)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("auc", "brier_score"),
        ("AUC difference", "Brier-score difference"),
        strict=True,
    ):
        values = plot_data.loc[plot_data["metric"] == metric].copy()
        y = np.arange(len(values))[::-1]
        axis.errorbar(
            values["estimate"],
            y,
            xerr=np.vstack(
                [
                    values["estimate"] - values["ci_lower"],
                    values["ci_upper"] - values["estimate"],
                ]
            ),
            fmt="o",
            color="#24557A",
            ecolor="#24557A",
            capsize=2,
            markersize=4,
        )
        axis.axvline(0, color="#999999", linewidth=0.8)
        axis.set_yticks(y, values["label"] if metric == "auc" else [""] * len(values))
        axis.set_xlabel("Paired difference (95% patient-bootstrap CI)")
        axis.set_title(title, loc="left", fontweight="bold")
    axes[0].text(-0.12, 1.05, "A", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.12, 1.05, "B", transform=axes[1].transAxes, fontweight="bold")

    output_dir = config.paths.results_dir / "publication"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Any] = {}
    for output_format in config.publication.formats:
        path = output_dir / f"figure_1_annotation_effects.{output_format}"
        options: dict[str, Any] = {"dpi": config.publication.dpi, "facecolor": "white"}
        if output_format == "tiff":
            options["pil_kwargs"] = {"compression": "tiff_lzw"}
        figure.savefig(path, **options)
        output_paths[f"figure_{output_format}"] = path
    plt.close(figure)

    caption_path = output_dir / "figure_1_caption.md"
    caption_path.write_text(
        "# Figure 1. Label-assignment and case-exclusion effects\n\n"
        "(A) Paired AUC differences and (B) paired Brier-score differences for the "
        "fixed combined-logistic predictions. Label assignment compares the two "
        "full-cohort median rules on identical nodules (median-3-malignant minus "
        "median-3-benign). Case exclusion compares the restricted population with "
        "the full population while retaining the stated label rule. Intervals use "
        "patient-level bootstrap resampling.\n",
        encoding="utf-8",
    )
    output_paths["figure_caption"] = caption_path

    table_one = references.loc[
        references["reference"].isin(
            [
                "median_exclude_3",
                "median_3_benign",
                "median_3_malignant",
                "any_reader_4_or_5",
                "random_reader",
            ]
        ),
        [
            "reference",
            "description",
            "available_nodules",
            "malignant_nodules",
            "prevalence",
            "retained_after_excluding_median_3",
            "malignant_after_excluding_median_3",
        ],
    ]
    table_one_path = output_dir / "table_1_reference_definitions.csv"
    table_one.to_csv(table_one_path, index=False, lineterminator="\n")
    output_paths["table_1"] = table_one_path

    table_two = shared[
        [
            "geometry",
            "evaluation",
            "metric",
            "effect_definition",
            "mean_effect",
            "ci_lower",
            "ci_upper",
            "assignment_sd",
            "partition_sd",
        ]
    ]
    table_two_path = output_dir / "table_2_shared_assessment_effects.csv"
    table_two.to_csv(table_two_path, index=False, lineterminator="\n")
    output_paths["table_2"] = table_two_path

    manifest = base_manifest(config.project_root, config.config_path, "publication")
    manifest.update(
        {
            "figure": "label_assignment_and_case_exclusion_effects",
            "tables": ["reference_definitions", "matched_shared_assessment_effects"],
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
