r"""
Generate baseline and HiFi-GAN tables/graphs directly from local experiment outputs.

The script recursively scans your project folders, detects the relevant CSV files,
normalizes common column-name variations, calculates summary tables, and saves
all outputs into a separate report folder.

Recommended command
-------------------
python generate_reports_from_local_outputs.py ^
  --project-root "C:\pd-speech-crosslingual" ^
  --output-dir "C:\pd-speech-crosslingual\supervisor_report_outputs"

Optional explicit paths
-----------------------
python generate_reports_from_local_outputs.py ^
  --project-root "C:\pd-speech-crosslingual" ^
  --baseline-file "C:\pd-speech-crosslingual\outputs\tables\full_model_comparison.csv" ^
  --hifigan-comparison-file "C:\pd-speech-crosslingual\voice_conversion\logs_full\full_evaluation_comparison_summary.csv" ^
  --duration-file "C:\pd-speech-crosslingual\voice_conversion\logs_full\full_original_vs_generated_duration_comparison.csv" ^
  --audio-validation-file "C:\pd-speech-crosslingual\voice_conversion\logs_full\generated_full_audio_inspection_summary.csv" ^
  --output-dir "C:\pd-speech-crosslingual\supervisor_report_outputs"

Dependencies
------------
pip install pandas numpy matplotlib
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =====================================================================
# General configuration
# =====================================================================

DPI = 300

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": DPI,
    }
)


# =====================================================================
# Filename candidates
# =====================================================================

BASELINE_FILENAME_CANDIDATES = [
    "full_model_comparison.csv",
    "model_layer_comparison.csv",
    "baseline_model_comparison.csv",
    "baseline_summary.csv",
]

HIFIGAN_COMPARISON_FILENAME_CANDIDATES = [
    "full_evaluation_comparison_summary.csv",
    "original_vs_generated_classification_summary.csv",
    "original_vs_reconstructed_classification_summary.csv",
    "hifigan_evaluation_comparison_summary.csv",
    "subset_80_evaluation_comparison_summary.csv",
]

DURATION_FILENAME_CANDIDATES = [
    "full_original_vs_generated_duration_comparison.csv",
    "original_vs_generated_duration_comparison.csv",
    "original_vs_reconstructed_duration_comparison.csv",
]

AUDIO_VALIDATION_FILENAME_CANDIDATES = [
    "generated_full_audio_inspection_summary.csv",
    "full_inspection_summary.csv",
    "generated_audio_inspection_summary.csv",
    "audio_validation_summary.csv",
]

DATASET_INDEX_FILENAME_CANDIDATES = [
    "dataset_index_readtext.csv",
    "dataset_index.csv",
]


# =====================================================================
# Column aliases
# =====================================================================

ALIASES = {
    "scenario": [
        "scenario",
        "classification_scenario",
        "evaluation_scenario",
        "experiment",
        "direction",
        "train_test",
        "train_test_scenario",
        "setting",
    ],
    "train_language": [
        "train_language",
        "training_language",
        "train_lang",
        "source_language",
        "source_lang",
    ],
    "test_language": [
        "test_language",
        "testing_language",
        "test_lang",
        "target_language",
        "target_lang",
    ],
    "model": [
        "model",
        "ssl_model",
        "feature_model",
        "representation_model",
        "encoder",
    ],
    "layer": [
        "layer",
        "model_layer",
        "hidden_layer",
        "feature_layer",
        "selected_layer",
    ],
    "classifier": [
        "classifier",
        "classification_model",
        "clf",
        "classifier_name",
    ],
    "uar": [
        "uar",
        "test_uar",
        "mean_uar",
        "uar_mean",
        "best_uar",
        "score_uar",
    ],
    "original_uar": [
        "original_uar",
        "uar_orig",
        "original_mean_uar",
        "uar_original",
        "mean_original_uar",
        "baseline_uar",
        "original_score",
        "original",
    ],
    "generated_uar": [
        "generated_uar",
        "uar_gen",
        "reconstructed_uar",
        "generated_mean_uar",
        "reconstructed_mean_uar",
        "uar_generated",
        "uar_reconstructed",
        "mean_generated_uar",
        "mean_reconstructed_uar",
        "generated_score",
        "reconstructed_score",
        "generated",
        "reconstructed",
    ],
    "difference": [
        "difference",
        "uar_diff",
        "uar_difference",
        "delta_uar",
        "generated_minus_original",
        "reconstructed_minus_original",
        "mean_difference",
    ],
    "absolute_difference": [
        "absolute_difference",
        "abs_difference",
        "absolute_uar_difference",
        "abs_uar_difference",
        "mean_absolute_difference",
    ],
    "language": [
        "language",
        "lang",
        "dataset_language",
    ],
    "label": [
        "label",
        "class",
        "diagnosis",
        "group",
        "target",
        "condition",
    ],
    "original_duration": [
        "original_duration",
        "original_duration_sec",
        "original_duration_seconds",
        "duration_original",
        "source_duration",
    ],
    "generated_duration": [
        "generated_duration",
        "reconstructed_duration",
        "generated_duration_sec",
        "reconstructed_duration_sec",
        "duration_generated",
        "duration_reconstructed",
        "output_duration",
    ],
    "duration_difference": [
        "duration_difference",
        "absolute_duration_difference",
        "abs_duration_difference",
        "duration_diff",
        "duration_difference_sec",
    ],
    "original_rms": [
        "original_rms",
        "original_rms_energy",
        "rms_original",
        "source_rms",
    ],
    "generated_rms": [
        "generated_rms",
        "reconstructed_rms",
        "generated_rms_energy",
        "reconstructed_rms_energy",
        "rms_generated",
        "rms_reconstructed",
        "output_rms",
    ],
    "valid": [
        "valid",
        "is_valid",
        "validation_passed",
        "passed_validation",
        "audio_valid",
        "status",
    ],
    "file": [
        "file",
        "filename",
        "file_name",
        "audio_file",
        "path",
        "audio_path",
    ],
}


# =====================================================================
# Utility functions
# =====================================================================

def normalize_name(value: object) -> str:
    """Normalize a filename, column name, model name, or scenario string."""
    text = str(value).strip().lower()
    text = text.replace("→", " to ")
    text = text.replace("->", " to ")
    text = text.replace("+", " plus ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read CSV using several common encodings and delimiter detection."""
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError(
        f"Could not read CSV: {path}\n" + "\n".join(errors)
    )


def find_column(df: pd.DataFrame, canonical_name: str) -> Optional[str]:
    """Find a column using aliases and normalized comparisons."""
    normalized_columns = {normalize_name(col): col for col in df.columns}

    for alias in ALIASES[canonical_name]:
        normalized_alias = normalize_name(alias)
        if normalized_alias in normalized_columns:
            return normalized_columns[normalized_alias]

    # Partial fallback, but only for aliases of reasonable length.
    for alias in ALIASES[canonical_name]:
        normalized_alias = normalize_name(alias)
        if len(normalized_alias) < 4:
            continue
        for normalized_col, original_col in normalized_columns.items():
            if normalized_alias in normalized_col:
                return original_col

    return None


def numeric_series(series: pd.Series) -> pd.Series:
    """Convert common numeric strings to floating point."""
    cleaned = (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def recursive_csvs(root: Path) -> list[Path]:
    """Return all CSVs below root, excluding report-output folders."""
    blocked_parts = {
        "supervisor_report_outputs",
        "baseline_hifigan_report_outputs",
        "report_outputs",
    }

    result = []
    for path in root.rglob("*.csv"):
        lowered_parts = {part.lower() for part in path.parts}
        if lowered_parts.intersection(blocked_parts):
            continue
        result.append(path)
    return result


def score_candidate(path: Path, names: Iterable[str], keywords: Iterable[str]) -> int:
    """Score a candidate file based on filename and path keywords."""
    score = 0
    filename = path.name.lower()
    full_path = str(path).lower()

    for rank, candidate in enumerate(names):
        if filename == candidate.lower():
            score += 1000 - rank * 10
        elif candidate.lower().replace(".csv", "") in filename:
            score += 500 - rank * 5

    for keyword in keywords:
        if keyword.lower() in filename:
            score += 80
        if keyword.lower() in full_path:
            score += 20

    # Prefer full results over subset/pilot where possible.
    if "full" in filename or "logs_full" in full_path:
        score += 100
    if "subset" in filename or "pilot" in filename:
        score -= 50

    return score


def autodetect_file(
    root: Path,
    candidates: Iterable[str],
    keywords: Iterable[str],
    description: str,
) -> Optional[Path]:
    """Detect the most likely relevant CSV."""
    csv_files = recursive_csvs(root)
    scored = [
        (score_candidate(path, candidates, keywords), path)
        for path in csv_files
    ]
    scored = [(score, path) for score, path in scored if score > 0]

    if not scored:
        print(f"[WARNING] Could not auto-detect {description}.")
        return None

    scored.sort(key=lambda item: (item[0], item[1].stat().st_mtime), reverse=True)
    selected = scored[0][1]
    print(f"[FOUND] {description}: {selected}")
    return selected


def require_path(path: Optional[Path], description: str) -> Path:
    if path is None:
        raise FileNotFoundError(
            f"Required input not found: {description}. "
            "Pass its path explicitly on the command line."
        )
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def create_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "figures_png": output_dir / "figures_png",
        "figures_pdf": output_dir / "figures_pdf",
        "logs": output_dir / "logs",
    }
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    return dirs


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_figure(fig: plt.Figure, stem: str, dirs: dict[str, Path]) -> None:
    fig.tight_layout()
    fig.savefig(dirs["figures_png"] / f"{stem}.png", bbox_inches="tight")
    fig.savefig(dirs["figures_pdf"] / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(
    ax: plt.Axes,
    bars,
    decimals: int = 4,
    offset: float = 0.005,
    signed: bool = False,
) -> None:
    for bar in bars:
        value = float(bar.get_height())
        y = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        template = f"{{:{'+' if signed else ''}.{decimals}f}}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            template.format(value),
            ha="center",
            va=va,
            fontsize=8.2,
        )


def scenario_display(value: object) -> str:
    """Standardize common scenario names for readable outputs."""
    norm = normalize_name(value)

    rules = [
        (["spanish_to_spanish", "sp_to_sp"], "Spanish → Spanish"),
        (["german_to_german", "de_to_de"], "German → German"),
        (["spanish_to_german", "sp_to_de"], "Spanish → German"),
        (["german_to_spanish", "de_to_sp"], "German → Spanish"),
        (
            [
                "spanish_plus_german",
                "combined",
                "sp_plus_de",
                "spanish_german_to_spanish_german",
            ],
            "Spanish + German",
        ),
    ]

    for patterns, display in rules:
        if any(pattern in norm for pattern in patterns):
            return display

    return str(value).strip()


def model_display(value: object) -> str:
    norm = normalize_name(value)
    if "xlsr" in norm:
        return "XLSR"
    if "wavlm" in norm:
        return "WavLM"
    if "wav2vec" in norm or "wav_2_vec" in norm:
        return "Wav2Vec2"
    return str(value).strip()


def label_display(value: object) -> str:
    norm = normalize_name(value)
    if norm in {"pd", "parkinson", "parkinsons", "parkinson_disease", "1"}:
        return "PD"
    if norm in {"hc", "healthy", "healthy_control", "control", "0"}:
        return "HC"
    return str(value).strip()


# =====================================================================
# Data normalization
# =====================================================================

def load_baseline_results(path: Path) -> pd.DataFrame:
    raw = read_csv_flexible(path)

    scenario_col = find_column(raw, "scenario")
    train_language_col = find_column(raw, "train_language")
    test_language_col = find_column(raw, "test_language")
    model_col = find_column(raw, "model")
    layer_col = find_column(raw, "layer")
    classifier_col = find_column(raw, "classifier")
    uar_col = find_column(raw, "uar")

    if scenario_col is None:
        if train_language_col is not None and test_language_col is not None:
            train_values = raw[train_language_col].astype(str).str.strip()
            test_values = raw[test_language_col].astype(str).str.strip()
            scenario_values = train_values + " → " + test_values
        else:
            raise ValueError(
                "Baseline CSV has neither a scenario column nor both "
                "train_language and test_language columns.\n"
                f"File: {path}\nColumns: {list(raw.columns)}"
            )
    else:
        scenario_values = raw[scenario_col]

    missing = [
        name
        for name, col in [
            ("model", model_col),
            ("layer", layer_col),
            ("classifier", classifier_col),
            ("UAR", uar_col),
        ]
        if col is None
    ]

    if missing:
        raise ValueError(
            f"Baseline CSV is missing recognizable columns: {missing}\n"
            f"File: {path}\nColumns: {list(raw.columns)}"
        )

    df = pd.DataFrame(
        {
            "Scenario": scenario_values.map(scenario_display),
            "Model": raw[model_col].map(model_display),
            "Layer": raw[layer_col],
            "Classifier": raw[classifier_col].astype(str).str.strip(),
            "UAR": numeric_series(raw[uar_col]),
        }
    )

    df["Layer"] = pd.to_numeric(df["Layer"], errors="coerce")
    df = df.dropna(subset=["Scenario", "Model", "UAR"]).copy()
    return df


def load_hifigan_comparison(path: Path) -> pd.DataFrame:
    raw = read_csv_flexible(path)

    scenario_col = find_column(raw, "scenario")
    train_language_col = find_column(raw, "train_language")
    test_language_col = find_column(raw, "test_language")
    model_col = find_column(raw, "model")
    layer_col = find_column(raw, "layer")
    classifier_col = find_column(raw, "classifier")
    original_col = find_column(raw, "original_uar")
    generated_col = find_column(raw, "generated_uar")
    difference_col = find_column(raw, "difference")
    absolute_difference_col = find_column(raw, "absolute_difference")

    if scenario_col is None:
        if train_language_col is not None and test_language_col is not None:
            train_values = raw[train_language_col].astype(str).str.strip()
            test_values = raw[test_language_col].astype(str).str.strip()
            scenario_values = train_values + " → " + test_values
        else:
            raise ValueError(
                "HiFi-GAN comparison CSV has neither a scenario column nor "
                "both train_language and test_language columns.\n"
                f"File: {path}\nColumns: {list(raw.columns)}"
            )
    else:
        scenario_values = raw[scenario_col]

    missing = [
        name
        for name, col in [
            ("model", model_col),
            ("original UAR", original_col),
            ("generated/reconstructed UAR", generated_col),
        ]
        if col is None
    ]

    if missing:
        raise ValueError(
            f"HiFi-GAN comparison CSV is missing recognizable columns: {missing}\n"
            f"File: {path}\nColumns: {list(raw.columns)}"
        )

    data = {
        "Scenario": scenario_values.map(scenario_display),
        "Model": raw[model_col].map(model_display),
        "Original UAR": numeric_series(raw[original_col]),
        "Reconstructed UAR": numeric_series(raw[generated_col]),
    }

    if layer_col is not None:
        data["Layer"] = pd.to_numeric(raw[layer_col], errors="coerce")
    else:
        data["Layer"] = np.nan

    if classifier_col is not None:
        data["Classifier"] = raw[classifier_col].astype(str).str.strip()
    else:
        data["Classifier"] = "Unknown"

    df = pd.DataFrame(data)

    if difference_col is not None:
        df["Difference"] = numeric_series(raw[difference_col])
    else:
        df["Difference"] = df["Reconstructed UAR"] - df["Original UAR"]

    if absolute_difference_col is not None:
        df["Absolute difference"] = numeric_series(raw[absolute_difference_col])
    else:
        df["Absolute difference"] = df["Difference"].abs()

    df = df.dropna(
        subset=["Scenario", "Model", "Original UAR", "Reconstructed UAR"]
    ).copy()
    return df


def load_dataset_index(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None

    raw = read_csv_flexible(path)
    language_col = find_column(raw, "language")
    label_col = find_column(raw, "label")

    if language_col is None or label_col is None:
        print(
            "[WARNING] Dataset index found, but language/label columns "
            f"could not be recognized: {path}"
        )
        return None

    df = pd.DataFrame(
        {
            "Language": raw[language_col].astype(str).str.strip().str.title(),
            "Class": raw[label_col].map(label_display),
        }
    )
    return df.dropna()


def load_duration_results(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None

    raw = read_csv_flexible(path)
    original_col = find_column(raw, "original_duration")
    generated_col = find_column(raw, "generated_duration")
    difference_col = find_column(raw, "duration_difference")

    if original_col is None and generated_col is None and difference_col is None:
        print(
            "[WARNING] Duration CSV found, but duration columns could not "
            f"be recognized: {path}"
        )
        return None

    df = pd.DataFrame(index=raw.index)

    if original_col is not None:
        df["Original duration"] = numeric_series(raw[original_col])

    if generated_col is not None:
        df["Reconstructed duration"] = numeric_series(raw[generated_col])

    if difference_col is not None:
        df["Duration difference"] = numeric_series(raw[difference_col]).abs()
    elif original_col is not None and generated_col is not None:
        df["Duration difference"] = (
            df["Reconstructed duration"] - df["Original duration"]
        ).abs()

    return df


def load_audio_validation(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None

    raw = read_csv_flexible(path)
    original_rms_col = find_column(raw, "original_rms")
    generated_rms_col = find_column(raw, "generated_rms")
    valid_col = find_column(raw, "valid")
    file_col = find_column(raw, "file")

    df = pd.DataFrame(index=raw.index)

    if file_col is not None:
        df["File"] = raw[file_col].astype(str)

    if original_rms_col is not None:
        df["Original RMS"] = numeric_series(raw[original_rms_col])

    if generated_rms_col is not None:
        df["Reconstructed RMS"] = numeric_series(raw[generated_rms_col])

    if valid_col is not None:
        valid_raw = raw[valid_col]
        if pd.api.types.is_bool_dtype(valid_raw):
            df["Valid"] = valid_raw
        else:
            normalized = valid_raw.astype(str).map(normalize_name)
            valid_values = {
                "true",
                "1",
                "yes",
                "valid",
                "pass",
                "passed",
                "ok",
                "success",
                "successful",
            }
            df["Valid"] = normalized.isin(valid_values)

    if df.empty:
        print(
            "[WARNING] Audio-validation CSV found, but RMS/validation columns "
            f"could not be recognized: {path}"
        )
        return None

    return df


# =====================================================================
# Summary tables
# =====================================================================

def create_baseline_best_table(baseline: pd.DataFrame) -> pd.DataFrame:
    idx = baseline.groupby("Scenario")["UAR"].idxmax()
    best = baseline.loc[idx].copy()
    best = best[
        ["Scenario", "Model", "Layer", "Classifier", "UAR"]
    ].rename(
        columns={
            "Model": "Best model",
            "UAR": "Best UAR",
        }
    )

    scenario_order = [
        "Spanish → Spanish",
        "German → German",
        "Spanish → German",
        "German → Spanish",
        "Spanish + German",
    ]
    best["Scenario"] = pd.Categorical(
        best["Scenario"], categories=scenario_order, ordered=True
    )
    return best.sort_values("Scenario").reset_index(drop=True)


def create_baseline_gap_table(best: pd.DataFrame) -> pd.DataFrame:
    scores = dict(zip(best["Scenario"].astype(str), best["Best UAR"]))

    pairs = [
        (
            "Spanish reference vs. Spanish → German",
            "Spanish → Spanish",
            "Spanish → German",
        ),
        (
            "German reference vs. German → Spanish",
            "German → German",
            "German → Spanish",
        ),
    ]

    rows = []
    for comparison, within_name, cross_name in pairs:
        if within_name not in scores or cross_name not in scores:
            continue
        within = float(scores[within_name])
        cross = float(scores[cross_name])
        absolute_reduction = within - cross
        relative_reduction = (
            absolute_reduction / within * 100 if within != 0 else np.nan
        )
        rows.append(
            [
                comparison,
                within,
                cross,
                absolute_reduction,
                relative_reduction,
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Comparison",
            "Within-language UAR",
            "Crosslingual UAR",
            "Absolute reduction",
            "Relative reduction (%)",
        ],
    )


def create_overall_hifigan_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    within_005 = int((comparison["Absolute difference"] <= 0.05).sum())
    total = int(len(comparison))

    return pd.DataFrame(
        [
            ["Number of paired configurations", total],
            ["Mean original UAR", comparison["Original UAR"].mean()],
            [
                "Mean reconstructed UAR",
                comparison["Reconstructed UAR"].mean(),
            ],
            ["Mean UAR difference", comparison["Difference"].mean()],
            [
                "Mean absolute UAR difference",
                comparison["Absolute difference"].mean(),
            ],
            [
                "Median absolute UAR difference",
                comparison["Absolute difference"].median(),
            ],
            [
                "Maximum absolute UAR difference",
                comparison["Absolute difference"].max(),
            ],
            ["Configurations within ±0.05 UAR", f"{within_005} of {total}"],
        ],
        columns=["Measure", "Result"],
    )


def create_scenario_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    summary = (
        comparison.groupby("Scenario", as_index=False)
        .agg(
            original_mean_uar=("Original UAR", "mean"),
            reconstructed_mean_uar=("Reconstructed UAR", "mean"),
            mean_difference=("Difference", "mean"),
            mean_absolute_difference=("Absolute difference", "mean"),
            configuration_count=("Difference", "size"),
        )
        .rename(
            columns={
                "Scenario": "Evaluation scenario",
                "original_mean_uar": "Original mean UAR",
                "reconstructed_mean_uar": "Reconstructed mean UAR",
                "mean_difference": "Difference",
                "mean_absolute_difference": "Mean absolute difference",
                "configuration_count": "Configurations",
            }
        )
    )

    scenario_order = [
        "German → German",
        "German → Spanish",
        "Spanish → German",
        "Spanish → Spanish",
        "Spanish + German",
    ]
    summary["Evaluation scenario"] = pd.Categorical(
        summary["Evaluation scenario"],
        categories=scenario_order,
        ordered=True,
    )
    return summary.sort_values("Evaluation scenario").reset_index(drop=True)


def create_model_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    return (
        comparison.groupby("Model", as_index=False)
        .agg(
            original_mean_uar=("Original UAR", "mean"),
            reconstructed_mean_uar=("Reconstructed UAR", "mean"),
            mean_difference=("Difference", "mean"),
            mean_absolute_difference=("Absolute difference", "mean"),
            configuration_count=("Difference", "size"),
        )
        .rename(
            columns={
                "original_mean_uar": "Original mean UAR",
                "reconstructed_mean_uar": "Reconstructed mean UAR",
                "mean_difference": "Difference",
                "mean_absolute_difference": "Mean absolute difference",
                "configuration_count": "Configurations",
            }
        )
        .sort_values("Model")
        .reset_index(drop=True)
    )


def create_layer_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    valid = comparison.dropna(subset=["Layer"]).copy()
    if valid.empty:
        return pd.DataFrame()

    return (
        valid.groupby("Layer", as_index=False)
        .agg(
            original_mean_uar=("Original UAR", "mean"),
            reconstructed_mean_uar=("Reconstructed UAR", "mean"),
            mean_difference=("Difference", "mean"),
            mean_absolute_difference=("Absolute difference", "mean"),
            configuration_count=("Difference", "size"),
        )
        .rename(
            columns={
                "original_mean_uar": "Original mean UAR",
                "reconstructed_mean_uar": "Reconstructed mean UAR",
                "mean_difference": "Difference",
                "mean_absolute_difference": "Mean absolute difference",
                "configuration_count": "Configurations",
            }
        )
        .sort_values("Layer")
        .reset_index(drop=True)
    )


def create_classifier_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    valid = comparison[
        comparison["Classifier"].astype(str).str.lower() != "unknown"
    ].copy()
    if valid.empty:
        return pd.DataFrame()

    return (
        valid.groupby("Classifier", as_index=False)
        .agg(
            original_mean_uar=("Original UAR", "mean"),
            reconstructed_mean_uar=("Reconstructed UAR", "mean"),
            mean_difference=("Difference", "mean"),
            mean_absolute_difference=("Absolute difference", "mean"),
            configuration_count=("Difference", "size"),
        )
        .rename(
            columns={
                "original_mean_uar": "Original mean UAR",
                "reconstructed_mean_uar": "Reconstructed mean UAR",
                "mean_difference": "Difference",
                "mean_absolute_difference": "Mean absolute difference",
                "configuration_count": "Configurations",
            }
        )
        .sort_values("Classifier")
        .reset_index(drop=True)
    )


def create_dataset_summary(dataset_index: Optional[pd.DataFrame]) -> pd.DataFrame:
    if dataset_index is None or dataset_index.empty:
        return pd.DataFrame()

    table = pd.crosstab(
        dataset_index["Language"],
        dataset_index["Class"],
        margins=False,
    ).reset_index()

    for needed in ("PD", "HC"):
        if needed not in table.columns:
            table[needed] = 0

    table["Total"] = table.drop(columns=["Language"]).sum(axis=1)
    total_row = {
        "Language": "Total",
        "PD": table["PD"].sum(),
        "HC": table["HC"].sum(),
        "Total": table["Total"].sum(),
    }
    table = pd.concat([table, pd.DataFrame([total_row])], ignore_index=True)

    return table[["Language", "PD", "HC", "Total"]]


def create_duration_summary(
    duration: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if duration is None or duration.empty:
        return pd.DataFrame()

    rows = [["Number of compared recordings", len(duration)]]

    if "Duration difference" in duration:
        rows.extend(
            [
                [
                    "Mean absolute duration difference (s)",
                    duration["Duration difference"].mean(),
                ],
                [
                    "Median absolute duration difference (s)",
                    duration["Duration difference"].median(),
                ],
                [
                    "Maximum absolute duration difference (s)",
                    duration["Duration difference"].max(),
                ],
            ]
        )

    if "Original duration" in duration:
        rows.append(
            ["Mean original duration (s)", duration["Original duration"].mean()]
        )

    if "Reconstructed duration" in duration:
        rows.append(
            [
                "Mean reconstructed duration (s)",
                duration["Reconstructed duration"].mean(),
            ]
        )

    return pd.DataFrame(rows, columns=["Measure", "Result"])


def create_audio_summary(
    audio: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if audio is None or audio.empty:
        return pd.DataFrame()

    rows = [["Rows in audio validation file", len(audio)]]

    if "Original RMS" in audio:
        rows.append(["Mean original RMS energy", audio["Original RMS"].mean()])

    if "Reconstructed RMS" in audio:
        rows.append(
            [
                "Mean reconstructed RMS energy",
                audio["Reconstructed RMS"].mean(),
            ]
        )

    if "Valid" in audio:
        rows.extend(
            [
                ["Recordings passing validation", int(audio["Valid"].sum())],
                [
                    "Recordings failing validation",
                    int((~audio["Valid"]).sum()),
                ],
            ]
        )

    return pd.DataFrame(rows, columns=["Measure", "Result"])


# =====================================================================
# Graph generation
# =====================================================================

def plot_baseline_best(best: pd.DataFrame, dirs: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    bars = ax.bar(best["Scenario"].astype(str), best["Best UAR"])
    ax.set_title("Best Original-Speech Baseline Result by Scenario")
    ax.set_ylabel("Best UAR")
    lower = max(0, float(best["Best UAR"].min()) - 0.12)
    upper = min(1.0, float(best["Best UAR"].max()) + 0.06)
    ax.set_ylim(lower, upper)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    add_bar_labels(ax, bars)
    save_figure(fig, "01_baseline_best_result_by_scenario", dirs)


def plot_baseline_gap(gaps: pd.DataFrame, dirs: dict[str, Path]) -> None:
    if gaps.empty:
        return

    labels = ["Spanish → German", "German → Spanish"]
    x = np.arange(len(gaps))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    within_bars = ax.bar(
        x - width / 2,
        gaps["Within-language UAR"],
        width,
        label="Within-language",
    )
    cross_bars = ax.bar(
        x + width / 2,
        gaps["Crosslingual UAR"],
        width,
        label="Crosslingual",
    )
    ax.set_xticks(x, labels[: len(gaps)])
    ax.set_title("Within-Language vs. Crosslingual Baseline UAR")
    ax.set_ylabel("UAR")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    add_bar_labels(ax, within_bars)
    add_bar_labels(ax, cross_bars)
    save_figure(fig, "02_baseline_within_vs_crosslingual", dirs)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(labels[: len(gaps)], gaps["Absolute reduction"])
    ax.set_title("Absolute Crosslingual UAR Reduction")
    ax.set_ylabel("UAR reduction")
    ax.grid(axis="y", alpha=0.25)
    add_bar_labels(ax, bars)
    save_figure(fig, "03_baseline_crosslingual_reduction", dirs)


def grouped_original_reconstructed_plot(
    summary: pd.DataFrame,
    category_col: str,
    title: str,
    stem: str,
    dirs: dict[str, Path],
    rotation: int = 0,
) -> None:
    if summary.empty:
        return

    x = np.arange(len(summary))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    original_bars = ax.bar(
        x - width / 2,
        summary["Original mean UAR"],
        width,
        label="Original",
    )
    reconstructed_bars = ax.bar(
        x + width / 2,
        summary["Reconstructed mean UAR"],
        width,
        label="Reconstructed",
    )
    ax.set_xticks(x, summary[category_col].astype(str), rotation=rotation)
    ax.set_title(title)
    ax.set_ylabel("Mean UAR")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    add_bar_labels(ax, original_bars)
    add_bar_labels(ax, reconstructed_bars)
    save_figure(fig, stem, dirs)


def difference_plot(
    summary: pd.DataFrame,
    category_col: str,
    title: str,
    stem: str,
    dirs: dict[str, Path],
    horizontal: bool = False,
) -> None:
    if summary.empty:
        return

    if horizontal:
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        bars = ax.barh(summary[category_col].astype(str), summary["Difference"])
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Reconstructed − original UAR")
        for bar in bars:
            value = float(bar.get_width())
            offset = 0.0008 if value >= 0 else -0.0008
            ax.text(
                value + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.4f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8.2,
            )
        ax.grid(axis="x", alpha=0.25)
    else:
        fig, ax = plt.subplots(figsize=(8.0, 4.7))
        bars = ax.bar(summary[category_col].astype(str), summary["Difference"])
        ax.axhline(0, linewidth=1)
        ax.set_ylabel("Reconstructed − original UAR")
        ax.grid(axis="y", alpha=0.25)
        add_bar_labels(ax, bars, offset=0.0008, signed=True)

    ax.set_title(title)
    save_figure(fig, stem, dirs)


def plot_parity(
    comparison: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(
        comparison["Original UAR"],
        comparison["Reconstructed UAR"],
        alpha=0.7,
        s=35,
    )

    values = pd.concat(
        [comparison["Original UAR"], comparison["Reconstructed UAR"]]
    )
    lower = max(0, float(values.min()) - 0.04)
    upper = min(1, float(values.max()) + 0.04)
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_title("Original vs. Reconstructed UAR Parity")
    ax.set_xlabel("Original UAR")
    ax.set_ylabel("Reconstructed UAR")
    ax.grid(alpha=0.25)
    save_figure(fig, "10_original_reconstructed_parity", dirs)


def plot_difference_distribution(
    comparison: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    ax.hist(comparison["Difference"].dropna(), bins=20)
    ax.axvline(0, linewidth=1)
    ax.axvline(
        comparison["Difference"].mean(),
        linestyle="--",
        linewidth=1.2,
        label=f"Mean = {comparison['Difference'].mean():+.4f}",
    )
    ax.set_title("Distribution of UAR Changes After Reconstruction")
    ax.set_xlabel("Reconstructed − original UAR")
    ax.set_ylabel("Number of configurations")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "11_uar_difference_distribution", dirs)


def plot_absolute_difference_distribution(
    comparison: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    ax.hist(comparison["Absolute difference"].dropna(), bins=20)
    ax.axvline(
        0.05,
        linestyle="--",
        linewidth=1.2,
        label="±0.05 threshold",
    )
    ax.set_title("Distribution of Absolute UAR Differences")
    ax.set_xlabel("Absolute UAR difference")
    ax.set_ylabel("Number of configurations")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "12_absolute_uar_difference_distribution", dirs)


def plot_model_scenario_heatmap(
    comparison: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    pivot = comparison.pivot_table(
        index="Model",
        columns="Scenario",
        values="Difference",
        aggfunc="mean",
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_xticks(
        np.arange(len(pivot.columns)),
        pivot.columns.astype(str),
        rotation=20,
        ha="right",
    )
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index.astype(str))
    ax.set_title("Mean UAR Change by Model and Scenario")

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            if pd.notna(value):
                ax.text(
                    col,
                    row,
                    f"{value:+.4f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Reconstructed − original UAR")
    save_figure(fig, "13_model_scenario_difference_heatmap", dirs)


def plot_dataset_summary(
    dataset_summary: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    if dataset_summary.empty:
        return

    language_df = dataset_summary[
        dataset_summary["Language"].astype(str).str.lower() != "total"
    ].copy()

    x = np.arange(len(language_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    pd_bars = ax.bar(x - width / 2, language_df["PD"], width, label="PD")
    hc_bars = ax.bar(x + width / 2, language_df["HC"], width, label="HC")
    ax.set_xticks(x, language_df["Language"])
    ax.set_title("PD and HC Distribution by Language")
    ax.set_ylabel("Number of recordings")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    add_bar_labels(ax, pd_bars, decimals=0, offset=1)
    add_bar_labels(ax, hc_bars, decimals=0, offset=1)
    save_figure(fig, "14_dataset_pd_hc_distribution", dirs)


def plot_duration_results(
    duration: Optional[pd.DataFrame],
    dirs: dict[str, Path],
) -> None:
    if (
        duration is None
        or duration.empty
        or "Duration difference" not in duration
    ):
        return

    differences_ms = duration["Duration difference"].dropna() * 1000

    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    ax.hist(differences_ms, bins=25)
    ax.set_title("Distribution of Reconstruction Duration Differences")
    ax.set_xlabel("Absolute duration difference (ms)")
    ax.set_ylabel("Number of recordings")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "15_duration_difference_distribution", dirs)

    values = [
        differences_ms.mean(),
        differences_ms.median(),
        differences_ms.max(),
    ]
    labels = ["Mean", "Median", "Maximum"]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(labels, values)
    ax.set_title("Full-Dataset Duration Difference Summary")
    ax.set_ylabel("Absolute duration difference (ms)")
    ax.grid(axis="y", alpha=0.25)
    add_bar_labels(ax, bars, decimals=2, offset=max(values) * 0.02)
    save_figure(fig, "16_duration_difference_summary", dirs)


def plot_audio_results(
    audio: Optional[pd.DataFrame],
    dirs: dict[str, Path],
) -> None:
    if audio is None or audio.empty:
        return

    if "Original RMS" in audio and "Reconstructed RMS" in audio:
        means = [
            audio["Original RMS"].mean(),
            audio["Reconstructed RMS"].mean(),
        ]

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        bars = ax.bar(["Original", "Reconstructed"], means)
        ax.set_title("Mean RMS Energy Before and After Reconstruction")
        ax.set_ylabel("Mean RMS energy")
        ax.grid(axis="y", alpha=0.25)
        add_bar_labels(ax, bars)
        save_figure(fig, "17_rms_energy_comparison", dirs)

        paired = audio.dropna(
            subset=["Original RMS", "Reconstructed RMS"]
        )
        if not paired.empty:
            fig, ax = plt.subplots(figsize=(6.2, 5.8))
            ax.scatter(
                paired["Original RMS"],
                paired["Reconstructed RMS"],
                alpha=0.65,
                s=28,
            )
            values = pd.concat(
                [paired["Original RMS"], paired["Reconstructed RMS"]]
            )
            lower = max(0, float(values.min()))
            upper = float(values.max()) * 1.05
            ax.plot([lower, upper], [lower, upper], linestyle="--")
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.set_title("Original vs. Reconstructed RMS Energy")
            ax.set_xlabel("Original RMS")
            ax.set_ylabel("Reconstructed RMS")
            ax.grid(alpha=0.25)
            save_figure(fig, "18_rms_energy_parity", dirs)

    if "Valid" in audio:
        passed = int(audio["Valid"].sum())
        failed = int((~audio["Valid"]).sum())

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        bars = ax.bar(["Passed", "Failed"], [passed, failed])
        ax.set_title("Generated Audio Validation Results")
        ax.set_ylabel("Number of recordings")
        ax.grid(axis="y", alpha=0.25)
        add_bar_labels(ax, bars, decimals=0, offset=max(1, passed * 0.02))
        save_figure(fig, "19_audio_validation_counts", dirs)


# =====================================================================
# Main execution
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read local baseline and HiFi-GAN result CSV files and generate "
            "supervisor-report tables and graphs."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"C:\pd-speech-crosslingual"),
        help="Root folder of the local thesis repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Separate output directory. Default: "
            "<project-root>/supervisor_report_outputs"
        ),
    )
    parser.add_argument("--baseline-file", type=Path, default=None)
    parser.add_argument(
        "--hifigan-comparison-file",
        type=Path,
        default=None,
    )
    parser.add_argument("--duration-file", type=Path, default=None)
    parser.add_argument("--audio-validation-file", type=Path, default=None)
    parser.add_argument("--dataset-index-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    if not project_root.exists():
        raise FileNotFoundError(
            f"Project root does not exist: {project_root}"
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "supervisor_report_outputs"
    )
    dirs = create_output_dirs(output_dir)

    print("=" * 78)
    print("LOCAL BASELINE + HIFI-GAN REPORT GENERATOR")
    print("=" * 78)
    print(f"Project root: {project_root}")
    print(f"Output dir : {output_dir}")
    print()

    baseline_file = args.baseline_file
    if baseline_file is None:
        baseline_file = autodetect_file(
            project_root,
            BASELINE_FILENAME_CANDIDATES,
            ["baseline", "model_comparison", "classification_results"],
            "baseline results",
        )

    hifigan_file = args.hifigan_comparison_file
    if hifigan_file is None:
        hifigan_file = autodetect_file(
            project_root,
            HIFIGAN_COMPARISON_FILENAME_CANDIDATES,
            [
                "evaluation_comparison",
                "original_vs_generated",
                "original_vs_reconstructed",
                "classification_summary",
            ],
            "HiFi-GAN original-vs-reconstructed results",
        )

    duration_file = args.duration_file
    if duration_file is None:
        duration_file = autodetect_file(
            project_root,
            DURATION_FILENAME_CANDIDATES,
            ["duration", "original_vs_generated"],
            "duration comparison results",
        )

    audio_file = args.audio_validation_file
    if audio_file is None:
        audio_file = autodetect_file(
            project_root,
            AUDIO_VALIDATION_FILENAME_CANDIDATES,
            ["inspection", "audio_validation", "rms"],
            "audio validation results",
        )

    dataset_file = args.dataset_index_file
    if dataset_file is None:
        dataset_file = autodetect_file(
            project_root,
            DATASET_INDEX_FILENAME_CANDIDATES,
            ["dataset_index", "readtext"],
            "dataset index",
        )

    baseline_file = require_path(baseline_file, "Baseline results CSV")
    hifigan_file = require_path(
        hifigan_file,
        "HiFi-GAN comparison CSV",
    )

    # Load local data.
    baseline = load_baseline_results(baseline_file)
    hifigan = load_hifigan_comparison(hifigan_file)
    duration = load_duration_results(duration_file)
    audio = load_audio_validation(audio_file)
    dataset_index = load_dataset_index(dataset_file)

    # Save normalized raw inputs for traceability.
    save_table(
        baseline,
        dirs["tables"] / "00_normalized_baseline_input.csv",
    )
    save_table(
        hifigan,
        dirs["tables"] / "00_normalized_hifigan_comparison_input.csv",
    )
    if duration is not None:
        save_table(
            duration,
            dirs["tables"] / "00_normalized_duration_input.csv",
        )
    if audio is not None:
        save_table(
            audio,
            dirs["tables"] / "00_normalized_audio_validation_input.csv",
        )

    # Calculate summaries from the actual local outputs.
    baseline_best = create_baseline_best_table(baseline)
    baseline_gaps = create_baseline_gap_table(baseline_best)
    overall_summary = create_overall_hifigan_summary(hifigan)
    scenario_summary = create_scenario_summary(hifigan)
    model_summary = create_model_summary(hifigan)
    layer_summary = create_layer_summary(hifigan)
    classifier_summary = create_classifier_summary(hifigan)
    dataset_summary = create_dataset_summary(dataset_index)
    duration_summary = create_duration_summary(duration)
    audio_summary = create_audio_summary(audio)

    # Save calculated tables.
    tables = {
        "01_baseline_best_results.csv": baseline_best,
        "02_baseline_crosslingual_gaps.csv": baseline_gaps,
        "03_hifigan_overall_summary.csv": overall_summary,
        "04_hifigan_scenario_summary.csv": scenario_summary,
        "05_hifigan_model_summary.csv": model_summary,
        "06_hifigan_layer_summary.csv": layer_summary,
        "07_hifigan_classifier_summary.csv": classifier_summary,
        "08_dataset_summary.csv": dataset_summary,
        "09_duration_summary.csv": duration_summary,
        "10_audio_validation_summary.csv": audio_summary,
    }

    for filename, table in tables.items():
        if not table.empty:
            save_table(table, dirs["tables"] / filename)

    # Generate graphs from actual local data.
    plot_baseline_best(baseline_best, dirs)
    plot_baseline_gap(baseline_gaps, dirs)

    grouped_original_reconstructed_plot(
        scenario_summary,
        "Evaluation scenario",
        "Full-Dataset Classification Before and After HiFi-GAN Reconstruction",
        "04_hifigan_scenario_original_vs_reconstructed",
        dirs,
        rotation=18,
    )
    difference_plot(
        scenario_summary,
        "Evaluation scenario",
        "Scenario-Level UAR Change After Reconstruction",
        "05_hifigan_scenario_difference",
        dirs,
        horizontal=True,
    )

    grouped_original_reconstructed_plot(
        model_summary,
        "Model",
        "Classification Comparison by Speech Representation Model",
        "06_hifigan_model_original_vs_reconstructed",
        dirs,
    )
    difference_plot(
        model_summary,
        "Model",
        "Model-Level UAR Change After Reconstruction",
        "07_hifigan_model_difference",
        dirs,
    )

    if not layer_summary.empty:
        grouped_original_reconstructed_plot(
            layer_summary,
            "Layer",
            "Classification Comparison by Representation Layer",
            "08_hifigan_layer_original_vs_reconstructed",
            dirs,
        )
        difference_plot(
            layer_summary,
            "Layer",
            "Layer-Level UAR Change After Reconstruction",
            "09_hifigan_layer_difference",
            dirs,
        )

    plot_parity(hifigan, dirs)
    plot_difference_distribution(hifigan, dirs)
    plot_absolute_difference_distribution(hifigan, dirs)
    plot_model_scenario_heatmap(hifigan, dirs)
    plot_dataset_summary(dataset_summary, dirs)
    plot_duration_results(duration, dirs)
    plot_audio_results(audio, dirs)

    # Save an input-file manifest.
    manifest = pd.DataFrame(
        [
            ["Project root", str(project_root)],
            ["Baseline file", str(baseline_file)],
            ["HiFi-GAN comparison file", str(hifigan_file)],
            ["Duration file", str(duration_file) if duration_file else "Not found"],
            ["Audio validation file", str(audio_file) if audio_file else "Not found"],
            ["Dataset index file", str(dataset_file) if dataset_file else "Not found"],
            ["Output directory", str(output_dir)],
        ],
        columns=["Item", "Path"],
    )
    save_table(manifest, dirs["logs"] / "input_manifest.csv")

    print()
    print("=" * 78)
    print("COMPLETED")
    print("=" * 78)
    print(f"Tables saved to     : {dirs['tables']}")
    print(f"PNG figures saved to: {dirs['figures_png']}")
    print(f"PDF figures saved to: {dirs['figures_pdf']}")
    print(f"Input manifest      : {dirs['logs'] / 'input_manifest.csv'}")
    print()
    print(
        "Important: all summary values were calculated from the CSV files "
        "found in your local project."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[ERROR]", exc, file=sys.stderr)
        sys.exit(1)
