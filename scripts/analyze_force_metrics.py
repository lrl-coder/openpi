"""Analyze force-guided pi0 training metrics exported by scripts/train.py.

The script reads metrics.csv and writes:
  - report.md: human-readable summary and diagnostics
  - key_metrics_summary.csv: first/final/min/max/delta for important metrics
  - axis_variance_summary.csv: per-axis force variance calibration at the final row
  - plots/*.png: loss, calibration, variance-gap, and log-sigma trend plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = Path(
    "/root/autodl-fs/openpi_checkpoints/"
    "pi0_flexiv_pump_1bottle_inputForce_lora_force_guided/"
    "flexiv_pump_lora_force_guided/metrics.csv"
)

KEY_METRICS = [
    "loss",
    "diagnostic_loss",
    "loss_fm",
    "force_nll",
    "force_target_nll",
    "loss_force_weighted",
    "loss_force_target_weighted",
    "loss_force_physical_anchor",
    "loss_force_physical_anchor_weighted",
    "loss_force_distill",
    "loss_force_distill_weighted",
    "loss_force_semantic_align",
    "loss_force_semantic_align_weighted",
    "force_distill_cosine_mean",
    "force_semantic_cosine_mean",
    "force_physical_summary_mse",
    "force_physical_summary_mae",
    "force_physical_summary_target_std_mean",
    "force_physical_summary_pred_std_mean",
    "grad_norm",
    "force_pred_sigma_mean",
    "force_pred_sigma_to_true_std_ratio_mean",
    "force_pred_var_abs_gap_mean",
    "force_pred_var_abs_gap_within_horizon_mean",
    "force_pred_var_to_residual_mse_ratio_mean",
    "force_residual_rmse_to_pred_sigma_ratio_mean",
    "force_target_pred_sigma_mean",
    "force_target_pred_sigma_to_true_std_ratio_mean",
    "force_target_pred_var_abs_gap_mean",
    "force_target_pred_var_abs_gap_within_horizon_mean",
    "force_target_pred_var_to_residual_mse_ratio_mean",
    "force_target_residual_rmse_to_pred_sigma_ratio_mean",
]

PLOT_GROUPS = {
    "training_losses": [
        "loss",
        "diagnostic_loss",
        "loss_fm",
        "loss_force_weighted",
        "loss_force_target_weighted",
        "loss_force_physical_anchor_weighted",
        "loss_force_distill_weighted",
        "loss_force_semantic_align_weighted",
    ],
    "contact_dynamics_token": [
        "loss_force_physical_anchor",
        "loss_force_distill",
        "force_distill_cosine_mean",
        "force_physical_summary_mse",
        "force_physical_summary_mae",
        "force_physical_summary_target_std_mean",
        "force_physical_summary_pred_std_mean",
    ],
    "force_nll": [
        "force_nll",
        "force_target_nll",
        "force_nll_negative_frac",
        "force_target_nll_negative_frac",
    ],
    "force_predictor_calibration": [
        "force_pred_sigma_to_true_std_ratio_mean",
        "force_pred_var_to_residual_mse_ratio_mean",
        "force_residual_rmse_to_pred_sigma_ratio_mean",
    ],
    "force_target_calibration": [
        "force_target_pred_sigma_to_true_std_ratio_mean",
        "force_target_pred_var_to_residual_mse_ratio_mean",
        "force_target_residual_rmse_to_pred_sigma_ratio_mean",
    ],
    "variance_abs_gap": [
        "force_pred_var_abs_gap_mean",
        "force_pred_var_abs_gap_within_horizon_mean",
        "force_target_pred_var_abs_gap_mean",
        "force_target_pred_var_abs_gap_within_horizon_mean",
    ],
    "log_sigma_ranges": [
        "force_pred_log_sigma_min",
        "force_pred_log_sigma_mean",
        "force_pred_log_sigma_max",
        "force_target_pred_log_sigma_min",
        "force_target_pred_log_sigma_mean",
        "force_target_pred_log_sigma_max",
    ],
    "clip_fractions": [
        "force_pred_log_sigma_min_clip_frac",
        "force_pred_log_sigma_max_clip_frac",
        "force_target_pred_log_sigma_min_clip_frac",
        "force_target_pred_log_sigma_max_clip_frac",
    ],
    "true_variance": [
        "force_true_var_mean",
        "force_true_var_within_horizon_mean",
        "force_target_true_var_mean",
        "force_target_true_var_within_horizon_mean",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_METRICS, help="Path to metrics.csv.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <metrics.csv parent>/metrics_analysis.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Rolling mean window for plots. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--last-k",
        type=int,
        default=5,
        help="Number of final logged rows used for the stable-final summary.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"metrics CSV not found: {path}")
    df = pd.read_csv(path)
    if "step" not in df.columns:
        raise ValueError(f"{path} does not contain a 'step' column")
    df = df.sort_values("step").reset_index(drop=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def existing(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def final_window(df: pd.DataFrame, last_k: int) -> pd.DataFrame:
    return df.tail(max(1, min(last_k, len(df))))


def make_metric_summary(df: pd.DataFrame, last_k: int) -> pd.DataFrame:
    rows = []
    stable = final_window(df, last_k)
    for col in existing(df, KEY_METRICS):
        series = df[col].dropna()
        if series.empty:
            continue
        first = float(series.iloc[0])
        final = float(series.iloc[-1])
        row = {
            "metric": col,
            "first": first,
            "final": final,
            "delta": final - first,
            "min": float(series.min()),
            "min_step": int(df.loc[series.idxmin(), "step"]),
            "max": float(series.max()),
            "max_step": int(df.loc[series.idxmax(), "step"]),
            f"last_{len(stable)}_mean": float(stable[col].mean()) if col in stable else np.nan,
            f"last_{len(stable)}_std": float(stable[col].std(ddof=0)) if col in stable else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def axis_variance_summary(df: pd.DataFrame) -> pd.DataFrame:
    final = df.iloc[-1]
    rows = []
    for prefix in ["force", "force_target"]:
        pred_prefix = "force_pred" if prefix == "force" else "force_target_pred"
        true_prefix = "force_true" if prefix == "force" else "force_target_true"
        for axis in range(6):
            row = {"head": prefix, "axis": axis}
            for name, col in [
                ("pred_var", f"{pred_prefix}_var_axis_{axis}"),
                ("true_var", f"{true_prefix}_var_axis_{axis}"),
                ("var_gap", f"{pred_prefix}_var_minus_true_var_axis_{axis}"),
                ("true_var_within_horizon", f"{true_prefix}_var_within_horizon_axis_{axis}"),
                ("var_gap_within_horizon", f"{pred_prefix}_var_minus_true_var_within_horizon_axis_{axis}"),
                ("residual_mse", f"{prefix}_residual_mse_axis_{axis}"),
            ]:
                row[name] = float(final[col]) if col in df.columns and pd.notna(final[col]) else np.nan
            if np.isfinite(row["pred_var"]) and np.isfinite(row["true_var"]) and row["true_var"] > 0:
                row["pred_var_to_true_var"] = row["pred_var"] / row["true_var"]
            else:
                row["pred_var_to_true_var"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_series(df: pd.DataFrame, columns: list[str], path: Path, title: str, smooth_window: int) -> bool:
    cols = existing(df, columns)
    if not cols:
        return False
    plt.figure(figsize=(10, 5.8))
    x = df["step"]
    for col in cols:
        y = df[col]
        if smooth_window > 1 and len(y) >= smooth_window:
            y = y.rolling(smooth_window, min_periods=1).mean()
        plt.plot(x, y, marker="o", linewidth=1.8, markersize=3, label=col)
    plt.title(title)
    plt.xlabel("step")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def plot_axis_bars(axis_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for head in ["force", "force_target"]:
        sub = axis_df[axis_df["head"] == head]
        if sub.empty:
            continue
        x = np.arange(len(sub))
        width = 0.36
        path = out_dir / f"{head}_axis_variance_final.png"
        plt.figure(figsize=(9.5, 5.6))
        plt.bar(x - width / 2, sub["pred_var"], width, label="pred_var")
        plt.bar(x + width / 2, sub["true_var"], width, label="true_var")
        plt.xticks(x, [f"axis_{int(a)}" for a in sub["axis"]])
        plt.ylabel("variance")
        plt.title(f"{head}: final predicted variance vs true variance")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)
    return paths


def describe_metric(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return "N/A"
    series = df[col].dropna()
    if series.empty:
        return "N/A"
    return f"{series.iloc[-1]:.6g} (first {series.iloc[0]:.6g}, min {series.min():.6g}, max {series.max():.6g})"


def markdown_table(df: pd.DataFrame, *, floatfmt: str = ".5g") -> str:
    """Formats a small DataFrame as a markdown table without optional dependencies."""
    if df.empty:
        return ""
    columns = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(format(float(value), floatfmt))
            elif isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            else:
                values.append(str(value))
        rows.append(values)

    widths = [
        max(len(columns[i]), *(len(row[i]) for row in rows))
        for i in range(len(columns))
    ]
    header = "| " + " | ".join(columns[i].ljust(widths[i]) for i in range(len(columns))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def add_warning(lines: list[str], condition: bool, message: str) -> None:
    if condition:
        lines.append(f"- WARNING: {message}")


def build_report(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    axis_df: pd.DataFrame,
    plot_paths: list[Path],
    csv_path: Path,
    out_dir: Path,
    last_k: int,
) -> str:
    final = df.iloc[-1]
    stable = final_window(df, last_k)
    lines = []
    lines.append("# Force Metrics Analysis")
    lines.append("")
    lines.append(f"- Source CSV: `{csv_path}`")
    lines.append(f"- Output directory: `{out_dir}`")
    lines.append(f"- Logged rows: {len(df)}")
    lines.append(f"- Step range: {int(df['step'].iloc[0])} -> {int(df['step'].iloc[-1])}")
    lines.append(f"- Columns: {len(df.columns)}")
    lines.append("")

    lines.append("## Key Metrics")
    for col in existing(df, KEY_METRICS[:12]):
        lines.append(f"- `{col}`: {describe_metric(df, col)}")
    lines.append("")

    if existing(df, ["loss", "diagnostic_loss", "loss_fm"]):
        lines.append("## Training Trend")
        for col in existing(df, ["loss", "diagnostic_loss", "loss_fm", "grad_norm"]):
            series = df[col].dropna()
            if series.empty:
                continue
            pct = 100.0 * (series.iloc[-1] - series.iloc[0]) / (abs(series.iloc[0]) + 1e-12)
            lines.append(f"- `{col}` changed by {pct:.2f}% from first to final logged row.")
        lines.append("")

    lines.append("## Force Predictor Calibration")
    for col in existing(
        df,
        [
            "force_nll",
            "force_nll_negative_frac",
            "force_pred_sigma_to_true_std_ratio_mean",
            "force_pred_var_abs_gap_mean",
            "force_pred_var_abs_gap_within_horizon_mean",
            "force_pred_var_to_residual_mse_ratio_mean",
            "force_residual_rmse_to_pred_sigma_ratio_mean",
            "force_pred_log_sigma_min_clip_frac",
            "force_pred_log_sigma_max_clip_frac",
        ],
    ):
        lines.append(f"- `{col}`: {describe_metric(df, col)}")
    lines.append("")

    lines.append("## VLM Force Target Calibration")
    for col in existing(
        df,
        [
            "force_target_nll",
            "force_target_nll_negative_frac",
            "force_target_pred_sigma_to_true_std_ratio_mean",
            "force_target_pred_var_abs_gap_mean",
            "force_target_pred_var_abs_gap_within_horizon_mean",
            "force_target_pred_var_to_residual_mse_ratio_mean",
            "force_target_residual_rmse_to_pred_sigma_ratio_mean",
            "force_target_pred_log_sigma_min_clip_frac",
            "force_target_pred_log_sigma_max_clip_frac",
        ],
    ):
        lines.append(f"- `{col}`: {describe_metric(df, col)}")
    lines.append("")

    warnings = []
    add_warning(
        warnings,
        "force_pred_log_sigma_min_clip_frac" in df.columns
        and float(final["force_pred_log_sigma_min_clip_frac"]) > 0.05,
        "`F_phi` log_sigma is often clipped at the lower bound; predicted force variance may be collapsing.",
    )
    add_warning(
        warnings,
        "force_target_pred_log_sigma_min_clip_frac" in df.columns
        and float(final["force_target_pred_log_sigma_min_clip_frac"]) > 0.05,
        "VLM target log_sigma is often clipped at the lower bound; target variance may be collapsing.",
    )
    add_warning(
        warnings,
        "force_pred_sigma_to_true_std_ratio_mean" in df.columns
        and float(final["force_pred_sigma_to_true_std_ratio_mean"]) < 0.3,
        "`F_phi` predicted std is less than 30% of the batch true std.",
    )
    add_warning(
        warnings,
        "force_target_pred_sigma_to_true_std_ratio_mean" in df.columns
        and float(final["force_target_pred_sigma_to_true_std_ratio_mean"]) < 0.3,
        "VLM target predicted std is less than 30% of the true target std.",
    )
    add_warning(
        warnings,
        "force_residual_rmse_to_pred_sigma_ratio_mean" in df.columns
        and float(final["force_residual_rmse_to_pred_sigma_ratio_mean"]) > 2.0,
        "`F_phi` residual RMSE is more than 2x predicted sigma; uncertainty may be under-estimated.",
    )
    add_warning(
        warnings,
        "force_target_residual_rmse_to_pred_sigma_ratio_mean" in df.columns
        and float(final["force_target_residual_rmse_to_pred_sigma_ratio_mean"]) > 2.0,
        "VLM target residual RMSE is more than 2x predicted sigma; target uncertainty may be under-estimated.",
    )
    add_warning(
        warnings,
        "grad_norm" in df.columns and float(df["grad_norm"].max()) > 10 * max(float(stable["grad_norm"].median()), 1e-12),
        "Large early/isolated grad_norm spike detected. Check whether this is just step-0 warmup or persistent instability.",
    )
    nan_cols = [col for col in df.columns if df[col].isna().any()]
    add_warning(warnings, bool(nan_cols), f"NaNs found in columns: {', '.join(nan_cols[:20])}")

    lines.append("## Automatic Checks")
    if warnings:
        lines.extend(warnings)
    else:
        lines.append("- No obvious variance-collapse or numeric warning from the configured checks.")
    lines.append("")

    if not axis_df.empty:
        lines.append("## Final Per-Axis Variance Calibration")
        show = axis_df[
            [
                "head",
                "axis",
                "pred_var",
                "true_var",
                "pred_var_to_true_var",
                "var_gap",
                "residual_mse",
            ]
        ].copy()
        lines.append(markdown_table(show, floatfmt=".4g"))
        lines.append("")

    lines.append("## Generated Files")
    lines.append("- `key_metrics_summary.csv`")
    lines.append("- `axis_variance_summary.csv`")
    for path in plot_paths:
        lines.append(f"- `{path.relative_to(out_dir)}`")
    lines.append("")

    if not summary_df.empty:
        lines.append("## Compact Summary Table")
        compact = summary_df[["metric", "first", "final", "delta", "min", "min_step", "max", "max_step"]].head(24)
        lines.append(markdown_table(compact, floatfmt=".5g"))
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    csv_path = args.csv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else csv_path.parent / "metrics_analysis"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(csv_path)
    summary_df = make_metric_summary(df, args.last_k)
    axis_df = axis_variance_summary(df)

    summary_df.to_csv(out_dir / "key_metrics_summary.csv", index=False)
    axis_df.to_csv(out_dir / "axis_variance_summary.csv", index=False)

    plot_paths = []
    for name, columns in PLOT_GROUPS.items():
        path = plot_dir / f"{name}.png"
        if plot_series(df, columns, path, name.replace("_", " ").title(), args.smooth_window):
            plot_paths.append(path)
    plot_paths.extend(plot_axis_bars(axis_df, plot_dir))

    report = build_report(df, summary_df, axis_df, plot_paths, csv_path, out_dir, args.last_k)
    report_path = out_dir / "report.md"
    report_path.write_text(report)

    print(f"Loaded {len(df)} rows x {len(df.columns)} columns from {csv_path}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote summaries: {out_dir / 'key_metrics_summary.csv'}, {out_dir / 'axis_variance_summary.csv'}")
    print(f"Wrote {len(plot_paths)} plots under: {plot_dir}")


if __name__ == "__main__":
    main()
