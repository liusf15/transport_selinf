"""Aggregate spline power-comparison replications and draw the power curve.

Run this after the batch runs have completed. By default the command
expects seeds 0--499 for each signal in the planned experiment and refuses to
produce a final summary if any replication is missing or failed:

    python -m experiments.spline.summarize_power_comparison \
      --results_dir experiments/results/spline_power/<configuration>

Use ``--allow_incomplete`` only for interim diagnostics. Such summaries retain
the actual successful-replication counts so they cannot be mistaken for the
complete 500-replication result.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SIGNALS = (0.0, 0.25, 0.5, 0.75, 1.0)
# Map internal method keys to plot labels and source p-value columns.
METHODS = {
    "transport": (
        r"$\left\|\hat{\tau}^{-1}(T)\right\|_2$",
        "transport_pvalue",
    ),
    "whitened_l2": (r"$\|\Sigma_M^{-1} T\|_2$", "whitened_l2_pvalue"),
    "empirical_l2": (r"$\|T\|_2$", "empirical_l2_pvalue"),
}
METHOD_STYLES = {
    "transport": {"linestyle": "-", "marker": "o"},
    "whitened_l2": {"linestyle": "--", "marker": "s"},
    "empirical_l2": {"linestyle": ":", "marker": "^"},
}
REQUIRED_COLUMNS = {
    "configuration_id",
    "seed",
    "signal_fac",
    "alpha",
    "status",
    "transport_pvalue",
    "whitened_l2_pvalue",
    "empirical_l2_pvalue",
}


def wilson_interval(successes, total, confidence_z=1.959963984540054):
    """Return a two-sided 95% Wilson interval for a binomial proportion."""
    # successes/total define the estimate; confidence_z selects confidence level.
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1.0 + confidence_z ** 2 / total
    center = (proportion + confidence_z ** 2 / (2.0 * total)) / denominator
    radius = (
        confidence_z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + confidence_z ** 2 / (4.0 * total ** 2)
        )
        / denominator
    )
    return center - radius, center + radius


def load_results(results_dir):
    """Load combined batch files or legacy per-seed files."""
    # results_dir may be a configuration root or one signal-specific directory.
    results_dir = Path(results_dir)
    # Batch outputs live immediately inside signal directories. Avoid loading
    # derived files such as selected_knots_5/results.csv as raw input.
    paths = sorted(results_dir.glob("signal_*/results.csv"))
    if results_dir.name.startswith("signal_") and (results_dir / "results.csv").is_file():
        paths.append(results_dir / "results.csv")
    combined_signal_directories = {path.parent.resolve() for path in paths}
    paths.extend(
        path
        for path in sorted(results_dir.rglob("seed_*.csv"))
        if path.parent.resolve() not in combined_signal_directories
    )
    if not paths:
        raise ValueError(
            f"no results.csv or seed_*.csv files found below {results_dir}"
        )

    # Enforce a common schema before concatenating source files.
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing_columns)}"
            )
        if path.name.startswith("seed_") and len(frame) != 1:
            raise ValueError(f"legacy per-seed file {path} must contain one row")
        if frame.empty:
            raise ValueError(f"{path} contains no result rows")
        frame["source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_results(
    results,
    signals=DEFAULT_SIGNALS,
    expected_replications=500,
    allow_incomplete=False,
):
    """Validate configuration identity, seed coverage, status, and p-values."""
    # signals and expected_replications define the required grid; allow_incomplete
    # permits interim summaries while preserving explicit coverage diagnostics.
    results = results.copy()
    results["seed"] = pd.to_numeric(results["seed"], errors="raise").astype(int)
    results["signal_fac"] = pd.to_numeric(
        results["signal_fac"], errors="raise"
    ).astype(float)
    signals = tuple(float(signal) for signal in signals)

    configuration_ids = results["configuration_id"].dropna().unique()
    if len(configuration_ids) != 1:
        raise ValueError(
            "results must contain exactly one configuration_id; found "
            f"{configuration_ids.tolist()}"
        )

    duplicate_mask = results.duplicated(["signal_fac", "seed"], keep=False)
    if duplicate_mask.any():
        duplicates = results.loc[duplicate_mask, ["signal_fac", "seed"]]
        raise ValueError(
            "duplicate (signal_fac, seed) rows found: "
            f"{duplicates.drop_duplicates().to_dict(orient='records')}"
        )

    unexpected_signals = sorted(set(results["signal_fac"]).difference(signals))
    if unexpected_signals:
        raise ValueError(f"unexpected signal levels found: {unexpected_signals}")

    # Check each signal independently for missing, failed, or out-of-range seeds.
    expected_seeds = set(range(expected_replications))
    diagnostics = []
    has_incomplete_signal = False
    for signal in signals:
        signal_rows = results[np.isclose(results["signal_fac"], signal)]
        observed_seeds = set(signal_rows["seed"])
        extra_seeds = sorted(observed_seeds.difference(expected_seeds))
        if extra_seeds:
            raise ValueError(
                f"signal {signal:g} contains seeds outside 0--"
                f"{expected_replications - 1}: {extra_seeds}"
            )

        missing_seeds = sorted(expected_seeds.difference(observed_seeds))
        failed_seeds = sorted(
            signal_rows.loc[signal_rows["status"] != "success", "seed"].tolist()
        )
        successful_rows = signal_rows[signal_rows["status"] == "success"]
        diagnostics.append(
            {
                "signal_fac": signal,
                "found": len(signal_rows),
                "successful": len(successful_rows),
                "missing_seeds": missing_seeds,
                "failed_seeds": failed_seeds,
            }
        )
        if missing_seeds or failed_seeds:
            has_incomplete_signal = True

    # Only successful rows contribute, but every contributing p-value must be valid.
    successful = results[results["status"] == "success"].copy()
    for _, (_, pvalue_column) in METHODS.items():
        pvalues = pd.to_numeric(successful[pvalue_column], errors="coerce")
        invalid = pvalues.isna() | (pvalues < 0.0) | (pvalues > 1.0)
        if invalid.any():
            bad_rows = successful.loc[invalid, ["signal_fac", "seed"]]
            raise ValueError(
                f"successful rows contain invalid {pvalue_column}: "
                f"{bad_rows.to_dict(orient='records')}"
            )
        successful[pvalue_column] = pvalues

    if has_incomplete_signal and not allow_incomplete:
        messages = []
        for diagnostic in diagnostics:
            if diagnostic["missing_seeds"] or diagnostic["failed_seeds"]:
                messages.append(
                    f"signal {diagnostic['signal_fac']:g}: "
                    f"{len(diagnostic['missing_seeds'])} missing, "
                    f"{len(diagnostic['failed_seeds'])} failed"
                )
        raise ValueError(
            "incomplete experiment; " + "; ".join(messages)
            + ". Use --allow_incomplete only for an interim summary."
        )

    return successful, diagnostics


def summarize_results(successful_results, signals=DEFAULT_SIGNALS):
    """Calculate paired rejection rates and Wilson intervals by signal."""
    # successful_results contains validated p-values; signals fixes output ordering.
    rows = []
    for signal in (float(value) for value in signals):
        signal_rows = successful_results[
            np.isclose(successful_results["signal_fac"], signal)
        ]
        if signal_rows.empty:
            alpha = np.nan
        else:
            alphas = signal_rows["alpha"].astype(float).unique()
            if len(alphas) != 1:
                raise ValueError(f"signal {signal:g} contains multiple alpha values")
            alpha = alphas[0]

        # Summarize each method on the same replication subset for paired comparison.
        for method, (method_label, pvalue_column) in METHODS.items():
            total = len(signal_rows)
            rejections = int((signal_rows[pvalue_column] <= alpha).sum()) if total else 0
            power = rejections / total if total else np.nan
            standard_error = (
                np.sqrt(power * (1.0 - power) / total) if total else np.nan
            )
            ci_low, ci_high = wilson_interval(rejections, total)
            rows.append(
                {
                    "signal_fac": signal,
                    "estimand": "type_i_error" if signal == 0.0 else "power",
                    "method": method,
                    "method_label": method_label,
                    "alpha": alpha,
                    "num_successful": total,
                    "num_rejections": rejections,
                    "rejection_rate": power,
                    "standard_error": standard_error,
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def plot_power_curve(summary, output_stem):
    """Save a PDF power curve with two-binomial-standard-error bars."""
    # summary is the method-by-signal table; output_stem omits the file extension.
    # Import plotting only when figures are requested so validation helpers can
    # still be used in lightweight result-checking environments.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_stem = Path(output_stem)
    with sns.plotting_context("paper", font_scale=1.5), sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for method, (method_label, _) in METHODS.items():
            method_rows = summary[summary["method"] == method].sort_values(
                "signal_fac"
            )
            valid_rows = method_rows.dropna(subset=["rejection_rate"])
            ax.errorbar(
                valid_rows["signal_fac"],
                valid_rows["rejection_rate"],
                yerr=2.0 * valid_rows["standard_error"],
                capsize=3,
                linewidth=1.8,
                label=method_label,
                **METHOD_STYLES[method],
            )

        # Mark the nominal type-I error level when it is common across rows.
        finite_alpha = summary["alpha"].dropna().unique()
        if len(finite_alpha) == 1:
            ax.axhline(
                finite_alpha[0],
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label="_nolegend_",
            )
        signals = sorted(summary["signal_fac"].unique())
        tick_labels = [
            "0\n(type-I error)" if value == 0 else f"{value:g}"
            for value in signals
        ]
        ax.set_xticks(signals, tick_labels)
        ax.set_xlabel("Signal factor")
        ax.set_ylabel("Rejection rate")
        ax.set_title("Comparison of power")
        ax.set_ylim(0.0, 1.0)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_stem.with_suffix(".pdf"))
        plt.close(fig)


def _print_diagnostics(diagnostics):
    # Print compact seed-coverage previews without flooding output for large gaps.
    for diagnostic in diagnostics:
        message = (
            f"signal {diagnostic['signal_fac']:g}: "
            f"{diagnostic['successful']} successful / {diagnostic['found']} found"
        )
        if diagnostic["missing_seeds"]:
            preview = diagnostic["missing_seeds"][:20]
            message += f", missing seeds {preview}"
            if len(diagnostic["missing_seeds"]) > len(preview):
                message += " ..."
        if diagnostic["failed_seeds"]:
            preview = diagnostic["failed_seeds"][:20]
            message += f", failed seeds {preview}"
            if len(diagnostic["failed_seeds"]) > len(preview):
                message += " ..."
        print(message)


def _build_parser():
    # Configure expected simulation grid, incomplete-run policy, and output basename.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", "--results-dir", required=True)
    parser.add_argument(
        "--signals",
        nargs="+",
        type=float,
        default=list(DEFAULT_SIGNALS),
    )
    parser.add_argument(
        "--expected_replications",
        "--expected-replications",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--allow_incomplete",
        "--allow-incomplete",
        action="store_true",
    )
    parser.add_argument(
        "--output_stem",
        "--output-stem",
        default="power_comparison",
        help="output basename, relative to results_dir unless absolute",
    )
    return parser


def main(argv=None):
    # Validate raw records before creating any final table or figure.
    args = _build_parser().parse_args(argv)
    if args.expected_replications <= 0:
        raise ValueError("--expected_replications must be positive")

    results = load_results(args.results_dir)
    successful, diagnostics = validate_results(
        results,
        signals=args.signals,
        expected_replications=args.expected_replications,
        allow_incomplete=args.allow_incomplete,
    )
    _print_diagnostics(diagnostics)
    summary = summarize_results(successful, signals=args.signals)

    # Resolve relative outputs below the result directory and save both deliverables.
    output_stem = Path(args.output_stem)
    if not output_stem.is_absolute():
        output_stem = Path(args.results_dir) / output_stem
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_stem.with_suffix(".csv")
    summary.to_csv(summary_path, index=False)
    plot_power_curve(summary, output_stem)
    print(f"Saved summary to {summary_path}")
    print(f"Saved figure to {output_stem.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
