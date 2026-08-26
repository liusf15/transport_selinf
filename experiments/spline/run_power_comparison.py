"""Evaluate spline power replications using pre-calibrated transport maps.

First create the four knot-conditioned maps with
``experiments.spline.train_power_transports``. Evaluate a complete seed range
in one process and save it as one CSV (the upper endpoint is exclusive):

    python -m experiments.spline.run_power_comparison \
      --map_dir experiments/results/spline_power_maps/<configuration> \
      --signal_fac 1 --seeds 0 500

The legacy ``--seed`` option remains available for scheduler-based runs.
"""

import argparse
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer, StandardScaler

from experiments.spline.power_transport import (
    float_slug,
    load_transport_artifact,
    transport_pvalue,
)
from experiments.spline.spline_selector import SplineSelection


COEFFICIENT_DIRECTION = np.array([1.0, -1.0, 1.0, 1.0])


@lru_cache(maxsize=None)
def _fixed_design(n, design_seed):
    # n sets the sample size; design_seed makes covariates common to all replicates.
    design_rng = np.random.default_rng(design_seed)
    x = design_rng.uniform(size=(n, 1))
    design_matrix = SplineTransformer(
        n_knots=3,
        include_bias=False,
    ).fit_transform(x)
    return x, StandardScaler().fit_transform(design_matrix)


def fixed_design(args):
    """Return the reproducible covariate and standardized generating basis."""
    # args contributes n and design_seed from the experiment configuration.
    return _fixed_design(args.n, args.design_seed)


def coefficient_direction(args):
    """Return a direction normalized to the reference fixed-design signal SD."""
    # args may override four coefficients; normalization isolates direction effects.
    direction = np.asarray(
        getattr(args, "coefficient_direction", COEFFICIENT_DIRECTION),
        dtype=float,
    )
    if direction.shape != COEFFICIENT_DIRECTION.shape:
        raise ValueError("--coefficient-direction requires exactly four values")
    if not np.all(np.isfinite(direction)) or np.all(direction == 0.0):
        raise ValueError("--coefficient-direction must be finite and nonzero")

    _, design_matrix = fixed_design(args)
    reference_signal_sd = np.std(design_matrix @ COEFFICIENT_DIRECTION)
    direction_signal_sd = np.std(design_matrix @ direction)
    if not np.isfinite(direction_signal_sd) or direction_signal_sd <= 0.0:
        raise ValueError("--coefficient-direction generates no varying signal")
    return direction * reference_signal_sd / direction_signal_sd


def direction_slug(direction):
    """Format an alternative direction for deterministic result namespacing."""
    # direction is the normalized four-coefficient signal vector.
    return "_".join(float_slug(value) for value in direction)


def empirical_l2_pvalue(observed, null_samples):
    """Return the finite-Monte-Carlo conditional p-value based on L2 norm.

    The added one in both numerator and denominator prevents a zero p-value and
    gives the usual valid randomization-test correction. Ties are counted in
    the upper tail, making the test conservative for discrete samples.
    """
    # observed is one selected statistic; null_samples are draws under its event.
    observed = np.asarray(observed, dtype=float)
    null_samples = np.asarray(null_samples, dtype=float)
    if null_samples.ndim != 2 or null_samples.shape[0] == 0:
        raise ValueError("null_samples must be a non-empty two-dimensional array")
    if observed.ndim != 1 or null_samples.shape[1] != observed.shape[0]:
        raise ValueError("observed and null_samples must have matching dimensions")

    observed_norm = np.linalg.norm(observed)
    null_norms = np.linalg.norm(null_samples, axis=1)
    exceedances = np.count_nonzero(null_norms >= observed_norm)
    return (1.0 + exceedances) / (null_samples.shape[0] + 1.0)


def empirical_whitened_l2_pvalue(observed, null_samples, covariance):
    """Return an empirical p-value based on the selected-model Mahalanobis norm."""
    # covariance defines the linear whitening applied to observed and null draws.
    observed = np.asarray(observed, dtype=float)
    null_samples = np.asarray(null_samples, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if null_samples.ndim != 2 or null_samples.shape[0] == 0:
        raise ValueError("null_samples must be a non-empty two-dimensional array")
    dimension = observed.shape[0]
    if observed.ndim != 1 or null_samples.shape[1] != dimension:
        raise ValueError("observed and null_samples must have matching dimensions")
    if covariance.shape != (dimension, dimension):
        raise ValueError("covariance must match the statistic dimension")

    precision = np.linalg.inv(covariance)
    observed_norm_sq = observed @ precision @ observed
    null_norms_sq = np.einsum(
        "bi,ij,bj->b",
        null_samples,
        precision,
        null_samples,
    )
    exceedances = np.count_nonzero(null_norms_sq >= observed_norm_sq)
    return (1.0 + exceedances) / (null_samples.shape[0] + 1.0)


def configuration_id(args):
    """Identify the saved-map collection and the testing level."""
    # Include every option that changes inference results in the namespace.
    map_configuration = Path(args.map_dir).resolve().name
    return (
        f"maps_{map_configuration}_methods_transport_raw_whitened"
        f"_direction_{direction_slug(coefficient_direction(args))}"
        f"_alpha_{float_slug(args.alpha)}"
    )


def result_path(args):
    """Return the deterministic path for one signal/seed result."""
    return (
        Path(args.output_root)
        / "spline_power_pretrained"
        / configuration_id(args)
        / f"signal_{float_slug(args.signal_fac)}"
        / f"seed_{args.seed}.csv"
    )


def batch_result_path(args):
    """Return the deterministic combined-result path for one signal."""
    return (
        Path(args.output_root)
        / "spline_power_pretrained"
        / configuration_id(args)
        / f"signal_{float_slug(args.signal_fac)}"
        / "results.csv"
    )


def _generate_data(args):
    """Generate the fixed spline design and one response replication."""
    # args supplies signal size/direction, replication seed, and randomization variance.
    x, design_matrix = fixed_design(args)

    beta = coefficient_direction(args) * args.signal_fac
    mu = design_matrix @ beta
    data_rng = np.random.default_rng(args.seed)
    y = mu + data_rng.normal(size=args.n)
    y_perturb = data_rng.normal(size=args.n) * np.sqrt(args.nu_sq)
    snr = np.sqrt(np.var(mu))
    return x, y, y_perturb, snr


def _base_result(args):
    """Create a complete result record whose diagnostics can be filled later."""
    # Prepopulate failure-safe fields so every attempted replication has one schema.
    return {
        "configuration_id": configuration_id(args),
        "seed": args.seed,
        "signal_fac": args.signal_fac,
        "coefficient_direction": ",".join(
            format(value, ".12g") for value in coefficient_direction(args)
        ),
        "alpha": args.alpha,
        "n": args.n,
        "design_seed": args.design_seed,
        "n_train": np.nan,
        "n_val": np.nan,
        "n_fold": args.n_fold,
        "max_knots": args.max_knots,
        "nu_sq": args.nu_sq,
        "map_dir": str(Path(args.map_dir).resolve()),
        "snr": np.nan,
        "selected_n_knots": np.nan,
        "dimension": np.nan,
        "num_conditional_samples": 0,
        "mean_num_tries": np.nan,
        "max_num_tries": np.nan,
        "empirical_l2_pvalue": np.nan,
        "whitened_l2_pvalue": np.nan,
        "transport_pvalue": np.nan,
        "empirical_l2_reject": np.nan,
        "whitened_l2_reject": np.nan,
        "transport_reject": np.nan,
        "flow_seed": np.nan,
        "flow_learning_rate": np.nan,
        "flow_initial_val_loss": np.nan,
        "flow_final_val_loss": np.nan,
        "flow_best_val_loss": np.nan,
        "map_configuration_id": "",
        "status": "started",
        "error_type": "",
        "error_message": "",
    }


def _validate_artifact(args, metadata, selected_dimension):
    """Prevent evaluation with a map calibrated for a different experiment."""
    # metadata describes the saved calibration; selected_dimension comes from this fit.
    expected = {
        "n": args.n,
        "design_seed": args.design_seed,
        "n_fold": args.n_fold,
        "max_knots": args.max_knots,
        "nu_sq": args.nu_sq,
        "dimension": selected_dimension,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"saved map is incompatible with this run: {mismatches}")


def run_experiment(args):
    """Run one complete simulation replication and return its result record."""
    # args identifies the map, design, signal, seed, and rejection threshold.
    result = _base_result(args)
    try:
        x, y, y_perturb, snr = _generate_data(args)
        selector = SplineSelection(
            x,
            y,
            sigma=1.0,
            maximum_knots=args.max_knots,
            n_fold=args.n_fold,
            scale=True,
            nu=np.sqrt(args.nu_sq),
            y_perturb=y_perturb,
        )
        result.update(
            snr=snr,
            selected_n_knots=selector.n_knots,
            dimension=selector.d,
        )

        # Load the reference law matching the selected knot event and verify its design.
        artifact = load_transport_artifact(args.map_dir, selector.n_knots)
        metadata = artifact["metadata"]
        _validate_artifact(args, metadata, selector.d)
        null_samples = artifact["null_samples"]
        result.update(
            num_conditional_samples=null_samples.shape[0],
            n_train=metadata["n_train"],
            n_val=metadata["n_val"],
            mean_num_tries=metadata["mean_num_tries"],
            max_num_tries=metadata["max_num_tries"],
            flow_seed=metadata["flow_seed"],
            flow_learning_rate=metadata["learning_rate"],
            flow_initial_val_loss=metadata["initial_val_loss"],
            flow_final_val_loss=metadata["final_val_loss"],
            flow_best_val_loss=metadata["best_val_loss"],
            map_configuration_id=metadata["map_configuration_id"],
        )

        # Both tests use the same saved conditional reference distribution.
        empirical_pvalue = empirical_l2_pvalue(selector.beta_hat, null_samples)
        result.update(
            empirical_l2_pvalue=empirical_pvalue,
            empirical_l2_reject=bool(empirical_pvalue <= args.alpha),
        )

        # With known sigma=1, the lower-right block is Cov(beta_hat) after
        # accounting for the fitted intercept in the selected spline model.
        selected_covariance = np.linalg.inv(selector.X.T @ selector.X)[1:, 1:]
        whitened_pvalue = empirical_whitened_l2_pvalue(
            selector.beta_hat,
            null_samples,
            selected_covariance,
        )
        result.update(
            whitened_l2_pvalue=whitened_pvalue,
            whitened_l2_reject=bool(whitened_pvalue <= args.alpha),
        )

        # The proposed test uses the nonlinear inverse map and a chi-square radius.
        fitted_transport_pvalue = transport_pvalue(selector.beta_hat, artifact)
        result.update(
            transport_pvalue=fitted_transport_pvalue,
            transport_reject=bool(fitted_transport_pvalue <= args.alpha),
            status="success",
        )
    except Exception as exc:
        result.update(
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    return result


def _validate_args(args):
    """Reject evaluation settings incompatible with the runner interface."""
    # Validate both single-seed and half-open seed-range execution modes.
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.seeds is not None:
        seed_start, seed_stop = args.seeds
        if seed_start < 0 or seed_stop <= seed_start:
            raise ValueError("--seeds requires 0 <= START < STOP")
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.n_fold < 2:
        raise ValueError("--n_fold must be at least 2")
    if args.max_knots < 2:
        raise ValueError("--max_knots must be at least 2")
    if args.nu_sq < 0:
        raise ValueError("--nu_sq must be non-negative")
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must lie strictly between 0 and 1")
    coefficient_direction(args)
    if not Path(args.map_dir).is_dir():
        raise ValueError(f"--map_dir does not exist: {args.map_dir}")


def _build_parser():
    # Expose single-replication and resumable batch interfaces through one parser.
    parser = argparse.ArgumentParser(description=__doc__)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument(
        "--seeds",
        nargs=2,
        type=int,
        metavar=("START", "STOP"),
        help="run seeds in range(START, STOP) and save one combined CSV",
    )
    parser.add_argument("--signal_fac", "--signal-fac", type=float, default=1.0)
    parser.add_argument(
        "--coefficient_direction",
        "--coefficient-direction",
        nargs=4,
        type=float,
        default=COEFFICIENT_DIRECTION.tolist(),
        metavar=("B1", "B2", "B3", "B4"),
        help=(
            "four-coefficient direction; automatically normalized to the "
            "fixed-design signal scale before multiplication by --signal-fac"
        ),
    )
    parser.add_argument("--map_dir", "--map-dir", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--design_seed", "--design-seed", type=int, default=2025)
    parser.add_argument("--n_fold", "--n-fold", type=int, default=10)
    parser.add_argument("--max_knots", "--max-knots", type=int, default=5)
    parser.add_argument("--nu_sq", "--nu-sq", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output_root",
        "--output-root",
        type=str,
        default="experiments/results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing result for the same configuration and seed",
    )
    return parser


def main(argv=None):
    # argv supports command-line execution and direct test invocation.
    args = _build_parser().parse_args(argv)
    if args.seed is None and args.seeds is None:
        args.seed = 0
    _validate_args(args)
    output_path = (
        batch_result_path(args) if args.seeds is not None else result_path(args)
    )
    if output_path.exists() and not args.overwrite and args.seeds is None:
        print(f"Result already exists; skipping: {output_path}")
        return 0

    # Run one record directly or merge missing seeds into a resumable batch file.
    if args.seeds is None:
        results = [run_experiment(args)]
        result_frame = pd.DataFrame(results)
    else:
        seed_start, seed_stop = args.seeds
        requested_seeds = set(range(seed_start, seed_stop))
        existing_frame = None
        if output_path.exists() and not args.overwrite:
            existing_frame = pd.read_csv(output_path)
            if "seed" not in existing_frame:
                raise ValueError(f"existing batch has no seed column: {output_path}")
            existing_frame["seed"] = pd.to_numeric(
                existing_frame["seed"], errors="raise"
            ).astype(int)
            if existing_frame["seed"].duplicated().any():
                raise ValueError(f"existing batch contains duplicate seeds: {output_path}")
            seeds_to_run = sorted(requested_seeds.difference(existing_frame["seed"]))
        else:
            seeds_to_run = sorted(requested_seeds)

        if not seeds_to_run:
            print(f"Combined result already covers requested seeds: {output_path}")
            return 0

        results = []
        for seed in seeds_to_run:
            seed_args = argparse.Namespace(**vars(args))
            seed_args.seed = seed
            results.append(run_experiment(seed_args))
        new_frame = pd.DataFrame(results)
        if existing_frame is None:
            result_frame = new_frame
        else:
            result_frame = pd.concat([existing_frame, new_frame], ignore_index=True)
            result_frame = result_frame.sort_values("seed").reset_index(drop=True)

    # Atomically replace the CSV so interrupted writes do not corrupt prior results.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    result_frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)
    requested_frame = result_frame[result_frame["seed"].isin(requested_seeds)] if args.seeds is not None else result_frame
    num_successful = int((requested_frame["status"] == "success").sum())
    print(
        f"Saved {num_successful}/{len(requested_frame)} successful results to "
        f"{output_path}"
    )
    return 0 if num_successful == len(requested_frame) else 1


if __name__ == "__main__":
    raise SystemExit(main())
