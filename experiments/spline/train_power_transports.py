"""Calibrate and save one global-null transport map per spline knot count.

For the 500-draw setup used in the small power experiment, run:

    python -m experiments.spline.train_power_transports \
      --n_train 400 --n_val 100

The printed map directory can then be passed to every invocation of
``experiments.spline.run_power_comparison`` regardless of signal or data seed.
"""

import argparse
from pathlib import Path

import numpy as np

from experiments.spline.power_transport import (
    HIDDEN_DIMS,
    N_LAYERS,
    NUM_BINS,
    artifact_directory,
    fit_transport_artifact,
    map_configuration_id,
    save_transport_artifact,
)
from experiments.spline.spline_selector import SplineSelection


def _fixed_design(n, design_seed):
    # n is the row count and design_seed fixes the covariates across calibrations.
    rng = np.random.default_rng(design_seed)
    return rng.uniform(size=(n, 1))


def train_all_maps(args):
    """Train the finite family of knot-conditioned global-null maps."""
    # args supplies design, sampling, optimizer, and output settings from the CLI.
    map_dir = (
        Path(args.output_root)
        / "spline_power_maps"
        / map_configuration_id(args)
    )
    x = _fixed_design(args.n, args.design_seed)
    required_samples = args.n_train + args.n_val

    # Each possible selected knot count defines a distinct conditional null law.
    for selected_n_knots in range(2, args.max_knots + 1):
        output_dir = artifact_directory(map_dir, selected_n_knots)
        required_files = (
            output_dir / "metadata.json",
            output_dir / "params.msgpack",
            output_dir / "reference.npz",
        )
        if all(path.exists() for path in required_files) and not args.overwrite:
            print(f"Knot count {selected_n_knots}: artifact exists; skipping")
            continue

        print(f"Knot count {selected_n_knots}: sampling {required_samples} draws")
        # Build an event-specific sampler without generating an observed response.
        selector = SplineSelection.for_global_null_event(
            x,
            selected_n_knots=selected_n_knots,
            sigma=1.0,
            maximum_knots=args.max_knots,
            n_fold=args.n_fold,
            scale=True,
            nu=np.sqrt(args.nu_sq),
        )
        # Derive an independent, reproducible stream for each knot event.
        sampling_rng = np.random.default_rng(
            np.random.SeedSequence(
                [args.calibration_seed, args.design_seed, selected_n_knots]
            )
        )
        samples, num_tries = selector.sample_from_global_null(
            sampling_rng,
            required_samples,
            return_num_tries=True,
        )

        print(f"Knot count {selected_n_knots}: fitting transport")
        # Fit on the leading draws and validate on the remaining reference draws.
        _, params, mean_shift, covariance_cholesky, diagnostics = (
            fit_transport_artifact(
                samples,
                n_train=args.n_train,
                max_iter=args.max_iter,
                checkpoint_every=args.checkpoint_every,
            )
        )
        # Store all configuration and diagnostics needed to audit compatibility.
        metadata = {
            "artifact_version": 1,
            "map_configuration_id": map_configuration_id(args),
            "selected_n_knots": selected_n_knots,
            "dimension": selector.d,
            "n": args.n,
            "design_seed": args.design_seed,
            "calibration_seed": args.calibration_seed,
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_fold": args.n_fold,
            "max_knots": args.max_knots,
            "nu_sq": args.nu_sq,
            "max_iter": args.max_iter,
            "checkpoint_every": args.checkpoint_every,
            "num_conditional_samples": int(samples.shape[0]),
            "mean_num_tries": float(np.mean(num_tries)),
            "max_num_tries": int(np.max(num_tries)),
            "n_layers": N_LAYERS,
            "hidden_dims": list(HIDDEN_DIMS),
            "num_bins": NUM_BINS,
            **diagnostics,
        }
        saved_dir = save_transport_artifact(
            map_dir,
            selected_n_knots,
            samples,
            mean_shift,
            covariance_cholesky,
            params,
            metadata,
        )
        print(f"Knot count {selected_n_knots}: saved to {saved_dir}")

    return map_dir


def _validate_args(args):
    # Fail early on invalid sample, CV, or checkpoint settings.
    if args.n <= 0 or args.n_train <= 0 or args.n_val <= 0:
        raise ValueError("sample sizes must be positive")
    if args.n_fold < 2:
        raise ValueError("--n_fold must be at least 2")
    if args.max_knots < 2:
        raise ValueError("--max_knots must be at least 2")
    if args.nu_sq < 0:
        raise ValueError("--nu_sq must be non-negative")
    if args.checkpoint_every <= 0 or args.max_iter < args.checkpoint_every:
        raise ValueError("--max_iter must be at least --checkpoint_every")
    if args.max_iter % args.checkpoint_every != 0:
        raise ValueError("--max_iter must be divisible by --checkpoint_every")


def _build_parser():
    # Define both underscore and hyphen spellings for script and shell callers.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--design_seed", "--design-seed", type=int, default=2025)
    parser.add_argument(
        "--calibration_seed", "--calibration-seed", type=int, default=0
    )
    parser.add_argument("--n_train", "--n-train", type=int, default=2000)
    parser.add_argument("--n_val", "--n-val", type=int, default=500)
    parser.add_argument("--n_fold", "--n-fold", type=int, default=10)
    parser.add_argument("--max_knots", "--max-knots", type=int, default=5)
    parser.add_argument("--nu_sq", "--nu-sq", type=float, default=0.0)
    parser.add_argument("--max_iter", "--max-iter", type=int, default=10000)
    parser.add_argument(
        "--checkpoint_every", "--checkpoint-every", type=int, default=1000
    )
    parser.add_argument(
        "--output_root",
        "--output-root",
        default="experiments/results",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    # argv permits direct CLI use and isolated tests with explicit arguments.
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    map_dir = train_all_maps(args)
    print(f"All knot-conditioned maps are available in {map_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
