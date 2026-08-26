"""Reusable knot-conditioned transport artifacts for the spline experiment."""

import json
import os
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from scipy.stats import chi2, kstest

from flows.realnvp import RealNVP
from flows.train import train_with_validation


CALIBRATION_ATTEMPTS = ((0, 1e-5), (1, 1e-5), (2, 1e-5))
# Shared architecture constants make saved artifacts reconstructible at load time.
N_LAYERS = 12
HIDDEN_DIMS = (8,)
NUM_BINS = 20


def float_slug(value):
    """Format a float for stable directory names."""
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def map_configuration_id(args):
    """Identify settings that determine the four conditional reference laws."""
    return (
        f"n_{args.n}_design_{args.design_seed}_train_{args.n_train}"
        f"_val_{args.n_val}_fold_{args.n_fold}_maxknots_{args.max_knots}"
        f"_nusq_{float_slug(args.nu_sq)}_iter_{args.max_iter}"
        f"_checkpoint_{args.checkpoint_every}"
    )


def artifact_directory(map_dir, selected_n_knots):
    """Return the directory holding one selected-knot transport artifact."""
    return Path(map_dir) / f"knots_{selected_n_knots}"


def fit_transport_artifact(
    samples,
    n_train,
    max_iter,
    checkpoint_every,
):
    """Fit a stable flow and return its parameters and affine preprocessing."""
    # samples contains training followed by validation draws; n_train is the
    # split point, while iteration arguments control checkpointed optimization.
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] <= n_train:
        raise ValueError("samples must include non-empty training and validation sets")

    # Affine prewhitening improves optimization without replacing the learned map.
    mean_shift = np.mean(samples, axis=0)
    covariance = np.atleast_2d(np.cov(samples.T))
    covariance_cholesky = np.linalg.cholesky(np.linalg.inv(covariance))
    centered_samples = (samples - mean_shift) @ covariance_cholesky
    dimension = samples.shape[1]
    errors = []
    best_candidate = None
    validation_samples = centered_samples[n_train:]
    # Use a standard Gaussian fit as a sanity baseline for candidate flows.
    gaussian_baseline_nll = float(
        np.mean(
            0.5
            * (
                dimension * np.log(2.0 * np.pi)
                + np.sum(validation_samples ** 2, axis=1)
            )
        )
    )

    # Retry independent initializations and retain the valid lowest-loss candidate.
    for flow_seed, learning_rate in CALIBRATION_ATTEMPTS:
        try:
            model, params, val_losses = train_with_validation(
                centered_samples[:n_train],
                None,
                validation_samples,
                None,
                learning_rate=learning_rate,
                max_iter=max_iter,
                checkpoint_every=checkpoint_every,
                hidden_dims=list(HIDDEN_DIMS),
                n_layers=N_LAYERS,
                num_bins=NUM_BINS,
                seed=flow_seed,
            )
        except Exception as exc:
            errors.append(f"seed {flow_seed}: {type(exc).__name__}: {exc}")
            continue
        losses = np.asarray([float(loss) for loss in val_losses])
        if losses.size == 0 or not np.all(np.isfinite(losses)):
            errors.append(f"seed {flow_seed}: non-finite validation loss")
            continue
        best_val_loss = float(np.min(losses))
        if best_val_loss > gaussian_baseline_nll + 0.5:
            errors.append(
                f"seed {flow_seed}: best validation loss {best_val_loss:.4f} "
                f"exceeds Gaussian baseline {gaussian_baseline_nll:.4f}"
            )
            continue

        # Reject flows whose held-out transform is non-finite or poorly standardized.
        validation_z = model.apply(
            params,
            jnp.asarray(validation_samples),
            context=None,
            method=model.inverse,
        )[0]
        validation_z = np.asarray(validation_z, dtype=float)
        if not np.all(np.isfinite(validation_z)):
            errors.append(f"seed {flow_seed}: non-finite validation transform")
            continue

        z_mean_max_abs = float(np.max(np.abs(np.mean(validation_z, axis=0))))
        z_covariance = np.atleast_2d(np.cov(validation_z.T))
        z_covariance_max_abs_error = float(
            np.max(np.abs(z_covariance - np.eye(dimension)))
        )
        if z_mean_max_abs > 0.75 or z_covariance_max_abs_error > 1.0:
            errors.append(
                f"seed {flow_seed}: transformed validation moments are unstable"
            )
            continue

        # Record a chi-square goodness-of-fit diagnostic for whitened radii.
        squared_z_norms = np.sum(validation_z ** 2, axis=1)
        ks_result = kstest(squared_z_norms, chi2(df=dimension).cdf)

        diagnostics = {
            "flow_seed": flow_seed,
            "learning_rate": learning_rate,
            "initial_val_loss": float(losses[0]),
            "final_val_loss": float(losses[-1]),
            "best_val_loss": best_val_loss,
            "gaussian_baseline_val_loss": gaussian_baseline_nll,
            "z_mean_max_abs": z_mean_max_abs,
            "z_covariance_max_abs_error": z_covariance_max_abs_error,
            "z_norm_chi2_ks_statistic": float(ks_result.statistic),
            "z_norm_chi2_ks_pvalue": float(ks_result.pvalue),
            "val_losses": losses.tolist(),
        }
        candidate = (
            best_val_loss,
            model,
            params,
            mean_shift,
            covariance_cholesky,
            diagnostics,
        )
        if best_candidate is None or candidate[0] < best_candidate[0]:
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("; ".join(errors))
    _, model, params, mean_shift, covariance_cholesky, diagnostics = best_candidate
    diagnostics["rejected_candidates"] = errors
    return model, params, mean_shift, covariance_cholesky, diagnostics


def save_transport_artifact(
    map_dir,
    selected_n_knots,
    samples,
    mean_shift,
    covariance_cholesky,
    params,
    metadata,
):
    """Persist one flow, its preprocessing, and the empirical null reference."""
    # map_dir identifies a calibration; selected_n_knots identifies its event.
    # Remaining arguments are the fitted reference law and descriptive metadata.
    output_dir = artifact_directory(map_dir, selected_n_knots)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write each component to a temporary sibling before atomically replacing it.
    arrays_path = output_dir / "reference.npz"
    arrays_temporary = output_dir / "reference.tmp"
    with arrays_temporary.open("wb") as file_handle:
        np.savez_compressed(
            file_handle,
            null_samples=np.asarray(samples),
            mean_shift=np.asarray(mean_shift),
            covariance_cholesky=np.asarray(covariance_cholesky),
        )
    os.replace(arrays_temporary, arrays_path)

    params_path = output_dir / "params.msgpack"
    params_temporary = output_dir / "params.tmp"
    params_temporary.write_bytes(serialization.to_bytes(params))
    os.replace(params_temporary, params_path)

    metadata_path = output_dir / "metadata.json"
    metadata_temporary = output_dir / "metadata.tmp"
    metadata_temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_temporary, metadata_path)
    return output_dir


@lru_cache(maxsize=None)
def load_transport_artifact(map_dir, selected_n_knots):
    """Load and validate one knot-conditioned flow artifact."""
    # map_dir is the calibration root; selected_n_knots chooses the event-specific map.
    input_dir = artifact_directory(map_dir, selected_n_knots)
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["selected_n_knots"] != selected_n_knots:
        raise ValueError("artifact knot count does not match requested knot count")

    with np.load(input_dir / "reference.npz") as arrays:
        null_samples = arrays["null_samples"]
        mean_shift = arrays["mean_shift"]
        covariance_cholesky = arrays["covariance_cholesky"]

    dimension = int(metadata["dimension"])
    if null_samples.ndim != 2 or null_samples.shape[1] != dimension:
        raise ValueError("artifact reference samples have an invalid shape")
    if mean_shift.shape != (dimension,):
        raise ValueError("artifact mean shift has an invalid shape")
    if covariance_cholesky.shape != (dimension, dimension):
        raise ValueError("artifact covariance factor has an invalid shape")

    # Rebuild the saved architecture, then deserialize parameters into its template.
    model = RealNVP(
        dim=dimension,
        n_layers=int(metadata["n_layers"]),
        hidden_dims=tuple(metadata["hidden_dims"]),
    )
    template = model.init(
        jax.random.key(0),
        jnp.ones((1, dimension)),
        context=None,
    )
    params = serialization.from_bytes(
        template,
        (input_dir / "params.msgpack").read_bytes(),
    )
    return {
        "model": model,
        "params": params,
        "null_samples": null_samples,
        "mean_shift": mean_shift,
        "covariance_cholesky": covariance_cholesky,
        "metadata": metadata,
    }


def transport_pvalue(observed, artifact):
    """Transform an observed statistic with a saved map and compute its p-value."""
    # observed is a selected coefficient vector; artifact supplies preprocessing,
    # the inverse map, and the reference dimension.
    observed = np.asarray(observed, dtype=float)
    centered_observed = artifact["covariance_cholesky"].T @ (
        observed - artifact["mean_shift"]
    )
    z_value = artifact["model"].apply(
        artifact["params"],
        jnp.asarray(centered_observed),
        context=None,
        method=artifact["model"].inverse,
    )[0]
    z_value = np.asarray(z_value, dtype=float)
    if not np.all(np.isfinite(z_value)):
        raise RuntimeError("saved transport produced a non-finite value")
    return float(chi2.sf(np.sum(z_value ** 2), df=observed.shape[0]))
