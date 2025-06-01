import numpy as np
from scipy.stats import norm
import argparse
import pickle
import os
from sklearn.preprocessing import PolynomialFeatures
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from experiments.polynomial_regression.polynomial_selector import PolynomialSelection
from flows.train import train_with_validation
from utils.utils import ci_bisection


def inference(model, params, beta_hat, Sigma, mean_shift, cov_chol, sig_level=0.05, compute_ci=False):
    d = len(beta_hat)
    beta_sd = np.sqrt(np.diag(Sigma))
    pvalues = np.zeros(d)
    cis = np.zeros((d, 2))

    if d == 1:
        beta_hat_center = cov_chol.T @ (beta_hat - mean_shift)
        def get_pvalue(beta_null):
            z_value = model.apply(params, beta_hat_center, context=jnp.array([beta_null]), method=model.inverse)[0]
            return 2 * norm.cdf(-np.abs(z_value))
        pvalues[0] = get_pvalue(0.)
        if compute_ci:
            cis[0] = ci_bisection(get_pvalue, beta_sd[0], beta_hat[0] + 5 * beta_sd[0], beta_hat[0] - 5 * beta_sd[0], sig_level=sig_level, tol=1e-4)
    else:
        beta_hat_center = cov_chol.T @ (beta_hat - mean_shift)

        def neg_loglik(beta_hat, beta_null):
            beta_hat_center_ = cov_chol.T @ (beta_hat - mean_shift)
            return model.apply(params, beta_hat_center_, beta_null, method=model.forward_kl)
        
        for j in range(d):
            eta = np.eye(d)[j]
            c = Sigma @ eta / (np.dot(eta, Sigma @ eta))
            beta_perp = beta_hat - c * beta_hat[j]
            gridsize = 200
            sd_j = np.sqrt(Sigma[j, j])
            grid = jnp.linspace(-10, 10, gridsize) * sd_j + beta_hat[j]

            @jax.jit
            def logp_j(beta_hat_j, beta_null_j):
                beta_ = jnp.copy(beta_hat)
                beta_ = beta_.at[j].set(beta_null_j)
                return -neg_loglik(beta_perp + c * beta_hat_j, beta_)

            @jax.jit
            def get_pvalue(beta_null_j):
                logp = jax.vmap(logp_j, in_axes=(0, None))(grid, beta_null_j)
                isnan = jnp.all(jnp.isnan(logp))
                logp = jnp.nan_to_num(logp, nan=-np.inf)
                logp -= logp.max()
                log_normalization_const = logsumexp(logp)

                idx_left = (grid <= beta_hat[j]) 
                log_numerator_left = logsumexp(jnp.where(idx_left, logp, -jnp.inf))
                pval = jnp.exp(log_numerator_left - log_normalization_const)
                pval = jax.lax.select(pval < 0.5, 2 * pval, 2 * (1 - pval))
                return jax.lax.select(isnan, 0., pval)

            pvalues[j] = get_pvalue(0.)
            if compute_ci:
                cis[j] = ci_bisection(get_pvalue, sd_j, beta_hat[j] + 5 * sd_j, beta_hat[j] - 5 * sd_j, sig_level=sig_level, tol=1e-4)
    if compute_ci:
        return  pvalues, cis
    return pvalues

def generate_data(seed, nu):
    rng = np.random.default_rng(seed)    
    y = mu + rng.normal(size=(n,)) * sigma
    y_perturb = nu * rng.normal(size=(n,))
    return y, y_perturb

def run(seed, n_train, n_val=1000, hidden_dim=8, nu_sq=0.):
    nu = np.sqrt(nu_sq)
    y, y_perturb = generate_data(seed, nu)
    selector = PolynomialSelection(X, y, sigma, nu, y_perturb)
    d = selector.selected_deg
    print('selected degree:', d)
    if d < 1:
        return
    
    beta_target = (np.linalg.pinv(X[:, :d+1]) @ mu)[1:]
    beta_hat = selector.beta_hat
    Sigma = selector.Sigma

    sig_level = 0.05
    
    pvalues_all = {}
    intervals_all = {}

    pvalues_all['naive'], intervals_all['naive'] = selector.naive_inference(sig_level=sig_level)
    if nu > 0:
        pvalues_all['splitting'], intervals_all['splitting'] = selector.splitting_inference(sig_level=sig_level)

    print("Generating samples ...")
    rng = np.random.default_rng(0)
    train_samples, train_contexts = selector.generate_training_data(rng, n_train+n_val, max_try=100)
    print("Generated", train_samples.shape[0], 'samples')

    if train_samples.shape[0] == 0:
        print("Failed to generate training data")
        return
    
    mean_shift = np.mean(train_samples, axis=0)
    cov_chol = np.linalg.cholesky(np.linalg.inv(np.atleast_2d(np.cov(train_samples.T))))
    samples_center = (train_samples - mean_shift) @ cov_chol

    val_samples = samples_center[n_train:]
    val_contexts = train_contexts[n_train:]
    train_samples = samples_center[:n_train]
    train_contexts = train_contexts[:n_train]

    def train_and_inference(seed):
        model, params, val_losses = train_with_validation(train_samples, train_contexts, val_samples, val_contexts, learning_rate=1e-4, max_iter=10000, checkpoint_every=1000, hidden_dims=[hidden_dim], n_layers=12, num_bins=20, seed=seed)
        val_losses = np.array(val_losses)
        
        pvals, cis = inference(model, params, beta_hat, Sigma, mean_shift, cov_chol, compute_ci=True, sig_level=sig_level)
        return pvals, cis, val_losses
    
    for _seed in range(10):
        print("Training seed: ", _seed)
        pvalues_all['nf'] , intervals_all['nf'], val_losses = train_and_inference(seed=_seed)
        if (not np.isnan(pvalues_all['nf']).any()) and (not np.isinf(intervals_all['nf']).any()):
            break

    coverages_all = {}
    for key, ci in intervals_all.items():
        coverages_all[key] = (ci[:, 0] <= beta_target) & (beta_target <= ci[:, 1])
    print(coverages_all)
    print(pvalues_all)

    return {'coverages': coverages_all, 'pvalues': pvalues_all, 'intervals': intervals_all, 'losses': val_losses}
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='2025')
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--signal_fac', type=float, default=1.)
    parser.add_argument('--mixed_sign', default=False, action='store_true')
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--nu_sq', type=float, default=0.)
    parser.add_argument('--rootdir', type=str, default='experiments/results')
    args = parser.parse_args()

    n = args.n
    p = 4 
    sigma = 1.
    rng = np.random.default_rng(2025)
    beta = np.array([0, 0, 1, 1]) * args.signal_fac * np.sqrt(n)

    x = rng.normal(size=(n,))
    poly = PolynomialFeatures(p, include_bias=True)
    X = poly.fit_transform(x.reshape(-1, 1)) 
    X[:, 1:] -= X[:, 1:].mean(axis=0)
    X[:, 1:] /= (X[:, 1:].std(axis=0) * np.sqrt(n))
    X[:, 0] /= np.sqrt(n)
    mu = X[:, 1:] @ beta
    snr = np.sqrt(np.var(mu) / sigma**2)

    seed = args.seed
    results = run(seed, n_train=args.n_train, n_val=args.n_val, hidden_dim=args.hidden_dim, nu_sq=args.nu_sq)
    if results is not None:
        savepath = os.path.join(args.rootdir, args.date, 'poly')

        prefix = f'poly_{n}_{p}_signal_{args.signal_fac}_nusq_{args.nu_sq}_train_{args.n_train}_val_{args.n_val}_hidden_{args.hidden_dim}'
        path = os.path.join(savepath, prefix)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f'{seed}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f'Saved to {filename}')

