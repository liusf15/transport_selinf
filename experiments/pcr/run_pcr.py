import numpy as np
import os
import argparse
import pickle
import pandas as pd
import jax
from jax.scipy.stats import norm
import jax.numpy as jnp
from scipy.stats import chi2
from jax.scipy.special import logsumexp

from experiments.pcr.pcr_selector import PCRSelection
from flows.train import train_with_validation
from utils.utils import ci_bisection

n, p = 100, 50

def inference(model, params, suff_stat, sd_suff_stat, beta_hat, sd_beta, mean_shift, cov_chol, sig_level=0.05, compute_ci=False):
    d = len(suff_stat)
    pvalues = np.zeros(d)
    cis = np.zeros((d, 2))

    if d == 1:
        suff_stat_center = cov_chol.T @ (suff_stat - mean_shift)
        def get_pvalue(beta_null):
            z_value = model.apply(params, suff_stat_center, context=jnp.array([beta_null]), method=model.inverse)[0]
            return 2 * norm.cdf(-np.abs(z_value))
        pvalues[0] = get_pvalue(0.)[0]
        global_pval = pvalues[0]
        if compute_ci:
            cis[0] = ci_bisection(get_pvalue, sd_beta[0], beta_hat[0] + 5 * sd_beta[0], beta_hat[0] - 5 * sd_beta[0], sig_level=sig_level, tol=1e-4)
    else:
        suff_stat_center = cov_chol.T @ (suff_stat - mean_shift)
        z_value = model.apply(params, suff_stat_center, context=jnp.zeros(d), method=model.inverse)[0]
        global_pval = 1 - chi2.cdf(np.sum(z_value**2), df=d)

        def neg_loglik(suff_stat, beta_null):
            suff_stat_center_ = cov_chol.T @ (suff_stat - mean_shift)
            return model.apply(params, suff_stat_center_, beta_null, method=model.forward_kl)
        
        for j in range(d):
            c = np.eye(d)[j]
            beta_perp = suff_stat - c * suff_stat[j]
            gridsize = 200
            grid = jnp.linspace(-10, 10, gridsize) * sd_suff_stat[j] + suff_stat[j]

            @jax.jit
            def logp_j(beta_hat_j, beta_null_j):
                beta_ = jnp.zeros(d)
                beta_ = beta_.at[j].set(beta_null_j)
                return -neg_loglik(beta_perp + c * beta_hat_j, beta_)

            @jax.jit
            def get_pvalue(beta_null_j):
                logp = jax.vmap(logp_j, in_axes=(0, None))(grid, beta_null_j)
                logp = jnp.nan_to_num(logp)
                logp -= logp.max()
                log_normalization_const = logsumexp(logp)

                idx_left = (grid <= suff_stat[j]) 
                log_numerator_left = logsumexp(jnp.where(idx_left, logp, -jnp.inf))
                pval = jnp.exp(log_numerator_left - log_normalization_const)
                return jax.lax.select(pval < 0.5, 2 * pval, 2 * (1 - pval))

            pvalues[j] = get_pvalue(0.)
            if compute_ci:
                cis[j] = ci_bisection(get_pvalue, sd_beta[j], beta_hat[j] + 5 * sd_beta[j], beta_hat[j] - 5 * sd_beta[j], sig_level=sig_level, tol=1e-4)

    if compute_ci:
        return  global_pval, pvalues, cis
    return global_pval, pvalues


def generate_data(seed, rho):
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(mean=np.zeros(p), cov=rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p))), size=n)
    y = rng.binomial(1, 0.5, size=n)
    return X, y

def run(seed, rho, n_train, n_val, n_fold=5):
    X, y = generate_data(seed, rho)
    selector = PCRSelection(X, y, n_fold=n_fold)
    d = selector.d
    if d == 0:
        print("Selected empty model")
        return
    
    sig_level = 0.05
    pvalues_all = {}
    intervals_all = {}
    global_pvalues_all = {}
    intervals_all['naive'], pvalues_all['naive'], global_pvalues_all['naive'] = selector.naive_inference(sig_level)

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

    suff_stat = selector.suff_stat
    sd_suff_stat = np.sqrt(np.diag(selector.cov_suff_stat))
    beta_hat = selector.beta_hat
    sd_beta = np.sqrt(np.diag(selector.Sigma))

    def train_and_inference(seed):
        model, params, val_losses = train_with_validation(train_samples, train_contexts, val_samples, val_contexts, learning_rate=1e-4, max_iter=10000, checkpoint_every=1000, hidden_dims=[8], n_layers=12, num_bins=20, seed=seed)
        val_losses = np.array(val_losses)
        global_pval, pvalues, cis = inference(model, params, suff_stat, sd_suff_stat, beta_hat, sd_beta, mean_shift, cov_chol, sig_level, compute_ci=True)
        return global_pval, pvalues, cis, val_losses
    
    for _seed in range(10):
        print("Training seed: ", _seed)
        global_pvalues_all['nf'], pvalues_all['nf'] , intervals_all['nf'], val_losses = train_and_inference(seed=_seed)
        if (not np.isnan(val_losses[-1])) and (not np.isnan(pvalues_all['nf']).any()) and (not np.isinf(intervals_all['nf']).any()):
            break
    
    print(global_pvalues_all)
    print(pvalues_all)
    print(intervals_all)
    return {'global_pvalue': global_pvalues_all, 'pvalues': pvalues_all, 'intervals': intervals_all}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='20250409')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rho', type=float, default=.9)
    parser.add_argument('--n_train', type=int, default=2000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--max_iter', type=int, default=3000)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--rootdir', type=str, default='/mnt/ceph/users/sliu1/transport_selinf/')
    args = parser.parse_args()

    savepath = os.path.join(args.rootdir, args.date, 'pcr')

    results = run(seed=args.seed, rho=args.rho, n_train=args.n_train, n_val=args.n_val, n_fold=5)
    if results is not None:
        savepath = os.path.join(args.rootdir, args.date, 'pcr')

        prefix = f'pcr_rho_{args.rho}_train_{args.n_train}_val_{args.n_val}'
        savepath = os.path.join(savepath, prefix)
        os.makedirs(savepath, exist_ok=True)
        filename = os.path.join(savepath, f'{args.seed}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f'Saved to {filename}')
