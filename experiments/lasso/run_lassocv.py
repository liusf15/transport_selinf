import numpy as np
import argparse
import os
import pickle
from jax.scipy.stats import multivariate_normal as mvn
import jax

from experiments.lasso.randomized_lasso import RandomLassoCV
from experiments.lasso.regression_designs import gaussian_instance
from flows.train import train_with_validation

def run(seed, p, s, signal_fac, nu, rho, n_train, n_val=1000, hidden_dim=8, savepath=None):
    n = 100
    sigma = 1.
    signal = np.sqrt(signal_fac * 2 * np.log(p))
    equi = False
    random_signs = False
    rng = np.random.default_rng(seed)
    X, y, beta = gaussian_instance(rng, n, p, s, sigma, rho, signal, random_signs=random_signs, scale=True, center=True, equicorrelated=equi)
    w_y = nu * rng.normal(size=(n,))
    w = X.T @ w_y

    alphas = np.logspace(-2, np.log10(5), 10) * np.sqrt(np.log(p)) / n
    rl = RandomLassoCV(X, y, sigma, alphas, nu=nu, w=w, nfold=10)
    d = rl.d
    print("selected", d, "variables")
    if d == 0:
        return

    sig_level = 0.05
    intervals_all = {}
    pvalues_all = {}

    beta_target = np.linalg.pinv(rl.X_E) @ X @ beta

    pvalues_all['naive'], intervals_all['naive'] = rl.naive_inference(sig_level=sig_level)

    if nu > 0:
        y_indep = y - w_y * (sigma**2 / nu**2)
        pvalues_all['splitting'], intervals_all['splitting'] = rl.splitting_inference(y_indep, sig_level=sig_level)

    def neg_loglik(beta_hat, beta_null):
        return -mvn.logpdf(beta_hat, mean=beta_null, cov=rl.Sigma)
    
    # unadjusted
    if nu > 0:
        methods = ['hard_threshold', 'bivnormal', 'sov']
    else:
        methods = ['hard_threshold']
    for method in methods:
        print(method)
        pvalues_all[method], intervals_all[method] = rl.adjusted_inference(neg_loglik, method_sel_prob=method, compute_ci=True, sig_level=sig_level)

    print("Generating samples ...")
    train_samples, train_contexts = rl.generate_training_data(rng, n_train+n_val, resample_scale=1., max_try=100)
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

    def train():
        for _seed in range(10):
            model, params, val_losses = train_with_validation(train_samples, train_contexts, val_samples, val_contexts, learning_rate=1e-4, max_iter=10000, checkpoint_every=1000, hidden_dims=[hidden_dim], n_layers=12, num_bins=20, seed=_seed)
            val_losses = np.array(val_losses)

            def neg_loglik_adjusted(beta_hat, beta_null):
                beta_hat_center_ = cov_chol.T @ (beta_hat - mean_shift)
                return model.apply(params, beta_hat_center_, beta_null, method=model.forward_kl)

            pval, ci = rl.adjusted_inference(neg_loglik_adjusted, method_sel_prob='hard_threshold', compute_ci=True, sig_level=sig_level)

            if (not np.isnan(pval).any()) and (not np.isinf(ci).any()):
                return model, params
            print("NaN or Inf in validation loss, retrying with new seed")
        return model, params

    model, params = train()

    @jax.jit
    def neg_loglik_adjusted(beta_hat, beta_null):
        beta_hat_center_ = cov_chol.T @ (beta_hat - mean_shift)
        return model.apply(params, beta_hat_center_, beta_null, method=model.forward_kl)
    
    for method in methods:
        print("adjusted", method)
        pvalues_all['adjusted_' + method], intervals_all['adjusted_' + method] = rl.adjusted_inference(neg_loglik_adjusted, method_sel_prob=method, compute_ci=True, sig_level=sig_level)

    print(pvalues_all)
    print(intervals_all)

    false_rejects_all = {}
    coverages_all = {}
    for key, item in pvalues_all.items():
        false_rejects_all[key] = item < sig_level
        coverages_all[key] = (intervals_all[key][:, 0] <= beta_target) * (beta_target <= intervals_all[key][:, 1])
    print(false_rejects_all)
    print(coverages_all)
    
    results_all = {'pvalues': pvalues_all, 'intervals': intervals_all, 'false_rejects': false_rejects_all, 'coverages': coverages_all}
    if savepath is not None:
        prefix = f'lassocv_{n}_{p}_{s}_{round(nu, 3)}_{signal_fac}_rho_{rho}_train_{n_train}_val_{n_val}_hidden_{hidden_dim}'
        path = os.path.join(savepath, prefix)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f'{seed}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(results_all, f)
        print(f'Saved to {filename}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cv rlasso')
    parser.add_argument('--date', type=str, default='20250221')
    parser.add_argument('--p', type=int, default=20)
    parser.add_argument('--s', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--signal_fac', type=float, default=.6)
    parser.add_argument('--nu_sq', type=float, default=.1)
    parser.add_argument('--rho', type=float, default=.5)
    parser.add_argument('--n_train', type=int, default=2000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--max_iter', type=int, default=3000)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--rootdir', type=str, default='experiments/results')
    args = parser.parse_args()

    savepath = os.path.join(args.rootdir, args.date, 'lassocv')
    nu = np.sqrt(args.nu_sq)
    run(p=args.p, s=args.s, seed=args.seed, signal_fac=args.signal_fac, nu=nu, rho=args.rho, n_train=args.n_train, n_val=args.n_val, hidden_dim=args.hidden_dim, savepath=savepath)
