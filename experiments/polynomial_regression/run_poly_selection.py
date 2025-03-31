import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
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
        global_pval = pvalues[0]
        if compute_ci:
            cis[0] = ci_bisection(get_pvalue, beta_sd[0], beta_hat[0] + 5 * beta_sd[0], beta_hat[0] - 5 * beta_sd[0], sig_level=sig_level, tol=1e-4)
    else:
        beta_hat_center = cov_chol.T @ (beta_hat - mean_shift)
        z_value = model.apply(params, beta_hat_center, context=jnp.zeros(d), method=model.inverse)[0]
        global_pval = 1 - chi2.cdf(np.sum(z_value**2), df=d)

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
                logp = jnp.nan_to_num(logp)
                logp -= logp.max()
                log_normalization_const = logsumexp(logp)

                idx_left = (grid <= beta_hat[j]) 
                log_numerator_left = logsumexp(jnp.where(idx_left, logp, -jnp.inf))
                pval = jnp.exp(log_numerator_left - log_normalization_const)
                return jax.lax.select(pval < 0.5, 2 * pval, 2 * (1 - pval))

            pvalues[j] = get_pvalue(0.)
            if compute_ci:
                cis[j] = ci_bisection(get_pvalue, sd_j, beta_hat[j] + 5 * sd_j, beta_hat[j] - 5 * sd_j, sig_level=sig_level, tol=1e-4)

    if compute_ci:
        return  global_pval, pvalues, cis
    return global_pval, pvalues


def generate_data(seed):
    rng = np.random.default_rng(seed)    
    y = mu + rng.normal(size=(n,)) * sigma
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(p+1)])
    df['y'] = y 
    return df

def run(seed, n_train, n_val=1000, hidden_dim=8):
    df = generate_data(seed)
    selector = PolynomialSelection(df, sigma)
    d = selector.selected_deg
    print('selected degree:', d)
    if d < 1:
        return
    
    beta_target = (np.linalg.pinv(X[:, :d+1]) @ mu)[1:]
    beta_hat = selector.beta_hat
    Sigma = selector.Sigma
    beta_sd = np.sqrt(np.diag(Sigma))

    sig_level = 0.05
    q = norm.ppf(sig_level / 2)
    lower = beta_hat + q * beta_sd
    upper = beta_hat - q * beta_sd
    
    pvalues_all = {}
    intervals_all = {}
    global_pvalues_all = {}
    # intervals_all['naive'] = pd.DataFrame({0: lower, 1: upper}, index=[f'x{i}' for i in range(1, d+1)])
    intervals_all['naive'] = np.stack([lower, upper]).T
    pvalues_all['naive'] = 2 * norm.cdf(-np.abs(beta_hat) / beta_sd)
    naive_zvalue = np.linalg.cholesky(np.linalg.inv(Sigma)).T @ beta_hat
    global_pvalues_all['naive'] = 1 - chi2.cdf(np.sum(naive_zvalue**2), df=d)

    print("Generating samples ...")
    rng = np.random.default_rng(0)
    train_samples, train_contexts = selector.generate_training_data(rng, n_train+n_val, resample_scale=1., max_try=100)
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
        
        global_pval, pvals, cis = inference(model, params, beta_hat, Sigma, mean_shift, cov_chol, compute_ci=True, sig_level=sig_level)
        return global_pval, pvals, cis, val_losses
    
    for _seed in range(10):
        print("Training seed: ", _seed)
        global_pvalues_all['nf'], pvalues_all['nf'] , intervals_all['nf'], val_losses = train_and_inference(seed=_seed)
        if (not np.isnan(pvalues_all['nf']).any()) and (not np.isinf(intervals_all['nf']).any()):
            break

    # if d == 1:
    #     model, params, losses = train_nf(samples_center, contexts, learning_rate=1e-1, max_iter=max_iter, hidden_dims=[8], num_bins=20)
    #     losses = np.array(losses)
    #     if np.isnan(losses[-1]) or np.std(losses[-100:]) / np.mean(losses[-100:]) > 0.1:
    #         print("Unstable training, try learning rate 1e-3")
    #         model, params, losses = train_nf(samples_center, contexts, learning_rate=1e-2, max_iter=max_iter)
        
    #     beta_hat_center = cov_chol.T @ (selector.beta_hat - mean_shift)
    #     def get_pvalue(beta_null):
    #         # z_value = model.inverse(params, beta_hat_center, np.array([beta_null]))[0]
    #         z_value = model.apply(params, beta_hat_center, context=jnp.array([beta_null]), method=model.inverse)[0]
    #         return 2 * norm.cdf(-np.abs(z_value))
        
    #     nf_ci = ci_bisection(get_pvalue, beta_sd[0], beta_hat[0] + 5 * beta_sd[0], beta_hat[0] - 5 * beta_sd[0], sig_level=sig_level, tol=1e-4)
    #     if np.any(np.isinf(np.array(nf_ci))):
    #         print("Got Inf, retraining with learning rate 1e-2")
    #         model, params, losses = train_nf(samples_center, contexts, learning_rate=1e-2, max_iter=max_iter, hidden_dims=[8], num_bins=20)
    #         losses = np.array(losses)
    #         nf_ci = ci_bisection(get_pvalue, beta_sd[0], beta_hat[0] + 5 * beta_sd[0], beta_hat[0] - 5 * beta_sd[0], sig_level=sig_level, tol=1e-4)

    #     nf_ci = pd.DataFrame(np.array(nf_ci).reshape(1, 2), index=['x1'])
    #     nf_pvalue = get_pvalue(0.)
    # else:
    #     def logdensity_fn(model, params, beta_hat, beta_null):
    #         beta_hat_center_ = cov_chol.T @ (beta_hat - mean_shift)
    #         # return -model.forward_kl(beta_hat_center_, params, beta_null)
    #         return -model.apply(params, beta_hat_center_, beta_null, method=model.forward_kl)
    #     # logdensity_fn = jax.jit(logdensity_fn, static_argnums=(0,))

    #     def train_and_inference(learning_rate, max_iter, n_layers):
    #         model, params, losses = train_nf(samples_center, contexts, learning_rate=learning_rate, max_iter=max_iter, hidden_dims=[2*d, 2*d], n_layers=n_layers)
    #         losses = np.array(losses)
    #         logp_fn = lambda beta_hat, beta_null: logdensity_fn(model, params, beta_hat, beta_null)
    #         nf_pvalue, nf_ci = inference(logp_fn, beta_hat, Sigma, sig_level=sig_level, compute_ci=True)
    #         nf_ci = pd.DataFrame(nf_ci, index=naive_ci.index)
    #         return nf_ci, nf_pvalue, losses
        
    #     print("Training with learning rate 1e-3, iteration=2000, n_layers=8")
    #     nf_ci, nf_pvalue, losses = train_and_inference(1e-4, 2000, 8)
    #     if np.any(np.isinf(np.array(nf_ci))):
    #         print("Got Inf, retraining with learning rate 1e-4, iteration=2000, n_layers=8")
    #         nf_ci, nf_pvalue, losses = train_and_inference(1e-4, 3000, 8)
    #         if np.any(np.isinf(np.array(nf_ci))):
    #             print("Got Inf, retraining with learning rate 1e-3, iteration=2000, n_layers=12")
    #             nf_ci, nf_pvalue, losses = train_and_inference(1e-3, 2000, 12)
    #             if np.any(np.isinf(np.array(nf_ci))):
    #                 print("Got Inf, retraining with learning rate 1e-4, iteration=3000, n_layers=12")
    #                 nf_ci, nf_pvalue, losses = train_and_inference(1e-4, 3000, 12)


    coverages_all = {}
    for key, ci in intervals_all.items():
        coverages_all[key] = (ci[:, 0] <= beta_target) & (beta_target <= ci[:, 1])
    print(coverages_all)
    print(pvalues_all)

    return {'coverages': coverages_all, 'pvalues': pvalues_all, 'intervals': intervals_all, 'global_pvalues': global_pvalues_all, 'losses': val_losses}
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='20250308')
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--signal_fac', type=float, default=1.)
    parser.add_argument('--mixed_sign', default=False, action='store_true')
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--rootdir', type=str, default='/mnt/ceph/users/sliu1/transport_selinf/')
    args = parser.parse_args()

    n = args.n
    p = 4 #5
    sigma = 1.
    rng = np.random.default_rng(2025)
    # beta = np.array([0., 0., 1., 2., -2.]) * args.signal_fac * np.sqrt(n)
    # x = np.linspace(0, 2, n)
    # if args.mixed_sign:
    #     beta = np.array([0, 0, -1, 1]) * args.signal_fac * np.sqrt(n)
    # else:
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
    results = run(seed, n_train=args.n_train, n_val=args.n_val, hidden_dim=args.hidden_dim)
    if results is not None:
        savepath = os.path.join(args.rootdir, args.date, 'poly')

        prefix = f'poly_{n}_{p}_signal_{args.signal_fac}_train_{args.n_train}_val_{args.n_val}_hidden_{args.hidden_dim}'
        path = os.path.join(savepath, prefix)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f'{seed}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f'Saved to {filename}')

