import numpy as np
import argparse
import os
import pickle
from jax.scipy.stats import multivariate_normal as mvn
import jax
import jax.numpy as jnp 
from scipy.special import logsumexp

from experiments.lasso.randomized_lasso import RandomLassoCV
from experiments.lasso.regression_designs import gaussian_instance
from flows.train import train_nf
from utils.utils import ci_bisection

def inference(logdensity_fn, select_prob_fn, beta_hat, Sigma, compute_ci=True, sig_level=0.05):
    d = len(beta_hat)
    pvalues = np.zeros(d)
    cis = np.zeros((d, 2))
    for j in range(d):
        eta = np.eye(d)[j]
        c = Sigma @ eta / (np.dot(eta, Sigma @ eta))
        beta_perp = beta_hat - c * beta_hat[j]
        gridsize = 100
        sd_j = np.sqrt(Sigma[j, j])
        grid = jnp.linspace(-10, 10, gridsize) * sd_j + beta_hat[j]
        grids = np.outer(grid, c) + beta_perp
        select_prob = select_prob_fn(grids, j)

        @jax.jit
        def adjusted_logdensity(beta_hat_j, beta_null_j):
            beta_ = jnp.zeros(d)
            beta_ = beta_.at[j].set(beta_null_j)
            return logdensity_fn(beta_perp + c * beta_hat_j, beta_)

        def get_pvalue(beta_null_j):
            logp = jax.vmap(adjusted_logdensity, in_axes=(0, None))(grid, beta_null_j)
            logp = jnp.nan_to_num(logp)
            logp -= logp.max()
            log_normalization_const = logsumexp(logp, b=select_prob)

            idx_left = (grid <= beta_hat[j]) 
            log_numerator_left = logsumexp(logp[idx_left], b=select_prob[idx_left])
            pval = np.exp(log_numerator_left - log_normalization_const)
            return 2 * min(pval, 1 - pval)
    
        pvalues[j] = get_pvalue(0.)
        if compute_ci:
            cis[j] = ci_bisection(get_pvalue, sd_j, beta_hat[j] + 5 * sd_j, beta_hat[j] - 5 * sd_j, sig_level=sig_level, tol=1e-4)
    if compute_ci:
        return pvalues, cis
    return pvalues

def run(seed, signal_fac, nu, rho, n_train, max_iter, savepath=None):
    n = 100
    p = 5
    s = 0
    sigma = 1.
    signal = np.sqrt(signal_fac * 2 * np.log(p))
    equi = False
    random_signs = False
    rng = np.random.default_rng(seed)
    X, y, beta = gaussian_instance(rng, n, p, s, sigma, rho, signal, random_signs=random_signs, scale=True, center=True, equicorrelated=equi)
    w = nu * X.T @ rng.normal(size=(n, ))

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
    beta_sd = np.sqrt(np.diag(rl.Sigma))

    result_mle = rl.mle_inference(w=w, sig_level=sig_level)
    intervals_all['mle'] = np.array([result_mle['lower_confidence'], result_mle['upper_confidence']]).T
    pvalues_all['mle'] = np.array(result_mle['pvalue'])
    pvalues_all['naive'], intervals_all['naive'] = rl.naive_inference(sig_level=sig_level)

    def neg_loglik(beta_hat, beta_null):
        return -mvn.logpdf(beta_hat, mean=beta_null, cov=rl.Sigma)
    
    pvalues_all['preselect'] = rl.adjusted_inference(neg_loglik, compute_ci=False, sig_level=sig_level)

    print("Generating samples ...")
    samples, contexts = rl.generate_training_data(rng, n_train, resample_scale=1., max_try=100)
    print("Generated", samples.shape[0], 'samples')

    if samples.shape[0] == 0:
        print("Failed to generate training data")
        return
    
    mean_shift = np.mean(samples, axis=0)
    cov_chol = np.linalg.cholesky(np.linalg.inv(np.atleast_2d(np.cov(samples.T))))
    samples_center = (samples - mean_shift) @ cov_chol


    model, params, losses = train_nf(samples_center, contexts, learning_rate=1e-4, max_iter=max_iter, hidden_dims=[2*d, 2*d], n_layers=8)
    losses = np.array(losses)
    print(losses[-10:])
    print(np.std(losses[-100:]) / np.mean(losses[-100:]))
    # num_increases = np.sum(np.diff(losses[-100:]) > 0.)
    # if np.isnan(losses[-1]) or np.std(losses[-100:]) / np.mean(losses[-100:]) > 0.01 or num_increases > 10:
    #     print("Unstable training, try learning rate 1e-4")
    #     model, params, losses = train_nf(samples_center, contexts, learning_rate=1e-4, max_iter=max_iter, hidden_dims=[2*d, 2*d], n_layers=8)
    #     losses = np.array(losses)
    #     if np.isnan(losses[-1]) or np.std(losses[-100:]) / np.mean(losses[-100:]) > 0.01 or num_increases > 10:
    #         print("Unstable training!!!")

    def neg_loglik_adjusted(beta_hat, beta_null):
        beta_hat_center_ = cov_chol.T @ (beta_hat - mean_shift)
        return model.apply(params, beta_hat_center_, beta_null, method=model.forward_kl)
    pvalues_all['adjusted'] = rl.adjusted_inference(neg_loglik_adjusted, compute_ci=False, sig_level=sig_level)

    print(pvalues_all)

    false_rejects_all = {}
    for key, item in pvalues_all.items():
        false_rejects_all[key] = item < sig_level
    print(false_rejects_all)
    
    results_all = {'pvalues': pvalues_all}
    if savepath is not None:
        prefix = f'lassocv_{n}_{p}_{s}_{nu}_{signal_fac}_rho_{rho}_train_{n_train}_iter_{max_iter}'
        path = os.path.join(savepath, prefix)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f'{seed}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(results_all, f)
        print(f'Saved to {filename}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cv rlasso')
    parser.add_argument('--date', type=str, default='20250221')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--signal_fac', type=float, default=.6)
    parser.add_argument('--nu', type=float, default=.3)
    parser.add_argument('--rho', type=float, default=.5)
    parser.add_argument('--n_train', type=int, default=2000)
    parser.add_argument('--max_iter', type=int, default=3000)
    parser.add_argument('--rootdir', type=str, default='/mnt/ceph/users/sliu1/transport_selinf/')
    args = parser.parse_args()

    savepath = os.path.join(args.rootdir, args.date, 'lassocv')

    run(seed=args.seed, signal_fac=args.signal_fac, nu=args.nu, rho=args.rho, n_train=args.n_train, max_iter=args.max_iter, savepath=savepath)
