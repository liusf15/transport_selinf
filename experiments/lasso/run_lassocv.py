import numpy as np
import argparse
import os
import pickle
from jax.scipy.stats import multivariate_normal as mvn
import time
import jax
import jax.numpy as jnp 
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.special import logsumexp
from selectinf.randomized.lasso import lasso
from selectinf.base import selected_targets

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

def run(seed, signal_fac, nu, n_train, max_iter, savepath=None):
    n = 100
    p = 5
    s = 0
    sigma = 1.
    rho = 0.5
    signal_fac = 1.5
    signal = np.sqrt(signal_fac * 2 * np.log(p))
    equi = False
    random_signs = False
    rng = np.random.default_rng(seed)
    X, y, beta = gaussian_instance(rng, n, p, s, sigma, rho, signal, random_signs=random_signs, scale=True, center=True, equicorrelated=equi)
    w = nu * X.T @ rng.normal(size=(n, ))

    alphas = np.logspace(-2, np.log10(5), 10) * np.sqrt(np.log(p)) / n
    rl = RandomLassoCV(X, y, sigma, alphas, nu=nu, w=w)
    d = rl.d
    print("selected", d, "variables")
    if d == 0:
        return

    sig_level = 0.05
    intervals_all = {}
    pvalues_all = {}

    beta_target = np.linalg.pinv(rl.X_E) @ X @ beta
    # result_mle = rl.mle_inference(w=w, sig_level=sig_level)
    # intervals_all['mle'] = np.array([result_mle['lower_confidence'], result_mle['upper_confidence']]).T
    # pvalues_all['mle'] = np.array(result_mle['pvalue'])

    # feature_weights_ = np.ones(p) * rl.lbd
    # selector = lasso.gaussian(X, y, feature_weights_, sigma=sigma, ridge_term=0.)
    # signs = selector.fit(perturb=-w)
    # nonzero = signs != 0
    # if nonzero.sum() != d:
    #     print("different selection")
    #     return
    # selector.setup_inference(dispersion=sigma**2)
    # target_spec = selected_targets(selector.loglike, selector.observed_soln, dispersion=sigma**2)
    # result_mle = selector.inference(target_spec, 'selective_MLE', level=1-sig_level)

    result_mle = rl.mle_inference(w=w, sig_level=sig_level)
    intervals_all['mle'] = np.array([result_mle['lower_confidence'], result_mle['upper_confidence']]).T
    pvalues_all['mle'] = np.array(result_mle['pvalue'])

    # result_exact = selector.inference(target_spec, 'exact', level=1-sig_level)
    # intervals_all['exact'] = np.array([result_exact['lower_confidence'], result_exact['upper_confidence']]).T
    # pvalues_all['exact'] = np.array(result_exact['pvalue'])
    pvalues_all['naive'], intervals_all['naive'] = rl.naive_inference(sig_level=sig_level)

    beta_sd = np.sqrt(np.diag(rl.Sigma))

    print("Generating samples ...")
    # def generate_one_sample(seed):
    #     _rng = np.random.default_rng(seed)
    #     beta_null = _rng.standard_normal(d) * beta_sd + rl.beta_hat
    #     X = rl.resample(_rng, beta_null, num_samples=1, max_try=100)
    #     if len(X) > 0:
    #         return X[0], beta_null
    #     return None, None

    # start = time.time()
    # seeds = rng.integers(low=0, high=2**32 - 1, size=n_train)
    # results = Parallel(n_jobs=-1)(
    #         delayed(generate_one_sample)(seed)
    #         for seed in tqdm(seeds)
    #     )
    # end = time.time()

    # samples = np.array([r[0] for r in results if r[0] is not None])
    # contexts = np.array([r[1] for r in results if r[0] is not None])
    samples, contexts = rl.generate_training_data(rng, n_train, resample_scale=1., max_try=100)
    print("Generated", samples.shape[0], 'samples')

    if samples.shape[0] == 0:
        print("Failed to generate training data")
        return
    
    mean_shift = np.mean(samples, axis=0)
    cov_chol = np.linalg.cholesky(np.linalg.inv(np.atleast_2d(np.cov(samples.T))))
    samples_center = (samples - mean_shift) @ cov_chol
    model, params, losses = train_nf(samples_center, contexts, learning_rate=1e-3, max_iter=max_iter)
    losses = np.array(losses)
    if np.isnan(losses[-1]) or np.std(losses[-100:]) / np.mean(losses[-100:]) > 0.1:
        print("Unstable training, try learning rate 1e-4")
        model, params, losses = train_nf(samples_center, contexts, learning_rate=1e-4, max_iter=max_iter)
    
    def preselect_logdensity(beta_hat, beta_null):
        return mvn.logpdf(beta_hat, mean=beta_null, cov=rl.Sigma)

    @jax.jit
    def adjusted_logdensity_fn(beta_hat, beta_null):
        beta_hat_center_ = cov_chol.T @ (beta_hat - mean_shift)
        # return -model.forward_kl(beta_hat_center_, params, beta_null)
        return -model.apply(params, beta_hat_center_, beta_null, method=model.forward_kl)

    def get_logdensity_fn(adjust):
        if adjust == 'adjusted':
            return adjusted_logdensity_fn
        elif adjust == 'preselect':
            return preselect_logdensity
        else:
            raise ValueError(f"Invalid method {method}")
    
    def get_select_prob_fn(method):
        if method == 'biv':
            return rl.select_prob_bivnormal
        elif method == 'hard_threshold':
            return rl.select_prob_hard_threshold
        elif method == 'sov':
            return rl.select_prob_sov
        else:
            raise ValueError(f"Invalid method {method}")
    
    for adjust in ['adjusted', 'preselect']:
        logdensity_fn = get_logdensity_fn(adjust)
        for method in ['sov']:
            select_prob_fn = get_select_prob_fn(method)
            pvalues = inference(logdensity_fn, select_prob_fn, rl.beta_hat, rl.Sigma, compute_ci=False, sig_level=sig_level)
            # intervals_all[adjust + '_' + method] = ci
            pvalues_all[adjust + '_' + method] = pvalues

    # pvalues_biv, ci_biv = inference(logdensity_fn, rl.select_prob_bivnormal, rl.beta_hat, rl.Sigma, sig_level)
    # pvalues_hardthre, ci_hardthre = inference(logdensity_fn, rl.select_prob_hard_threshold, rl.beta_hat, rl.Sigma, sig_level)
    # pvalues_sov, ci_sov = inference(logdensity_fn, rl.select_prob_sov, rl.beta_hat, rl.Sigma, sig_level)

    # coverages_all = {}
    # lengths_all = {}
    false_rejects_all = {}
    for key, item in pvalues_all.items():
        false_rejects_all[key] = item < sig_level
        # coverages_all[key] = (interval[:, 0] <= beta_target) * (interval[:, 1] >= beta_target)
    #     lengths_all[key] = interval[:, 1] - interval[:, 0]
    print(false_rejects_all)
    
    results_all = {'pvalues': pvalues_all}
    if savepath is not None:
        prefix = f'cv_rlasso_sov_{n}_{p}_{s}_{nu}_{signal_fac}_train_{n_train}_iter_{max_iter}'
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
    parser.add_argument('--n_train', type=int, default=2000)
    parser.add_argument('--max_iter', type=int, default=3000)
    parser.add_argument('--rootdir', type=str, default='/mnt/ceph/users/sliu1/transport_selinf/')
    args = parser.parse_args()

    savepath = os.path.join(args.rootdir, args.date, 'cv_rlasso')

    run(seed=args.seed, signal_fac=args.signal_fac, nu=args.nu, n_train=args.n_train, max_iter=args.max_iter, savepath=savepath)
