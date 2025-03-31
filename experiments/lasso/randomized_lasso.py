from tqdm import trange
import numpy as np
from scipy.special import ndtri, ndtr
from sklearn.linear_model import Lasso, LassoCV
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import regreg.api as rr
from selectinf.algorithms import lasso
from selectinf.randomized.lasso import lasso
from selectinf.base import selected_targets, full_targets
from sampling.cython_core import sample_sov
from utils.utils import ci_bisection

from experiments.selector import Selector

class RandomLasso(Selector):
    """
    Lasso selector with fixed lbd
    X: design matrix
    y: response vector
    sigma: noise level
    lbd: fixed lasso penalty
    nu: randomization level
    w: randomization variable

    if nu = 0, then no randomization is used
    """
    def __init__(self, X, y, sigma, lbd, nu=0., w=None) -> None:
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        self.sigma = sigma
        self.lbd = lbd

        self.nu = nu
        if w is None:
            self.w = np.zeros(self.p)
        else:
            self.w = w

        self.beta_sol = self.solve_lasso(y, w, self.lbd)
        E = abs(self.beta_sol) > 1e-10
        self.s_E = np.sign(self.beta_sol[E])
        self.E = E
        self.d = E.sum()
        self.X_E = X[:, E]
        self.Sigma = sigma**2 * np.linalg.inv(self.X_E.T @ self.X_E)       
        self.Sigma_sqrt = np.linalg.cholesky(self.Sigma) 
        self.beta_hat = np.linalg.pinv(self.X_E) @ self.y

        self.K_E = np.zeros((self.p - self.d, self.p))
        self.K_E[:, E] = (self.X[:, np.logical_not(E)].T @ self.X[:, E]) @ np.linalg.inv(self.X[:, E].T @ self.X[:, E])
        self.K_E[:, np.logical_not(E)] = -np.eye(self.p - self.d)
        self.KX = self.K_E @ X.T
        self.u_obs = self.KX @ y
        self.proj_y = self.KX.T @ np.linalg.inv(self.KX @ self.KX.T)

    def solve_lasso(self, y, w, lbd=None):
        if lbd is None:
            l1_norm = rr.weighted_l1norm([self.lbd] * self.p, lagrange=1.)
        else:
            l1_norm = rr.weighted_l1norm([lbd] * self.p, lagrange=1.)
        quad = rr.identity_quadratic(coef=0, center=0, linear_term=w)
        loglike = rr.glm.gaussian(self.X, y, coef=1)
        problem = rr.simple_problem(loglike, l1_norm)
        solve_args={'tol': 1.e-12, 'min_its': 50}
        beta_sol = problem.solve(quadratic=quad, **solve_args)

        # debug
        # E = (beta_sol != 0)
        # s_E = np.sign(beta_sol[E])
        # beta_hat = np.linalg.pinv(self.X[:, E]) @ y

        # assert np.allclose(beta_sol[E], np.linalg.inv(self.X[:, E].T @ self.X[:, E]) @ (self.X[:, E].T @ y - self.lbd * s_E - w[E]))

        # assert np.allclose(np.sign(beta_hat - np.linalg.inv(self.X[:, E].T @ self.X[:, E]) @ (self.lbd * s_E + w[E])), s_E)
        # d = E.sum()
        # K_E = np.zeros((self.p - d, self.p))
        # K_E[:, E] = self.X[:, np.logical_not(E)].T @ self.X[:, E] @ np.linalg.inv(self.X[:, E].T @ self.X[:, E])
        # K_E[:, np.logical_not(E)] = -np.eye(self.p - d)
        # assert max(abs(K_E @ self.X.T @ y - K_E @ w - self.lbd * self.X[:, np.logical_not(E)].T @ self.X[:, E] @ np.linalg.inv(self.X[:, E].T @ self.X[:, E]) @ s_E)) <= self.lbd
        return beta_sol


    def select_prob_hard_threshold(self, beta_hat, _):
        """
        selection probability conditioned on beta_hat, and all randomization variables
        The probability is an indicator function
        """
        Sigma_ = np.linalg.inv(self.X_E.T @ self.X_E)
        scaled_s = self.s_E * (self.lbd * Sigma_ @ self.s_E)
        scaled_w = self.s_E * (Sigma_ @ self.w[self.E])
        beta_hat = np.atleast_2d(beta_hat)
        return np.all((beta_hat - scaled_s - scaled_w) * self.s_E > 0, axis=1) * 1.

    def select_prob_bivnormal(self, beta_hat, j):
        """
        selection probability conditioned on beta_hat=beta_hat_j, beta_hat^\perp, 
        and all the randomization variables except for ((X_E'X_E)^{-1} w)_j
        """
        Sigma_ = np.linalg.inv(self.X_E.T @ self.X_E)
        scaled_s = self.s_E * (self.lbd * Sigma_ @ self.s_E)
        scaled_w = self.s_E * (Sigma_ @ self.w[self.E])
        # s_j * beta_hat_j - scaled_s[j] > scaled_w[j]
        # scaled_w[j] | scaled_w[-j]
        D = np.diag(self.s_E)
        scaled_w_var = self.nu**2 * D @ Sigma_ @ D
        mask_j = np.ones(self.d, dtype=bool)
        mask_j[j] = False
        cond_var = scaled_w_var[j, j] - scaled_w_var[j, mask_j] @ np.linalg.inv(scaled_w_var[mask_j][:, mask_j]) @ scaled_w_var[mask_j, j]
        cond_mean = scaled_w_var[j, mask_j] @ np.linalg.inv(scaled_w_var[mask_j][:, mask_j]) @ scaled_w[mask_j] 
        sel_prob = norm.cdf((self.s_E[j] * np.atleast_2d(beta_hat)[:, j] - scaled_s[j]), loc=cond_mean, scale=np.sqrt(cond_var))
        indi = np.all((beta_hat[:, mask_j] - scaled_s[mask_j] - scaled_w[mask_j]) * self.s_E[mask_j] > 0, axis=1) * 1.
        return sel_prob * indi
    
    def select_prob_sov(self, beta_hat, _):
        """
        selection probability conditioned on beta_hat, beta_hat^\perp, w_E^\perp
        while marginalizing over w_E
        """
        Sigma_ = np.linalg.inv(self.X_E.T @ self.X_E)
        scaled_s = self.s_E * (self.lbd * Sigma_ @ self.s_E)
        # 
        D = np.diag(self.s_E)
        scaled_w_var = self.nu**2 * (D @ Sigma_ @ D)

        L = np.linalg.cholesky(scaled_w_var)
        a = np.ones(self.d) * (-np.inf)
        beta_hat = np.atleast_2d(beta_hat)
        sel_probs = np.zeros(beta_hat.shape[0])
        for i in range(beta_hat.shape[0]):
            b = self.s_E * beta_hat[i] - scaled_s
            weights = sample_sov(a, b, L, 512)[1]
            sel_probs[i] = np.mean(weights)
        return sel_probs

    def mle_inference(self, w, target='selected', sig_level=0.05):
        feature_weights_ = np.ones(self.p) * self.lbd
        selector = lasso.gaussian(self.X, self.y, feature_weights_, sigma=self.sigma, ridge_term=0.)
        signs = selector.fit(perturb=-w)
        nonzero = signs != 0
        if sum(nonzero) != self.d:
            raise AssertionError("different selection")

        selector.setup_inference(dispersion=self.sigma**2)
        if target == 'selected':
            target_spec = selected_targets(selector.loglike, selector.observed_soln, dispersion=self.sigma**2)
        elif target == 'full':
            target_spec = full_targets(selector.loglike, selector.observed_soln, nonzero, dispersion=self.sigma**2)
        else:
            raise NotImplementedError
        result = selector.inference(target_spec, 'selective_MLE', level=1-sig_level)
        return result

    def naive_inference(self, sig_level=0.05, beta=None):
        sd = np.sqrt(np.diag(self.Sigma))
        q = ndtri(sig_level / 2)
        lower = self.beta_hat + q * sd
        upper = self.beta_hat - q * sd
        if beta is None:
            pvals = 2 * ndtr(-abs(self.beta_hat / sd))
        else:
            pvals = 2 * ndtr(-abs((self.beta_hat - beta) / sd))
        return pvals, np.stack([lower, upper]).T
    
    def get_hard_threshold(self, a, b):
        """
        find {t: a * t - b > 0 }, where a, b \in \R^d
        """
        zero_idx = a == 0
        if sum(zero_idx) and b[zero_idx] < 0:
            return None
        pos_idx = a > 0
        if sum(pos_idx):
            lb = np.max(b[pos_idx] / a[pos_idx])
        else:
            lb = -np.inf
        neg_idx = a < 0
        if sum(neg_idx):
            ub = np.min(b[neg_idx] / a[neg_idx])
        else:
            ub = np.inf
        return lb, ub

    def adjusted_inference(self, neg_loglik, compute_ci=False, sig_level=0.05):
        d = self.d
        Sigma = self.Sigma
        beta_sd = np.sqrt(np.diag(Sigma))
        beta_hat = self.beta_hat
        cis = np.zeros((d, 2))
        pvalues = np.zeros(d)
        if self.nu == 0:
            for j in range(d):
                eta = np.eye(d)[j]
                c = Sigma @ eta / np.dot(eta, Sigma @ eta)
                theta_hat = eta.dot(beta_hat)
                beta_perp = beta_hat - c * theta_hat
                a = c * self.s_E
                b = -(beta_perp - self.lbd * np.linalg.inv(self.X_E.T @ self.X_E) @ self.s_E) * self.s_E
                lb, ub = self.get_hard_threshold(a, b)
                lb_finite = lb
                ub_finite = ub
                if np.isinf(lb):
                    lb_finite = -20 * beta_sd[j]
                if np.isinf(ub):
                    ub_finite = 20 * beta_sd[j]
                    
                def logp_j(beta_hat_j, beta_null_j):
                    beta_ = np.zeros(d)
                    beta_[j] = beta_null_j
                    return -neg_loglik(beta_perp + c * beta_hat_j, beta_)
                
                def _get_pvalue(beta_null_j):
                    beta_ = np.zeros(d)
                    beta_[j] = beta_null_j

                    logp1 = logp_j(lb_finite, beta_null_j)
                    logp2 = logp_j(ub_finite, beta_null_j)
                    logp3 = logp_j(beta_hat[j], beta_null_j)

                    _offset = np.nanmin([logp1, logp2, logp3])
                    if np.isnan(_offset):
                        return 0.
                    
                    grid = np.linspace(lb_finite, ub_finite, 200)
                    logp = jax.vmap(logp_j, in_axes=(0, None))(grid, beta_null_j)
                    logp = jnp.nan_to_num(logp, nan=-np.inf)
                    logp -= _offset
                    log_normalization_const = logsumexp(logp)
                    idx_left = (grid <= beta_hat[j]) 
                    log_numerator_left = logsumexp(logp[idx_left])
                    pval = np.exp(log_numerator_left - log_normalization_const)
                    return 2 * min(pval, 1 - pval)

                pvalues[j] = _get_pvalue(0.)
                if compute_ci:
                    cis[j] = ci_bisection(_get_pvalue, beta_sd[j], beta_hat[j] + 5 * beta_sd[j], beta_hat[j] - 5 * beta_sd[j], sig_level=sig_level, tol=1e-4)
        else:
            for j in range(d):
                eta = np.eye(d)[j]
                c = Sigma @ eta / (np.dot(eta, Sigma @ eta))
                beta_perp = beta_hat - c * beta_hat[j]
                gridsize = 200
                sd_j = np.sqrt(Sigma[j, j])
                grid = jnp.linspace(-10, 10, gridsize) * sd_j + beta_hat[j]
                grids = np.outer(grid, c) + beta_perp
                select_prob = self.select_prob_sov(grids, j)

                def logp_j(beta_hat_j, beta_null_j):
                    beta_ = np.zeros(d)
                    beta_[j] = beta_null_j
                    return -neg_loglik(beta_perp + c * beta_hat_j, beta_)

                def get_pvalue(beta_null_j):
                    logp = jax.vmap(logp_j, in_axes=(0, None))(grid, beta_null_j)
                    logp = jnp.nan_to_num(logp, nan=-np.inf)
                    logp -= logp.max()
                    log_normalization_const = logsumexp(logp, b=select_prob)

                    idx_left = (grid <= beta_hat[j]) 
                    log_numerator_left = logsumexp(logp[idx_left], b=select_prob[idx_left])
                    pval = np.exp(log_numerator_left - log_normalization_const)
                    return 2 * min(pval, 1 - pval)
            
                pvalues[j] = get_pvalue(0.)
                if compute_ci:
                    cis[j] = ci_bisection(get_pvalue, sd_j, beta_hat[j] + 3 * sd_j, beta_hat[j] - 3 * sd_j, sig_level=sig_level, tol=1e-4)

        if compute_ci:
            return pvalues, cis
        return pvalues

    
class RandomLassoCV(RandomLasso):
    def __init__(self, X, y, sigma, alphas, nfold=10, nu=0., w=None):

        
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        self.sigma = sigma
        self.alphas = alphas
        self.nfold = nfold

        self.nu = nu
        
        self.alpha = self.select_lambda(y)
        self.lbd = self.alpha * self.n

        super().__init__(X, y, sigma, lbd=self.lbd, nu=self.nu, w=w)

        # self.beta_sol = self.solve_lasso(y, w, self.lbd)
        # E = abs(self.beta_sol) > 1e-10
        # self.E = E
        # self.s_E = np.sign(self.beta_sol[E])
        # self.E = E
        # self.d = E.sum()
        # self.X_E = X[:, E]
        # self.Sigma = sigma**2 * np.linalg.inv(self.X_E.T @ self.X_E)        
        # self.beta_hat = np.linalg.pinv(self.X_E) @ self.y

        # self.K_E = np.zeros((self.p - self.d, self.p))
        # self.K_E[:, E] = (self.X[:, np.logical_not(E)].T @ self.X[:, E]) @ np.linalg.inv(self.X[:, E].T @ self.X[:, E])
        # self.K_E[:, np.logical_not(E)] = -np.eye(self.p - self.d)
        # self.KX = self.K_E @ X.T
        # self.u_obs = self.KX @ y
        # self.wperp_obs = self.K_E @ w
        # self.proj_y = self.KX.T @ np.linalg.inv(self.KX @ self.KX.T)
        # Omega = self.nu**2 * X.T @ X
        # self.proj_w = Omega @ self.K_E.T @ np.linalg.inv(self.K_E @ Omega @ self.K_E.T)
    
    def select_lambda(self, y):
        lasso_cv = LassoCV(alphas=self.alphas, fit_intercept=False, n_jobs=-1, cv=self.nfold, random_state=0)
        lasso_cv.fit(self.X, y)
        return lasso_cv.alpha_ 

    def select(self, y):
        return self.select_lambda(y)

    def _resample(self, rng, beta_null):
        y = self.X_E @ beta_null + rng.normal(size=(self.n, )) * self.sigma
        y = y - self.proj_y @ (self.KX @ y - self.u_obs)
        alpha_hat = self.select_lambda(y)
        if np.abs((alpha_hat - self.alpha) / self.alpha) <= 1e-2:
            return np.linalg.pinv(self.X_E) @ y
        else:
            return None
        
    # def _resample_condition_variable(self, rng, beta_null):
    #     y = self.X_E @ beta_null + rng.normal(size=(self.n, )) * self.sigma
    #     y = y - self.proj_y @ (self.KX @ y - self.u_obs)
    #     w = self.nu * self.X.T @ rng.normal(size=(self.n, ))
    #     w = w - self.proj_w @ (self.K_E @ w - self.wperp_obs)

    #     lbd_hat = self.select_lambda(y)
    #     E_hat = self.select(y, w, lbd=lbd_hat)
    #     if np.all(E_hat == self.E):
    #         return np.linalg.pinv(self.X_E) @ y
    #     else:
    #         return None

    # def resample(self, rng, beta_null, num_samples=1, max_try=1000, condition_on='lambda'):
    #     samples = []
    #     count = 0
    #     for _ in range(max_try):
    #         if condition_on == 'lambda':
    #             beta_hat = self._resample_condition_lambda(rng, beta_null)
    #         else:
    #             beta_hat = self._resample_condition_variable(rng, beta_null)
    #         if beta_hat is not None:
    #             samples.append(beta_hat)
    #             count += 1
    #             if count == num_samples:
    #                 break
    #     return np.array(samples)
    
    
class SelectLambda:
    def __init__(self, X, y, sigma, alphas):
        self.X = X
        self.y = y
        self.sigma = sigma
        self.n, self.p = X.shape
        self.alphas = alphas

        lasso_cv = LassoCV(alphas=self.alphas, fit_intercept=False, n_jobs=-1, random_state=0)
        lasso_cv.fit(self.X, y)
        self.alpha = lasso_cv.alpha_ 
        self.lbd = self.alpha * self.n

        beta_sol = lasso_cv.coef_
        E = np.abs(beta_sol) > 1e-10
        self.E = E
        self.d = E.sum()
        self.X_E = X[:, E]
        self.Sigma = sigma**2 * np.linalg.inv(self.X_E.T @ self.X_E)        
        self.beta_hat = np.linalg.pinv(self.X_E) @ self.y

        self.K_E = np.zeros((self.p - self.d, self.p))
        self.K_E[:, E] = (self.X[:, np.logical_not(E)].T @ self.X[:, E]) @ np.linalg.inv(self.X[:, E].T @ self.X[:, E])
        self.K_E[:, np.logical_not(E)] = -np.eye(self.p - self.d)
        self.KX = self.K_E @ X.T
        self.u_obs = self.KX @ y
        self.proj_y = self.KX.T @ np.linalg.inv(self.KX @ self.KX.T)
        
    def select(self, y):
        lasso_cv = LassoCV(alphas=self.alphas, fit_intercept=False, n_jobs=-1, random_state=0)
        lasso_cv.fit(self.X, y)
        return lasso_cv.alpha_ 
    
    def _resample(self, rng, beta_null):
        y = self.X_E @ beta_null + rng.normal(size=(self.n, )) * self.sigma
        y = y - self.proj_y @ (self.KX @ y - self.u_obs)
        # assert np.allclose(self.KX @ y, self.u_obs)

        # w = self.nu * self.X.T @ rng.normal(size=(self.n, ))
        # w = w - self.proj_w @ (self.K_E @ w - self.wperp_obs)
        # assert np.allclose(self.K_E @ w, self.wperp_obs)

        alpha_hat = self.select(y)
        if np.abs((alpha_hat - self.alpha) / self.alpha) <= 1e-2:
            return np.linalg.pinv(self.X_E) @ y
        else:
            return None

    def resample(self, rng, beta_null, num_samples=1, max_try=1000):
        samples = []
        count = 0
        for i in range(max_try):
            beta_hat = self._resample(rng, beta_null)
            if beta_hat is not None:
                samples.append(beta_hat)
                count += 1
                if count == num_samples:
                    break
        # print("generated", count, "samples", "out of", i)
        return np.array(samples)
    
    def naive_inference(self, sig_level=0.05, beta=None):
        sd = np.sqrt(np.diag(self.Sigma))
        q = ndtri(sig_level / 2)
        lower = self.beta_hat + q * sd
        upper = self.beta_hat - q * sd
        if beta is None:
            pvals = 2 * ndtr(-abs(self.beta_hat / sd))
        else:
            pvals = 2 * ndtr(-abs((self.beta_hat - beta) / sd))
        return pvals, np.stack([lower, upper]).T
    