import numpy as np
from scipy.special import ndtri, ndtr
from sklearn.linear_model import LassoCV, Lasso
from scipy.stats import norm
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from selectinf.algorithms import lasso
from selectinf.randomized.lasso import lasso
from selectinf.base import selected_targets, full_targets

from sampling.cython_core import sample_sov
from utils.utils import ci_bisection
from experiments.selector import Selector

class RandomLasso(Selector):
    def __init__(self, X, y, sigma, lbd, nu=0., y_perturb=None) -> None:
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        self.sigma = sigma
        self.lbd = lbd

        self.nu = nu
        if y_perturb is None:
            self.y_perturb = np.zeros(self.n)
            self.w = np.zeros(self.p)
        else:
            self.y_perturb = y_perturb
            self.w = X.T @ y_perturb

        self.beta_sol = self.solve_lasso(y, self.y_perturb, self.lbd)
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
        self.proj_y = self.KX.T @ np.linalg.inv(self.KX @ self.KX.T)
        self.A1_obs = self.KX @ y
        self.A2_obs = self.KX @ self.y_perturb

    def solve_lasso(self, y, y_perturb, lbd=None):
        # if lbd is None:
        #     l1_norm = rr.weighted_l1norm([self.lbd] * self.p, lagrange=1.)
        # else:
        #     l1_norm = rr.weighted_l1norm([lbd] * self.p, lagrange=1.)
        # quad = rr.identity_quadratic(coef=0, center=0, linear_term=self.w)
        # loglike = rr.glm.gaussian(self.X, y, coef=1)
        # problem = rr.simple_problem(loglike, l1_norm)
        # solve_args={'tol': 1.e-12, 'min_its': 50}
        # beta_sol = problem.solve(quadratic=quad, **solve_args)
        # return beta_sol
        # debug
        if lbd is None:
            lbd = self.lbd
        return Lasso(alpha=lbd/self.n, fit_intercept=False).fit(self.X, y - y_perturb).coef_
        # assert np.allclose(beta_sol, beta_sol2), f"beta_sol: {beta_sol}, beta_sol2: {beta_sol2}"
        # return beta_sol

    def select_prob_hard_threshold(self, beta_hat, _):
        Sigma_ = np.linalg.inv(self.X_E.T @ self.X_E)
        scaled_s = self.s_E * (self.lbd * Sigma_ @ self.s_E)
        scaled_w = self.s_E * (Sigma_ @ self.w[self.E])
        beta_hat = np.atleast_2d(beta_hat)
        return np.all((beta_hat - scaled_s - scaled_w) * self.s_E > 0, axis=1) * 1.

    def select_prob_bivnormal(self, beta_hat, j):
        Sigma_ = np.linalg.inv(self.X_E.T @ self.X_E)
        scaled_s = self.s_E * (self.lbd * Sigma_ @ self.s_E)
        scaled_w = self.s_E * (Sigma_ @ self.w[self.E])
        D = np.diag(self.s_E)
        scaled_w_var = self.nu**2 * D @ Sigma_ @ D

        eta = np.eye(self.d)[j]
        w_var_j = np.dot(eta, scaled_w_var @ eta)
        c = scaled_w_var @ eta / w_var_j
        w_perp = scaled_w - c * scaled_w[j]
        
        beta_hat = np.atleast_2d(beta_hat)
        sel_probs = np.zeros(beta_hat.shape[0])
        for i in range(beta_hat.shape[0]):
            b = self.s_E * beta_hat[i] - scaled_s
            lb, ub = self.get_hard_threshold(-c, -(b - w_perp))
            sel_probs[i] = norm.cdf(ub / np.sqrt(w_var_j)) - norm.cdf(lb / np.sqrt(w_var_j))
        return sel_probs
    
    def select_prob_sov(self, beta_hat, _):
        Sigma_ = np.linalg.inv(self.X_E.T @ self.X_E)
        scaled_s = self.s_E * (self.lbd * Sigma_ @ self.s_E)
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
    
    def splitting_inference(self, sig_level=0.05, beta=None):
        y_indep = self.y + self.y_perturb * (self.sigma**2 / self.nu**2)
        beta_hat_indep = np.linalg.pinv(self.X_E) @ y_indep
        Sigma_indep = self.Sigma * (1 + self.sigma**2 / self.nu**2)
        sd = np.sqrt(np.diag(Sigma_indep))
        q = ndtri(sig_level / 2)
        lower = beta_hat_indep + q * sd
        upper = beta_hat_indep - q * sd
        if beta is None:
            pvals = 2 * ndtr(-abs(beta_hat_indep / sd))
        else:
            pvals = 2 * ndtr(-abs((beta_hat_indep - beta) / sd))
        return pvals, np.stack([lower, upper]).T
    
    def get_hard_threshold(self, a, b):
        """
        get intervals x \in [lb, ub] that corresponds to a * x - b > 0
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

    def adjusted_inference(self, neg_loglik, method_sel_prob, compute_ci=False, sig_level=0.05):
        d = self.d
        Sigma = self.Sigma
        beta_sd = np.sqrt(np.diag(Sigma))
        beta_hat = self.beta_hat
        cis = np.zeros((d, 2))
        pvalues = np.zeros(d)
        if method_sel_prob == 'hard_threshold':
            for j in range(d):
                eta = np.eye(d)[j]
                c = Sigma @ eta / np.dot(eta, Sigma @ eta)
                theta_hat = eta.dot(beta_hat)
                beta_perp = beta_hat - c * theta_hat
                a = c * self.s_E
                b = -(beta_perp - np.linalg.inv(self.X_E.T @ self.X_E) @ (self.lbd * self.s_E + self.w[self.E])) * self.s_E
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
                sd_j = np.sqrt(Sigma[j, j])
                grid = jnp.linspace(-10, 10, 200) * sd_j + beta_hat[j]
                grids = np.outer(grid, c) + beta_perp
                if method_sel_prob == 'sov':
                    select_prob = self.select_prob_sov(grids, j)
                elif method_sel_prob == 'bivnormal':
                    select_prob = self.select_prob_bivnormal(grids, j)
                else:
                    raise NotImplementedError

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
    def __init__(self, X, y, sigma, alphas, nfold=10, nu=0., y_perturb=None):
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        self.sigma = sigma
        self.alphas = alphas
        self.nfold = nfold
        self.nu = nu
        if y_perturb is None:
            self.y_perturb = np.zeros(self.n)
        else:
            self.y_perturb = y_perturb
        self.alpha = self.select_lambda(y - y_perturb)
        self.lbd = self.alpha * self.n

        super().__init__(X, y, sigma, lbd=self.lbd, nu=self.nu, y_perturb=y_perturb)

    def select_lambda(self, y):
        lasso_cv = LassoCV(alphas=self.alphas, fit_intercept=False, n_jobs=-1, cv=self.nfold, random_state=0)
        lasso_cv.fit(self.X, y)
        return lasso_cv.alpha_ 

    def _resample(self, rng, beta_null):
        y = self.X_E @ beta_null + rng.normal(size=(self.n, )) * self.sigma
        y = y - self.proj_y @ (self.KX @ y - self.A1_obs)

        y_perturb = self.nu * rng.normal(size=(self.n, ))
        y_perturb = y_perturb - self.proj_y @ (self.KX @ y_perturb - self.A2_obs)

        alpha_hat = self.select_lambda(y - y_perturb)
        if np.abs((alpha_hat - self.alpha) / self.alpha) <= 1e-2:
            return np.linalg.pinv(self.X_E) @ y
        else:
            return None
    