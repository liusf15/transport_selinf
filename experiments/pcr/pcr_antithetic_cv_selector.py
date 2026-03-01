import numpy as np
import statsmodels.api as sm
from scipy.special import expit as sigmoid
from scipy.linalg import sqrtm

from experiments.selector import Selector
from scipy.optimize import minimize

class PCRAntitheticCVSelection(Selector):
    def __init__(self, X_PC, y, alpha_cv=0.1, K_cv=10):
        self.X_PC = X_PC
        self.y = y
        self.n = X_PC.shape[0]
        self.p = X_PC.shape[1]
        self.alpha_cv = alpha_cv
        self.K_cv = K_cv

        self.H_hat = self.estimate_score_cov()
        self.sqrtH = sqrtm(self.H_hat)

        self.best_k = self.select(self.X_PC.T @ y)
        self.d = self.best_k
        self.X_E = self.X_PC[:, :self.d]
        self.X_Ec = self.X_PC[:, self.d:]

        self.score = self.X_PC.T @ self.y
        self.beta_hat = self.logistic_regression_from_score(self.X_E, self.score[:self.d]) # mle in selected model
        logits = self.X_E @ self.beta_hat
        probs = sigmoid(logits)
        self.H_hat = self.X_PC.T @ np.diag(probs * (1 - probs)) @ self.X_PC # estimated H
        self.sqrtH = sqrtm(self.H_hat)

        self.K = np.zeros((self.p - self.d, self.p))
        self.K[:, :self.d] = -(self.X_Ec.T @ self.X_E) @ np.linalg.inv(self.X_E.T @ self.X_E) 
        self.K[:, self.d:] = np.eye(self.p - self.d)
        self.A_obs = self.K @ self.score
        self.proj = (self.H_hat @ self.K.T) @ np.linalg.inv(self.K @ self.H_hat @ self.K.T)

        self.Sigma = np.linalg.inv(self.H_hat)[:self.d, :self.d]
        self.Sigma_sqrt = np.linalg.cholesky(self.Sigma)

    def estimate_score_cov(self):
        logistic_model = sm.Logit(self.y, self.X_PC[:, :10]).fit(disp=False)
        logits = self.X_PC[:, :10] @ logistic_model.params
        probs = sigmoid(logits)
        return self.X_PC.T @ np.diag(probs * (1 - probs)) @ self.X_PC # estimated H

    def logistic_regression_from_score(self, X, score):
        l2_penalty = 1e-3
        d = X.shape[1]

        def nll(beta):
            return np.sum(np.logaddexp(0.0, X @ beta)) - np.dot(score, beta) + (l2_penalty / 2) * np.sum(beta**2) 

        def nll_grad(beta):
            return X.T @ sigmoid(X @ beta) - score + l2_penalty * beta

        res = minimize(nll, np.zeros(d), jac=nll_grad, method='BFGS')
        return res.x
    
    def antithetic_cv(self, X, s, sqrtH, alpha, K, rng):
        d = s.shape[0]
        Omega_bar = (1+1/(K-1))*np.eye(K-1) - np.ones((K-1,K-1))/(K-1)    
        W = rng.multivariate_normal(np.zeros(K-1), Omega_bar, size=d).T
        W = np.vstack([W, -np.sum(W, axis=0)[np.newaxis, :]])
        err = 0
        for k in range(K):
            wk = sqrtH[:d, :d] @ W[k]
            s_train = s + np.sqrt(alpha)*wk
            s_test = s - wk/np.sqrt(alpha)
            beta_hat = self.logistic_regression_from_score(X, s_train)
            err += np.sum(np.logaddexp(0.0, X @ beta_hat)) / self.n - np.sum(beta_hat * s_test) / self.n
        return err / K

    def select(self, score):
        rng = np.random.default_rng(0)
        best_cv_score = np.inf
        best_k = 0
        for k in range(1, 6):
            cv_score = self.antithetic_cv(self.X_PC[:, :k], score[:k], self.sqrtH, self.alpha_cv, self.K_cv, rng)
            if cv_score < best_cv_score:
                best_k = k
                best_cv_score = cv_score
        return best_k

    def _resample(self, rng, beta_null):
        logits = self.X_E @ beta_null
        y = rng.binomial(1, 1 / (1 + np.exp(-logits)), size=self.n)
        score = self.X_PC.T @ y
        score = score - self.proj @ (self.K @ score - self.A_obs)

        best_k = self.select(score)
        if best_k == self.best_k:
            return self.X_E.T @ y
        else:
            return None

    def naive_inference(self, sig_level=0.05):
        logistic_model = sm.Logit(self.y, self.X_E).fit(disp=False)
        cis = logistic_model.conf_int(alpha=sig_level)
        pvalues = logistic_model.pvalues
        llr_pvalue = logistic_model.llr_pvalue
        return cis, pvalues, llr_pvalue