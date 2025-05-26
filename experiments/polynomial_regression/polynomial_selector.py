import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy.special import ndtr, ndtri
from experiments.selector import Selector

class PolynomialSelection(Selector):
    def __init__(self, X, y, sigma, nu=0, y_perturb=None):
        self.X = X
        self.df_X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
        self.y = y
        self.n = X.shape[0]
        self.p = X.shape[1] - 1
        self.sigma = sigma
        self.nu = nu
        if y_perturb is None:
            self.y_perturb = np.zeros(self.n)
        else:
            self.y_perturb = y_perturb

        self.selected_deg, self.selected_model = self.select(y + self.y_perturb)
        self.d = self.selected_deg
        self.X_E = X[:, :self.selected_deg+1]
        self.beta_hat = np.array(self.selected_model.params.iloc[1:])
        self.Sigma = np.linalg.inv(self.X_E.T @ self.X_E) * sigma**2
        self.Sigma = self.Sigma[1:, 1:]
        self.Sigma_sqrt = np.linalg.cholesky(self.Sigma)
    
    def select(self, y):
        df = self.df_X.copy()
        df['y'] = y 
        models = {}
        for deg in range(self.p+1):
            formula = 'y ~ ' + ' + '.join(df.columns[:deg+1]) + ' - 1'
            models[deg] = smf.ols(formula, data=df).fit()
        anova_results = anova_lm(*models.values())
        anova_results.index = np.arange(self.p+1)
        pvalues = anova_results['Pr(>F)'][1:]
        if len(pvalues.index[pvalues >= 0.05]) > 0:
            selected_deg = pvalues.index[pvalues >= 0.05].min() - 1
        else:
            selected_deg = self.p
        return selected_deg, models[selected_deg]

    def _resample(self, rng, beta_null):
        y = self.X_E[:, 1:] @ beta_null + rng.normal(size=(self.n, )) * self.sigma
        y_perturb = rng.normal(size=(self.n, )) * self.nu
        selected_deg, selected_model = self.select(y + y_perturb)
        if selected_deg == self.selected_deg:
            return np.array(selected_model.params.iloc[1:])
        else:
            return None

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
        y_indep = self.y - self.y_perturb * (self.sigma**2 / self.nu**2)
        beta_hat_indep = (np.linalg.pinv(self.X_E) @ y_indep)[1:]
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
    