import numpy as np
from scipy.special import ndtr, ndtri
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from experiments.selector import Selector

class PolynomialSelection(Selector):
    def __init__(self, df, sigma):
        self.df = df
        self.n = df.shape[0]
        self.p = df.shape[1] - 2
        self.sigma = sigma

        self.selected_deg, self.selected_model = self.select(df)
        self.d = self.selected_deg
        self.X_E = np.array(df.iloc[:, :self.selected_deg+1])
        self.beta_hat = np.array(self.selected_model.params.iloc[1:])
        self.Sigma = np.linalg.inv(self.X_E.T @ self.X_E) * sigma**2
        self.Sigma = self.Sigma[1:, 1:]
        self.Sigma_sqrt = np.linalg.cholesky(self.Sigma)
    
    def select(self, df):
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
        df_ = self.df.copy()
        df_['y'] = y
        selected_deg, selected_model = self.select(df_)
        if selected_deg == self.selected_deg:
            return np.array(selected_model.params.iloc[1:])
        else:
            return None
