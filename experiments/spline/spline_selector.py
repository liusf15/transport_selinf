import numpy as np
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from joblib import Parallel, delayed
from tqdm import tqdm

from experiments.selector import Selector

class SplineSelection(Selector):
    def __init__(self, x, y, sigma=1., maximum_knots=5):
        self.x = x
        self.y = y
        self.sigma = sigma
        self.maximum_knots = maximum_knots  
        self.n = x.shape[0]

        self.n_knots = self.select(y)
        spline_transformer = SplineTransformer(n_knots=self.n_knots, include_bias=False)
        X_basis = spline_transformer.fit_transform(x)
        scalar_transformer = StandardScaler()
        self.X = scalar_transformer.fit_transform(X_basis)
        self.X = sm.add_constant(self.X)
        self.selected_model = sm.OLS(y, self.X).fit()
        # X = sm.add_constant(x)
        self.beta_hat = np.array(self.selected_model.params[1:])
        self.d = len(self.beta_hat)
        self.intercept = self.selected_model.params[0]
    
    def select(self, y):
        pipe = make_pipeline(
            SplineTransformer(n_knots=2, knots='quantile', include_bias=False),
            LinearRegression(fit_intercept=True)
        )
        param_grid = {
            "splinetransformer__n_knots": np.arange(2, self.maximum_knots + 1),
        }
        grid = GridSearchCV(pipe, param_grid, scoring='neg_mean_squared_error', cv=5)
        grid.fit(self.x, y)

        n_knots = grid.best_params_['splinetransformer__n_knots']
        return n_knots

    def _resample(self, rng, beta_null):
        y = self.X[:, 1:] @ beta_null + self.intercept + rng.normal(size=(self.n, )) * self.sigma
        n_knots = self.select(y)
        if n_knots == self.n_knots:
            selected_model = sm.OLS(y, self.X).fit()
            return selected_model.params[1:]
        else:
            return None

    def naive_F_test(self):
        const_model = sm.OLS(self.y, np.ones((self.n, 1))).fit()
        _, F_pval, _ = self.selected_model.compare_f_test(const_model)
        return F_pval

    def sample_from_global_null(self, rng, n_train):
        beta_null = np.zeros(self.d)
        def _generator(seed):
            _rng = np.random.default_rng(seed)
            X = self.resample(_rng, beta_null, num_samples=1, max_try=100)
            if len(X) > 0:
                return X[0]
            return None

        seeds = rng.integers(low=0, high=2**32 - 1, size=n_train)
        samples = Parallel(n_jobs=-1)(
                delayed(_generator)(seed)
                for seed in tqdm(seeds)
            )
        return np.array([r for r in samples if r is not None])
