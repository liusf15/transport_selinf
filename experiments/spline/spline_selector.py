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
    # Select spline complexity by cross-validation and expose conditional resampling.
    def __init__(self, x, y, sigma=1., maximum_knots=5, n_fold=10, scale=False, nu=0., y_perturb=None):
        # x/y are observed data; sigma and nu are response/randomization scales.
        # maximum_knots and n_fold define the CV search; scale standardizes bases.
        self.x = x
        self.y = y
        self.sigma = sigma
        self.nu = nu
        self.maximum_knots = maximum_knots  
        self.n_fold = n_fold
        self.n = x.shape[0]
        if y_perturb is None:
            self.y_perturb = np.zeros(y.shape)
        else:
            self.y_perturb = y_perturb
        
        # Select on the randomized response, then fit the target on the original one.
        self.n_knots = self.select(y + self.y_perturb)
        spline_transformer = SplineTransformer(n_knots=self.n_knots, include_bias=False)
        X_basis = spline_transformer.fit_transform(x)
        if scale:
            scalar_transformer = StandardScaler()
            self.X = scalar_transformer.fit_transform(X_basis)
        else:
            self.X = X_basis
        self.X = sm.add_constant(self.X)
        # Exclude the intercept from the statistic transported downstream.
        self.selected_model = sm.OLS(y, self.X).fit()
        self.beta_hat = np.array(self.selected_model.params[1:])
        self.d = len(self.beta_hat)
        self.intercept = self.selected_model.params[0]

    @classmethod
    def for_global_null_event(
        cls,
        x,
        selected_n_knots,
        sigma=1.0,
        maximum_knots=5,
        n_fold=10,
        scale=False,
        nu=0.0,
    ):
        """Build a sampler for one fixed knot-selection event under the null.

        Unlike the regular constructor, this factory does not need an observed
        response or rerun model selection. It is used to calibrate the finite
        collection of global-null laws indexed by the possible knot counts.
        """
        if selected_n_knots < 2 or selected_n_knots > maximum_knots:
            raise ValueError(
                "selected_n_knots must be between 2 and maximum_knots"
            )

        # Construct only the state needed for null resampling, bypassing observed CV.
        selector = cls.__new__(cls)
        selector.x = x
        selector.y = np.zeros(x.shape[0])
        selector.sigma = sigma
        selector.nu = nu
        selector.maximum_knots = maximum_knots
        selector.n_fold = n_fold
        selector.n = x.shape[0]
        selector.y_perturb = np.zeros(x.shape[0])
        selector.n_knots = selected_n_knots

        spline_transformer = SplineTransformer(
            n_knots=selected_n_knots,
            include_bias=False,
        )
        X_basis = spline_transformer.fit_transform(x)
        if scale:
            X_basis = StandardScaler().fit_transform(X_basis)
        selector.X = sm.add_constant(X_basis)
        selector.d = selector.X.shape[1] - 1
        selector.intercept = 0.0
        selector.beta_hat = np.zeros(selector.d)
        selector.selected_model = None
        return selector

    def select(self, y):
        # y is the response whose prediction error chooses the number of knots.
        pipe = make_pipeline(
            SplineTransformer(n_knots=2, knots='quantile', include_bias=False),
            LinearRegression(fit_intercept=True)
        )
        param_grid = {
            "splinetransformer__n_knots": np.arange(2, self.maximum_knots + 1),
        }
        # Refit every candidate across the shared CV folds and keep the best score.
        grid = GridSearchCV(pipe, param_grid, scoring='neg_mean_squared_error', cv=self.n_fold)
        grid.fit(self.x, y)

        n_knots = grid.best_params_['splinetransformer__n_knots']
        return n_knots

    def _resample(self, rng, beta_null):
        # Simulate at beta_null and accept only responses selecting the same knot count.
        y = self.X[:, 1:] @ beta_null + self.intercept + rng.normal(size=(self.n, )) * self.sigma
        y_perturb = rng.normal(size=(self.n, )) * self.nu
        n_knots = self.select(y + y_perturb)
        if n_knots == self.n_knots:
            selected_model = sm.OLS(y, self.X).fit()
            return selected_model.params[1:]
        else:
            return None

    def naive_F_test(self):
        # Compare the selected spline to an intercept-only model without adjustment.
        const_model = sm.OLS(self.y, np.ones((self.n, 1))).fit()
        _, F_pval, _ = self.selected_model.compare_f_test(const_model)
        return F_pval
    
    def splitting_F_test(self):
        # Remove the selection-correlated randomization before the classical F test.
        y_indep = self.y - self.y_perturb * (self.sigma**2 / self.nu**2)
        null_model = sm.OLS(y_indep, np.ones((self.n, 1))).fit()
        alt_model = sm.OLS(y_indep, self.X).fit()
        _, F_pval, _ = alt_model.compare_f_test(null_model)
        return F_pval

    def sample_from_global_null(self, rng, n_train, return_num_tries=False):
        """Sample coefficient estimates conditional on the observed knot count.

        Parameters
        ----------
        rng : numpy.random.Generator
            Generator used to create an independent seed for each rejection sampler.
        n_train : int
            Number of accepted conditional draws to return. Rare exhausted
            rejection samplers are replaced with fresh independent attempts.
        return_num_tries : bool, default=False
            If true, also return the number of rejection-sampling attempts used by
            each accepted draw.

        Returns
        -------
        samples : numpy.ndarray
            Accepted coefficient estimates with shape ``(n_accepted, self.d)``.
        num_tries : numpy.ndarray, optional
            Attempt counts aligned row-for-row with ``samples``.
        """
        beta_null = np.zeros(self.d)

        def _generator(seed):
            _rng = np.random.default_rng(seed)
            if not return_num_tries:
                X = self.resample(_rng, beta_null, num_samples=1, max_try=100)
                if len(X) > 0:
                    return X[0]
                return None
            else:
                X, num_tries = self.resample(
                    _rng,
                    beta_null,
                    num_samples=1,
                    max_try=100,
                    return_num_tries=True,
                )
                if len(X) > 0:
                    return X[0], num_tries
                return None, num_tries

        # Retry failed rejection-sampling jobs in batches until exactly n_train accept.
        samples = []
        rounds = 0
        while len(samples) < n_train:
            rounds += 1
            if rounds > 10:
                raise RuntimeError(
                    f"failed to obtain {n_train} accepted samples after 10 batches"
                )
            num_missing = n_train - len(samples)
            seeds = rng.integers(low=0, high=2**32 - 1, size=num_missing)
            batch = Parallel(n_jobs=-1)(
                delayed(_generator)(seed)
                for seed in tqdm(seeds)
            )
            if return_num_tries:
                samples.extend(result for result in batch if result[0] is not None)
            else:
                samples.extend(result for result in batch if result is not None)

        if not return_num_tries:
            return np.asarray(samples)

        # Separate accepted estimates from their aligned proposal counts.
        accepted_samples = np.asarray([result[0] for result in samples])
        num_tries = np.asarray([result[1] for result in samples], dtype=int)
        return accepted_samples, num_tries
