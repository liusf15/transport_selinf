import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from experiments.selector import Selector
from scipy.special import expit as sigmoid

class PCRSelection(Selector):
    def __init__(self, X, y, n_fold=5):
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.n_fold = n_fold

        self.best_k, self.selected_model = self.select(X, y)
        self.d = self.best_k

        scaled_X = self.selected_model.named_steps['scale'].transform(self.X)
        self.X_PC = self.selected_model.named_steps['pca'].transform(scaled_X)
        self.suff_stat = self.X_PC.T @ self.y

        X_pcs_with_intercept = sm.add_constant(self.X_PC) 
        self.logistic_model = sm.Logit(self.y, X_pcs_with_intercept).fit(disp=False)
        self.intercept = self.logistic_model.params[0]
        self.beta_hat = self.logistic_model.params[1:]
        logits = X_pcs_with_intercept @ self.logistic_model.params
        probs = sigmoid(logits)
        self.cov_suff_stat = self.X_PC.T @ np.diag(probs * (1 - probs)) @ self.X_PC

        self.Sigma = self.logistic_model.cov_params()
        self.Sigma = self.Sigma[1:, 1:]
        self.Sigma_sqrt = np.linalg.cholesky(self.Sigma)

    def select(self, X, y):
        cv = StratifiedKFold(n_splits=self.n_fold, shuffle=False)
        intercept_only_score = cross_val_score(
            LogisticRegression(solver='lbfgs', max_iter=1000), 
            np.zeros((self.n, 1)),  
            y, 
            scoring='neg_log_loss', 
            cv=cv
        ).mean()

        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('pca', PCA()),
            ('logreg', LogisticRegression(solver='lbfgs', max_iter=1000))
        ])

        param_grid = {
            'pca__n_components': list(range(1, 5))  # trying 1 to 20 PCs
        }

        gridcv = GridSearchCV(pipe, param_grid, scoring='neg_log_loss', cv=cv)
        gridcv.fit(X, y)

        if np.max(gridcv.cv_results_['mean_test_score']) < intercept_only_score:
            best_k = 0
        else:
            best_k = gridcv.best_params_['pca__n_components']
        return best_k, gridcv.best_estimator_

    def _resample(self, rng, beta_null):
        logits = self.X_PC @ beta_null
        y = rng.binomial(1, 1 / (1 + np.exp(-logits)), size=self.n)
        best_k, _ = self.select(self.X, y)
        if best_k == self.best_k:
            return self.X_PC.T @ y
        else:
            return None

    def naive_inference(self, sig_level=0.05):
        
        cis = self.logistic_model.conf_int(alpha=sig_level)[1:]
        pvalues = self.logistic_model.pvalues[1:]
        llr_pvalue = self.logistic_model.llr_pvalue
        return cis, pvalues, llr_pvalue