import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

class Selector:
    def __init__(self):
        pass

    def select(self):
        raise NotImplementedError
    
    def _resample(self, rng, beta_null):
        raise NotImplementedError
    
    def resample(self, rng, beta_null, num_samples=1, max_try=100):
        samples = []
        count = 0
        for i in range(max_try):
            beta_hat = self._resample(rng, beta_null)
            if beta_hat is not None:
                samples.append(beta_hat)
                count += 1
                if count == num_samples:
                    break
        return np.array(samples)
    
    def generate_training_data(self, rng, n_train, resample_scale=1., max_try=100):
        d = self.d
        def generator(seed):
            _rng = np.random.default_rng(seed)
            beta_null = self.Sigma_sqrt @ _rng.standard_normal(d) * resample_scale + self.beta_hat
            X = self.resample(_rng, beta_null, num_samples=1, max_try=max_try)
            if len(X) > 0:
                return X[0], beta_null
            return None, None
    
        seeds = rng.integers(low=0, high=2**32 - 1, size=n_train)
        results = Parallel(n_jobs=-1)(
                delayed(generator)(seed)
                for seed in tqdm(seeds)
            )

        samples = np.array([r[0] for r in results if r[0] is not None])
        contexts = np.array([r[1] for r in results if r[0] is not None])
        return samples, contexts
