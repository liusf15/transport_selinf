import numpy as np
import os
import argparse
import pandas as pd
from scipy.stats import chi2
from sklearn.preprocessing import SplineTransformer, StandardScaler

from experiments.spline.spline_selector import SplineSelection
from flows.train import train_with_validation


def generate_data(seed):
    rng = np.random.default_rng(seed)    
    return mu + rng.normal(size=(n,)) * sigma

def run(seed, n_train, n_val):
    y = generate_data(seed)

    selector = SplineSelection(x, y, sigma)
    d = selector.d
    naive_pval = selector.naive_F_test()

    print("Generating samples ...")
    rng = np.random.default_rng(0)
    train_samples = selector.sample_from_global_null(rng, n_train+n_val)
    print("Generated", train_samples.shape[0], 'samples')

    if train_samples.shape[0] == 0:
        print("Failed to generate training data")
        return

    mean_shift = np.mean(train_samples, axis=0)
    cov_chol = np.linalg.cholesky(np.linalg.inv(np.atleast_2d(np.cov(train_samples.T))))
    samples_center = (train_samples - mean_shift) @ cov_chol
    beta_hat_center = cov_chol.T @ (selector.beta_hat - mean_shift)

    def train_and_inference(seed, learning_rate, hidden_dim):
        model, params, val_losses = train_with_validation(samples_center[:n_train], None, samples_center[n_train:], None, learning_rate=learning_rate, max_iter=10000, checkpoint_every=1000, hidden_dims=[hidden_dim], n_layers=12, num_bins=20, seed=seed)
        z_value = model.apply(params, beta_hat_center, context=None, method=model.inverse)[0]
        pval = chi2.sf(np.sum(z_value**2), df=d)
        if np.isinf(z_value).any():
            return np.nan, val_losses
        return pval, val_losses
    
    learning_rate = 1e-4
    hidden_dim = max(8, 2*d)
    flag = False
    for _seed in range(3):
        print("Training seed: ", _seed, "lr: ", learning_rate)
        pval, val_losses = train_and_inference(seed=_seed, learning_rate=learning_rate, hidden_dim=hidden_dim)
        if np.isnan(val_losses[-1]) or (val_losses[-1] - val_losses[0] > 1e3) or (np.isnan(pval)):
            print("Training failed")
            continue
        else:
            print("Training succeeded")
            flag = True
            break
    
    if not flag:
        learning_rate = 1e-5
        for _seed in range(3, 6):
            print("Training seed: ", _seed, "lr: ", learning_rate)
            pval, val_losses = train_and_inference(seed=_seed, learning_rate=learning_rate, hidden_dim=hidden_dim)
            if np.isnan(val_losses[-1]) or (val_losses[-1] - val_losses[0] > 1e3) or (np.isnan(pval)):
                print("Training failed")
                continue
            else:
                print("Training succeeded")
                flag = True
                break
    
    if not flag:
        print("Training failed, setting pvalue to 2")
        pval = 2.

    pvalues_all = {'naive': naive_pval, 'adjusted': pval}
    print(pvalues_all)
    return pvalues_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='20250403')
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--signal_fac', type=float, default=1.)
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--rootdir', type=str, default='/mnt/ceph/users/sliu1/transport_selinf/')
    args = parser.parse_args()

    n = args.n
    sigma = 1.
    rng = np.random.default_rng(2025)
    x = rng.uniform(size=(n, 1))
    spline_transformer = SplineTransformer(n_knots=3, include_bias=False)
    X = spline_transformer.fit_transform(x)
    scalar_transformer = StandardScaler()
    X = scalar_transformer.fit_transform(X)
    beta = np.array([1, -1, 1, 1]) * args.signal_fac
    mu = X @ beta
    snr = np.sqrt(np.var(mu) / sigma**2)

    seed = args.seed
    results = run(args.seed, n_train=args.n_train, n_val=args.n_val)
    if results is not None:
        savepath = os.path.join(args.rootdir, args.date, 'spline')
        prefix = f'spline_{n}_signal_{args.signal_fac}_train_{args.n_train}_val_{args.n_val}'
        path = os.path.join(savepath, prefix)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f'{seed}.csv')
        pd.DataFrame(results, index=[0]).to_csv(filename)
        print(f'Saved to {filename}')
