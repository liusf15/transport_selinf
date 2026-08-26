import numpy as np

def _design(p, rho, equicorrelated):
    # p is the feature count; rho controls either common or AR(1) correlation.
    if equicorrelated:
        Sigma = rho * np.ones([p, p]) + (1 - rho) * np.eye(p)
    else:
        Sigma = rho ** abs(np.arange(p).reshape(1, -1) - np.arange(p).reshape(-1, 1))
    return np.linalg.cholesky(Sigma)

def gaussian_instance(rng,
                      n=100,
                      p=200,
                      s=7,
                      sigma=1.,
                      rho=0.,
                      signal=7,
                      random_signs=False,
                      scale=True,
                      center=True,
                      equicorrelated=True):
    # n and p set the design shape; s is the number of nonzero coefficients.
    # signal sets their magnitude/range, while sigma scales signal and noise.
    # The remaining flags control signs and column preprocessing.
    # Correlate initially independent Gaussian design columns.
    chol = _design(p, rho, equicorrelated)
    X = rng.standard_normal((n, p)).dot(chol.T)

    if center:
        X -= X.mean(0, keepdims=True)

    # Populate the active coefficients before randomly assigning feature labels.
    beta = np.zeros(p) 
    signal = np.atleast_1d(signal)
    if signal.shape == (1,):
        beta[:s] = signal[0] 
    else:
        beta[:s] = np.linspace(signal[0], signal[1], s)
    if random_signs:
        beta[:s] *= (2 * rng.binomial(1, 0.5, size=(s,)) - 1.)
    rng.shuffle(beta)
    beta /= np.sqrt(n)

    # Normalize columns to unit Euclidean scale for comparable lasso penalties.
    if scale:
        scaling = X.std(0) * np.sqrt(n)
        X /= scaling[None, :]
        beta *= np.sqrt(n)

    noise = rng.standard_normal(n)
    
    # Apply sigma last so the returned coefficients match the response scale.
    Y = (X.dot(beta) + noise) * sigma
    return X, Y, beta * sigma
