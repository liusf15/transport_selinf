import jax
import jax.numpy as jnp
import jax.random as random

from flows.realnvp import RealNVP
from train.train import train

if __name__ == "__main__":
    
    key = random.PRNGKey(0)
    dim = 2
    n_layers = 8
    hidden_dims = [8, 8]

    model = RealNVP(dim=dim, n_layers=n_layers, hidden_dims=hidden_dims)
    params = model.init(random.PRNGKey(0), jnp.ones((1, dim)), context=jnp.ones((1, dim)))

    ntrain = 2000
    context = jax.random.uniform(key, (ntrain, dim))
    data = jax.random.normal(key, (ntrain, dim)) + context

    @jax.jit
    def loss_fn(params):
        return model.apply(params, data, context=context, method=model.forward_kl)
    
    print(loss_fn(params))
    # print(jax.grad(loss_fn)(params))
    optim_params, losses = train(model, params, data, context, learning_rate=1e-3, max_iter=2000)
    print(losses)
