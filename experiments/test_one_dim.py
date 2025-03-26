import jax
import jax.numpy as jnp
import jax.random as random

from flows.one_dim_flow import OneDSplineFlow
from train.train import train

if __name__ == "__main__":
    
    key = random.PRNGKey(0)
    dim = 1
    hidden_dims = [32]

    model = OneDSplineFlow(context_dim=1, hidden_dims=hidden_dims, num_bins=20)
    params = model.init(random.PRNGKey(0), jnp.ones((1, )), context=jnp.ones((1, 1)))

    ntrain = 2000
    context = jax.random.uniform(key, (ntrain, 1))
    data = jax.random.normal(key, (ntrain, )) + context.flatten()
    # context = None

    @jax.jit
    def loss_fn(params):
        return model.apply(params, data, context=context, method=model.forward_kl)
    
    print(loss_fn(params))
    # print(jax.grad(loss_fn)(params))
    optim_params, losses = train(model, params, data, context, learning_rate=1e-3, max_iter=2000)
    print(losses)
