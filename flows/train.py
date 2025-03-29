import jax
import jax.numpy as jnp
import optax

from flows.realnvp import RealNVP
from flows.one_dim_flow import OneDSplineFlow

def train(model, params, samples, contexts, learning_rate=0.01, max_iter=500):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params):
        return model.apply(params, samples, context=contexts, method=model.forward_kl)
    
    @jax.jit
    def train_step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    init_carry = (params, opt_state)
    carry, losses = jax.lax.scan(train_step, init_carry, None, length=max_iter)
    params, opt_state = carry
    losses = list(losses)
    return params, losses


def train_nf(samples, contexts, learning_rate, max_iter, hidden_dims=[32, 32], n_layers=8, num_bins=20):
    d = samples.shape[1]
    if d == 1:
        model = OneDSplineFlow(context_dim=1, hidden_dims=hidden_dims, num_bins=num_bins)
        params = model.init(jax.random.key(0), jnp.ones((1, )), context=jnp.ones((1, 1)))
        samples = samples.flatten()
    else:
        model = RealNVP(dim=d, n_layers=n_layers, hidden_dims=hidden_dims)
        params = model.init(jax.random.key(0), jnp.ones((1, d)), context=jnp.ones((1, d)))

    params, losses = train(model, params, samples, contexts, learning_rate=learning_rate, max_iter=max_iter)
    return model, params, losses
