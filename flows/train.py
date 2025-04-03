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

def train_with_validation(train_samples, train_contexts, val_samples, val_contexts, learning_rate, max_iter=5000, checkpoint_every=1000, hidden_dims=[8], n_layers=12, num_bins=20, seed=0):

    d = train_samples.shape[1]
    if d == 1:
        model = OneDSplineFlow(context_dim=1, hidden_dims=hidden_dims, num_bins=num_bins)
        if train_contexts is not None:
            params = model.init(jax.random.key(seed), jnp.ones((1, )), context=jnp.ones((1, 1)))
            val_samples = val_samples.flatten()
        else:
            params = model.init(jax.random.key(seed), jnp.ones((1, )), context=None)
        train_samples = train_samples.flatten()
    else:
        model = RealNVP(dim=d, n_layers=n_layers, hidden_dims=hidden_dims)
        if train_contexts is not None:
            params = model.init(jax.random.key(seed), jnp.ones((1, d)), context=jnp.ones((1, d)))
        else:
            params = model.init(jax.random.key(seed), jnp.ones((1, d)), context=None)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params):
        return model.apply(params, train_samples, context=train_contexts, method=model.forward_kl)
    
    @jax.jit
    def val_loss_fn(params):
        return model.apply(params, val_samples, context=val_contexts, method=model.forward_kl)
    
    @jax.jit
    def train_step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss
    
    def train_chunk(params, opt_state):
        carry, training_losses = jax.lax.scan(train_step, (params, opt_state), None, length=checkpoint_every)
        return carry

    n_chunks = max_iter // checkpoint_every
    val_losses = []
    min_val_loss = float('inf')
    best_params = params
    for n in range(n_chunks):
        params, opt_state = train_chunk(params, opt_state)
        val_loss = val_loss_fn(params)
        print("Iteration: ", (n + 1) * checkpoint_every, "Validation loss: ", val_loss)
        val_losses.append(val_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_params = params
        if val_loss > 1e10:
            break
    return model, best_params, val_losses
