import jax
import jax.numpy as jnp
import distrax
from flax import linen as nn
from typing import Sequence, Callable

inverse_softplus = lambda x: jnp.log(jnp.exp(x) - 1.)

class ConditionerMLP(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_dims:
            x = self.activation(nn.Dense(h, 
                                         kernel_init=nn.initializers.variance_scaling(scale=0.1, mode="fan_in", distribution="normal"))(x))
        x = nn.Dense(
            2 * self.output_dim,
            # kernel_init=nn.initializers.zeros_init(),
            kernel_init=nn.initializers.variance_scaling(scale=0.01, mode="fan_in", distribution="truncated_normal"),
            bias_init=nn.initializers.zeros_init())(x)
        return x

class RealNVP(nn.Module):
    dim: int
    n_layers: int
    hidden_dims: Sequence[int]

    def setup(self):
        """
        Create:
         - self.masks: an array of binary masks, one per layer
         - self.conditioners: a list of MLP submodules, each with unique names
        """
        # 1) Build an alternating mask for each layer
        self.masks = [
            jnp.array([((i + j) % 2) == 0 for j in range(self.dim)], dtype=bool)
            for i in range(self.n_layers)
        ]

        # 2) Create a ConditionerMLP submodule for each layer
        self.conditioners = [
            ConditionerMLP(
                hidden_dims=self.hidden_dims, 
                output_dim=self.dim,
                name=f"conditioner_mlp_{i}"
            )
            for i in range(self.n_layers)
        ]

    @nn.compact
    def __call__(self, x, context=None, inverse=False):
        logdet = 0.
        for i in range(self.n_layers):
            mask = self.masks[i]
            conditioner_mlp = self.conditioners[i]
            
            def bijector_fn(params) -> distrax.Bijector:
                scale_logit, shift = jnp.split(params, 2, axis=-1)
                scale = jax.nn.softplus(scale_logit + inverse_softplus(1.))
                return distrax.ScalarAffine(shift=shift, scale=scale)
            
            def conditioner_fn(x_masked):
                if context is not None:
                    x_masked = jnp.concatenate([x_masked, context], axis=-1)
                return conditioner_mlp(x_masked)  
            
            bij = distrax.MaskedCoupling(
                mask=mask,
                conditioner=conditioner_fn,
                bijector=bijector_fn,
            )
            if not inverse:
                x, ld = bij.forward_and_log_det(x)
            else:
                x, ld = bij.inverse_and_log_det(x)

            logdet += ld

        return x, logdet

    def forward(self, x, context=None):
        return self(x, context, False)  
    
    def inverse(self, y, context=None):
        return self(y, context, True)

    def forward_kl(self, y, context=None, base_dist=None):
        x, log_det = self.inverse(y, context)
        if base_dist is None:
            base_dist = distrax.MultivariateNormalDiag(jnp.zeros(self.dim), jnp.ones(self.dim))
        log_q = base_dist.log_prob(x)
        logp = log_q + log_det
        logp = jnp.where(jnp.isinf(logp), jnp.nan, logp)
        return -jnp.nanmean(logp)

