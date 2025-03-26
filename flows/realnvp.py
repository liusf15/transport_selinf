import jax
import jax.numpy as jnp
import distrax
from flax import linen as nn
from typing import Sequence, Callable


class ConditionerMLP(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_dims:
            x = self.activation(nn.Dense(h)(x))
        return nn.Dense(2 * self.output_dim)(x)

class RealNVP(nn.Module):
    dim: int
    n_layers: int
    hidden_dims: Sequence[int]

    @nn.compact
    def _flow(self, x, inverse=False):
        bijectors = []
        for i in range(self.n_layers):
            mask = jnp.array([((i + j) % 2) == 0 for j in range(self.dim)], dtype=bool)

            conditioner_fn = ConditionerMLP(
                hidden_dims=self.hidden_dims, 
                output_dim=self.dim
            )

            def bijector_fn(params) -> distrax.Bijector:
                scale_logit, shift = jnp.split(params, 2, axis=-1)
                scale = jax.nn.softplus(scale_logit)
                return distrax.ScalarAffine(shift=shift, scale=scale)
            
            bij = distrax.MaskedCoupling(
                mask=mask,
                conditioner=conditioner_fn,
                bijector=bijector_fn,
            )
            bijectors.append(bij)

        chain = distrax.Chain(bijectors[::-1])
       
        if not inverse:
            return chain.forward_and_log_det(x)
        else:
            return chain.inverse_and_log_det(x)

    def forward(self, x):
        return self._flow(x, False)  
    
    def inverse(self, y):
        return self._flow(y, True)

    def forward_kl(self, y, base_dist=None):
        x, log_det = self.inverse(y)
        if base_dist is None:
            base_dist = distrax.MultivariateNormalDiag(jnp.zeros(self.dim), jnp.ones(self.dim))
        log_q = base_dist.log_prob(x)
        return -jnp.mean(log_q + log_det)

