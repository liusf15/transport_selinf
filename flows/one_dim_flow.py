import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Callable, Optional
import distrax

class SplineParamsMLP(nn.Module):
    hidden_dims: Sequence[int]
    num_bins: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x, context: Optional[jnp.ndarray] = None):
        if context is not None:
            x = jnp.concatenate([x.unsqueeze(), context.unsqueeze()], axis=-1)

        for h in self.hidden_dims:
            x = self.activation(nn.Dense(h)(x))

        out_dim = 3 * self.num_bins + 1
        x = nn.Dense(out_dim)(x)

        return x


class OneDSplineFlow(nn.Module):
    context_dim: int = 0
    hidden_dims: Sequence[int] = 1
    num_bins: int = 10
    range_min: float = -5.0
    range_max: float = 5.0

    def setup(self):
        if self.context_dim > 0:
            self.mlp = SplineParamsMLP(
                hidden_dims=self.hidden_dims,
                num_bins=self.num_bins,
                activation=nn.relu
            )
        else:
            param_shape = (3 * self.num_bins + 1,)
            self.unconditional_params = self.param(
                'uncond_spline_params',
                nn.initializers.zeros_init(),
                param_shape
            )

    @nn.compact
    def __call__(self, x: jnp.ndarray, context: Optional[jnp.ndarray] = None, inverse: bool = False):
        if self.context_dim > 0 and context is not None:
            raw_params = self.mlp(context)
        else:
            raw_params = jnp.broadcast_to(
                self.unconditional_params, 
                (x.shape[0], 3 * self.num_bins + 1)
            )

        spline = distrax.RationalQuadraticSpline(
            params=raw_params,
            range_min=self.range_min,
            range_max=self.range_max,
            boundary_slopes='unconstrained'
        )

        if not inverse:
            y, logdet = spline.forward_and_log_det(x)
        else:
            y, logdet = spline.inverse_and_log_det(x)

        return y, logdet

    def forward(self, x, context=None):
        return self(x, context=context, inverse=False)

    def inverse(self, z, context=None):
        return self(z, context=context, inverse=True)

    def forward_kl(self, y, context=None, base_dist=None):
        x, log_det = self.inverse(y, context)
        if base_dist is None:
            base_dist = distrax.Normal(0., 1.)
        log_q = base_dist.log_prob(x)
        logp = log_q + log_det
        logp = jnp.where(jnp.isinf(logp), jnp.nan, logp)
        return -jnp.nanmean(logp)
    