import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Callable, Optional
import distrax

class SplineParamsMLP(nn.Module):
    """
    A small MLP to output parameters for a 1D Rational Quadratic Spline.
    Typically, we need `3 * num_bins - 1` parameters per spline:
      - `num_bins` for widths
      - `num_bins` for heights
      - `num_bins - 1` for derivatives (or boundary slopes)
    We'll do that for each input dimension (which is 1D here) = total needed is
      `3 * num_bins - 1`
    If we use context, we concatenate it with x before the feedforward.
    """
    hidden_dims: Sequence[int]
    num_bins: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x, context: Optional[jnp.ndarray] = None):
        """
        x: shape (batch, 1) for 1D
        context: optional shape (batch, context_dim)
        Output shape: (batch, 3 * num_bins + 1)
        """
        if context is not None:
            # Concatenate x and context along the last axis
            x = jnp.concatenate([x.unsqueeze(), context.unsqueeze()], axis=-1)

        # Pass through hidden layers
        for h in self.hidden_dims:
            x = self.activation(nn.Dense(h)(x))

        # Final layer: outputs the parameters for 1D RQS
        out_dim = 3 * self.num_bins + 1
        x = nn.Dense(out_dim)(x)

        return x


class OneDSplineFlow(nn.Module):
    """
    A chain of 1D spline transformations, each parameterized by a SplineParamsMLP.
    - dim is implicitly 1, but we keep 'dim' for clarity if you wanted to extend to multi-1D flows.
    - n_layers: how many spline transformations to chain.
    - hidden_dims: MLP hidden layers
    - num_bins: how many bins for each rational-quadratic spline
    - range_min, range_max: domain for the spline
    """
    context_dim: int = 0
    hidden_dims: Sequence[int] = 1
    num_bins: int = 10
    range_min: float = -5.0
    range_max: float = 5.0

    def setup(self):
        if self.context_dim > 0:
            # Conditional: build an MLP that produces (3*num_bins + 1) params from context
            self.mlp = SplineParamsMLP(
                hidden_dims=self.hidden_dims,
                num_bins=self.num_bins,
                activation=nn.relu
            )
        else:
            # Unconditional: define a single trainable param vector
            # shape (3*num_bins + 1,)
            param_shape = (3 * self.num_bins + 1,)
            # We'll store it as a parameter in setup (using self.param)
            # Initialize to zeros or something small so it starts near identity
            self.unconditional_params = self.param(
                'uncond_spline_params',
                nn.initializers.zeros_init(),
                param_shape
            )

    @nn.compact
    def __call__(self, x: jnp.ndarray, context: Optional[jnp.ndarray] = None, inverse: bool = False):
        """
        If `inverse=False`: forward pass x -> z.
        If `inverse=True`:  inverse pass z -> x.
        Returns (output, logdet).
        """
        if self.context_dim > 0 and context is not None:
            # shape (batch, 3*num_bins+1)
            raw_params = self.mlp(context)
        else:
            # shape (3*num_bins+1,) => we broadcast to match batch dimension of x
            raw_params = jnp.broadcast_to(
                self.unconditional_params, 
                (x.shape[0], 3 * self.num_bins + 1)
            )

        # 2) Build the RQS bijector
        #    boundary_slopes='unconstrained' => no boundary slope constraints
        spline = distrax.RationalQuadraticSpline(
            params=raw_params,
            range_min=self.range_min,
            range_max=self.range_max,
            boundary_slopes='unconstrained'
        )

        # 3) Forward or inverse pass
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
            # base_dist = distrax.MultivariateNormalDiag(jnp.zeros(1), jnp.ones(1))
        log_q = base_dist.log_prob(x)
        logp = log_q + log_det
        logp = jnp.where(jnp.isinf(logp), jnp.nan, logp)
        return -jnp.nanmean(logp)
    