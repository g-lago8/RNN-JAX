import warnings

import jax
import jax.numpy as jnp
from jax import random as jr
import equinox as eqx
from typing import Sequence, Optional
from jaxtyping import  Array
from rnn_jax.ssm.base import BaseSSMLayer
from rnn_jax.ssm.mixers import Mixer


class DeepStateSpaceModelEncoder(eqx.Module):
    n_layers : int
    in_dim : int
    state_dim : Sequence[int]
    model_dim : Sequence[int]
    layers : Sequence[BaseSSMLayer]
    mixers : Sequence[Mixer]
    in_projection : eqx.nn.Linear | eqx.nn.Identity
    out_projection : eqx.nn.Linear | eqx.nn.Identity
    pool_method : str

    def __init__(
        self,
        in_dim: int,
        layers: Sequence[BaseSSMLayer],
        mixers: Sequence[Mixer],
        in_projection: bool = True,
        out_dim: Optional[int] = None,
        pool_method: str = 'none',
        *,
        key: Array
    ):
        assert len(mixers) == len(layers), "number of layers and mixers must be the same"
        self.n_layers = len(layers)
        self.in_dim = in_dim
        self.layers = layers
        self.mixers = mixers
        self.state_dim = [m.state_dim for m in layers]
        self.model_dim = [m.model_dim for m in layers]
        self.pool_method = pool_method
        in_key, out_key = jr.split(key)
        if in_projection:
            projected_dim = self.state_dim[0]
            self.in_projection = eqx.nn.Linear(in_features=in_dim, out_features=projected_dim, key=in_key)
            # After projection the first layer receives vectors of size projected_dim.
            # Its W_in must match, i.e. layer.in_dim == projected_dim.
            first_layer_in = layers[0].in_dim
            if first_layer_in != projected_dim:
                raise ValueError(
                    f"in_projection maps inputs from {in_dim} to {projected_dim} "
                    f"(state_dim of the first layer), but the first layer expects "
                    f"in_dim={first_layer_in}. Create the first layer with "
                    f"in_dim={projected_dim} when using in_projection=True."
                )
        else:
            self.in_projection = eqx.nn.Identity()
            if layers[0].in_dim != in_dim:
                raise ValueError(
                    f"in_projection is disabled but in_dim ({in_dim}) != "
                    f"first layer's in_dim ({layers[0].in_dim}). "
                    f"Either set in_projection=True or create the first layer "
                    f"with in_dim={in_dim}."
                )

        # Validate inter-layer dimension compatibility.
        for i in range(len(layers) - 1):
            out_i = self.model_dim[i]
            in_next = layers[i + 1].in_dim
            if out_i != in_next:
                raise ValueError(
                    f"Dimension mismatch between layer {i} and layer {i+1}: "
                    f"layer {i} outputs model_dim={out_i} but layer {i+1} "
                    f"expects in_dim={in_next}."
                )

        if out_dim is not None:
            self.out_projection = eqx.nn.Linear(in_features=self.model_dim[-1], out_features=out_dim, key=out_key)
        else:
            self.out_projection = eqx.nn.Identity()

    def __call__(self, xs:Array) -> Array:
        xs = jax.vmap(self.in_projection)(xs)
        
        for ssm_layer, mixer in zip(self.layers, self.mixers):
            xs = ssm_layer(xs)
            xs = eqx.filter_vmap(mixer)(xs) # type:ignore

        if self.pool_method == 'none':
            return jax.vmap(self.out_projection)(xs)
        if self.pool_method == 'mean':
            xs = jnp.mean(xs, axis=0)
        elif self.pool_method == 'last':
            xs = xs[-1]
        return self.out_projection(xs)
