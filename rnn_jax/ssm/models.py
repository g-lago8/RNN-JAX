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

    def __init__(
        self,
        in_dim: int,
        layers: Sequence[BaseSSMLayer],
        mixers: Sequence[Mixer],
        in_projection: bool = True,
        out_dim: Optional[int] = None,
        *,
        key: Array
    ):
        assert len(mixers) == len(layers), "number of layers and mixers must be the same"
        self.layers = layers
        self.mixers = mixers
        self.state_dim = [m.state_dim for m in layers]
        self.model_dim = [m.model_dim for m in layers]
        in_key, out_key = jr.split(key)
        if in_projection:
            self.in_projection = eqx.nn.Linear(in_features=in_dim, out_features=self.state_dim[0], key = in_key)
        else:
            self.in_projection = eqx.nn.Identity()
    
        if out_dim is not None:
            self.out_projection = eqx.nn.Linear(in_features=self.model_dim[-1], out_features=out_dim, key=out_key)
        else:
            self.out_projection = eqx.nn.Identity()

    def __call__(self, xs:Array) -> Array:
        xs = self.in_projection(xs)
        for ssm_layer, mixer in zip(self.layers, self.mixers):
            xs = ssm_layer(xs)
            xs = eqx.filter_vmap(mixer)(xs) # type:ignore
        return eqx.filter_vmap(self.out_projection)(xs)

 





