import jax
from jax import random as jr
from jax import nn
import numpy as np
from jax import numpy as jnp
import equinox as eqx
from jax.lax import associative_scan
from typing import TypeVar, Tuple, Callable, Sequence, Optional
from jaxtyping import Inexact, Array
from rnn_jax.ssm.base import BaseSSMLayer
import einops


class Mixer(eqx.Module):
    def __init__(self, in_dim, out_dim, *, key):
        pass

    def __call__(self, x):
        return x


class GLUMixer(Mixer):
    gate_projection: eqx.nn.Linear
    main_projection: eqx.nn.Linear
    nonlinearity: Callable

    def __init__(
        self,
        in_dim,
        out_dim,
        nonlinearity=jax.nn.swish,
        *,
        key
        ):
        key1, key2 = jr.split(key)
        self.gate_projection = eqx.nn.Linear(in_dim, out_dim, key=key)
        self.main_projection = eqx.nn.Linear(in_dim, out_dim, key=key)
        self.nonlinearity = nonlinearity

    def __call__(self, x):
        return self.main_projection(x) * self.nonlinearity(self.gate_projection(x))


class FFNMixer(Mixer):
    pre_nonlinearity: Callable
    post_nonlinearity: Callable
    pre_projection: eqx.nn.Linear
    post_projection: eqx.nn.Linear

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        pre_nonlinearity=jax.nn.swish,
        post_nonlinearity=jax.nn.swish,
        *,
        key
        ):
        key1, key2 = jr.split(key)
        self.pre_projection = eqx.nn.Linear(in_dim, hidden_dim, key=key1)
        self.post_projection = eqx.nn.Linear(hidden_dim, out_dim, key=key2)
        self.pre_nonlinearity = pre_nonlinearity
        self.post_nonlinearity = post_nonlinearity

    def __call__(self, x):
        h = self.pre_nonlinearity(self.pre_projection(x))
        return self.post_nonlinearity(self.post_projection(h))
    
class IdentityMixer(Mixer):
    def __init__(self, in_dim, out_dim, *, key):
        pass

    def __call__(self, x):
        return x