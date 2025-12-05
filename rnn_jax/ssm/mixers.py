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


class GLUMixer(eqx.Module):
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
