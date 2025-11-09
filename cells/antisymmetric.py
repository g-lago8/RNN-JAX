from typing import Tuple
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
from typing import Callable
from jaxtyping import Inexact, Array, PRNGKeyArray
from cells.base import BaseCell
from jax.nn.initializers import Initializer


class AntiSymmetricRNNCell(BaseCell):
    w_hh: Array
    w_in: Array
    b: Array
    stepsize: float
    diffusion: float
    nonlinearity: Callable

    def __init__(
        self,
        idim,
        hdim,
        stepsize,
        diffusion,
        in_std=None,
        h_std=None,
        nonlinearity=jax.nn.relu,
        *,
        key,
    ):
        self.states_shapes = (hdim,)
        self.complex_state = False
        self.stepsize = stepsize
        self.diffusion = diffusion
        self.nonlinearity = nonlinearity
        inkey, hkey, bkey = jr.split(key, 3)
        if in_std == None:
            in_std = 1 / idim
        if h_std == None:
            h_std = 1 / hdim
        self.w_in = jr.normal(inkey, (hdim, idim)) * h_std
        self.w_h = jr.normal(hkey, (hdim, hdim)) * in_std
        self.b = jnp.zeros((hdim,))

    def __call__(self, x: Array, state: Tuple[Array]):
        (h,) = state
        h = h + self.stepsize * self.nonlinearity(
            (self.w_hh - self.w_hh.T) @ h - self.diffusion + self.w_in @ x + self.b
        )
        return (h,), h


class GatedAntiSymmetricRNNCell(AntiSymmetricRNNCell):
    w_g: Array

    def __init__(
        self,
        idim,
        hdim,
        stepsize,
        diffusion,
        in_std=None,
        h_std=None,
        gate_std=None,
        *,
        key,
    ):
        key1, key2 = jr.split(key)
        super().__init__(idim, hdim, stepsize, diffusion, in_std, h_std, key=key1)
        if gate_std == None:
            gate_std = 1 / idim
        self.w_g = jr.normal(key2, (hdim, idim))
        self.b_g = jnp.zeros((hdim,))

    def __call__(self, x: Array, state: Tuple[Array]):
        (h,) = state
        gate = jax.nn.sigmoid(
            (self.w_hh - self.w_hh.T) @ h - self.diffusion + self.w_g @ x + self.b_g
        )
        h = h + self.stepsize * gate * self.nonlinearity(
            (self.w_hh - self.w_hh.T) @ h - self.diffusion + self.w_in @ x + self.b
        )
        return (h,), h
