import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from base import BaseCell
from typing import Sequence, Callable, Tuple
from jaxtyping import PyTree, Array, ArrayLike, Float, Inexact, Int, PRNGKeyArray, Complex

nonlinearity_dict={
    'relu': jax.nn.relu,
    'tanh': jax.nn.tanh,
    'sigmoid': jax.nn.sigmoid
}

class CoupledOscillatoryRNNCell(BaseCell):
    gamma: float
    eps: float
    dt: float
    w_in: Float[Array, "hdim idim"]
    w_h: Float[Array, "hdim hdim"]
    w_z: Float[Array, "hdim hdim"] 
    b: Float[Array, "hdim"]
    nonlinearity:Callable
    def __init__(self, idim, hdim, gamma, eps, dt, nonlinearity='relu', *, key):
        super().__init__(idim, hdim)
        self.states_shapes = ((hdim,), (hdim,))
        self.complex_state = False
        self.gamma = gamma
        self.eps = eps
        self.dt = dt
        ikey, hkey, zkey = jr.split(key, 3)
        self.w_in = jr.uniform(ikey, (hdim, idim), minval= - 1 / jnp.sqrt(idim), maxval = 1 / jnp.sqrt(hdim))
        self.w_h = jr.uniform(hkey, (hdim, hdim), minval= - 1 / jnp.sqrt(hdim), maxval = 1 / jnp.sqrt(hdim))
        self.w_z = jr.uniform(zkey, (hdim, hdim), minval= - 1 / jnp.sqrt(hdim), maxval = 1 / jnp.sqrt(hdim))
        self.b = jnp.zeros((hdim,))
        self.nonlinearity = nonlinearity_dict[nonlinearity]


    def __call__(self, x: Array, state: Tuple[Array, Array]) -> Tuple[Tuple[Array, Array], Array]:
        h, z = state
        z_new = z + self.dt * self.nonlinearity(self.w_h @ h + self.w_z @ z + self.w_in @ x + self.b ) \
            - self.gamma * h - self.eps * z
        h_new = h + self.dt * z_new
        return (h, z), h
    

