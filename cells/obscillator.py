import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from cells.base import BaseCell
from typing import Sequence, Callable, Tuple
from jaxtyping import Array, Float
from cells.utils import nonlinearity_dict


class CoupledOscillatoryRNNCell(BaseCell):
    gamma: float | Array
    eps: float | Array
    dt: float
    w_in: Float[Array, "hdim idim"]
    w_h: Float[Array, "hdim hdim"]
    w_z: Float[Array, "hdim hdim"] 
    b: Float[Array, "hdim"]
    nonlinearity:Callable
    def __init__(self, idim, hdim, gamma, eps, dt, nonlinearity='relu', heterogeneous=False, *, key):
        super().__init__(idim, hdim)
        self.states_shapes = ((hdim,), (hdim,))
        self.complex_state = False
        if heterogeneous:
            assert isinstance(gamma, Sequence) and len(gamma) == 2, "porcoddue bro"
            assert isinstance(eps, Sequence) and len(eps) == 2, "ziopera fra"
            gamma_key, eps_key, key = jr.split(key, 3)
            self.gamma = jr.uniform(gamma_key, (hdim,), minval=gamma[0], maxval=gamma[1])
            self.eps = jr.uniform(eps_key, (hdim,), minval=gamma[0], maxval=gamma[1])
        else:
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
            - jax.lax.stop_gradient(self.gamma) * h - jax.lax.stop_gradient(self.eps) * z
        h_new = h + self.dt * z_new
        return (h_new, z_new), h_new
    

