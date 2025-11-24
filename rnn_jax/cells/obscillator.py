import numpy as np
import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from rnn_jax.cells.base import BaseCell
from typing import Sequence, Callable, Tuple
from jaxtyping import Array, Float


class CoupledOscillatoryRNNCell(BaseCell):
    gamma: float | Array
    eps: float | Array
    dt: float
    w_in: Float[Array, "hdim idim"]
    w_h: Float[Array, "hdim hdim"]
    w_z: Float[Array, "hdim hdim"]
    b: Float[Array, "hdim"]
    nonlinearity: Callable

    def __init__(
        self,
        idim,
        hdim,
        gamma,
        eps,
        dt,
        nonlinearity=jax.nn.relu,
        heterogeneous=False,
        *,
        key,
    ):
        super().__init__(idim, hdim)
        self.states_shapes = ((hdim,), (hdim,))
        self.complex_state = False
        if heterogeneous:
            assert isinstance(gamma, Sequence) and len(gamma) == 2, (
                "if heterogeneous==True, gamma must be a tuple (gamma_min, gamma_max)"
            )
            assert isinstance(eps, Sequence) and len(eps) == 2, (
                "if heterogeneous==True, eps must ne a tuple (eps_min, eps_max)"
            )
            gamma_key, eps_key, key = jr.split(key, 3)
            self.gamma = jr.uniform(
                gamma_key, (hdim,), minval=gamma[0], maxval=gamma[1]
            )
            self.eps = jr.uniform(eps_key, (hdim,), minval=gamma[0], maxval=gamma[1])
        else:
            assert jnp.isscalar(gamma), (
                "if heterogeneous==False, gamma must be a scalar"
            )
            assert jnp.isscalar(eps), "if heterogeneous==False, eps must be a scalar"
            self.gamma = gamma
            self.eps = eps
        self.dt = dt
        ikey, hkey, zkey = jr.split(key, 3)
        self.w_in = jr.uniform(
            ikey, (hdim, idim), minval=-1 / jnp.sqrt(idim), maxval=1 / jnp.sqrt(hdim)
        )
        self.w_h = jr.uniform(
            hkey, (hdim, hdim), minval=-1 / jnp.sqrt(hdim), maxval=1 / jnp.sqrt(hdim)
        )
        self.w_z = jr.uniform(
            zkey, (hdim, hdim), minval=-1 / jnp.sqrt(hdim), maxval=1 / jnp.sqrt(hdim)
        )
        self.b = jnp.zeros((hdim,))
        self.nonlinearity = nonlinearity

    def __call__(
        self, x: Array, state: Tuple[Array, Array]
    ) -> Tuple[Tuple[Array, Array], Array]:
        h, z = state
        z_new = (
            z
            + self.dt
            * self.nonlinearity(self.w_h @ h + self.w_z @ z + self.w_in @ x + self.b)
            - jax.lax.stop_gradient(self.gamma) * h
            - jax.lax.stop_gradient(self.eps) * z
        )
        h_new = h + self.dt * z_new
        return (h_new, z_new), h_new
