from typing import Tuple
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
from typing import Callable
from jaxtyping import Array, PRNGKeyArray, Float
from rnn_jax.cells.base import BaseCell
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
        """Initialize an Antisymmetric RNN Cell.

        Args:
            idim (int): Input dimension.
            hdim (int): Hidden dimension.
            stepsize (float): Step size for the update in the Euler discretization.
            diffusion (float): Diffusion diagonal term to ensure stability.
            key (PRNGKeyArray): JAX random key for parameter initialization.
            in_std (float, optional): Standard deviation for input weights initialization. If None, defaults to 1/idim.
            h_std (float, optional): Standard deviation for hidden weights initialization. If None, defaults to 1/hdim.
            nonlinearity (Callable, optional): Activation function. Defaults to jax.nn.relu.
        """
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

    def __call__(self, x: Float[Array, "idim"], state: Tuple[Array]):
        """Call the Antisymmetric RNN Cell.

        Args:
            x (Array): Input array of shape (idim,).
            state (Tuple[Array]): Tuple (h,) containing the hidden state array of shape (hdim,).

        Returns:
            (h,), h (Tuple[Tuple[Array], Array]): Updated state tuple and the new hidden state.
        """
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
        nonlinearity=jax.nn.relu,
        *,
        key,
    ):
        """Initialize an Antisymmetric RNN Cell.

        Args:
            idim (int): Input dimension.
            hdim (int): Hidden dimension.
            stepsize (float): Step size for the update in the Euler discretization.
            diffusion (float): Diffusion diagonal term to ensure stability.
            key (PRNGKeyArray): JAX random key for parameter initialization.
            in_std (float, optional): Standard deviation for input weights initialization. If None, defaults to 1/idim.
            h_std (float, optional): Standard deviation for hidden weights initialization. If None, defaults to 1/hdim.
            gate_std (float, optional): Standard deviation for gate weights initialization. If None, defaults to 1/idim.
            nonlinearity (Callable, optional): Activation function. Defaults to jax.nn.relu.
        """
        key1, key2 = jr.split(key)
        super().__init__(
            idim, hdim, stepsize, diffusion, in_std, h_std, nonlinearity, key=key1
        )
        if gate_std == None:
            gate_std = 1 / idim
        self.w_g = jr.normal(key2, (hdim, idim))
        self.b_g = jnp.zeros((hdim,))

    def __call__(self, x: Array, state: Tuple[Array]):
        """Call the Gated Antisymmetric RNN Cell.
        Args:
            x (Array): Input array of shape (idim,).
            state (Tuple[Array]): Tuple (h,) containing the hidden state array of shape (hdim,).
        Returns:
            Tuple[Tuple[Array], Array]: (h,), h Updated state tuple and the new hidden state.
        """

        (h,) = state
        gate = jax.nn.sigmoid(
            (self.w_hh - self.w_hh.T) @ h - self.diffusion + self.w_g @ x + self.b_g
        )
        h = h + self.stepsize * gate * self.nonlinearity(
            (self.w_hh - self.w_hh.T) @ h - self.diffusion + self.w_in @ x + self.b
        )
        return (h,), h
