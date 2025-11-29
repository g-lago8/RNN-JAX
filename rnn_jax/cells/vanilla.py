from typing import Tuple
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
from typing import Callable
from jaxtyping import Inexact, Array, PRNGKeyArray
from rnn_jax.cells.base import BaseCell
from jax.nn.initializers import Initializer


class ElmanRNNCell(BaseCell):
    w_hh: Inexact[Array, "hdim hdim"]
    w_ih: Inexact[Array, "hdim idim"]
    b: Inexact[Array, "hdim"]
    nonlinearity: Callable

    def __init__(
        self,
        idim,
        hdim,
        kernel_init=jax.nn.initializers.glorot_normal(),
        recurrent_kernel_init=jax.nn.initializers.orthogonal(),
        bias_init=jax.nn.initializers.zeros,
        nonlinearity=jax.nn.relu,
        *,
        key,
    ):
        """Elman RNN, also known as Vanilla RNN or Simple RNN, cell. its update follows the equation
        `h(t+1) = nonlinearity(W_h h(t) + W_x x(t+1) + b)`

        Args:
            idim (int): input dimension
            hdim (int): hidden dimension
            key (PRNGKeyArray): pseudoRNG key
            kernel_init (jax.nn.initializers.Initializer, optional): _description_. Defaults to `initializers.glorot_normal()`.
            recurrent_kernel_init (jax.nn.initializers.Initializer, optional): _description_. Defaults to a `initializers.normal(stddev=1)`.
            bias_init (jax.nn.initializers.Initializer, optional): _description_. Defaults to `initializers.zeros`.
            nonlinearity (Callable, optional): _description_. Defaults to `jax.nn.relu`.
        """
        super().__init__(idim, hdim)
        self.states_shapes = (hdim,)
        self.complex_state = False
        ikey, hkey, bkey = jr.split(key, 3)
        self.w_ih = kernel_init(ikey, (hdim, idim))
        self.w_hh = recurrent_kernel_init(hkey, (hdim, hdim))
        self.b = bias_init(bkey, hdim)
        self.nonlinearity = nonlinearity

    def __call__(
        self, x: Array, state: Tuple[Array, ...]
    ) -> Tuple[Tuple[Array], Array]:
        (h,) = state
        new_h = self.nonlinearity(self.w_hh @ h + self.w_ih @ x + self.b)
        return (new_h,), new_h


class IndRNNCell(BaseCell):
    w_hh: Inexact[Array, "hdim"]
    w_ih: Inexact[Array, "hdim idim"]
    b: Inexact[Array, "hdim"]
    nonlinearity: Callable

    def __init__(
        self,
        idim,
        hdim,
        kernel_init=jax.nn.initializers.glorot_normal(),
        recurrent_kernel_init=jax.nn.initializers.normal(stddev=1.0),
        bias_init=jax.nn.initializers.zeros,
        nonlinearity=jax.nn.relu,
        *,
        key,
    ):
        """Independent RNN cell. This type of RNN is similar to a simple RNN, but the h-to-h matrix is diagonal, therefore neurons update independently

        Args:
            idim (int): input dimension
            hdim (int): hidden dimension
            key (PRNGKeyArray): pseudoRNG key
            kernel_init (jax.nn.initializers.Initializer, optional): _description_. Defaults to `initializers.glorot_normal()`.
            recurrent_kernel_init (jax.nn.initializers.Initializer, optional): _description_. Defaults to a `initializers.normal(stddev=1)`.
            bias_init (jax.nn.initializers.Initializer, optional): _description_. Defaults to `initializers.zeros`.
            nonlinearity (Callable, optional): _description_. Defaults to `jax.nn.relu`.
        """
        super().__init__(idim, hdim)
        self.states_shapes = (hdim,)
        self.complex_state = False
        ikey, hkey, bkey = jr.split(key, 3)
        self.w_ih = kernel_init(ikey, (hdim, idim))
        self.w_hh = recurrent_kernel_init(hkey, (hdim,))
        self.b = bias_init(bkey, hdim)
        self.nonlinearity = nonlinearity

    def __call__(self, x: Array, state: Tuple[Array]) -> Tuple[Tuple[Array], Array]:
        (h,) = state
        new_h = self.nonlinearity(self.w_hh * h + self.w_ih @ x + self.b)
        return (new_h,), new_h
