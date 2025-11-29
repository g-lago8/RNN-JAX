from typing import Tuple, Callable
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Inexact
from rnn_jax.cells.base import BaseCell
from typing import Tuple
from jaxtyping import Array


class LipschitzRNNCell(BaseCell):
    beta_a: float
    gamma_a: float
    beta_w: float
    gamma_w: float
    dt: float
    M_a: Array
    M_w: Array
    b: Array
    W_in: Array
    discretization: str
    nonlinearity: Callable

    def __init__(
        self,
        idim,
        hdim,
        beta_a,
        gamma_a,
        beta_w,
        gamma_w,
        dt,
        weight_std,
        discretization="euler",
        nonlinearity=jax.nn.tanh,
        *,
        key,
    ):
        """Initialize the Lipschitz RNN cell

        Args:
            idim (int): Input dimension
            hdim (int): Hidden dimension
            beta_a (float): Proportion of antisymmetric part in A
            gamma_a (float): Diagonal stabilization term for A
            gamma_w (float): Diagonal stabilization term for W
            dt (float): Step size for the discretization
            weight_std (float): Standard deviation for weight initialization
            key (PRNGKey): JAX PRNG key for initialization
            discretization (str, optional): Dicretization type, the accepted values are 'euler' and 'rk2'. Defaults to "euler".
            nonlinearity (Callable, optional): Nonlinear activation function. Defaults to jax.nn.tanh.
        """
        super().__init__(idim, hdim)
        self.complex_state = False
        self.states_shapes = ((hdim,), (hdim,))
        self.beta_a = beta_a
        self.gamma_a = gamma_a
        self.beta_w = beta_w
        self.gamma_w = gamma_w
        self.dt = dt
        self.discretization = discretization
        self.nonlinearity = nonlinearity
        a_key, w_key, in_key = jr.split(key, 3)
        self.M_a = jr.normal(w_key, (hdim, hdim)) * weight_std
        self.M_w = jr.normal(a_key, (hdim, hdim)) * weight_std
        self.W_in = jr.normal(in_key, (hdim, idim)) * weight_std
        self.b = jnp.zeros((hdim,))

    def _A(self):
        """return A as a convex combination of a symmetric and antisymmetric matrix.
        **CAREFUL**: differently from the paper, A doesn't take in account the diagonal term gamma_a * I, that
        is directly added to the vector resulting from the matvec multiplication

        Returns:
            Array: A = (1-beta_a) (M_a + M_a^T) + beta_a (M_a - M_a^T)
        """
        return (1 - self.beta_a) * (self.M_a + self.M_a.T) + self.beta_a * (
            self.M_a - self.M_a.T
        )

    def _W(self):
        """return W as a convex combination of a symmetric and antisymmetric matrix.
        **CAREFUL**: differently from the paper, A doesn't take in account the diagonal term gamma_a * I, that
        is directly added to the vector resulting from the matvec multiplication

        Returns:
            Array: W = (1-beta_w) (M_w + M_w^T) + beta_a (M_w - M_w^T)
        """
        return (1 - self.beta_w) * (self.M_w + self.M_w.T) + self.beta_w * (
            self.M_w - self.M_w.T
        )

    def _update_euler(self, x, h, z, W, A):
        return h + self.dt * (A @ h - self.gamma_a) + self.dt * self.nonlinearity(z)

    def _update_rk2(self, x, h, z, W, A):
        h_tilde = (
            h
            + (self.dt / 2) * (A @ h - self.gamma_a)
            + (self.dt / 2) * self.nonlinearity(z)
        )  # intermediate state
        z_tilde = W @ h_tilde - self.gamma_w + self.W_in @ x + self.b
        return (
            h
            + self.dt * (A @ h_tilde - self.gamma_a)
            + self.dt * self.nonlinearity(z_tilde)
        )

    def _update_h(self, x, h, z, W, A):
        if self.discretization == "euler":
            return self._update_euler(x, h, z, W, A)
        elif self.discretization == "rk2":
            return self._update_rk2(x, h, z, W, A)
        else:
            raise ValueError("unrecognized discretization method")

    def __call__(
        self, x: Array, state: Tuple[Array, Array]
    ) -> Tuple[Tuple[Array, Array], Array]:
        h, z = state
        W = self._W()
        A = self._A()
        z_new = W @ h - self.gamma_w + self.W_in @ x + self.b
        h_new = self._update_h(x, h, z, W, A)
        return (h_new, z_new), h_new
