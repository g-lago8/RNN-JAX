import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jax.scipy.linalg import expm  # optimized matrix exponentiation
from rnn_jax.cells._base import BaseCell
from typing import Optional, Sequence, Callable, Tuple
from jaxtyping import (
    PyTree,
    Array,
    ArrayLike,
    Float,
    Inexact,
    Int,
    PRNGKeyArray,
    Complex,
)


class ExpRNNCell(eqx.Module):
    A: Inexact[Array, "hdim hdim"]  # expm(A - A.T) is the normal matrix
    w_in: Inexact[Array, "hdim, idim"]  # input to hidden matrix
    bias: Inexact[Array, "hdim"]
    nonlinearity: Callable

    def __init__(
        self,
        idim,
        hdim,
        nonlinearity: Optional[Callable[[Array], Array]] = None,
        complex_weights: bool = False,
        *,
        key,
    ):
        """The ExpRNN cell, that parametrizes the hidden-to-hidden matrix as the matrix exponential of a skew-symmetric matrix,
        ensuring that the hidden-to-hidden matrix is normal and has eigenvalues of module 1.
        """
        A_key, w_in_key, bias_key = jr.split(key, 3)
        self.A = (
            jax.random.orthogonal(A_key, hdim)
            if not complex_weights
            else jax.random.orthogonal(A_key, hdim, dtype=jnp.complex_)
        )
        self.w_in = (
            jax.random.normal(w_in_key, (hdim, idim))
            if not complex_weights
            else jax.random.normal(w_in_key, (hdim, idim), dtype=jnp.complex_)
        )
        self.bias = (
            jax.random.normal(bias_key, (hdim,))
            if not complex_weights
            else jax.random.normal(bias_key, (hdim,), dtype=jnp.complex_)
        )
        self.nonlinearity = (
            nonlinearity if nonlinearity is not None else lambda x: jnp.maximum(0, x)
        )  # default nonlinearity is ReLU

    def __call__(self, x, h, *, key: Optional[Array] = None):
        # Materialize the normal matrix P
        P = expm(self.A - jnp.conjugate(self.A).T)
        return self.nonlinearity((P @ h + self.w_in @ x + self.bias).real)


class NonNormalRNNCells(eqx.Module):
    A: Inexact[Array, "hdim hdim"]  # expm(A - A.T) is the normal matrix
    tril_wh: Inexact[
        Array, "hdim, hdim"
    ]  # lower triangular matrix with zeros on the diagonal, used to parametrize the non-normal part of the hidden-to-hidden matrix
    diag_wh: Inexact[
        Array, "hdim"
    ]  # diagonal part with the eigenvalues of the non-normal hidden-to-hidden matrix
    w_in: Inexact[Array, "hdim, idim"]  # input-to-hidden matrix
    bias: Inexact[Array, "hdim"]
    nonlinearity: Callable

    def __init__(
        self,
        idim,
        hdim,
        nonlinearity: Optional[Callable[[Array], Array]] = None,
        complex_weights: bool = False,
        *,
        key,
    ):
        """The ExpRNN cell, that parametrizes the hidden-to-hidden matrix as the Schur decomposition of a non-normal matrix"""
        A_key, tril_wh_key, diag_wh_key, w_in_key, bias_key = jr.split(key, 5)
        self.A = (
            jax.random.orthogonal(A_key, hdim)
            if not complex_weights
            else jax.random.orthogonal(A_key, hdim, dtype=jnp.complex_)
        )
        self.tril_wh = jnp.tril(
            jax.random.normal(tril_wh_key, (hdim, hdim))
            if not complex_weights
            else jax.random.normal(tril_wh_key, (hdim, hdim), dtype=jnp.complex_),
            k=-1,
        )
        self.diag_wh = (
            jax.random.normal(diag_wh_key, (hdim,))
            if not complex_weights
            else jax.random.normal(diag_wh_key, (hdim,), dtype=jnp.complex_)
        )
        self.w_in = (
            jax.random.normal(w_in_key, (hdim, idim))
            if not complex_weights
            else jax.random.normal(w_in_key, (hdim, idim), dtype=jnp.complex_)
        )
        self.bias = (
            jax.random.normal(bias_key, (hdim,))
            if not complex_weights
            else jax.random.normal(bias_key, (hdim,), dtype=jnp.complex_)
        )
        self.nonlinearity = (
            nonlinearity if nonlinearity is not None else lambda x: jnp.maximum(0, x)
        )  # default nonlinearity is ReLU

    def __call__(self, x, h, *, key: Optional[Array] = None):
        # Materialize the normal matrix P
        P = expm(self.A - jnp.conjugate(self.A).T)
        # input processing
        wx = self.w_in @ x
        # output processing h <- (P * T * P^t) * h
        new_h = P.T @ h
        new_h = jnp.tril(self.tril_wh, k=-1) @ new_h + self.diag_wh * new_h
        new_h = P @ new_h
        return self.nonlinearity((new_h + wx + self.bias).real)
