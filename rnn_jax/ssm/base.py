# This file implements the base class that should be subclassed when creating a new SSM.
# The structure of a forward pass of a linear SSM-lke model on a sequence xs is, using parallel scan
# ========================================================================================
# 1. Preprocess the input sequence xs in batches
#    e.g. Bx_t = B @ x_t for each x_t in xs
# 2. Apply the parallel scan on the binary associative function f(a, b)
# implementing the recursive pass, getting hidden states hs,
#    e.g. h_t = A @ h_t-1 + Bx
# 3. Postprocess the states in batches
#    e.g. y_t = C @ h_t + D @ x_t
# ========================================================================================


from abc import ABC, abstractmethod
import equinox as eqx
from typing import TypeVar, Tuple
import jax
from jaxtyping import Inexact, Array
from jax.lax import associative_scan
T = TypeVar("T")


class BaseSSMLayer(eqx.Module, ABC):
    """Base SSM module"""

    in_dim: int
    state_dim: int
    model_dim: int

    def __init__(self, in_dim: int, state_dim: int, model_dim=None):
        self.in_dim = in_dim
        self.state_dim = state_dim
        if model_dim is None:
            model_dim = state_dim
        self.model_dim = model_dim

    def preprocess_inputs(self, xs: Inexact[Array, "seq_len model_dim"], w_in: Inexact[Array, "state_dim in_dim"]) -> Inexact[Array, " seq_len state_dim"]:
        return jax.vmap(lambda x: w_in @ x)(xs)

    @abstractmethod
    def discretize(self, *args)->Tuple[Array, ...]:
        raise NotImplementedError("discretize and postprocess_outputs should be implemented in the subclass")

    def ssm_cell(
        self, a: Tuple[Array, Array], b: Tuple[Array, Array]
    ) -> Tuple[Array, Array]:
        matrix_pow_i, bx_i = a
        matrix_pow_j, bx_j = b
        return matrix_pow_i * matrix_pow_i, matrix_pow_j * bx_i + bx_j

    @abstractmethod
    def postprocess_outputs(self, xs, hs)->Array:
        raise NotImplementedError("discretize and postprocess_outputs should be implemented in the subclass")

    def __call__(self, xs, h0=None)->Array:
        seq_len = xs.shape[0]
        lambda_elements, w_in = self.discretize(seq_len)
        w_in_xs = self.preprocess_inputs(xs, w_in)
        scan_elements = (lambda_elements, w_in_xs)
        lambda_powers, hs = associative_scan(self.ssm_cell, scan_elements)
        return self.postprocess_outputs(xs, hs)
