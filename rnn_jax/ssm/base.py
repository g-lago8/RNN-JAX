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
from jaxtyping import Inexact, Array

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

    @abstractmethod
    def preprocess_inputs(self, xs: Inexact[Array, " seq_len model_dim"]):
        return xs

    @abstractmethod
    def preprocess_matrix(self, *args):
        pass

    @abstractmethod
    def ssm_cell(self, a: T, b: T) -> T:
        """A binary function given to the associative scan. Must be associative to work correctly

        Args:
            a (Array): first input
            b (Array): second input
        """
        pass

    @abstractmethod
    def postprocess_outputs(self, xs, hs):
        pass

    @abstractmethod
    def __call__(self, xs, h0=None):
        pass
