import equinox as eqx
import jax
from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import Tuple, Any
from jaxtyping import Array, Inexact


class BaseCell(eqx.Module, ABC):
    idim: int
    hdim: int
    states_shapes: Tuple[Tuple[int], ...]
    complex_state: bool

    def __init__(self, idim, hdim):
        """Initialize the base class, assigning the basic attributes.

        Args:
            idim (int): Input dimension
            hdim (int): Hidden dimension
        """
        self.idim = idim
        self.hdim = hdim

    @abstractmethod
    def __call__(
        self, x: Inexact[Array, "hdim"], state: Tuple[Array, ...]
    ) -> Tuple[Tuple[Array, ...], Array]:
        """Call the RNN cell. Should be implemented in subclasses.
        Args:
            x (Array): Input array of shape (idim,).
            state (Tuple[Array, ...]): Tuple of state arrays.
        Returns:
            Tuple[Tuple[Array, ...], Array]: Updated state tuple and output array.
        """
        return state, state[0]

    # . TODO: delete attributes states_shapes, complex_state and make something like
    # @abstractmethod
    # def init_state(self, *, key, batch_dims=())->Tuple[Array, ...]:
    #     pass
