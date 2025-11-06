import equinox as eqx
import jax
from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import Tuple, Any
from jaxtyping import Array, Inexact


class BaseCell(eqx.Module, ABC):
    idim: int
    hdim: int
    states_shapes:Tuple[Tuple[int],...]
    complex_state:bool
    def __init__(self, idim, hdim):
        self.idim = idim
        self.hdim = hdim
    @abstractmethod
    def __call__(self, x: Inexact[Array, "hdim"], state:Tuple[Array,...]) ->Tuple[Tuple[Array,...], Array]:
        pass
