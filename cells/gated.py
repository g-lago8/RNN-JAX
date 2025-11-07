from typing import Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Inexact
from cells.base import BaseCell
from typing import Tuple
from jaxtyping import Array


# TODO: reimplement from zero, since the implementations in Equinox does
#  not seem to have particular optimization tricks anyway.

class LongShortTermMemory(BaseCell):
    lstm: eqx.nn.LSTMCell

    def __init__(self, idim:int, hdim:int, *, key, **lstm_kwargs):
        """Wrapper for `equinox.nn.LSTMCell`

        Args:
            idim (int): input dimension
            hdim (int): hidden dimension
            key (PRNGKey): pseudo-RNG key
        """
        super().__init__(idim, hdim)
        self.complex_state = False
        self.states_shapes = ((hdim,), (hdim,))
        self.lstm = eqx.nn.LSTMCell(idim, hdim, key=key, **lstm_kwargs)

    def __call__(self, x: jax.Array, state: Tuple[Array, ...]) -> Tuple[Tuple[Array, Array], Array]:
        h, c = self.lstm(x, state)
        return (h, c), h


class GatedRecurrentUnit(BaseCell):
    gru: eqx.nn.GRUCell

    def __init__(self, idim, hdim, *, key, **gru_kwargs):
        """Wrapper for `equinox.nn.GRUCell`

        Args:
            idim (int): input dimension
            hdim (int): hidden dimension
            key (PRNGKey): pseudo-RNG key
        """
        super().__init__(idim, hdim)
        self.complex_state = False
        self.states_shapes = ((hdim,))
        self.gru =eqx.nn.GRUCell(idim, hdim, key=key, **gru_kwargs)
    
    def __call__(self, x: Array, state: Tuple[Array]) -> Tuple[Tuple[Array], Array]:
        h = self.gru(x, state[0])
        return (h,), h
    
