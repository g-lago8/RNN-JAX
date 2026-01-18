from typing import Sequence
from rnn_jax.cells import BaseCell
import jax
import equinox as eqx


class MessagePassingNN(eqx.Module):
    """General message passing neural network
    """
    n_nodes: int
    layers: Sequence[BaseCell]
    