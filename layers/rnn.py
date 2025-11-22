"""
This file implements base RNN classes using jax.lax.scan on custom cells implemented in `cells/`
The implemented classes are
- RNN: base RNN class
- BidirectionalRNN: bidirectional RNN class
"""

import sys
from typing import Sequence

sys.path.append("..")
sys.path.append("../cells")
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Inexact, Array, Complex
from cells.base import BaseCell
from utils.utils import concat_real_imag
from layers.encoder import RNNEncoder, BidirectionalRNNEncoder
jax.config.update("jax_debug_nans", 'true')


class RNN(eqx.Module):
    encoder: RNNEncoder
    hdim: int
    odim: int
    out_layer: eqx.nn.Linear

    def __init__(self, cell: BaseCell, odim, *, key):
        """RNN, implemented with `jax.lax.scan`.

        This class takes a cell and iterates it through an input sequence, from first to last, 
        then transforms linearly the last hidden state

        Args:
            cell (BaseCell): the cell implementing the logic of a single forward pass in time
            odim (int): output dimension
            key (PRNGKeyArray): random key
        """
        encoder_key, key = jr.split(key)
        self.encoder = RNNEncoder(cell, key=encoder_key)
        self.odim = odim
        out_key, key = jr.split(key)
        self.hdim = self.encoder.hdim
        self.out_layer = eqx.nn.Linear(self.hdim, odim, key=out_key)

    def __call__(self, x: Inexact[Array, "seq_len idim"]):
        """Calls the encoder on an input sequence x and 
        Args:
            x (Array): input sequence, an array of shape (seq_len, idim)

        Returns:
            y (Array): output array, obtained applying the output transformation to the last state of the network
        """
        all_outs = self.encoder(x)
        return self.out_layer(all_outs[-1]), all_outs


class BidirectionalRNN(eqx.Module):
    encoder: BidirectionalRNNEncoder
    hdim: int
    odim: int
    out_layer: eqx.nn.Linear

    def __init__(self, cell, odim, *, key):
        """Bidirectional RNN, implemented using `jax.lax.scan`. This class takes a cell and iterates it through an input sequence, from first to last and from last to first.

        Args:
            cell (BaseCell): the cell implementing the logic of a single forward pass in time
            odim (int): output dimension
            key (PRNGKeyArray): random key
        """
        self.encoder = BidirectionalRNNEncoder(cell)
        self.odim = odim
        out_key, key = jr.split(key)
        self.hdim = self.encoder.hdim
        self.out_layer = eqx.nn.Linear(self.hdim, odim, key=out_key)

    def __call__(self, x: Inexact[Array, "seq_len idim"]):
        """Calls the cell on an input sequence x, in both directions,
        and applies a linear layer
        Args:
            x (Array): input sequence, an array of shape (seq_len, idim)

        Returns:
            y (Array): output array, obtained applying the output transformation
            to a concatenation of the two states obtained iterating the network
            from first to last and form last to first
            all_hidden (Tuple[Array, Array]): all hidden states.
            all_hidden[0] are the first-to-last states, all_hidden[1] are the last-to-first states
        """
        hidden, hidden_reverse = self.encoder(x)
        return self.out_layer(jnp.concat([hidden[-1], hidden_reverse[-1]])), (hidden, hidden_reverse)




if __name__ == "__main__":
    from cells import (
        UnitaryEvolutionRNNCell,
        LongShortTermMemory,
        ElmanRNNCell,
        CoupledOscillatoryRNNCell,
        LipschitzRNNCell
    )

    key = jr.key(0)
    idim = 10
    hdim = 16
    urnn_cell = UnitaryEvolutionRNNCell(idim, hdim, use_bias_in=True, key=key)
    urnn = RNN(urnn_cell, 1, key=key)
    x = jr.normal(key, (100, idim))
    print("uRNN")
    print(urnn(x))
    lstm_cell = LongShortTermMemory(idim, hdim, key=key)
    lstm = RNN(lstm_cell, 1, key=key)
    print("LSTM")
    print(lstm(x))
    rnn_cell = ElmanRNNCell(idim, hdim, key=key)
    rnn = RNN(rnn_cell, 1, key=key)
    print("RNN")
    print(rnn(x))
    cornn_cell = CoupledOscillatoryRNNCell(idim, hdim, 1.0, 1.0, 0.01, key=key)
    cornn = RNN(cornn_cell, 1, key=key)
    print("coRNN")
    print(cornn(x))
    hcornn_cell = CoupledOscillatoryRNNCell(
        idim, hdim, (0, 1.0), (-1.0, 1.0), dt=0.01, heterogeneous=True, key=key
    )
    hcornn = RNN(hcornn_cell, 1, key=key)
    print("hcoRNN")
    print(hcornn(x))
    # try the bidirectional impl.
    print("Bidirectional LSTM")
    bidirectional_lstm = BidirectionalRNN(lstm_cell, 1, key=key)
    print(bidirectional_lstm(x))

    liprnn_cell = LipschitzRNNCell(idim, hdim, 0.65, 1., 0.65, 1., 0.001, 1/16, key=key)
    liprnn = RNN(liprnn_cell, 1, key=key)
    print("Lipschitz RNN, Euler")
    print(liprnn(x))

    liprnn_cell = LipschitzRNNCell(idim, hdim, 0.65, 1., 0.65, 1., 0.001, 1/16, key=key, discretization='rk2')
    liprnn = RNN(liprnn_cell, 1, key=key)
    print("Lipschitz RNN, RK2")
    print(liprnn(x))