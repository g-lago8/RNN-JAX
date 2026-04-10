"""
This file implements base RNN classes using jax.lax.scan on custom cells implemented in `cells/`
The implemented classes are
- RNN: base RNN class
- BidirectionalRNN: bidirectional RNN class
"""

import sys
from typing import Sequence, Union
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Inexact, Array, Complex
from rnn_jax.cells._base import BaseCell
from rnn_jax.utils.utils import concat_real_imag
from rnn_jax.layers._encoder import RNNEncoder, BidirectionalRNNEncoder


class RNN(eqx.Module):
    encoder: RNNEncoder
    hdim: int
    odim: int
    out_layer: eqx.nn.Linear

    def __init__(self, cell: BaseCell, odim, *, key, use_bias_out=True):
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
        self.out_layer = eqx.nn.Linear(
            self.hdim, odim, key=out_key, use_bias=use_bias_out
        )

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

    def __init__(self, fw_cell, bw_cell, odim, *, key, use_bias_out=True):
        """Bidirectional RNN, implemented using `jax.lax.scan`. This class takes a cell and iterates it through an input sequence, from first to last and from last to first.

        Args:
            cell (BaseCell): the cell implementing the logic of a single forward pass in time
            odim (int): output dimension
            key (PRNGKeyArray): random key
        """
        self.encoder = BidirectionalRNNEncoder(fw_cell, bw_cell)
        self.odim = odim
        out_key, key = jr.split(key)
        self.hdim = self.encoder.hdim
        self.out_layer = eqx.nn.Linear(
            self.hdim, odim, key=out_key, use_bias=use_bias_out
        )

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
        return self.out_layer(jnp.concat([hidden[-1], hidden_reverse[-1]])), (
            hidden,
            hidden_reverse[::-1],  # return backward states matching the forward order
        )


class DeepRNN(eqx.Module):
    layers: Sequence[RNNEncoder]
    n_layers: int
    hdim: Sequence[int]
    odim: int
    out_layer: eqx.nn.Linear

    def __init__(self, layers: Sequence[BaseCell], odim, *, key, use_bias_out=True):
        assert len(layers) >= 1, (
            "layers must be a non-empty sequence of Encoder objects, got an empty sequence"
        )
        self.layers = [RNNEncoder(cell) for cell in layers]
        self.odim = odim
        self.n_layers = len(self.layers)
        out_key, key = jr.split(key)
        self.hdim = [l.hdim for l in layers]
        self.out_layer = eqx.nn.Linear(
            self.hdim[-1], self.odim, key=out_key, use_bias=use_bias_out
        )

    def __call__(self, x, *, key=None):
        all_hidden = []
        for layer in self.layers:
            x = layer(x)
            all_hidden.append(x)
        return self.out_layer(all_hidden[-1][-1]), all_hidden


class DeepBidirectionalRNN(eqx.Module):
    layers: Sequence[BidirectionalRNNEncoder]
    n_layers: int
    hdim: Sequence[int]
    odim: int
    out_layer: eqx.nn.Linear

    def __init__(
        self,
        fw_layers: Sequence[BaseCell],
        bw_layers: Sequence[BaseCell],
        odim,
        *,
        key,
        use_bias_out=True,
    ):
        assert len(fw_layers) >= 1, (
            "fw_layers must be a non-empty sequence of Encoder objects, got an empty sequence"
        )
        assert len(fw_layers) >= 1, (
            "fw_layers must be a non-empty sequence of Encoder objects, got an empty sequence"
        )
        assert (lfw := len(fw_layers)) == (lbw := len(bw_layers)), (
            f"layers must be of the same lenght, got {lfw} and {lbw}"
        )
        self.layers = [
            BidirectionalRNNEncoder(fw_cell, bw_cell)
            for fw_cell, bw_cell in zip(fw_layers, bw_layers)
        ]
        self.odim = odim
        self.n_layers = len(self.layers)
        out_key, key = jr.split(key)
        self.hdim = [l.hdim for l in fw_layers]
        self.out_layer = eqx.nn.Linear(
            self.hdim[-1] * 2, self.odim, key=out_key, use_bias=use_bias_out
        )

    def __call__(self, x, *, key=None):
        all_hidden = []
        for layer in self.layers:
            h, h_reverse = layer(x)
            h_reverse = h_reverse[::-1]  # reverse the states to match the forward order
            all_hidden.append((h, h_reverse))
            x = jnp.concat([h, h_reverse], axis=1)
        return self.out_layer(
            jnp.concat([all_hidden[-1][0][-1], all_hidden[-1][1][0]])
        ), all_hidden