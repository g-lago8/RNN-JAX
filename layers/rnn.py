"""
This file implements base RNN classes using jax.lax.scan on custom cells implemented in `cells/`
The implemented classes are
- RNN: base RNN class
- BidirectionalRNN: bidirectional RNN class
"""
import sys
sys.path.append("..")
sys.path.append("../cells")
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Inexact, Array, Complex
from cells.base import BaseCell

def concat_real_imag(x:Complex[Array, "..."], axis=-1):
    """Concatenate real and image parts of an array

    Args:
        x (Complex[Array]): complex array
        axis (int, optional): axis of the concatenation. Defaults to -1.

    Returns:
        Array: a real concatenated array 
    """
    x_real = jnp.real(x)
    x_imag = jnp.imag(x)
    return jnp.concatenate([x_real, x_imag], axis=axis)


# TODO: is no output layer cleaner, giving the user freedom to choose the readout function?

class RNN(eqx.Module):

    cell: BaseCell
    hdim: int
    odim: int
    out_layer: eqx.nn.Linear
    def __init__(self, cell:BaseCell, odim, *, key):
        """RNN, implemented with `jax.lax.scan`. 
        
        This class takes a cell and iterates it through an input sequence, from first to last

        Args:
            cell (BaseCell): the cell implementing the logic of a single forward pass in time
            odim (int): output dimension
            key (PRNGKeyArray): random key
        """
        self.cell = cell
        self.odim = odim
        out_key, key = jr.split(key)
        self.hdim = self.cell.hdim if not self.cell.complex_state else self.cell.hdim * 2
        self.out_layer = eqx.nn.Linear(self.hdim, odim, key=out_key)

    def __call__(self, x:Inexact[Array, "seq_len idim"]):
        """Calls the cell on an input sequence x

        Args:
            x (Array): input sequence, an array of shape (seq_len, idim)

        Returns:
            y (Array): output array, obtained applying the output transformation to the last state of the network
        """
        scan_fn = lambda state, x_t : self.cell(x_t, state)
        dtype = jnp.complex64 if self.cell.complex_state else jnp.float32
        initial_state = tuple(jnp.zeros(s, dtype=dtype) for s in self.cell.states_shapes)
        last_state, all_outs = jax.lax.scan(scan_fn, initial_state, x)
        if self.cell.complex_state:
            all_outs= concat_real_imag(all_outs)
        return self.out_layer(all_outs[-1]), all_outs


class BidirectionalRNN(eqx.Module):
    cell:BaseCell
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
        self.cell = cell
        self.odim = odim
        out_key, key = jr.split(key)
        self.hdim = self.cell.hdim * 2 if not self.cell.complex_state else self.cell.hdim * 4 
        self.out_layer = eqx.nn.Linear(self.hdim, odim, key=out_key)

    def __call__(self, x:Inexact[Array, "seq_len idim"]):
        """Calls the cell on an input sequence x, in both directions

        Args:
            x (Array): input sequence, an array of shape (seq_len, idim)

        Returns:
            y (Array): output array, obtained applying the output transformation 
            to a concatenation of the two states obtained iterating the network 
            from first to last and form last to first 
        """
        scan_fn = lambda state, x_t : self.cell(x_t, state)
        dtype = jnp.complex64 if self.cell.complex_state else jnp.float32
        initial_state = tuple(jnp.zeros(s, dtype=dtype) for s in self.cell.states_shapes)
        last_state, outs = jax.lax.scan(scan_fn, initial_state, x)
        last_state_reverse, outs_reverse = jax.lax.scan(scan_fn, initial_state, x[::-1])
        if self.cell.complex_state:
            outs = concat_real_imag(outs)
            outs_reverse = concat_real_imag(outs_reverse)
        all_outs = jnp.concat([outs[-1], outs_reverse[-1]])
        return self.out_layer(all_outs[-1]), all_outs





if __name__ == "__main__":
    from cells import UnitaryEvolutionRNNCell, LongShortTermMemory, ElmanRNNCell, CoupledOscillatoryRNNCell
    key = jr.key(0)
    idim = 10
    hdim = 16
    urnn_cell = UnitaryEvolutionRNNCell(idim, hdim, use_bias_in=True, key = key)
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
    cornn_cell = CoupledOscillatoryRNNCell(idim, hdim, 1., 1., 0.01, key=key)
    cornn = RNN(cornn_cell, 1, key=key)
    print("coRNN")
    print(cornn(x))
    hcornn_cell = CoupledOscillatoryRNNCell(idim, hdim, (0, 1.), (-1., 1.), dt=0.01, heterogeneous=True, key=key)
    hcornn = RNN(hcornn_cell, 1, key=key)
    print("hcoRNN")
    print(hcornn(x))
    # try the bidirectional impl.
    print("Bidirectional LSTM")
    bidirectional_lstm = BidirectionalRNN(lstm_cell, 1, key=key)
    print(bidirectional_lstm(x))
