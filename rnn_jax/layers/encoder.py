import sys
from typing import Sequence

sys.path.append("..")
sys.path.append("../cells")
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Inexact, Array, Complex
from rnn_jax.cells.base import BaseCell
from rnn_jax.utils.utils import concat_real_imag
jax.config.update("jax_debug_nans", 'true')



class RNNEncoder(eqx.Module):
    cell: BaseCell
    hdim: int

    def __init__(self, cell: BaseCell, *, key=None):
        """RNNEncoder, implemented with `jax.lax.scan`.

        This class takes a cell and iterates it through an input sequence, from first to last

        Args:
            cell (BaseCell): the cell implementing the logic of a single forward pass in time
            key (PRNGKeyArray): random key
        """
        self.cell = cell
        self.hdim = (
            self.cell.hdim if not self.cell.complex_state else self.cell.hdim * 2
        )

    def __call__(self, x: Inexact[Array, "seq_len idim"]):
        """Calls the cell on an input sequence x

        Args:
            x (Array): input sequence, an array of shape (seq_len, idim)

        Returns:
            y (Array): output array, obtained applying the output transformation to the last state of the network
        """
        scan_fn = lambda state, x_t: self.cell(x_t, state)
        dtype = jnp.complex64 if self.cell.complex_state else jnp.float32
        initial_state = tuple(
            jnp.zeros(s, dtype=dtype) for s in self.cell.states_shapes
        )
        last_state, all_outs = jax.lax.scan(scan_fn, initial_state, x)
        if self.cell.complex_state:
            all_outs = concat_real_imag(all_outs)
        return all_outs
    

class BidirectionalRNNEncoder(eqx.Module):
    cell: BaseCell
    hdim: int

    def __init__(self, cell, *, key=None):
        """Bidirectional RNN, implemented using `jax.lax.scan`.
        This class takes a cell and iterates it through an input sequence, 
        from first to last and from last to first.

        Args:
            cell (BaseCell): the cell implementing the logic of a single forward pass in time
            odim (int): output dimension
            key (PRNGKeyArray): random key
        """
        self.cell = cell
        self.hdim = (
            self.cell.hdim * 2 if not self.cell.complex_state else self.cell.hdim * 4
        )
        
    def __call__(self, x: Inexact[Array, "seq_len idim"]):
        """Calls the cell on an input sequence x, in both directions

        Args:
            x (Array): input sequence, an array of shape (seq_len, idim)

        Returns:
            y (Array): output array, obtained applying the output transformation
            to a concatenation of the two states obtained iterating the network
            from first to last and form last to first
        """
        scan_fn = lambda state, x_t: self.cell(x_t, state)
        dtype = jnp.complex64 if self.cell.complex_state else jnp.float32
        initial_state = tuple(
            jnp.zeros(s, dtype=dtype) for s in self.cell.states_shapes
        )
        last_state, outs = jax.lax.scan(scan_fn, initial_state, x)
        last_state_reverse, outs_reverse = jax.lax.scan(scan_fn, initial_state, x[::-1])
        if self.cell.complex_state:
            outs = concat_real_imag(outs)
            outs_reverse = concat_real_imag(outs_reverse)
        return outs, outs_reverse
