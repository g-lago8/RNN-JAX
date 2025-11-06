"""Implement a base RNN class using jax.lax.scan on custom cells implemented in cells/
    """
import sys
sys.path.append("..")
sys.path.append("../cells")
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Inexact, Array
from cells.base import BaseCell

def get_default_dtype(complex: bool = False):
    """Return default dtype respecting current JAX precision setting."""

    
class RNN(eqx.Module):
    cell: BaseCell
    odim: int
    out_layer: eqx.nn.Linear
    def __init__(self, cell, odim, *, key):
        self.cell = cell
        self.odim = odim
        out_key, key = jr.split(key)
        self.out_layer = eqx.nn.Linear(self.cell.hdim, odim, key=key)

    def __call__(self, x:Inexact[Array, "seq_len idim"]):
        scan_fn = lambda state, x_t : self.cell(x_t, state)
        dtype = jnp.complex64 if self.cell.complex_state else jnp.float32
        print(dtype)
        initial_state = tuple(jnp.zeros(s, dtype=dtype) for s in self.cell.states_shapes)
        last_state, all_outs = jax.lax.scan(scan_fn, initial_state, x)
        return self.out_layer(all_outs[-1])


if __name__ == "__main__":
    from cells import UnitaryEvolutionRNNCell, LongShortTermMemory, ElmanRNNCell
    key = jr.key(0)
    urnn_cell = UnitaryEvolutionRNNCell(10, 16, use_bias_in=True, key = key)
    urnn = RNN(urnn_cell, 1, key=key)
    x = jr.normal(key, (100, 10))
    print(urnn(x))
    lstm_cell = LongShortTermMemory(10, 16, key=key)
    lstm = RNN(lstm_cell, 1, key=key)
    print(lstm(x))
    rnn_cell = ElmanRNNCell(10, 16, key=key)
    rnn = RNN(rnn_cell, 1, key=key)
    print(rnn(x))

