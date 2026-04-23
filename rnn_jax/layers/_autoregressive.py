from typing import Sequence

import jax
import jax.numpy as jnp
import equinox as eqx
from rnn_jax.cells._base import BaseCell
from rnn_jax.layers._encoder import RNNEncoder
from rnn_jax.utils.utils import concat_real_imag

class AutoregressiveRNN(eqx.Module):
    layers: Sequence[BaseCell]
    n_layers: int
    hdim: Sequence[int]
    odim: int
    out_layer: eqx.nn.Linear

    def __init__(self, layers: Sequence[BaseCell], *, key):
        self.layers = layers
        self.n_layers = len(layers)
        self.hdim = [l.hdim for l in layers]
        self.odim = self.layers[0].idim # output dimension must match input dimension for autoregressive
        out_key, key = jax.random.split(key)
        self.out_layer = eqx.nn.Linear(self.hdim[-1], self.odim, key=out_key)
    
    def open_loop(self, xs, initial_states=None):
        """Performs a single step of the RNN in open loop, i.e., feeding the input x to the first layer"""
        last_states = []

        for i, cell in enumerate(self.layers):
            scan_fn = lambda state, x_t: cell(x_t, state)
            dtype = jnp.complex64 if cell.complex_state else jnp.float32

            if initial_states is None:
                # initialize the state to zeros
                initial_state = tuple(
                    jnp.zeros(s, dtype=dtype) for s in cell.states_shapes
                )

            else:
                initial_state = initial_states[i]
            last_state, all_outs = jax.lax.scan(scan_fn, initial_state, xs)
            if cell.complex_state:
                xs = concat_real_imag(all_outs)
            else:
                xs = all_outs
            last_states.append(last_state)
        
        return tuple(last_states), jax.vmap(self.out_layer)(xs)
    

    def __call__(self, xs, n_steps):
        """Applies the autoregressive RNN for n_steps, starting from an input sequence xs

        Args:
            xs (Array): input sequence, of shape (seq_len, idim)
            n_steps (int): number of autoregressive steps to perform after the initial sequence
        Returns:
            Array: output sequence, of shape (seq_len + n_steps, odim)
        """
        initial_state, xs = self.open_loop(xs)
        carry = (initial_state, xs[-1])
        
        # Materialize attributes to avoid errors with JIT
        n_layers = self.n_layers
        layers = self.layers
        out_layer = self.out_layer

        def _step_autoregressive(carry, ins):
            states, x = carry
            new_states = []
            for i in range(n_layers):
                new_state, x = layers[i](x, states[i])
                new_states.append(new_state)
            x = out_layer(x)
            return (tuple(new_states), x), x

        _, autoregressive_outputs = jax.lax.scan(_step_autoregressive, carry, None, length=n_steps)
        return jnp.concatenate([xs, autoregressive_outputs], axis=0)
    
if __name__ == "__main__":
    from rnn_jax.cells import ElmanRNNCell, LeakyElmanCell
    key = jax.random.PRNGKey(0)
    cell1 = ElmanRNNCell(10, 20, key=key)
    cell2 = LeakyElmanCell(20, 20, key=key)
    rnn = AutoregressiveRNN([cell1, cell2], key=key)
    x = jax.random.normal(key, (5, 10)) # (seq_len, idim)
    out = rnn(x, n_steps=3)
    print(out[-1]) # print last output  
    print(out.shape)
    new_key = jax.random.PRNGKey(1)
    new_layers = [LeakyElmanCell(10, 20, key=new_key), LeakyElmanCell(20, 20, key=new_key)]
    new_rnn = eqx.tree_at(lambda r: r.layers, rnn, new_layers)
    new_out = new_rnn(x, n_steps=3)
    print(new_out[-1]) # print last output of new rnn
    print(new_out.shape)