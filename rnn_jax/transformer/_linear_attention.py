"""
Implementation of Linear Attention (Katharopolous et al., 2020)
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array
from typing import Callable
import einops

class LinearAttentionLayer(eqx.Module):
    d_model: int
    n_heads: int
    d_head: int
    activation: Callable
    wq: Array
    wk: Array
    wv: Array
    mlp: eqx.nn.Sequential

    def __init__(self, d_model: int, d_head: int, n_heads: int, *, key: Array):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.activation = lambda x: jax.nn.elu(x) + 1
        wq_key, wk_key, wv_key = jax.random.split(key, 3)
        self.wq = jax.random.normal(wq_key, (n_heads, self.d_head, d_model))
        self.wk = jax.random.normal(wk_key, (n_heads, self.d_head, d_model))
        self.wv = jax.random.normal(wv_key, (n_heads, self.d_head, d_model))
        self.mlp = eqx.nn.Sequential([
            eqx.nn.Linear(d_head * n_heads, d_model, key=key),
            eqx.nn.Lambda(lambda x: jax.nn.gelu(x)),
            eqx.nn.Linear(d_model, d_head * n_heads, key=key),
        ])

    def sequential_cell(self, state, x):
        """Sequential application of lin-attention

        Args:
            state (Tuple[Array (n_heads, d_head, d_head), Array (n_heads, d_head)]): S and Z states
            x (Tuple[Array, Array, Array]): Query, Key and Values, each of shape (n_heads, d_head)

        Returns:
            Tuple[Tuple[Array, Array], Array]: Updated state and output
        """
        s, z = state 
        q, k, v = x
        k = self.activation(k)
        s += einops.einsum(k, v, "n_heads d_head_k, n_heads d_head_v -> n_heads d_head_k d_head_v")  # (n_heads, d_head, d_head)
        z += k # (n_heads, d_head)
        q = self.activation(q)
        sq = einops.einsum(s, q, "n_heads d_head_k d_head_v, n_heads d_head_q -> n_heads d_head_v") 
        zq = einops.einsum(z, q, "n_heads d_head_k, n_heads d_head_q -> n_heads")
        out = sq / zq[:, None]  # (n_heads, d_head)
        return (s, z), out

    def call_sequential(self, xs):
        cell = lambda state, x: self.sequential_cell(state, x) # this must be done because of a bug in jax (see issue #13554)
        # xs: (seq_len, d_model)
        q = einops.einsum(self.wq, xs, "n_heads d_head d_model, seq_len d_model -> seq_len n_heads d_head")  # (seq_len, n_heads, d_head)
        k = einops.einsum(self.wk, xs, "n_heads d_head d_model, seq_len d_model -> seq_len n_heads d_head")  # (seq_len, n_heads, d_head)
        v = einops.einsum(self.wv, xs, "n_heads d_head d_model, seq_len d_model -> seq_len n_heads d_head")  # (seq_len, n_heads, d_head)
        # scan over the sequence length dimension
        init_state = (jnp.zeros((self.n_heads, self.d_head, self.d_head)), jnp.zeros((self.n_heads, self.d_head)))  # (s, z)
        _, output = jax.lax.scan(cell, init_state, (q, k, v)) # vmap on the heads, scan on the sequence length
        output = einops.rearrange(output, "seq_len n_heads d_head -> seq_len (n_heads d_head)")  
        output = eqx.filter_vmap(self.mlp)(output)
        return output
    