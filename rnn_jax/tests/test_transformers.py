import importlib.util
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from rnn_jax.transformer import LinearAttentionLayer

# ---------------------------------------------------------------------------
# Fixtures / Constants
# ---------------------------------------------------------------------------
D_MODEL = 12
N_HEADS = 3
D_HEAD = 4
SEQ_LEN = 10
BATCH_SIZE = 5
KEY = jr.key(42)


def _make_layer(key=KEY):
    """Construct a linear attention layer instance."""
    return LinearAttentionLayer(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        key=key
    )


def _make_input(key, seq_len=SEQ_LEN, d_model=D_MODEL):
    """Create a random input sequence of shape (seq_len, d_model)."""
    return jr.normal(key, (seq_len, d_model))


def _zero_state():
    """Initial recurrent state for sequential linear attention."""
    s = jnp.zeros((N_HEADS, D_HEAD, D_HEAD))
    z = jnp.zeros((N_HEADS, D_HEAD))
    return s, z


class TestLinearAttentionLayer:
    """Tests for the linear attention transformer layer.
    """
    def test_init(self):
        layer = _make_layer()
        assert layer.d_model == D_MODEL
        assert layer.n_heads == N_HEADS
        assert layer.d_head == D_HEAD
        assert layer.wq.shape == (N_HEADS, D_HEAD, D_MODEL)
        assert layer.wk.shape == (N_HEADS, D_HEAD, D_MODEL)
        assert layer.wv.shape == (N_HEADS, D_HEAD, D_MODEL)

    def test_activation_positive(self):
        layer = _make_layer()
        x = jr.normal(KEY, (N_HEADS, D_HEAD))
        y = layer.activation(x)
        assert jnp.all(y > 0)

    def test_sequential_cell_shapes(self):
        layer = _make_layer()
        state = _zero_state()
        q = jr.normal(KEY, (N_HEADS, D_HEAD))
        k = jr.normal(KEY, (N_HEADS, D_HEAD))
        v = jr.normal(KEY, (N_HEADS, D_HEAD))

        new_state, out = layer.sequential_cell(state, (q, k, v))

        assert len(new_state) == 2
        assert new_state[0].shape == (N_HEADS, D_HEAD, D_HEAD)
        assert new_state[1].shape == (N_HEADS, D_HEAD)
        assert out.shape == (N_HEADS, D_HEAD)

    def test_sequential_cell_updates_state(self):
        layer = _make_layer()
        state = _zero_state()
        q = jr.normal(KEY, (N_HEADS, D_HEAD))
        k = jr.normal(jr.key(1), (N_HEADS, D_HEAD))
        v = jr.normal(jr.key(2), (N_HEADS, D_HEAD))

        new_state, _ = layer.sequential_cell(state, (q, k, v))

        assert jnp.any(jnp.abs(new_state[0]) > 0)
        assert jnp.any(jnp.abs(new_state[1]) > 0)

    def test_forward_shape(self):
        layer = _make_layer()
        x = _make_input(KEY)
        y = layer.call_sequential(x)
        assert y.shape == (SEQ_LEN, D_HEAD * N_HEADS)

	
    def test_seq_len_one(self):
        layer = _make_layer()
        x = _make_input(KEY, seq_len=1)
        y = layer.call_sequential(x)
        assert y.shape == (1, D_HEAD * N_HEADS)

    def test_jit(self):
        layer = _make_layer()
        x = _make_input(KEY)
        jit_fn = eqx.filter_jit(layer.call_sequential)
        y = jit_fn(x)
        assert y.shape == (SEQ_LEN, D_HEAD * N_HEADS)

    def test_grad(self):
        layer = _make_layer()
        x = _make_input(KEY)

        @eqx.filter_value_and_grad
        def loss_fn(model):
            return jnp.sum(model.call_sequential(x))

        val, grads = loss_fn(layer)
        assert jnp.isfinite(val)
        assert grads.wq.shape == layer.wq.shape
        assert grads.wk.shape == layer.wk.shape
        assert grads.wv.shape == layer.wv.shape

    def test_determinism(self):
        layer = _make_layer()
        x = _make_input(KEY)
        y1 = layer.call_sequential(x)
        y2 = layer.call_sequential(x)
        assert jnp.allclose(y1, y2)

    def test_different_inputs_different_outputs(self):
        layer = _make_layer()
        x1 = _make_input(KEY)
        x2 = _make_input(jr.key(1))
        y1 = layer.call_sequential(x1)
        y2 = layer.call_sequential(x2)
        assert not jnp.allclose(y1, y2)

    def test_vmap(self):
        layer = _make_layer()
        x = jax.vmap(_make_input)(jnp.repeat(KEY, BATCH_SIZE, axis=0))  # (BATCH_SIZE, seq_len, d_model)
        vmap_fn = eqx.filter_vmap(layer.call_sequential)
        y = vmap_fn(x)
        assert y.shape == (BATCH_SIZE, SEQ_LEN, D_HEAD * N_HEADS)