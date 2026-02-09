"""
Test cases for State Space Model (SSM) components in rnn_jax.

Tests cover:
  - BaseSSMLayer interface (via concrete subclasses)
  - LinearRecurrentUnit (LRU): init, discretize, full forward, jit, grad
  - SimplifiedStateSpaceLayer (S5): init, discretize, full forward, jit, grad
  - Mixers: IdentityMixer, GLUMixer, FFNMixer, NonLinearIdentityMixer
  - DeepStateSpaceModelEncoder: stacking, pooling, jit, grad
  - Edge cases: sequence length 1, validation errors, different nonlinearities
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
import pytest

from rnn_jax.ssm import (
    BaseSSMLayer,
    LinearRecurrentUnit,
    SimplifiedStateSpaceLayer,
    GLUMixer,
    IdentityMixer,
    DeepStateSpaceModelEncoder,
)
from rnn_jax.ssm.mixers import FFNMixer, NonLinearIdentityMixer, Mixer
from rnn_jax.ssm.s5 import leg_s_matrix, leg_n_matrix

# ---------------------------------------------------------------------------
# Fixtures / Constants
# ---------------------------------------------------------------------------
IN_DIM = 6
STATE_DIM = 8
MODEL_DIM = 10
ODIM = 4
SEQ_LEN = 20
KEY = jr.key(42)


def _make_input(key, seq_len=SEQ_LEN, in_dim=IN_DIM):
    """Random input sequence of shape (seq_len, in_dim)."""
    return jr.normal(key, (seq_len, in_dim))


# ===================================================================
# HELPER / MATRIX TESTS
# ===================================================================


class TestLegMatrices:
    """Tests for the LegS and LegN helper matrices used by S5."""

    def test_leg_s_shape(self):
        A = leg_s_matrix(STATE_DIM)
        assert A.shape == (STATE_DIM, STATE_DIM)

    def test_leg_s_lower_triangular_plus_diag(self):
        """LegS should be zero above the diagonal (strictly upper part)."""
        A = leg_s_matrix(STATE_DIM)
        # entries above the diagonal should be zero
        assert np.allclose(np.triu(A, k=1), 0.0)

    def test_leg_n_shape(self):
        A = leg_n_matrix(STATE_DIM)
        assert A.shape == (STATE_DIM, STATE_DIM)

    def test_leg_n_skew_symmetric_off_diagonal(self):
        """Off-diagonal part of LegN should be antisymmetric."""
        A = leg_n_matrix(STATE_DIM)
        off = A - np.diag(np.diag(A))
        assert np.allclose(off, -off.T, atol=1e-12)


# ===================================================================
# LINEAR RECURRENT UNIT (LRU) TESTS
# ===================================================================


class TestLinearRecurrentUnit:
    """Tests for the LRU layer."""

    def test_init(self):
        lru = LinearRecurrentUnit(IN_DIM, STATE_DIM, key=KEY)
        assert lru.in_dim == IN_DIM
        assert lru.state_dim == STATE_DIM
        # when model_dim is not provided, default to state_dim
        assert lru.model_dim == STATE_DIM

    def test_init_custom_model_dim(self):
        lru = LinearRecurrentUnit(IN_DIM, STATE_DIM, model_dim=MODEL_DIM, key=KEY)
        assert lru.model_dim == MODEL_DIM

    def test_parameter_shapes(self):
        lru = LinearRecurrentUnit(IN_DIM, STATE_DIM, model_dim=MODEL_DIM, key=KEY)
        assert lru.nu_log.shape == (STATE_DIM,)
        assert lru.theta_log.shape == (STATE_DIM,)
        assert lru.W_in.shape == (STATE_DIM, IN_DIM)
        assert lru.W_out.shape == (MODEL_DIM, STATE_DIM)
        assert lru.W_skip.shape == (MODEL_DIM, IN_DIM)
        assert lru.gamma_log.shape == (STATE_DIM,)

    def test_complex_weights(self):
        lru = LinearRecurrentUnit(IN_DIM, STATE_DIM, key=KEY)
        assert jnp.iscomplexobj(lru.W_in)
        assert jnp.iscomplexobj(lru.W_out)

    def test_discretize_shapes(self):
        lru = LinearRecurrentUnit(IN_DIM, STATE_DIM, key=KEY)
        lambda_elements, w_in = lru.discretize(SEQ_LEN)
        # lambda_elements comes back as a 1-tuple from lru.discretize
        assert w_in.shape == (STATE_DIM, IN_DIM)

    def test_forward_shape_default_model_dim(self):
        lru = LinearRecurrentUnit(IN_DIM, STATE_DIM, key=KEY)
        x = _make_input(KEY)
        y = lru(x)
        assert y.shape == (SEQ_LEN, STATE_DIM)

    def test_forward_shape_custom_model_dim(self):
        lru = LinearRecurrentUnit(IN_DIM, STATE_DIM, model_dim=MODEL_DIM, key=KEY)
        x = _make_input(KEY)
        y = lru(x)
        assert y.shape == (SEQ_LEN, MODEL_DIM)

    def test_output_is_real(self):
        lru = LinearRecurrentUnit(IN_DIM, STATE_DIM, key=KEY)
        x = _make_input(KEY)
        y = lru(x)
        assert not jnp.iscomplexobj(y)

    def test_jit(self):
        lru = LinearRecurrentUnit(IN_DIM, STATE_DIM, key=KEY)
        x = _make_input(KEY)
        y = eqx.filter_jit(lru)(x)
        assert y.shape == (SEQ_LEN, STATE_DIM)

    def test_grad(self):
        lru = LinearRecurrentUnit(IN_DIM, STATE_DIM, key=KEY)
        x = _make_input(KEY)

        @eqx.filter_value_and_grad
        def loss_fn(model):
            return jnp.sum(model(x))

        val, grads = loss_fn(lru)
        assert jnp.isfinite(val)

    def test_seq_len_one(self):
        lru = LinearRecurrentUnit(IN_DIM, STATE_DIM, key=KEY)
        x = _make_input(KEY, seq_len=1)
        y = lru(x)
        assert y.shape == (1, STATE_DIM)

    def test_rho_range_validation(self):
        with pytest.raises(ValueError, match="rho_max must be larger"):
            LinearRecurrentUnit(IN_DIM, STATE_DIM, rho_min=0.99, rho_max=0.5, key=KEY)

    def test_rho_bounds_validation(self):
        with pytest.raises(ValueError, match="rho_min and rho_max must be in"):
            LinearRecurrentUnit(IN_DIM, STATE_DIM, rho_min=-0.1, rho_max=0.9, key=KEY)

    def test_theta_range_validation(self):
        with pytest.raises(ValueError, match="theta_max must be larger"):
            LinearRecurrentUnit(
                IN_DIM, STATE_DIM, theta_min=3.0, theta_max=1.0, key=KEY
            )

    def test_custom_nonlinearity(self):
        lru = LinearRecurrentUnit(
            IN_DIM, STATE_DIM, nonlinearity=jax.nn.relu, key=KEY
        )
        x = _make_input(KEY)
        y = lru(x)
        # relu output should be >= 0
        assert jnp.all(y >= 0)

    def test_determinism(self):
        lru = LinearRecurrentUnit(IN_DIM, STATE_DIM, key=KEY)
        x = _make_input(KEY)
        y1 = lru(x)
        y2 = lru(x)
        assert jnp.allclose(y1, y2)


# ===================================================================
# SIMPLIFIED STATE SPACE LAYER (S5) TESTS
# ===================================================================


class TestSimplifiedStateSpaceLayer:
    """Tests for the S5 layer."""

    def test_init(self):
        s5 = SimplifiedStateSpaceLayer(IN_DIM, STATE_DIM, key=KEY)
        assert s5.in_dim == IN_DIM
        assert s5.state_dim == STATE_DIM
        assert s5.model_dim == STATE_DIM  # default

    def test_init_custom_model_dim(self):
        s5 = SimplifiedStateSpaceLayer(IN_DIM, STATE_DIM, model_dim=MODEL_DIM, key=KEY)
        assert s5.model_dim == MODEL_DIM

    def test_parameter_shapes(self):
        s5 = SimplifiedStateSpaceLayer(IN_DIM, STATE_DIM, model_dim=MODEL_DIM, key=KEY)
        assert s5.Lambda.shape == (STATE_DIM,)
        assert s5.W_in.shape == (STATE_DIM, IN_DIM)
        assert s5.W_out.shape == (MODEL_DIM, STATE_DIM)
        assert s5.W_skip.shape == (MODEL_DIM, IN_DIM)

    def test_discretize_shapes(self):
        s5 = SimplifiedStateSpaceLayer(IN_DIM, STATE_DIM, key=KEY)
        lambda_elements, w_in = s5.discretize(SEQ_LEN)
        assert lambda_elements.shape == (SEQ_LEN, STATE_DIM)
        assert w_in.shape == (STATE_DIM, IN_DIM)

    def test_forward_shape(self):
        s5 = SimplifiedStateSpaceLayer(IN_DIM, STATE_DIM, key=KEY)
        x = _make_input(KEY)
        y = s5(x)
        assert y.shape == (SEQ_LEN, STATE_DIM)

    def test_forward_shape_custom_model_dim(self):
        s5 = SimplifiedStateSpaceLayer(IN_DIM, STATE_DIM, model_dim=MODEL_DIM, key=KEY)
        x = _make_input(KEY)
        y = s5(x)
        assert y.shape == (SEQ_LEN, MODEL_DIM)

    def test_output_is_real(self):
        s5 = SimplifiedStateSpaceLayer(IN_DIM, STATE_DIM, key=KEY)
        x = _make_input(KEY)
        y = s5(x)
        assert not jnp.iscomplexobj(y)

    def test_jit(self):
        s5 = SimplifiedStateSpaceLayer(IN_DIM, STATE_DIM, key=KEY)
        x = _make_input(KEY)
        y = eqx.filter_jit(s5)(x)
        assert y.shape == (SEQ_LEN, STATE_DIM)

    def test_grad(self):
        s5 = SimplifiedStateSpaceLayer(IN_DIM, STATE_DIM, key=KEY)
        x = _make_input(KEY)

        @eqx.filter_value_and_grad
        def loss_fn(model):
            return jnp.sum(model(x))

        val, grads = loss_fn(s5)
        assert jnp.isfinite(val)

    def test_seq_len_one(self):
        s5 = SimplifiedStateSpaceLayer(IN_DIM, STATE_DIM, key=KEY)
        x = _make_input(KEY, seq_len=1)
        y = s5(x)
        assert y.shape == (1, STATE_DIM)

    def test_multi_block_init(self):
        """S5 with multiple blocks for the Lambda matrix."""
        # state_dim must be divisible by n_blocks
        s5 = SimplifiedStateSpaceLayer(
            IN_DIM, STATE_DIM, blocks_lambda=2, key=KEY
        )
        x = _make_input(KEY)
        y = s5(x)
        assert y.shape == (SEQ_LEN, STATE_DIM)

    def test_blocks_not_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            SimplifiedStateSpaceLayer(IN_DIM, 7, blocks_lambda=2, key=KEY)

    def test_unknown_init_raises(self):
        with pytest.raises(NotImplementedError):
            SimplifiedStateSpaceLayer(
                IN_DIM, STATE_DIM, init_w_h="unknown", key=KEY
            )

    def test_custom_nonlinearity(self):
        s5 = SimplifiedStateSpaceLayer(
            IN_DIM, STATE_DIM, nonlinearity=jax.nn.relu, key=KEY
        )
        x = _make_input(KEY)
        y = s5(x)
        assert jnp.all(y >= 0)

    def test_determinism(self):
        s5 = SimplifiedStateSpaceLayer(IN_DIM, STATE_DIM, key=KEY)
        x = _make_input(KEY)
        y1 = s5(x)
        y2 = s5(x)
        assert jnp.allclose(y1, y2)


# ===================================================================
# MIXER TESTS
# ===================================================================


class TestIdentityMixer:
    """Tests for the IdentityMixer."""

    def test_real_passthrough(self):
        mixer = IdentityMixer(key=KEY)
        x = jr.normal(KEY, (MODEL_DIM,))
        assert jnp.allclose(mixer(x), x)

    def test_complex_returns_real(self):
        mixer = IdentityMixer(key=KEY)
        x = jr.normal(KEY, (MODEL_DIM,)) + 1j * jr.normal(jr.key(1), (MODEL_DIM,))
        y = mixer(x)
        assert not jnp.iscomplexobj(y)
        assert jnp.allclose(y, x.real)


class TestGLUMixer:
    """Tests for the GLU (Gated Linear Unit) mixer."""

    def test_init(self):
        mixer = GLUMixer(IN_DIM, MODEL_DIM, key=KEY)
        assert mixer.gate_projection.in_features == IN_DIM
        assert mixer.main_projection.out_features == MODEL_DIM

    def test_forward_shape(self):
        mixer = GLUMixer(IN_DIM, MODEL_DIM, key=KEY)
        x = jr.normal(KEY, (IN_DIM,))
        y = mixer(x)
        assert y.shape == (MODEL_DIM,)

    def test_jit(self):
        mixer = GLUMixer(IN_DIM, MODEL_DIM, key=KEY)
        x = jr.normal(KEY, (IN_DIM,))
        y = eqx.filter_jit(mixer)(x)
        assert y.shape == (MODEL_DIM,)

    def test_vmap(self):
        mixer = GLUMixer(IN_DIM, MODEL_DIM, key=KEY)
        xs = jr.normal(KEY, (SEQ_LEN, IN_DIM))
        ys = eqx.filter_vmap(mixer)(xs)
        assert ys.shape == (SEQ_LEN, MODEL_DIM)


class TestFFNMixer:
    """Tests for the feed-forward network mixer."""

    def test_forward_shape(self):
        mixer = FFNMixer(IN_DIM, 2 * IN_DIM, MODEL_DIM, key=KEY)
        x = jr.normal(KEY, (IN_DIM,))
        y = mixer(x)
        assert y.shape == (MODEL_DIM,)

    def test_jit(self):
        mixer = FFNMixer(IN_DIM, 2 * IN_DIM, MODEL_DIM, key=KEY)
        x = jr.normal(KEY, (IN_DIM,))
        y = eqx.filter_jit(mixer)(x)
        assert y.shape == (MODEL_DIM,)


class TestNonLinearIdentityMixer:
    """Tests for the NonLinearIdentityMixer."""

    def test_applies_nonlinearity(self):
        mixer = NonLinearIdentityMixer(nonlinearity=jax.nn.relu, key=KEY)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = mixer(x)
        assert jnp.allclose(y, jnp.array([0.0, 0.0, 1.0]))

    def test_complex_to_real(self):
        mixer = NonLinearIdentityMixer(nonlinearity=jax.nn.relu, key=KEY)
        x = jnp.array([1.0 + 2j, -1.0 + 0j])
        y = mixer(x)
        assert not jnp.iscomplexobj(y)


# ===================================================================
# DEEP STATE SPACE MODEL ENCODER TESTS
# ===================================================================


class TestDeepStateSpaceModelEncoder:
    """Tests for stacking SSM layers with mixers."""

    def _make_encoder(self, n_layers=2, pool_method="none", out_dim=None):
        """Helper to build a DeepStateSpaceModelEncoder with n LRU layers."""
        keys = jr.split(KEY, 2 * n_layers + 1)
        layers = []
        mixers = []
        first_model_dim = STATE_DIM
        for i in range(n_layers):
            layers.append(
                LinearRecurrentUnit(
                    in_dim=first_model_dim,
                    state_dim=STATE_DIM,
                    model_dim=first_model_dim,
                    key=keys[i],
                )
            )
            mixers.append(IdentityMixer(key=keys[n_layers + i]))
        return DeepStateSpaceModelEncoder(
            in_dim=IN_DIM,
            layers=layers,
            mixers=mixers,
            in_projection=True,
            out_dim=out_dim,
            pool_method=pool_method,
            key=keys[-1],
        )

    def test_init(self):
        model = self._make_encoder()
        assert model.n_layers == 2
        assert model.in_dim == IN_DIM

    def test_forward_shape_no_pool(self):
        model = self._make_encoder(pool_method="none")
        x = _make_input(KEY)
        y = model(x)
        assert y.shape == (SEQ_LEN, STATE_DIM)

    def test_forward_shape_mean_pool(self):
        model = self._make_encoder(pool_method="mean", out_dim=ODIM)
        x = _make_input(KEY)
        y = model(x)
        assert y.shape == (ODIM,)

    def test_forward_shape_last_pool(self):
        model = self._make_encoder(pool_method="last", out_dim=ODIM)
        x = _make_input(KEY)
        y = model(x)
        assert y.shape == (ODIM,)

    def test_forward_shape_with_out_dim(self):
        model = self._make_encoder(pool_method="none", out_dim=ODIM)
        x = _make_input(KEY)
        y = model(x)
        assert y.shape == (SEQ_LEN, ODIM)

    def test_no_in_projection(self):
        """When in_projection=False, identity is used."""
        keys = jr.split(KEY, 3)
        lru = LinearRecurrentUnit(IN_DIM, STATE_DIM, model_dim=IN_DIM, key=keys[0])
        mixer = IdentityMixer(key=keys[1])
        model = DeepStateSpaceModelEncoder(
            in_dim=IN_DIM,
            layers=[lru],
            mixers=[mixer],
            in_projection=False,
            pool_method="none",
            key=keys[2],
        )
        x = _make_input(KEY)
        y = model(x)
        assert y.shape == (SEQ_LEN, IN_DIM)

    def test_jit(self):
        model = self._make_encoder()
        x = _make_input(KEY)
        y = eqx.filter_jit(model)(x)
        assert y.shape == (SEQ_LEN, STATE_DIM)

    def test_grad(self):
        model = self._make_encoder()
        x = _make_input(KEY)

        @eqx.filter_value_and_grad
        def loss_fn(m):
            return jnp.sum(m(x))

        val, grads = loss_fn(model)
        assert jnp.isfinite(val)

    def test_mismatched_layers_mixers_raises(self):
        k1, k2 = jr.split(KEY)
        lru = LinearRecurrentUnit(STATE_DIM, STATE_DIM, key=k1)
        with pytest.raises(AssertionError):
            DeepStateSpaceModelEncoder(
                in_dim=IN_DIM,
                layers=[lru],
                mixers=[],  # no mixers
                key=k2,
            )

    def test_with_glu_mixer(self):
        """Use GLUMixer instead of IdentityMixer."""
        keys = jr.split(KEY, 4)
        lru = LinearRecurrentUnit(STATE_DIM, STATE_DIM, model_dim=STATE_DIM, key=keys[0])
        mixer = GLUMixer(STATE_DIM, STATE_DIM, key=keys[1])
        model = DeepStateSpaceModelEncoder(
            in_dim=IN_DIM,
            layers=[lru],
            mixers=[mixer],
            in_projection=True,
            pool_method="none",
            key=keys[2],
        )
        x = _make_input(KEY)
        y = model(x)
        assert y.shape == (SEQ_LEN, STATE_DIM)

    def test_with_s5_layer(self):
        """Build an encoder using S5 layers instead of LRU."""
        keys = jr.split(KEY, 4)
        s5 = SimplifiedStateSpaceLayer(STATE_DIM, STATE_DIM, model_dim=STATE_DIM, key=keys[0])
        mixer = IdentityMixer(key=keys[1])
        model = DeepStateSpaceModelEncoder(
            in_dim=IN_DIM,
            layers=[s5],
            mixers=[mixer],
            in_projection=True,
            pool_method="none",
            key=keys[2],
        )
        x = _make_input(KEY)
        y = model(x)
        assert y.shape == (SEQ_LEN, STATE_DIM)

    def test_seq_len_one(self):
        model = self._make_encoder()
        x = _make_input(KEY, seq_len=1)
        y = model(x)
        assert y.shape == (1, STATE_DIM)

    def test_determinism(self):
        model = self._make_encoder()
        x = _make_input(KEY)
        y1 = model(x)
        y2 = model(x)
        assert jnp.allclose(y1, y2)

    def test_single_layer(self):
        model = self._make_encoder(n_layers=1)
        x = _make_input(KEY)
        y = model(x)
        assert y.shape == (SEQ_LEN, STATE_DIM)

    def test_three_layers(self):
        model = self._make_encoder(n_layers=3)
        x = _make_input(KEY)
        y = model(x)
        assert y.shape == (SEQ_LEN, STATE_DIM)


# ===================================================================
# CROSS-CUTTING / INTEGRATION TESTS
# ===================================================================


class TestSSMIntegration:
    """Integration-style tests combining SSM components."""

    def test_vmap_over_batch_lru(self):
        lru = LinearRecurrentUnit(IN_DIM, STATE_DIM, key=KEY)
        batch = jr.normal(KEY, (5, SEQ_LEN, IN_DIM))
        ys = jax.vmap(lru)(batch)
        assert ys.shape == (5, SEQ_LEN, STATE_DIM)

    def test_vmap_over_batch_s5(self):
        s5 = SimplifiedStateSpaceLayer(IN_DIM, STATE_DIM, key=KEY)
        batch = jr.normal(KEY, (5, SEQ_LEN, IN_DIM))
        ys = jax.vmap(s5)(batch)
        assert ys.shape == (5, SEQ_LEN, STATE_DIM)

    def test_vmap_deep_encoder(self):
        keys = jr.split(KEY, 4)
        lru = LinearRecurrentUnit(
            STATE_DIM, STATE_DIM, model_dim=STATE_DIM, key=keys[0]
        )
        mixer = IdentityMixer(key=keys[1])
        model = DeepStateSpaceModelEncoder(
            in_dim=IN_DIM,
            layers=[lru],
            mixers=[mixer],
            in_projection=True,
            pool_method="none",
            key=keys[2],
        )
        batch = jr.normal(KEY, (3, SEQ_LEN, IN_DIM))
        ys = jax.vmap(model)(batch)
        assert ys.shape == (3, SEQ_LEN, STATE_DIM)

    def test_different_keys_different_outputs(self):
        k1, k2 = jr.split(KEY)
        lru_a = LinearRecurrentUnit(IN_DIM, STATE_DIM, key=k1)
        lru_b = LinearRecurrentUnit(IN_DIM, STATE_DIM, key=k2)
        x = _make_input(KEY)
        ya = lru_a(x)
        yb = lru_b(x)
        assert not jnp.allclose(ya, yb)

    def test_grad_deep_encoder_mean_pool(self):
        """End-to-end gradient through a deep encoder with mean pooling."""
        keys = jr.split(KEY, 5)
        layers = [
            LinearRecurrentUnit(STATE_DIM, STATE_DIM, model_dim=STATE_DIM, key=keys[0]),
            LinearRecurrentUnit(STATE_DIM, STATE_DIM, model_dim=STATE_DIM, key=keys[1]),
        ]
        mixers = [
            IdentityMixer(key=keys[2]),
            IdentityMixer(key=keys[3]),
        ]
        model = DeepStateSpaceModelEncoder(
            in_dim=IN_DIM,
            layers=layers,
            mixers=mixers,
            in_projection=True,
            out_dim=ODIM,
            pool_method="mean",
            key=keys[4],
        )
        x = _make_input(KEY)

        @eqx.filter_value_and_grad
        def loss_fn(m):
            return jnp.sum(m(x))

        val, grads = loss_fn(model)
        assert jnp.isfinite(val)
