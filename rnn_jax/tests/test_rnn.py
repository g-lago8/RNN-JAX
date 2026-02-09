"""
Test cases for RNN cells and layers in JAX.

Tests cover:
  - Every cell in rnn_jax.cells (initialization, single-step shapes, JIT, grads)
  - Every layer in rnn_jax.layers (RNN, BidirectionalRNN, DeepRNN,
    DeepBidirectionalRNN, RNNEncoder, BidirectionalRNNEncoder, ReservoirComputer)
  - Edge cases: sequence length 1, large hidden dim, different nonlinearities
"""

from typing import Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array
import pytest

from rnn_jax.cells import (
    BaseCell,
    ElmanRNNCell,
    IndRNNCell,
    LongShortTermMemoryCell,
    LongShortTermMemory,
    GatedRecurrentUnit,
    AntiSymmetricRNNCell,
    GatedAntiSymmetricRNNCell,
    CoupledOscillatoryRNNCell,
    UnitaryEvolutionRNNCell,
    LipschitzRNNCell,
    ClockWorkRNNCell,
)
from rnn_jax.layers import (
    RNN,
    BidirectionalRNN,
    DeepRNN,
    DeepBidirectionalRNN,
)
from rnn_jax.layers.encoder import RNNEncoder, BidirectionalRNNEncoder
from rnn_jax.layers.reservoir import ReservoirComputer, init_reservoir_esn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
IDIM = 8
HDIM = 16
ODIM = 4
SEQ_LEN = 20
KEY = jr.key(42)


def _make_input(key, seq_len=SEQ_LEN, idim=IDIM):
    """Create a random input sequence of shape (seq_len, idim)."""
    return jr.normal(key, (seq_len, idim))


def _zeros_state(cell)->Tuple[Array, ...]:
    """Build the zero initial state expected by a cell."""
    dtype = jnp.complex64 if cell.complex_state else jnp.float32
    return tuple(jnp.zeros(s, dtype=dtype) for s in cell.states_shapes)


# ===================================================================
# CELL TESTS
# ===================================================================


class TestElmanRNNCell:
    """Tests for the Elman (vanilla) RNN cell."""

    def test_init(self):
        cell = ElmanRNNCell(IDIM, HDIM, key=KEY)
        assert cell.idim == IDIM
        assert cell.hdim == HDIM
        assert cell.complex_state is False

    def test_forward_shapes(self):
        cell = ElmanRNNCell(IDIM, HDIM, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = cell(x, state)
        assert out.shape == (HDIM,)
        assert new_state[0].shape == (HDIM,)

    def test_jit(self):
        cell = ElmanRNNCell(IDIM, HDIM, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        jit_fn = eqx.filter_jit(cell)
        new_state, out = jit_fn(x, state)
        assert out.shape == (HDIM,)

    def test_grad(self):
        cell = ElmanRNNCell(IDIM, HDIM, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)

        @eqx.filter_value_and_grad
        def loss_fn(cell):
            _, out = cell(x, state)
            return jnp.sum(out)

        val, grads = loss_fn(cell)
        assert jnp.isfinite(val)

    def test_nonlinearity_kwarg(self):
        cell = ElmanRNNCell(IDIM, HDIM, nonlinearity=jax.nn.tanh, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        _, out = cell(x, state)
        # tanh output is bounded in (-1, 1)
        assert jnp.all(jnp.abs(out) <= 1.0)


class TestIndRNNCell:
    """Tests for the Independent RNN cell (diagonal recurrence)."""

    def test_init(self):
        cell = IndRNNCell(IDIM, HDIM, key=KEY)
        assert cell.w_hh.shape == (HDIM,)  # diagonal, not a matrix

    def test_forward_shapes(self):
        cell = IndRNNCell(IDIM, HDIM, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = cell(x, state)
        assert out.shape == (HDIM,)

    def test_jit(self):
        cell = IndRNNCell(IDIM, HDIM, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = eqx.filter_jit(cell)(x, state)
        assert out.shape == (HDIM,)


class TestLongShortTermMemoryCell:
    """Tests for the custom LSTM cell implementation."""

    def test_init(self):
        cell = LongShortTermMemoryCell(IDIM, HDIM, key=KEY)
        assert cell.idim == IDIM
        assert cell.hdim == HDIM
        # forget gate bias should be initialised to ones
        assert jnp.allclose(cell.b_f, jnp.ones(HDIM))

    def test_forward_shapes(self):
        cell = LongShortTermMemoryCell(IDIM, HDIM, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)  # (h, c)
        new_state, out = cell(x, state)
        assert len(new_state) == 2
        assert new_state[0].shape == (HDIM,)
        assert new_state[1].shape == (HDIM,)
        assert out.shape == (HDIM,)

    def test_jit(self):
        cell = LongShortTermMemoryCell(IDIM, HDIM, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = eqx.filter_jit(cell)(x, state)
        assert out.shape == (HDIM,)

    def test_grad(self):
        cell = LongShortTermMemoryCell(IDIM, HDIM, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)

        @eqx.filter_value_and_grad
        def loss_fn(cell):
            _, out = cell(x, state)
            return jnp.sum(out)

        val, grads = loss_fn(cell)
        assert jnp.isfinite(val)


class TestLongShortTermMemory:
    """Tests for the equinox-wrapper LSTM cell."""

    def test_init(self):
        cell = LongShortTermMemory(IDIM, HDIM, key=KEY)
        assert cell.idim == IDIM
        assert cell.hdim == HDIM
        assert cell.complex_state is False

    def test_forward_shapes(self):
        cell = LongShortTermMemory(IDIM, HDIM, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = cell(x, state)
        assert len(new_state) == 2
        assert out.shape == (HDIM,)

    def test_jit_and_grad(self):
        cell = LongShortTermMemory(IDIM, HDIM, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)

        @eqx.filter_value_and_grad
        def loss_fn(cell):
            _, out = cell(x, state)
            return jnp.sum(out)

        val, grads = loss_fn(cell)
        assert jnp.isfinite(val)


class TestGatedRecurrentUnit:
    """Tests for the equinox-wrapper GRU cell."""

    def test_init(self):
        cell = GatedRecurrentUnit(IDIM, HDIM, key=KEY)
        assert cell.states_shapes == (HDIM,)

    def test_forward_shapes(self):
        cell = GatedRecurrentUnit(IDIM, HDIM, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = cell(x, state)
        assert out.shape == (HDIM,)
        assert new_state[0].shape == (HDIM,)

    def test_jit(self):
        cell = GatedRecurrentUnit(IDIM, HDIM, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = eqx.filter_jit(cell)(x, state)
        assert out.shape == (HDIM,)


class TestAntiSymmetricRNNCell:
    """Tests for the Antisymmetric RNN cell."""

    def test_init(self):
        cell = AntiSymmetricRNNCell(IDIM, HDIM, stepsize=0.01, diffusion=0.1, key=KEY)
        assert cell.idim == IDIM
        assert cell.hdim == HDIM

    def test_forward_shapes(self):
        cell = AntiSymmetricRNNCell(IDIM, HDIM, stepsize=0.01, diffusion=0.1, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = cell(x, state)
        assert out.shape == (HDIM,)

    def test_jit(self):
        cell = AntiSymmetricRNNCell(IDIM, HDIM, stepsize=0.01, diffusion=0.1, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = eqx.filter_jit(cell)(x, state)
        assert out.shape == (HDIM,)


class TestGatedAntiSymmetricRNNCell:
    """Tests for the Gated Antisymmetric RNN cell."""

    def test_init(self):
        cell = GatedAntiSymmetricRNNCell(
            IDIM, HDIM, stepsize=0.01, diffusion=0.1, key=KEY
        )
        assert cell.idim == IDIM

    def test_forward_shapes(self):
        cell = GatedAntiSymmetricRNNCell(
            IDIM, HDIM, stepsize=0.01, diffusion=0.1, key=KEY
        )
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = cell(x, state)
        assert out.shape == (HDIM,)

    def test_jit(self):
        cell = GatedAntiSymmetricRNNCell(
            IDIM, HDIM, stepsize=0.01, diffusion=0.1, key=KEY
        )
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = eqx.filter_jit(cell)(x, state)
        assert out.shape == (HDIM,)


class TestCoupledOscillatoryRNNCell:
    """Tests for the coRNN cell (homogeneous and heterogeneous)."""

    def test_homogeneous_init(self):
        cell = CoupledOscillatoryRNNCell(
            IDIM, HDIM, gamma=1.0, eps=1.0, dt=0.01, key=KEY
        )
        assert cell.states_shapes == ((HDIM,), (HDIM,))

    def test_homogeneous_forward(self):
        cell = CoupledOscillatoryRNNCell(
            IDIM, HDIM, gamma=1.0, eps=1.0, dt=0.01, key=KEY
        )
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = cell(x, state)
        assert out.shape == (HDIM,)
        assert len(new_state) == 2

    def test_heterogeneous_init(self):
        cell = CoupledOscillatoryRNNCell(
            IDIM,
            HDIM,
            gamma=(0.0, 1.0),
            eps=(-1.0, 1.0),
            dt=0.01,
            heterogeneous=True,
            key=KEY,
        )
        # gamma and eps should be arrays
        assert cell.gamma.shape == (HDIM,)
        assert cell.eps.shape == (HDIM,)

    def test_heterogeneous_forward(self):
        cell = CoupledOscillatoryRNNCell(
            IDIM,
            HDIM,
            gamma=(0.0, 1.0),
            eps=(-1.0, 1.0),
            dt=0.01,
            heterogeneous=True,
            key=KEY,
        )
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = cell(x, state)
        assert out.shape == (HDIM,)

    def test_jit(self):
        cell = CoupledOscillatoryRNNCell(
            IDIM, HDIM, gamma=1.0, eps=1.0, dt=0.01, key=KEY
        )
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = eqx.filter_jit(cell)(x, state)
        assert out.shape == (HDIM,)


class TestUnitaryEvolutionRNNCell:
    """Tests for the uRNN cell (complex-valued hidden state)."""

    def test_init_identity_perm(self):
        cell = UnitaryEvolutionRNNCell(
            IDIM, HDIM, permutation_type="identity", key=KEY
        )
        assert cell.complex_state is True
        assert jnp.array_equal(cell.perm, jnp.arange(HDIM))

    def test_init_random_perm(self):
        cell = UnitaryEvolutionRNNCell(
            IDIM, HDIM, permutation_type="random", key=KEY
        )
        assert cell.complex_state is True
        # random permutation should be a permutation of arange
        assert set(cell.perm.tolist()) == set(range(HDIM))

    def test_invalid_perm_raises(self):
        with pytest.raises(ValueError):
            UnitaryEvolutionRNNCell(
                IDIM, HDIM, permutation_type="invalid", key=KEY
            )

    def test_forward_shapes(self):
        cell = UnitaryEvolutionRNNCell(IDIM, HDIM, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)  # complex zeros
        new_state, out = cell(x, state)
        assert out.shape == (HDIM,)
        assert jnp.iscomplexobj(out)

    def test_jit(self):
        cell = UnitaryEvolutionRNNCell(IDIM, HDIM, key=KEY)
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = eqx.filter_jit(cell)(x, state)
        assert out.shape == (HDIM,)


class TestLipschitzRNNCell:
    """Tests for the Lipschitz RNN cell (Euler and RK2)."""

    @pytest.fixture(params=["euler", "rk2"])
    def cell(self, request):
        return LipschitzRNNCell(
            IDIM,
            HDIM,
            beta_a=0.65,
            gamma_a=1.0,
            beta_w=0.65,
            gamma_w=1.0,
            dt=0.001,
            weight_std=1 / 16,
            discretization=request.param,
            key=KEY,
        )

    def test_init(self, cell):
        assert cell.idim == IDIM
        assert cell.hdim == HDIM

    def test_forward_shapes(self, cell):
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = cell(x, state)
        assert out.shape == (HDIM,)
        assert len(new_state) == 2  # (h, z)

    def test_jit(self, cell):
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = eqx.filter_jit(cell)(x, state)
        assert out.shape == (HDIM,)

    def test_grad(self, cell):
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)

        @eqx.filter_value_and_grad
        def loss_fn(cell):
            _, out = cell(x, state)
            return jnp.sum(out)

        val, _ = loss_fn(cell)
        assert jnp.isfinite(val)


class TestClockWorkRNNCell:
    """Tests for the ClockWork RNN cell."""

    def test_init_uniform_blocks(self):
        cell = ClockWorkRNNCell(
            IDIM,
            block_sizes=8,
            periods=[2, 4, 8],
            nonlinearity=jax.nn.relu,
            key=KEY,
        )
        assert cell.hdim == 24  # 8*3

    def test_init_variable_blocks(self):
        cell = ClockWorkRNNCell(
            IDIM,
            block_sizes=[8, 7, 9],
            periods=[2, 4, 8],
            nonlinearity=jax.nn.relu,
            key=KEY,
        )
        assert cell.hdim == 24  # 8+7+9

    def test_forward_shapes(self):
        cell = ClockWorkRNNCell(
            IDIM,
            block_sizes=[HDIM, HDIM - 1, HDIM + 1],
            periods=[2, 4, 8],
            nonlinearity=jax.nn.relu,
            key=KEY,
        )
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = cell(x, state)
        # output is the concatenation of all blocks
        assert out.shape == (cell.hdim,)

    def test_time_counter_increments(self):
        """The last element of the state is a time counter that increments."""
        cell = ClockWorkRNNCell(
            IDIM,
            block_sizes=4,
            periods=[1, 2],
            nonlinearity=jax.nn.relu,
            key=KEY,
        )
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, _ = cell(x, state)
        # time counter should be 1 after one step
        assert jnp.allclose(new_state[-1], jnp.array([1.0]))

    def test_jit(self):
        cell = ClockWorkRNNCell(
            IDIM,
            block_sizes=4,
            periods=[1, 2, 4],
            nonlinearity=jax.nn.relu,
            key=KEY,
        )
        x = jr.normal(KEY, (IDIM,))
        state = _zeros_state(cell)
        new_state, out = eqx.filter_jit(cell)(x, state)
        assert out.shape == (cell.hdim,)


# ===================================================================
# ENCODER TESTS
# ===================================================================


class TestRNNEncoder:
    """Tests for the RNNEncoder (scans a cell over a sequence)."""

    def test_output_shape_real(self):
        cell = ElmanRNNCell(IDIM, HDIM, key=KEY)
        enc = RNNEncoder(cell, key=KEY)
        x = _make_input(KEY)
        out = enc(x)
        assert out.shape == (SEQ_LEN, HDIM)

    def test_output_shape_complex(self):
        """For complex-state cells, encoder concatenates real/imag parts."""
        cell = UnitaryEvolutionRNNCell(IDIM, HDIM, key=KEY)
        enc = RNNEncoder(cell, key=KEY)
        x = _make_input(KEY)
        out = enc(x)
        # complex -> concat real + imag => 2*HDIM
        assert out.shape == (SEQ_LEN, 2 * HDIM)

    def test_custom_initial_state(self):
        cell = ElmanRNNCell(IDIM, HDIM, key=KEY)
        enc = RNNEncoder(cell, key=KEY)
        x = _make_input(KEY)
        custom_state = (jnp.ones((HDIM,)),)
        out = enc(x, initial_state=custom_state)
        assert out.shape == (SEQ_LEN, HDIM)

    def test_jit(self):
        cell = ElmanRNNCell(IDIM, HDIM, key=KEY)
        enc = RNNEncoder(cell, key=KEY)
        x = _make_input(KEY)
        out = eqx.filter_jit(enc)(x)
        assert out.shape == (SEQ_LEN, HDIM)


class TestBidirectionalRNNEncoder:
    """Tests for the BidirectionalRNNEncoder."""

    def test_output_shapes(self):
        cell = ElmanRNNCell(IDIM, HDIM, key=KEY)
        enc = BidirectionalRNNEncoder(cell, cell, key=KEY)
        x = _make_input(KEY)
        fwd, bwd = enc(x)
        assert fwd.shape == (SEQ_LEN, HDIM)
        assert bwd.shape == (SEQ_LEN, HDIM)

    def test_hdim(self):
        cell = ElmanRNNCell(IDIM, HDIM, key=KEY)
        enc = BidirectionalRNNEncoder(cell, cell, key=KEY)
        # hdim should be 2 * cell.hdim for real-valued cells
        assert enc.hdim == 2 * HDIM

    def test_jit(self):
        cell = ElmanRNNCell(IDIM, HDIM, key=KEY)
        enc = BidirectionalRNNEncoder(cell, cell, key=KEY)
        x = _make_input(KEY)
        fwd, bwd = eqx.filter_jit(enc)(x)
        assert fwd.shape == (SEQ_LEN, HDIM)


# ===================================================================
# LAYER TESTS
# ===================================================================


class TestRNN:
    """Tests for the RNN layer (encoder + output linear)."""

    @pytest.fixture(
        params=[
            "elman",
            "lstm_cell",
            "lstm_eqx",
            "gru",
            "indrnn",
            "antisym",
            "cornn",
            "lipschitz",
            "urnn",
        ]
    )
    def rnn(self, request):
        """Parametrised fixture: build an RNN layer for every cell type."""
        k1, k2 = jr.split(KEY)
        cell_map = {
            "elman": lambda: ElmanRNNCell(IDIM, HDIM, key=k1),
            "lstm_cell": lambda: LongShortTermMemoryCell(IDIM, HDIM, key=k1),
            "lstm_eqx": lambda: LongShortTermMemory(IDIM, HDIM, key=k1),
            "gru": lambda: GatedRecurrentUnit(IDIM, HDIM, key=k1),
            "indrnn": lambda: IndRNNCell(IDIM, HDIM, key=k1),
            "antisym": lambda: AntiSymmetricRNNCell(
                IDIM, HDIM, stepsize=0.01, diffusion=0.1, key=k1
            ),
            "cornn": lambda: CoupledOscillatoryRNNCell(
                IDIM, HDIM, gamma=1.0, eps=1.0, dt=0.01, key=k1
            ),
            "lipschitz": lambda: LipschitzRNNCell(
                IDIM, HDIM, 0.65, 1.0, 0.65, 1.0, 0.001, 1 / 16, key=k1
            ),
            "urnn": lambda: UnitaryEvolutionRNNCell(IDIM, HDIM, key=k1),
        }
        cell = cell_map[request.param]()
        return RNN(cell, ODIM, key=k2)

    def test_output_shape(self, rnn):
        x = _make_input(KEY)
        y, all_outs = rnn(x)
        assert y.shape == (ODIM,)
        assert all_outs.shape[0] == SEQ_LEN

    def test_jit(self, rnn):
        x = _make_input(KEY)
        y, all_outs = eqx.filter_jit(rnn)(x)
        assert y.shape == (ODIM,)

    def test_grad(self, rnn):
        x = _make_input(KEY)

        @eqx.filter_value_and_grad
        def loss_fn(model):
            y, _ = model(x)
            return jnp.sum(y)

        val, grads = loss_fn(rnn)
        assert jnp.isfinite(val)

    def test_seq_len_one(self, rnn):
        """Edge case: single time step."""
        x = _make_input(KEY, seq_len=1)
        y, all_outs = rnn(x)
        assert y.shape == (ODIM,)
        assert all_outs.shape[0] == 1


class TestRNNClockWork:
    """Separate tests for ClockWork RNN wrapped in the RNN layer
    (ClockWork has a non-standard state tuple so it deserves dedicated tests).
    """

    def test_output_shape(self):
        k1, k2 = jr.split(KEY)
        cell = ClockWorkRNNCell(
            IDIM,
            block_sizes=[HDIM, HDIM],
            periods=[1, 2],
            nonlinearity=jax.nn.relu,
            key=k1,
        )
        rnn = RNN(cell, ODIM, key=k2)
        x = _make_input(KEY)
        y, all_outs = rnn(x)
        assert y.shape == (ODIM,)

    def test_jit(self):
        k1, k2 = jr.split(KEY)
        cell = ClockWorkRNNCell(
            IDIM,
            block_sizes=[HDIM, HDIM],
            periods=[1, 2],
            nonlinearity=jax.nn.relu,
            key=k1,
        )
        rnn = RNN(cell, ODIM, key=k2)
        x = _make_input(KEY)
        y, all_outs = eqx.filter_jit(rnn)(x)
        assert y.shape == (ODIM,)


class TestBidirectionalRNN:
    """Tests for the BidirectionalRNN layer."""

    def test_output_shape(self):
        k1, k2 = jr.split(KEY)
        fw = ElmanRNNCell(IDIM, HDIM, key=k1)
        bw = ElmanRNNCell(IDIM, HDIM, key=k2)
        k3, _ = jr.split(k2)
        model = BidirectionalRNN(fw, bw, ODIM, key=k3)
        x = _make_input(KEY)
        y, (h_fwd, h_bwd) = model(x)
        assert y.shape == (ODIM,)
        assert h_fwd.shape == (SEQ_LEN, HDIM)
        assert h_bwd.shape == (SEQ_LEN, HDIM)

    def test_with_lstm(self):
        k1, k2, k3 = jr.split(KEY, 3)
        fw = LongShortTermMemory(IDIM, HDIM, key=k1)
        bw = LongShortTermMemory(IDIM, HDIM, key=k2)
        model = BidirectionalRNN(fw, bw, ODIM, key=k3)
        x = _make_input(KEY)
        y, _ = model(x)
        assert y.shape == (ODIM,)

    def test_jit(self):
        k1, k2, k3 = jr.split(KEY, 3)
        fw = ElmanRNNCell(IDIM, HDIM, key=k1)
        bw = ElmanRNNCell(IDIM, HDIM, key=k2)
        model = BidirectionalRNN(fw, bw, ODIM, key=k3)
        x = _make_input(KEY)
        y, _ = eqx.filter_jit(model)(x)
        assert y.shape == (ODIM,)

    def test_grad(self):
        k1, k2, k3 = jr.split(KEY, 3)
        fw = ElmanRNNCell(IDIM, HDIM, key=k1)
        bw = ElmanRNNCell(IDIM, HDIM, key=k2)
        model = BidirectionalRNN(fw, bw, ODIM, key=k3)
        x = _make_input(KEY)

        @eqx.filter_value_and_grad
        def loss_fn(m):
            y, _ = m(x)
            return jnp.sum(y)

        val, grads = loss_fn(model)
        assert jnp.isfinite(val)


class TestDeepRNN:
    """Tests for the DeepRNN layer (stacked encoders)."""

    def test_output_shape(self):
        k1, k2, k3 = jr.split(KEY, 3)
        cells = [
            LongShortTermMemory(IDIM, HDIM, key=k1),
            LongShortTermMemory(HDIM, HDIM, key=k2),
        ]
        model = DeepRNN(cells, ODIM, key=k3)
        x = _make_input(KEY)
        y, all_hidden = model(x)
        assert y.shape == (ODIM,)
        assert len(all_hidden) == 2

    def test_three_layers(self):
        k1, k2, k3, k4 = jr.split(KEY, 4)
        cells = [
            ElmanRNNCell(IDIM, HDIM, key=k1),
            ElmanRNNCell(HDIM, HDIM, key=k2),
            ElmanRNNCell(HDIM, HDIM, key=k3),
        ]
        model = DeepRNN(cells, ODIM, key=k4)
        x = _make_input(KEY)
        y, all_hidden = model(x)
        assert y.shape == (ODIM,)
        assert len(all_hidden) == 3

    def test_jit(self):
        k1, k2, k3 = jr.split(KEY, 3)
        cells = [
            LongShortTermMemory(IDIM, HDIM, key=k1),
            LongShortTermMemory(HDIM, HDIM, key=k2),
        ]
        model = DeepRNN(cells, ODIM, key=k3)
        x = _make_input(KEY)
        y, _ = eqx.filter_jit(model)(x)
        assert y.shape == (ODIM,)

    def test_grad(self):
        k1, k2, k3 = jr.split(KEY, 3)
        cells = [
            LongShortTermMemory(IDIM, HDIM, key=k1),
            LongShortTermMemory(HDIM, HDIM, key=k2),
        ]
        model = DeepRNN(cells, ODIM, key=k3)
        x = _make_input(KEY)

        @eqx.filter_value_and_grad
        def loss_fn(m):
            y, _ = m(x)
            return jnp.sum(y)

        val, _ = loss_fn(model)
        assert jnp.isfinite(val)

    def test_empty_layers_raises(self):
        with pytest.raises(AssertionError):
            DeepRNN([], ODIM, key=KEY)


class TestDeepBidirectionalRNN:
    """Tests for the DeepBidirectionalRNN layer."""

    def test_output_shape(self):
        k1, k2, k3, k4, k5 = jr.split(KEY, 5)
        fw_cells = [
            LongShortTermMemory(IDIM, HDIM, key=k1),
            LongShortTermMemory(2 * HDIM, HDIM, key=k2),
        ]
        bw_cells = [
            LongShortTermMemory(IDIM, HDIM, key=k3),
            LongShortTermMemory(2 * HDIM, HDIM, key=k4),
        ]
        model = DeepBidirectionalRNN(fw_cells, bw_cells, ODIM, key=k5)
        x = _make_input(KEY)
        y, all_hidden = model(x)
        assert y.shape == (ODIM,)
        assert len(all_hidden) == 2

    def test_jit(self):
        k1, k2, k3, k4, k5 = jr.split(KEY, 5)
        fw_cells = [
            LongShortTermMemory(IDIM, HDIM, key=k1),
            LongShortTermMemory(2 * HDIM, HDIM, key=k2),
        ]
        bw_cells = [
            LongShortTermMemory(IDIM, HDIM, key=k3),
            LongShortTermMemory(2 * HDIM, HDIM, key=k4),
        ]
        model = DeepBidirectionalRNN(fw_cells, bw_cells, ODIM, key=k5)
        x = _make_input(KEY)
        y, _ = eqx.filter_jit(model)(x)
        assert y.shape == (ODIM,)

    def test_mismatched_layers_raises(self):
        k1, k2, k3 = jr.split(KEY, 3)
        fw_cells = [LongShortTermMemory(IDIM, HDIM, key=k1)]
        bw_cells = [
            LongShortTermMemory(IDIM, HDIM, key=k2),
            LongShortTermMemory(2 * HDIM, HDIM, key=k3),
        ]
        with pytest.raises(AssertionError):
            DeepBidirectionalRNN(fw_cells, bw_cells, ODIM, key=KEY)

    def test_grad(self):
        k1, k2, k3, k4, k5 = jr.split(KEY, 5)
        fw_cells = [
            LongShortTermMemory(IDIM, HDIM, key=k1),
            LongShortTermMemory(2 * HDIM, HDIM, key=k2),
        ]
        bw_cells = [
            LongShortTermMemory(IDIM, HDIM, key=k3),
            LongShortTermMemory(2 * HDIM, HDIM, key=k4),
        ]
        model = DeepBidirectionalRNN(fw_cells, bw_cells, ODIM, key=k5)
        x = _make_input(KEY)

        @eqx.filter_value_and_grad
        def loss_fn(m):
            y, _ = m(x)
            return jnp.sum(y)

        val, _ = loss_fn(model)
        assert jnp.isfinite(val)


# ===================================================================
# RESERVOIR TESTS
# ===================================================================


class TestReservoirComputer:
    """Tests for the ReservoirComputer and ESN initialization."""

    def test_init_and_forward(self):
        k1, k2 = jr.split(KEY)
        cell = ElmanRNNCell(IDIM, HDIM, key=k1)
        rc = ReservoirComputer(cell, ODIM, key=k2)
        x = _make_input(KEY)
        y = rc(x)
        assert y.shape == (SEQ_LEN, ODIM)

    def test_compute_reservoir(self):
        k1, k2 = jr.split(KEY)
        cell = ElmanRNNCell(IDIM, HDIM, key=k1)
        rc = ReservoirComputer(cell, ODIM, key=k2)
        x = _make_input(KEY)
        h = rc.compute_reservoir(x)
        assert h.shape == (SEQ_LEN, HDIM)

    def test_fit_readout_with_bias(self):
        k1, k2 = jr.split(KEY)
        cell = ElmanRNNCell(IDIM, HDIM, key=k1)
        rc = ReservoirComputer(cell, ODIM, key=k2)
        x = _make_input(KEY)
        h = rc.compute_reservoir(x)
        y_target = jr.normal(KEY, (SEQ_LEN, ODIM))
        rc_fitted = rc.fit_readout(h, y_target, ridge=1e-6, train_bias=True)
        y_pred = rc_fitted(x)
        assert y_pred.shape == (SEQ_LEN, ODIM)

    def test_fit_readout_without_bias(self):
        k1, k2 = jr.split(KEY)
        cell = ElmanRNNCell(IDIM, HDIM, key=k1)
        rc = ReservoirComputer(cell, ODIM, key=k2)
        x = _make_input(KEY)
        h = rc.compute_reservoir(x)
        y_target = jr.normal(KEY, (SEQ_LEN, ODIM))
        rc_fitted = rc.fit_readout(h, y_target, ridge=1e-6, train_bias=False)
        y_pred = rc_fitted(x)
        assert y_pred.shape == (SEQ_LEN, ODIM)

    def test_esn_init(self):
        k1, k2, k3 = jr.split(KEY, 3)
        cell = ElmanRNNCell(IDIM, HDIM, key=k1)
        rc = ReservoirComputer(cell, ODIM, key=k2)
        rc_esn = init_reservoir_esn(
            k3,
            rc,
            spectral_radius=0.9,
            input_scaling=0.5,
            bias_scaling=0.1,
        )
        # the reservoir should still work
        x = _make_input(KEY)
        y = rc_esn(x)
        assert y.shape == (SEQ_LEN, ODIM)

    def test_esn_spectral_radius(self):
        """After init_reservoir_esn the spectral radius should match."""
        k1, k2, k3 = jr.split(KEY, 3)
        cell = ElmanRNNCell(IDIM, HDIM, key=k1)
        rc = ReservoirComputer(cell, ODIM, key=k2)
        target_rho = 0.9
        rc_esn = init_reservoir_esn(
            k3, rc, spectral_radius=target_rho, input_scaling=0.5, bias_scaling=0.1
        )
        w_hh = rc_esn.reservoir.cell.w_hh
        rho = jnp.max(jnp.abs(jnp.linalg.eigvals(w_hh)))
        assert jnp.allclose(rho, target_rho, atol=1e-5)

    def test_esn_wrong_cell_raises(self):
        """init_reservoir_esn should only work with ElmanRNNCell reservoirs."""
        k1, k2, k3 = jr.split(KEY, 3)
        cell = LongShortTermMemory(IDIM, HDIM, key=k1)
        rc = ReservoirComputer(cell, ODIM, key=k2)
        with pytest.raises(AssertionError):
            init_reservoir_esn(
                k3,
                rc,
                spectral_radius=0.9,
                input_scaling=0.5,
                bias_scaling=0.1,
            )


# ===================================================================
# CROSS-CUTTING / INTEGRATION TESTS
# ===================================================================


class TestCrossCutting:
    """Integration-style tests that exercise multiple components together."""

    def test_vmap_over_batch(self):
        """vmap an RNN over a batch of sequences."""
        k1, k2 = jr.split(KEY)
        cell = ElmanRNNCell(IDIM, HDIM, key=k1)
        model = RNN(cell, ODIM, key=k2)
        batch = jr.normal(KEY, (5, SEQ_LEN, IDIM))
        batched_fn = jax.vmap(model)
        ys, all_outs = batched_fn(batch)
        assert ys.shape == (5, ODIM)
        assert all_outs.shape == (5, SEQ_LEN, HDIM)

    def test_determinism(self):
        """Same key, same input → same output."""
        k1, k2 = jr.split(KEY)
        cell = LongShortTermMemory(IDIM, HDIM, key=k1)
        model = RNN(cell, ODIM, key=k2)
        x = _make_input(KEY)
        y1, _ = model(x)
        y2, _ = model(x)
        assert jnp.allclose(y1, y2)

    def test_different_keys_different_params(self):
        """Different keys should produce different model parameters."""
        k1, k2, k3 = jr.split(KEY, 3)
        model_a = RNN(ElmanRNNCell(IDIM, HDIM, key=k1), ODIM, key=k2)
        model_b = RNN(ElmanRNNCell(IDIM, HDIM, key=k2), ODIM, key=k3)
        x = _make_input(KEY)
        ya, _ = model_a(x)
        yb, _ = model_b(x)
        assert not jnp.allclose(ya, yb)

    def test_complex_cell_in_rnn_layer(self):
        """uRNN (complex state) should work through the full RNN layer."""
        k1, k2 = jr.split(KEY)
        cell = UnitaryEvolutionRNNCell(IDIM, HDIM, key=k1)
        model = RNN(cell, ODIM, key=k2)
        x = _make_input(KEY)
        y, all_outs = model(x)
        assert y.shape == (ODIM,)
        # all_outs should be real (encoder applies concat_real_imag)
        assert not jnp.iscomplexobj(all_outs)

