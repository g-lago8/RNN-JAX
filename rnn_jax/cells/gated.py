from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Inexact
from rnn_jax.cells.base import BaseCell
from typing import Tuple
from jaxtyping import Array


class LongShortTermMemoryCell(BaseCell):
    w_ih: Array  # w input gate
    w_ii: Array  # w input gate
    b_i: Array  # b input gate
    w_fh: Array  # w forget gate
    w_fi: Array  # w forget gate
    b_f: Array  # b forget gate
    w_oh: Array  # w out gate
    w_oi: Array  # w out gate
    b_o: Array  # b forget gate
    w_ci: Array  # w for computing c
    w_ch: Array  # w for computing c
    b_c: Array  # b for computing c
    nonlinearity: Callable

    def __init__(
        self,
        idim,
        hdim,
        nonlinearity=jax.nn.tanh,
        kernel_init:Callable=jax.nn.initializers.glorot_normal(),
        recurrent_kernel_init:Callable=jax.nn.initializers.uniform(),
        bias_init:Callable=jax.nn.initializers.zeros,
        *,
        key,
        use_bias=None,
    ):
        """LSTM cell

        Args:
            idim (int): input dimension
            hdim (int): hidden dimension
            key (PRNGKey): pseudo-RNG key
            nonlinearity (Callable, optional): activation function (for gates, sigmoid is used). Defaults to jax.nn.relu.
            kernel_init (jax.nn.Initializer, optional): weights initializer. Defaults to jax.nn.initializers.glorot_normal().
            bias_init (jax.nn.Initializer, optional): bias initializer. Defaults to jax.nn.initializers.zeros, except for the *forget gate*, where the bias is always initialized to 1.
            To change the forget bias initialization (or to have finer-grained control on the initialized weights), one could use `equinox.tree_at` to perform model surgery as in the example below
            ```
            key = jax.random.key(42)
            model_key, bias_key = jax.random.split(key, 2)
            my_custom_bias = jax.nn.initializers.normal()(bias_key, hdim)
            my_model = equinox.tree_at(
                lambda model: model.b_f, # function that takes a pytree and return the leaf to modify
                LongShortTermMemoryCell(idim, hdim, key=model_key), # the model to modify
                my_custom_bias # the value to put in place of the chose leaf
            )
            ```
            use_bias (bool, optional): not used for now (bias always on). Defaults to None.
        """
        self.idim = idim
        self.hdim = hdim
        self.complex_state = False
        self.states_shapes = (hdim, hdim)
        self.nonlinearity = nonlinearity
        *subkeys, key = jr.split(key, 12)
        self.w_ih = recurrent_kernel_init(subkeys[0], (hdim, hdim))
        self.w_fh = recurrent_kernel_init(subkeys[1], (hdim, hdim))
        self.w_oh = recurrent_kernel_init(subkeys[2], (hdim, hdim))
        self.w_ch = recurrent_kernel_init(subkeys[3], (hdim, hdim))
        self.w_ii = kernel_init(subkeys[4], (hdim, idim))
        self.w_fi = kernel_init(subkeys[5], (hdim, idim))
        self.w_oi = kernel_init(subkeys[6], (hdim, idim))
        self.w_ci = kernel_init(subkeys[7], (hdim, idim))
        self.b_i = bias_init(subkeys[8], (hdim,))
        self.b_c = bias_init(subkeys[9], (hdim,))
        self.b_o = bias_init(subkeys[10], (hdim,))
        self.b_f = jnp.ones((hdim,))

    def __call__(
        self, x: Array, state: Tuple[Array, Array]
    ) -> Tuple[Tuple[Array, Array], Array]:
        """Call the LSTM cell

        Args:
            x (Array): Input array
            state (Tuple[Array, Array]): Cell state and hidden state of the LSTM

        Returns:
            (h, c), h (Tuple[Array, Array], Array): Tuple of new cell state and hidden state, and the new hidden state
        """
        h, c = state
        input_gate = jax.nn.sigmoid(self.w_ii @ x + self.w_ih @ h + self.b_i)
        forget_gate = jax.nn.sigmoid(self.w_fi @ x + self.w_fh @ h + self.b_f)
        output_gate = jax.nn.sigmoid(self.w_oi @ x + self.w_oh @ h + self.b_o)
        c_new = self.nonlinearity(self.w_ci @ x + self.w_ch @ h + self.b_c)
        c_new = input_gate * c_new + forget_gate * c
        h_new = self.nonlinearity(c_new) * output_gate
        return (h_new, c_new), h_new



class GatedRecurrentUnitCell(BaseCell):
    w_ih: Array  # w input gate
    w_fh: Array  # w forget gate
    b_i: Array  # b input gate
    b_f: Array  # b forget gate
    w_ii: Array  # w input gate
    w_fi: Array  # w forget gate
    nonlinearity: Callable

    def __init__(
        self,
        idim,
        hdim,
        nonlinearity=jax.nn.relu,
        kernel_init=jax.nn.initializers.glorot_normal(),
        bias_init=jax.nn.initializers.zeros,
        *,
        key,
        use_bias=None,
    ):
        """GRU cell

        Args:
            idim (int): input dimension
            hdim (int): hidden dimension
            key (PRNGKey): pseudo-RNG key
            nonlinearity (Callable, optional): activation function (for gates, sigmoid is used). Defaults to jax.nn.relu.
            kernel_init (jax.nn.Initializer, optional): weights initializer. Defaults to jax.nn.initializers.glorot_normal().
            bias_init (jax.nn.Initializer, optional): bias initializer. Defaults to jax.nn.initializers.zeros.
            use_bias (bool, optional): not used for now (bias always on). Defaults to None.
        """
        self.idim = idim
        self.hdim = hdim
        self.complex_state = False
        self.states_shapes = (hdim,)
        self.nonlinearity = nonlinearity
        *subkeys, key = jr.split(key, 7)
        self.w_ih = kernel_init(subkeys[0], (hdim, hdim))
        self.w_fh = kernel_init(subkeys[1], (hdim, hdim))
        self.w_ii = kernel_init(subkeys[2], (hdim, idim))
        self.w_fi = kernel_init(subkeys[3], (hdim, idim))
        self.b_i = bias_init(subkeys[4], (hdim,))
        self.b_f = bias_init(subkeys[5], (hdim,))

    def __call__(self, x: Array, state: Tuple[Array]) -> Tuple[Tuple[Array], Array]:
        """Call the GRU cell

        Args:
            x (Array): Input array
            state (Tuple[Array]): Hidden state of the GRU
        Returns:
            (h,), h (Tuple[Array], Array): Tuple of new hidden state, and the new hidden state
        """
        h = state[0]
        reset_gate = jax.nn.sigmoid(self.w_fi @ x + self.w_fh @ h + self.b_f)
        input_gate = jax.nn.sigmoid(self.w_ii @ x + self.w_ih @ (reset_gate * h) + self.b_i)
        h_new = self.nonlinearity(self.w_ii @ x + self.w_ih @ (input_gate * h))
        h_new = (1 - input_gate) * h + input_gate * h_new
        return (h_new,), h_new


# =======================  wrappers from equinox.nn =======================


class LongShortTermMemory(BaseCell):
    lstm: eqx.nn.LSTMCell

    def __init__(self, idim: int, hdim: int, *, key, **lstm_kwargs):
        """Wrapper for `equinox.nn.LSTMCell`

        Args:
            idim (int): input dimension
            hdim (int): hidden dimension
            key (PRNGKey): pseudo-RNG key
        """
        super().__init__(idim, hdim)
        self.complex_state = False
        self.states_shapes = ((hdim,), (hdim,))
        self.lstm = eqx.nn.LSTMCell(idim, hdim, key=key, **lstm_kwargs)
        bias_modified = self.lstm.bias.at[2 * hdim : 3 * hdim].set(1.0)  # type: ignore
        self.lstm = eqx.tree_at(
            lambda tree: tree.bias, self.lstm, bias_modified
        )  # Initialize the forget gate bias to ones (biases are incorporated in one single vector)

    def __call__(
        self, x: jax.Array, state: Tuple[Array, ...]
    ) -> Tuple[Tuple[Array, Array], Array]:
        h, c = self.lstm(x, state)
        return (h, c), h


class GatedRecurrentUnit(BaseCell):
    gru: eqx.nn.GRUCell

    def __init__(self, idim, hdim, *, key, **gru_kwargs):
        """Wrapper for `equinox.nn.GRUCell`

        Args:
            idim (int): input dimension
            hdim (int): hidden dimension
            key (PRNGKey): pseudo-RNG key
        """
        super().__init__(idim, hdim)
        self.complex_state = False
        self.states_shapes = (hdim,)
        self.gru = eqx.nn.GRUCell(idim, hdim, key=key, **gru_kwargs)

    def __call__(self, x: Array, state: Tuple[Array]) -> Tuple[Tuple[Array], Array]:
        h = self.gru(x, state[0])
        return (h,), h
