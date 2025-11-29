from typing import Callable, List, Tuple, Sequence, Union
import collections.abc as abc
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Inexact
from rnn_jax.cells.base import BaseCell
from jax.nn.initializers import Initializer
from typing import Tuple
from jaxtyping import Array

type Block = Union[Array, None]


def block_multiply(block, y):
    def _mult_block(b, y):
        if b is not None:
            return b @ y
        return jnp.array(0)

    multiplied = jax.tree.map(_mult_block, block, y, is_leaf=lambda b: b is None)
    return sum(multiplied)


def masked_block_triangular_matrix_mult(
    matrix: Sequence[Sequence[Block]], mask: Array, y: List[Array]
):
    y_new = y.copy()
    for i, block in enumerate(matrix):
        y_new[i] = jax.lax.cond(
            mask[i, 0], lambda _: block_multiply(block, y), lambda _: y[i], None
        )
    return y_new


def masked_multiply(blocks, mask, x):
    wx = []
    for i, block in enumerate(blocks):
        wx.append(mask[i] * block @ x)
    return wx


class ClockWorkRNNCell(BaseCell):
    W_h: Sequence[Sequence[Block]]
    W_i: Sequence[Array]
    b: Sequence[Array]
    nonlinearity: Callable
    periods: Array

    def __init__(
        self,
        idim: int,
        block_sizes: int | Sequence[int],
        periods: Sequence[int],
        nonlinearity: Callable,
        kernel_init: Initializer = jax.nn.initializers.glorot_normal(),
        recurrent_kernel_init: Initializer = jax.nn.initializers.orthogonal(),
        bias_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: Array,
    ):
        """Initialize the clockwork RNN

        Args:
            idim (int): Input dimension
            block_sizes (int | Sequence[int]): block sizes (if int, all blocks have the same size)
            periods (Sequence[int]): periods of the blocks
            nonlinearity (Callable): activation function
            kernel_init (Initializer, optional): input weights initializer. Defaults to jax.nn.initializers.glorot_normal().
            recurrent_kernel_init (Initializer, optional): recurrent weights initializer. Defaults to jax.nn.initializers.orthogonal().
            bias_init (Initializer, optional): bias initializer. Defaults to jax.nn.initializers.zeros.

        Kwargs:
            key (Array): JAX PRNG key for initialization
        """
        self.idim = idim
        self.complex_state = False
        assert isinstance(periods, abc.Sequence) and len(periods) > 0, (
            "periods must be a sequence"
        )
        if isinstance(block_sizes, int):
            block_sizes = [block_sizes] * len(periods)
        self.hdim = sum(block_sizes)
        self.periods = jnp.array(periods)
        self.states_shapes = (
            tuple((int(s),) for s in block_sizes) + ((1,),)
        )  # states (divided by period) + time ( array of shape () ) which is also part of the state
        w_h_key, key = jr.split(key)
        self.W_h = self._init_w_h(block_sizes, recurrent_kernel_init, w_h_key)
        w_i_key, key = jr.split(key)
        self.W_i = self._init_w_i(idim, block_sizes, kernel_init, w_i_key)
        b_key, key = jr.split(key)
        self.b = self._init_b(block_sizes, bias_init, b_key)
        self.nonlinearity = nonlinearity

    def _init_w_h(self, ns, initializer, key):
        """Initialize the block upper-triangular recurrent weights

        Args:
            ns: list of block sizes [n1, ..., nk]
            initializer: function (key, shape) -> array
            key: anything you want passed to the initializer (e.g., PRNGKey in JAX)

        Returns:
            List[List[Block]]: block upper-triangular matrix
        """
        k = len(ns)
        blocks = []
        for i in range(k):
            row_blocks = []
            for j in range(k):
                if j < i:
                    # zero block below diagonal
                    row_blocks.append(None)
                else:
                    # create block with the given initializer
                    block_key, key = jr.split(key, 2)  # or split key if using JAX
                    row_blocks.append(initializer(block_key, (ns[i], ns[j])))
            blocks.append(row_blocks)
        return blocks

    def _init_w_i(self, idim, ns, initializer, key):
        """initialize the input weights

        Args:
            idim (int): input dimension
            ns (Sequence[int]): number of units per block
            initializer (Initializer): initializer function for each block
            key (Array): JAX PRNG key

        Returns:
            List[Array]: list of input weight blocks
        """
        k = len(ns)
        blocks = []
        for i in range(k):
            block_key, key = jr.split(key)
            blocks.append(initializer(block_key, (ns[i], idim)))
        return blocks

    def _init_b(self, ns, initializer, key):
        k = len(ns)
        blocks = []
        for i in range(k):
            block_key, key = jr.split(key)
            blocks.append(initializer(block_key, (ns[i],)))
        return blocks

    def __call__(
        self, x: Array, state: Tuple[Array, ...]
    ) -> Tuple[Tuple[Array, ...], Array]:
        """Call the clockwork RNN

        Args:
            x (Array): Input array
            state (Tuple[Array, ...]): Hidden state, an array for each block

        Returns:
            new_state, h: (Tuple[Tuple[Array, ...], Array]): New state as a tuple and as concatenated array
        """
        *h, t = state
        mask = jax.vmap(lambda T, t: t % T == 0, in_axes=(0, None))(
            self.periods, t
        )  # mask out the blocks whose period is not a divisor of t
        wx = masked_multiply(self.W_i, mask, x)
        wh = masked_block_triangular_matrix_mult(self.W_h, mask, h)
        masked_bias = [b_i * m_i for b_i, m_i in zip(self.b, mask)]
        new_h = jax.tree.map(
            lambda wxi, whi, bi: self.nonlinearity(wxi + whi + bi), wx, wh, masked_bias
        )
        t = t + 1
        return tuple(new_h) + (t,), jnp.concat(new_h)
