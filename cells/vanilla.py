from typing import Tuple
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
from typing import Callable
from jaxtyping import Inexact, Array
from cells.base import BaseCell


init_dict = {
    'glorot': jax.nn.initializers.glorot_normal(),
    'orthogonal': jax.nn.initializers.orthogonal(),
    'uniform': lambda key, shape: jr.uniform(key, shape, minval=-1, maxval=1),
    'zeros': jax.nn.initializers.zeros
}

nonlinearity_dict={
    'relu': jax.nn.relu,
    'tanh': jax.nn.tanh,
    'sigmoid': jax.nn.sigmoid
}

class ElmanRNNCell(BaseCell):
    w_hh:Inexact[Array, "hdim hdim"]
    w_ih:Inexact[Array, "hdim idim"]
    b: Inexact[Array, "hdim"]
    nonlinearity: Callable
    def __init__(self, idim, hdim, kernel_init="glorot", recurrent_kernel_init="orthogonal", bias_init="zeros", nonlinearity='relu', *, key):
        super().__init__(idim, hdim)
        self.states_shapes=((hdim,))
        self.complex_state=False
        ikey, hkey, bkey = jr.split(key, 3)
        self.w_ih = init_dict[kernel_init](ikey, (hdim, idim))
        self.w_hh = init_dict[recurrent_kernel_init](hkey, (hdim, hdim))
        self.b = init_dict[bias_init](bkey, hdim)
        self.nonlinearity = nonlinearity_dict[nonlinearity]

    def __call__(self, x: Array, state: Tuple[Array, ...]) -> Tuple[Tuple[Array, ...], Array]:
        h, = state
        new_h = self.nonlinearity(self.w_hh @ h + self.w_ih @ x + self.b)
        return (new_h,), new_h
        