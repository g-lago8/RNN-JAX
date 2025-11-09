"""Implementation of Unitary Evolution Recurrent Neural Networks (Arjovski et al. 2016), see https://arxiv.org/abs/1511.06464 for the paper 
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from cells.base import BaseCell
from typing import Sequence, Callable, Tuple
from jaxtyping import PyTree, Array, ArrayLike, Float, Inexact, Int, PRNGKeyArray, Complex


class ModReLU(eqx.Module):
    b: Float[Array, "hdim"]        
    def __call__(self, x):
        return (jnp.absolute(x) + self.b) * (x / jnp.absolute(x))


class UnitaryEvolutionRNNCell(BaseCell):
    v1:Inexact[Array, "hdim"]
    v2:Inexact[Array, "hdim"]
    diag1:Inexact[Array, "hdim"]
    diag2:Inexact[Array, "hdim"]
    diag3:Inexact[Array, "hdim"]
    perm:Int[Array, "hdim"]
    #h0: Float[Array, "hdim"]
    in_layer: eqx.nn.Linear
    modrelu: Callable
    states_shapes: Tuple
    def __init__(self, idim, hdim, permutation_type='identity', nonlinerity:Callable= jax.nn.relu, *, key, use_bias_in):
        super().__init__(idim, hdim)
        self.states_shapes = ((hdim,))
        self.complex_state = True
        key, *subkeys = jr.split(key, 9)
        v1_key1, v1_key2 = jr.split(subkeys[0])
        self.v1 = jr.uniform(v1_key1, (hdim,), minval=-1, maxval=1) + 1j * jr.uniform(v1_key2, (hdim,), minval=-1, maxval=1)
        v2_key1, v2_key2 = jr.split(subkeys[1])
        self.v2 = jr.uniform(v2_key1, (hdim,), minval=-1, maxval=1) + 1j * jr.uniform(v2_key2, (hdim,), minval=-1, maxval=1)
        self.diag1 = jr.uniform(subkeys[2], (hdim,), minval=-np.pi, maxval=np.pi)
        self.diag2 = jr.uniform(subkeys[3], (hdim,), minval=-np.pi, maxval=np.pi)
        self.diag3 = jr.uniform(subkeys[4], (hdim,), minval=-np.pi, maxval=np.pi)
     
        #self.h0 = jr.uniform(subkeys[5], (hdim,), minval= - np.sqrt(3 / (2 * hdim)), maxval = np.sqrt(3 / (2 * hdim)))
        self.modrelu = ModReLU(jnp.zeros(hdim)) # initialize biases in the ModReLU non-linearity
        allowed_permutations = ['random', 'identity']
        if permutation_type == 'random':
            perm = jr.permutation(subkeys[6], hdim)
        elif permutation_type == 'identity':
            perm = jnp.arange(hdim)
        else: 
            raise ValueError(f"allowed values for permutation_type are {allowed_permutations}, got {permutation_type}")
        self.perm = perm
        self.in_layer = eqx.nn.Linear(idim, hdim, use_bias=use_bias_in, key=subkeys[7])

    def _hh_layer (self, h:Inexact[Array, "hdim"]):
        h = jnp.exp(1j * self.diag1) * h
        h = jnp.fft.fft(h)
        h = 1 - 2 * self.v1 * (jnp.dot(self.v1, h)) / jnp.linalg.norm(self.v1, ord=2)
        h = h[self.perm]
        h = jnp.exp(1j * self.diag2) * h
        h = jnp.fft.ifft(h)
        h = 1 - 2 * self.v2 * (jnp.dot(self.v2, h)) / jnp.linalg.norm(self.v2, ord=2)
        h = jnp.exp(1j * self.diag3) * h
        return h
    
    def __call__(self, x:Inexact[Array, 'idim'], state)->Tuple[Tuple[Array], Array]:
        """Call the unitary evolution RNN. The update at time t is given by

            h_t = modReLU(W @ h_{t-1} + V @ x_t), 
            
            with W a unitary matrix parametrized as follows:

            W = D_3 * R_2 * F^{-1} * D_2 * P _ R_1 * F * D_1, 
            
            where
            - D matrices are diagonal complex matrices
            - R matrices are Householder reflectors
            - F is the DFT operation
            - P is a permuation matrix
         

        Args:
            x (Inexact[Array, &#39;idim&#39;]): input vector
            carry (_type_): state of the system (i.e. h in this case)
        """
        h, = state # carry for this cell is only the hidden state!
        x = self.in_layer(x)
        h = self._hh_layer(h)
        h = self.modrelu(h + x)
        return (h,), h
        


if __name__ == "__main__":
    key = jr.key(0)
    keys = jr.split(key, 3)
    rnn = UnitaryEvolutionRNNCell(10, 16, 'random', key=keys[0], use_bias_in=False)
    x = jr.uniform(keys[1], (10,))
    h = jr.uniform(keys[2], (16,))
    print(rnn(x, (h,)))