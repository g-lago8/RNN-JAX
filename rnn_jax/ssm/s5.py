import jax
from jax import random as jr
from jax import nn
import numpy as np
from jax import numpy as jnp
import equinox as eqx
from jax.lax import associative_scan
from typing import TypeVar, Tuple
from jaxtyping import Inexact, Array, Complex, Real
from rnn_jax.ssm.base import BaseSSMLayer
import einops
from itertools import product


def leg_s_matrix(N):
    """Materialize a n x n LegS matrix

    Args:
        n (int): Dimension of the matrix
    """
    A = np.zeros((N, N))

    for n, k in product(range(N), range(N)):
        if  n > k:
            A[n, k] = (2.*float(n) + 1.) ** (1/2) * (2.*k + 1.) ** (1/2)
        elif n == k:
            A[n, k] = n + 1
        # if n < k keep it 0
    return - A

def leg_n_matrix(N):
    """Materialize an N x N LegN matrix 
    i.e. the normal matrix A' such that A = A' + P @ P.T, where P # P.T is the low rank
    matrix in the NPLR decomposition of A 

    Args:
        N (int): Dimension of the matrix
    """
    A = np.zeros((N, N))

    for n, k in product(range(N), range(N)):
        if  n > k:
            A[n, k] = (n + 1/2) ** (1/2) * (k + 1./2) ** (1/2)
        elif n == k:
            A[n, k] = 1 / 2
        else:
            A[n, k] = - (n + 1/2) ** (1/2) * (k + 1./2) ** (1/2)
    return -A


class SimplifiedStateSpaceLayer(BaseSSMLayer):
    """Implementation of the Simplified SSM layer (S5)
    """
    Delta: Real[Array, "state_dim"]
    W_in: Complex[Array, "state_dim in_dim"]
    W_out: Complex[Array, "model_dim state_dim"]
    W_skip: Real[Array, "model_dim in_dim"]
    #Lambda_complex: Complex[Array, "state_dim_c"]
    #Lambda_real: Complex[Array, "state_dim_r"]
    Lambda: Array
    
    def __init__(
        self,
        in_dim,
        state_dim: int,
        model_dim=None,
        init_w_h = 'leg-n',
        blocks_lambda = 1,
        delta_min=0.001,
        delta_max=0.1,
        *,
        key,
    ):
        super().__init__(in_dim, state_dim, model_dim)
        in_key, out_key, skip_key, delta_key = jr.split(key, 4)
        
        # init timescale
        self.Delta = jr.uniform(delta_key, minval=delta_min, maxval=delta_max)
        self.Lambda, V = self.init_lambda(blocks_lambda, init_w_h)
        # init input matrix
        B = jr.normal(in_key, (self.state_dim, self.in_dim))
        self.W_in = V.T @ B
        # init mixing/out matrix
        C = jr.normal(out_key, (self.model_dim, self.state_dim))
        self.W_out = C @ V
        # init skip matrix
        self.W_skip = jr.normal(skip_key, (self.model_dim, self.in_dim))

    def init_lambda(self, n_blocks, init_type):
        if self.state_dim % n_blocks != 0:
            raise ValueError("state_dim must be divisible by n_blocks")
        if init_type == 'leg-n':
            A = leg_n_matrix(self.state_dim // n_blocks)
            A = np.kron(np.eye(n_blocks), A)
            L, V = np.linalg.eig(A)
            return jnp.array(L), jnp.array(V)
        else:
            raise NotImplementedError("The only possible initialization is 'leg-n' for now.")

    def preprocess_inputs(self, xs: Array, B_bar: Array):
        return jax.vmap(lambda x: B_bar @ x)(xs)
        
    
    def postprocess_outputs(self, xs, hs):
        return (jax.vmap(lambda h: self.W_out @ h)(hs)).real + jax.vmap(lambda x: self.W_skip @ x)(xs)

    def ssm_cell(
        self, a: Tuple[Array, Array], b: Tuple[Array, Array]
    ) -> Tuple[Array, Array]:
        matrix_pow_i, bx_i = a
        matrix_pow_j, bx_j = b
        return matrix_pow_i * matrix_pow_i, matrix_pow_j * bx_i + bx_j
    
    def preprocess_matrix(self, seq_len):
        Lambda_bar = jnp.exp(self.Lambda * self.Delta)
        zoh_in_scaling = (1 / self.Lambda * (Lambda_bar - 1))
        W_in_bar = einops.einsum(self.W_in, zoh_in_scaling, "state_dim in_dim, state_dim -> state_dim in_dim")
        return einops.repeat(
            Lambda_bar, "state_dim -> seq_len state_dim", seq_len=seq_len
        ), W_in_bar
    
    def __call__(self, xs):
        seq_len = xs.shape[0]
        lambda_elements, B_bar = self.preprocess_matrix(seq_len)
        w_in_xs = self.preprocess_inputs(xs, B_bar)
        scan_elements = (lambda_elements, w_in_xs)
        lambda_powers, hs = associative_scan(self.ssm_cell, scan_elements)
        return self.postprocess_outputs(xs, hs)


def main():

    xs = jr.uniform(jr.key(0), (1000, 1))
    key = jr.key(1)
    model = SimplifiedStateSpaceLayer(1, 16, 32, key=key)
    jit_model = eqx.filter_jit(model)
    ys = model(xs)
    ys_prime = jit_model(xs)
    assert jnp.allclose(ys, ys_prime, atol=1e-4)
    print(ys[-1])
    print(ys_prime[-1])
    print("Check passed with rtol 1e-4")

    print(ys.shape)


if __name__ == "__main__":
    main()


