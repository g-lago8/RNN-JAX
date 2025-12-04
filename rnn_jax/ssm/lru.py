import jax
from jax import random as jr
from jax import nn
import numpy as np
from jax import numpy as jnp
import equinox as eqx
from jax.lax import associative_scan
from typing import TypeVar, Tuple, Callable
from jaxtyping import Inexact, Array
from rnn_jax.ssm.base import BaseSSMLayer
import einops


class LinearRecurrentUnit(BaseSSMLayer):
    nu_log: Inexact[Array, "state_dim"]
    theta_log: Inexact[Array, "state_dim"]
    W_in: Inexact[Array, "state_dim in_dim"]
    W_out: Inexact[Array, "model_dim state_dim"]
    W_skip: Inexact[Array, "model_dim in_dim"]
    gamma_log: Inexact[Array, "state_dim"]
    nonlinearity: Callable

    def __init__(
        self,
        in_dim,
        state_dim: int,
        model_dim=None,
        rho_min=0.0,
        rho_max=0.99,
        theta_min=0,
        theta_max=2 * np.pi,
        nonlinearity=jax.nn.gelu,
        *,
        key,
    ):
        super().__init__(in_dim, state_dim, model_dim)
        in_key, h_key, skip_key, out_key = jr.split(key, 4)
        # init transition matrix parameters
        self.nu_log, self.theta_log = self._init_W_hh(
            rho_min, rho_max, theta_min, theta_max, h_key
        )
        # init input matrix
        in_re_key, in_im_key = jr.split(in_key)
        w_in_real = nn.initializers.glorot_normal()(
            in_re_key, (self.state_dim, self.in_dim)
        )
        w_in_imag = nn.initializers.glorot_normal()(
            in_im_key, (self.state_dim, self.in_dim)
        )
        self.W_in = w_in_real + 1j * w_in_imag
        # init skip matrix
        self.W_skip = nn.initializers.glorot_normal()(
            skip_key, (self.model_dim, self.in_dim)
        )
        out_re_key, out_im_key = jr.split(out_key)
        # init out matrix
        w_out_real = nn.initializers.glorot_normal()(
            out_re_key, (self.model_dim, self.state_dim)
        )
        w_out_imag = nn.initializers.glorot_normal()(
            out_im_key, (self.model_dim, self.state_dim)
        )
        self.W_out = w_out_real + 1j * w_out_imag
        # init gamma_log to
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        self.gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))
        self.nonlinearity = nonlinearity

    def _init_W_hh(self, rho_min, rho_max, theta_min, theta_max, key):
        key1, key2 = jr.split(key)

        u1 = jr.uniform(key1, shape=(self.state_dim,))
        u2 = jr.uniform(key2, shape=(self.state_dim,))

        # sample the lenght of the eigenvalues
        if rho_max < rho_min:
            raise ValueError("rho_max must be larger than rho_min")
        if rho_min < 0 or rho_max > 1:
            raise ValueError("rho_min and rho_max must be in [0, 1]")
        nu_log = np.log(-0.5 * np.log(u1 * (rho_max**2 - rho_min**2) + rho_min**2))
        if theta_max < theta_min:
            raise ValueError("theta_max must be larger than theta_min")

        # sample the angles of the eigenvalues
        # Make sure thetas are positive. One could want to sample negative thetas too,
        # but the log breaks, so we just shift them by 2pi
        while theta_min < 0:
            theta_min += 2 * np.pi
            theta_max += 2 * np.pi

        theta_log = np.log(u2 * (theta_max - theta_min) + theta_min)

        return nu_log, theta_log

    def preprocess_inputs(self, xs):
        scaled_W_in = einops.einsum(
            self.W_in,
            jnp.exp(self.gamma_log),
            "state_dim in_dim, state_dim -> state_dim in_dim"
        )
        return jax.vmap(lambda x: scaled_W_in @ x)(xs)

    def discretize(self, seq_len):
        lambda_matrix = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        return einops.repeat(
            lambda_matrix, "state_dim -> seq_len state_dim", seq_len=seq_len
        )

    def ssm_cell(
        self, a: Tuple[Array, Array], b: Tuple[Array, Array]
    ) -> Tuple[Array, Array]:
        matrix_pow_i, bx_i = a
        matrix_pow_j, bx_j = b
        return matrix_pow_i * matrix_pow_i, matrix_pow_j * bx_i + bx_j

    def postprocess_outputs(self, xs, hs):
        zs = (jax.vmap(lambda h: self.W_out @ h)(hs)).real + jax.vmap(lambda x: self.W_skip @ x)(xs)
        return self.nonlinearity(zs)

    def __call__(self, xs, h0=None):
        seq_len = xs.shape[0]
        lambda_elements = self.discretize(seq_len)
        w_in_xs = self.preprocess_inputs(xs)
        scan_elements = (lambda_elements, w_in_xs)
        lambda_powers, hs = associative_scan(self.ssm_cell, scan_elements)
        return self.postprocess_outputs(xs, hs)
