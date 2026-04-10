from typing import Any, Optional, Sequence, Union, Callable
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Inexact, Bool, Array
from rnn_jax.cells._base import BaseCell
from rnn_jax.utils.utils import (
    concat_real_imag,
    filter_stack_model,
    filter_unstack_model,
)
from rnn_jax.layers._encoder import RNNEncoder, BidirectionalRNNEncoder


class ModularRNN(eqx.Module):
    model_list: Sequence[BaseCell]  # sequence of BaseCell
    connection_matrix: jnp.ndarray  # (n_modules, n_modules) bool
    in_connections: jnp.ndarray  # (n_modules,) bool
    out_connections: jnp.ndarray  # (n_modules,) bool

    # optional linear projections
    in_projection: Optional[jnp.ndarray]  # (in_dim, per-module-input_dim)
    out_projection: Optional[jnp.ndarray]  # (per-module-state_dim, out_dim)

    aggregate_fn: Callable = eqx.static_field()  # e.g. jnp.sum

    def __call__(self, x_seq):
        """
        x_seq: (T, in_dim)
        Returns:
            outputs: (T, out_dim)
            states: final states of all modules
        """

        T, _ = x_seq.shape
        n_modules = len(self.model_list)

        # initialize hidden states of each module
        states = [[jnp.zeros(s) for s in m.states_shapes] for m in self.model_list]

        def step(carry, x_t):
            states = carry

            # ==== 1. Build input for each module ====
            module_inputs = []

            for j in range(n_modules):
                # (1) recurrent inputs: sum of states from all i → j
                incoming = [
                    states[i][1]  # h_i^t
                    for i in range(n_modules)
                    if self.connection_matrix[i, j]
                ]
                if incoming:
                    rec_sum = self.aggregate_fn(jnp.stack(incoming, axis=0), axis=0)
                else:
                    rec_sum = 0.0

                # (2) external input: optional projection
                if self.in_connections[j]:
                    if self.in_projection is None:
                        ext = x_t
                    else:
                        ext = x_t @ self.in_projection  # (in_dim → module_dim)
                else:
                    ext = 0.0

                module_inputs.append(rec_sum + ext)

            # ==== 2. Apply each module ====
            new_states = []
            for j in range(n_modules):
                cell = self.model_list[j]
                s_prev, _ = states[j]
                s_new, h_new = cell(module_inputs[j], s_prev)  # BaseCell API
                new_states.append((s_new, h_new))

            # ==== 3. Build output ====
            outgoing = [
                new_states[j][1]  # h_j
                for j in range(n_modules)
                if self.out_connections[j]
            ]
            if outgoing:
                out_sum = self.aggregate_fn(jnp.stack(outgoing, axis=0), axis=0)
            else:
                out_sum = jnp.zeros(())

            if self.out_projection is not None:
                out = out_sum @ self.out_projection
            else:
                out = out_sum

            return new_states, out

        # scan over sequence
        states, outputs = jax.lax.scan(step, states, x_seq)
        return outputs, states


class StackedModularRNN(eqx.Module):
    stacked_models: BaseCell  # sequence of BaseCell
    n_modules: int
    state_shapes: tuple
    complex_state: bool
    template: BaseCell
    connection_matrix: jnp.ndarray  # (n_modules, n_modules) bool
    in_connections: jnp.ndarray  # (n_modules,) bool
    out_connections: jnp.ndarray  # (n_modules,) bool

    # optional linear projections
    in_projection: jnp.ndarray  # (in_dim, per-module-input_dim)
    out_projection: jnp.ndarray  # (per-module-state_dim, out_dim)

    aggregate_fn: Callable = eqx.static_field()  # e.g. jnp.sum

    def __init__(
        self,
        model_list: Sequence[BaseCell],
        connection_matrix,
        in_connections,
        out_connections,
    ):
        self.state_shapes = model_list[0].states_shapes
        self.complex_state = model_list[0].complex_state
        self.n_modules = len(model_list)
        self.model_list, self.template = filter_stack_model(model_list)
        self.connection_matrix = connection_matrix
        self.in_connections = in_connections
        self.out_connections = out_connections

    def __call__(self, x_seq, initial_state=None):
        if initial_state is None:
            initial_state = jnp.zeros(self.n_modules, *self.state_shapes)

        for x in x_seq:
            x = self.in_connections * self.in_projection @ x
