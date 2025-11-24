import jax
import jax.random as jr
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from rnn_jax.layers import RNNEncoder
from rnn_jax.cells import BaseCell, ElmanRNNCell
from jaxtyping import Float, Array


class ReservoirComputer(eqx.Module):
    reservoir: RNNEncoder
    hdim: int
    odim: int
    readout:  eqx.nn.Linear
    def __init__(self, cell, odim, *, key):
        """A general reservoir computer. 
        When instanced with Cell = ElmanRNN(), it returns an Echo State Network 

        Args:
            cell (BaseCell): Cell representing the readout
            odim (int): output dimension
            key (PRNGKeyArray): rng key

        Returns:
            ReservoirComputer: a reservoir computer with a `cell` reservoir and a linear readout
        """
        res_key, out_key = jr.split(key)
        self.reservoir = RNNEncoder(cell, key=res_key)
        self.hdim = self.reservoir.hdim
        self.odim = odim
        self.readout = eqx.nn.Linear(self.hdim, self.odim, key=out_key)

    def compute_reservoir(self, x):
        """Call the encoder to compute the reservoir states

        Args:
            x (Array["seq_len in_dim"]): input sequence

        Returns:
            h (Array["seq_len hdim]): reservoir states
        """
        return self.reservoir(x)
    
    def fit_readout(self, reservoir:Float[Array, "Nseq Nres"], y:Float[Array, "Nseq"], ridge = 1e-8, train_bias: bool = True) -> "ReservoirComputer":
        """Fit the readout using (optional ridge) least-squares and attach it to self.out_layer.weight and optionally bias.
        
        Args:
            reservoir: Array[N, Nres] reservoir states, computed through self.compute_reservoir
            y: Array[N,] (or [N, odim]) output sequence
            ridge: regularization scalar (added to normal equations)
            train_bias: whether to fit an additive bias term (default True)
        Returns:
            new ReservoirComputer with updated readout.weight (and readout.bias if train_bias)
        """
        R = jnp.asarray(reservoir)
        Y = jnp.asarray(y)
        if Y.ndim == 1:
            Y = Y[:, None]  # (N, odim)

        if train_bias:
            # augment reservoir with constant 1s column to solve jointly for weights and bias
            ones = jnp.ones((R.shape[0], 1), dtype=R.dtype)
            R_aug = jnp.concatenate([R, ones], axis=1)  # (N, Nres+1)
            RtR = R_aug.T @ R_aug
            D = RtR.shape[0]
            dtype = RtR.dtype
            reg = ridge * jnp.eye(D, dtype=dtype)
            Wcorr = jnp.linalg.solve(RtR + reg, R_aug.T @ Y)  # (Nres+1, odim)
            # split into weights and bias
            W_readout = Wcorr[:-1, :].T  # (odim, Nres)
            b_readout = Wcorr[-1, :].reshape(-1)  # (odim,)
            # update both weight and bias
            def get_readout_params(rc: ReservoirComputer):
                return (rc.readout.weight, rc.readout.bias)
            return eqx.tree_at(get_readout_params, self, (W_readout, b_readout))
        else:
            # only fit weight, keep existing bias unchanged
            RtR = R.T @ R
            D = RtR.shape[0]
            dtype = RtR.dtype
            reg = ridge * jnp.eye(D, dtype=dtype)
            Wcorr = jnp.linalg.solve(RtR + reg, R.T @ Y)  # (Nres, odim)
            W_readout = Wcorr.T  # (odim, Nres) matches out_layer.weight

            def get_readout_kernel(rc: ReservoirComputer):
                return rc.readout.weight
            return eqx.tree_at(get_readout_kernel, self, W_readout)


    def __call__(self, x, last_state=None, batch_size=-1):
        """Call the reservoir computer (Reservoir + Readout)

        Args:
            x (Array[N, idim]): input array
            batch_size (int): batch size of the readout layer. If -1, `jax.vmap`is used, processing all elements of the sequence at once. Defaults to -1
        """
        reservoir_states = self.reservoir(x, initial_state=last_state)
        if batch_size <= 0:
            return jax.vmap(self.readout)(reservoir_states)
        else:
            return jax.lax.map(self.reservoir, reservoir_states, batch_size=batch_size)
        

def init_reservoir_esn(
    key,
    esn:ReservoirComputer, 
    spectral_radius: float, 
    input_scaling: float,
    bias_scaling: float,
    kernel_init=jr.uniform,
    rec_kernel_init=jr.uniform,
    bias_init=jr.uniform,
    ):
    assert isinstance(esn.reservoir.cell, ElmanRNNCell), """This function is meant to be used on Echo State Networks, "
        i.e. reservoir computers with cell of type ElmanRNNCell."""
    
    # get the shapes

    w_ih_shape = esn.reservoir.cell.w_ih.shape
    w_hh_shape = esn.reservoir.cell.w_hh.shape
    b_shape = esn.reservoir.cell.b.shape
    
    ikey, hkey, bkey = jr.split(key, 3)
    
    w_ih = kernel_init(ikey, w_ih_shape) * input_scaling
    b = bias_init(bkey, b_shape) * bias_scaling
    w_hh = rec_kernel_init(hkey, w_hh_shape)
    rho_h = jnp.max(jnp.absolute(jnp.linalg.eigvals(w_hh)))
    w_hh =  w_hh * spectral_radius / rho_h

    esn = eqx.tree_at(
        lambda m: (
            m.reservoir.cell.w_ih,
            m.reservoir.cell.b,
            m.reservoir.cell.w_hh),
        esn,
        (w_ih, b, w_hh)
    )
    return esn