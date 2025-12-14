"""Test stack / unstack functionality
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy
from rnn_jax.cells import LongShortTermMemoryCell
from rnn_jax.layers import RNNEncoder  
from rnn_jax.utils.utils import filter_stack_model, filter_unstack_model

def test_stack_unstack():
    key = jax.random.PRNGKey(0)
    idim = 4
    hdim = 8
    n_models = 5
    keys = jax.random.split(key, n_models)
    models = [
       RNNEncoder(LongShortTermMemoryCell(idim, hdim, key=k)) for k in keys
    ]

    stacked_model, template = filter_stack_model(models)
    unstacked_models = filter_unstack_model(stacked_model, template)
    for m1, m2 in zip(models, unstacked_models):
        jax.tree_util.tree_map(
            lambda x, y: numpy.testing.assert_array_equal(x, y), m1, m2
        )


def test_stacked_fw_pass():
    key = jax.random.PRNGKey(0)
    idim = 4
    hdim = 8
    n_models = 3
    batch_size = 2
    seq_len = 5

    keys = jax.random.split(key, n_models)
    models = [
        RNNEncoder(LongShortTermMemoryCell(idim, hdim, key=k)) for k in keys
    ]

    stacked_model, template = filter_stack_model(models)
    x = jax.random.normal(key, (n_models, batch_size, seq_len, idim))

    def model_fw(model, x):
        model = eqx.combine(model, template) # Combine with static structure
        return eqx.filter_vmap(model)(x)
    stacked_out = eqx.filter_vmap(model_fw)(stacked_model, x)

    unstacked_models = filter_unstack_model(stacked_model, template)
    unstacked_outs = []
    for i in range(n_models):
        out = model_fw(unstacked_models[i], x[i])
        unstacked_outs.append(out)
    unstacked_outs = jnp.stack(unstacked_outs, axis=0)

    numpy.testing.assert_array_equal(stacked_out, unstacked_outs)


if __name__ == "__main__":
    test_stack_unstack()
    print("test_stack_unstack passed")
    test_stacked_fw_pass()
    print("test_stacked_fw_pass passed")
