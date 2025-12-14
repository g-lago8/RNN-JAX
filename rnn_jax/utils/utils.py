import jax
import jax.numpy as jnp
from jaxtyping import Complex, Array
import equinox as eqx

def concat_real_imag(x: Complex[Array, "..."], axis=-1):
    """Concatenate real and image parts of an array

    Args:
        x (Complex[Array]): complex array
        axis (int, optional): axis of the concatenation. Defaults to -1.

    Returns:
        Array: a real concatenated array
    """
    x_real = jnp.real(x)
    x_imag = jnp.imag(x)
    return jnp.concatenate([x_real, x_imag], axis=axis)


def filter_stack_model(models):
    """
    Given a list of models, returns a model where each parameter is a stack of the corresponding parameters in the input models.
    Requires that all models have the same structure and parameter shapes.
    Args:
        models (list[eqx.Module]): list of equinox models

    Returns:
        (eqx.Module, eqx.Module: filtered stacked model (with parameters stacked along a new first axis), and a template with the static structure
    """

    models_filtered, static = eqx.partition(models, eqx.is_array)
    # Assert static structure is the same
    for s in static[1:]:
        assert static[0] == s, "All models must have the same static structure"

    stacked_model = jax.tree.map(
        lambda *args: jnp.stack(args, axis=0), *models_filtered
    )
    return stacked_model, static[0]


def filter_unstack_model(stacked_model, template):
    """
    Given a stacked model (as returned by `filter_stack_model`), and a template model, returns a list of models where each parameter is taken from the corresponding slice of the stacked model.
    Args:
        stacked_model (eqx.Module): stacked model
        template (eqx.Module): template model
    Returns:
        list[eqx.Module]: list of unstacked models
    """
    n_models = jax.tree_util.tree_leaves(stacked_model)[0].shape[0]

    def get_model(i):
        model = jax.tree.map(
            lambda x: x[i], stacked_model
        )
        # re-insert static structure
        model = eqx.combine(model, template)
        return model
    return [get_model(i) for i in range(n_models)]

