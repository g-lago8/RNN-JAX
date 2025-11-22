import jax
import jax.numpy as jnp
from jaxtyping import Complex, Array


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

