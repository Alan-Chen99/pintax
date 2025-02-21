from jax import Array
from jax._src.typing import ArrayLike

from ._core import value_and_unit


def dimensionless_zero() -> Array:
    """
    return a zero that is dimensionless
    (rather than symbolic_zero)
    """
    val, _ = value_and_unit(0.0)
    return val


def get_unit(v: ArrayLike) -> Array:
    _, unit = value_and_unit(v)
    return unit


def get_value(v: ArrayLike) -> Array:
    value, _ = value_and_unit(v)
    return value
