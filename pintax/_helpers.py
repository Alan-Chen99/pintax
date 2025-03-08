from jax import Array
from jax._src.typing import ArrayLike

from ._api import Quantity, QuantityLike, Unit
from ._core import dimensionless
from ._primitives import value_and_unit
from ._utils import check_arraylike


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


def quantity(x: QuantityLike) -> Quantity:
    if isinstance(x, Quantity):
        return x
    elif isinstance(x, Unit):
        return Quantity._create(1, x)
    elif isinstance(x, Array):
        m, u = value_and_unit(x)
        return Quantity._create(m, Unit._traced(u))
    else:
        check_arraylike(x)
        return Quantity._create(x, Unit._concrete(dimensionless))
