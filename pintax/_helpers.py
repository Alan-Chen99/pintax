from __future__ import annotations

from typing import Any, Literal, overload

from jax import Array
from jax import tree_util as jtu
from jax.typing import ArrayLike

from ._api import Quantity, QuantityLike, Unit, ensure_arraylike
from ._core import dimensionless
from ._primitives import prim_convert_unit, prim_value_and_unit
from ._utils import check_arraylike, ensure_jax


def quantity(x: QuantityLike) -> Quantity:
    r"""
    convert an Array under unitify in to a Quantity, which allow access to its magnitude and unit.

    >>> from pintax import *
    >>> @unitify
    >>> def main():
    ...     x = 10.0 * areg.m
    ...     q = quantity(x)
    ...     print(x, q, q.m, q.u, q.a, sep="\n")
    >>> main()
    UnitTracer(Array(10., dtype=float32, weak_type=True), 'meter')
    Quantity(Array(10., dtype=float32, weak_type=True), 'meter')
    UnitTracer(Array(10., dtype=float32, weak_type=True), 'dimensionless')
    Unit('meter')
    UnitTracer(Array(10., dtype=float32), 'meter')
    """
    if isinstance(x, Quantity):
        return x
    elif isinstance(x, Unit):
        return Quantity._create(1, x)
    elif isinstance(x, Array):
        m, u = prim_value_and_unit(x)
        return Quantity._create(m, Unit._traced(u))
    else:
        x = check_arraylike(x)
        return Quantity._create(x, Unit._concrete(dimensionless))


def value_and_unit(x: QuantityLike) -> tuple[ArrayLike, Unit]:
    ans = quantity(x)
    return ans.m, ans.u


def get_unit(x: QuantityLike) -> Unit:
    return quantity(x).u


@overload
def convert_unit(
    x: QuantityLike, u: QuantityLike, return_array: Literal[True] = True
) -> Array: ...
@overload
def convert_unit(
    x: QuantityLike, u: QuantityLike, return_array: Literal[False]
) -> Quantity: ...


def convert_unit(
    x: QuantityLike, u: QuantityLike, return_array=True
) -> Array | Quantity:
    """
    convert x to the units of u, and return the result. only the unit of u is used; value of u is ignored.

    if return_array==True (default), return an Array. this only works under unitify.

    if return_array==False, return a Quantity.
    """
    if return_array:
        return prim_convert_unit(ensure_arraylike(x), ensure_arraylike(u))
    else:
        return quantity(x).to(quantity(u).u)


@overload
def magnitude(x: Array, u: QuantityLike | None = None) -> Array: ...
@overload
def magnitude(x: QuantityLike, u: QuantityLike | None = None) -> ArrayLike: ...


def magnitude(x: QuantityLike, u: QuantityLike | None = None) -> ArrayLike:
    """
    get the magnitude of x, in the units of u. if u is None, just get the magnitude of x.
    """
    if u is None:
        ans = quantity(x).m
    else:
        ans = quantity(x).to(quantity(u).u).m
    if isinstance(x, Array):
        return ensure_jax(ans)
    else:
        return ans


@overload
def sync_units(x: QuantityLike) -> tuple[ArrayLike, Unit]: ...
@overload
def sync_units(x: tuple[QuantityLike, ...]) -> tuple[tuple[ArrayLike, ...], Unit]: ...
@overload
def sync_units(x: list[QuantityLike]) -> tuple[list[ArrayLike], Unit]: ...
@overload
def sync_units[T](x: T) -> tuple[T, Unit]: ...


def sync_units(x) -> tuple[Any, Unit]:
    """
    convert arrays of the same dimensionality into the same unit.

    Args:
      x: pytree of Quantity, Unit or ArrayLike, containing the arrays
    Returns:
      A tuple (magnitudes, unit)

      the first is a pytree with the same structure as x, with the resulting magnitudes.
      the second is the shared unit.
    """
    args, pytree = jtu.tree_flatten(x)
    assert len(args) > 0
    args_q = [quantity(x) for x in args]
    u = args_q[0].u
    out_bufs = [x.to(u).m for x in args_q]
    return jtu.tree_unflatten(pytree, out_bufs), u
