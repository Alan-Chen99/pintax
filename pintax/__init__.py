from . import _rules, unstable
from ._api import Quantity, areg, unitify, ureg
from ._helpers import dimensionless_zero, get_unit, get_value, quantity
from ._primitives import value_and_unit

__all__ = [
    "Quantity",
    "areg",
    "dimensionless_zero",
    "quantity",
    "unitify",
    "ureg",
    "value_and_unit",
]


class _dummy:
    pass


for _x in [
    Quantity,
    dimensionless_zero,
    get_unit,
    get_value,
    quantity,
    type(areg),
    type(ureg),
    unitify,
    value_and_unit,
]:
    _x.__module__ = _dummy.__module__

_ = [
    _rules,
    unstable,
]
