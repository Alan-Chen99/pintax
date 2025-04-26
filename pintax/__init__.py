import os as _os
import sys as _sys

from . import _primitives as _primitives
from . import _rules as _rules
from . import _rules_basic as _rules_basic
from . import _rules_controlflow as _rules_controlflow
from . import functions, unstable
from ._api import Quantity, QuantityLike, Unit, areg, quantity_to_arr, unitify, ureg
from ._helpers import convert_unit, magnitude, quantity, sync_units

dimensionless: Unit = ureg.dimensionless

__all__ = [
    "Quantity",
    "QuantityLike",
    "Unit",
    "areg",
    "convert_unit",
    "dimensionless",
    "magnitude",
    "quantity",
    "sync_units",
    "unitify",
    "ureg",
]


class _dummy:
    pass


if not _os.path.basename(_sys.argv[0]) == "sphinx-build":
    for _x in [
        Quantity,
        Unit,
        convert_unit,
        magnitude,
        quantity,
        quantity_to_arr,
        sync_units,
        type(areg),
        type(ureg),
        unitify,
    ]:
        _x.__module__ = _dummy.__module__

_ = [
    functions,
    unstable,
]
