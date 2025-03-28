import os as _os
import sys as _sys

from ._core import (
    AnyUnit,
    PintaxDimensionalityError,
    PintaxError,
    PintaxError_forward,
    PintaxNotImplementedError,
    PintaxRuntimeError,
    PintaxTypeError,
    PintaxZeroDivisionError,
    Qt,
    anyunit,
    pint_registry,
    quantity,
)
from ._core import rules_complex as unitify_rules
from ._primitives import (
    convert_unit_p,
    make_unit_p,
    prim_convert_unit,
    prim_make_unit,
    value_and_unit_p,
)
from ._rules import unitify_jaxpr


class _dummy:
    pass


if not _os.path.basename(_sys.argv[0]) == "sphinx-build":
    for _x in [
        AnyUnit,
        PintaxDimensionalityError,
        PintaxError,
        PintaxError_forward,
        PintaxNotImplementedError,
        PintaxRuntimeError,
        PintaxTypeError,
        PintaxZeroDivisionError,
        Qt,
        pint_registry,
        prim_convert_unit,
        prim_make_unit,
        quantity,
    ]:
        _x.__module__ = _dummy.__module__

_ = [
    anyunit,
    convert_unit_p,
    make_unit_p,
    unitify_jaxpr,
    unitify_rules,
    value_and_unit_p,
]
