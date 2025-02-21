from ._core import (
    PintaxError,
    PintaxError_forward,
    PintaxNotImplementedError,
    PintaxRuntimeError,
    PintaxTypeError,
    PintaxZeroDivisionError,
    Quantity,
)
from ._core import _rules_complex as unitify_rules
from ._core import (
    dimensionless,
    make_unit,
    make_unit_p,
    pint_registry,
    quantity,
    symbolic_zero,
    value_and_unit_p,
)
from ._registry import qreg
from ._rules import unitify_jaxpr


class _dummy:
    pass


for _x in [
    PintaxError,
    PintaxError_forward,
    PintaxNotImplementedError,
    PintaxRuntimeError,
    PintaxTypeError,
    PintaxZeroDivisionError,
    Quantity,
    make_unit,
    pint_registry,
    quantity,
    type(qreg),
]:
    _x.__module__ = _dummy.__module__

_ = [
    dimensionless,
    symbolic_zero,
    unitify_jaxpr,
    unitify_rules,
    make_unit_p,
    value_and_unit_p,
]
