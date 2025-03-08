from ._core import (
    PintaxError,
    PintaxError_forward,
    PintaxNotImplementedError,
    PintaxRuntimeError,
    PintaxTypeError,
    PintaxZeroDivisionError,
    Qt,
    anyunit,
    dimensionless,
    pint_registry,
    quantity,
)
from ._core import rules_complex as unitify_rules
from ._primitives import (
    convert_unit,
    convert_unit_p,
    make_unit,
    make_unit_p,
    value_and_unit_p,
)
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
    Qt,
    convert_unit,
    make_unit,
    pint_registry,
    quantity,
]:
    _x.__module__ = _dummy.__module__

_ = [
    anyunit,
    convert_unit_p,
    dimensionless,
    make_unit_p,
    unitify_jaxpr,
    unitify_rules,
    value_and_unit_p,
]
