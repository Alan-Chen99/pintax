from jax import Array
from jax._src import traceback_util
from jax._src.traceback_util import api_boundary
from pint import Unit

from ._core import Quantity, _global_ureg, make_unit

traceback_util.register_exclusion(__file__)


class _registry_wrapper:
    @api_boundary
    def __getattr__(self, name: str) -> Array:
        return self(name)

    @api_boundary
    def __call__(self, name: str) -> Array:
        ans = _global_ureg(name).units
        assert isinstance(ans, Unit)
        return make_unit(ans)


ureg = _registry_wrapper()


class _quantity_registry_wrapper:
    @api_boundary
    def __getattr__(self, name: str) -> Quantity:
        return self(name)

    @api_boundary
    def __call__(self, name: str) -> Quantity:
        ans = _global_ureg(name).units
        assert isinstance(ans, Unit)
        return Quantity._create(ans)


qreg = _quantity_registry_wrapper()
