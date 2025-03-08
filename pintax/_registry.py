from jax._src import traceback_util

traceback_util.register_exclusion(__file__)


# class _quantity_registry_wrapper:
#     @api_boundary
#     def __getattr__(self, name: str) -> Qt:
#         return self(name)

#     @api_boundary
#     def __call__(self, name: str) -> Qt:
#         ans = _global_ureg(name).units
#         assert isinstance(ans, Unit)
#         return Qt._create(ans)


# qreg = _quantity_registry_wrapper()
