import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
import jax
import numpy as np
from jax import Array, core, lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax._src.interpreters.batching import BatchTracer
from jax.experimental.sparse import SparseTracer, sparsify
from pint import UnitRegistry

from pintax import dimensionless_zero, unitify, ureg, value_and_unit
from pintax._core import _global_ureg as gr
from pintax.unstable import Quantity, make_unit, make_unit_p, qreg

jax.config.update("jax_enable_x64", True)
q = unitify(lambda: ureg.meter, unwrap_outs=False)()
# zz = q + q


def print_fn(x):
    jax.debug.callback(lambda x: print(x * qreg.meter), x)
    # jax.debug.callback(lambda x: print(x), x)


def testfn():
    # TODO
    return unitify(print)(object())


@unitify
def main():

    return unitify(jnp.isinf)(qreg.meter)

    return unitify(lambda x: jnp.array(x))(qreg.meter)

    print("trace", ureg.meter * 2)

    ar = (jnp.arange(5) + 0.0) * ureg.meter

    def inner(x):
        # return x * ureg.inch

        v, u = value_and_unit(x)
        print(x)
        print(u)
        return jnp.sum(v / u) * ureg.inch + 5.0

    return jax.vmap(jax.vmap(inner))(jnp.ones((5, 5, 5)) * ureg.meter)
    return

    def inner(x: Array):
        print(x)
        ans = lax.concatenate([x, ar], 0)
        print(ans)
        return ans

    f = lambda g: jax.jvp(inner, (ar,), (g * ctx.unit("inch"),))[1]

    # return jax.make_jaxpr(f)(ar)
    return f(ar)


if __name__ == "__main__":
    main()
