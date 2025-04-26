import jax
from jax import numpy as jnp

from pintax import areg, quantity, unitify, ureg


def f(x, y, z):
    x = jnp.minimum(x, y) + x
    x = jnp.abs(x)
    x += y**2 / x
    abc = x * z * areg.kg * quantity(x).m / quantity(x).u
    return abc


@unitify
def test_grad():
    _ = jax.grad(f)(1.0 * areg.m, 2.0 * areg.inch, 3.0 * areg.s)


@unitify
def test_vmap():
    _ = jax.vmap(f)(
        (jnp.arange(5) + 1.0) * areg.m,
        (jnp.arange(5) + 2.0) * areg.inch,
        (jnp.arange(5) + 3.0) * areg.s,
    )


def test_jit():
    a, b, c = 1.0 * ureg.m, 2.0 * ureg.inch, 3.0 * ureg.s

    ans1 = jax.jit(unitify(f))(a, b, c)
    ans2 = unitify(jax.jit(f))(a, b, c)
    assert unitify(jnp.allclose, static_typed=False)(
        ans1, ans2, atol=1e-8 * ureg.kg * ureg.s
    )

    comp = jax.jit(unitify(f)).lower(a, b, c).compile()
    print("comp", comp(a, b, c))


def f2(x, y, z):
    return f(x, y, z), f(1 / x, 1 / y, 1 / z)


@unitify
def test_jvp():
    _ = jax.jvp(
        f2,
        (1.0 * areg.m, 2.0 * areg.inch, 3.0 * areg.s),
        (1.0 * areg.m / areg.kg, 2.0 * areg.inch / areg.kg, 3.0 * areg.s / areg.kg),
    )


@unitify
def test_vjp():
    (o1, o2), vjp_fn = jax.vjp(f2, 1.0 * areg.m, 2.0 * areg.inch, 3.0 * areg.s)
    _ = vjp_fn((1 / o1, 1 / o2))
