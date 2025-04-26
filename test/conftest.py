import jax

jax.config.update("jax_check_tracer_leaks", True)
print("setting jax_platform_name")
jax.config.update("jax_platform_name", "cpu")
