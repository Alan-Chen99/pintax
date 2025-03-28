import oryx

from ._core import Unit, rules


@rules(oryx.core.interpreters.harvest.sow_p)
def _(*args: Unit, **_):
    return args
