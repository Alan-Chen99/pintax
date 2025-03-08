import oryx
from pint import Unit

from ._core import rules


@rules(oryx.core.interpreters.harvest.sow_p)
def _(*args: Unit, **_):
    return args
