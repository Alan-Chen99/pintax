import pytest
from _pytest.python import Function

from pintax._core import PintaxNotImplementedError


def pytest_runtest_makereport(item: Function, call: pytest.CallInfo):
    if (
        item.function.__module__ == "test_jnp"
        and call.excinfo is not None
        and isinstance(call.excinfo.value, PintaxNotImplementedError)
    ):
        report = pytest.TestReport.from_item_and_call(item, call)
        report.outcome = "xfailed"  # type: ignore
        return report
