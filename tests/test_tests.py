import pytest


def test_iterable_fixture(iter_fix):
    print(iter_fix)


@pytest.mark.parametrize(
    "get_i, _x", list(zip(range(10), range(10))), indirect=["get_i"]
)
def test_get_i(get_i, _x):
    assert get_i == _x**2
