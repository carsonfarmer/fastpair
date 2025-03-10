import numpy
import pytest


def rand_tuple(dim: int = 2, seed: None | int = None) -> tuple[float, ...]:
    rng = numpy.random.default_rng(seed)
    return tuple([round(rng.uniform(), 4) for j in range(dim)])


def point_set(n: int = 50, d: int = 10, seed: int = 0) -> list[tuple[float, ...]]:
    """Return ``numpy.array`` of shape ``(n, d)``."""
    return [rand_tuple(dim=d, seed=seed + i) for i in range(n)]


def pytest_configure(config):  # noqa: ARG001
    """PyTest session attributes, methods, etc."""

    pytest.rand_tuple = rand_tuple
    pytest.point_set = point_set


@pytest.fixture
def image_array():
    return [
        (b"\x00\x00\x07\x20\x00\x00\x03\x21\x08\x02\x00\x00\x00", "0"),
        (b"\x00\x50\x07\x60\x00\x00\x03\x21\x06\x02\x00\x00\x00", "1"),
        (b"\x00\x00\x07\x20\x00\x00\x03\x21\x08\x02\x00\x08\x00", "2"),
        (b"\x00\x50\x07\x60\x00\x00\x03\x21\x06\x02\x00\x60\x00", "3"),
        (b"\x00\x00\x07\x20\x00\x00\x03\x21\x08\x02\x00\x30\x01", "4"),
        (b"\x00\x50\x07\x60\x00\x00\x03\x21\x06\x02\x00\x00\x10", "5"),
    ]
