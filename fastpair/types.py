# Copyright (c) 2016, Carson J. Q. Farmer <carson.farmer@gmail.com>
# Copyright (c) 2002-2015, David Eppstein
# Licensed under the MIT Licence (http://opensource.org/licenses/MIT).

import sys
from collections.abc import Callable, Generator, Iterable
from typing import Annotated, TypeAlias, TypedDict, TypeVar

################################################################
# See GH#73 -- remove once Python 3.11 is minimum
if sys.version_info.minor < 11:
    from typing_extensions import Self
else:
    from typing import Self
################################################################

import numpy

__all__ = [
    "Callable",
    "Iterable",
    "Self",
    "NumericCoord",
    "PointCoords",
    "PointsInput",
    "Dist",
    "DistNeighTuple",
    "AllNeighs",
    "DistNeighDict",
    "NeighCoords",
    "ClosestPair",
]


# point typing ---------------------------------------------------------------------
_Numeric = TypeVar("_Numeric", float, int, complex, numpy.number)
_NumericAnnotation = Annotated[_Numeric, "Any numeric value."]
NumericCoord: TypeAlias = _NumericAnnotation

_PointCoords = TypeVar("_PointCoords", bound=tuple[NumericCoord])
PointCoords = Annotated[_PointCoords, "A tuple of at least 2 numeric elements."]

_PointsInput = TypeVar("_PointsInput", bound=Iterable[PointCoords])
PointsInput = Annotated[_PointsInput, "Input for ``.build()``."]


# distance typing ------------------------------------------------------------------
Dist: TypeAlias = _NumericAnnotation

_DistNeighTuple = TypeVar("_DistNeighTuple", bound=tuple[Dist, PointCoords])
DistNeighTuple = Annotated[_DistNeighTuple, "Returned from ``.__getitem__()``."]

_AllNeighs = Generator[DistNeighTuple, None, None]
AllNeighs = Annotated[_AllNeighs, "Returned from ``.sdist()``."]


# pair typing ----------------------------------------------------------------------
class DistNeighDict(TypedDict):
    """The values of the ``FastPair.neighbors`` dictionary. Also potentially
    returned from ``._update_point()``. See GH#69.
    """

    dist: Dist
    neigh: PointCoords


_NeighCoords = tuple[PointCoords, PointCoords]
NeighCoords = Annotated[_NeighCoords, "2 instances of ``PointCoords``."]

_ClosestPair = TypeVar("_ClosestPair", bound=tuple[Dist, NeighCoords])
ClosestPair = Annotated[_ClosestPair, "Returned from ``.closest_pair()``."]
