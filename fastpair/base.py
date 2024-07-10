# Copyright (c) 2016, Carson J. Q. Farmer <carson.farmer@gmail.com>
# Copyright (c) 2002-2015, David Eppstein
# Licensed under the MIT Licence (http://opensource.org/licenses/MIT).

from collections import defaultdict
from collections.abc import Callable, Generator, Iterable
from itertools import combinations, cycle
from typing import Annotated, Self, TypeAlias, TypedDict, TypeVar

import numpy
import scipy.spatial.distance as dist

__all__ = ["FastPair", "dist"]


# point typing ---------------------------------------------------------------------
_Numeric = TypeVar("_Numeric", float, int, complex, numpy.number)
NumericCoord: TypeAlias = Annotated[_Numeric, "Any numeric value."]

_PointCoords = TypeVar("_PointCoords", bound=tuple[NumericCoord])
PointCoords = Annotated[_PointCoords, "Must be a tuple of at least 2 elements."]

_PointsInput = TypeVar("_PointsInput", bound=Iterable[PointCoords])
PointsInput = Annotated[_PointsInput, "Input for ``.build()``."]


# distance typing ------------------------------------------------------------------
Dist: TypeAlias = Annotated[_Numeric, "Any numeric value."]

_DistNeighTuple = TypeVar("_DistNeighTuple", bound=tuple[Dist, PointCoords])
DistNeighTuple = Annotated[_DistNeighTuple, "Returned from ``.__getitem__()``."]

_AllNeighs = Generator[DistNeighTuple, None, None]
AllNeighs = Annotated[_AllNeighs, "Returned from ``.sdist()``."]


# pair typing ----------------------------------------------------------------------
class DistNeighDict(TypedDict):
    """The values of the ``FastPair.neighbors`` dictionary.
    Also potentially returned from ``._update_point()``.
    See GH#69.
    """

    dist: Dist
    neigh: PointCoords


_NeighCoords = tuple[PointCoords, PointCoords]
NeighCoords = Annotated[_NeighCoords, "2 instances of ``PointCoords``."]

_ClosestPair = TypeVar("_ClosestPair", bound=tuple[Dist, NeighCoords])
ClosestPair = Annotated[_ClosestPair, "Returned from ``.closest_pair()``."]


class AttrDict(dict):
    """Simple ``dict`` with support for accessing elements as attributes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class FastPair:
    """FastPair: Data-structure for the dynamic closest-pair problem.

    This data structure is based on the observation that the conga line data
    structure, in practice, does better the more subsets you give to it: even
    though the worst case time for :math:`k` subsets is :math:`O(nk log (n/k))`,
    that worst case seems much harder to reach than the nearest neighbor algorithm.

    In the limit of arbitrarily many subsets, each new addition or point moved
    by a deletion will be in a singleton subset, and the algorithm will
    differ from nearest neighbors in only a couple of ways:

        1. when we create the initial data structure, we use a conga line rather than
            all nearest neighbors, to keep the indegree of each point low, and
        2. when we insert a point, we don't bother updating other points' neighbors.

    Notes
    -----
    * Total space: :math:`20n` bytes, which could be reduced to
        :math:`4n` at some cost in update time.
    * Time per insertion or single distance update: :math:`O(n)`
    * Time per deletion or point update: :math:`O(n)` expected,
        :math:`O(n^2)` worst case
    * Time per closest pair: :math:`O(n)`

    References
    ----------
    [1] Eppstein, David: Fast hierarchical clustering and other applications of
        dynamic closest pairs. Journal of Experimental Algorithmics 5 (2000) 1.
        https://doi.org/10.1145/351827.351829
    """

    def __init__(self, min_points: int = 10, dist: Callable = dist.euclidean) -> None:
        """Initialize an empty ``FastPair`` data-structure.

        Parameters
        ----------
        min_points : int, default=10
            The minimum number of points to add before initializing the
            data structure. Queries *before* ``min_points`` have been added to
            the data structure will be brute-force.
        dist : Callable, default=scipy.spatial.distance.euclidean
            Can be any Python function that returns a distance (float) between
            between two vectors (tuples) ``u`` and ``v``. Any distance function
            from ``scipy.spatial.distance`` will do the trick. By default, the
            Euclidean distance function is used. This function should play
            nicely with the ``merge`` function.

        Attributes
        ----------
        initialized : bool
            ``True`` after the ``build()`` method is called.
        neighbors : defaultdict[AttrDict]
            Internally defined ``dict`` with support for accessing
            elements as attributes where keys are ``PointCoords``
            and values are ``DistNeighDict``.
        points : None | PointsInput
            An list of 2D (two-dimensional) point tuples in the point set.
            All objects in ``points`` **must** have equivalent dimensions.

        Examples
        --------
        Create empty data structure with default values.

        >>> import fastpair
        >>> fp = fastpair.FastPair()
        >>> fp
        <FastPair[min_points=10, dist=euclidean]
         Current state: not initialized with 0 0D points>

        Add points and build conga line.

        >>> points_2d = ((1, 1), (2, 2), (4, 4))
        >>> fp.build(points_2d)
        <FastPair[min_points=10, dist=euclidean]
         Current state: initialized with 3 2D points>

        Query closest pair.

        >>> fp.closest_pair()
        (np.float64(1.4142135623730951), ((1, 1), (2, 2)))

        Add a point to the data structure.

        >>> fp += (1, 0)
        >>> fp
        <FastPair[min_points=10, dist=euclidean]
         Current state: initialized with 4 2D points>

        Query again for closest pair.

        >>> fp.closest_pair()
        (np.float64(1.0), ((1, 0), (1, 1)))

        Remove a point from the data structure.

        >>> fp -= (1, 1)
        >>> fp
        <FastPair[min_points=10, dist=euclidean]
         Current state: initialized with 3 2D points>

        Query again for closest pair.

        >>> fp.closest_pair()
        (np.float64(2.23606797749979), ((1, 0), (2, 2)))

        The default distance metric is ``scipy.spatial.distance.euclidean``.
        Let's rebuild the ``FastPair`` with another metric. Many of the
        distance functions available via ``scipy.spatial.distance`` can used.
        Here we'll try ``cityblock``, which Manhattan distance.

        >>> from scipy.spatial import distance
        >>> fp = fastpair.FastPair(min_points=3, dist=distance.cityblock)
        >>> fp.build(points_2d)
        <FastPair[min_points=3, dist=cityblock]
         Current state: initialized with 3 2D points>

        While the closet pair remains the same, the distance between the
        pair has increased with the new distance function.

        >>> fp.closest_pair()
        (np.int64(2), ((1, 1), (2, 2)))
        """

        self.min_points = min_points
        self.dist = dist
        # Has the data-structure been initialized?
        self.initialized = False
        # Dict of neighbor points and dists
        self.neighbors: defaultdict = defaultdict(AttrDict)
        # Internal point set; entries may be non-unique
        self.points: list = []

    def __repr__(self) -> str:
        mp = self.min_points
        d = self.dist.__name__
        init = "" if self.initialized else "not "
        curr_pnts = len(self)
        dim = 0 if not self.points else len(self.points[0])
        return (
            f"<FastPair[min_points={mp}, dist={d}]\n"
            f" Current state: {init}initialized with {curr_pnts} {dim}D points>"
        )

    def __add__(self, p: PointCoords) -> Self:
        """Add a point and find its nearest neighbor.

        There is some extra logic here to allow us to gradually build up the
        ``FastPair`` data structure until we reach ``min_points``. Once ``min_points``
        has been reached, we initialize the data structure and start to take
        advantage of the ``FastPair`` efficiencies.
        """
        self.points.append(p)
        if self.initialized:
            self._find_neighbor(p)
        elif len(self) >= self.min_points:
            self.build()
        return self

    def __sub__(self, p: PointCoords) -> Self:
        """Remove a point and update neighbors."""
        self.points.remove(p)
        if self.initialized:
            # We must update neighbors of points for which `p` had been nearest.
            for q in self.points:
                if self.neighbors[q].neigh == p:
                    self._find_neighbor(q)
        return self

    def __len__(self) -> int:
        """Number of points in data structure."""
        return len(self.points)

    def __call__(self) -> ClosestPair:
        """Find closest pair by scanning list of nearest neighbors."""
        return self.closest_pair()

    def __contains__(self, p: PointCoords) -> bool:
        return p in self.points

    def __iter__(self) -> Iterable:
        return iter(self.points)

    def __getitem__(self, item: PointCoords) -> DistNeighTuple:
        if item not in self:
            raise KeyError(f"{item} not found")
        return self.neighbors[item]

    def __setitem__(self, item: PointCoords, value: Dist | PointCoords):
        if item not in self:
            raise KeyError(f"{item} not found")
        self._update_point(item, value)

    def build(self, points: None | PointsInput = None) -> Self:
        """Build a ``FastPair`` data-structure from a set of (new) points.

        Here we use a conga line rather than calling explicitly (re)building
        the neighbors map multiple times as it is more efficient. This method
        needs to be called *before* querying the data structure or adding/
        removing any new points. Once it has been called, the internal
        ``initialized`` flag will be set to ``True``. Otherwise, simple
        brute-force versions of queries and calculations will be used.

        Parameters
        ----------
        points : None | PointsInput, default=None
            An optional list of 2D+ (two-dimensional or greater) point tuples
            to be added to the point set, prior to computing the conga line
            and main ``FastPair`` data structure. All objects in ``points``
            **must** have equivalent dimensions.

        Examples
        --------

        Initialize and build the data structure 300 equally-spaced 2D points.

        >>> import fastpair
        >>> fp = fastpair.FastPair().build([(i, 1) for i in range(300)])
        >>> fp
        <FastPair[min_points=10, dist=euclidean]
         Current state: initialized with 300 2D points>

        Ensure all neighbor distances are exactly 1.0 (except the last point
        which has no conga-style neighbor).

        >>> all([fp.neighbors[i]["dist"] == 1.0 for i in fp.points[:-1]])
        True

        Since the last point has no neighbors, its distance is assigned as ``inf``.

        >>> fp.neighbors[fp.points[-1]]["dist"]
        inf
        """
        if points is not None:
            self.points += list(points)
        np = len(self)

        # Go through and find all neighbors, placing them in a conga line
        for i in range(np - 1):
            # Find neighbor to p[0] to start
            nbr = i + 1
            nbd = float("inf")
            for j in range(i + 1, np):
                d = self.dist(self.points[i], self.points[j])
                if d < nbd:
                    nbr = j
                    nbd = d
            # Add that edge, move nbr to points[i+1]
            self.neighbors[self.points[i]].dist = nbd
            self.neighbors[self.points[i]].neigh = self.points[nbr]
            self.points[nbr] = self.points[i + 1]
            self.points[i + 1] = self.neighbors[self.points[i]].neigh
        # No more neighbors, terminate conga line.
        # Last person on the line has no neigbors :(
        self.neighbors[self.points[np - 1]].neigh = self.points[np - 1]
        self.neighbors[self.points[np - 1]].dist = float("inf")
        self.initialized = True
        return self

    def closest_pair(self) -> ClosestPair:
        """Find closest pair by scanning list of nearest neighbors.

        If ``npoints`` is less than ``min_points``, a brute-force version
        of the closest pair algorithm is used.

        Examples
        --------
        Create and build the data structure with default values.

        >>> import fastpair
        >>> fp = fastpair.FastPair(min_points=3).build(((1, 1), (2, 2), (4, 4)))
        >>> fp
        <FastPair[min_points=3, dist=euclidean]
         Current state: initialized with 3 2D points>

        Query closest pair.

        >>> fp.closest_pair()
        (np.float64(1.4142135623730951), ((1, 1), (2, 2)))
        """
        npoints = len(self)
        if npoints < 2:
            raise ValueError("Must have `npoints >= 2` to form a pair.")
        elif not self.initialized:
            return self.closest_pair_brute_force()
        a = self.points[0]  # Start with first point
        d = self.neighbors[a].dist
        for p in self.points:
            if self.neighbors[p].dist < d:
                a = p  # Update `a` and distance `d`
                d = self.neighbors[p].dist
        b = self.neighbors[a].neigh
        return d, (a, b)

    def closest_pair_brute_force(self) -> ClosestPair:
        """Find closest pair using brute-force algorithm."""
        return _closest_pair_brute_force(self.points, self.dist)

    def sdist(self, p: PointCoords) -> AllNeighs:
        """Compute distances from input to all other points in data structure.

        This returns an iterator over all other points and their distance
        from the input point ``p``. The resulting iterator returns tuples with
        the first item giving the distance, and the second item giving in
        neighbor point. The ``min`` of this iterator is essentially a brute-
        force 'nearest-neighbor' calculation. To do this, supply ``operator.itemgetter``
        (or a ``lambda`` function) as the ``key`` argument to ``min``.

        Examples
        --------
        Initialize and build the data structure.

        >>> import fastpair, operator
        >>> points = ((1, 1), (2, 2), (4, 4))
        >>> fp = fastpair.FastPair().build(points)

        Query the data structure using the ``.sdist()`` method for the closest
        single point to ``(1, 1)``.

        >>> focal_point = (1, 1)
        >>> min(fp.sdist(focal_point), key=operator.itemgetter(0))
        (np.float64(1.4142135623730951), (2, 2))
        """

        return ((self.dist(a, b), b) for a, b in zip(cycle([p]), self.points) if b != a)

    def _find_neighbor(self, p: PointCoords):
        """Find and update nearest neighbor of a given point."""
        # If no neighbors available, set flag for `_update_point` to find
        if len(self) < 2:
            self.neighbors[p].neigh = p
            self.neighbors[p].dist = float("inf")
        else:
            # Find first point unequal to `p` itself
            first_nbr = 0
            if p == self.points[first_nbr]:
                first_nbr = 1
            self.neighbors[p].neigh = self.points[first_nbr]
            self.neighbors[p].dist = self.dist(p, self.neighbors[p].neigh)
            # Now test whether each other point is closer
            for q in self.points[first_nbr + 1 :]:
                if p != q:
                    d = self.dist(p, q)
                    if d < self.neighbors[p].dist:
                        self.neighbors[p].dist = d
                        self.neighbors[p].neigh = q
        return dict(self.neighbors[p])  # Return plain ol' dict

    def _update_point(self, old: PointCoords, new: PointCoords):
        """Update point location, neighbors, and distances.

        All distances to point have changed, we need to recompute all aspects
        of the data structure that may be affected. Note that although we
        completely recompute the neighbors of the original point (``old``), we
        don't explicitly rebuild the neighbors map, since that would double the
        number of distance computations made by this routine. Also, like
        deletion, we don't change any *other* point's neighbor to the updated point.
        """
        # Out with the old, in with the new...
        self.points.remove(old)
        self.points.append(new)
        if not self.initialized:
            return new
        del self.neighbors[old]
        self.neighbors[new].neigh = new  # Flag for not yet found any
        self.neighbors[new].dist = float("inf")
        for q in self.points:
            if q != new:
                d = self.dist(new, q)
                if d < self.neighbors[new].dist:
                    self.neighbors[new].dist = d
                    self.neighbors[new].neigh = q
                if self.neighbors[q].neigh == old:
                    if d > self.neighbors[q].dist:
                        self._find_neighbor(q)
                    else:
                        self.neighbors[q].neigh = new
                        self.neighbors[q].dist = d
        return dict(self.neighbors[new])

    # def merge_closest(self):
    #     dist, (a, b) = self.closest_pair()
    #     c = self.merge(a, b)
    #     self -= b
    #     return self._update_point(a, c)


def _closest_pair_brute_force(
    pts: PointsInput, dst: Callable = dist.euclidean
) -> ClosestPair:
    """Compute closest pair of points using brute-force algorithm.

    Notes
    -----
    Computes all possible combinations of points and compares their distances.
    This is *not* efficient, nor scalable, but it provides a useful reference
    for more efficient algorithms. This version is actually pretty fast due
    to its use of fast Python builtins.
    """
    return min((dst(p1, p2), (p1, p2)) for p1, p2 in combinations(pts, r=2))
