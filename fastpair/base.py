#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FastPair: Data-structure for the dynamic closest-pair problem.

This data-structure is based on the observation that the conga line data
structure, in practice, does better the more subsets you give to it: even
though the worst case time for k subsets is O(nk log (n/k)), that worst case
seems much harder to reach than the nearest neighbor algorithm.

In the limit of arbitrarily many subsets, each new addition or point moved
by a deletion will be in a singleton subset, and the algorithm will
differ from nearest neighbors in only a couple of ways: (1) when we
create the initial data structure, we use a conga line rather than
all nearest neighbors, to keep the indegree of each point low, and
(2) when we insert a point, we don't bother updating other points'
neighbors.

Notes
-----
Total space: 20n bytes. (Could be reduced to 4n at some cost in update time.)
Time per insertion or single distance update: O(n)
Time per deletion or point update: O(n) expected, O(n^2) worst case
Time per closest pair: O(n)

References
----------
[1] Eppstein, David: Fast hierarchical clustering and other applications of
    dynamic closest pairs. Journal of Experimental Algorithmics 5 (2000) 1.
"""

# Copyright (c) 2016, Carson J. Q. Farmer <carsonfarmer@gmail.com>
# Copyright (c) 2002-2015, David Eppstein
# Licensed under the MIT Licence (http://opensource.org/licenses/MIT).

from __future__ import print_function, division, absolute_import
from itertools import combinations, cycle
from operator import itemgetter
from collections import defaultdict
import scipy.spatial.distance as dist


__all__ = ["FastPair", "dist"]


class attrdict(dict):
    """Simple dict with support for accessing elements as attributes."""

    def __init__(self, *args, **kwargs):
        super(attrdict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class FastPair(object):
    """FastPair 'sketch' class.
    """

    def __init__(self, min_points=10, dist=dist.euclidean):
        """Initialize an empty FastPair data-structure.

        Parameters
        ----------
        min_points : int, default=10
            The minimum number of points to add before initializing the
            data-structure. Queries _before_ `min_points` have been added to
            the data-structure will be brute-force.
        dist : function, default=scipy.spatial.distance.euclidean
            Can be any Python function that returns a distance (float) between
            between two vectors (tuples) `u` and `v`. Any distance function
            from `scipy.spatial.distance` will do the trick. By default, the
            Euclidean distance function is used. This function should play
            nicely with the `merge` function.
        """
        self.min_points = min_points
        self.dist = dist
        self.initialized = False  # Has the data-structure been initialized?
        self.neighbors = defaultdict(attrdict)  # Dict of neighbor points and dists
        self.points = list()  # Internal point set; entries may be non-unique

    def __add__(self, p):
        """Add a point and find its nearest neighbor.

        There is some extra logic here to allow us to gradually build up the
        FastPair data-structure until we reach `min_points`. Once `min_points`
        has been reached, we initialize the data-structure and start to take
        advantage of the FastPair efficiencies.
        """
        self.points.append(p)
        if self.initialized:
            self._find_neighbor(p)
        elif len(self) >= self.min_points:
            self.build()
        return self

    def __sub__(self, p):
        """Remove a point and update neighbors."""
        self.points.remove(p)
        if self.initialized:
            # We must update neighbors of points for which `p` had been nearest.
            for q in self.points:
                if self.neighbors[q].neigh == p:
                    res = self._find_neighbor(q)
        return self

    def __len__(self):
        """Number of points in data structure."""
        return len(self.points)

    def __call__(self):
        """Find closest pair by scanning list of nearest neighbors."""
        return self.closest_pair()

    def __contains__(self, p):
        return p in self.points

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, item):
        if not item in self:
            raise KeyError("{} not found".format(item))
        return self.neighbors[item]

    def __setitem__(self, item, value):
        if not item in self:
            raise KeyError("{} not found".format(item))
        self._update_point(item, value)

    def build(self, points=None):
        """Build a FastPairs data-structure from a set of (new) points.

        Here we use a conga line rather than calling explicitly (re)building
        the neighbors map multiple times as it is more efficient. This method
        needs to be called _before_ querying the data-structure or adding/
        removing any new points. Once it has been called, the internal
        `initialized` flag will be set to True. Otherwise, simple brute-force
        versions of queries and calculations will be used.

        Parameters
        ----------
        points : list of tuples/vectors, default=None
            An optional list of point tuples to be added to the point set,
            prior to computing the conga line and main FastPair data structure.
        """
        if points is not None:
            self.points += list(points)
        np = len(self)

        # Go through and find all neighbors, placing then in a conga line
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

    def closest_pair(self):
        """Find closest pair by scanning list of nearest neighbors.

        If `npoints` is less than `min_points`, a brute-force version
        of the closest pair algrithm is used.
        """
        if len(self) < 2:
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

    def closest_pair_brute_force(self):
        """Find closest pair using brute-force algorithm."""
        return _closest_pair_brute_force(self.points, self.dist)

    def sdist(self, p):
        """Compute distances from input to all other points in data-structure.

        This returns an iterator over all other points and their distance
        from the input point `p`. The resulting iterator returns tuples with
        the first item giving the distance, and the second item giving in
        neighbor point. The `min` of this iterator is essentially a brute-
        force 'nearest-neighbor' calculation. To do this, supply `itemgetter`
        (or a lambda version) as the `key` argument to `min`.

        Examples
        --------
        >>> fp = FastPair().build(points)
        >>> min(fp.sdist(point), key=itemgetter(0))
        """
        return ((self.dist(a, b), b) for a, b in zip(cycle([p]), self.points) if b != a)

    def _find_neighbor(self, p):
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

    def _update_point(self, old, new):
        """Update point location, neighbors, and distances.

        All distances to point have changed, we need to recompute all aspects
        of the data structure that may be affected. Note that although we
        completely recompute the neighbors of the original point (`old`), we
        don't explicitly rebuild the neighbors map, since that would double the
        number of distance computations made by this routine. Also, like
        deletion, we don't change any _other_ point's neighbor to the updated
        point.
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


def _closest_pair_brute_force(pts, dst=dist.euclidean):
    """Compute closest pair of points using brute-force algorithm.

    Notes
    -----
    Computes all possible combinations of points and compares their distances.
    This is _not_ efficient, nor scalable, but it provides a useful reference
    for more efficient algorithms. This version is actually pretty fast due
    to its use of fast Python builtins.
    """
    return min((dst(p1, p2), (p1, p2)) for p1, p2 in combinations(pts, r=2))
