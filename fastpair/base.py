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

from __future__ import (print_function as _print_function,
                        division as _division,
                        absolute_import as _absolute_import)

from itertools import combinations as _combs, cycle as _cycle
from operator import itemgetter as _getter
from collections import defaultdict as _ddict
import scipy.spatial.distance as dist
from scipy import mean as _mean, array as _array

__all__ = ["interact", "FastPair", "dist", "default_dist"]

default_dist = dist.euclidean


def interact(u, v):
    """Compute element-wise mean(s) from two arrays."""
    return tuple(_mean(_array([u, v]), axis=0))


class _adict(dict):
    """Simple dict with support for accessing elements as attributes."""
    def __init__(self, *args, **kwargs):
        super(_adict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class FastPair(object):
    """FastPair 'sketch' class.
    """
    def __init__(self, min_points=10, dist=default_dist, merge=interact):
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
        merge : function, default=scipy.mean
            Can be any Python function that returns a single 'point' from two
            input 'points'. By default, the element-wise mean(s) from two input
            point arrays is used. If a user has a 'special' point class; for
            example, one that represents cluster centroids, then the user can
            specify a function that returns valid clusters. This function
            should play nicely with the `dist` function.
        """
        self.min_points = min_points
        self.dist = dist
        self.merge = merge
        self.initialized = False  # Has the data-structure been initialized?
        self.neighbors = _ddict(_adict)  # Dict of neighbor points and dists
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
        for i in range(np-1):
            # Find neighbor to p[0] to start
            nbr = i + 1
            nbd = float("inf")
            for j in range(i+1, np):
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
        self.neighbors[self.points[np-1]].neigh = self.points[np-1]
        self.neighbors[self.points[np-1]].dist = float("inf")
        self.initialized = True
        return self

    def closest_pair(self):
        """Find closest pair by scanning list of nearest neighbors.

        If `npoints` is less than `min_points`, a brute-force version
        of the closest pair algrithm is used.
        """
        if len(self) < 2:
            raise ValueError("Must have `npoints >= 2` to form a pair.")
        elif len(self) < self.min_points:
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
        return _closest_pair_brute_force(self.points)

    def closest_pair_divide_conquer(self):
        """Find closest pair using divide-and-conquer algorithm."""
        return _closest_pair_divide_conquer(self.points)

    def _find_neighbor(self, p):
        """Find and update nearest neighbor of a given point."""
        # If no neighbors available, set flag for `update_point` to find
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
            for q in self.points[first_nbr+1:]:
                if p != q:
                    d = self.dist(p, q)
                    if d < self.neighbors[p].dist:
                        self.neighbors[p].dist = d
                        self.neighbors[p].neigh = q
        return dict(self.neighbors[p])  # Return plain ol' dict

    def merge_closest(self):
        dist, (a, b) = self.closest_pair()
        c = self.merge(a, b)
        self -= b
        return self.update_point(a, c)

    def update_point(self, old, new):
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

    def update_dist(self, p, q):
        """Single distance has changed, check if structures are ok."""
        # This is rarely called for most applications I'm interested in.
        # TODO: Decide if its worth keeping...?
        d = self.dist(p, q)
        if d < self.neighbors[p].dist:
            self.neighbors[p].dist = d
            self.neighbors[p].neigh = q
        elif self.neighbors[p].neigh == q and d > self.neighbors[p].dist:
            self._find_neighbor(p)

        if d < self.neighbors[q].dist:
            self.neighbors[q].dist = d
            self.neighbors[q].neigh = q
        elif self.neighbors[q].neigh == p and d > self.neighbors[q].dist:
            self._find_neighbor(q)
        return d

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
        return ((self.dist(a, b), b) for a, b in
                zip(_cycle([p]), self.points) if b != a)


def _closest_pair_brute_force(pts, dst=default_dist):
    """Compute closest pair of points using brute-force algorithm.

    Notes
    -----
    Computes all possible combinations of points and compares their distances.
    This is _not_ efficient, nor scalable, but it provides a useful reference
    for more efficient algorithms. This version is actually pretty fast due
    to its use of fast Python builtins.
    """
    return min((dst(p1, p2), (p1, p2)) for p1, p2 in _combs(pts, r=2))

def _closest_pair_divide_conquer(pts, dst=default_dist):
    """Compute closest pair of points using divide and conquer algorithm.

    References
    ----------
    https://www.cs.iupui.edu/~xkzou/teaching/CS580/Divide-and-conquer-closestPair.ppt
    https://www.rosettacode.org/wiki/Closest-pair_problem#Python
    """
    xp = sorted(pts, key=_getter(0))  # Sort by x
    yp = sorted(pts, key=_getter(1))  # Sort by y
    return _divide_and_conquer(xp, yp)

def _divide_and_conquer(xp, yp, dst=default_dist):
    np = len(xp)
    if np <= 3:
        return _closest_pair_brute_force(xp, dst=dst)
    Pl = xp[:np//2]
    Pr = xp[np//2:]
    Yl, Yr = [], []
    divider = Pl[-1][0]
    for p in yp:
        if p[0] <= divider:
            Yl.append(p)
        else:
            Yr.append(p)
    dl, pairl = _divide_and_conquer(Pl, Yl)
    dr, pairr = _divide_and_conquer(Pr, Yr)
    dm, pairm = (dl, pairl) if dl < dr else (dr, pairr)
    # Points within dm of divider sorted by Y coord
    # We use abs here because we're only measuring distance in one direction
    close = [p for p in yp  if abs(p[0] - divider) < dm]
    num_close = len(close)
    if num_close > 1:
        # There is a proof that you only need compare a max of 7 next points
        closest = min(((dst(close[i], close[j]), (close[i], close[j]))
                       for i in range(num_close-1)
                       for j in range(i+1, min(i+8, num_close))),
                      key=_getter(0))
        return (dm, pairm) if dm <= closest[0] else closest
    else:
        return dm, pairm
