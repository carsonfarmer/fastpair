#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FastPair: Data-structure for the dynamic closest-pair problem.

Testing module for FastPair.
"""

# Copyright (c) 2016, Carson J. Q. Farmer <carsonfarmer@gmail.com>
# Copyright (c) 2002-2015, David Eppstein
# Licensed under the MIT Licence (http://opensource.org/licenses/MIT).

from __future__ import print_function, division, absolute_import

from operator import itemgetter
from types import FunctionType
from itertools import cycle, combinations, groupby
import random
import pytest
from fastpair import FastPair, interact
from math import isinf, isnan

from scipy import mean, array, unique

def contains_same(s, t):
    s, t = set(s), set(t)
    return s >= t and s <= t


def all_close(s, t, tol=1e-8):
    # Ignores inf and nan values...
    return all(abs(a - b) < tol for a, b in zip(s, t)
               if not isinf(a) and not isinf(b) and
                  not isnan(a) and not isnan(b))


def rand_tuple(dim=2):
    return tuple([random.random() for _ in range(dim)])


def to_codebook(X, part):
    """Calculates centroids according to flat cluster assignment."""
    codebook = []
    X = array(X)
    for i in unique(part):
        codebook.append(tuple(X[part == i].mean(0)))
    return codebook


# Setup fixtures
@pytest.fixture(scope="module")
def PointSet(n=50, d=10):
    """Return numpy array of shape `n`x`d`."""
    # random.seed(8714)
    return [rand_tuple(d) for _ in range(n)]


class TestFastPairs:
    """Main test class."""

    def test_init(self):
        fp = FastPair()
        assert fp.min_points == 10
        assert isinstance(fp.dist, FunctionType)
        assert fp.initialized is False
        assert len(fp.points) == 0
        assert len(fp.neighbors) == 0

    def test_build(self):
        ps = PointSet()
        fp = FastPair().build(ps)
        assert len(fp) == len(ps)
        assert len(fp.neighbors) == len(ps)
        assert fp.initialized is True

    def test_add(self):
        ps = PointSet()
        fp = FastPair()
        for p in ps[:9]:
            fp += p
        assert fp.initialized is False
        assert len(fp) == 9
        for p in ps[9:]:
            fp += p
        assert fp.initialized is True

    def test_sub(self):
        ps = PointSet()
        fp = FastPair().build(ps)
        start = fp._find_neighbor(ps[-1])
        fp -= ps[-1]
        end = fp._find_neighbor(start["neigh"])
        assert end["neigh"] != ps[-1]
        # This is risky, because it might legitimately be the same...?
        assert start["dist"] != end["dist"]
        assert len(fp) == len(ps)-1
        with pytest.raises(ValueError):
            fp -= rand_tuple(len(ps[0]))

    def test_len(self):
        ps = PointSet()
        fp = FastPair()
        assert len(fp) == 0
        fp.build(ps)
        assert len(fp) == len(ps)

    def test_contains(self):
        ps = PointSet()
        fp = FastPair()
        assert ps[0] not in fp
        fp.build(ps)
        assert ps[0] in fp

    def test_call_and_closest_pair(self):
        ps = PointSet()
        fp = FastPair().build(ps)
        cp = fp.closest_pair()
        bf = fp.closest_pair_brute_force()
        assert fp() == cp
        assert abs(cp[0] - bf[0]) < 1e-8
        assert cp[1] == bf[1]

    def test_all_closest_pairs(self):
        ps = PointSet()
        fp = FastPair().build(ps)
        cp = fp.closest_pair()
        bf = fp.closest_pair_brute_force()  # Ordering should be the same
        # dc = fp.closest_pair_divide_conquer()  # Maybe different ordering
        assert abs(cp[0] - bf[0]) < 1e-8
        assert cp[1] == bf[1]  # Tuple comparison
        test = min([(fp.dist(a, b), (a, b)) for a, b in combinations(ps, r=2)], key=itemgetter(0))
        assert abs(cp[0] - test[0]) < 1e-8
        assert sorted(cp[1]) == sorted(test[1])  # Tuple comparison
        # assert abs(dc[0] - cp[0]) < 1e-8  # Compare distance
        # Ordering may be different, but both should be in there
        # assert dc[1][0] in cp[1] and dc[1][1] in cp[1]

    def test_find_neighbor_and_sdist(self):
        ps = PointSet()
        fp = FastPair().build(ps)
        rando = rand_tuple(len(ps[0]))
        neigh = fp._find_neighbor(rando)  # Abusing find_neighbor!
        dist = fp.dist(rando, neigh["neigh"])
        assert  abs(dist - neigh["dist"]) < 1e-8
        assert len(fp) == len(ps)  # Make sure we didn't add a point...
        l = [(fp.dist(a, b), b) for a, b in zip(cycle([rando]), ps)]
        res = min(l, key=itemgetter(0))
        assert abs(res[0] - neigh["dist"]) < 1e-8
        assert res[1] == neigh["neigh"]
        res = min(fp.sdist(rando), key=itemgetter(0))
        assert abs(neigh["dist"] - res[0]) < 1e-8
        assert neigh["neigh"] == res[1]

    def test_cluster(self):
        ps = PointSet()
        fp = FastPair().build(ps)
        for i in range(len(fp)-1):
            # Version one
            dist, (a, b) = fp.closest_pair()
            c = interact(a, b)
            fp -= b  # Drop b
            fp -= a
            fp += c
            # Order gets reversed here...
            d, (e, f) = min([(fp.dist(i, j), (i, j)) for i, j in
                             combinations(ps, r=2)], key=itemgetter(0))
            g = interact(e, f)
            assert abs(d - dist) < 1e-8
            assert (a == e or b == e) and (b == f or a == f)
            assert c == g
            ps.remove(e)
            ps.remove(f)
            ps.append(g)
            assert contains_same(fp.points, ps)
        assert len(fp.points) == len(ps) == 1

    def test_update_point(self):
        # Still failing sometimes...
        ps = PointSet()
        fp = FastPair().build(ps)
        assert len(fp) == len(ps)
        old = ps[0]  # Just grab the first point...
        new = rand_tuple(len(ps[0]))
        res = fp.update_point(old, new)
        assert old not in fp
        assert new in fp
        assert len(fp) == len(ps)  # Size shouldn't change
        l = [(fp.dist(a, b), b) for a, b in zip(cycle([new]), ps)]
        res = min(l, key=itemgetter(0))
        neigh = fp._find_neighbor(new)  # Abusing fin_neighbor!
        assert abs(res[0] - neigh["dist"]) < 1e-8
        assert res[1] == neigh["neigh"]

    def test_merge_closest(self):
        # Still failing sometimes...
        ps = PointSet()
        fp1 = FastPair().build(ps)
        fp2 = FastPair().build(ps)
        n = len(ps)
        while n >= 2:
            dist, (a, b) = fp1.closest_pair()
            new = interact(a, b)
            fp1 -= b  # Drop b
            fp1.update_point(a, new)
            fp2.merge_closest()
            n -= 1
        assert len(fp1) == len(fp2) == 1 # == len(fp2)
        assert fp1.points == fp2.points # == fp2.points
        # Compare points
        assert contains_same(list(fp1.neighbors.keys()), list(fp2.neighbors.keys()))
        # Compare neighbors
        assert contains_same([n["neigh"] for n in fp1.neighbors.values()],
                             [n["neigh"] for n in fp2.neighbors.values()])
        # Compare dists
        assert all_close([n["dist"] for n in fp1.neighbors.values()],
                         [n["dist"] for n in fp2.neighbors.values()])
        # Should have < 2 points now...
        with pytest.raises(ValueError):
            fp1.closest_pair()
            fp2.closest_pair()
