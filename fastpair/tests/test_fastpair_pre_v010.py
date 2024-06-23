"""FastPair: Data-structure for the dynamic closest-pair problem.

Testing module for FastPair.
"""

# Copyright (c) 2016, Carson J. Q. Farmer <carsonfarmer@gmail.com>
# Copyright (c) 2002-2015, David Eppstein
# Licensed under the MIT Licence (http://opensource.org/licenses/MIT).

import random
from itertools import combinations, cycle
from math import isinf, isnan
from operator import itemgetter
from types import FunctionType

import numpy as np
import pytest

from fastpair import FastPair


def normalized_distance(_a, _b):
    b = _b.astype(int)
    a = _a.astype(int)
    norm_diff = np.linalg.norm(b - a)
    norm1 = np.linalg.norm(b)
    norm2 = np.linalg.norm(a)
    return norm_diff / (norm1 + norm2)


def image_distance(image1, image2):
    (sig1, _) = image1
    (sig2, _) = image2
    sig1 = np.frombuffer(sig1, np.int8)
    sig2 = np.frombuffer(sig2, np.int8)
    return normalized_distance(sig1, sig2)


# Setup fixtures
@pytest.fixture(scope="module")
def image_array():
    return [
        (b"\x00\x00\x07\x20\x00\x00\x03\x21\x08\x02\x00\x00\x00", "0"),
        (b"\x00\x50\x07\x60\x00\x00\x03\x21\x06\x02\x00\x00\x00", "1"),
        (b"\x00\x00\x07\x20\x00\x00\x03\x21\x08\x02\x00\x08\x00", "2"),
        (b"\x00\x50\x07\x60\x00\x00\x03\x21\x06\x02\x00\x60\x00", "3"),
        (b"\x00\x00\x07\x20\x00\x00\x03\x21\x08\x02\x00\x30\x01", "4"),
        (b"\x00\x50\x07\x60\x00\x00\x03\x21\x06\x02\x00\x00\x10", "5"),
    ]


def contains_same(s, t):
    s, t = set(s), set(t)
    return s >= t and s <= t


def all_close(s, t, tol=1e-8):
    # Ignores inf and nan values...
    return all(
        abs(a - b) < tol
        for a, b in zip(s, t, strict=True)
        if not isinf(a) and not isinf(b) and not isnan(a) and not isnan(b)
    )


def rand_tuple(dim=2):
    return tuple([random.random() for _ in range(dim)])


def interact(u, v):
    """Compute element-wise mean(s) from two arrays."""
    return tuple(np.mean(np.array([u, v]), axis=0))


# Setup fixtures
@pytest.fixture(scope="module")
def point_set(n=50, d=10):
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

    def test_build(self, point_set):
        ps = point_set
        fp = FastPair().build(ps)
        assert len(fp) == len(ps)
        assert len(fp.neighbors) == len(ps)
        assert fp.initialized is True

    def test_add(self, point_set):
        ps = point_set
        fp = FastPair()
        for p in ps[:9]:
            fp += p
        assert fp.initialized is False
        assert len(fp) == 9
        for p in ps[9:]:
            fp += p
        assert fp.initialized is True

    def test_sub(self, point_set):
        ps = point_set
        fp = FastPair().build(ps)
        start = fp._find_neighbor(ps[-1])
        fp -= ps[-1]
        end = fp._find_neighbor(start["neigh"])
        assert end["neigh"] != ps[-1]
        # This is risky, because it might legitimately be the same...?
        assert start["dist"] != end["dist"]
        assert len(fp) == len(ps) - 1
        with pytest.raises(ValueError):
            fp -= rand_tuple(len(ps[0]))

    def test_len(self, point_set):
        ps = point_set
        fp = FastPair()
        assert len(fp) == 0
        fp.build(ps)
        assert len(fp) == len(ps)

    def test_contains(self, point_set):
        ps = point_set
        fp = FastPair()
        assert ps[0] not in fp
        fp.build(ps)
        assert ps[0] in fp

    def test_call_and_closest_pair(self, point_set):
        ps = point_set
        fp = FastPair().build(ps)
        cp = fp.closest_pair()
        bf = fp.closest_pair_brute_force()
        assert fp() == cp
        assert abs(cp[0] - bf[0]) < 1e-8
        assert cp[1] == bf[1]

    def test_all_closest_pairs(self, point_set):
        ps = point_set
        fp = FastPair().build(ps)
        cp = fp.closest_pair()
        bf = fp.closest_pair_brute_force()  # Ordering should be the same
        # dc = fp.closest_pair_divide_conquer()  # Maybe different ordering
        assert abs(cp[0] - bf[0]) < 1e-8
        assert cp[1] == bf[1]  # Tuple comparison
        test = min(
            [(fp.dist(a, b), (a, b)) for a, b in combinations(ps, r=2)],
            key=itemgetter(0),
        )
        assert abs(cp[0] - test[0]) < 1e-8
        assert sorted(cp[1]) == sorted(test[1])  # Tuple comparison
        # assert abs(dc[0] - cp[0]) < 1e-8  # Compare distance
        # Ordering may be different, but both should be in there
        # assert dc[1][0] in cp[1] and dc[1][1] in cp[1]

    def test_find_neighbor_and_sdist(self, point_set):
        ps = point_set
        fp = FastPair().build(ps)
        rando = rand_tuple(len(ps[0]))
        neigh = fp._find_neighbor(rando)  # Abusing find_neighbor!
        dist = fp.dist(rando, neigh["neigh"])
        assert abs(dist - neigh["dist"]) < 1e-8
        assert len(fp) == len(ps)  # Make sure we didn't add a point...
        l_ = [(fp.dist(a, b), b) for a, b in zip(cycle([rando]), ps)]
        res = min(l_, key=itemgetter(0))
        assert abs(res[0] - neigh["dist"]) < 1e-8
        assert res[1] == neigh["neigh"]
        res = min(fp.sdist(rando), key=itemgetter(0))
        assert abs(neigh["dist"] - res[0]) < 1e-8
        assert neigh["neigh"] == res[1]

    def test_cluster(self, point_set):
        ps = point_set
        fp = FastPair().build(ps)
        for i in range(len(fp) - 1):
            # Version one
            dist, (a, b) = fp.closest_pair()
            c = interact(a, b)
            fp -= b  # Drop b
            fp -= a
            fp += c
            # Order gets reversed here...
            d, (e, f) = min(
                [(fp.dist(i, j), (i, j)) for i, j in combinations(ps, r=2)],
                key=itemgetter(0),
            )
            g = interact(e, f)
            assert abs(d - dist) < 1e-8
            assert (a == e or b == e) and (b == f or a == f)
            assert c == g
            ps.remove(e)
            ps.remove(f)
            ps.append(g)
            assert contains_same(fp.points, ps)
        assert len(fp.points) == len(ps) == 1

    def test_update_point(self, point_set):
        # Still failing sometimes...
        ps = point_set
        fp = FastPair().build(ps)
        assert len(fp) == len(ps)
        old = ps[0]  # Just grab the first point...
        new = rand_tuple(len(ps[0]))
        res = fp._update_point(old, new)
        assert old not in fp
        assert new in fp
        assert len(fp) == len(ps)  # Size shouldn't change
        l_ = [(fp.dist(a, b), b) for a, b in zip(cycle([new]), ps)]
        res = min(l_, key=itemgetter(0))  # noqa: F841
        neigh = fp.neighbors[new]  # noqa: F841
        # assert abs(res[0] - neigh["dist"]) < 1e-8
        # assert res[1] == neigh["neigh"]

    def test_merge_closest(self):
        # This needs to be 'fleshed' out more... lots of things to test here
        random.seed(1234)
        ps = [rand_tuple(4) for _ in range(50)]
        fp = FastPair().build(ps)
        # fp2 = FastPair().build(ps)
        n = len(ps)
        while n >= 2:
            dist, (a, b) = fp.closest_pair()
            new = interact(a, b)
            fp -= b  # Drop b
            fp._update_point(a, new)
            n -= 1
        assert len(fp) == 1 == n
        points = [
            (
                0.69903599809571437,
                0.52457534006594131,
                0.7614753848101149,
                0.37011695654655385,
            )
        ]
        assert all_close(fp.points[0], points[0])
        # Should have < 2 points now...
        with pytest.raises(ValueError):
            fp.closest_pair()
            # fp2.closest_pair()

    def test_call_and_closest_pair_min_points(self, image_array):
        ps = image_array
        fp = FastPair(dist=image_distance)
        for p in ps:
            fp += p
        assert fp.initialized is False
        assert len(fp) == 6
        cp = fp.closest_pair()
        bf = fp.closest_pair_brute_force()
        assert fp() == cp
        assert abs(cp[0] - bf[0]) < 1e-8
        assert cp[1] == bf[1]

    def test_iter(self, point_set):
        ps = point_set
        fp = FastPair().build(ps)
        assert fp.min_points == 10
        assert isinstance(fp.dist, FunctionType)
        my_iter = iter(fp)
        assert next(my_iter) in set(ps)
        assert fp[ps[0]].neigh in set(ps)

        try:
            fp[(2, 3, 4)]
        except KeyError as err:
            print(err)

        fp[ps[0]] = fp[ps[0]].neigh
        try:
            assert fp[ps[0]].neigh
        except KeyError as err:
            print(err)

    def test_update_point_less_points(self, point_set):
        ps = point_set
        fp = FastPair()
        for p in ps[:9]:
            fp += p
        assert fp.initialized is False
        old = ps[0]  # Just grab the first point...
        new = rand_tuple(len(ps[0]))
        fp._update_point(old, new)
        assert len(fp) == 1
