"""FastPair: Data-structure for the dynamic closest-pair problem.

Testing module for FastPair.
"""

# Copyright (c) 2016, Carson J. Q. Farmer <carsonfarmer@gmail.com>
# Copyright (c) 2002-2015, David Eppstein
# Licensed under the MIT Licence (http://opensource.org/licenses/MIT).

import re
from itertools import combinations, cycle
from operator import itemgetter
from types import FunctionType

import numpy
import pytest

from fastpair import FastPair


def normalized_distance(_a: numpy.array, _b: numpy.array) -> float:
    """Compute the normalized distance between 2 arrays."""
    b = _b.astype(int)
    a = _a.astype(int)
    norm_diff = numpy.linalg.norm(b - a)
    norm1 = numpy.linalg.norm(b)
    norm2 = numpy.linalg.norm(a)
    return norm_diff / (norm1 + norm2)


def image_distance(image1: tuple, image2: tuple) -> float:
    """Custom distance metric."""
    (sig1, _) = image1
    (sig2, _) = image2
    sig1 = numpy.frombuffer(sig1, numpy.int8)
    sig2 = numpy.frombuffer(sig2, numpy.int8)
    return normalized_distance(sig1, sig2)


def contains_same(s: list, t: list) -> bool:
    """Determine if 2 lists contain the same (set-theoretic) elements."""
    s, t = set(s), set(t)
    return s >= t and s <= t


def interact(u: tuple[float, float], v: tuple[float, float]) -> tuple[float, float]:
    """Compute element-wise mean(s) from two arrays."""
    return tuple(numpy.mean(numpy.array([u, v]), axis=0))


class TestFastPairInit:
    def setup_method(self):
        self.fp = FastPair()

    def test_min_points(self):
        assert self.fp.min_points == 10

    def test_dist_callable(self):
        assert isinstance(self.fp.dist, FunctionType)

    def test_not_initialized(self):
        assert self.fp.initialized is False

    def test_len_points(self):
        assert len(self.fp.points) == 0

    def test_len_neighbors(self):
        assert len(self.fp.neighbors) == 0


class TestFastPairBuild:
    def setup_method(self):
        self.fp = FastPair().build(pytest.point_set())

    def test_len_fp(self):
        assert len(self.fp.neighbors) == len(pytest.point_set())

    def test_len_neighbors(self):
        assert len(self.fp.neighbors) == len(pytest.point_set())

    def test_is_initialized(self):
        assert self.fp.initialized is True


class TestFastPairAdd:
    def setup_method(self):
        self.fp = FastPair()

    def test_add(self):
        for p in pytest.point_set()[:9]:
            self.fp += p

        assert self.fp.initialized is False

        assert len(self.fp) == 9

        for p in pytest.point_set()[9:]:
            self.fp += p

        assert self.fp.initialized is True


class TestFastPairSub:
    def setup_method(self):
        self.ps = pytest.point_set()
        self.fp = FastPair().build(self.ps)

    def test_sub(self):
        start = self.fp._find_neighbor(self.ps[-1])
        self.fp -= self.ps[-1]
        end = self.fp._find_neighbor(start["neigh"])

        assert end["neigh"] != self.ps[-1]

        # This is risky, because it might legitimately be the same...?
        assert start["dist"] != end["dist"]

        assert len(self.fp) == len(self.ps) - 1

        with pytest.raises(
            ValueError, match=re.escape("list.remove(x): x not in list")
        ):
            self.fp -= pytest.rand_tuple(len(self.ps[0]))


class TestFastPairLen:
    def setup_method(self):
        self.ps = pytest.point_set()
        self.fp = FastPair()

    def test_pre_build(self):
        assert len(self.fp) == 0

    def test_post_build(self):
        self.fp.build(self.ps)
        assert len(self.fp) == len(self.ps)


class TestFastPairContains:
    def setup_method(self):
        self.ps = pytest.point_set()
        self.fp = FastPair()

    def test_pre_build(self):
        assert self.ps[0] not in self.fp

    def test_post_build(self):
        self.fp.build(self.ps)
        assert self.ps[0] in self.fp


class TestFastPairGetSet:
    def setup_method(self):
        self.ps = pytest.point_set(n=3, d=2, seed=0)
        self.fp = FastPair().build(self.ps)

    def test_get_set(self):
        known_get = {"dist": 0.6983581029815579, "neigh": (0.5118, 0.9505)}
        observed_get = self.fp[(0.2616, 0.2985)]
        assert known_get == pytest.approx(observed_get)

        known_set = {"dist": 0.6983581029815579, "neigh": (19, 99)}
        self.fp[(0.2616, 0.2985)].neigh = (19, 99)
        observed_set = self.fp[(0.2616, 0.2985)]
        assert known_set == pytest.approx(observed_set)

    def test_get_raise_1(self):
        with pytest.raises(KeyError, match=re.escape("('a', 'b') not found")):
            self.fp[("a", "b")]

    def test_get_raise_2(self):
        with pytest.raises(KeyError, match=re.escape("('a', 'b') not found")):
            self.fp.__getitem__(("a", "b"))

    def test_set_raise_1(self):
        with pytest.raises(KeyError, match=re.escape("('a', 'b') not found")):
            self.fp[("a", "b")].neigh = (19, 99)

    def test_set_raise_2(self):
        with pytest.raises(KeyError, match=re.escape("('a', 'b') not found")):
            self.fp.__setitem__(("a", "b"), {"dist": 1, "neigh": (20, 20)})


class TestFastPairIter:
    def test_basic(self):
        ps = [(1, 1), (2, 2), (3, 3)]
        fp = FastPair().build(ps)

        # see GH#58
        assert fp.min_points == 10
        assert isinstance(fp.dist, FunctionType)

        known_iter = iter(fp)

        assert next(known_iter) in set(ps)
        assert fp[ps[0]].neigh in set(ps)

        with pytest.raises(KeyError, match=re.escape("(2, 3, 4) not found")):
            fp[(2, 3, 4)]

        fp[ps[0]] = fp[ps[0]].neigh
        with pytest.raises(KeyError, match=re.escape("(1, 1) not found")):
            _ = fp[ps[0]].neigh

    def test_complex(self):
        ps = pytest.point_set(n=300, d=10, seed=0)
        fp = FastPair().build(ps)

        # see GH#58
        assert fp.min_points == 10
        assert isinstance(fp.dist, FunctionType)

        known_iter = iter(fp)

        assert next(known_iter) in set(ps)
        assert fp[ps[0]].neigh in set(ps)

        with pytest.raises(KeyError, match=re.escape("(2, 3, 4) not found")):
            fp[(2, 3, 4)]

        fp[ps[0]] = fp[ps[0]].neigh
        with pytest.raises(KeyError, match=re.escape("not found")):
            _ = fp[ps[0]].neigh


class TestFastPairCallAndClosestPair:
    def setup_method(self):
        self.fp = FastPair().build(pytest.point_set(n=11, d=2, seed=0))
        self.cp = self.fp.closest_pair()
        self.bf = self.fp.closest_pair_brute_force()

    def test_call_eq_cp(self):
        assert self.fp() == self.cp

    def test_cp_eq_bf_dist_approx(self):
        assert abs(self.cp[0] - self.bf[0]) < 1e-8

    def test_cp_eq_bf_pair(self):
        assert self.cp[1] == self.bf[1]

    def test_cp_eq_bf_exact(self):
        known = (
            pytest.approx(0.11669811480910905),
            ((0.8702, 0.2868), (0.956, 0.2077)),
        )
        observed_cp = self.cp
        observed_bf = self.bf

        assert known == observed_cp
        assert known == observed_bf


class TestFastPairAllClosestPairs:
    def setup_method(self):
        self.ps = pytest.point_set(n=7, d=3, seed=0)
        self.fp = FastPair().build(self.ps)
        self.cp = self.fp.closest_pair()
        self.bf = self.fp.closest_pair_brute_force()
        # self.dc = self.fp.closest_pair_divide_conquer()  # Maybe different ordering
        _all_closest = [
            (0.6997697978621256, ((0.637, 0.2698, 0.041), (0.5118, 0.9505, 0.1442))),
            (0.859992494153292, ((0.637, 0.2698, 0.041), (0.2616, 0.2985, 0.8142))),
            (0.9397803200748566, ((0.637, 0.2698, 0.041), (0.0856, 0.2368, 0.8013))),
            (1.013221841454279, ((0.637, 0.2698, 0.041), (0.9431, 0.5113, 0.9762))),
            (0.7367062508218591, ((0.637, 0.2698, 0.041), (0.805, 0.8079, 0.5153))),
            (0.3504472856222459, ((0.637, 0.2698, 0.041), (0.5382, 0.3433, 0.3691))),
            (0.9677830542017152, ((0.5118, 0.9505, 0.1442), (0.2616, 0.2985, 0.8142))),
            (1.0596199979237841, ((0.5118, 0.9505, 0.1442), (0.0856, 0.2368, 0.8013))),
            (1.0349590958100712, ((0.5118, 0.9505, 0.1442), (0.9431, 0.5113, 0.9762))),
            (0.4939799692295225, ((0.5118, 0.9505, 0.1442), (0.805, 0.8079, 0.5153))),
            (0.6480500057865904, ((0.5118, 0.9505, 0.1442), (0.5382, 0.3433, 0.3691))),
            (0.18694731878259177, ((0.2616, 0.2985, 0.8142), (0.0856, 0.2368, 0.8013))),
            (0.7320997814505888, ((0.2616, 0.2985, 0.8142), (0.9431, 0.5113, 0.9762))),
            (0.8025665891376242, ((0.2616, 0.2985, 0.8142), (0.805, 0.8079, 0.5153))),
            (0.5259549505423445, ((0.2616, 0.2985, 0.8142), (0.5382, 0.3433, 0.3691))),
            (0.9171949138541928, ((0.0856, 0.2368, 0.8013), (0.9431, 0.5113, 0.9762))),
            (0.9620226452636133, ((0.0856, 0.2368, 0.8013), (0.805, 0.8079, 0.5153))),
            (0.6348116649841905, ((0.0856, 0.2368, 0.8013), (0.5382, 0.3433, 0.3691))),
            (0.5652185241125772, ((0.9431, 0.5113, 0.9762), (0.805, 0.8079, 0.5153))),
            (0.7488246924347515, ((0.9431, 0.5113, 0.9762), (0.5382, 0.3433, 0.3691))),
            (0.5553465944795196, ((0.805, 0.8079, 0.5153), (0.5382, 0.3433, 0.3691))),
        ]
        self.all_closest = [(pytest.approx(i), j) for i, j in _all_closest]

    def test_all_closest(self):
        known = self.all_closest
        observed = [(self.fp.dist(a, b), (a, b)) for a, b in combinations(self.ps, r=2)]
        assert known == observed

    def test_cp_eq_bf_dist_approx(self):
        assert abs(self.cp[0] - self.bf[0]) < 1e-8

    def test_cp_eq_bf_pair(self):
        assert self.cp[1] == self.bf[1]

    # def test_membership(self):
    #    # Ordering may be different, but both should be in there
    #    assert self.dc[1][0] in self.cp[1] and self.dc[1][1] in self.cp[1]

    def test_min(self):
        known = min(
            [(self.fp.dist(a, b), (a, b)) for a, b in combinations(self.ps, r=2)],
            key=itemgetter(0),
        )
        assert abs(self.cp[0] - known[0]) < 1e-8

    # def test_divide_conquer(self):
    #     assert abs(self.dc[0] - self.cp[0]) < 1e-8  # Compare distance

    def test_cp_eq_bf_exact(self):
        known = (
            0.18694731878259177,
            ((0.2616, 0.2985, 0.8142), (0.0856, 0.2368, 0.8013)),
        )
        observed_cp = self.cp
        observed_bf = self.bf

        assert known == observed_cp
        assert known == observed_bf


class TestFastPairFindNeighborAndSDist:
    def setup_method(self):
        self.ps = pytest.point_set(n=3, d=2, seed=0)
        self.fp = FastPair().build(self.ps)
        self.rando = pytest.rand_tuple(dim=len(self.ps[0]), seed=22)
        self.neigh = self.fp._find_neighbor(self.rando)  # Abusing find_neighbor!
        self.dist = self.fp.dist(self.rando, self.neigh["neigh"])

    def test_dist_observed(self):
        assert abs(self.dist - self.neigh["dist"]) < 1e-8

    def test_point_consistency(self):
        assert len(self.fp) == len(self.ps)  # Make sure we didn't add a point...

    def test_min_nearest_1(self):
        known = (0.14423151528012176, (0.2616, 0.2985))
        nearest = [
            (self.fp.dist(a, b), b) for a, b in zip(cycle([self.rando]), self.ps)
        ]
        observed = min(nearest, key=itemgetter(0))

        assert abs(observed[0] - self.neigh["dist"]) < 1e-8
        assert observed[1] == self.neigh["neigh"]
        assert known == observed

    def test_min_nearest_2(self):
        known = (0.14423151528012176, (0.2616, 0.2985))
        observed = min(self.fp.sdist(self.rando), key=itemgetter(0))

        assert abs(observed[0] - self.neigh["dist"]) < 1e-8
        assert observed[1] == self.neigh["neigh"]
        assert known == observed


class TestFastPairCluster:
    def test_basic(self):
        ps = [(1, 1), (2, 2), (3, 3)]
        known_coords = [1.5, 2.25]
        known_dists = [
            pytest.approx(1.4142135623730951),
            pytest.approx(2.121320343559643),
        ]

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

            assert c == g == (known_coords[i], known_coords[i])
            assert dist == d == known_dists[i]

            ps.remove(e)
            ps.remove(f)
            ps.append(g)
            assert contains_same(fp.points, ps)

        assert len(fp.points) == len(ps) == 1

    def test_complex(self):
        ps = pytest.point_set(n=50, d=5, seed=0)

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

        assert dist == d == 0.7905894069674386


class TestFastPairUpdatePoint:
    def test_basic(self):
        ps = [(1, 1), (2, 2), (3, 3)]
        fp = FastPair().build(ps)
        assert len(fp) == len(ps)

        old = ps[0]  # Just grab the first point...
        new = (4.0, 4.0)
        new_neigh = fp._update_point(old, new)

        assert old not in fp
        assert new in fp
        assert new_neigh == {"neigh": (3, 3), "dist": 1.4142135623730951}
        assert len(fp) == len(ps)  # Size shouldn't change

        nearest = [(fp.dist(a, b), b) for a, b in zip(cycle([new]), ps)]
        min_nearest = min(nearest, key=itemgetter(0))
        neigh = fp.neighbors[new]

        assert abs(min_nearest[0] - neigh["dist"]) < 1e-8
        assert min_nearest[1] == neigh["neigh"]

        assert list(fp.neighbors.items()) == (
            [
                ((2, 2), {"dist": numpy.float64(1.4142135623730951), "neigh": (3, 3)}),
                ((3, 3), {"neigh": (3, 3), "dist": numpy.inf}),
                ((4.0, 4.0), {"neigh": (3, 3), "dist": 1.4142135623730951}),
            ]
        )

    def test_intermediate(self):
        ps = pytest.point_set(n=10, d=10, seed=0)
        fp = FastPair().build(ps)
        assert len(fp) == len(ps)

        old = ps[0]  # Just grab the first point...
        new = pytest.rand_tuple(len(ps[0]), seed=10)
        new_neigh = fp._update_point(old, new)

        assert old not in fp
        assert new in fp
        assert new_neigh == {
            "neigh": (
                0.9431,
                0.5113,
                0.9762,
                0.0808,
                0.6074,
                0.3765,
                0.8019,
                0.1745,
                0.8716,
                0.5439,
            ),
            "dist": 1.006573613800799,
        }
        assert len(fp) == len(ps)  # Size shouldn't change

        nearest = [(fp.dist(a, b), b) for a, b in zip(cycle([new]), ps)]
        min_nearest = min(nearest, key=itemgetter(0))
        neigh = fp.neighbors[new]

        assert abs(min_nearest[0] - neigh["dist"]) < 1e-8
        assert min_nearest[1] == neigh["neigh"]

    @pytest.mark.xfail(reason="Still failing sometimes with larger ``(n, d)`` inputs.")
    def test_complex(self):
        ps = pytest.point_set(n=100, d=100, seed=0)
        fp = FastPair().build(ps)
        assert len(fp) == len(ps)

        old = ps[0]  # Just grab the first point...
        new = pytest.rand_tuple(len(ps[0]), seed=10)
        new_neigh = fp._update_point(old, new)

        assert old not in fp
        assert new in fp
        assert new_neigh["dist"] == 3.4977092074670817

        assert len(fp) == len(ps)  # Size shouldn't change

        # Still failing sometimes with larger ``(n, d)`` inputs
        nearest = [(fp.dist(a, b), b) for a, b in zip(cycle([new]), ps)]
        min_nearest = min(nearest, key=itemgetter(0))
        neigh = fp.neighbors[new]

        assert abs(min_nearest[0] - neigh["dist"]) < 1e-8
        assert min_nearest[1] == neigh["neigh"]


class TestFastPairCustomMetric:
    def test_image(self, image_array):
        ps = image_array
        fp = FastPair(dist=image_distance)
        for p in ps:
            fp += p

        assert fp.initialized is False
        assert len(fp) == 6

        cp = fp.closest_pair()
        bf = fp.closest_pair_brute_force()

        assert (
            fp()
            == cp
            == (
                numpy.float64(0.061482537729360145),
                (
                    (b"\x00P\x07`\x00\x00\x03!\x06\x02\x00\x00\x00", "1"),
                    (b"\x00P\x07`\x00\x00\x03!\x06\x02\x00\x00\x10", "5"),
                ),
            )
        )
        assert abs(cp[0] - bf[0]) < 1e-8
        assert cp[1] == bf[1]


class TestFastPairMergeClosest:
    def test_manual_basic(self):
        ps = [tuple([i for j in range(4)]) for i in range(10)]
        known_dists = (2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 6.0, 11.0)

        fp = FastPair().build(ps)
        n = len(ps)
        ix = 0
        while n >= 2:
            dist, (a, b) = fp.closest_pair()
            assert known_dists[ix] == dist

            new = interact(a, b)
            fp -= b  # Drop b
            fp._update_point(a, new)
            ix += 1
            n -= 1

        assert len(fp) == 1 == n

        points = [
            (
                4.25,
                4.25,
                4.25,
                4.25,
            )
        ]
        assert numpy.allclose(fp.points[0], points[0])

        # Should have < 2 points now...
        with pytest.raises(
            ValueError, match=re.escape("Must have `npoints >= 2` to form a pair.")
        ):
            fp.closest_pair()

    def test_manual_complex(self):
        ps = pytest.point_set(n=500, d=4, seed=2020)

        fp = FastPair().build(ps)
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
                numpy.float64(0.29924683837890625),
                numpy.float64(0.3623512680053711),
                numpy.float64(0.7459307929992676),
                numpy.float64(0.5862821228027344),
            )
        ]
        assert numpy.allclose(fp.points[0], points[0])

        # Should have < 2 points now...
        with pytest.raises(
            ValueError, match=re.escape("Must have `npoints >= 2` to form a pair.")
        ):
            fp.closest_pair()


class TestFastPairUpdatePointLessPoints:
    ####################################################################
    #                              See GH#60                           #
    ####################################################################

    def test_basic(self):
        ps = pytest.point_set()
        fp = FastPair()
        for p in ps[:9]:
            fp += p
        assert fp.initialized is False
        old = ps[0]  # Just grab the first point...
        new = pytest.rand_tuple(len(ps[0]))
        fp._update_point(old, new)

        try:
            _len = 1
            _msg = f"length {_len} passed?"
            assert len(fp) == _len, _msg
        except AssertionError as e:
            if str(e).startswith(_msg):
                try:
                    _len = 9
                    _msg = f"length {_len} passed?"
                    assert len(fp) == _len, _msg
                except AssertionError as e:
                    if not str(e).startswith(_msg):
                        raise (e)
            else:
                raise (e)
