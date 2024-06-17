# FastPair

Data-structure for the dynamic closest-pair problem.

## Overview

![tag](https://img.shields.io/github/v/release/carsonfarmer/fastpair?include_prereleases&sort=semver)

This project is an implementation of the FastPair dynamic closest-pair data-structure described in David Eppstein's [Fast Hierarchical Clustering and Other Applications of Dynamic Closest Pairs](http://dl.acm.org/citation.cfm?id=351829).
The data-structure is based on the observation that the [conga line data-structure](https://www.ics.uci.edu/~eppstein/projects/pairs/Methods/), in practice, does better the more subsets you give to it: even though the worst case time for $k$ subsets is $O(nk\log{(n/k)})$, that worst case seems much harder to reach than the nearest neighbor algorithm.

In the limit of arbitrarily many subsets, each new addition or point moved by a deletion will be in a singleton subset, and the algorithm will differ from nearest neighbors in only a couple of ways:

1. When we create the initial data structure, we use a conga line rather than all nearest neighbors, to keep the in-degree of each point low, and
2. When we insert a point, we don't bother updating other points' neighbors.

<table>
  <tr>
    <td align="center" colspan="2"><b>Complexity</b></td>
  </tr>
  <tr>
    <td><i>Total space</i></td>
    <td>$20n$ bytes (could be reduced to $4n$ at some cost in update time)</td>
  </tr>
 <tr>
    <td><i>Time per insertion or single distance update</i></td>
    <td>$O(n)$ </td>
  </tr>
 <tr>
    <td><i>Time per deletion or point update</i></td>
    <td>$O(n)$ expected, $O(n^2)$ worst case</td>
  </tr>
 <tr>
    <td><i>Time per closest pair</i></td>
    <td>$O(n)$</td>
  </tr>
</table>

This `Python` version of the algorithm combines ideas and code from the [closest-pair data structure testbed (C++)](https://www.ics.uci.edu/~eppstein/projects/pairs/Source/testbed/) developed around a [series of papers](https://www.ics.uci.edu/~eppstein/projects/pairs/Papers/) by Eppstein *et al.*

## Installation

`FastPairs` has not yet been uploaded to [PyPi](https://pypi.python.org/pypi), as we are currently at the 'pre-release' stage\*. Having said that you should be able to install it via `pip` directly from the GitHub repository with:

```bash
pip install git+git://github.com/carsonfarmer/fastpair.git
```

You can also install `FastPair` by cloning the [GitHub repository](https://github.com/carsonfarmer/fastpair) and using the setup script:

```bash
git clone https://github.com/carsonfarmer/fastpair.git
cd fastpair
pip install .
```

\* *This means the API is not set, and subject to crazy changes at any time!*

## Testing

[![Continuous Integration](https://github.com/carsonfarmer/fastpair/actions/workflows/testing.yml/badge.svg)](https://github.com/carsonfarmer/fastpair/actions/workflows/testing.yml)
[![codecov](https://codecov.io/gh/carsonfarmer/fastpair/branch/main/graph/badge.svg)](https://codecov.io/gh/carsonfarmer/fastpair)

`FastPair` comes with a <del>comprehensive</del> preliminary range of tests. To run tests, install as an editable, development package:

```bash
pip install -e .[tests]
```

This will install `fastpair` itself, its functional dependencies, and the testing/development dependencies. Tests can be run with [`pytest`](http://pytest.org/latest/) as follows:

```bash
pytest -v fastpair --cov fastpair
```

Currently `fastpair` is tested against Python 3.{10,11,12}.

## Utilizing `FastPair`

This notebooks linked below are designed as interactive, minimum tutorials in working with `fastpair` and require additional dependencies, which can be installed with:

```bash
pip install -e .[tests,notebooks]
```

* [`basics_usage.iypnb`](https://github.com/carsonfarmer/fastpair/notebooks/basics_usage.iypnb): Understanding the `fastpair` functionality and data structure
* [`n-dimensional_pointsets`](https://github.com/carsonfarmer/fastpair/notebooks/n-dimensional_pointsets.iypnb): Querying point clouds

## License

Copyright © 2016, [Carson J. Q. Farmer](http://carsonfarmer.com/)  
Copyright © 2002-2015, [David Eppstein](https://www.ics.uci.edu/~eppstein/)  
Licensed under the [MIT License](http://opensource.org/licenses/MIT).  
