---
title: munkres — Munkres implementation for Python
layout: withTOC
---

## Introduction

The Munkres module provides an implementation of the Munkres algorithm
(also called the [Hungarian algorithm][] or the Kuhn-Munkres algorithm).
The algorithm models an assignment problem as an NxM cost matrix, where
each element represents the cost of assigning the ith worker to the jth
job, and it figures out the least-cost solution, choosing a single item
from each row and column in the matrix, such that no row and no column are
used more than once.

[Hungarian algorithm]: http://en.wikipedia.org/wiki/Hungarian_algorithm

## Getting and installing *munkres*

### Installing

Because *munkres* is available via [PyPI][], if you have [pip][]
installed on your system, installing *munkres* is as easy as running this
command:

    pip install munkres

### Installing from source

You can also install *munkres* from source. Either download the source (as
a zip or tarball) from <http://github.com/bmc/munkres/downloads>, or make
a local read-only clone of the [Git repository][] using one of the
following commands:

    $ git clone git://github.com/bmc/munkres.git
    $ git clone http://github.com/bmc/munkres.git

[pip]: https://pip.pypa.io/en/stable
[PyPI]: http://pypi.python.org/pypi
[Git repository]: http://github.com/bmc/munkres

Once you have a local `munkres` source directory, change your working directory
to the source directory, and type:

    python setup.py install

To install it somewhere other than the default location (such as in your
home directory) type:

    python setup.py install --prefix=$HOME

## Documentation

Consult the [API documentation](api/index.html) for details. The API 
documentation is generated from the source code, so you can also just browse
[the source](https://github.com/bmc/munkres/blob/master/munkres.py).

### References

1. <http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html>
2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
   *Naval Research Logistics Quarterly*, 2:83-97, 1955.
3. Harold W. Kuhn. Variants of the Hungarian method for assignment
   problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.
4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
   *Journal of the Society of Industrial and Applied Mathematics*,
   5(1):32-38, March, 1957.
5. <http://en.wikipedia.org/wiki/Hungarian_algorithm>

## License

This module is released under the Apache Software License, version 2.
See the [license][] file for details.

[license]: https://github.com/bmc/munkres/blob/master/LICENSE.md
