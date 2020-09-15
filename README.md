Munkres implementation for Python
---------------------------------

<a href="https://pypi.python.org/pypi/munkres" target="_blank">
<img src="https://img.shields.io/pypi/v/munkres.png">
</a>
<a href="https://travis-ci.org/bmc/munkres">
<img src="https://travis-ci.org/bmc/munkres.svg?branch=master"/>
</a>

## Introduction

The Munkres module provides an O(n^3) implementation of the Munkres algorithm
(also called the [Hungarian algorithm][] or the Kuhn-Munkres algorithm).
The algorithm models an assignment problem as an NxM cost matrix, where
each element represents the cost of assigning the ith worker to the jth
job, and it figures out the least-cost solution, choosing a single item
from each row and column in the matrix, such that no row and no column are
used more than once.

This particular implementation is based on 
<https://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html>.

[Hungarian algorithm]: https://en.wikipedia.org/wiki/Hungarian_algorithm

See the docs on the [project page][] for more details.

**WARNING**: As of version 1.1.0, _munkres_ no longer supports Python 2.
If you need to use this package with Python 2, install an earlier version.
See [the installation instructions](http://software.clapper.org/munkres/#installing) 
for details.

[project page]: https://software.clapper.org/munkres/

## Copyright

&copy; 2008-2019 Brian M. Clapper

## License

Licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE.md) for details.
