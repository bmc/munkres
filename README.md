Munkres implementation for Python
---------------------------------

<a href="https://pypi.python.org/pypi/munkres" target="_blank">
<img src="https://img.shields.io/pypi/v/munkres.png">
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
<http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html>.

[Hungarian algorithm]: http://en.wikipedia.org/wiki/Hungarian_algorithm

See the docs in munkres.py and the [home page][] for more details.

[home page]: http://software.clapper.org/munkres/

## Copyright

&copy; 2008 Brian M. Clapper

## License

Licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE.md) for details.
