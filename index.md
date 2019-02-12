---
title: munkres — Munkres implementation for Python
layout: withTOC
---

## Introduction

### Assignment Problem

Let *C* be an *n* by *n* matrix representing the costs of each of *n* workers
to perform any of *n* jobs. The assignment problem is to assign jobs to
workers in a way that minimizes the total cost. Since each worker can perform
only one job and each job can be assigned to only one worker the assignments
represent an independent set of the matrix *C*.

One way to generate the optimal set is to create all permutations of
the indexes necessary to traverse the matrix so that no row and column
are used more than once. For instance, given this matrix (expressed in
Python):

```python
matrix = [[5, 9, 1],
          [10, 3, 2],
          [8, 7, 4]]
```

You could use this code to generate the traversal indexes:

```python
def permute(a, results):
    if len(a) == 1:
        results.insert(len(results), a)
    else:
        for i in range(0, len(a)):
            element = a[i]
            a_copy = [a[j] for j in range(0, len(a)) if j != i]
            subresults = []
            permute(a_copy, subresults)
            for subresult in subresults:
                result = [element] + subresult
                results.insert(len(results), result)

results = []
permute(range(len(matrix)), results) # [0, 1, 2] for a 3x3 matrix
```

After the call to permute(), the results matrix would look like this:

```python
[[0, 1, 2],
 [0, 2, 1],
 [1, 0, 2],
 [1, 2, 0],
 [2, 0, 1],
 [2, 1, 0]]
 ```

You could then use that index matrix to loop over the original cost matrix
and calculate the smallest cost of the combinations:

```python
minval = sys.maxsize
for indexes in results:
    cost = 0
    for row, col in enumerate(indexes):
        cost += matrix[row][col]
    minval = min(cost, minval)

print(minval)
```

While this approach works fine for small matrices, it does not scale. It
executes in O(*n*!) time: Calculating the permutations for an *n* x *n*
matrix requires *n*! operations. For a 12x12 matrix, that's 479,001,600
traversals. Even if you could manage to perform each traversal in just one
millisecond, it would still take more than 133 hours to perform the entire
traversal. A 20x20 matrix would take 2,432,902,008,176,640,000 operations. At
an optimistic millisecond per operation, that's more than 77 million years.

The Munkres algorithm runs in O(*n*^3) time, rather than O(*n*!). This
package provides an implementation of that algorithm.

This version is based on
<http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html>

This version was written for Python by Brian Clapper from the algorithm
at the above web site. (The `Algorithm:Munkres` Perl version, in CPAN, was
clearly adapted from the same web site.)

### Usage

Construct a Munkres object:

```python
from munkres import Munkres

m = Munkres()
```

Then use it to compute the lowest cost assignment from a cost matrix. Here's
a sample program:

```python
from munkres import Munkres, print_matrix

matrix = [[5, 9, 1],
          [10, 3, 2],
          [8, 7, 4]]
m = Munkres()
indexes = m.compute(matrix)
print_matrix(matrix, msg='Lowest cost through this matrix:')
total = 0
for row, column in indexes:
    value = matrix[row][column]
    total += value
    print(f'({row}, {column}) -> {value}')
print(f'total cost: {total}')'
```

Running that program produces:

```
Lowest cost through this matrix:
[5, 9, 1]
[10, 3, 2]
[8, 7, 4]
(0, 0) -> 5
(1, 1) -> 3
(2, 2) -> 4
total cost=12
```

The instantiated Munkres object can be used multiple times on different
matrices.

### Non-square Cost Matrices

The Munkres algorithm assumes that the cost matrix is square. However, it's
possible to use a rectangular matrix if you first pad it with 0 values to make
it square. This module automatically pads rectangular cost matrices to make
them square.

Notes:

- The module operates on a *copy* of the caller's matrix, so any padding will
  not be seen by the caller.
- The cost matrix must be rectangular or square. An irregular matrix will
  *not* work.

### Calculating Profit, Rather than Cost

The cost matrix is just that: A cost matrix. The Munkres algorithm finds
the combination of elements (one from each row and column) that results in
the smallest cost. It's also possible to use the algorithm to maximize
profit. To do that, however, you have to convert your profit matrix to a
cost matrix. The simplest way to do that is to subtract all elements from a
large value. For example:

```python
from munkres import Munkres, print_matrix

matrix = [[5, 9, 1],
          [10, 3, 2],
          [8, 7, 4]]
cost_matrix = []
for row in matrix:
    cost_row = []
    for col in row:
        cost_row += [sys.maxsize - col]
    cost_matrix += [cost_row]

m = Munkres()
indexes = m.compute(cost_matrix)
print_matrix(matrix, msg='Highest profit through this matrix:')
total = 0
for row, column in indexes:
    value = matrix[row][column]
    total += value
    print(f'({row}, {column}) -> {value}')

print(f'total profit={total}')'
```

Running that program produces:

```
Highest profit through this matrix:
[5, 9, 1]
[10, 3, 2]
[8, 7, 4]
(0, 1) -> 9
(1, 0) -> 10
(2, 2) -> 4
total profit=23
```

The `munkres` module provides a convenience method for creating a cost
matrix from a profit matrix. By default, it calculates the maximum profit
and subtracts every profit from it to obtain a cost. If, however, you
need a more general function, you can provide the
conversion function; but the convenience method takes care of the actual
creation of the matrix:

```python
import munkres
import math

cost_matrix = munkres.make_cost_matrix(
    matrix,
    lambda profit: 1000.0 - math.sqrt(profit)
)
```

So, the above profit-calculation program can be recast as:

```python
from munkres import Munkres, print_matrix, make_cost_matrix

matrix = [[5, 9, 1],
          [10, 3, 2],
          [8, 7, 4]]
cost_matrix = make_cost_matrix(matrix)
# cost_matrix == [[5, 1, 9],
#                 [0, 7, 8],
#                 [2, 3, 6]]
m = Munkres()
indexes = m.compute(cost_matrix)
print_matrix(matrix, msg='Highest profits through this matrix:')
total = 0
for row, column in indexes:
    value = matrix[row][column]
    total += value
    print(f'(${row}, ${column}) -> ${total}')
print(f'total profit=${total}')
```

### Disallowed Assignments

You can also mark assignments in your cost or profit matrix as disallowed.
Simply use the munkres.DISALLOWED constant.

```python
from munkres import Munkres, print_matrix, make_cost_matrix, DISALLOWED

matrix = [[5, 9, DISALLOWED],
          [10, DISALLOWED, 2],
          [8, 7, 4]]
cost_matrix = make_cost_matrix(matrix, lambda cost: (sys.maxsize - cost) if
                                      (cost != DISALLOWED) else DISALLOWED)
m = Munkres()
indexes = m.compute(cost_matrix)
print_matrix(matrix, msg='Highest profit through this matrix:')
total = 0
for row, column in indexes:
    value = matrix[row][column]
    total += value
    print(f'({row}, {column}) -> {value}')
print(f'total profit={total}')
```

Running this program produces:

```
Lowest cost through this matrix:
[ 5,  9,  D]
[10,  D,  2]
[ 8,  7,  4]
(0, 1) -> 9
(1, 0) -> 10
(2, 2) -> 4
total profit=23
```

### References

1. <http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html>

2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
   *Naval Research Logistics Quarterly*, 2:83-97, 1955.

3. Harold W. Kuhn. Variants of the Hungarian method for assignment
   problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.

4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
   *Journal of the Society of Industrial and Applied Mathematics*,
   5(1):32-38, March, 1957.

5. <http://en.wikipedia.org/wiki/Hungarian_algorithm>


## Getting and installing *munkres*

### Installing

Because *munkres* is available via [PyPI][], if you have [pip][]
installed on your system, installing *munkres* is as easy as running this
command:

```
pip install munkres
```

**WARNING:** As of version 1.1.0, *munkres* no longer supports Python 2.
If you need to use it with Python 2, install an earlier version (e.g., 1.0.12):

```
pip install munkres==1.0.12
```

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
