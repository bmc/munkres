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

### Installing via EasyInstall

Because *munkres* is available via [PyPI][], if you have [EasyInstall][]
installed on your system, installing *munkres* is as easy as running this
command (usually as `root` or the system administrator):

    easy_install munkres

### Installing from source

You can also install *munkres* from source. Either download the source (as
a zip or tarball) from <http://github.com/bmc/munkres/downloads>, or make
a local read-only clone of the [Git repository][] using one of the
following commands:

    $ git clone git://github.com/bmc/munkres.git
    $ git clone http://github.com/bmc/munkres.git

[EasyInstall]: http://peak.telecommunity.com/DevCenter/EasyInstall
[PyPI]: http://pypi.python.org/pypi
[Git repository]: http://github.com/bmc/munkres

Once you have a local `munkres` source directory, change your working directory
to the source directory, and type:

    python setup.py install

To install it somewhere other than the default location (such as in your
home directory) type:

    python setup.py install --prefix=$HOME

## The Assignment Problem

These next few sections describe the purpose of the *munkres* module in more
detail.

Let *C* be an *n*x*n* matrix representing the costs of each of *n* workers
to perform any of *n* jobs. The assignment problem is to assign jobs to
workers in a way that minimizes the total cost. Since each worker can perform
only one job and each job can be assigned to only one worker the assignments
represent an independent set of the matrix *C*.

One way to generate the optimal set is to create all permutations of
the indexes necessary to traverse the matrix so that no row and column
are used more than once. For instance, given this matrix (expressed in
Python):

    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]

You could use this code to generate the traversal indexes:

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

After the call to permute(), the results matrix would look like this:

    [[0, 1, 2],
     [0, 2, 1],
     [1, 0, 2],
     [1, 2, 0],
     [2, 0, 1],
     [2, 1, 0]]

You could then use that index matrix to loop over the original cost matrix
and calculate the smallest cost of the combinations:

    n = len(matrix)
    minval = sys.maxint
    for row in range(n):
        cost = 0
        for col in range(n):
            cost += matrix[row][col]
        minval = min(cost, minval)

    print minval

While this approach works fine for small matrices, it does not scale. It
executes in O(*n*!) time: Calculating the permutations for an *n*x*n*
matrix requires *n*! operations. For a 12x12 matrix, that's 479,001,600
traversals. Even if you could manage to perform each traversal in just one
millisecond, it would still take more than 133 hours to perform the entire
traversal. A 20x20 matrix would take 2,432,902,008,176,640,000 operations. At
an optimistic millisecond per operation, that's more than 77 million years.

The Munkres algorithm runs in O(*n*^3) time, rather than O(*n*!). This
package provides an implementation of that algorithm, based on the
algorithm outlined at
<http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html>.

This version was written for Python by Brian Clapper from the (Ada) algorithm
at the above web site. (The ``Algorithm::Munkres`` Perl version, in CPAN, was
clearly adapted from the same web site.)

### Usage

Construct a Munkres object:

    from munkres import Munkres

    m = Munkres()

Then use it to compute the lowest cost assignment from a cost matrix. Here's
a sample program:

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
        print '(%d, %d) -> %d' % (row, column, value)
    print 'total cost: %d' % total

Running that program produces:

    Lowest cost through this matrix:
    [5, 9, 1]
    [10, 3, 2]
    [8, 7, 4]
    (0, 0) -> 5
    (1, 1) -> 3
    (2, 2) -> 4
    total cost=12

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

The cost matrix is just that: a cost matrix. The Munkres algorithm finds
the combination of elements (one from each row and column) that results in
the smallest cost. It's also possible to use the algorithm to maximize
profit. To do that, however, you have to convert your profit matrix to a
cost matrix. The simplest way to do that is to subtract all elements from a
large value. For example:

    from munkres import Munkres, print_matrix

    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]
    cost_matrix = []
    for row in matrix:
        cost_row = []
        for col in row:
            cost_row += [sys.maxint - col]
        cost_matrix += [cost_row]

    m = Munkres()
    indexes = m.compute(cost_matrix)
    print_matrix(matrix, msg='Lowest cost through this matrix:')
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print '(%d, %d) -> %d' % (row, column, value)

    print 'total profit=%d' % total

Running that program produces:

    Highest profit through this matrix:
    [5, 9, 1]
    [10, 3, 2]
    [8, 7, 4]
    (0, 1) -> 9
    (1, 0) -> 10
    (2, 2) -> 4
    total profit=23

The ``munkres`` module provides a convenience method for creating a cost
matrix from a profit matrix. Since it doesn't know whether the matrix contains
floating point numbers, decimals, or integers, you have to provide the
conversion function; but the convenience method takes care of the actual
creation of the cost matrix:

    import munkres

    cost_matrix = munkres.make_cost_matrix(matrix,
                                           lambda cost: sys.maxint - cost)

So, the above profit-calculation program can be recast as:

    from munkres import Munkres, print_matrix, make_cost_matrix

    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]
    cost_matrix = make_cost_matrix(matrix, lambda cost: sys.maxint - cost)
    m = Munkres()
    indexes = m.compute(cost_matrix)
    print_matrix(matrix, msg='Lowest cost through this matrix:')
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print '(%d, %d) -> %d' % (row, column, value)
    print 'total profit=%d' % total

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

This module is released under a BSD license. See the accompanying
[license][] file.

[license]: license.html
