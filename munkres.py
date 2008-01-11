"""
INTRODUCTION

The Munkres module provides an implementation of the Munkres algorithm
(also called the Hungarian algorithm or the Kuhn-Munkres algorithm) for
solving the Assignment Problem. 

Assignment Problem

Let C be an nxn matrix representing the costs of each of n workers to
perform any of n jobs. The assignment problem is to assign jobs to workers
in a way that minimizes the total cost. Since each worker can perform only
one job and each job can be assigned to only one worker the assignments
represent an independent set of the matrix C.

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
executes in O(n!) time: Calculating the permutations for an NxN matrix
requires N! operations. For a 12x12 matrix, that's 479,001,600 traversals.
Even if you could manage to perform each traversal in just one millisecond,
it would still take more than 133 hours to perform the entire traversal. A
20x20 matrix would take 2,432,902,008,176,640,000 operations. At an
optimistic millisecond per operation, that's more than 77 million years.

The Munkres algorithm runs in O(n^3) time, rather than O(n!). This package
provides an implementation of that algorithm.

This version is based on
http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html

This version was written for Python by Brian Clapper (bmc@clapper.org),
from the (Ada) algorithm at the above web site. (The Algorithm::Munkres
Perl version, in CPAN, was clearly adapted from the same web site.)

USAGE

Construct a Munkres object:

    from munkres import Munkres

    m = Munkres()

Then use it to compute the lowest cost assignment from a cost matrix. Here's
a sample program:

    from munkres import Munkres

    def print_matrix(msg, matrix):
        print msg
        for row in matrix:
            print row


    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]
    m = Munkres()
    indexes = m.compute(matrix)
    print_matrix('Lowest cost through this matrix:', matrix)
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

CALCULATING PROFIT, RATHER THAN COST

The cost matrix is just that: A cost matrix. The Munkres algorithm finds
the combination of elements (one from each row and column) that results in
the smallest cost. It's also possible to use the algorithm to maximize
profit. To do that, however, you have to convert your profit matrix to a
cost matrix. The simplest way to do that is to subtract all elements from a
large value. For example:

    from munkres import Munkres

    def print_matrix(msg, matrix):
        print msg
        for row in matrix:
            print row


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
    print_matrix('Lowest cost through this matrix:', matrix)
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print '(%d, %d) -> %d' % (row, column, value)

    Print 'total profit=%d' % total

Running that program produces:

    Highest profit through this matrix:
    [5, 9, 1]
    [10, 3, 2]
    [8, 7, 4]
    (0, 1) -> 9
    (1, 0) -> 10
    (2, 2) -> 4
    total profit=23

The Munkres class provides a convenience method for creating a cost matrix
from a profit matrix. Since it doesn't know whether the matrix contains
floating point numbers, decimals, or integers, you have to provide the
conversion function; but the convenience method takes care of the actual
creation of the cost matrix:

    cost_matrix = Munkres.make_cost_matrix(matrix,
                                           lambda cost: sys.maxint - cost)

So, the above profit-calculation program can be recast as:

    from munkres import Munkres

    def print_matrix(msg, matrix):
        print msg
        for row in matrix:
            print row


    matrix = [[5, 9, 1],
                 [10, 3, 2],
                 [8, 7, 4]]
    cost_matrix = Munkres.make_cost_matrix(matrix,
                                           lambda cost: sys.maxint - cost)
    m = Munkres()
    indexes = m.compute(cost_matrix)
    print_matrix('Lowest cost through this matrix:', matrix)
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print '(%d, %d) -> %d' % (row, column, value)
    print 'total profit=%d' % total

REFERENCES

1. http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html

2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
   Naval Research Logistics Quarterly, 2:83-97, 1955.

3. Harold W. Kuhn. Variants of the Hungarian method for assignment
   problems. Naval Research Logistics Quarterly, 3: 253-258, 1956.

4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
   Journal of the Society of Industrial and Applied Mathematics,
   5(1):32-38, March, 1957.

5. http://en.wikipedia.org/wiki/Hungarian_algorithm

COPYRIGHT AND LICENSE

Copyright (c) 2008 Brian M. Clapper

This is free software, released under the following BSD-like license:

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. The end-user documentation included with the redistribution, if any,
   must include the following acknowlegement:

      This product includes software developed by Brian M. Clapper
      (bmc@clapper.org, http://www.clapper.org/bmc/). That software is
      copyright (c) 2008 Brian M. Clapper.

    Alternately, this acknowlegement may appear in the software itself, if
    and wherever such third-party acknowlegements normally appear.

THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESSED OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL BRIAN M. CLAPPER BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

$Id$
"""

import sys

# Info about the module
__version__   = "1.0"
__author__    = "Brian Clapper, bmc <at> clapper <dot> org"
__url__       = "http://www.clapper.org/software/python/munkres/"
__copyright__ = "(c) 2008 Brian M. Clapper"
__license__   = "BSD-style license"

# Package stuff

__all__     = ["Munkres"]

class Munkres:
    """
Calculate the Munkres solution to the classical assignment problem.
See the module documentation for usage.
    """

    def __init__(self):
        """Create a new instance"""
        self.C = None
        self.row_covered = []
        self.col_covered = []
        self.n = 0
        self.Z0_r = 0
        self.Z0_c = 0
        self.marked = None
        self.path = None

    def make_cost_matrix(profit_matrix, inversion_function):
        """
        Create a cost matrix from a profit matrix by calling
        'inversion_function' to invert each value. The inversion
        function must take one numeric argument (of any type) and return
        another numeric argument which is presumed to be the cost inverse
        of the original profit.

        This is a static method. Call it like this:

        cost_matrix = Munkres.make_cost_matrix(matrix, inversion_func)

        For example:

        cost_matrix = Munkres.make_cost_matrix(matrix,
                                               lambda x : sys.maxint - x
        """
        cost_matrix = []
        for row in profit_matrix:
            cost_row = []
            for value in row:
                cost_row += [inversion_function(value)]
            cost_matrix += [cost_row]
        return cost_matrix

    make_cost_matrix = staticmethod(make_cost_matrix)

    def compute(self, cost_matrix):
        """
        Compute the indexes for the lowest-cost pairings between rows and
        columns in the database. Returns a list of (row, column) tuples
        that can be used to traverse the matrix.

        The matrix must be square.
        """
        self.C = self.__copy_matrix(cost_matrix)
        self.n = len(cost_matrix)
        self.row_covered = [False for i in range(self.n)]
        self.col_covered = [False for i in range(self.n)]
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = self.__make_matrix(self.n, 0)
        self.marked = self.__make_matrix(self.n, 0)

        done = False
        step = 1

        steps = { 1 : self.__step1,
                  2 : self.__step2,
                  3 : self.__step3,
                  4 : self.__step4,
                  5 : self.__step5,
                  6 : self.__step6 }

        while not done:
            try:
                func = steps[step]
                #print 'calling ' + str(func)
                step = func()
            except KeyError:
                done = True

        # Look for the starred columns
        results = []
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 1:
                    results += [(i, j)]
        assert(len(results) == self.n)

        return results

    def __copy_matrix(self, matrix):
        """Return an exact copy of the supplied matrix"""
        copy = []
        for row in matrix:
            new_row = []
            for item in row:
                new_row += [item]
            copy += [new_row]
        return copy

    def __make_matrix(self, n, val):
        """Create an NxN matrix, populating it with the specific value."""
        matrix = []
        for i in range(n):
            matrix += [[val for j in range(n)]]
        return matrix

    def __step1(self):
        """
        For each row of the matrix, find the smallest element and
        subtract it from every element in its row. Go to Step 2.
        """
        C = self.C
        n = self.n
        for i in range(n):
            minval = self.C[i][0]
            # Find the minimum value for this row
            for j in range(1, n):
                if minval > self.C[i][j]:
                    minval = self.C[i][j]

            # Subtract that minimum from every element in the row.
            for j in range(n):
                self.C[i][j] -= minval

        return 2

    def __step2(self):
        """
        Find a zero (Z) in the resulting matrix. If there is no starred
        zero in its row or column, star Z. Repeat for each element in the
        matrix. Go to Step 3.
        """
        n = self.n
        for i in range(n):
            for j in range(n):
                if (self.C[i][j] == 0) and \
                   (not self.col_covered[j]) and \
                   (not self.row_covered[i]):
                    self.marked[i][j] = 1
                    self.col_covered[j] = True
                    self.row_covered[i] = True

        self.__clear_covers()
        return 3

    def __step3(self):
        """
        Cover each column containing a starred zero. If K columns are
        covered, the starred zeros describe a complete set of unique
        assignments. In this case, Go to DONE, otherwise, Go to Step 4.
        """
        n = self.n
        count = 0
        for i in range(n):
            for j in range(n):
                if self.marked[i][j] == 1:
                    self.col_covered[j] = True
                    count += 1

        if count >= n:
            step = 7 # done
        else:
            step = 4

        return step

    def __step4(self):
        """
        Find a noncovered zero and prime it. If there is no starred zero
        in the row containing this primed zero, Go to Step 5. Otherwise,
        cover this row and uncover the column containing the starred
        zero. Continue in this manner until there are no uncovered zeros
        left. Save the smallest uncovered value and Go to Step 6.
        """
        step = 0
        done = False
        row = -1
        col = -1
        star_col = -1
        while not done:
            (row, col) = self.__find_a_zero()
            if row < 0:
                done = True
                step = 6
            else:
                self.marked[row][col] = 2
                star_col = self.__find_star_in_row(row)
                if star_col >= 0:
                    col = star_col
                    self.row_covered[row] = True
                    self.col_covered[col] = False
                else:
                    done = True
                    self.Z0_r = row
                    self.Z0_c = col
                    step = 5

        return step

    def __step5(self):
        """
        Construct a series of alternating primed and starred zeros as
        follows. Let Z0 represent the uncovered primed zero found in Step
        4. Let Z1 denote the starred zero in the column of Z0 (if any).
        Let Z2 denote the primed zero in the row of Z1 (there will always
        be one). Continue until the series terminates at a primed zero
        that has no starred zero in its column. Unstar each starred zero
        of the series, star each primed zero of the series, erase all
        primes and uncover every line in the matrix. Return to Step 3
        """
        count = 0
        path = self.path
        path[count][0] = self.Z0_r
        path[count][1] = self.Z0_c
        done = False
        while not done:
            row = self.__find_star_in_col(path[count][1])
            if row >= 0:
                count += 1
                path[count][0] = row
                path[count][1] = path[count-1][1]
            else:
                done = True

            if not done:
                col = self.__find_prime_in_row(path[count][0])
                count += 1
                path[count][0] = path[count-1][0]
                path[count][1] = col

        self.__convert_path(path, count)
        self.__clear_covers()
        self.__erase_primes()
        return 3

    def __step6(self):
        """
        Add the value found in Step 4 to every element of each covered
        row, and subtract it from every element of each uncovered column.
        Return to Step 4 without altering any stars, primes, or covered
        lines.
        """
        min = self.__find_smallest()
        for i in range(self.n):
            for j in range(self.n):
                if self.row_covered[i]:
                    self.C[i][j] += min
                if not self.col_covered[j]:
                    self.C[i][j] -= min
        return 4

    def __find_smallest(self):
        """Find the smallest uncovered value in the matrix."""
        minval = sys.maxint
        for i in range(self.n):
            for j in range(self.n):
                if (not self.row_covered[i]) and (not self.col_covered[j]):
                    if minval > self.C[i][j]:
                        minval = self.C[i][j]
        return minval

    def __find_a_zero(self):
        """Find the first uncovered element with value 0"""
        row = -1
        col = -1
        i = 0
        n = self.n
        done = False

        while not done:
            j = 0
            while True:
                if (self.C[i][j] == 0) and \
                   (not self.row_covered[i]) and \
                   (not self.col_covered[j]):
                    row = i
                    col = j
                    done = True
                j += 1
                if j >= n:
                    break
            i += 1
            if i >= n:
                done = True

        return (row, col)

    def __find_star_in_row(self, row):
        """
        Find the first starred element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 1:
                col = j
                break

        return col

    def __find_star_in_col(self, col):
        """
        Find the first starred element in the specified row. Returns
        the row index, or -1 if no starred element was found.
        """
        row = -1
        for i in range(self.n):
            if self.marked[i][col] == 1:
                row = i
                break

        return row

    def __find_prime_in_row(self, row):
        """
        Find the first prime element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 2:
                col = j
                break

        return col

    def __convert_path(self, path, count):
        for i in range(count+1):
            if self.marked[path[i][0]][path[i][1]] == 1:
                self.marked[path[i][0]][path[i][1]] = 0
            else:
                self.marked[path[i][0]][path[i][1]] = 1

    def __clear_covers(self):
        """Clear all covered matrix cells"""
        for i in range(self.n):
            self.row_covered[i] = False
            self.col_covered[i] = False

    def __erase_primes(self):
        """Erase all prime markings"""
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 2:
                    self.marked[i][j] = 0



if __name__ == '__main__':
    cost = [[999600, 999850, 999600],
            [999600, 999550, 999400],
            [999700, 999775, 999700]]
    m = Munkres()
    indexes = m.compute(cost)
    total_ctr = 0
    for r, c in indexes:
        x = cost[r][c]
        ctr = 1000000 - x
        total_ctr += ctr
        print '(%d, %d) -> %d (ctr=%d)' % (r, c, x, ctr)

    print 'total CTR=%d' % total_ctr
