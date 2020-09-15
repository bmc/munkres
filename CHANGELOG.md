# Change Log, munkres.py

Version 1.1.4 (September, 2020)

- Switched from Nose to Pytest for testing. Patch courtesy @kmosiejczuk,
  [PR #32](https://github.com/bmc/munkres/pull/32), with some additional
  cleanup by me.
- Fix to [Issue #34](https://github.com/bmc/munkres/issues/34), in which
  `print_matrix` wasn't handling non-integral values. Patch courtesy @finn0,
  via [PR #35](https://github.com/bmc/munkres/pull/35).
- Various changes from `http:` URLs to `https:` URLs, courtesy @finn0
  via [PR #36](https://github.com/bmc/munkres/pull/36/).

Version 1.1.3:

**Nonexistent**. Accidentally published before check-in. Deleted from
PyPI. Use version 1.1.4.

Version 1.1.2 (February, 2019)

- Removed `NoReturn` type annotations, to allow compatibility with Python 3.5 
  releases prior to 3.5.4. Thanks to @jackwilsdon for catching that issue.

Version 1.1.1 (February, 2019)

- Version bump to get past a PyPI publishing issue. (Can't republish
  partially published 1.1.0.)

Version 1.1.0 (February, 2019)

- Only supports Python 3.5 or better, from this version forward (since Python
  2 is at end of life in 11 months).
- Added `typing` type hints.
- Updated docs to use `pdoc`, since `epydoc` is pretty much dead.

Version 1.0.12 (June, 2017)

- Merged [Pull Request #11](https://github.com/bmc/munkres/pull/11), from
  [@brunokim](https://github.com/brunokim), which simplifies conversion of a 
  profit matrix to a cost matrix, in the default case.
  
- Merged [Pull Request #7](https://github.com/bmc/munkres/pull/7), from
  [@mdxs](https://github.com/mdxs), which fixes a message.
  
- Added more tests.

Version 1.0.11 (June, 2017)

- Docs are now generated with [pdoc](https://github.com/BurntSushi/pdoc).

- Merged [Pull Request 24](https://github.com/bmc/munkres/pull/24), from
  [@czlee](https://github.com/czlee): 
    - Change to step 4: When it looks for a uncovered zero, rather than starting 
      at row 0, column 0, it starts where it left off, i.e. at the last 
      uncovered zero it found. Since it doesn't start at (0,0), when it gets to 
      the last column it now loops around to the first, and exits unsuccessfully 
      if it got back to where it started. This change reduces this reduces the 
      solving time for (certain) large matrices. For instance, in tests, 
      solving a matrix of size 394×394  goes from about 2 minutes to about 4 
      seconds.
    - Since Python 3 started cracking down on unnatural comparisons, the 
      `DISALLOWED` constant added in 
      [Pull Request 19](https://github.com/bmc/munkres/issues/19) no longer 
      works. (It raises a TypeError for unorderable types, as is expected in 
      Python 3.) Since this constant is meant to act like infinity, this 
      modification just changes the two lines where it would otherwise try to 
      make an illegal (in Python 3) comparison between a number and 
      `DISALLOWED_OBJ()` and gets it to behave as if `DISALLOWED` is always 
      larger.

- Added Travis CI integration.

- Added some unit tests. See `tests` and `tests/README.md`.

Version 1.0.10 (May, 2017)

- Updated `setup.py` to produce a wheel.

Version 1.0.9 (Jan, 2017)

- Fixed URL to original implementation. Addresses
  [Issue #4](https://github.com/bmc/munkres/issues/4).
- Fixes from [@kylemcdonald](https://github.com/kylemcdonald):
    - `print_matrix()` no longer crashes on 0. Fixes
      [Issue #1](https://github.com/bmc/munkres/issues/4).
    - Fixed bug where step 3 could quit early. Fixes 
      [Issue #16](https://github.com/bmc/munkres/issues/16).
    - Added step 2 break for a small optimization.
    - Added time bound to README. Addresses 
      [Issue #15](https://github.com/bmc/munkres/issues/15).
- Versioning will now adhere to
  [semantic version specification](https://semver.org).

Version 1.0.8 (June, 2016)

- License is now ASL.

Version 1.0.7 (December, 2014)

Fix from Stephan Porz (s.porz /at/ uni-bonn.de):
- Fix pad_matrix: pad_value now actually used

Version 1.0.6 (December, 2013)

Fixes from Markus Padourek (markus.padourek /at/ gmail.com):
- sys.maxsize fix and bump to 1.0.6
- Updated to Python 3.x 

Version 1.0.5.3 (2 August, 2009)

- Fixed documentation of print_matrix() in module docs.

Version 1.0.5.2 (30 June, 2008):

- Incorporated some suggestions optimizations from Mark Summerfield
  (mark /at/ qtrac.eu)
- Munkres.make_cost_matrix() is now deprecated, in favor of a module-level
  function.
- The module now provides a print_matrix() convenience function.
- Fixed some bugs related to the padding of non-square matrics.

Version 1.0.5.1 (26 June, 2008)

- Some minor doc changes.

Version 1.0.5 (26 June, 2008)

- Now handles non-square cost matrices by padding them with zeros.
- Converted Epydocs to use reStructuredText.

Version 1.0.4 (13 June, 2008)

- Minor bug fix in main (tester) program in munkres.py

Version 1.0.3 (16 March, 2008)

- Minor change to prevent shadowing of built-in min() function. Thanks to
  Nelson Castillo (nelson /at/ emqbit.com) for pointing it out.

Version 1.0.2 (21 February, 2008)

- Fixed an overindexing bug reported by Chris Willmore (willmc <at> rpi.edu)

Version 1.0.1 (16 February, 2008)

- Documentation now processed by Epydoc.

Version 1.0 (January, 2008)

- Initial release.
