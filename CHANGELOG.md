# Change Log, munkres.py

Version 1.0.11 (June, 2017)

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
  [semantic version specification](http://semver.org).

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
