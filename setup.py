#!/usr/bin/env python
#
# Distutils setup script for Munkres
#
# $Id$
# ---------------------------------------------------------------------------

from distutils.core import setup
import re, munkres

VERSION = str(munkres.__version__)
(AUTHOR, EMAIL) = re.match('^(.*),\s*(.*)$', munkres.__author__).groups()
URL = munkres.__url__
LICENSE = munkres.__license__

setup (name="munkres",
       version=VERSION,
       description="munkres algorithm for the Assignment Problem",
       url=URL,
       license=LICENSE,
       author=AUTHOR,
       author_email=EMAIL,
       py_modules=["munkres"])
