#!/usr/bin/env python
#
# Distutils setup script for Munkres
# ---------------------------------------------------------------------------

from setuptools import setup
import re
import os
import sys
import imp

# Load the data.

here = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path = [here] + sys.path
mf = os.path.join(here, 'munkres.py')
munkres = imp.load_module('munkres', open(mf), mf,
                          ('__init__.py', 'r', imp.PY_SOURCE))
long_description = munkres.__doc__
version = str(munkres.__version__)
(author, email) = re.match('^(.*),\s*(.*)$', munkres.__author__).groups()
url = munkres.__url__
license = munkres.__license__

# Run setup

setup(
    name="munkres",
    version=version,
    description="munkres algorithm for the Assignment Problem",
    long_description=long_description,
    url=url,
    license=license,
    author=author,
    author_email=email,
    py_modules=["munkres"],
    classifiers = [
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Mathematics', 
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
