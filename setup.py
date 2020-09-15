#!/usr/bin/env python
#
# Distutils setup script for Munkres
# ---------------------------------------------------------------------------

from setuptools import setup
import re
import os
import sys
from distutils.cmd import Command
from abc import abstractmethod

if sys.version_info[0:2] < (3, 5):
    columns = int(os.environ.get('COLUMNS', '80')) - 1
    msg = ('As of version 1.1.0, this munkres package no longer supports ' +
           'Python 2. Either upgrade to Python 3.5 or better, or use an ' +
           'older version of munkres (e.g., 1.0.12).')
    sys.stderr.write(msg + '\n')
    raise Exception(msg)

# Load the module.

here = os.path.dirname(os.path.abspath(sys.argv[0]))

def import_from_file(file, name):
    # See https://stackoverflow.com/a/19011259/53495
    import importlib.machinery
    import importlib.util
    loader = importlib.machinery.SourceFileLoader(name, file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod

mf = os.path.join(here, 'munkres.py')
munkres = import_from_file(mf, 'munkres')
long_description = munkres.__doc__
version = str(munkres.__version__)
(author, email) = re.match('^(.*),\s*(.*)$', munkres.__author__).groups()
url = munkres.__url__
license = munkres.__license__

API_DOCS_BUILD = 'apidocs'

class CommandHelper(Command):
    user_options = []

    def __init__(self, dist):
        Command.__init__(self, dist)

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    @abstractmethod
    def run(self):
        pass

class Doc(CommandHelper):
    description = 'create the API docs'

    def run(self):
        os.environ['PYTHONPATH'] = '.'
        cmd = 'pdoc --html --html-dir {} --overwrite --html-no-source munkres'.format(
            API_DOCS_BUILD
        )
        print('+ {}'.format(cmd))
        rc = os.system(cmd)
        if rc != 0:
            raise Exception("Failed to run pdoc. rc={}".format(rc))

class Test(CommandHelper):

    def run(self):
        import pytest
        os.environ['PYTHONPATH'] = '.'
        rc = pytest.main(['-W', 'ignore', '-ra', '--cache-clear', 'test', '.'])
        if rc != 0:
            raise Exception('*** Tests failed.')

# Run setup

setup(
    name="munkres",
    version=version,
    description="Munkres (Hungarian) algorithm for the Assignment Problem",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=url,
    license=license,
    author=author,
    author_email=email,
    py_modules=["munkres"],
    cmdclass = {
        'doc': Doc,
        'docs': Doc,
        'apidoc': Doc,
        'apidocs': Doc,
        'test': Test
    },
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
