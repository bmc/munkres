# $Id$

all: dist

.PHONY: all sdist dist clean

sdist: dist
dist:
	python setup.py sdist --formats=gztar,zip

clean:
	rm -fr html dist build MANIFEST munkres.pyc
