# $Id$

all: sdist

sdist:
	python setup.py sdist --formats=gztar,zip

clean:
	rm -fr html dist build MANIFEST munkres.py
