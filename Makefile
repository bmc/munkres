# $Id$

all: dist

.PHONY: all sdist dist clean test

test:
	nosetests

dist: test
	python setup.py sdist bdist_wheel

doc: apidoc

apidoc:
	pdoc --html --html-no-source --overwrite --html-dir apidocs munkres.py

publish: dist
	python setup.py bdist_wheel upload

clean:
	rm -fr html dist build MANIFEST munkres.pyc apidocs
