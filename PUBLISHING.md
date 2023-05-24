# Publishing to PyPI

(Notes to self...)

Adapted from <https://realpython.com/pypi-publish-python-package/>

Python 3 only (from version 1.1.0 forward):

```
$ pip install twine
$ pip install 'readme_renderer[md]'
$ python setup.py test # run tests. Make sure they pass.
$ python setup.py docs # create API docs (for main project page)
$ python setup.py sdist bdist_wheel
$ twine check dist/*
```

If all looks good, then:

```
$ twine -r pypi upload dist/*
```

Note: This assumes the existence of something like the following in
`~/.pypirc`:

```
[pypi]
[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = <pypi API token goes here>
```
