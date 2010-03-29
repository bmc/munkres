---
title: pyutmp — Python interface to Unix utmp
layout: withTOC
---

## Introduction

The `pyutmp` module provides a Python-oriented interface to the *utmp* file
on Unix-like operating systems. To paraphrase the *Linux Programmer's
Manual* page *utmp*(5), the *utmp* file allows one to discover information
about who is currently using (i.e., is logged into) the system. The *utmp*
file is a series of entries whose structure is typically defined by the
`utmp.h` C header file.

This module provides an read-only interface to the underlying operating
system's C *utmp* API.

## Getting and installing *pyutmp*

*pyutmp* relies on the presence of the *utmp* library, which
isn't automatically present on all operating systems. In addition, *pyutmp*
uses [Cython][]-generated C files to provide Python access to the C *utmp*
libraries.

It is impractical to provide binaries of *pyutmp* for every combination of
Unix-like operating system and operating system release. So, currently, you
must build *pyutmp* from source code, as described below.

First, obtain the source code. You can download the source (as a zip or
tarball) from <http://github.com/bmc/pyutmp/downloads>, or you can make a
local read-only clone of the [Git repository][] using one of the following
commands:

    $ git clone git://github.com/bmc/pyutmp.git
    $ git clone http://github.com/bmc/pyutmp.git

[Git repository]: http://github.com/bmc/pyutmp
[Cython]: http://www.cython.org/

Once you have a local `pyutmp` source directory, change your working directory
to the source directory, and type:

    python setup.py install

To install it somewhere other than the default location (such as in your
home directory) type:

    python setup.py install --prefix=$HOME

## Interface and Usage

The `pyutmp` module supplies two classes: `UtmpFile` and `Utmp`. A
`UtmpFile` object represents the open *utmp* file; when you iterate over a
`UtmpFile` object, it yields successive `Utmp` objects. For example:

    from pyutmp import UtmpFile
    import time

    for utmp in UtmpFile():
        # utmp is a Utmp object
        if utmp.ut_user_process:
            print '%s logged in at %s on tty %s' % (utmp.ut_user, time.ctime(utmp.ut_time), utmp.ut_line)

### UtmpFile

In addition to the `__iter__()` generator method, allowing iteration over
the contents of the *utmp* file, the `UtmpFile` class provides a `rewind()`
method that permits you to reset the file pointer to the top of the file.
See the class documentation for details.

### Utmp

The fields of the `Utmp` class are operating system-dependent. However, they
will *always* include at least the following fields:

* `ut_user` (string): The user associated with the *utmp* entry, if any.
* `ut_line` (string): The tty or pseudo-tty associated with the entry, if any.
  In this API, the line will *always* be the full path to the device.
* `ut_host` (string): The host name associated with the entry, if any.
* `ut_time` (timestamp:) The timestamp associated with the entry. This timestamp
  is in the form returnd by `time.time()` and may be passed directly to methods
  like `time.ctime()`.
* `ut_user_process` (bool): Whether or not the *utmp* entry is a user process
  (as opposed to a reboot or some other system event). 

On some operating systems, other fields may be present. For instance, on
Linux and Solaris systems (and other System V-derived systems), `Utmp` also
contains the following fields:

* `ut_type` (string): The type of the entry, typically one of the following 
  string values. See the *utmp*(5) manual page for a description of these
  strings.
    * "RUN_LVL"
    * "BOOT_TIME"
    * "NEW_TIME"
    * "OLD_TIME"
    * "INIT_PROCESS" 
    * "LOGIN_PROCESS"
    * "USER_PROCESS"
    * "DEAD_PROCESS"
    * "ACCOUNTING".          
* `ut_pid` (int): The associated process ID, if any.
* `ut_id` (string): The *init*(8) ID, or the abbreviated tty name.
* `ut_exit_code` (int): The process exit code, if applicable.
* `ut_session` (int): Session ID, for windowing.
* `ut_addr` (int array): IPv4 address of remote host (if applicable), one
  octet per array element.

If you're writing portable code, you should not count on the presence of
this secont set of attributes--or, at the very least, you should wrap
access to them in a `try/catch` block that catches `AttributeError`.

## Notes

This module has been tested on the following operating systems:

* Ubuntu Linux, version 8.04
* FreeBSD
* Mac OS X 10.4 (Tiger)
* OpenSolaris (2008.05, x86, using the SunStudio 12 compiler suite)

Adding support for other Unix variants should be straightforward.

## Restrictions

- Access to the *utmp* file is read-only. There is no provision for writing
  to the file.

## License

This module is released under a BSD license. See the accompanying
[license][] file.

[license]: license.html
