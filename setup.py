#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FastPair: Data-structure for the dynamic closest-pair problem.

Installation script for FastPair.
"""

# Copyright (c) 2016, Carson J. Q. Farmer <carsonfarmer@gmail.com>
# Copyright (c) 2002-2015, David Eppstein
# Licensed under the MIT Licence (http://opensource.org/licenses/MIT).

import sys
import os
import warnings

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

PACKAGE_NAME = "FastPair"
DESCRIPTION = "FastPair: Data-structure for the dynamic closest-pair problem."

setup(name=PACKAGE_NAME, description=DESCRIPTION,
      license='MIT', author='Carson J. Q. Farmer',
      author_email='carsonfarmer@gmail.com',
      keywords="closest-pair points algorithm fastpair",
      long_description=DESCRIPTION, packages=find_packages("."),
      install_requires=["scipy"], zip_safe=True,
      setup_requires=["pytest-runner",], tests_require=["pytest",],
      classifiers=["Development Status :: 2 - Pre-Alpha",
                   "Environment :: Console",
                   "Intended Audience :: Science/Research",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Information Technology",
                   "License :: OSI Approved :: MIT License",
                   "Natural Language :: English",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python",
                   "Topic :: Scientific/Engineering :: Information Analysis",
                   "Topic :: System :: Distributed Computing",
                   ])
