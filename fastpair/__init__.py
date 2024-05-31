#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FastPair: Data-structure for the dynamic closest-pair problem.

Init module for FastPair.
"""

# Copyright (c) 2016, Carson J. Q. Farmer <carsonfarmer@gmail.com>
# Copyright (c) 2002-2015, David Eppstein
# Licensed under the MIT Licence (http://opensource.org/licenses/MIT).

import contextlib
from importlib.metadata import PackageNotFoundError, version

from .base import FastPair

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("fastpair")
