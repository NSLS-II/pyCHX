#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

import sys
import warnings
import versioneer

from setuptools import setup, find_packages

no_git_reqs = []
with open('requirements.txt') as f:
    required = f.read().splitlines()
    for r in required:
        if not (r.startswith('git') or r.startswith('#') or r.strip() == ''):
            no_git_reqs.append(r)

setup(
    name='chxanalys',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Brookhaven National Laboratory_CHX',
    packages=find_packages(),
    install_requires=no_git_reqs,
)
