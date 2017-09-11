#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import setuptools


os.environ['SKIP_GENERATE_AUTHORS'] = '1'
os.environ['SKIP_WRITE_GIT_CHANGELOG'] = '1'


def setup_package():
    setuptools.setup(setup_requires=['pbr'], pbr=True)


if __name__ == "__main__":
    setup_package()
