#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools


def setup_package():
    setuptools.setup(setup_requires=['pbr'], pbr=True)


if __name__ == "__main__":
    setup_package()
