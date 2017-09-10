#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup


def setup_package():
    setup(
        name="face_shazam",
        version="1.0.0",
        install_requires=["numpy>=1.13.1, < 2"],
        setup_requires=['six', 'pyscaffold>=2.5a0,<2.6a0'],
        use_pyscaffold=True,

    )


if __name__ == "__main__":
    setup_package()
