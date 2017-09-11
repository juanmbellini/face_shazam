# -*- coding: utf-8 -*-
import pkg_resources
import logging

_logger = logging.getLogger(__name__)

# noinspection PyBroadException
try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception as e:
    _logger.warn("Unknown version")
    __version__ = 'unknown'
