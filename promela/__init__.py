"""Promela parser and syntax tree."""
from .yacc import Parser
try:
    from ._version import version as __version__
except:
    __version__ = None
