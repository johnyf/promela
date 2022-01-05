"""Promela parser and syntax tree."""
from promela.yacc import Parser
try:
    from promela import _version
    __version__ = _version.version
except:
    __version__ = None
