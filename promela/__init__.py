"""Promela parser and syntax tree."""
from promela.yacc import Parser
try:
    import promela._version as _version
    __version__ = _version.version
except:
    __version__ = None
