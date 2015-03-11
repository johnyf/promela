"""This subpackage contains a Promela parser and interpreter.

It uses PLY, the Python Lex-Yacc implementation.
"""
from .yacc import Parser
try:
    from .version import version as __version__
except:
    __version__ = None
