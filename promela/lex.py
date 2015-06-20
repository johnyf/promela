"""Lexer for Promela, using Python Lex-Yacc (PLY)."""
import logging
import ply.lex


logger = logging.getLogger(__name__)


class Lexer(object):
    """Lexer for the Promela modeling language."""

    states = tuple([('ltl', 'inclusive')])
    reserved = {
        '_np': 'NONPROGRESS',
        'true': 'TRUE',
        'false': 'FALSE',
        'active': 'ACTIVE',
        'assert': 'ASSERT',
        'atomic': 'ATOMIC',
        'bit': 'BIT',
        'bool': 'BOOL',
        'break': 'BREAK',
        'byte': 'BYTE',
        'chan': 'CHAN',
        'd_step': 'D_STEP',
        'd_proctype': 'D_PROCTYPE',
        'do': 'DO',
        'else': 'ELSE',
        'empty': 'EMPTY',
        'enabled': 'ENABLED',
        'eval': 'EVAL',
        'fi': 'FI',
        'full': 'FULL',
        'get_priority': 'GET_P',
        'goto': 'GOTO',
        'hidden': 'HIDDEN',
        'if': 'IF',
        'init': 'INIT',
        'int': 'INT',
        'len': 'LEN',
        'local': 'ISLOCAL',
        'ltl': 'LTL',
        'mtype': 'MTYPE',
        'nempty': 'NEMPTY',
        'never': 'CLAIM',
        'nfull': 'NFULL',
        'od': 'OD',
        'of': 'OF',
        'pc_value': 'PC_VAL',
        'pid': 'PID',
        'printf': 'PRINT',
        'priority': 'PRIORITY',
        'proctype': 'PROCTYPE',
        'provided': 'PROVIDED',
        'R': 'RELEASE',
        'return': 'RETURN',
        'run': 'RUN',
        'short': 'SHORT',
        'skip': 'TRUE',
        'show': 'SHOW',
        'timeout': 'TIMEOUT',
        'typedef': 'TYPEDEF',
        'U': 'UNTIL',
        'unless': 'UNLESS',
        'unsigned': 'UNSIGNED',
        'X': 'NEXT',
        'xr': 'XR',
        'xs': 'XS',
        'W': 'WEAK_UNTIL'}
    values = {'skip': 'true'}
    delimiters = ['LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET',
                  'LBRACE', 'RBRACE', 'COMMA', 'PERIOD',
                  'SEMI', 'COLONS', 'COLON', 'ELLIPSIS']
    # remember to check precedence
    operators = ['PLUS', 'INCR', 'MINUS', 'DECR', 'TIMES', 'DIVIDE',
                 'MOD', 'OR', 'AND', 'NOT', 'XOR', 'IMPLIES', 'EQUIV',
                 'LOR', 'LAND', 'LNOT', 'LT', 'GT',
                 'LE', 'GE', 'EQ', 'NE',
                 'RCV', 'R_RCV', 'TX2', 'LSHIFT', 'RSHIFT', 'AT',
                 'ALWAYS', 'EVENTUALLY']
    misc = ['EQUALS', 'ARROW', 'STRING', 'NAME', 'INTEGER',
            'PREPROC', 'NEWLINE', 'COMMENT']

    def __init__(self, debug=False):
        self.tokens = (
            self.delimiters + self.operators +
            self.misc + list(set(self.reserved.values())))
        self.build(debug=debug)

    def build(self, debug=False, debuglog=None, **kwargs):
        """Create a lexer.

        @param kwargs: Same arguments as C{ply.lex.lex}:

          - except for C{module} (fixed to C{self})
          - C{debuglog} defaults to the module's logger.
        """
        if debug and debuglog is None:
            debuglog = logger
        self.lexer = ply.lex.lex(
            module=self,
            debug=debug,
            debuglog=debuglog,
            **kwargs)

    # check for reserved words
    def t_NAME(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        t.value = self.values.get(t.value, t.value)
        t.type = self.reserved.get(t.value, 'NAME')
        # switch to LTL context
        if t.value == 'ltl':
            self.lexer.level = 0
            self.lexer.begin('ltl')
        return t

    def t_STRING(self, t):
        r'"[^"]*"'
        return t

    # operators
    t_PLUS = r'\+'
    t_INCR = r'\+\+'
    t_MINUS = r'-'
    t_DECR = r'--'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_MOD = r'%'
    t_OR = r'\|'
    t_AND = r'&'
    t_NOT = r'~'
    t_XOR = r'\^'
    t_LOR = r'\|\|'
    t_LAND = r'&&'
    t_LNOT = r'!'
    t_TX2 = r'!!'
    t_LT = r'<'
    t_LSHIFT = r'<<'
    t_GT = r'>'
    t_RSHIFT = r'>>'
    t_LE = r'<='
    t_GE = r'>='
    t_EQ = r'=='
    t_NE = r'!='
    t_RCV = r'\?'
    t_R_RCV = r'\?\?'
    t_AT = r'@'
    t_EQUIV = r'<->'
    # assignment
    t_EQUALS = r'='
    # temporal operators
    t_ALWAYS = r'\[\]'
    t_EVENTUALLY = r'\<\>'
    # delimeters
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_COMMA = r','
    t_PERIOD = r'\.'
    t_SEMI = r';'
    t_COLONS = r'::'
    t_COLON = r':'
    t_ELLIPSIS = r'\.\.\.'

    def t_ltl_LBRACE(self, t):
        r'\{'
        self.lexer.level += 1
        return t

    def t_ltl_RBRACE(self, t):
        r'\}'
        self.lexer.level -= 1
        if self.lexer.level == 0:
            self.lexer.begin('INITIAL')
        return t

    def t_ltl_ARROW(self, t):
        r'->'
        t.type = 'IMPLIES'
        return t

    t_INITIAL_ARROW = r'->'

    def t_PREPROC(self, t):
        r'\#.*'
        pass

    def t_INTEGER(self, t):
        r'\d+([uU]|[lL]|[uU][lL]|[lL][uU])?'
        return t

    # t_ignore is reserved by lex to provide
    # much more efficient internal handling by lex
    #
    # A string containing ignored characters (spaces and tabs)
    t_ignore = ' \t'

    def t_NEWLINE(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count('\n')

    def t_COMMENT(self, t):
        r' /\*(.|\n)*?\*/'
        t.lineno += t.value.count('\n')

    def t_error(self, t):
        logger.error('Illegal character "{s}"'.format(s=t.value[0]))
        t.lexer.skip(1)
