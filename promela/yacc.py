"""Parser for Promela, using Python Lex-Yacc (PLY).


References
==========

Holzmann G.J., The SPIN Model Checker,
    Addison-Wesley, 2004, pp. 365--368
    http://spinroot.com/spin/Man/Quick.html
"""
from __future__ import absolute_import
from __future__ import division
import logging
import os
import subprocess
import warnings
import ply.yacc
# inline
#
# import promela.ast as promela_ast
# from promela import lex


TABMODULE = 'promela.promela_parsetab'
logger = logging.getLogger(__name__)


class Parser(object):
    """Production rules for Promela parser."""

    logger = logger
    tabmodule = TABMODULE
    start = 'program'
    # http://spinroot.com/spin/Man/operators.html
    # spin.y
    # lowest to highest
    precedence = (
        ('right', 'EQUALS'),
        ('left', 'TX2', 'RCV', 'R_RCV'),
        ('left', 'IMPLIES', 'EQUIV'),
        ('left', 'LOR'),
        ('left', 'LAND'),
        ('left', 'ALWAYS', 'EVENTUALLY'),
        ('left', 'UNTIL', 'WEAK_UNTIL', 'RELEASE'),
        ('right', 'NEXT'),
        ('left', 'OR'),
        ('left', 'XOR'),
        ('left', 'AND'),
        ('left', 'EQ', 'NE'),
        ('left', 'LT', 'LE', 'GT', 'GE'),
        ('left', 'LSHIFT', 'RSHIFT'),
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIVIDE', 'MOD'),
        ('left', 'INCR', 'DECR'),
        ('right', 'LNOT', 'NOT', 'UMINUS', 'NEG'),  # LNOT is also SND
        ('left', 'DOT'),
        ('left', 'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET'))

    def __init__(self, ast=None, lexer=None):
        if ast is None:
            import promela.ast as ast
        if lexer is None:
            from promela import lex
            lexer = lex.Lexer()
        self.lexer = lexer
        self.ast = ast
        self.tokens = self.lexer.tokens
        self.build()

    def build(self, tabmodule=None, outputdir='', write_tables=False,
              debug=False, debuglog=None, errorlog=None):
        """Build parser using `ply.yacc`.

        Default table module is `self.tabmodule`.
        Module logger used as default debug logger.
        Default error logger is that created by PLY.
        """
        if tabmodule is None:
            tabmodule = self.tabmodule
        if debug and debuglog is None:
            debuglog = self.logger
        self.parser = ply.yacc.yacc(
            method='LALR',
            module=self,
            start=self.start,
            tabmodule=tabmodule,
            outputdir=outputdir,
            write_tables=write_tables,
            debug=debug,
            debuglog=debuglog,
            errorlog=errorlog)

    def parse(self, promela):
        """Parse string of Promela code."""
        s = cpp(promela)
        program = self.parser.parse(
            s, lexer=self.lexer.lexer, debug=self.logger)
        return program

    def _iter(self, p):
        if p[2] is not None:
            p[1].append(p[2])
        return p[1]

    def _end(self, p):
        if p[1] is None:
            return list()
        else:
            return [p[1]]

    # Top-level constructs
    # ====================

    def p_program(self, p):
        """program : units"""
        p[0] = self.ast.Program(p[1])

    def p_units_iter(self, p):
        """units : units unit"""
        p[0] = self._iter(p)

    def p_units_end(self, p):
        """units : unit"""
        p[0] = self._end(p)

    # TODO: events, c_fcts, ns, error
    def p_unit_proc(self, p):
        """unit : proc
                | init
                | claim
                | ltl
        """
        p[0] = p[1]

    def p_unit_decl(self, p):
        """unit : one_decl
                | utype
        """
        p[0] = p[1]

    def p_unit_semi(self, p):
        """unit : semi"""

    def p_proc(self, p):
        ("""proc : prefix_proctype NAME"""
         """       LPAREN decl RPAREN"""
         """       opt_priority opt_enabler"""
         """       body
         """)
        inst = p[1]
        name = p[2]
        args = p[4]
        priority = p[6]
        enabler = p[7]
        body = p[8]

        p[0] = self.ast.Proctype(
            name, body, args=args, priority=priority,
            provided=enabler, **inst)

    # instantiator
    def p_inst(self, p):
        """prefix_proctype : ACTIVE opt_index proctype"""
        d = p[3]
        if p[2] is None:
            n_active = self.ast.Integer('1')
        else:
            n_active = p[2]
        d['active'] = n_active
        p[0] = d

    def p_inactive_proctype(self, p):
        """prefix_proctype : proctype"""
        p[0] = p[1]

    def p_opt_index(self, p):
        """opt_index : LBRACKET expr RBRACKET
                     | LBRACKET NAME RBRACKET
        """
        p[0] = p[2]

    def p_opt_index_empty(self, p):
        """opt_index : empty"""

    def p_init(self, p):
        """init : INIT opt_priority body"""
        p[0] = self.ast.Init(name='init', body=p[3], priority=p[2])

    def p_claim(self, p):
        """claim : CLAIM optname body"""
        name = p[2] if p[2] else 'never'
        p[0] = self.ast.NeverClaim(name=name, body=p[3])

    # user-defined type
    def p_utype(self, p):
        """utype : TYPEDEF NAME LBRACE decl_lst RBRACE"""
        seq = self.ast.Sequence(p[4])
        p[0] = self.ast.TypeDef(p[2], seq)

    def p_ltl(self, p):
        """ltl : LTL LBRACE expr RBRACE"""
        p[0] = self.ast.LTL(p[3])

    # Declarations
    # ============

    def p_decl(self, p):
        """decl : decl_lst"""
        p[0] = self.ast.Sequence(p[1])

    def p_decl_empty(self, p):
        """decl : empty"""

    def p_decl_lst_iter(self, p):
        """decl_lst : one_decl SEMI decl_lst"""
        p[0] = [p[1]] + p[3]

    def p_decl_lst_end(self, p):
        """decl_lst : one_decl"""
        p[0] = [p[1]]

    def p_one_decl_visible(self, p):
        """one_decl : vis typename var_list
                    | vis NAME var_list
        """
        visible = p[1]
        typ = p[2]
        var_list = p[3]

        p[0] = self.one_decl(typ, var_list, visible)

    def p_one_decl(self, p):
        """one_decl : typename var_list
                    | NAME var_list
        """
        typ = p[1]
        var_list = p[2]
        p[0] = self.one_decl(typ, var_list)

    def one_decl(self, typ, var_list, visible=None):
        c = list()
        for d in var_list:
            v = self.ast.VarDef(vartype=typ, visible=visible, **d)
            c.append(v)
        return self.ast.Sequence(c)

    # message type declaration
    def p_one_decl_mtype_vis(self, p):
        """one_decl : vis MTYPE asgn LBRACE name_list RBRACE"""
        p[0] = self.ast.MessageType(p[5], visible=p[1])

    def p_one_decl_mtype(self, p):
        """one_decl : MTYPE asgn LBRACE name_list RBRACE"""
        p[0] = self.ast.MessageType(p[4])

    def p_name_list_iter(self, p):
        """name_list : name_list COMMA NAME"""
        p[1].append(p[3])
        p[0] = p[1]

    def p_name_list_end(self, p):
        """name_list : NAME"""
        p[0] = [p[1]]

    def p_var_list_iter(self, p):
        """var_list : ivar COMMA var_list"""
        p[0] = [p[1]] + p[3]

    def p_var_list_end(self, p):
        """var_list : ivar"""
        p[0] = [p[1]]

    # TODO: vardcl asgn LBRACE c_list RBRACE

    # ivar = initialized variable
    def p_ivar(self, p):
        """ivar : vardcl"""
        p[0] = p[1]

    def p_ivar_asgn(self, p):
        """ivar : vardcl asgn expr"""
        expr = self.ast.Expression(p[3])
        p[1]['initval'] = expr
        p[0] = p[1]

    def p_vardcl(self, p):
        """vardcl : NAME"""
        p[0] = {'name': p[1]}

    # p.403, SPIN manual
    def p_vardcl_unsigned(self, p):
        """vardcl : NAME COLON const"""
        p[0] = {'name': p[1], 'bitwidth': p[3]}

    def p_vardcl_array(self, p):
        """vardcl : NAME LBRACKET const_expr RBRACKET"""
        p[0] = {'name': p[1], 'length': p[3]}

    def p_vardcl_chan(self, p):
        """vardcl : vardcl EQUALS ch_init"""
        p[1].update(p[3])
        p[0] = p[1]

    def p_typename(self, p):
        """typename : BIT
                    | BOOL
                    | BYTE
                    | CHAN
                    | INT
                    | PID
                    | SHORT
                    | UNSIGNED
                    | MTYPE
        """
        p[0] = p[1]

    def p_ch_init(self, p):
        ("""ch_init : LBRACKET const_expr RBRACKET """
         """ OF LBRACE typ_list RBRACE""")
        p[0] = {'length': p[2], 'msg_types': p[6]}

    def p_typ_list_iter(self, p):
        """typ_list : typ_list COMMA basetype"""
        p[1].append(p[3])
        p[0] = p[1]

    def p_typ_list_end(self, p):
        """typ_list : basetype"""
        p[0] = [p[1]]

    # TODO: | UNAME | error
    def p_basetype(self, p):
        """basetype : typename"""
        p[0] = p[1]

    # References
    # ==========

    def p_varref(self, p):
        """varref : cmpnd"""
        p[0] = p[1]

    def p_cmpnd_iter(self, p):
        """cmpnd : cmpnd PERIOD cmpnd %prec DOT"""
        p[0] = self.ast.VarRef(extension=p[3], **p[1])

    def p_cmpnd_end(self, p):
        """cmpnd : pfld"""
        p[0] = self.ast.VarRef(**p[1])

    # pfld = prefix field
    def p_pfld_indexed(self, p):
        """pfld : NAME LBRACKET expr RBRACKET"""
        p[0] = {'name': p[1], 'index': p[3]}

    def p_pfld(self, p):
        """pfld : NAME"""
        p[0] = {'name': p[1]}

    # Attributes
    # ==========

    def p_opt_priority(self, p):
        """opt_priority : PRIORITY number"""
        p[0] = p[2]

    def p_opt_priority_empty(self, p):
        """opt_priority : empty"""

    def p_opt_enabler(self, p):
        """opt_enabler : PROVIDED LPAREN expr RPAREN"""
        p[0] = p[3]

    def p_opt_enabler_empty(self, p):
        """opt_enabler : empty"""

    def p_body(self, p):
        """body : LBRACE sequence os RBRACE"""
        p[0] = p[2]

    # Sequence
    # ========

    def p_sequence(self, p):
        """sequence : sequence msemi step"""
        p[1].append(p[3])
        p[0] = p[1]

    def p_sequence_ending_with_atomic(self, p):
        """sequence : seq_block step"""
        p[1].append(p[2])
        p[0] = p[1]

    def p_sequence_single(self, p):
        """sequence : step"""
        p[0] = self.ast.Sequence([p[1]])

    def p_seq_block(self, p):
        """seq_block : sequence msemi atomic
                     | sequence msemi dstep
        """
        p[1].append(p[3])
        p[0] = p[1]

    def p_seq_block_iter(self, p):
        """seq_block : seq_block atomic
                     | seq_block dstep
        """
        p[1].append(p[2])
        p[0] = p[1]

    def p_seq_block_single(self, p):
        """seq_block : atomic
                     | dstep
        """
        p[0] = [p[1]]

    # TODO: XU vref_lst
    def p_step_1(self, p):
        """step : one_decl
                | stmnt
        """
        p[0] = p[1]

    def p_step_labeled(self, p):
        """step : NAME COLON one_decl"""
        raise Exception(
            'label preceding declaration: {s}'.format(s=p[3]))

    def p_step_3(self, p):
        """step : NAME COLON XR
                | NAME COLON XS
        """
        raise Exception(
            'label preceding xr/xs claim')

    def p_step_4(self, p):
        """step : stmnt UNLESS stmnt"""
        p[0] = (p[1], 'unless', p[3])
        self.logger.warning('UNLESS not interpreted yet')

    # Statement
    # =========

    def p_stmnt(self, p):
        """stmnt : special
                 | statement
        """
        p[0] = p[1]

    # Stmnt in spin.y
    def p_statement_asgn(self, p):
        """statement : varref asgn full_expr"""
        p[0] = self.ast.Assignment(var=p[1], value=p[3])

    def p_statement_incr(self, p):
        """statement : varref INCR"""
        one = self.ast.Integer('1')
        expr = self.ast.Expression(self.ast.Binary('+', p[1], one))
        p[0] = self.ast.Assignment(p[1], expr)

    def p_statement_decr(self, p):
        """statement : varref DECR"""
        one = self.ast.Integer('1')
        expr = self.ast.Expression(self.ast.Binary('-', p[1], one))
        p[0] = self.ast.Assignment(p[1], expr)

    def p_statement_assert(self, p):
        """statement : ASSERT full_expr"""
        p[0] = self.ast.Assert(p[2])

    def p_statement_fifo_receive(self, p):
        """statement : varref RCV rargs"""
        p[0] = self.ast.Receive(p[1], p[3])

    def p_statement_copy_fifo_receive(self, p):
        """statement : varref RCV LT rargs GT"""
        p[0] = self.ast.Receive(p[1], p[4])

    def p_statement_random_receive(self, p):
        """statement : varref R_RCV rargs"""
        p[0] = self.ast.Receive(p[1], p[3])

    def p_statement_copy_random_receive(self, p):
        """statement : varref R_RCV LT rargs GT"""
        p[0] = self.ast.Receive(p[1], p[4])

    def p_statement_tx2(self, p):
        """statement : varref TX2 margs"""
        p[0] = self.ast.Send(p[1], p[3])

    def p_statement_full_expr(self, p):
        """statement : full_expr"""
        p[0] = p[1]

    def p_statement_else(self, p):
        """statement : ELSE"""
        p[0] = self.ast.Else()

    def p_statement_atomic(self, p):
        """statement : atomic"""
        p[0] = p[1]

    def p_atomic(self, p):
        """atomic : ATOMIC LBRACE sequence os RBRACE"""
        s = p[3]
        s.context = 'atomic'
        p[0] = s

    def p_statement_dstep(self, p):
        """statement : dstep"""
        p[0] = p[1]

    def p_dstep(self, p):
        """dstep : D_STEP LBRACE sequence os RBRACE"""
        s = p[3]
        s.context = 'd_step'
        p[0] = s

    def p_statement_braces(self, p):
        """statement : LBRACE sequence os RBRACE"""
        p[0] = p[2]

    # the stmt of line 696 in spin.y collects the inline ?
    def p_statement_call(self, p):
        """statement : NAME LPAREN args RPAREN"""
        # NAME = INAME = inline
        c = self.ast.Inline(p[1], p[3])
        p[0] = self.ast.Sequence([c])

    def p_statement_assgn_call(self, p):
        """statement : varref asgn NAME LPAREN args RPAREN statement"""
        inline = self.ast.Inline(p[3], p[5])
        p[0] = self.ast.Assignment(p[1], inline)

    def p_statement_return(self, p):
        """statement : RETURN full_expr"""
        p[0] = self.ast.Return(p[2])

    def p_printf(self, p):
        """statement : PRINT LPAREN STRING prargs RPAREN"""
        p[0] = self.ast.Printf(p[3], p[4])

    # yet unimplemented for statement:
        # SET_P l_par two_args r_par
        # PRINTM l_par varref r_par
        # PRINTM l_par CONST r_par
        # ccode

    # Special
    # =======

    def p_special(self, p):
        """special : varref RCV"""
        p[0] = self.ast.Receive(p[1])

    def p_varref_lnot(self, p):
        """special : varref LNOT margs"""
        raise NotImplementedError

    def p_break(self, p):
        """special : BREAK"""
        p[0] = self.ast.Break()

    def p_goto(self, p):
        """special : GOTO NAME"""
        p[0] = self.ast.Goto(p[2])

    def p_labeled_stmt(self, p):
        """special : NAME COLON stmnt"""
        p[0] = self.ast.Label(p[1], p[3])

    def p_labeled(self, p):
        """special : NAME COLON"""
        p[0] = self.ast.Label(
            p[1],
            self.ast.Expression(self.ast.Bool('true')))

    def p_special_if(self, p):
        """special : IF options FI"""
        p[0] = self.ast.Options('if', p[2])

    def p_special_do(self, p):
        """special : DO options OD"""
        p[0] = self.ast.Options('do', p[2])

    def p_options_end(self, p):
        """options : option"""
        p[0] = [p[1]]

    def p_options_iter(self, p):
        """options : options option"""
        p[1].append(p[2])
        p[0] = p[1]

    def p_option(self, p):
        """option : COLONS sequence os"""
        s = p[2]
        s.is_option = True
        p[0] = s

    # Expressions
    # ===========

    def p_full_expr(self, p):
        """full_expr : expr
                     | pexpr
        """
        p[0] = self.ast.Expression(p[1])

    # probe expr = no negation allowed (positive)
    def p_pexpr(self, p):
        """pexpr : probe
                 | LPAREN pexpr RPAREN
                 | pexpr LAND pexpr
                 | pexpr LAND expr
                 | expr LAND pexpr
                 | pexpr LOR pexpr
                 | pexpr LOR expr
                 | expr LOR pexpr
        """
        p[0] = 'pexpr'

    def p_probe(self, p):
        """probe : FULL LPAREN varref RPAREN
                 | NFULL LPAREN varref RPAREN
                 | EMPTY LPAREN varref RPAREN
                 | NEMPTY LPAREN varref RPAREN
        """
        p[0] = 'probe'

    def p_expr_paren(self, p):
        """expr : LPAREN expr RPAREN"""
        p[0] = p[2]

    def p_expr_arithmetic(self, p):
        """expr : expr PLUS expr
                | expr MINUS expr
                | expr TIMES expr
                | expr DIVIDE expr
                | expr MOD expr
        """
        p[0] = self.ast.Binary(p[2], p[1], p[3])

    def p_expr_not(self, p):
        """expr : NOT expr
                | MINUS expr %prec UMINUS
                | LNOT expr %prec NEG
        """
        p[0] = self.ast.Unary(p[1], p[2])

    def p_expr_logical(self, p):
        """expr : expr AND expr
                | expr OR expr
                | expr XOR expr
                | expr LAND expr
                | expr LOR expr
        """
        p[0] = self.ast.Binary(p[2], p[1], p[3])

    # TODO: cexpr

    def p_expr_shift(self, p):
        """expr : expr LSHIFT expr
                | expr RSHIFT expr
        """
        p[0] = p[1]

    def p_expr_const_varref(self, p):
        """expr : const
                | varref
        """
        p[0] = p[1]

    def p_expr_varref(self, p):
        """expr : varref RCV LBRACKET rargs RBRACKET
                | varref R_RCV LBRACKET rargs RBRACKET
        """
        p[0] = p[1]
        warnings.warn('not implemented')

    def p_expr_other(self, p):
        """expr : LPAREN expr ARROW expr COLON expr RPAREN
                | LEN LPAREN varref RPAREN
                | ENABLED LPAREN expr RPAREN
                | GET_P LPAREN expr RPAREN
        """
        p[0] = p[1]
        warnings.warn('"{s}" not implemented'.format(s=p[1]))

    def p_expr_run(self, p):
        """expr : RUN aname LPAREN args RPAREN opt_priority"""
        p[0] = self.ast.Run(p[2], p[4], p[6])

    def p_expr_other_2(self, p):
        """expr : TIMEOUT
                | NONPROGRESS
                | PC_VAL LPAREN expr RPAREN
        """
        raise NotImplementedError()

    def p_expr_remote_ref_proctype_pc(self, p):
        """expr : NAME AT NAME
        """
        p[0] = self.ast.RemoteRef(p[1], p[3])

    def p_expr_remote_ref_pid_pc(self, p):
        """expr : NAME LBRACKET expr RBRACKET AT NAME"""
        p[0] = self.ast.RemoteRef(p[1], p[6], pid=p[3])

    def p_expr_remote_ref_var(self, p):
        """expr : NAME LBRACKET expr RBRACKET COLON pfld"""
        # | NAME COLON pfld %prec DOT2
        raise NotImplementedError()

    def p_expr_comparator(self, p):
        """expr : expr EQ expr
                | expr NE expr
                | expr LT expr
                | expr LE expr
                | expr GT expr
                | expr GE expr
        """
        p[0] = self.ast.Binary(p[2], p[1], p[3])

    def p_binary_ltl_expr(self, p):
        """expr : expr UNTIL expr
                | expr WEAK_UNTIL expr
                | expr RELEASE expr
                | expr IMPLIES expr
                | expr EQUIV expr
        """
        p[0] = self.ast.Binary(p[2], p[1], p[3])

    def p_unary_ltl_expr(self, p):
        """expr : NEXT expr
                | ALWAYS expr
                | EVENTUALLY expr
        """
        p[0] = self.ast.Unary(p[1], p[2])

    # Constants
    # =========

    def p_const_expr_const(self, p):
        """const_expr : const"""
        p[0] = p[1]

    def p_const_expr_unary(self, p):
        """const_expr : MINUS const_expr %prec UMINUS"""
        p[0] = self.ast.Unary(p[1], p[2])

    def p_const_expr_binary(self, p):
        """const_expr : const_expr PLUS const_expr
                      | const_expr MINUS const_expr
                      | const_expr TIMES const_expr
                      | const_expr DIVIDE const_expr
                      | const_expr MOD const_expr
        """
        p[0] = self.ast.Binary(p[2], p[1], p[3])

    def p_const_expr_paren(self, p):
        """const_expr : LPAREN const_expr RPAREN"""
        p[0] = p[2]

    def p_const(self, p):
        """const : boolean
                 | number
        """
        # lex maps `skip` to `TRUE`
        p[0] = p[1]

    def p_bool(self, p):
        """boolean : TRUE
                   | FALSE
        """
        p[0] = self.ast.Bool(p[1])

    def p_number(self, p):
        """number : INTEGER"""
        p[0] = self.ast.Integer(p[1])

    # Auxiliary
    # =========

    def p_two_args(self, p):
        """two_args : expr COMMA expr"""

    def p_args(self, p):
        """args : arg"""
        p[0] = p[1]

    def p_prargs(self, p):
        """prargs : COMMA arg"""
        p[0] = p[2]

    def p_prargs_empty(self, p):
        """prargs : empty"""

    def p_args_empty(self, p):
        """args : empty"""

    def p_margs(self, p):
        """margs : arg
                 | expr LPAREN arg RPAREN
        """

    def p_arg(self, p):
        """arg : expr
               | expr COMMA arg
        """
        p[0] = 'arg'

    # TODO: CONST, MINUS CONST %prec UMIN
    def p_rarg(self, p):
        """rarg : varref
                | EVAL LPAREN expr RPAREN
        """
        p[0] = 'rarg'

    def p_rargs(self, p):
        """rargs : rarg
                 | rarg COMMA rargs
                 | rarg LPAREN rargs RPAREN
                 | LPAREN rargs RPAREN
        """

    def p_proctype(self, p):
        """proctype : PROCTYPE
                    | D_PROCTYPE
        """
        if p[1] == 'proctype':
            p[0] = dict(d_proc=False)
        else:
            p[0] = dict(d_proc=True)

    # PNAME
    def p_aname(self, p):
        """aname : NAME"""
        p[0] = p[1]

    # optional name
    def p_optname(self, p):
        """optname : NAME"""
        p[0] = p[1]

    def p_optname_empty(self, p):
        """optname : empty"""

    # optional semi
    def p_os(self, p):
        """os : empty
              | semi
        """
        p[0] = ';'

    # multi-semi
    def p_msemi(self, p):
        """msemi : semi
                 | msemi semi
        """
        p[0] = ';'

    def p_semi(self, p):
        """semi : SEMI
                | ARROW
        """
        p[0] = ';'

    def p_asgn(self, p):
        """asgn : EQUALS
                | empty
        """
        p[0] = None

    def p_visible(self, p):
        """vis : HIDDEN
               | SHOW
               | ISLOCAL
        """
        p[0] = {'visible': p[1]}

    def p_empty(self, p):
        """empty : """

    def p_error(self, p):
        raise Exception('syntax error at: {p}'.format(p=p))


def cpp(s):
    """Call the C{C} preprocessor with input C{s}."""
    try:
        p = subprocess.Popen(['cpp', '-E', '-x', 'c'],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
    except OSError as e:
        if e.errno == os.errno.ENOENT:
            raise Exception('C preprocessor (cpp) not found in path.')
        else:
            raise
    logger.debug('cpp input:\n' + s)
    stdout, stderr = p.communicate(s)
    logger.debug('cpp returned: {c}'.format(c=p.returncode))
    logger.debug('cpp stdout:\n {out}'.format(out=stdout))
    return stdout


def rebuild_table(parser, tabmodule):
    # log details to file
    h = logging.FileHandler('log.txt', mode='w')
    debuglog = logging.getLogger()
    debuglog.addHandler(h)
    debuglog.setLevel('DEBUG')
    import os
    outputdir = './'
    # rm table files to force rebuild to get debug output
    tablepy = tabmodule + '.py'
    tablepyc = tabmodule + '.pyc'
    try:
        os.remove(tablepy)
    except:
        print('no "{t}" found'.format(t=tablepy))
    try:
        os.remove(tablepyc)
    except:
        print('no "{t}" found'.format(t=tablepyc))
    parser.build(tabmodule, outputdir=outputdir,
                 write_tables=True, debug=True,
                 debuglog=debuglog)


if __name__ == '__main__':
    rebuild_table(Parser(), TABMODULE.split('.')[-1])


# TODO
#
# expr << expr
# expr >> expr
# (expr -> expr : expr)
# run func(args) priority
# len(varref)
# enabled(expr)
# get_p(expr)
# var ? [rargs]
# var ?? [rargs]
# timeout
# nonprogress
# pc_val(expr)
# name[expr] @ name
# name[expr] : pfld
# name @ name
# name : pfld
