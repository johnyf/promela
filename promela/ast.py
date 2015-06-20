"""Abstract syntax tree for Promela."""
from __future__ import absolute_import
from __future__ import division
import logging
import copy
import ctypes
import pprint
import networkx as nx
from networkx.utils import misc


logger = logging.getLogger(__name__)
DATATYPES = {
    'bit': ctypes.c_bool,
    'bool': ctypes.c_bool,
    'byte': ctypes.c_ubyte,
    'pid': ctypes.c_ubyte,
    'short': ctypes.c_short,
    'int': ctypes.c_int,
    'unsigned': None,
    'chan': None,
    'mtype': None}
N = 0


def generate_unique_node():
    """Return a fresh integer to be used as node."""
    global N
    N += 1
    return N


def _indent(s, depth=1, skip=0):
    w = []
    for i, line in enumerate(s.splitlines()):
        indent = '' if i < skip else depth * '\t'
        w.append(indent + line)
    return '\n'.join(w)


def to_str(x):
    try:
        return x.to_str()
    except AttributeError:
        return str(x)


class Proctype(object):
    def __init__(self, name, body, args=None,
                 active=None, d_proc=False,
                 priority=None, provided=None):
        self.name = name
        self.body = body
        self.args = args
        if active is None:
            active = 0
        else:
            active = int(active.value)
        if priority is not None:
            priority = int(priority.value)
        self.active = active
        self.priority = priority
        self.provided = provided

    def __str__(self):
        return "Proctype('{name}')".format(name=self.name)

    def to_str(self):
        return (
            '{active} proctype {name}({args}){{\n'
            '{body}\n'
            '}}\n\n').format(
                active=self._active_str(),
                name=self.name,
                args=self._args_str(),
                body=_indent(to_str(self.body)))

    def _active_str(self):
        if self.active is None:
            active = ''
        else:
            active = 'active [' + to_str(self.active) + '] '
        return active

    def _args_str(self):
        if self.args is None:
            args = ''
        else:
            args = to_str(self.args)
        return args

    def to_pg(self, syntactic_else=False):
        """Return program graph of proctype.

        @param syntactic_else: if `True`, then "else"
            statements in directly nested "if" or "do"
            take precedence based on syntactic context.
            The Promela language reference defines
            "else" semantically, with respect to
            the program graph.
        """
        global N
        N = 1
        g = nx.MultiDiGraph(name=self.name)
        g.locals = set()
        ine, out = self.body.to_pg(g, syntactic_else=syntactic_else)
        # root: explicit is better than implicit
        u = generate_unique_node()
        g.add_node(u, color='red', context=None)
        g.root = u
        for v, d in ine:
            g.add_edge(u, v, **d)
        # rm unreachable nodes
        S = nx.descendants(g, g.root)
        S.add(g.root)
        [g.remove_node(x) for x in g.nodes() if x not in S]
        if logger.getEffectiveLevel() == 1:
            dump_graph(
                g, 'dbg.pdf', node_label='context',
                edge_label='stmt', relabel=True)
        # contract goto edges
        assert_gotos_are_admissible(g)
        for u in sorted(g.nodes()):
            contract_goto_edges(g, u)
        h = map_uuid_to_int(g)
        # other out-edges of each node with an `else`
        if not syntactic_else:
            semantic_else(h)
        return h


def contract_goto_edges(g, u):
    """Identify nodes connected with `goto` edge."""
    assert u in g
    assert g.root in g
    n = g.out_degree(u)
    if n == 0 or 1 < n:
        return
    assert n == 1, n
    # single outgoing edge: safe to contract
    _, q, d = next(g.edges_iter(u, data=True))
    if d['stmt'] != 'goto':
        return
    # goto
    assert u != q, 'loop of `goto`s detected'
    # the source node context (atomic or d_step) is overwritten
    for p, _, d in g.in_edges_iter(u, data=True):
        g.add_edge(p, q, **d)
    # but the source node label is preserved
    u_label = g.node[u].get('labels')
    if u_label is not None:
        g.node[q].setdefault('labels', set()).update(u_label)
    g.remove_node(u)
    if u == g.root:
        g.root = q


def assert_gotos_are_admissible(g):
    """Assert no branch node has outgoing `goto`."""
    # branch node cannot have goto
    # `goto` and `break` must have transformed to `true`
    # labels must have raised `Exception`
    for u in g:
        if g.out_degree(u) <= 1:
            continue
        for _, v, d in g.edges_iter(u, data=True):
            assert 'stmt' in d
            stmt = d['stmt']
            assert stmt != 'goto', stmt
    for u, d in g.nodes_iter(data=True):
        assert 'context' in d
    for u, v, d in g.edges_iter(data=True):
        assert 'stmt' in d


def map_uuid_to_int(g):
    """Reinplace uuid nodes with integers."""
    umap = {u: i for i, u in enumerate(sorted(g))}
    h = nx.MultiDiGraph(name=g.name)
    for u, d in g.nodes_iter(data=True):
        p = umap[u]
        h.add_node(p, **d)
    for u, v, key, d in g.edges_iter(keys=True, data=True):
        p = umap[u]
        q = umap[v]
        h.add_edge(p, q, key=key, **d)
    h.root = umap[g.root]
    h.locals = g.locals
    return h


def semantic_else(g):
    """Set `Else.other_guards` to other edges with same source."""
    for u, v, d in g.edges_iter(data=True):
        stmt = d['stmt']
        if not isinstance(stmt, Else):
            continue
        # is `Else`
        stmt.other_guards = [
            q['stmt'] for _, _, q in g.out_edges_iter(u, data=True)
            if q['stmt'] != stmt]


class NeverClaim(Proctype):
    """Subclass exists only for semantic purposes."""
    def to_str(self):
        name = '' if self.name is None else self.name
        s = (
            'never ' + name + '{\n' +
            _indent(to_str(self.body)) + '\n'
            '}\n\n')
        return s


class Init(Proctype):
    def to_str(self):
        return (
            'init ' + '{\n' +
            _indent(to_str(self.body)) + '\n'
            '}\n\n')


class Program(list):
    def __str__(self):
        return '\n'.join(to_str(x) for x in self)

    def __repr__(self):
        c = super(Program, self).__repr__()
        return 'Program({c})'.format(c=c)

    def to_table(self):
        """Return global definitions, proctypes, and LTL blocks.

        @rtype: 3-`tuple` of `set`
        """
        units = misc.flatten(self)
        ltl = {x for x in units if isinstance(x, LTL)}
        proctypes = {x for x in units if isinstance(x, Proctype)}
        global_defs = {x for x in units
                       if x not in proctypes and x not in ltl}
        return global_defs, proctypes, ltl


class LTL(object):
    """Used to mark strings as LTL blocks."""

    def __init__(self, formula):
        self.formula = formula

    def __repr__(self):
        return 'LTL({f})'.format(f=repr(self.formula))

    def __str__(self):
        return 'ltl {' + str(self.formula) + '}'


class Sequence(list):
    def __init__(self, iterable, context=None, is_option=False):
        super(Sequence, self).__init__(iterable)
        # "atomic" or "dstep"
        self.context = context
        self.is_option = is_option

    def to_str(self):
        if self.context is None:
            return '\n'.join(to_str(x) for x in self)
        else:
            return (
                self.context + '{\n' +
                _indent(to_str(self)) + '\n}\n')

    def __repr__(self):
        l = super(Sequence, self).__repr__()
        return 'Sequence({l}, context={c}, is_option={isopt})'.format(
            l=l, c=self.context, isopt=self.is_option)

    def to_pg(self, g, context=None, option_guard=None, **kw):
        # set context
        if context is None:
            context = self.context
        c = context
        assert c in {'atomic', 'd_step', None}
        # atomic cannot appear inside d_step
        if context == 'd_step' and c == 'atomic':
            raise Exception('atomic inside d_step')
        context = c
        # find first non-decl
        # option guard first
        option_guard = self.is_option or option_guard
        assert len(self) > 0
        stmts = iter(self)
        t = None
        for stmt in stmts:
            t = stmt.to_pg(g, context=context,
                           option_guard=option_guard, **kw)
            if t is not None:
                break
        # no statements ?
        if t is None:
            return None
        e, tail = t
        # guard can't be a goto or label
        # (should have been caught below)
        if option_guard:
            for u, d in e:
                assert d.get('stmt') != 'goto', self
        # other option statements
        for stmt in stmts:
            t = stmt.to_pg(g, context=context, option_guard=None, **kw)
            # decl ?
            if t is None:
                continue
            ine, out = t
            # connect tail to ine
            assert ine
            for v, d in ine:
                g.add_edge(tail, v, **d)
            # update tail
            assert out in g
            tail = out
        return e, tail


class Node(object):
    def to_pg(self, g, context=None, **kw):
        u = generate_unique_node()
        g.add_node(u, context=context)
        e = (u, dict(stmt=self))
        return [e], u


class Options(Node):
    def __init__(self, opt_type, options):
        self.type = opt_type
        self.options = options

    def to_str(self):
        a, b = self.entry_exit
        c = list()
        c.append(a)
        c.append('\n')
        for option in self.options:
            option_guard = _indent(to_str(option[0]), skip=1)
            w = [_indent(to_str(x)) for x in option[1:]]
            c.append(
                ':: {option_guard}{tail}\n'.format(
                    option_guard=option_guard,
                    tail=(' ->\n' + '\n'.join(w)) if w else ''))
        c.append(b)
        c.append(';\n')
        return ''.join(c)

    @property
    def entry_exit(self):
        if self.type == 'if':
            return ('if', 'fi')
        elif self.type == 'do':
            return ('do', 'od')

    def to_pg(self, g, od_exit=None,
              context=None, option_guard=None,
              syntactic_else=False, **kw):
        logger.info('-- start flattening {t}'.format(t=self.type))
        assert self.options
        assert self.type in {'if', 'do'}
        # create target
        target = generate_unique_node()
        g.add_node(target, context=context)
        # target != exit node ?
        if self.type == 'do':
            od_exit = generate_unique_node()
            g.add_node(od_exit, context=context)
        self_else = None
        self_has_else = False
        option_has_else = False
        edges = list()
        else_ine = None
        for option in self.options:
            logger.debug('option: {opt}'.format(opt=option))
            t = option.to_pg(g, od_exit=od_exit, context=context, **kw)
            assert t is not None  # decls filtered by `Sequence`
            ine, out = t
            assert out in g
            # detect `else`
            has_else = False
            for u, d in ine:
                stmt = d.get('stmt')
                if isinstance(stmt, Else):
                    has_else = True
                    self_else = stmt
                # option cannot start with goto (= contraction)
                assert stmt != 'goto'
            # who owns this else ?
            if has_else:
                if len(ine) == 1:
                    assert not self_has_else, option
                    self_has_else = True
                    if not syntactic_else:
                        assert not option_has_else, option
                elif len(ine) > 1:
                    option_has_else = True
                    if not syntactic_else:
                        assert not self_has_else, option
                else:
                    raise Exception('option with no in edges')
            # collect in edges, except for own `else`
            if not (has_else and self_has_else):
                edges.extend(ine)
            else:
                else_ine = ine  # keep for later
            # forward edges
            # goto from last option node to target node
            g.add_edge(out, target, stmt='goto')
            # backward edges
            if self.type == 'if':
                continue
            for u, d in ine:
                g.add_edge(target, u, **d)
        # handle else
        if self_has_else and not option_has_else:
            self_else.other_guards = [d['stmt'] for v, d in edges]
            # add back the `else` edge
            edges.extend(else_ine)
        # what is the exit node ?
        if self.type == 'if':
            out = target
        elif self.type == 'do':
            out = od_exit
        else:
            raise Exception('Unknown type: {t}'.format(t=out))
        # is not itself an option guard ?
        if option_guard:
            logger.debug('is option guard')
        if self.type == 'do' and option_guard is None:
            edge = (target, dict(stmt='goto'))
            in_edges = [edge]
        else:
            in_edges = edges
        logger.debug('in edges: {ie}, out: {out}\n'.format(
            ie=in_edges, out=out))
        logger.info('-- end flattening {t}'.format(t=self.type))
        assert out in g
        return in_edges, out


class Else(Node):
    def __init__(self):
        self.other_guards = None

    def __str__(self):
        return 'else'


class Break(Node):
    def __str__(self):
        return 'break'

    def to_pg(self, g, od_exit=None, **kw):
        if od_exit is None:
            raise Exception('Break outside repetition construct.')
        # like goto, but with: v = od_exit
        # context of od_tail is determined by do loop
        assert od_exit in g
        v = od_exit
        return goto_to_pg(g, v, **kw)


class Goto(Node):
    def __init__(self, label):
        self.label = label

    def __str__(self):
        return 'goto {l}'.format(l=self.label)

    def to_pg(self, g, context=None, **kw):
        v = _format_label(self.label)
        # ok, because source node context
        # is overwritten during contraction
        g.add_node(v, context=context)
        return goto_to_pg(g, v, context=context, **kw)


def goto_to_pg(g, v, option_guard=None, context=None, **kw):
    assert v in g
    if option_guard is None:
        stmt = 'goto'
    else:
        stmt = Bool('true')
    e = (v, dict(stmt=stmt))
    u = generate_unique_node()
    g.add_node(u, context=context)
    return [e], u


class Label(Node):
    def __init__(self, label, body):
        self.label = label
        self.body = body

    def to_str(self):
        return '{l}: {b}'.format(l=self.label, b=to_str(self.body))

    def to_pg(self, g, option_guard=None, context=None, **kw):
        if option_guard is not None:
            raise Exception('option guard cannot be labeled')
        # add label node, with context
        u = _format_label(self.label)
        g.add_node(u, context=context, labels={self.label})
        # flatten body
        t = self.body.to_pg(g, context=context, **kw)
        if t is None:
            raise Exception('Label of variable declaration.')
        ine, out = t
        assert out in g
        # make ine out edges of label node
        for v, d in ine:
            g.add_edge(u, v, **d)
        # appear like a goto (almost)
        e = (u, dict(stmt='goto'))
        return [e], out


def _format_label(label):
    return 'label_{l}'.format(l=label)


# TODO: check that referenced types exist, before adding typedef
# to symbol table

class VarDef(Node):
    def __init__(self, name, vartype, length=None,
                 visible=None, bitwidth=None,
                 msg_types=None, initval=None):
        self.name = name
        self.type = vartype
        if length is None:
            l = None
        else:
            l = eval(str(length))
            assert isinstance(l, int), l
        self.length = l
        self.visible = visible
        if bitwidth is not None:
            self.bitwidth = int(bitwidth.value)
        if vartype == 'bool':
            default_initval = Bool('false')
        else:
            default_initval = Integer('0')
        if initval is None:
            initval = Expression(default_initval)
        self.initial_value = initval
        # TODO message types

    def __repr__(self):
        return 'VarDef({t}, {v})'.format(t=self.type, v=self.name)

    def to_str(self):
        s = '{type} {varname}{len}'.format(
            type=self._type_str(),
            varname=self.name,
            len='[{n}]'.format(n=self.len) if self.len else '')
        return s

    def _type_str(self):
        return self.type

    def to_pg(self, g, **kw):
        # var declarations are collected before the process runs
        # man page: datatypes, p.405
        g.locals.add(self)
        return None

    def insert(self, symbol_table, pid):
        """Insert variable into table of symbols.

        @type symbol_table: L{SymbolTable}

        @type pid: int or None
        """
        t = self.type
        if t == 'chan':
            v = MessageChannel(self.len)
            # channels are always available globally
            # note how this differs from having global scope:
            # no name conflicts
            symbol_table.channels.add(v)
        elif t == 'mtype':
            raise NotImplementedError
        elif t in {'bit', 'bool', 'byte', 'pid', 'short', 'int'}:
            if self.len is None:
                v = DATATYPES[t]()
            else:
                v = [DATATYPES[t]() for i in xrange(self.len)]
        elif t == 'unsigned':
            n = self.bitwidth

            class Unsigned(ctypes.Structure):
                _fields_ = [('value', ctypes.c_uint, n)]

            if self.len is None:
                v = Unsigned()
            else:
                v = [Unsigned() for i in xrange(self.len)]
        else:
            raise TypeError('unknown type "{t}"'.format(t=t))
        # global scope ?
        if pid is None:
            d = symbol_table.globals
        else:
            d = symbol_table.locals[pid]
        name = self.name
        if name in d:
            raise Exception('variable "{name}" is already defined'.format(
                            name=name))
        else:
            d[name] = v


class SymbolTable(object):
    """Variable, user data and message type definitions.

    Attributes:

      - `globals`: `dict` of global vars
      - `locals`: `dict` of `dicts`, keys of outer `dict` are pids
      - `channels`: `dict` of global lists for channels
      - `pids`: map from:

          pid integers

          to:

            - name of proctype (name)
            - current value of program counter (pc)

      - `types`: `dict` of data types

    pids are non-negative integers.
    The type name "mtype" corresponds to a message type.
    """

    def __init__(self):
        # see Def. 7.6, p.157
        self.exclusive = None
        self.handshake = None
        self.timeout = False
        self.else_ = False
        self.stutter = False
        # tables of variables
        self.globals = dict()
        self.channels = set()
        self.locals = dict()
        self.pids = dict()
        self.types = DATATYPES

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        assert(isinstance(other, SymbolTable))
        if self.globals != other.globals:
            return False
        if self.channels != other.channels:
            return False
        if set(self.locals) != set(other.locals):
            return False
        for pid, d in self.locals.iteritems():
            if d != other.locals[pid]:
                return False
        if set(self.pids) != set(other.pids):
            return False
        for pid, d in self.pids.iteritems():
            q = other.pids[pid]
            if d['name'] != q['name']:
                return False
            if d['pc'] != q['pc']:
                return False
        return True

    def __str__(self):
        return (
            'globals: {g}\n'
            'channels: {c}\n'
            'pids: {p}\n\n'
            'types: {t}\n'
            'locals: {l}\n'
            'exclusive: {e}\n').format(
                g=self.globals,
                l=self.locals,
                p=pprint.pformat(self.pids, width=15),
                t=self.types,
                e=self.exclusive,
                c=self.channels)

    def copy(self):
        new = SymbolTable()
        # auxiliary
        new.exclusive = self.exclusive
        new.handshake = self.handshake
        new.timeout = self.timeout
        new.else_ = self.else_
        new.stutter = self.stutter
        # copy symbols
        new.globals = copy.deepcopy(self.globals)
        new.channels = copy.deepcopy(self.channels)
        new.locals = copy.deepcopy(self.locals)
        new.pids = {k: {'name': d['name'],
                        'pc': d['pc']}
                    for k, d in self.pids.iteritems()}
        new.types = self.types
        return new


class MessageChannel(object):
    def __init__(self, nslots):
        self.nslots = nslots
        self.contents = list()

    def send(self, x):
        if len(self.contents) < self.nslots:
            self.contents.append(x)
        else:
            raise Exception('channel {name} is full'.format(
                            name=self.name))

    def receive(self, x=None, random=False, rm=True):
        c = self.contents
        i = 0
        if x and random:
            i = c.index(x)
        m = c[i]
        if rm:
            c.pop(i)
        return m


class TypeDef(Node):
    def __init__(self, name, decls):
        self.name = name
        self.decls = decls

    def __str__(self):
        return 'typedef {name} {\ndecls\n}'.format(
            name=self.name, decls=to_str(self.decls))

    def exe(self, t):
        t.types[self.name] = self


class MessageType(Node):
    def __init__(self, values, visible=None):
        self.values = values

    def __str__(self):
        return 'mtype {{ {values} }}'.format(values=self.values)

    def exe(self, t):
        t.types[self.name] = self


class Return(Node):
    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return to_str(self.expr)


class Run(Node):
    def __init__(self, func, args=None, priority=None):
        self.func = func
        self.args = args
        self.priority = priority

    def __str__(self):
        return 'run({f})'.format(f=self.func)


class Inline(Node):
    def __init__(self, name, args):
        self.name = name
        self.args = args


class Call(Node):
    def __init__(self, func, args):
        self.func = func
        self.args = args


class Assert(Node):
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return 'assert({expr})'.format(expr=repr(self.expr))


class Expression(Node):
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return 'Expression({expr})'.format(expr=repr(self.expr))

    def __str__(self):
        return to_str(self.expr)

    def eval(self, g, l):
        s = str(self)
        g = dict(g)
        for k, v in g.iteritems():
            if 'ctypes' in str(type(v)):
                g[k] = int(v.value)
            elif isinstance(v, list):
                for x in v:
                    if 'ctypes' in str(type(x)):
                        v[v.index(x)] = int(x.value)
        l = dict(l)
        for k, v in l.iteritems():
            if 'ctypes' in str(type(v)):
                l[k] = int(v.value)
            elif isinstance(v, list):
                for x in v:
                    if 'ctypes' in str(type(x)):
                        v[v.index(x)] = int(x.value)

        v = eval(s, g, l)
        return v


class Assignment(Node):
    def __init__(self, var, value):
        self.var = var
        self.value = value

    def __repr__(self):
        return 'Assignment({var}, {val})'.format(
            var=repr(self.var), val=repr(self.value))

    def __str__(self):
        return '{var} = {val}'.format(var=self.var, val=self.value)

    def exe(self, g, l):
        logger.debug('Assign: {var} = {val}'.format(
                     var=self.var, val=self.value))
        s = self.to_str()
        og = g
        ol = l
        g = dict(g)
        for k, v in g.iteritems():
            if 'ctypes' in str(type(v)):
                g[k] = int(v.value)
            elif isinstance(v, list):
                for x in v:
                    if 'ctypes' in str(type(x)):
                        v[v.index(x)] = int(x.value)
        l = dict(l)
        for k, v in l.iteritems():
            if 'ctypes' in str(type(v)):
                l[k] = int(v.value)
            elif isinstance(v, list):
                for x in v:
                    if 'ctypes' in str(type(x)):
                        v[v.index(x)] = int(x.value)
        exec s in g, l
        for k in og:
            og[k] = g[k]
        for k in ol:
            ol[k] = l[k]


class Receive(Node):
    def __init__(self, varref, args=None):
        self.var = varref
        self.args = args

    def __str__(self):
        v = to_str(self.var)
        return 'Rx({v})'.format(v=v)


class Send(Node):
    def __init__(self, varref, args=None):
        self.varref = varref
        self.args = args

    def __str__(self):
        v = to_str(self.var)
        return 'Tx({v})'.format(v=v)


class Printf(Node):
    def __init__(self, s, args):
        self.s = s
        self.args = args

    def __str__(self):
        return 'printf()'.format(s=self.s, args=self.args)


class Operator(object):
    def __init__(self, operator, *operands):
        self.operator = operator
        self.operands = operands

    def __repr__(self):
        return 'Operator({op}, {xy})'.format(
            op=repr(self.operator),
            xy=', '.join(repr(x) for x in self.operands))

    def __str__(self):
        return '({op} {xy})'.format(
            op=self.operator,
            xy=' '.join(to_str(x) for x in self.operands))


class Binary(Operator):
    def __str__(self):
        return '({x} {op} {y})'.format(
            x=to_str(self.operands[0]),
            op=self.operator,
            y=to_str(self.operands[1]))


class Unary(Operator):
    pass


class Terminal(object):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return '{classname}({val})'.format(
            classname=type(self).__name__,
            val=repr(self.value))

    def __str__(self):
        return str(self.value)


class VarRef(Terminal):
    def __init__(self, name, index=None, extension=None):
        self.name = name
        if index is None:
            i = None
        else:
            i = index
        self.index = i
        self.extension = extension
        # used by some external methods
        self.value = name

    def __repr__(self):
        return 'VarRef({name}, {index}, {ext})'.format(
            name=repr(self.name),
            index=repr(self.index),
            ext=repr(self.extension))

    def __str__(self):
        if self.index is None:
            i = ''
        else:
            i = '[{i}]'.format(i=to_str(self.index))
        return '{name}{index}{ext}'.format(
            name=self.name,
            index=i,
            ext='' if self.extension is None else self.extension)


class Integer(Terminal):
    def __bool__(self):
        return bool(int(self.value))


class Bool(Terminal):
    def __init__(self, val):
        self.value = val.upper() == 'TRUE'

    def __bool__(self):
        return self.value

    def __repr__(self):
        return 'Bool({value})'.format(value=repr(self.value))

    def __str__(self):
        return str(self.value)


class RemoteRef(Terminal):
    def __init__(self, proctype, label, pid=None):
        self.proctype = proctype
        self.label = label
        self.pid = pid

    def __repr__(self):
        return 'RemoteRef({proc}, {label}, {pid})'.format(
            proc=self.proctype, label=self.label, pid=self.pid)

    def __str__(self):
        if self.pid is None:
            inst = ''
        else:
            inst = '[{pid}]'.format(pid=self.pid)
        return '{proc} {inst} @ {label}'.format(
            proc=self.proctype, inst=inst, label=self.label)


def dump_graph(g, fname='a.pdf', node_label='label',
               edge_label='label', relabel=False):
    """Write the program graph, annotated with formulae, to PDF file."""
    # map nodes to integers
    if relabel:
        mapping = {u: i for i, u in enumerate(g)}
        g = nx.relabel_nodes(g, mapping)
        inv_mapping = {v: k for k, v in mapping.iteritems()}
        s = list()
        s.append('mapping of nodes:')
        for k in sorted(inv_mapping):
            v = inv_mapping[k]
            s.append('{k}: {v}'.format(k=k, v=v))
        print('\n'.join(s))
    h = nx.MultiDiGraph()
    for u, d in g.nodes_iter(data=True):
        label = d.get(node_label, u)
        label = '"{label}"'.format(label=label)
        h.add_node(u, label=label)
    for u, v, d in g.edges_iter(data=True):
        label = d.get(edge_label, ' ')
        label = '"{label}"'.format(label=label)
        h.add_edge(u, v, label=label)
    pd = nx.to_pydot(h)
    pd.write_pdf(fname)
