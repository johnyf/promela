import logging
import networkx as nx
import networkx.algorithms.isomorphism as iso
from nose.tools import assert_raises
from promela import ast, yacc


logger = logging.getLogger(__name__)
logger.setLevel('WARNING')
log = logging.getLogger('promela.yacc')
log.setLevel(logging.ERROR)
h = logging.StreamHandler()
log = logging.getLogger('promela.ast')
log.setLevel('WARNING')
log.addHandler(h)


parser = yacc.Parser()


def parse_proctype_test():
    s = '''
    active [3] proctype main(){
        int x;
    }
    '''
    tree = parser.parse(s)
    assert isinstance(tree, ast.Program)
    assert len(tree) == 1
    p = tree[0]
    assert isinstance(p, ast.Proctype)
    assert p.name == 'main'
    assert isinstance(p.body, ast.Sequence)
    assert p.active == 3, type(p.active)
    assert p.args is None
    assert p.priority is None
    assert p.provided is None


def parse_if_test():
    s = '''
    proctype p (){
        if
        :: skip
        fi
    }
    '''
    tree = parser.parse(s)
    assert isinstance(tree, ast.Program)
    assert len(tree) == 1
    proc = tree[0]
    assert isinstance(proc, ast.Proctype)
    assert isinstance(proc.body, ast.Sequence)
    assert len(proc.body) == 1
    if_block = proc.body[0]
    assert isinstance(if_block, ast.Options)
    assert if_block.type == 'if'
    options = if_block.options
    assert isinstance(options, list)
    assert len(options) == 1
    opt0 = options[0]
    assert isinstance(opt0, ast.Sequence)
    assert len(opt0) == 1
    assert isinstance(opt0[0], ast.Expression)
    e = opt0[0].expr
    assert isinstance(e, ast.Bool)
    assert e.value, e


def parse_do_multiple_options_test():
    s = '''
    proctype p (){
        do
        :: x -> x = x + 1;
        :: (y == 0) -> y = x; y == 1;
        od
    }
    '''
    tree = parser.parse(s)
    assert isinstance(tree, ast.Program)
    assert len(tree) == 1
    proc = tree[0]
    assert isinstance(proc, ast.Proctype)
    assert isinstance(proc.body, ast.Sequence)
    assert len(proc.body) == 1
    do_block = proc.body[0]
    assert isinstance(do_block, ast.Options)
    assert do_block.type == 'do'
    options = do_block.options
    assert isinstance(options, list)
    assert len(options) == 2
    opt = options[0]
    assert isinstance(opt, ast.Sequence)
    assert len(opt) == 2
    assert isinstance(opt[0], ast.Expression)
    assert isinstance(opt[1], ast.Assignment)

    opt = options[1]
    assert isinstance(opt, ast.Sequence)
    assert len(opt) == 3
    assert isinstance(opt[0], ast.Expression)
    assert isinstance(opt[1], ast.Assignment)
    assert isinstance(opt[2], ast.Expression)


def if_one_option_pg_test():
    s = '''
    proctype p (){
        if
        :: skip
        fi
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([(0, 1)])
    assert nx.is_isomorphic(g, h)


def if_two_options_pg_test():
    s = '''
    proctype p(){
        if
        :: true
        :: false
        fi
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([(0, 1), (0, 1)])
    assert nx.is_isomorphic(g, h)


def do_one_option_pg_test():
    s = '''
    proctype p(){
        do
        :: skip
        od
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([(0, 0)])
    assert nx.is_isomorphic(g, h)


def do_two_options_pg_test():
    s = '''
    proctype p(){
        do
        :: true
        :: false
        od
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([(0, 0), (0, 0)])
    assert nx.is_isomorphic(g, h)


def nested_if_pg_test():
    s = '''
    proctype p(){
        bit x;
        if
        :: if
           :: true
           :: false
           fi
        :: x
        fi
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([(0, 1), (0, 1), (0, 1)])
    assert nx.is_isomorphic(g, h)


def nested_if_not_guard_pg_test():
    s = '''
    proctype p(){
        bit x;
        if
        :: true;
           if
           :: true
           :: false
           fi
        :: x
        fi
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([(0, 1), (0, 2), (2, 1), (2, 1)])
    assert nx.is_isomorphic(g, h)


def doubly_nested_if_pg_test():
    s = '''
    proctype p(){
        bit x;
        if
        :: if
           :: true
           :: if
              :: true
              :: skip
              fi
           :: false
           fi
        :: if
           :: if
              :: true
              :: false
              fi
           fi
        fi
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    for i in xrange(6):
        h.add_edge(0, 1)
    assert nx.is_isomorphic(g, h)


def nested_do_pg_test():
    s = '''
    proctype p(){
        bit x;
        if
        :: do
           :: true
           :: false
           od
        :: x
        fi
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([(0, 1), (0, 2), (0, 2), (2, 2), (2, 2)])
    assert nx.is_isomorphic(g, h)


def nested_do_not_guard_pg_test():
    s = '''
    proctype p(){
        bit x;
        if
        :: true;
           do
           :: true
           :: false
           od
        :: x
        fi
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([(0, 1), (0, 2), (2, 2), (2, 2)])
    assert nx.is_isomorphic(g, h)


def combined_if_do_program_graph_test():
    s = '''
    active proctype p(){
        int x, y, z;
        if /* 0 */
        ::  do
            :: x = 1; /* 2 */
               y == 5 /* 1 */

            :: z = 3; /* 3 */
               skip /* 1 */

            ::  if
                :: z = (3 - x) * y; /* 4 */
                   true; /* 5 */
                   y = 3 /* 1 */

                :: true /* 1 */
                fi
            od
          /* 1 */

        :: true; /* 6 */
           if
           :: true /* 7 */

           :: true -> /* 8 */
              x = y /* 7 */

           fi
        fi
        /* 7 */
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([
        (0, 2), (0, 3), (0, 4), (0, 1),
        (2, 1), (3, 1), (5, 1),
        (1, 2), (1, 3), (1, 4), (1, 1),
        (4, 5),
        # false; if ...
        (0, 6),
        (6, 7), (6, 8), (8, 7)])
    dump(g, node_label=None)
    assert iso.is_isomorphic(g, h)


def invalid_label_pg_test():
    s = '''
    proctype p(){
        do
        :: S0: x = 1;
        od
    }
    '''
    tree = parser.parse(s)
    with assert_raises(Exception):
        tree[0].to_pg()


def goto_pg_test():
    s = '''
    proctype p(){
        bit x;
        x = 1;
        goto S0;
        x = 2;
        S0: x = 3;
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([(0, 1), (1, 2)])
    assert nx.is_isomorphic(g, h)


def double_goto_pg_test():
    s = '''
    proctype p(){
        bit x;
        x = 1;
        goto S0;
        x = 2;
        S0: goto S1;
        x = 3;
        S1: skip
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([(0, 1), (1, 2)])
    assert nx.is_isomorphic(g, h)


def goto_inside_if_pg_test():
    s = '''
    proctype p(){
        bit x;
        if
        :: true; goto S0; x = 1;
        :: x = 3; false
        fi;
        S0: skip
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([(0, 1), (0, 2), (2, 1), (1, 3)])
    assert nx.is_isomorphic(g, h)


def goto_loop_pg_test():
    s = '''
    proctype p(){
        bit x;
        S0: if
        :: true; goto S0; x = 1;
        :: x = 3;
        fi;
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([(0, 1), (0, 0)])
    assert nx.is_isomorphic(g, h)


def goto_self_loop_pg_test():
    s = '''
    proctype p(){
        S0: goto S1;
        S1: goto S0
    }
    '''
    tree = parser.parse(s)
    with assert_raises(AssertionError):
        tree[0].to_pg()


def break_pg_test():
    s = '''
    proctype p(){
        bit x;
        do
        :: true; x = 1;
        :: break; x = 3;
        od
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([(0, 1), (1, 0), (0, 2)])
    assert nx.is_isomorphic(g, h)


def nested_break_pg_test():
    s = '''
    proctype p(){
        bit x;
        /* 0 */
        do
        :: true; /* 2 */
           x == 1; /* 0 */

        :: do
           :: x == 2;
              break /* 0 */

           :: false; /* 4 */
              x == 3 /* 5 */

           od
           /* 5 */

        :: break; /* 1 */
           x == 4;

        od
        /* 1 */
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([
        (0, 1),
        (0, 2), (2, 0),
        (0, 0),
        (0, 4), (4, 5),
        (5, 4), (5, 0)])
    assert nx.is_isomorphic(g, h)


def atomic_pg_test():
    s = '''
    proctype p(){
        bit x;
        x = 1;
        atomic { x = 2; }
        x = 3;
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_node(0, context=None)
    h.add_node(1, context=None)
    h.add_node(2, context='atomic')
    h.add_node(3, context=None)
    h.add_edges_from([(0, 1), (1, 2), (2, 3)])
    nm = lambda x, y: x['context'] == y['context']
    gm = iso.GraphMatcher(g, h, node_match=nm)
    assert gm.is_isomorphic()


def do_atomic_dissapears_pg_test():
    s = '''
    proctype p(){
        bit x, y;
        /* 0 */
        do
        :: true; /* 3 */
           atomic { x = 2; goto S0; /* 1 */ y = 1}
        :: x == 1; /* 4 */ y == 2; /* 0 */
        od;
        x = 3;
        /* 1 */
        S0: skip
        /* 2 */
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([
        (0, 3), (3, 1), (1, 2),
        (0, 4), (4, 0)])
    for u in h:
        h.add_node(u, context=None)
    nm = lambda x, y: x['context'] == y['context']
    gm = iso.GraphMatcher(g, h, node_match=nm)
    assert gm.is_isomorphic()


def do_atomic_pg_test():
    s = '''
    proctype p(){
        bit x, y;
        /* 0 */
        do
        :: true; /* 1 */
           atomic { x = 2; /* 2 */
           y = 1; goto S0; } /* 3 */
        :: x == 1; /* 4 */ y == 2; /* 0 */
        od;
        x = 3;
        /* 3 */
        S0: skip
        /* 5 */
    }
    '''
    tree = parser.parse(s)
    g = tree[0].to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([
        (0, 1), (1, 2), (2, 3),
        (3, 5), (0, 4), (4, 0)])
    for u in h:
        h.add_node(u, context=None)
    h.add_node(2, context='atomic')
    nm = lambda x, y: x['context'] == y['context']
    gm = iso.GraphMatcher(g, h, node_match=nm)
    assert gm.is_isomorphic()


def ltl_block_test():
    s = '''
    bit x, y, c;

    proctype p(){
        if
        :: x;
        fi
    }

    ltl { (x == 1) && []<>(y != 2) && <>[](c == 1) && (x U y) }
    '''
    tree = parser.parse(s)
    assert len(tree) == 3, repr(tree)
    decl, p, ltl = tree
    assert isinstance(p, ast.Proctype)
    assert isinstance(ltl, ast.LTL)
    s = str(ltl.formula)
    assert s == (
        '((((x == 1) && ([] (<> (y != 2)))) && '
        '(<> ([] (c == 1)))) && (x U y))'), s


def test_else():
    s = '''
    proctype p(){
        byte x;
        do
        :: x == 0
        :: x == 1
        :: else
        od
    }
    '''
    (proc,) = parser.parse(s)
    g = proc.to_pg()
    dump(g)
    for u, v, d in g.edges_iter(data=True):
        c = d['stmt']
        if isinstance(c, ast.Else):
            print c.other_guards


def test_nested_else():
    s = '''
    proctype p(){
        byte x;
        do
        ::
            if
            :: false
            :: else
            fi
        od
    }
    '''
    (proc,) = parser.parse(s)
    g = proc.to_pg()
    dump(g)
    h = nx.MultiDiGraph()
    h.add_edges_from([(0, 0), (0, 0)])
    for u in h:
        h.add_node(u, context=None)
    nm = lambda x, y: x['context'] == y['context']
    gm = iso.GraphMatcher(g, h, node_match=nm)
    assert gm.is_isomorphic()


def test_double_else():
    s = '''
    proctype foo(){
        bit x;
        do
        ::
            if
            :: x
            :: else
            fi
        :: else
        od
    }
    '''
    # syntactic else = Promela language definition
    (proc,) = parser.parse(s)
    with assert_raises(AssertionError):
        proc.to_pg()
    # different from Promela language definition
    g = proc.to_pg(syntactic_else=True)
    active_else = 0
    off_else = 0
    for u, v, d in g.edges_iter(data=True):
        stmt = d['stmt']
        if isinstance(stmt, ast.Else):
            other = stmt.other_guards
            if other is None:
                off_else += 1
            else:
                active_else += 1
                assert len(other) == 1, other
                (other_stmt,) = other
                s = str(other_stmt)
                assert s == 'x', s
    assert active_else == 1, active_else
    assert off_else == 1, off_else


def test_pg_node_order():
    s = '''
    proctype foo(){
        bit x;
        if
        ::
            do
            :: x > 2; x = 1
            :: else; break
            od;
            x = 1
        :: x = 2
        fi
    }
    '''
    (proc,) = parser.parse(s)
    g = proc.to_pg()
    dump(g)
    # Final indexing depends on the
    # aux goto nodes created and the contraction order.
    # The latter depend on the intermediate indexing,
    # which is fixed syntactically
    # (see `generate_unique_node`).
    edges = {(1, 2), (1, 3), (2, 0), (3, 1),
             (4, 0), (4, 2), (4, 3)}
    assert set(g) == set(xrange(5)), g.nodes()
    assert set(g.edges_iter()) == edges, g.edges()


def test_labels():
    s = '''
    active proctype foo(){
        progress:
        do
        :: true
        od
    }
    '''
    (proc,) = parser.parse(s)
    g = proc.to_pg()
    for u, d in g.nodes_iter(data=True):
        print d.get('label')


def test_remote_ref():
    s = '''
    proctype foo(){
        bar @ critical
    }
    '''
    (proc,) = parser.parse(s)
    g = proc.to_pg()
    (e,) = g.edges(data=True)
    u, v, d = e
    s = d['stmt']
    assert isinstance(s, ast.Expression), s
    ref = s.expr
    assert isinstance(ref, ast.RemoteRef), ref
    assert ref.proctype == 'bar', ref.proctype
    assert ref.label == 'critical', ref.label
    assert ref.pid is None, ref.pid


def dump(g, fname='g.pdf', node_label='context'):
    if logger.getEffectiveLevel() >= logging.DEBUG:
        return
    # map nodes to integers
    ast.dump_graph(
        g, fname, node_label=node_label, edge_label='stmt')


if __name__ == '__main__':
    test_labels()
