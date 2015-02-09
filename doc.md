This package provides a lexer, parser, and abstract syntax tree (AST) for the [Promela](http://en.wikipedia.org/wiki/Promela) modeling language.
The lexer and parser are generated using [PLY](https://pypi.python.org/pypi/ply/3.4) (Python `lex`-`yacc`).
The [grammar](http://spinroot.com/spin/Man/grammar.html) is based on that used in the [SPIN](http://spinroot.com/spin/whatispin.html) model checker (in the files `spin.y` and `spinlex.c` of SPIN's source distribution), with modifications where needed.

To instantiate a Promela parser:

```python
from promela.yacc import Parser
parser = Parser()
```

Then Promela code, as a string, can be parsed by:

```python
s = '''
active proctype foo(){
	int x;
	do
	:: x = x + 1;
	od
}
'''
program = parser.parse(s)
```

then

```python
>>> print(program)
active [1]  proctype foo(){
	int x
	do
	:: x = (x + 1)
	od;
}
```

The parser returns the result as an abstract syntax tree using classes from the `promela.ast` module.
The top production rule returns a `Program` instance, which itself is a `list`.
The contents of this list

There are two categories of AST classes: those that represent control flow constructs:

- `Proctype`, (`Init`, `NeverClaim`), `Node`, (`Expression`, `Assignment`, `Assert`, `Options` (if, do), `Else`, `Break`, `Goto`, `Label`, `Call`, `Return`, `Run`), `Sequence`

and those that represent only syntax inside an expression:

- `Terminal`, (`VarRef`, `Integer`, `Bool`), `Operator`, (`Binary`, `Unary`)

The classes in parentheses are subclasses of the last class preceding the parentheses.
Each control flow class has a method `to_pg` that recursively converts the abstract syntax tree to a program graph.

A program graph is a directed graph whose edges are labeled with statements from the program.
Nodes represent states of the program.
Note the difference with a control flow graph, whose nodes are program statements and edges are program states.
AST node classes correspond to nodes of the control flow graph and edges of the program graph (possibly with branching).

For some node classes like `Expression` and `Assignment`, the `to_pg` method returns themselves, a.
Almost all statements are represented as either an `Expression` or an `Assignment`.
These label edges in the program graph, using the edge attribute `"stmt"`.

The program graph is represented as a [multi-digraph](http://en.wikipedia.org/wiki/Multigraph) using [`networkx.MultiDiGraph`](https://networkx.github.io/documentation/latest/reference/classes.multidigraph.html).
A multi-digraph is necessary, because two nodes in the program graph may be connected by two edges, each edge labeled with a different statement.
For example, this is the case in the code fragment:

```promela
bit x;
do
:: x == 0
:: x == 1
od
```

The above defines a program graph with a single node and two self-loops, one labeled with the statement `x == 0` and another with the statement `x == 1`.
These two statements here are guards, so they only determine whether the edge can be traversed, without affecting the program's data state (variable values).

Program graph nodes are labeled with a `"context"` attribute that can take the values:
- `"atomic"`
- `"d_step"`
- `None`
The values `"atomic"` and `"d_step"` signify that the state is inside an atomic or deterministic step block.

Continuing our earlier example:

```python
>>> g = program[0].to_pg()
>>> g.nodes(data=True)
[(0, {'context': None}), (1, {'context': None})]
>>> g.edges(data=True)
[(0, 1, {'stmt': Assignment(
    VarRef('x', None, None),
    Expression(
        Operator('+',
            VarRef('x', None, None),
            Integer('1'))))}),
 (1, 0, {'stmt': Expression(
    Operator('==',
        VarRef('x', None, None),
        Integer('2')))})]
```
