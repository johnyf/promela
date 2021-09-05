[![Build Status][build_img]][github_actions]


About
=====

A parser for the [Promela modeling language](https://en.wikipedia.org/wiki/Promela).
[PLY](https://pypi.org/project/ply/3.4/) (Python `lex`-`yacc`) is used to
generate the parser. Classes for a Promela abstract tree are included and used
for representing the result of parsing.

A short tutorial can be found in the file [`doc.md`](
    https://github.com/johnyf/promela/blob/main/doc.md).
To install:

```
pip install promela
```


License
=======

[3-clause BSD](https://opensource.org/licenses/BSD-3-Clause),
see the file `LICENSE`.


[build_img]: https://github.com/johnyf/promela/actions/workflows/main.yml/badge.svg?branch=main
[github_actions]: https://github.com/johnyf/promela/actions
