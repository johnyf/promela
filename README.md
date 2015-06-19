[![Build Status][build_img]][travis]
[![Coverage Status][coverage]][coveralls]


About
=====

A parser for the Promela modeling language.
[PLY](https://pypi.python.org/pypi/ply/3.4) (Python `lex`-`yacc`) is used to generate the parser.
Classes for a Promela abstract tree are included and used for representing the result of parsing.
A short tutorial can be found in the file `doc.md`.
To install:

```
pip install promela
```


License
=======
[BSD-3](http://opensource.org/licenses/BSD-3-Clause), see `LICENSE` file.


[build_img]: https://travis-ci.org/johnyf/promela.svg?branch=master
[travis]: https://travis-ci.org/johnyf/promela
[coverage]: https://coveralls.io/repos/johnyf/promela/badge.svg?branch=master
[coveralls]: https://coveralls.io/r/johnyf/promela?branch=master