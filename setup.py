import pip
from setuptools import setup


description = (
    'Parser and abstract syntax tree for the Promela modeling language.')
README = 'README.md'
VERSION_FILE = 'promela/version.py'
MAJOR = 0
MINOR = 0
MICRO = 1
version = '{major}.{minor}.{micro}'.format(
    major=MAJOR, minor=MINOR, micro=MICRO)
s = (
    '# This file was generated from setup.py\n'
    "version = '{version}'\n").format(version=version)


def build_parser_table():
    from promela import yacc
    tabmodule = yacc.TABMODULE.split('.')[-1]
    outputdir = 'promela/'
    parser = yacc.Parser()
    parser.build(tabmodule, outputdir=outputdir, write_tables=True)


if __name__ == '__main__':
    pip.main(['install', 'ply == 3.4'])
    pip.main(['install',
              '--allow-unverified', 'networkx>=2.0dev',
              'https://github.com/networkx/networkx/archive/master.zip'])
    with open(VERSION_FILE, 'w') as f:
        f.write(s)
    build_parser_table()
    setup(
        name='promela',
        version=version,
        description=description,
        long_description=open(README).read(),
        author='Ioannis Filippidis',
        author_email='jfilippidis@gmail.com',
        url='https://github.com/johnyf/promela',
        license='BSD',
        install_requires=['networkx >= 2.0.dev', 'ply == 3.4'],
        extras_require={'dot': 'pydot'},
        tests_require=['nose'],
        packages=['promela'],
        package_dir={'promela': 'promela'})
