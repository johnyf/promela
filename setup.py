from setuptools import setup
# inline:
# from promela import yacc


description = (
    'Parser and abstract syntax tree for the Promela modeling language.')
README = 'README.md'
VERSION_FILE = 'promela/_version.py'
MAJOR = 0
MINOR = 0
MICRO = 1
version = '{major}.{minor}.{micro}'.format(
    major=MAJOR, minor=MINOR, micro=MICRO)
s = (
    '# This file was generated from setup.py\n'
    "version = '{version}'\n").format(version=version)
install_requires = [
    'ply >= 3.4',
    'networkx >= 1.9.1']


def build_parser_table():
    from promela import yacc
    tabmodule = yacc.TABMODULE.split('.')[-1]
    outputdir = 'promela/'
    parser = yacc.Parser()
    parser.build(tabmodule, outputdir=outputdir, write_tables=True)


if __name__ == '__main__':
    with open(VERSION_FILE, 'w') as f:
        f.write(s)
    try:
        build_parser_table()
    except ImportError:
        print('WARNING: `promela` could not cache parser tables '
              '(ignore this if running only for "egg_info").')
    setup(
        name='promela',
        version=version,
        description=description,
        long_description=open(README).read(),
        author='Ioannis Filippidis',
        author_email='jfilippidis@gmail.com',
        url='https://github.com/johnyf/promela',
        license='BSD',
        install_requires=install_requires,
        extras_require={'dot': 'pydot'},
        tests_require=['nose'],
        packages=['promela'],
        package_dir={'promela': 'promela'})
