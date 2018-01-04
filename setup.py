from setuptools import setup
# inline:
# from promela import yacc


description = (
    'Parser and abstract syntax tree for the Promela modeling language.')
README = 'README.md'
VERSION_FILE = 'promela/_version.py'
MAJOR = 0
MINOR = 0
MICRO = 3
version = '{major}.{minor}.{micro}'.format(
    major=MAJOR, minor=MINOR, micro=MICRO)
s = (
    '# This file was generated from setup.py\n'
    "version = '{version}'\n").format(version=version)
install_requires = [
    'networkx >= 2.0',
    'ply >= 3.4',
    'pydot >= 1.1.0']
classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering']
keywords = [
    'promela', 'parser', 'syntax tree', 'ply', 'lex', 'yacc']


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
        tests_require=['nose'],
        packages=['promela'],
        package_dir={'promela': 'promela'},
        classifiers=classifiers,
        keywords=keywords)
