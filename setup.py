"""Installation script."""
from setuptools import setup
# inline:
# from promela import yacc


DESCRIPTION = (
    'Parser and abstract syntax tree for the Promela modeling language.')
README = 'README.md'
PROJECT_URLS = {
    'Bug Tracker':
        'https://github.com/johnyf/promela/issues',
    'Documentation':
        'https://github.com/johnyf/promela/blob/main/doc.md',}
VERSION_FILE = 'promela/_version.py'
MAJOR = 0
MINOR = 0
MICRO = 4
VERSION = '{major}.{minor}.{micro}'.format(
    major=MAJOR, minor=MINOR, micro=MICRO)
VERSION_FILE_TEXT = (
    '# This file was generated from setup.py\n'
    "version = '{version}'\n").format(version=VERSION)
PYTHON_REQUIRES = '>=3.6'
INSTALL_REQUIRES = [
    'networkx >= 2.0',
    'ply >= 3.4, <= 3.10',
    'pydot >= 1.1.0']
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Topic :: Scientific/Engineering']
KEYWORDS = [
    'promela', 'parser', 'syntax tree', 'ply', 'lex', 'yacc']


def build_parser_table():
    from promela import yacc
    tabmodule = yacc.TABMODULE.split('.')[-1]
    outputdir = 'promela/'
    parser = yacc.Parser()
    parser.build(tabmodule, outputdir=outputdir, write_tables=True)


if __name__ == '__main__':
    with open(VERSION_FILE, 'w') as f:
        f.write(VERSION_FILE_TEXT)
    try:
        build_parser_table()
    except ImportError:
        print('WARNING: `promela` could not cache parser tables '
              '(ignore this if running only for "egg_info").')
    with open(README) as f:
        long_description = f.read()
    setup(
        name='promela',
        version=VERSION,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Ioannis Filippidis',
        author_email='jfilippidis@gmail.com',
        url='https://github.com/johnyf/promela',
        project_urls=PROJECT_URLS,
        license='BSD',
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        tests_require=['pytest'],
        packages=['promela'],
        package_dir={'promela': 'promela'},
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS)
