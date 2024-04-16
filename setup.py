"""Installation script."""
import setuptools
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
VERSION = f'{MAJOR}.{MINOR}.{MICRO}'
VERSION_FILE_TEXT = (
    '# This file was generated from setup.py\n'
    f"version = '{VERSION}'\n")
PYTHON_REQUIRES = '>=3.9'
INSTALL_REQUIRES = [
    'networkx >= 2.0',
    'ply >= 3.4, <= 3.10']
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering']
KEYWORDS = [
    'promela', 'parser', 'syntax tree', 'ply', 'lex', 'yacc']


def _build_parser_table():
    if not _ply_is_installed():
        return
    from promela import yacc
    tabmodule = yacc.TABMODULE.rpartition('.')[-1]
    outputdir = 'promela/'
    parser = yacc.Parser()
    parser.build(
        tabmodule,
        outputdir=outputdir,
        write_tables=True)


def _ply_is_installed():
    try:
        import ply
    except ImportError:
        print('WARNING: `promela` could not cache parser tables '
              '(ignore this if running only for "egg_info").')
        return False
    return True


def run_setup():
    with open(VERSION_FILE, 'w') as f:
        f.write(VERSION_FILE_TEXT)
    _build_parser_table()
    with open(README) as f:
        long_description = f.read()
    setuptools.setup(
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


if __name__ == '__main__':
    run_setup()
