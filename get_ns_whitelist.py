#!/usr/bin/env python

"""
This script creates a list of namespaces used by published packages.

In most cases, it is the same as the package name list, with few exceptions.
For example, NPM module @scope/package will be imported as
`require('@scope/package')`. However, in Python, module scikit-learn will
be imported as sklearn. So, *package name* `scikit-learn` is different from
namespace `sklearn`.

This list will be used to whitelist vocabulary tokens. We do so to filter out
local imports and WoC parsing artifacts. Examples:

* `pl:Project`: build script for Postgres
* `JS:com.unity.modules.vehicles`: looks to be parsed from an arbitrary .json
    file, not related to NPM
* `jl:"joinpath(dir, bubblesort"`: WoC import parsing artifact

"""

import argparse
import os

import pandas as pd

# array(['Alcatraz', 'Dub', 'Hex', 'Jam', 'Emacs', 'Bower', 'Sublime',
#        'Pub', 'NPM', 'Cargo', 'Pypi', 'Packagist', 'Rubygems', 'Hackage',
#        'Nimble', 'Maven', 'Go', 'Wordpress', 'NuGet', 'CPAN', 'CRAN',
#        'Meteor', 'Clojars', 'CocoaPods', 'Elm', 'Julia', 'PlatformIO',
#        'Atom', 'Inqlude', 'Homebrew', 'Carthage', 'Shards', 'SwiftPM',
#        'Haxelib', 'Puppet', 'Racket', 'PureScript'], dtype=object)
# From the ecosystems we use only Python and Java require parsing namespaces
LANGUAGES = {  # correspondence of libraries.io platforms to WoC imports
    # 'Pypi': 'PY',  # Python
    'CRAN': 'R',
    'NPM': 'JS',  # Javascript
    'Go': 'Go',
    # 'Maven': 'java',  # needs special parsing
    'Julia': 'jl',  # Julia
    'CPAN': 'pl',  # Perl
}


def get_python_namespaces(package_names):
    """Getting this data takes up to 20 hours of processing.
    It is better to cache the result"""
    cached_fname = 'python_namespaces.csv'
    if os.path.isfile(cached_fname):
        namespaces = pd.read_csv(
            cached_fname, index_col=0, squeeze=True, header=None,
            names=['package', 'namespaces'])
        namespaces = namespaces.apply(lambda x: str(x).split(','))
        return namespaces

    from stecosystems import pypi
    from stutils import mapreduce  # TODO: replace with joblib

    def get_module(i, package_name):
        try:
            namespaces = pypi.Package(package_name).modules()
        except:
            # Package either does not exist or its setup.py has errors
            namespaces = []
        return namespaces or [package_name]

    # higher number of workers hungs Docker
    namespaces = mapreduce.map(get_module, package_names, num_workers=8)
    namespaces = pd.Series(namespaces.values, index=list(package_names))
    namespaces.apply(lambda x: ','.join(str(s) for s in x)).to_csv(cached_fname)
    return namespaces


def main(projects_file_path, save_path):
    df = pd.read_csv(projects_file_path, index_col=False,
                     usecols=['Platform', 'Name'],
                     dtype={'Platform': str, 'Name': str})
    # strip github.com/.../ prefix from Go packages
    df.loc[df['Platform'] == 'Go', 'Name'] = df.loc[
        df['Platform'] == 'Go', 'Name'].apply(
        lambda namespace: namespace.rsplit('/', 1)[-1])

    for platform, language in LANGUAGES.items():
        fname = language + '_whitelist.csv'
        print("writing", fname)
        combined_packages = set(
            str(ns) for ns in df.loc[df['Platform'] == platform, 'Name'])
        with open(os.path.join(save_path, fname), 'wb') as fh:
            fh.write('\n'.join(combined_packages))

    # Python
    python_pkgnames = df.loc[df['Platform'] == 'Pypi', 'Name']
    python_namespaces = get_python_namespaces(python_pkgnames)
    # namespaces are multilevel, need to strip 2+ levels
    # e.g. lib3to2.fixes -> lib3to2
    # also, remove all the invalid ones (can't start with a digit)
    python_namespaces = python_namespaces.apply(
        lambda x: [s.split('.', 1)[0] for s in x if s and not s[0].isdigit()])
    # remove non-unique ones
    combined_namespaces = set().union(*python_namespaces)

    print("writing", 'PY_whitelist.csv')
    with open(os.path.join(save_path, 'PY_whitelist.csv'), 'wb') as fh:
        fh.write('\n'.join(combined_namespaces))

    # TODO: java
    # the idea here is to map classes back to package name,
    # or come up with a list of shortest prefix for each package
    # non-trivial in both cases


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build whitelist of namespaces')
    parser.add_argument('packages_file',
                        type=argparse.FileType('r'),
                        help='projects_with_repository* file from libraries.io')
    parser.add_argument('--save-path', default='ns_whitelist',
                        help='Directory to save whitelists to')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Log progress to stderr")
    args = parser.parse_args()

    main(args.packages_file, args.save_path)

