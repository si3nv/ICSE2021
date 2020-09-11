#!/usr/bin/env python

"""
Preprocessing
---------------
Supported languages:
    PY - Python
    R
    JS - Javascript
    Go
        includes github URLs
    ipy - Jupyter notebooks
    java
    J - Julia
    pl - Perl

On da4, run:
`./preprocess.py <lang> -v`

As a result, you should get these files:

<lang>_project_imports.csv - will not be used
<lang>_dev_commit_counts.csv - will be used to filter out bots
<lang>_namespace_counts_by_project.csv - will not be used
<lang>_namespace_counts_by_dev.csv - byproduct of filtering out infrequent NSs
<lang>_dev_year_imports.csv - training data
<lang>_dev_imports.csv - training data
<lang>_projects_imports.csv - training data for comparison to import2vec
<lang>_projects_year_imports.csv - intermediate comparison model data
<lang>_vocab.csv

Gory details:
This script does preliminary clean-up of WoC dependencies data
- leave only repository and author email
- take top level imorts only
- filter out built-ins

Input file format:
commit_sha;woc_proj_id;time;Author Name <email@address.tld>;blob_sha;<imports>
    where imports are semicolon-separated imports including dots
    and time is a unix epoch time

Author strings and imports often contain symbols in different encodings,
often incompatible with utf8. Non-ascii imports are ignored.
Author emails are checked for valid utf8 and ignored otherwise.


Full list of maps:
CobthruMaps:    Cobol*
CsthruMaps :    C#
CthruMaps:      C (includes, like string.h)
ErlangthruMaps: Erlang*
FmlthruMaps:    ???
FthruMaps:      FPGA?
GothruMaps      Go
ipythruMaps     Python from Jupyter notebooks
javathruMaps    Java
jlthruMaps      Julia  (.jl files - local imports only?)
JSthruMaps      JavaScript (require() args, filter out local (eg ./misc.js))
LispthruMaps    Lisp*
LuathruMaps     Lua*
phpthruMaps     PHP*
plthruMaps      Perl (use X; need to filter builtins)
PYthruMaps      Python
rbthruMaps      Ruby*
RthruMaps       R
RustthruMaps    Rust (use X::Y::Z - use X only)
ScalathruMaps   Scala
SqlthruMaps     SQL*
SwiftthruMaps   Swift*

* - incomplete

"""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import csv
from datetime import datetime
import gzip
import logging
import os
import random
from typing import Dict

import pandas as pd

import utils

# key: 0..31
WOC_VERSION = 'Q'
ON_DA0 = os.environ.get('HOSTNAME') == 'da0.eecs.utk.edu'
INPUT_PATH = ('/data' if ON_DA0 else '/da0_data'
              ) + '/play/{lang}thruMaps/c2bPtaPkg{ver}{lang}.{key}.gz'
MAX_DATE = '9999/99/99'


def _dot_normalizer(namespace):
    return namespace.split('.', 1)[0]


NORMALIZERS = {  # java and JS don't require normalization (?)
    # import matplotlib.pyplot as plt
    'PY': _dot_normalizer,
    # import "github.com/owner/package"
    'Go': lambda namespace: namespace.rsplit('/', 1)[-1],
    # same as Python
    'ipy': _dot_normalizer,
    # import A.p
    'jl': _dot_normalizer,
    # use Math::Calc;  # looks like such imports are not parsed as of ver P
    'pl': lambda namespace: namespace.split('::', 1)[0],
}


def is_proper_utf(s):
    try:
        s.decode('utf8')
    except UnicodeDecodeError:
        return False
    return True


def clean_import(namespace):
    try:
        namespace.decode('ascii')
    except UnicodeDecodeError:
        return None
    return namespace.strip()


def get_author_email(author_string):
    author_email = author_string.rsplit('<', 1)[-1].split('>', 1)[0].strip()
    if is_proper_utf(author_email):
        return author_email
    return None


def preprocess(lang):
    normalizer = NORMALIZERS.get(lang)
    for key in range(32):
        path = INPUT_PATH.format(key=key, lang=lang, ver=WOC_VERSION)
        fh = gzip.open(path)
        logging.info("Processing %s", path)
        for line in fh:
            chunks = line.strip().split(';')

            commit_sha, project_id, unix_ts, author, blob_sha = chunks[:5]
            author_email = get_author_email(author)
            try:
                dt = datetime.fromtimestamp(int(unix_ts))
            except ValueError:  # invalid timestamps
                continue
            if not author_email:
                continue

            imports = chunks[5:]
            if normalizer:
                imports = [normalizer(i) for i in imports]
            imports = [clean_import(i) for i in imports]
            imports = set(i for i in imports if i is not None)
            if not imports:  # only invalid imports
                continue

            yield project_id, author_email, dt, imports
        fh.close()


def get_counts(data):
    logging.info("Namespace count by projects...")
    ns_counts = Counter()
    for name, adoption_date in data.items():
        ns_counts.update(adoption_date.keys())
    return pd.Series(ns_counts).sort_values(ascending=False)


def write_imports(lang, mode, imports_dict, val_keys, test_keys,
                  data_path='data'):
    # type: (str, str, Dict, set, set, str) -> None
    train_fname = os.path.join(
        data_path, '%s_%s_imports_train.csv' % (lang, mode))
    with open(train_fname, 'wb') as imports_fh:
        writer = csv.writer(imports_fh)
        for key, namespaces in imports_dict.items():
            if key in val_keys or key in test_keys:
                continue
            writer.writerow([key] + list(namespaces))

    for split, keys in {'val': val_keys, 'test': test_keys}.items():
        fname = os.path.join(
            data_path, '%s_%s_imports_%s.csv' % (lang, mode, split))
        with open(fname, 'wb') as imports_fh:
            writer = csv.writer(imports_fh)
            for key in keys:
                writer.writerow([key] + list(imports_dict[key]))


def write_adoption_benchmark(
        imports_timeline, test_ids, benchmark_fname, benchmark_size=100000,
        min_adoption_months=3):
    with open(benchmark_fname, 'wb') as benchmark_fh:
        writer = csv.writer(benchmark_fh)
        collected = 0
        random.shuffle(test_ids)
        for id_ in test_ids:
            timeline = defaultdict(set)
            for namespace, date in imports_timeline[id_].items():
                timeline[date].add(namespace)
            if len(timeline) < min_adoption_months:
                continue
            timestamps = sorted(timeline.keys())
            # selected_timestamp = timestamps[-1]
            selected_timestamp = random.choice(timestamps[2:])
            adopted = timeline[selected_timestamp]
            previously_used = set().union(
                *(timeline[ts] for ts in timestamps
                  if ts < selected_timestamp))
            writer.writerow(
                [id_, ','.join(adopted), ','.join(previously_used)])
            collected += 1
            if collected >= benchmark_size:
                break


def months_since(start, date):
    # both dates are expected to be YYYY-MM-DD
    years = int(date[:4]) - int(start[:4])
    months = int(date[5:7]) - int(start[5:7])
    return months + years * 12


def get_adoption_timeline(imports_timeline, vocab, max_months=100,
                          stoplist=None):
    adoption_times = pd.DataFrame(0, index=vocab, columns=range(max_months))
    stoplist = stoplist or set()
    # project_imports[project][namespace] = earliest_date_used
    for entity_id, adoptions in imports_timeline.items():
        if entity_id in stoplist:
            continue
        start_date = min(adoptions.values())
        for namespace, date in adoptions.items():
            if namespace not in vocab:
                continue
            months_diff = months_since(start_date, date)
            if months_diff < max_months:
                adoption_times.loc[namespace, months_diff] += 1
    return adoption_times


def _lstmize_project_imports(d, separator='#'):
    for date in sorted(d.keys()):
        for namespace in sorted(d[date]):
            yield namespace
        yield separator


def write_lstm_timeline(
        entity_ids, imports_timeline, vocab, filename,
        min_seq_length=2, max_seq_length=100, size_limit=None, separator='#'):
    size_limit = size_limit or float('inf')
    with open(filename, 'wb') as fh:
        writer = csv.writer(fh)
        collected = 0
        for entity_id in entity_ids:
            row = [entity_id]
            adoptions = pd.Series({
                namespace: date
                for namespace, date in imports_timeline[entity_id].items()
                if namespace in vocab})
            d = adoptions.index.groupby(adoptions)
            if len(d) < min_seq_length:
                continue
            for namespace in _lstmize_project_imports(d, separator=separator):
                if len(row) > max_seq_length:
                    break
                row.append(namespace)
            writer.writerow(row)
            collected += 1
            if collected >= size_limit:
                return


def get_adjacency_matrix(imports_timeline, vocab):
    df = pd.DataFrame(0, index=vocab, columns=vocab)
    for adoptions in imports_timeline.values():
        imports = adoptions.values()
        for ns1, ns2 in combinations(imports, 2):
            df.loc[ns1, ns2] += 1
            df.loc[ns2, ns1] += 1
    return df


def main(lang, low_usage_threshold=100, test_val_share=0.1,
         benchmark_size=100000, bot_threshold=0.001,
         ns_whitelist_path='ns_whitelist', data_path='data',
         benchmark_path='benchmark_data',
         min_project_imports=2, max_project_imports=100):
    assert lang in utils.ECOSYSTEMS, "Invalid language"
    idx2namespace = {}
    vocab_fname = os.path.join(data_path, lang + '_vocab.csv')
    if os.path.isfile(vocab_fname):
        logging.info("Reading existing vocab...")
        with open(vocab_fname) as vocab_fh:
            idx2namespace = {i: line.strip() for i, line in enumerate(vocab_fh)}

    logging.info("Collecting stats...")

    # project_imports[project][namespace] = earliest_date_used
    project_imports_timeline = defaultdict(
        lambda: defaultdict(lambda: MAX_DATE))
    # project_devs[project][email] = earliest_date_contributed
    project_devs = defaultdict(
        lambda: defaultdict(lambda: MAX_DATE))
    # dev_imports[email][namespace] = earliest_date_used
    dev_imports_timeline = defaultdict(
        lambda: defaultdict(lambda: MAX_DATE))
    dev_commit_count = defaultdict(int)
    proj_commit_count = defaultdict(int)

    for project_id, author_email, dt, imports in preprocess(lang):
        day = dt.strftime('%Y-%m-%d')
        for namespace in imports:
            project_imports_timeline[project_id][namespace] = min(
                day, project_imports_timeline[project_id][namespace])
            project_devs[project_id][author_email] = min(
                day, project_devs[project_id][author_email])
            dev_imports_timeline[author_email][namespace] = min(
                day, dev_imports_timeline[author_email][namespace])
        dev_commit_count[author_email] += 1
        proj_commit_count[project_id] += 1

    ns_counts_by_proj = get_counts(project_imports_timeline)
    ns_counts_by_proj.to_csv(
        os.path.join(data_path, lang + '_namespace_counts_by_projects.csv'))
    ns_counts_by_dev = get_counts(dev_imports_timeline)
    ns_counts_by_dev.to_csv(
        os.path.join(data_path, lang + '_namespace_counts_by_dev.csv'))

    logging.info("Creating/updating vocabulary...")
    stoplist = set(
        ns_counts_by_dev.index[ns_counts_by_dev < low_usage_threshold])
    whitelist_fname = os.path.join(ns_whitelist_path, lang + '_whitelist.csv')
    with open(whitelist_fname) as whitelist_fh:
        whitelist = {ns.strip() for ns in whitelist_fh}
    missing_vocab_entities = ((
            set(ns_counts_by_dev.index) - stoplist).intersection(whitelist)
            - set(idx2namespace.values()))
    idx2namespace.update({
        i: namespace for i, namespace in
        enumerate(missing_vocab_entities, len(idx2namespace))})
    vocab = set(idx2namespace.values())
    # namespace2idx = {namespace: i for i, namespace in idx2namespace.items()}
    # check: !tail data/PY_vocab.csv
    with open(vocab_fname, 'wb') as vocab_fh:
        for i in range(len(idx2namespace)):
            vocab_fh.write(idx2namespace[i] + '\n')

    dev_commit_count = pd.Series(dev_commit_count).sort_values(ascending=False)
    dev_commit_count.to_csv(
        os.path.join(data_path, lang + '_dev_commit_counts.csv'))
    alleged_bots_num = int(len(dev_commit_count) * bot_threshold)
    alleged_bots = set(dev_commit_count.index[:alleged_bots_num])
    trivial_profiles = set(dev_commit_count.index[-alleged_bots_num:])

    # aggregate project imports
    project_import_counts = {}
    project_imports = {}
    for project_id, imports in project_imports_timeline.items():
        filtered_imports = vocab.intersection(imports.keys())
        project_import_counts[project_id] = len(filtered_imports)
        # ignore projects with less than two and 100+ imports
        if min_project_imports <= len(filtered_imports) <= max_project_imports:
            project_imports[project_id] = filtered_imports
    # project_import_counts = pd.Series(project_import_counts).sort_values()
    # project_import_counts.to_csv(
    #     os.path.join(data_path, lang + '_proj_imports_counts.csv'))

    # sample test and validation projects
    test_val_size = int(test_val_share * len(project_imports))
    val_projects = set(random.sample(project_imports.keys(), test_val_size))
    test_projects = set(random.sample(project_imports.keys(), test_val_size))

    logging.info("Writing project imports...")
    # NOTE: now imports contain namespaces, not IDs
    write_imports(lang, 'proj', project_imports, val_projects, test_projects,
                  data_path=data_path)

    # aggregate dev profiles - only for train/test/val, not benchmarks
    dev_imports = {}
    for email, imports in dev_imports_timeline.items():
        filtered_imports = vocab.intersection(imports.keys())
        if 1 < len(filtered_imports) < 1000:
            dev_imports[email] = filtered_imports

    # get developer test and validation
    # extract devs from val/test projects
    # PY will end up getting 395K test/val developers
    test_val_size = int(test_val_share * len(dev_imports))
    val_devs = set().union(
        *(project_devs[project].keys() for project in val_projects)
    ).intersection(dev_imports.keys())
    val_devs = set(list(val_devs - alleged_bots - trivial_profiles)[:test_val_size])
    test_devs = set().union(
        *(project_devs[project].keys() for project in test_projects)
    ).intersection(dev_imports.keys())
    test_devs = set(list(
        test_devs - val_devs - alleged_bots - trivial_profiles)[:test_val_size])

    logging.info("Writing dev imports...")
    write_imports(lang, 'dev', dev_imports, val_devs, test_devs,
                  data_path=data_path)

    # write prediction benchmarks
    # find 100k projects that adopted libraries on at least X different dates
    # predict libraries adopted on the last date

    # benchmark: what library(ies) will be adopted by a project or a developer
    # check:
    # !head benchmark_data/PY_proj_adoption_benchmark.csv
    # !head benchmark_data/PY_dev_adoption_benchmark.csv
    # lines are project/dev_id, newly adopted, previous history
    # TODO: replace with lstm
    write_adoption_benchmark(project_imports_timeline, test_projects,
                             lang + '_proj_adoption_benchmark.csv')

    write_adoption_benchmark(dev_imports_timeline, test_devs,
                             lang + '_dev_adoption_benchmark.csv')

    # benchmark: what kind of developer is going to join the project
    dev_profile_lengths = defaultdict(list)
    for email, adoptions in dev_imports_timeline.items():
        dev_profile_lengths[len(adoptions)].append(email)

    # Dev join benchmark
    benchmark_fname = os.path.join(
        benchmark_path, lang + '_dev_join_benchmark.csv')
    with open(benchmark_fname, 'wb') as benchmark_fh:
        writer = csv.writer(benchmark_fh)
        collected = 0
        for project_id, join_dates in project_devs.items():
            if len(join_dates) < 3:
                continue
            project_adoptions = project_imports_timeline[project_id]
            developers = sorted(
                join_dates.keys(), key=lambda dev: join_dates[dev])
            # alternatively, just get the last joining one
            # selected_dev = developers[-1]
            selected_dev = random.choice(developers[2:])
            join_date = join_dates[selected_dev]
            project_profile = set(
                (ns for ns, adoption_date in project_adoptions.items()
                 if adoption_date < join_date)).intersection(vocab)
            if len(project_profile) < 2:
                continue

            dev_adoptions = dev_imports_timeline[selected_dev]
            dev_profile = set(
                (ns for ns, adoption_date in dev_adoptions.items()
                  if adoption_date < join_date)).intersection(vocab)
            if len(dev_profile) < 2:
                continue

            if len(dev_profile_lengths[len(dev_profile)]) < 2:
                continue
            while True:
                decoy_dev = random.choice(dev_profile_lengths[len(dev_profile)])
                if decoy_dev == selected_dev:
                    continue
                decoy_adoptions = dev_imports_timeline[decoy_dev]
                false_profile = set(decoy_adoptions.keys()).intersection(vocab)
                if len(false_profile) < 2:
                    continue
                break

            writer.writerow(
                [project_id, selected_dev, ','.join(project_profile),
                 ','.join(false_profile), ','.join(dev_profile)])
            collected += 1
            if collected >= benchmark_size:
                break

    # LSTM: projects
    train_projects = set(project_imports.keys()) - test_projects - val_projects
    for category, entity_ids, size_limit in (
            ('train', train_projects, None),
            ('test', test_projects, benchmark_size),
            ('val', val_projects, benchmark_size)):
        fname = os.path.join(data_path, '%s_proj_lstm_%s.csv' % (lang, category))
        write_lstm_timeline(entity_ids, project_imports_timeline, vocab, fname,
                            size_limit=size_limit)

    # LSTM: developers
    train_devs = set(dev_imports.keys()) - test_devs - val_devs
    for category, entity_ids, size_limit in (
             ('train', train_devs, None),
             ('test', test_devs, benchmark_size),
             ('val', val_devs, benchmark_size)):
        fname = os.path.join(data_path, '%s_dev_lstm_%s.csv' % (lang, category))
        write_lstm_timeline(entity_ids, dev_imports_timeline, vocab, fname)

    # create adoption matrix vocab_size x 100
    get_adoption_timeline(project_imports_timeline, vocab).to_csv(
        os.path.join(data_path, lang + '_proj_adoption_times.csv'))

    get_adoption_timeline(
        dev_imports_timeline, vocab, stoplist=alleged_bots+trivial_profiles,
        max_months=300).to_csv(
        os.path.join(data_path, lang + '_dev_adoption_times.csv'))

    get_adjacency_matrix(project_imports_timeline, vocab).to_csv(
        os.path.join(data_path, lang + '_proj_co-occurrence.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess Python imoprts data from WoC dataset')
    parser.add_argument('language', choices=utils.ECOSYSTEMS,
                        help='Language map to process')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Log progress to stderr")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s',
                        level=logging.INFO if args.verbose else logging.WARNING)

    main(args.language)
