#!/usr/bin/env python3

"""
This file validates produced models, printing the report.
"""

import argparse
from collections import Counter
import csv
import logging
import os
import re
import sys

from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

supported_ecosystems = ('PY', 'R', 'JS', 'Go', 'jl', 'pl')
with open('global_vocab.csv') as fh:
    lib2id = pd.Series({line.strip(): i
                        for i, line in enumerate(fh)}).sort_values()
id2lib = pd.Series(lib2id.index, index=lib2id)


def numeric_ids_gen():
    # takes ~7 minutes
    with open('global_proj_imports.test.csv') as fh:
        for row in csv.reader(fh):
            yield [int(id_) for id_ in row[1:]]


def _knn(projects, embeddings, n, context_size):
    """
    n: either 1 or 20
    context_size: 2, 5, 10

        take w2v, N, ecosystem, context_size
        total_TP, total_FP = 0, 0
        select 1k projects. For each project,
            calculate the average embedding for libraries
            find top n similar projects
            get their libraries
            count their dependencies, get top context_size of them
            calculate  weighted TP and FP, add to total counter
        return total_TP / (total_TP + total_FP)
    """
    logging.info('\tKNN: n=%d, context_size=%d',
                 n, context_size)
    split_size = 1000
    logging.info('%d reference projects, %d total', split_size, len(projects))

    reference_tokens, other_tokens = projects[:split_size], projects[split_size:]
    reference_embeds, other_embeds = embeddings[:split_size], embeddings[split_size:]

    similarities = cosine_similarity(reference_embeds, other_embeds)
    # sorts such that last item in a column is the index of the biggest element
    # np.argpartition(a, -1, axis=0)
    most_similar_indexes = np.argpartition(similarities, -n, axis=1)[:, -n:]

    total_tp, total_fp = 0, 0
    for ref_tokens, msi in zip(reference_tokens, most_similar_indexes):
        counter = Counter()
        [counter.update(other_tokens[idx]) for idx in msi]
        for token, count in counter.most_common(context_size):
            if token in ref_tokens:
                total_tp += count / n
            else:
                total_fp += 1
    return total_tp / (total_tp + total_fp)


def knn(w2v, import2vec_ecosystem=None):
    ns = (1, 20)
    context_sizes = (2, 5, 10)
    ecosystems = ([import2vec_ecosystem] if import2vec_ecosystem is not None
                  else supported_ecosystems)
    df = pd.DataFrame(index=pd.MultiIndex.from_product([ns, context_sizes]),
                      columns=ecosystems)
    logging.info('Loading numeric IDs for KNN...')
    numeric_ids = list(numeric_ids_gen())

    for ecosystem in ecosystems:
        logging.info('Computing KNN for %s', ecosystem)
        if not ecosystem.endswith(':'):
            ecosystem += ':'
        projects = []
        embeddings = []
        for proj_ids in numeric_ids:
            proj_tokens = id2lib[proj_ids]
            if import2vec_ecosystem is None:
                proj_tokens = proj_tokens[proj_tokens.str.startswith(ecosystem)]
            else:
                proj_tokens = proj_tokens.str[len(ecosystem):]
                proj_tokens = proj_tokens[
                    [token in w2v.vocab for token in proj_tokens]]
            if len(proj_tokens) == 0:  # no imports in the target ecosystem
                continue
            if import2vec_ecosystem is None:
                avg_embedding = w2v.vectors[proj_tokens.index].mean(axis=0)
            else:  # slower
                avg_embedding = w2v[proj_tokens].mean(axis=0)

            projects.append(proj_tokens.to_list())
            embeddings.append(avg_embedding)

        embeddings = np.array(embeddings)

        for n in ns:
            for context_size in context_sizes:
                df.loc[(n, context_size), ecosystem] = _knn(
                    projects, embeddings, n, context_size)
    return df


def get_random_kv(embed_size):
    w2v = KeyedVectors(embed_size)
    w2v.add(list(id2lib), np.random.random((len(id2lib), embed_size)))
    return w2v


def get_kv(model_path, embed_size=None):
    # optional ecosystem is to limit to only one ecosystem
    if embed_size is None:
        embed_size = int(''.join(c for c in model_path if c.isdigit()))

    w2v = KeyedVectors(embed_size)
    model = tf.keras.models.load_model(model_path)

    all_weights = model.get_weights()
    # input_weights = all_weights[0]
    # embed_bias = all_weights[1]
    output_weights = all_weights[2].T
    # output_bias = all_weights[3]

    index = list(id2lib)
    weights = output_weights

    w2v.add(index, weights)
    return w2v


def truncate_w2v(w2v, new_dims):
    """Limit w2v to the specified number of dimensions, selected at random"""
    old_dims = w2v.vectors.shape[1]
    new_w2v = KeyedVectors(new_dims)
    vocab = list(w2v.vocab.keys())
    cols_idx = np.random.choice(old_dims, size=new_dims, replace=False)
    weights = w2v.vectors[:, cols_idx]
    new_w2v.add(vocab, weights)
    return new_w2v


def competition_benchmark(w2v, import2vec_ecosystem=None):
    competition_df = pd.read_csv('competition.csv')
    if import2vec_ecosystem is not None:
        # prevent matching packages with the same name in other ecosystems
        competition_df = competition_df[
            (competition_df['A_ecosystem'] == import2vec_ecosystem)
            & (competition_df['B_ecosystem'] == import2vec_ecosystem)]
    else:  # prepend ecosystem to the namespace
        competition_df['A'] = competition_df['A_ecosystem'] + ':' + competition_df['A']
        competition_df['B'] = competition_df['B_ecosystem'] + ':' + competition_df['B']

    def _similarity(row):
        res = pd.Series(name=row.name)
        if row['A'] in w2v.vocab and row['B'] in w2v.vocab:
            res['similarity'] = w2v.similarity(row['A'], row['B'])
            res['rank'] = len(w2v.closer_than(row['A'], row['B']))
            res['reverse_rank'] = len(w2v.closer_than(row['B'], row['A']))
        return res

    return pd.concat(
        [competition_df, competition_df.apply(_similarity, axis=1)], axis=1
    ).dropna()


def analogical_reasoning(w2v, import2vec=False, topn=1000):
    ar_df = pd.read_csv('analogical_reasoning.csv')

    if not import2vec:
        ar_df = 'JS:' + ar_df

    def _index(results, token):
        for i, (_token, _) in enumerate(results, 1):
            if _token == token:
                return i
        return len(results) + 1

    def _ar(row):
        if any(entity not in w2v for entity in row):
            print([(entity not in w2v, entity) for entity in row])
            return None
        pred = _index(w2v.most_similar(
            [row['a*'], row['b']], [row['a']], topn=topn), row['b*'])
        only_b = _index(w2v.most_similar(row['b'], topn=topn), row['b*'])
        return pd.Series({'pred': pred, 'only-b': only_b}, name=row.name)

    evaluation = ar_df.apply(_ar, axis=1)
    return pd.concat([ar_df, evaluation], axis=1)


def validation_loss():
    # import os

    from utils import dev2vecSequence, read_dev
    with open('global_vocab.csv') as fh:
        vocab_size = fh.read().count('\n') + 1

    input_types = ('proj', 'dev', 'dev-year')
    dimensions = (16, 32, 64, 100, 200)
    df = pd.DataFrame(columns=input_types, index=dimensions)
    for input_type in input_types:
        val_csv_path = 'global_%s_imports.val.csv' % input_type
        skip_fields = 2 if input_type == 'dev-year' else 1
        data, offsets = read_dev(val_csv_path, skip_fields)
        seq = dev2vecSequence(data, offsets, vocab_size=vocab_size)

        for dim in dimensions:
            model_path = 'models/%s_%d.model' % (input_type, dim)
            if not os.path.isdir(model_path):
                continue
            model = tf.keras.models.load_model(model_path)
            val_loss = model.evaluate(seq)
            df.loc[dim, input_type] = val_loss

    return df


def aggregate_competition(cb_df):
    cb_df = cb_df[(cb_df['A_ecosystem'] == 'PY') & (cb_df['B_ecosystem'] == 'PY')]
    return pd.Series({
        'avg': cb_df['similarity'].mean(),
        'std': cb_df['similarity'].std(),
        'n': len(cb_df)
    })


def cb_score():
    input_types = ('proj', 'dev', 'dev-year')
    dimensions = (16, 32, 64, 100, 200)
    idx = pd.MultiIndex.from_product([input_types, dimensions])
    df = pd.DataFrame(columns=['avg', 'std', 'n'], index=idx)

    for input_type in input_types:
        for dim in dimensions:
            model_path = 'models/%s_%d.model' % (input_type, dim)
            if not os.path.isdir(model_path):
                continue
            w2v = get_kv(model_path)
            cb_df = competition_benchmark(w2v)
            df.loc[(input_type, dim)] = aggregate_competition(cb_df)

    w2v = KeyedVectors.load_word2vec_format(
        'import2vec_data/python_w2v_dim100.txt.gz', binary=False)
    input_type = 'import2vec'
    for dim in dimensions:
        if dim > 100:
            continue
        tw2v = truncate_w2v(w2v, dim)
        cb_df = competition_benchmark(tw2v, 'PY')
        df.loc[(input_type, dim), :] = aggregate_competition(cb_df)

    df['ci'] = 1.96 * df['std'] / (df['n'] ** 0.5)
    return df[['avg', 'ci']].dropna()


def anomaly_detection(model_path='models/dev_32.model'):
    # import csv, re
    # import tensorflow as tf
    # model_path = 'models/dev_32.model'

    from utils import dev2vecSequence, read_dev
    with open('global_vocab.csv') as fh:
        vocab_size = fh.read().count('\n') + 1
    model = tf.keras.models.load_model(model_path)
    csv_path = 'unfiltered_dev_profiles.csv'
    data, offsets = read_dev(csv_path)
    seq = dev2vecSequence(data, offsets, vocab_size=vocab_size)
    bot_regexp = re.compile(r'\bbot\b', re.I)
    scores = {True: [], False: []}

    fh = open(csv_path)
    reader = csv.reader(fh)
    # There goal is to collect 1k bot records
    for i in range(len(seq)):
        bots = [bool(bot_regexp.search(next(reader)[0])) for _ in range(32)]
        if not any(bots):
            # if we store all scores, the model will run OOM
            # looks to be some compatibility issue with tf2 eager mode
            continue
        batch_input, batch_output = seq[i]
        batch_pred = model.predict(batch_input)
        batch_loss = tf.keras.losses.binary_crossentropy(batch_output, batch_pred).numpy()
        for is_bot, loss in zip(bots, batch_loss):
            scores[is_bot].append(loss)

    bots = np.array(scores[True])
    non_bots = np.array(scores[False])
    bots_d = np.histogram(np.log(bots), bins=50, range=range, density=True)
    non_bots_d = np.histogram(np.log(non_bots), bins=50, range=range, density=True)
    re_df = pd.DataFrame({'bots': bots_d[0], 'non_bots': non_bots_d[0]},
                         index=bots_d[1][:-1])
    return re_df
    # How to plot:
    # re_df = pd.read_csv('anomaly_detection.csv', index_col=0)
    # ax = re_df.plot()
    # ax.set_ylabel("probability density")
    # ax.set_xlabel("log reconstruction error")
    # plt.savefig('anomaly_detection.pdf')


def main(w2v, name, evaluation_path='evaluation', import2vec_ecosystem=None,
         force=False):
    if not os.path.isdir(evaluation_path):
        os.mkdir(evaluation_path)

    # knn_fname = os.path.join(evaluation_path, name + '.knn.csv')
    # if force or not os.path.isfile(knn_fname):
    #     sys.stderr.write('\n\nKNN evaluation:\n')
    #     knn_df = knn(w2v, import2vec_ecosystem)
    #     knn_df.to_csv(sys.stderr)
    #     knn_df.to_csv(knn_fname, float_format='%.3f')
    # else:
    #     logging.warning('KNN cached, wont overwrite without --force')

    cb_fname = os.path.join(evaluation_path, name + '.cb.csv')
    if force or not os.path.isfile(cb_fname):
        sys.stderr.write('\n\nCompetition benchmark:\n')
        cb_df = competition_benchmark(w2v, import2vec_ecosystem)
        cb_df.to_csv(sys.stderr)
        cb_df.to_csv(cb_fname, float_format='%.3f')

        group = cb_df.groupby(['A_ecosystem', 'B_ecosystem'])
        cb_series = pd.Series({
            'avg': group.mean(),
            'std': group.std(),
            'n': group.count()
        })
        cb_series.to_csv(sys.stderr)
    else:
        logging.warning('Competition benchmark is cached, '
                        'wont overwrite without --force')

    ar_fname = os.path.join(evaluation_path, name + '.ar.csv')
    if import2vec_ecosystem == 'PY':
        logging.warning('Analogous reasoning does not apply to Import2vec '
                        'pretrained on Python, skipping.')
    elif force or not os.path.isfile(ar_fname):
        sys.stderr.write('\n\nAnalogical reasoning:\n')
        ar_df = analogical_reasoning(w2v, bool(import2vec_ecosystem))
        ar_df.to_csv(sys.stderr)
        ar_df.to_csv(ar_fname)
    else:
        logging.warning(
            'Analogical reasoning cached, wont overwrite without --force')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validate the produced models, printing the report.')
    parser.add_argument('model_path',
                        help='''Model path. It has two special values:
    "random": will use random embeddings.
    "import2vec": use the previously trained model published by import2vec
    ''')
    parser.add_argument('--random-embed-size', default='100', type=int,
                        help='Embedding dimensionality for random embeddings')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Log progress to stderr")
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force recalculating the metrics')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s',
                        level=logging.INFO if args.verbose else logging.WARNING)

    mode = args.model_path
    import2vec_ecosystem = None
    if mode == 'random':
        w2v = get_random_kv(args.random_embed_size)
    elif mode.endswith('txt.gz'):  # import2vec
        import2vec_ecosystems = {
            'js': 'JS',
            'python': 'PY'
        }
        _ecosystem = mode.rsplit('/', 1)[-1].split('_', 1)[0]
        if _ecosystem not in import2vec_ecosystems:
            parser.exit(2, 'Unknown imoprt2vec ecosystem')
        # dimensionality is the XXX int in ...dimXXX.txt.gz
        # embed_dim = int(mode[:-6].rsplit('dim', 1)[-1])
        import2vec_ecosystem = import2vec_ecosystems[_ecosystem]
        w2v = KeyedVectors.load_word2vec_format(mode, binary=False)
    elif os.path.isdir(mode):
        w2v = get_kv(args.model_path)
    else:
        parser.exit(1, 'Invalid model')

    mode = mode.strip('/').rsplit('/', 1)[-1]

    main(w2v, mode, import2vec_ecosystem=import2vec_ecosystem, force=args.force)
