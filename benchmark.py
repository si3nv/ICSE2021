#!/usr/bin/env python
from __future__ import print_function

import argparse
from collections import defaultdict
import csv
from itertools import combinations
import logging
import random

from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy import stats
import tensorflow as tf
from typing import Dict, List, Iterable, Optional, Tuple

import utils

try:  # used by Naive model
    import textdistance
except ImportError:
    pass

# default number of topmost predictions
DEFAULT_MAX_LEN = 10


def cosine_similarity(vector, other_vectors):
    norm = np.linalg.norm(vector)
    all_norms = np.linalg.norm(other_vectors, axis=-1)
    return vector.dot(other_vectors.T) / (norm * all_norms)


class EmbeddingModel(object):
    """base class for uniform interface to predict and compute similarity
    Subclasses: Import2vec, Dev2vec, most used
    """
    def __init__(self, model, *args, **kwargs):
        """
        Args:
            model_path (str): model specific
            vocab Dict[int, str]: vocabulary of entries to be used
        """
        self.model = model

    def similarity(self, a, b):
        # type: (str, str) -> float
        raise NotImplemented

    def similarities(self, a, b):
        # type: (str, Iterable[str]) -> pd.Series
        raise NotImplemented

    # TODO: similarity rank?

    def predict_adoption(self, context, max_len=DEFAULT_MAX_LEN):
        # type: (List[str], int) -> List[str]
        """return list of projects ordered by probability of adoption"""
        raise NotImplemented

    def dev_project_match(self, project_profile, dev_profile):
        raise NotImplemented


class Dev2vec(EmbeddingModel):
    def __init__(self, model, idx2namespace):
        if isinstance(idx2namespace, str):
            idx2namespace, _ = utils.read_vocab(idx2namespace)
        self.vocab = idx2namespace
        idx = [idx2namespace[i] for i in range(len(idx2namespace))]
        if isinstance(model, str):
            model = tf.keras.models.load_model(model)
        super(Dev2vec, self).__init__(model)
        in_weights, embed_bias, out_weights, out_bias = model.weights
        self.embed_bias = embed_bias.numpy()
        self.out_bias = out_bias.numpy()
        self.in_weights = pd.DataFrame(in_weights.numpy(), index=idx)
        # in_weights are vocab x embed_size and out_weights are
        #   embed_size x vocab, so better transpose
        self.out_weights = pd.DataFrame(out_weights.numpy().T, index=idx)
        # now both self.in_weights and self.out_weights are vocab x embed_size

    def similarity(self, a, b):
        # type: (str, Iterable[str]) -> float
        try:
            return cosine_similarity(
                self.in_weights.loc[a], self.out_weights.loc[b])
        except KeyError:
            return np.nan

    similarities = similarity

    def predict_adoption(self, context, max_len=DEFAULT_MAX_LEN):
        # type: (List[str], int) -> List[str]
        max_len = max(max_len, len(context))
        context = [c for c in context if c in self.out_weights.index]
        logits = (self.in_weights.loc[context].sum(axis=0) + self.embed_bias
                  ).dot(self.out_weights.T).drop(context)
        top_idx = np.argpartition(logits.values, -max_len)[-max_len:]
        top = logits.iloc[top_idx].sort_values(ascending=False)
        return top.index.to_list()

    def dev_project_match(self, project_profile, dev_profile):
        # TODO: try different combinations of in/out
        try:
            in_weights = self.in_weights.loc[project_profile]
            out_weights = self.out_weights.loc[dev_profile]
        except KeyError:
            return None
        return cosine_similarity(
                in_weights.sum(axis=0),
                out_weights.sum(axis=0))


class LSTM(Dev2vec):
    def __init__(self, model, vocab_path):
        self.idx2namespace, self.namespace2idx = utils.read_vocab(
            vocab_path, lstm=True)
        if isinstance(model, str):
            model = utils.get_lstm_model(
                len(self.idx2namespace), checkpoint_path=model)
        idx = [self.idx2namespace[i] for i in range(len(self.idx2namespace))]
        self.in_weights = pd.DataFrame(model.weights[0].numpy(), index=idx)
        self.out_weights = self.in_weights
        self.model = model

    def predict_adoption(self, context, max_len=DEFAULT_MAX_LEN):
        # type: (List[str], int) -> List[str]
        max_len = max(max_len, len(context))
        max_misses = 20

        def gen(seq, max_len=max_len):
            if len(seq.shape) == 1:
                seq = np.array([seq])
            existing = set(seq.ravel())
            existing.add(0)
            misses = 0
            processed = 0
            while True:
                # 0: first element in the batch of one
                # -1: get the last prediction
                predictions = self.model.predict_on_batch(seq)[0, -1]
                prediction_id = predictions.argmax()
                try:
                    prediction = self.idx2namespace[prediction_id]
                except:
                    print(predictions, prediction_id, seq)
                    raise
                if prediction not in existing:
                    yield prediction
                    misses = 0
                    processed += 1
                    if processed >= max_len:
                        break
                    existing.add(prediction)
                else:
                    misses += 1
                    if misses > max_misses:
                        break
                seq = np.concatenate([seq, [[prediction_id]]], axis=1)

        return list(gen(np.array([self.namespace2idx[c] for c in context
                                  if c in self.namespace2idx])))


class Import2vec(EmbeddingModel):
    def __init__(self, model):
        if isinstance(model, str):
            model = KeyedVectors.load_word2vec_format(model, binary=False)
        self.weights = pd.DataFrame(
            model.vectors, index=list(model.vocab.keys()))
        super(Import2vec, self).__init__(model)

    def similarity(self, a, b):
        # type: (str, Iterable[str]) -> float
        try:
            return cosine_similarity(self.weights.loc[a], self.weights.loc[b])
        except KeyError:
            return np.nan

    similarities = similarity

    def predict_adoption(self, context, max_len=DEFAULT_MAX_LEN):
        # type: (List[str], int) -> List[str]
        max_len = max(max_len, len(context))
        context = [c for c in context if c in self.weights.index]
        logits = self.weights.loc[context].dot(self.weights.T)
        probabilities = sigmoid(logits).sum(axis=0).drop(context)
        top_idx = np.argpartition(probabilities.values, -max_len)[-max_len:]
        top = probabilities.iloc[top_idx].sort_values(ascending=False)
        return top.index.to_list()

        # return probabilities.sort_values(ascending=False)

    def dev_project_match(self, project_profile, dev_profile):
        try:
            in_weights = self.weights.loc[project_profile]
            out_weights = self.weights.loc[dev_profile]
        except KeyError:
            return None
        return cosine_similarity(
                in_weights.sum(axis=0),
                out_weights.sum(axis=0))


class Naive(EmbeddingModel):
    def __init__(self, counts_fname):
        model = pd.read_csv(
            counts_fname, header=None, squeeze=True, index_col=0
        ).sort_values(ascending=False)
        super(Naive, self).__init__(model[model > 10])

    def predict_adoption(self, context, max_len=DEFAULT_MAX_LEN):
        # type: (List[str], int) -> List[str]
        max_len = max(max_len, len(context))
        context = [c for c in context if c in self.model]
        # always predict most common libraries
        top = self.model.drop(context)[:max_len]
        return top.index.to_list()

    def dev_project_match(self, project_profile, dev_profile):
        # just compute the direct overlap of libraries
        pp = set(project_profile)
        # return len(pp.intersection(dev_profile)) / len(dev_profile)
        return len(pp.intersection(dev_profile)) / len(pp)

    def _txt_distance(self, a, b):
        return textdistance.levenshtein.normalized_similarity(a, b)

    def similarity(self, a, b):
        # type: (str, Iterable[str]) -> float
        if isinstance(b, str):
            return self._txt_distance(a, b)
        return pd.Series([self._txt_distance(a, b_) for b_ in b])

    similarities = similarity


class LSI(EmbeddingModel):
    def __init__(self, model, namespace2idx):
        if isinstance(namespace2idx, str):
            idx2namespace, namespace2idx = utils.read_vocab(namespace2idx)
        if isinstance(model, str):
            from gensim.models import LsiModel
            model = LsiModel.load(model)
        self.vocab = pd.Series(namespace2idx).sort_values()
        self.weights = pd.DataFrame(model.projection.u, index=self.vocab.index)
        super(LSI, self).__init__(model)

    def similarity(self, a, b):
        # type: (str, Iterable[str]) -> float
        try:
            return cosine_similarity(
                self.weights.loc[a], self.weights.loc[b])
        except KeyError:
            return np.nan

    similarities = similarity

    def doc_embed(self, context):
        return np.array(
            self.model[[(self.vocab[ns], 1) for ns in context]])[:, 1]

    def predict_adoption(self, context, max_len=DEFAULT_MAX_LEN):
        # type: (List[str], int) -> List[str]
        # filter out OOV items
        context = [ns for ns in context if ns in self.vocab]
        max_len = max(max_len, len(context))
        doc = self.doc_embed(context)  # / self.model.projection.s
        probs = self.weights.dot(doc).drop(context)
        top_idx = np.argpartition(probs.values, -max_len)[-max_len:]
        top = probs.iloc[top_idx].sort_values(ascending=False)
        return top.index.to_list()

    def dev_project_match(self, project_profile, dev_profile):
        return cosine_similarity(
            self.doc_embed(project_profile),
            self.doc_embed(dev_profile))


def create_prediction_benchmark(target_ecosystem, input_file, vocab_path):
    # for practical purposes it makes sense to limit the set by 50..100k entries
    ecosystems = []
    targets = set()
    with open(vocab_path) as vocab_fh:
        for i, name in enumerate(vocab_fh):
            ecosystem = name.strip().split(':', 1)[0]
            ecosystems.append(ecosystem)
            if ecosystem == target_ecosystem:
                targets.add(str(i))

    misssing_one_fname = 'predict_missing_%s.csv' % target_ecosystem
    extra_one_fname = 'detect_extra_%s.csv' % target_ecosystem
    with open(misssing_one_fname, 'w') as fh_missing, \
            open(extra_one_fname, 'w') as fh_extra:
        writer_missing = csv.writer(fh_missing)
        writer_extra = csv.writer(fh_extra)
        for line in input_file:
            chunks = line.strip().split(',')
            project_name = chunks[0]
            imports = set(i for i in chunks[1:] if i in targets)
            if len(imports) < 2:
                continue
            removed = random.choice(tuple(imports))
            added = random.choice(tuple(targets - imports))
            writer_extra.writerow(
                [project_name, added] + list(imports.union([added])))
            writer_missing.writerow(
                [project_name, removed] + list(imports - {removed}))


def run_prediction_benchmark(
        target_ecosystem='PY', input_file_path='predict_missing_PY.csv',
        vocab_path='global_vocab.csv', model_path='models/proj_100.model',
        import2vec_path='import2vec_data/python_w2v_dim100.txt.gz',
        max_records=100000, top_n=None):
    # type: (str, str, str, str, str, int, Optional[List[int]]) -> Dict
    """
    only use shared vocabulary
    Import2vec: (import x embed_size) x (embed_size x vocab_size),
        apply sigmoid, then log, then sum along one axis take argmax
    Dev
    """
    top_n = tuple(sorted(top_n or [1]))
    max_n = max(top_n)
    i2v = KeyedVectors.load_word2vec_format(import2vec_path, binary=False)
    i2v_vocab = dict(enumerate(i2v.vocab.keys()))
    i2v_reverse_vocab = {v: k for k, v in i2v_vocab.items()}
    d2v = tf.keras.models.load_model(model_path)

    vocab = {}  # type: Dict[int, str]
    reverse_vocab = {}  # type: Dict[str, int]
    vocab_size = 0  # type: int
    with open(vocab_path) as vocab_fh:
        for i, name in enumerate(vocab_fh):
            chunks = name.strip().split(':', 1)  # type: List[str]
            if chunks[0] == target_ecosystem:
                vocab[i] = chunks[-1]
                reverse_vocab[chunks[-1]] = i
            vocab_size += 1
    tokenizer = tf.keras.preprocessing.text.Tokenizer(vocab_size)
    d2v_mask = tokenizer.sequences_to_matrix([vocab.keys()])
    i2v_mask = np.array([w in reverse_vocab for w in i2v_reverse_vocab], dtype=int)

    hits = {
        'total': 0,
        'd2v': defaultdict(int),
        'i2v': defaultdict(int)
    }  # the format should be suitable for pd.DataFrame constructor
    with open(input_file_path) as input_file:
        for line in input_file:
            chunks = line.split(',')
            ground_truth = int(chunks[1])  # type: int
            imports = [int(i) for i in chunks[2:]]  # type: List[int]

            if (not all(vocab[i] in i2v for i in imports)
                    or vocab[ground_truth] not in i2v_reverse_vocab):
                # 501820 records total, 459071 have shared vocabulary
                continue
            i2v_ground_truth = i2v_reverse_vocab[vocab[ground_truth]]
            hits['total'] += 1

            d2v_inputs = tokenizer.sequences_to_matrix([imports])
            # exclude predictions outside of the target ecosystem
            d2v_probabilities = d2v.predict(d2v_inputs) * d2v_mask
            # also exclude inputs
            d2v_probabilities[:, imports] = 0
            # dv_top is a sorted list of top max_n indices
            d2v_top = np.argsort(d2v_probabilities.ravel())[-1:-max_n:-1]
            for n in top_n:
                hits['d2v'][n] += ground_truth in d2v_top[:n]

            i2v_inputs = [i2v_reverse_vocab[vocab[i]] for i in imports]
            i2v_logits = i2v.vectors[i2v_inputs, :].dot(i2v.vectors.T)
            # TODO: add bias to logits
            i2v_probabilities = sigmoid(i2v_logits).sum(axis=0) * i2v_mask
            i2v_probabilities[i2v_inputs] = 0
            i2v_top = np.argsort(i2v_probabilities)[-1:-max_n:-1]
            for n in top_n:
                hits['i2v'][n] += i2v_ground_truth in i2v_top[:n]

            print('Total:', hits['total'], '\td2v:', hits['d2v'][1],
                  '\ti2v:', hits['i2v'][1], end='\r')
            if hits['total'] > max_records:
                break

    return hits


def get_models(ecosystem):
    d2v_vocab_fn = 'data/%s_vocab.csv' % ecosystem
    i2v_ecosystem = {'JS': 'js', 'PY': 'python'}[ecosystem]
    return {
        'dev': Dev2vec(
            'models/%s_dev_100.model' % ecosystem, d2v_vocab_fn),
        'dev_norm': Dev2vec(
            'models/%s_dev_100_norm.model' % ecosystem, d2v_vocab_fn),
        'dev_weighted': Dev2vec(
            'models/%s_dev_100_weighted.model' % ecosystem, d2v_vocab_fn),
        'proj': Dev2vec(
            'models/%s_proj_100.model' % ecosystem, d2v_vocab_fn),
        'proj_norm': Dev2vec(
            'models/%s_proj_100_norm.model' % ecosystem, d2v_vocab_fn),
        'proj_weighted': Dev2vec(
            'models/%s_proj_100_weighted.model' % ecosystem, d2v_vocab_fn),
        'i2v': Import2vec(
            'import2vec_data/%s_w2v_dim100.txt.gz' % i2v_ecosystem),
        'lstm': LSTM(
            'models/%s_proj_lstm.model.chkpt' % ecosystem, d2v_vocab_fn),
        'lsi': LSI(
            'models/%s_proj_lstm.model.chkpt' % ecosystem, d2v_vocab_fn),
    }


def benchmark(func):
    def wrapper(ecosystem, test_file_path):
        def gen():
            for name, model in get_models(ecosystem).items():
                try:
                    yield func(model, test_file_path).rename(name)
                except:
                    logging.error("Benchmark error in model %s", name)
                    raise
        return pd.DataFrame(gen())

    return wrapper


def compare_series(series1, series2):
    # type: (pd.Series, pd.Series) -> pd.Series
    """Compute differences in two series

    Args:
        series1 (pd.Series): first series to compare
        series2 (pd.Series): first series to compare
    Returns:
        pd.Series: A series with three fields:
            `diff` - size of the difference
            `cohens_d` - Cohen's d  (effect size)
                https://en.wikipedia.org/wiki/Cohen%27s_d#Cohen's_d
            `p_value` - self explanatory
    """
    series1 = series1.dropna()
    series2 = series2.dropna()
    n1, n2 = len(series1), len(series2)
    diff = series1.mean() - series2.mean()
    s = (((n1 - 1) * series1.std() + (n2 - 1) * series2.std()) /
         (n1 + n2 - 2)) ** 0.5
    cohens_d = diff / s
    t_stat, p_value = stats.ttest_ind(series1, series2)
    return pd.Series({
        'diff': diff,
        's': s,
        'cohens_d': cohens_d,
        'p_value': p_value,
    })


def similarity_benchmark(model, input_file_path):
    """
    d2v = Dev2vec('models/PY_dev_100.model', 'data/PY_vocab.csv')
    similarity_benchmark(d2v, 'benchmark_data/similarity_benchmark.csv')
    """
    # A, B, A_ecosystem, B_ecosystem, relation_type
    input_df = pd.read_csv(input_file_path)
    input_df['similarity'] = input_df.apply(
        lambda row: model.similarity(row['A'], row['B']), axis=1)
    input_df['angle'] = np.arccos(input_df['similarity'])

    # def count(series):
    #     return series.count()
    # df = input_df['angle'].groupby(
    #     input_df['relation_type']).agg([np.mean, np.std, count])
    # >>> df
    #                    mean       std  count
    # relation_type
    # competing      1.197471  0.270938  107.0
    # complementary  1.066195  0.262586  101.0
    # orthogonal     1.542981  0.150828  103.0

    def gen():
        for rel1, rel2 in combinations(input_df['relation_type'].unique(), 2):
            yield compare_series(
                input_df.loc[input_df['relation_type'] == rel1, 'angle'],
                input_df.loc[input_df['relation_type'] == rel2, 'angle']
            ).rename(rel1 + '_vs_' + rel2)

    df = pd.DataFrame(gen())
    # >>> df
    #                                  diff  cohens_d       p_value
    # competing_vs_complementary   0.131276  0.254111  4.856633e-04
    # competing_vs_orthogonal     -0.345510 -0.750332  1.420649e-23
    # complementary_vs_orthogonal -0.476786 -1.050092  1.410972e-37

    df1 = df.stack().reset_index()
    return df1.set_index(df1['level_0'] + '_' + df1['level_1'])[0]


def predict_adoption(model, input_file_path, sep=utils.DEFAULT_LSTM_SEPARATOR):
    # type: (EmbeddingModel, str, str) -> pd.Series
    """
    predict_adoption(lsi_bm, 'data/PY_proj_lstm_test.csv')
    # NOT benchmark(lsi_bm, 'benchmark_data/PY_dev_adoption_benchmark.csv')

    # out of date - LSTM data is used
    # predict library adoption by a project or a developer
    # input format is CSV with three columns.
    #     First column is a project or developer id
    #     second column is a comma-separated string of adopted namespaces
    #     third column is a comma-separated string of previously used namespaces

    Computes top-1,3,5 accuracy (any predicted library is in the adopted list)
        and accuracy treating each adopted library as a separate record

    Returns:
         pd.Series with keys:
            (top1, top3, top5, intersection, total_records, total_libraries)

        top1,3,5 correct predictions (ony of 1,3,5 top predicted libraries is
            adopted)
        number of adopted libraries that were predicted correctly in the top
            N predicted, where N is the number of adopted libraries
        number of processed records - divide top1,3,5 to get accuracy
        number of processed libraries - divide intersection to get accuracy
    """
    with open(input_file_path) as input_fh:
        res = None
        for i, _line in enumerate(input_fh):
            # print(i)
            # first column is entry id - project id or a developer email
            line = _line.split(',', 1)[-1].strip('\r\n,'+sep)
            try:
                context_, ground_truth_ = line.rsplit(sep + ',', 1)
            except:
                print(i, line)
                continue  # raise
            context = (context_ + sep).split(',')
            ground_truth = ground_truth_.split(',')
            predicted = model.predict_adoption(context)
            s = pd.Series({
                'top1': predicted[0] in ground_truth,
                'top3': any(p in ground_truth for p in predicted[:3]),
                'top5': any(p in ground_truth for p in predicted[:5]),
                'intersection': len(
                    set(ground_truth).intersection(predicted[:len(ground_truth)])),
                'total_records': 1,
                'total_libraries': len(ground_truth)
            }, dtype=int)

            res = s if res is None else res + s
            if res['total_records'] >= 100000:
                break
        return res


def dev_matching_benchmark(model, input_file_path):
    # type: (EmbeddingModel, str) -> pd.DataFrame
    """
    ~4 minutes per model

    """

    def gen():
        with open(input_file_path) as input_fh:
            reader = csv.reader(input_fh)
            # from preprocess.py:
            # writer.writerow(
            #     [project_id, selected_dev, ','.join(project_profile),
            #      ','.join(false_profile), ','.join(dev_profile)])
            for _, _, project_, false_profile_, true_profile_ in reader:
                project_profile = project_.split(',')
                false_profile = false_profile_.split(',')
                true_profile = true_profile_.split(',')
                yield (model.dev_project_match(project_profile, true_profile),
                       model.dev_project_match(project_profile, false_profile))

    df = pd.DataFrame(gen(), columns=['true', 'false'])

    return compare_series(
        np.arccos(df['true']),
        np.arccos(df['false'])
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create prediction benchmarks')
    parser.add_argument('ecosystem', choices=ECOSYSTEMS,
                        help='Ecosystem to create benchmark for')
    parser.add_argument('-i', '--input', type=argparse.FileType('r'),
                        help='Path to the file with project imports. The first '
                             'column is ignored (expected to contain project '
                             'name), the rest is numerical IDs of the imports')
    parser.add_argument('--vocab-path', default='global_vocab.csv',
                        help='Path to the vocabulary file. Each line is '
                             'expected to contain a token, '
                             '<ecosystem>:<namespace>.') 
    args = parser.parse_args()

    create_prediction_benchmark(args.ecosystem, args.input, args.vocab_path)