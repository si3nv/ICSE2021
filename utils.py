#!/usr/bin/env python3

import csv
import math
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf


DEFAULT_BATCH_SIZE = 32
DEFAULT_MODEL_PATH = 'models'
DEFAULT_DATA_PATH = 'data'
DEFAULT_BENCHMARK_PATH = 'benchmark_data'
DEFAULT_LSTM_SEPARATOR = '#'
ECOSYSTEMS = (
    'PY',  # Python
    'R',
    'JS',  # Javascript
    'Go',  # imports include github URLs - take the last part only
    'ipy',  # Jupyter notebooks
    # 'java',
    'jl',  # Julia
    'pl',  # Perl
)

#
# class dev2vecSequence(tf.keras.utils.Sequence):
#     def __init__(self, input_data, input_offsets,
#                  output_data=None, output_offsets=None,
#                  vocab_size=None, batch_size=DEFAULT_BATCH_SIZE):
#         if not vocab_size:
#             raise ValueError('Vocabulary size is expected')
#         self.input_data = input_data
#         self.input_offsets = input_offsets
#         self.output_data = output_data
#         self.output_offsets = output_offsets
#         self.batch_size = batch_size
#         self.tokenizer = tf.keras.preprocessing.text.Tokenizer(vocab_size)
#
#     def __len__(self):
#         return math.floor(len(self.input_offsets) / self.batch_size)
#
#     def __getitem__(self, idx):
#         # mode='binary' by default
#         batch_input = self.tokenizer.sequences_to_matrix(
#             [list(self.input_data[start:end])
#              for start, end in self.input_offsets[
#                                idx*self.batch_size:(idx+1)*self.batch_size]])
#         if self.output_data is not None:
#             batch_output = self.tokenizer.sequences_to_matrix(
#                 [list(self.output_data[start:end])
#                  for start, end in self.output_offsets[
#                                    idx*self.batch_size:(idx+1)*self.batch_size]])
#         else:
#             batch_output = batch_input
#         return batch_input, batch_output


def read_vocab(vocab_path, lstm=False):
    # type: (str, bool) -> Tuple[Dict[int, str], Dict[str, int]]
    idx2namespace = {}
    namespace2idx = {}
    start_idx = 1 if lstm else 0
    with open(vocab_path) as vocab_fh:
        for idx, line in enumerate(vocab_fh, start_idx):
            namespace = line.strip()
            idx2namespace[idx] = namespace
            namespace2idx[namespace] = idx
    if lstm:
        idx2namespace[0] = DEFAULT_LSTM_SEPARATOR
        namespace2idx[DEFAULT_LSTM_SEPARATOR] = 0
    return idx2namespace, namespace2idx


def read_dev(imports_fname, namespace2idx, skip_fields=1):
    # type: (str, Dict[str, int], int) -> Tuple[np.ndarray, np.ndarray]
    # dev: each record is: dev,*imports => skip_files=1
    with open(imports_fname) as imports_fh:
        reader = csv.reader(imports_fh)
        row_lengths = np.array([len(row) - skip_fields for row in reader])
    row_indices = np.cumsum(np.concatenate([[0], row_lengths]))
    offsets = np.array(list(zip(row_indices, row_indices[1:])))
    total_imports = row_indices[-1]

    with open(imports_fname) as imports_fh:
        reader = csv.reader(imports_fh)
        data = np.empty(total_imports, dtype=int)
        for row, (start_offset, end_offset) in zip(reader, offsets):
            data[start_offset:end_offset] = list(
                namespace2idx[namespace] for namespace in row[skip_fields:])
    return data, offsets


def get_lstm_model(vocab_size, embed_size=100, rnn_size=100,
                   checkpoint_path=None):
    """
    If load_path is not None, we need a model for prediction. In this case,
        - use batch_size of 1
    """
    for_training = checkpoint_path is None
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocab_size, embed_size, batch_input_shape=[1, None]),
        tf.keras.layers.GRU(
            rnn_size, return_sequences=for_training, stateful=False),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])
    if checkpoint_path:
        model.load_weights(checkpoint_path)
    else:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model


def read_lstm_dataset(file_path, namespace2idx):
    with open(file_path) as fh:
        reader = csv.reader(fh)
        return np.array([
            # first entry is entity id - a project id or a developer email
            np.array([namespace2idx[ns] for ns in row[1:] if ns])
            for row in reader
        ])


class LSTMSequence(tf.keras.utils.Sequence):
    def __init__(self, file_path, namespace2idx):
        self.data = read_lstm_dataset(file_path, namespace2idx)

    def __len__(self):
        return math.floor(len(self.data) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
        return np.array([chunk[:-1] for chunk in batch]), np.array([chunk[1:] for chunk in batch])
