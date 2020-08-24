# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2019 Embeddia project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import argparse
import configparser
import csv
import logging
import os
import pickle
import random
import json
import sys
from collections import defaultdict

import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
# from seqeval.metrics.sequence_labeling import internal_report
from torch import nn

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig,
                                              BertForTokenClassification, BertForTokenClassificationUdExpanded)
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from seqeval.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

##########################################
# next 4 functions are copied from seqeval
def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end

def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks

def internal_report(y_true, y_pred, digits=2, suffix=False):
    """Build a text report showing the main classification metrics.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
        digits : int. Number of digits for formatting output floating point values.

    Returns:
        report : string. Text summary of the precision, recall, F1 score for each class.

    Examples:
        >>> from seqeval.metrics import classification_report
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> print(classification_report(y_true, y_pred))
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
        avg / total       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    last_line_heading = 'avg / total'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    type_names, ps, rs, f1s, s = [], [], [], [], []
    for type_name, true_entities in d1.items():
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        type_names.append(type_name)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'

    # compute averages
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=width, digits=digits)

    return [type_names, ps, rs, f1s, s], [last_line_heading, np.average(ps, weights=s), np.average(rs, weights=s), np.average(f1s, weights=s), np.sum(s)], report


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

ud_translation = {}
ud_map = {
    '': 0,
    'ADJ': 1,
    'ADV': 2,
    'INTJ': 3,
    'NOUN': 4,
    'PROPN': 5,
    'VERB': 6,
    'ADP': 7,
    'AUX': 8,
    'CCONJ': 9,
    'DET': 10,
    'NUM': 11,
    'PRON': 12,
    'SCONJ': 13,
    'PUNCT': 14,
    'SYM': 15,
    'X': 16,
    'PART': 17
}

universal_features_map = {
    # lexical features
    'PronType': {
        '': 0,
        'Art': 11,
        'Dem': 1,
        'Emp': 2,
        'Exc': 3,
        'Ind': 4,
        'Int': 5,
        'Neg': 6,
        'Prs': 7,
        'Rcp': 8,
        'Rel': 9,
        'Tot': 10,

    },
    'NumType': {
        '': 0,
        'Card': 7,
        'Dist': 1,
        'Frac': 2,
        'Mult': 3,
        'Ord': 4,
        'Range': 5,
        'Sets': 6
    },
    'Poss': {
        '': 0,
        'Yes': 1
    },
    'Reflex': {
        '': 0,
        'Yes': 1
    },
    'Foreign': {
        '': 0,
        'Yes': 1
    },
    'Abbr': {
        '': 0,
        'Yes': 1
    },

    # Inflectional features (nominal)
    'Gender': {
        '': 0,
        'Com': 4,
        'Fem': 1,
        'Masc': 2,
        'Neut': 3
    },
    'Animacy': {
        '': 0,
        'Anim': 4,
        'Hum': 1,
        'Inan': 2,
        'Nhum': 3,
    },
    'NounClass': {
        '': 0,
        'Bantu1': 20,
        'Bantu2': 1,
        'Bantu3': 2,
        'Bantu4': 3,
        'Bantu5': 4,
        'Bantu6': 5,
        'Bantu7': 6,
        'Bantu8': 7,
        'Bantu9': 8,
        'Bantu10': 9,
        'Bantu11': 10,
        'Bantu12': 11,
        'Bantu13': 12,
        'Bantu14': 13,
        'Bantu15': 14,
        'Bantu16': 15,
        'Bantu17': 16,
        'Bantu18': 17,
        'Bantu19': 18,
        'Bantu20': 19
    },
    'Number': {
        '': 0,
        'Coll': 11,
        'Count': 1,
        'Dual': 2,
        'Grpa': 3,
        'Grpl': 4,
        'Inv': 5,
        'Pauc': 6,
        'Plur': 7,
        'Ptan': 8,
        'Sing': 9,
        'Tri': 10
    },
    'Case': {
        '': 0,
        'Abs': 34,
        'Acc': 1,
        'Erg': 2,
        'Nom': 3,
        'Abe': 4,
        'Ben': 5,
        'Cau': 6,
        'Cmp': 7,
        'Cns': 8,
        'Com': 9,
        'Dat': 10,
        'Dis': 11,
        'Equ': 12,
        'Gen': 13,
        'Ins': 14,
        'Par': 15,
        'Tem': 16,
        'Tra': 17,
        'Voc': 18,
        'Abl': 19,
        'Add': 20,
        'Ade': 21,
        'All': 22,
        'Del': 23,
        'Ela': 24,
        'Ess': 25,
        'Ill': 26,
        'Ine': 27,
        'Lat': 28,
        'Loc': 29,
        'Per': 30,
        'Sub': 31,
        'Sup': 32,
        'Ter': 33
    },
    'Definite': {
        '': 0,
        'Com': 5,
        'Cons': 1,
        'Def': 2,
        'Ind': 3,
        'Spec': 4
    },
    'Degree': {
        'Abs': 0,
        'Cmp': 1,
        'Equ': 2,
        'Pos': 3,
        'Sup': 4
    },

    # Inflectional features (verbal)
    'VerbForm': {
        '': 0,
        'Conv': 8,
        'Fin': 1,
        'Gdv': 2,
        'Ger': 3,
        'Inf': 4,
        'Part': 5,
        'Sup': 6,
        'Vnoun': 7
    },
    'Mood': {
        '': 0,
        'Adm': 12,
        'Cnd': 1,
        'Des': 2,
        'Imp': 3,
        'Ind': 4,
        'Jus': 5,
        'Nec': 6,
        'Opt': 7,
        'Pot': 8,
        'Prp': 9,
        'Qot': 10,
        'Sub': 11
    },
    'Tense': {
        '': 0,
        'Fut': 5,
        'Imp': 1,
        'Past': 2,
        'Pqp': 3,
        'Pres': 4
    },
    'Aspect': {
        '': 0,
        'Hab': 6,
        'Imp': 1,
        'Iter': 2,
        'Perf': 3,
        'Prog': 4,
        'Prosp': 5
    },
    'Voice': {
        '': 0,
        'Act': 8,
        'Antip': 1,
        'Cau': 2,
        'Dir': 3,
        'Inv': 4,
        'Mid': 5,
        'Pass': 6,
        'Rcp': 7
    },
    'Evident': {
        '': 0,
        'Fh': 2,
        'Nfh': 1
    },
    'Polarity': {
        '': 0,
        'Neg': 2,
        'Pos': 1
    },
    'Person': {
        '': 0,
        '0': 5,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4
    },
    'Polite': {
        '': 0,
        'Elev': 4,
        'Form': 1,
        'Humb': 2,
        'Infm': 3
    },
    'Clusivity': {
        '': 0,
        'Ex': 2,
        'In': 1
    }
}

universal_features_list = universal_features_map.keys()

ud_list = ['', 'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB', 'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PRON', 'SCONJ',
           'PUNCT', 'SYM', 'X', 'PART']

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, other=None):
        """Constructs a InputExample.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.other = other


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, other_ids=None, fix_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.other_ids = other_ids
        self.fix_ids = fix_ids


def readfile(filename):
    '''
  read file
  return format :
  [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
  '''
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
    return data


def readfilesl(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''

    df = pd.read_csv(filename, sep='\t', keep_default_na=False)
    first_sentence_i = df['sentence_id'][0]
    last_sentence_i = df['sentence_id'].tail(1).iloc[0]

    output = []

    for i in range(first_sentence_i, last_sentence_i):
        df_sentence = df.loc[df['sentence_id'] == i]
        sentence = []
        labels = []
        for _, data in df_sentence.iterrows():
            if isinstance(data['word'], float):
                data['word'] = ''
            sentence.append(data['word'])
            if not isinstance(data['label'], str) and math.isnan(data['label']):
                labels.append('O')
            else:
                labels.append(data['label'])
        output.append((sentence, labels))

    return output


def readfile_embeddia(filename, cv_part):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    if cv_part != -1:
        filename = filename % cv_part
    df = pd.read_csv(filename, sep='\t', keep_default_na=False)
    df = df.fillna('')
    first_sentence_i = df['sentence_id'][0]
    last_sentence_i = df['sentence_id'].tail(1).iloc[0]

    output = []

    for i in range(first_sentence_i, last_sentence_i):
        df_sentence = df.loc[df['sentence_id'] == i]
        sentence = []
        labels = []
        others = []
        for _, data in df_sentence.iterrows():
            if isinstance(data['word'], float):
                data['word'] = ''
            sentence.append(data['word'])
            if data['word'] == '"':
                continue
            other = {}
            if 'msd' in data:
                other['msd'] = data['msd']
            if 'upos' in data:
                other['upos'] = data['upos']
            if 'feats' in data:
                other['feats'] = data['feats']
            if 'xpos' in data:
                other['xpos'] = data['xpos']
            if 'lemma' in data:
                other['lemma'] = data['lemma']
            if 'dependency_relation' in data:
                other['dependency_relation'] = data['dependency_relation']
            if 'prefixes' in data:
                other['prefixes'] = data['prefixes']
            if 'suffixes' in data:
                other['suffixes'] = data['suffixes']

            others.append(other)
            if not isinstance(data['label'], str) and math.isnan(data['label']):
                labels.append('O')
            else:
                labels.append(data['label'])
        output.append((sentence, labels, others))

    return output


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, cv_part, folds, partial_train_data_usage=1.0):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, cv_part):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerEmbeddiaProcessor(DataProcessor):
    """Processor for the formatted embeddia data sets."""

    def get_train_examples(self, data_dir, cv_part, folds, partial_train_data_usage=1.0):
        """See base class."""

        if cv_part == -1:
            data = readfile_embeddia(os.path.join(data_dir, "train_msd.tsv"), -1)
        else:
            # always ignore cv_part. In order to get same amount of ~same amount of data when learning if cv_part <= data_
            # used_parts add 1 part to train data
            data_used_parts = int(folds * partial_train_data_usage)
            if data_used_parts == folds:
                num_train_parts = folds
            else:
                if cv_part <= data_used_parts:
                    num_train_parts = data_used_parts + 1
                else:
                    num_train_parts = data_used_parts
            data = []
            for i in range(1, num_train_parts + 1):
                if i != cv_part:
                    data.extend(readfile_embeddia(os.path.join(data_dir, "ext_%d_msd.tsv"), i))
        return self._create_examples(
            data, "train")

    def get_dev_examples(self, data_dir, cv_part):
        """See base class."""
        if cv_part == -1:
            data = readfile_embeddia(os.path.join(data_dir, "test_msd.tsv"), -1)
        else:
            data = readfile_embeddia(os.path.join(data_dir, "ext_%d_msd.tsv"), cv_part)
        return self._create_examples(data, "dev")

    def get_test_examples(self, data_dir, cv_part):
        """See base class."""
        if cv_part == -1:
            data = readfile_embeddia(os.path.join(data_dir, "test_msd.tsv"), -1)
        else:
            data = readfile_embeddia(os.path.join(data_dir, "ext_%d_msd.tsv"), cv_part)
        return self._create_examples(data, "test")

    def get_labels(self):
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def get_ud_tags(self):
        return ud_list

    def get_prefix_tags(self):
        return prefix_list

    def get_suffix_tags(self):
        return suffix_list

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label, other) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, other=other))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, other=None):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        if textlist == ['']:
            continue
        labellist = example.label
        if other['upos'] or other['feats']:
            otherlist = example.other
        tokens = []
        labels = []
        others = {}
        if other['upos']:
            others['upos'] = []
        if other['feats']:
            others['feats'] = []
        if other['suffixes']:
            others['suffixes'] = []
        if other['prefixes']:
            others['prefixes'] = []
        parentheses_occurences = 0
        for i, word in enumerate(textlist):
            if i >= len(labellist):
                continue

            token = tokenizer.tokenize(word)
            tokens.extend(token)

            # error hotfix because parentheses should be ignored
            if word == '"':
                parentheses_occurences += 1
                label_1 = 'O'
                if other['upos']:
                    upos = "ADP"
                if other['feats']:
                    feats = "_"
            else:
                i = i - parentheses_occurences
                label_1 = labellist[i]
                if other['upos']:
                    upos = otherlist[i]['upos']
                if other['feats']:
                    feats = otherlist[i]['feats']
                if other['prefixes']:
                    prefixes = otherlist[i]['prefixes']
                if other['suffixes']:
                    suffixes = otherlist[i]['suffixes']
            for m in range(len(token)):
                if other['upos']:
                    others['upos'].append(upos)
                if other['feats']:
                    others['feats'].append(feats)
                if other['prefixes']:
                    others['prefixes'].append(prefixes)
                if other['suffixes']:
                    others['suffixes'].append(suffixes)
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        # create array of arrays of universal_feature_map and upos
        if other['upos'] or other['feats']:
            # for beginning tag use 0 - for separators
            if other['feats']:
                other_ids = [[0] for i in range(len(universal_features_map) + 1)]
            else:
                other_ids = [[0] for i in range(1)]

        if other['prefixes']:
            fix_ids = [[0], [0]]
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])

            if other['upos'] or other['feats']:
                this_feat_dict = {}
                if other['feats'] and others['feats'][i] != '_':
                    word_feats = others['feats'][i].split('|')
                    for feat in word_feats:
                        feat_split = feat.split('=')
                        this_feat_dict[feat_split[0]] = feat_split[1]
                if other['upos']:
                    upos_append = others['upos'][i]
                    other_ids[0].append(ud_map[upos_append])
                else:
                    other_ids[0].append(0)
                # add 0 if no feature of specific type is given or correct index of feature if it is given
                if other['feats']:
                    for index, key in enumerate(universal_features_map):
                        if key in this_feat_dict and this_feat_dict[key] in universal_features_map[key]:
                            other_ids[index + 1].append(universal_features_map[key][this_feat_dict[key]])
                        else:
                            other_ids[index + 1].append(0)

            if other['fixes']:
                prefixes_append = others['prefixes'][i]
                suffixes_append = others['suffixes'][i]
                fix_ids[0].append(prefix_map[prefixes_append])
                fix_ids[1].append(suffix_map[suffixes_append])

        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        if other['upos'] or other['feats']:
            for i in range(len(other_ids)):
                other_ids[i].append(0)
        if other['fixes']:
            fix_ids[0].append(0)
            fix_ids[1].append(0)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)

        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            if other['upos'] or other['feats']:
                for i in range(len(other_ids)):
                    other_ids[i].append(0)
            if other['fixes']:
                fix_ids[0].append(0)
                fix_ids[1].append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        if other['upos'] or other['feats']:
            assert len(other_ids[0]) == max_seq_length
        if other['fixes']:
            assert len(fix_ids[0]) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     if other['upos']:
        #         logger.info("ud_ids: %s" % " ".join([str(x) for x in other_ids[0]]))
        #     if other['fixes']:
        #         logger.info("fix_ids: %s" % " ".join([str(x) for x in fix_ids[0]]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))


        if other['upos'] or other['feats']:
            if not other['fixes']:
                fix_ids = None
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              other_ids=other_ids,
                              fix_ids=fix_ids))
        else:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids))

    return features


def read_ud_translations(path):
    ud_translation = {}
    df = pd.read_csv(path, sep='\t', keep_default_na=False)
    for index, row in df.iterrows():
        ud_translation[row['msd']] = [row['word_type'], row['ud']]

    return ud_translation

prefix_map = {}
suffix_map = {}
prefix_list = []
suffix_list = []
test = ['a']
def main():
    def bert_cross_validation_iteration(cross_validation_part, partial_train_data_usage):
        global universal_features_map
        if args.server_ip and args.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd
            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
            ptvsd.wait_for_attach()

        processors = {"nerembeddia": NerEmbeddiaProcessor}

        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16))

        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if not args.do_train and not args.do_eval:
            raise ValueError("At least one of `do_train` or `do_eval` must be True.")

        if os.path.exists(args.output_dir) and os.path.exists(args.output_dir + '/cv_%d' % cross_validation_part) and os.listdir(args.output_dir + '/cv_%d' % cross_validation_part) and args.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir + '/cv_%d' % cross_validation_part))
        if not os.path.exists(args.output_dir + '/cv_%d' % cross_validation_part):
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            os.makedirs(args.output_dir + '/cv_%d' % cross_validation_part)

        task_name = args.task_name.lower()

        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))

        processor = processors[task_name]()
        label_list = processor.get_labels()
        ud_tags_list = processor.get_ud_tags()
        prefix_tags_list = processor.get_prefix_tags()
        suffix_tags_list = processor.get_suffix_tags()
        num_labels = len(label_list) + 1
        num_ud_dependencies = 0
        if args.upos:
            num_ud_dependencies = len(ud_tags_list) + 1
        if not args.feats:
            universal_features_map = None

        num_prefixes = len(prefix_tags_list) + 1
        num_suffixes = len(suffix_tags_list) + 1

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

        train_examples = None
        num_train_optimization_steps = None
        if args.do_train:
            train_examples = processor.get_train_examples(args.data_dir, cross_validation_part, args.folds, partial_train_data_usage)
            num_train_optimization_steps = int(
                len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
            if args.local_rank != -1:
                num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed_{}'.format(args.local_rank))
        if args.upos or args.feats:
            model = BertForTokenClassificationUdExpanded.from_pretrained(args.bert_model,
                                                                         cache_dir=cache_dir,
                                                                         num_labels=num_labels,
                                                                         upos_num=num_ud_dependencies,
                                                                         prefix_num=num_prefixes,
                                                                         suffix_num=num_suffixes,
                                                                         others_map=universal_features_map)
        else:
            model = BertForTokenClassification.from_pretrained(args.bert_model,
                                                               cache_dir=cache_dir,
                                                               num_labels=num_labels)
        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

        global_step = 0
        if args.do_train:
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer, other=other_features)
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            if args.upos:

                all_other_ids = [torch.tensor([f.other_ids[other_i] for f in train_features], dtype=torch.long) for other_i in range(len(train_features[0].other_ids))]

                if args.fixes:
                    all_prefixes_ids = torch.tensor([f.fix_ids[0] for f in train_features], dtype=torch.long)
                    all_suffixes_ids = torch.tensor([f.fix_ids[1] for f in train_features], dtype=torch.long)
                    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_prefixes_ids, all_suffixes_ids,
                                               *all_other_ids)
                else:
                    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, *all_other_ids)
            else:
                train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

            model.train()

            if args.do_eval_in_training:
                epochs_specific = []
                epochs_overall = []

            for _ in trange(int(args.num_train_epochs), desc="Epoch"):
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=True)):
                    batch = tuple(t.to(device) for t in batch)
                    if args.upos:
                        if args.fixes:
                            input_ids, input_mask, segment_ids, label_ids, prefix_ids, suffix_ids, *other_ids = batch
                        else:
                            input_ids, input_mask, segment_ids, label_ids, *other_ids = batch
                            prefix_ids = None
                            suffix_ids = None
                        loss = model(input_ids, segment_ids, input_mask, label_ids, other_ids, prefix_ids, suffix_ids, use_ud=args.upos, use_fixes=args.fixes)
                    else:
                        input_ids, input_mask, segment_ids, label_ids = batch
                        ud_ids = None
                        loss = model(input_ids, segment_ids, input_mask, label_ids)

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            # modify learning rate with special warm up BERT uses
                            # if args.fp16 is False, BertAdam is used that handles this automatically
                            lr_this_step = args.learning_rate * warmup_linear(
                                global_step / num_train_optimization_steps,
                                args.warmup_proportion)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                if args.do_eval_in_training:
                    eval_examples = processor.get_dev_examples(args.data_dir, cross_validation_part)
                    eval_features = convert_examples_to_features(
                        eval_examples, label_list, args.max_seq_length, tokenizer, other=other_features)
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)
                    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

                    if args.upos:
                        all_other_ids = [torch.tensor([f.other_ids[other_i] for f in eval_features], dtype=torch.long)
                                         for other_i in range(len(eval_features[0].other_ids))]
                        if args.fixes:
                            all_prefixes_ids = torch.tensor([f.fix_ids[0] for f in eval_features], dtype=torch.long)
                            all_suffixes_ids = torch.tensor([f.fix_ids[1] for f in eval_features], dtype=torch.long)
                            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                                       all_prefixes_ids, all_suffixes_ids,
                                                       *all_other_ids)
                        else:
                            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                                      *all_other_ids)
                    else:
                        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

                    # Run prediction for full data
                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
                    model.eval()
                    y_true = []
                    y_pred = []
                    label_map = {i: label for i, label in enumerate(label_list, 1)}
                    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):
                        if args.upos:
                            if args.fixes:
                                input_ids, input_mask, segment_ids, label_ids, prefix_ids, suffix_ids, *other_ids = batch
                            else:
                                input_ids, input_mask, segment_ids, label_ids, *other_ids = batch
                                prefix_ids = None
                                suffix_ids = None
                        else:
                            input_ids, input_mask, segment_ids, label_ids = batch

                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)
                        if args.upos:
                            new_other_ids = []
                            for other_id in other_ids:
                                new_other_ids.append(other_id.to(device))
                            other_ids = new_other_ids

                        if args.fixes:
                            prefix_ids = prefix_ids.to(device)
                            suffix_ids = suffix_ids.to(device)

                        with torch.no_grad():
                            if args.upos:
                                if args.fixes:
                                    logits = model(input_ids, segment_ids, input_mask, other_ids=other_ids, prefix_ids=prefix_ids, suffix_ids=suffix_ids, use_ud=args.upos, use_fixes=args.fixes)
                                else:
                                    logits = model(input_ids, segment_ids, input_mask, other_ids=other_ids, prefix_ids=None, suffix_ids=None
                                                   , use_ud=args.upos, use_fixes=args.fixes)
                            else:
                                logits = model(input_ids, segment_ids, input_mask)

                        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        input_mask = input_mask.to('cpu').numpy()
                        for i, mask in enumerate(input_mask):
                            temp_1 = []
                            temp_2 = []
                            for j, m in enumerate(mask):
                                if j == 0:
                                    continue
                                if m:
                                    if label_map[label_ids[i][j]] != "X":
                                        temp_1.append(label_map[label_ids[i][j]])
                                        temp_2.append(label_map[logits[i][j]])
                                else:
                                    temp_1.pop()
                                    temp_2.pop()
                                    y_true.append(temp_1)
                                    y_pred.append(temp_2)
                                    break
                    report = classification_report(y_true, y_pred, digits=4)
                    # epochs_specific.append(specific_history)
                    # epochs_overall.append(overall_accuracy)
                    output_eval_file = os.path.join(args.output_dir + '/cv_%d' % cross_validation_part, "eval_results.txt")
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Eval results *****")
                        logger.info("\n%s", report)

            # Save a trained model and the associated configuration
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir + '/cv_%d' % cross_validation_part, WEIGHTS_NAME)
            # torch.save(model_to_save.state_dict(), output_model_file)
            output_config_file = os.path.join(args.output_dir + '/cv_%d' % cross_validation_part, CONFIG_NAME)
            # with open(output_config_file, 'w') as f:
            #     f.write(model_to_save.config.to_json_string())
            label_map = {i: label for i, label in enumerate(label_list, 1)}
            # model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
            #                 "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
            #                 "label_map": label_map}
            # json.dump(model_config, open(os.path.join(args.output_dir + '/cv_%d' % cross_validation_part, "model_config.json"), "w"))

        model.to(device)

        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            eval_examples = processor.get_dev_examples(args.data_dir, cross_validation_part)
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, other=other_features)
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            if args.upos:
                all_other_ids = [torch.tensor([f.other_ids[other_i] for f in eval_features], dtype=torch.long)
                                 for other_i in range(len(eval_features[0].other_ids))]

                if args.fixes:
                    all_prefixes_ids = torch.tensor([f.fix_ids[0] for f in eval_features], dtype=torch.long)
                    all_suffixes_ids = torch.tensor([f.fix_ids[1] for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                              all_prefixes_ids, all_suffixes_ids,
                                              *all_other_ids)
                else:
                    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                              *all_other_ids)
            else:
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            model.eval()
            y_true = []
            y_pred = []
            label_map = {i: label for i, label in enumerate(label_list, 1)}
            for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):
                if args.upos:
                    if args.fixes:
                        input_ids, input_mask, segment_ids, label_ids, prefix_ids, suffix_ids, *other_ids = batch
                    else:
                        input_ids, input_mask, segment_ids, label_ids, *other_ids = batch
                        prefix_ids = None
                        suffix_ids = None
                else:
                    input_ids, input_mask, segment_ids, label_ids = batch

                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                if args.upos:
                    new_other_ids = []
                    for other_id in other_ids:
                        new_other_ids.append(other_id.to(device))
                    other_ids = new_other_ids

                if args.fixes:
                    prefix_ids = prefix_ids.to(device)
                    suffix_ids = suffix_ids.to(device)

                with torch.no_grad():
                    if args.upos:
                        if args.fixes:
                            logits = model(input_ids, segment_ids, input_mask, other_ids=other_ids,
                                           prefix_ids=prefix_ids, suffix_ids=suffix_ids, use_ud=args.upos,
                                           use_fixes=args.fixes)
                        else:
                            logits = model(input_ids, segment_ids, input_mask, other_ids=other_ids, prefix_ids=None,
                                           suffix_ids=None, use_ud=args.upos, use_fixes=args.fixes)
                    else:
                        logits = model(input_ids, segment_ids, input_mask)

                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
                for i, mask in enumerate(input_mask):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(mask):
                        if j == 0:
                            continue
                        if m:
                            if label_map[label_ids[i][j]] != "X":
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])
                        else:
                            temp_1.pop()
                            temp_2.pop()
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            break
            specific_history, overall_accuracy, report = internal_report(y_true, y_pred, digits=4)
            output_eval_file = os.path.join(args.output_dir + '/cv_%d' % cross_validation_part, "eval_results.txt")

            # find errors:
            errors = []
            for i, (tru, pre) in enumerate(zip(y_true, y_pred)):
                if tru != pre:
                    errors.append((i, eval_examples[i], tru, pre))

            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                logger.info("\n%s", report)
                writer.write(report)
            # results_history_file = os.path.join(args.output_dir + '/cv_%d' % cross_validation_part, "results_history.pkl")
            mydict = {'specific_history': specific_history, 'overall_accuracy': overall_accuracy}
            # if args.do_eval_in_training:
            #     with open(results_history_file, "wb") as writer:
            #         output = open('myfile.pkl', 'wb')
            #         pickle.dump(mydict, writer)
            #         output.close()

            return mydict
        return None
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--config",
                        default='config.ini',
                        type=str,
                        help="Path to config.ini file.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval_in_training",
                        action='store_true',
                        help="Whether to do evaluation after each epoch during training.")
    parser.add_argument("--train_data_usage",
                        default=1.0,
                        type=float,
                        help="% of data used for training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    args.upos = config.getboolean('settings', 'upos')
    args.feats = config.getboolean('settings', 'feats')
    args.fixes = config.getboolean('settings', 'fixes')
    args.fixes_path = config.get('settings', 'fixes_path')
    args.cross_validation = config.getboolean('settings', 'cross_validation')
    args.train_test = config.getboolean('settings', 'train_test')
    args.folds = config.getint('settings', 'folds')

    other_features = {
        'upos': config.getboolean('settings', 'upos'),
        'feats': config.getboolean('settings', 'feats'),
        'xpos': config.getboolean('settings', 'xpos'),
        'lemma': config.getboolean('settings', 'lemma'),
        'dependency_relation': config.getboolean('settings', 'dependency_relation'),
        'fixes': config.getboolean('settings', 'fixes'),
        'prefixes': config.getboolean('settings', 'fixes'),
        'suffixes': config.getboolean('settings', 'fixes'),
    }

    if args.fixes:
        # with open() as fh:
        # all_prefixes = []
        global prefix_map
        global suffix_map
        global prefix_list
        global suffix_list
        max_prefix_len = 0
        with open(os.path.join(args.fixes_path, 'prefixes.csv'), 'r') as csvFile:
            reader = csv.reader(csvFile)
            first_el = True
            for row in reader:
                # erase predponsko
                if first_el:
                    first_el = False
                    continue
                # erase prefixes with length smaller than 2 characters
                if len(row[0]) > 2:
                    # add new prefix and erase _
                    prefix_list.append(row[0][:-1])

                # find longest prefix
                if len(row[0]) - 1 > max_prefix_len:
                    max_prefix_len = len(row[0]) - 1

        prefix_list = [''] + prefix_list

        max_suffix_len = 0
        with open(os.path.join(args.fixes_path, 'suffixes.csv'), 'r') as csvFile:
            reader = csv.reader(csvFile)
            first_el = True
            for row in reader:
                # erase priponsko
                if first_el:
                    first_el = False
                    continue

                # erase suffixes with length smaller than 2 characters
                if len(row[0]) > 2:
                    # add new suffix and erase _
                    suffix_list.append(row[0][1:])

                if len(row[0]) - 1 > max_suffix_len:
                    max_suffix_len = len(row[0]) - 1

        suffix_list = [''] + suffix_list

        prefix_map = {val: i for i, val in enumerate(prefix_list)}
        suffix_map = {val: i for i, val in enumerate(suffix_list)}

    def print_report(accuracies, digits=4):
        name_width = 11

        last_line_heading = 'avg / total'
        width = max(name_width, len(last_line_heading))

        headers = ["precision", "recall", "f1-score", "support"]
        head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
        report = head_fmt.format(u'', *headers, width=width)
        report += u'\n\n'

        row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

        for k, v in accuracies.items():
            report += row_fmt.format(*[k, v[0], v[1], v[2], v[3]], width=width, digits=digits)
        report += u'\n'
        return report

    accuracies = []

    # actually executing code
    if args.train_test:
        accuracies.append(bert_cross_validation_iteration(-1, args.train_data_usage))
    else:
        for i in range(1, args.folds + 1):
            accuracies.append(bert_cross_validation_iteration(i, args.train_data_usage))
            if not args.cross_validation:
                break

    # testing on one part of cross validation
    # accuracies.append(bert_cross_validation_iteration(1))

    accuracies_sum = {att:[0, 0, 0, 0] for att in accuracies[0]['specific_history'][0]}
    accuracies_sum['avg / total'] = [0, 0, 0, 0]
    for accuracy in accuracies:
        #save specific sums
        for i, att_name in enumerate(accuracy['specific_history'][0]):
            accuracies_sum[att_name][0] += accuracy['specific_history'][1][i] * accuracy['specific_history'][4][i]
            accuracies_sum[att_name][1] += accuracy['specific_history'][2][i] * accuracy['specific_history'][4][i]
            accuracies_sum[att_name][2] += accuracy['specific_history'][3][i] * accuracy['specific_history'][4][i]
            accuracies_sum[att_name][3] += accuracy['specific_history'][4][i]
        # save total sum
        accuracies_sum['avg / total'][0] += accuracy['overall_accuracy'][1] * accuracy['overall_accuracy'][4]
        accuracies_sum['avg / total'][1] += accuracy['overall_accuracy'][2] * accuracy['overall_accuracy'][4]
        accuracies_sum['avg / total'][2] += accuracy['overall_accuracy'][3] * accuracy['overall_accuracy'][4]
        accuracies_sum['avg / total'][3] += accuracy['overall_accuracy'][4]
    final_accuracy = {}
    for k, v in accuracies_sum.items():
        final_accuracy[k] = [v[0]/v[3], v[1]/v[3], v[2]/v[3], v[3]]

    report_final = print_report(final_accuracy)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

    logger.info("***** Eval results *****")
    logger.info("\n%s", report_final)


    with open(output_eval_file, "w") as writer:
        writer.write(report_final)

if __name__ == "__main__":
    main()
