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

# --data_dir="/media/luka/Portable Disk/Datasets/named_entity_recognition/train_test/" --bert_model=bert-base-multilingual-cased --task_name=nersl --output_dir=out_slovene_multilingual --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.4
# --data_dir="/media/luka/Portable Disk/Datasets/named_entity_recognition/train_test/" --bert_model=out_slovene_multilingual --task_name=nersl --output_dir=out_slovene_multilingual --max_seq_length=128 --do_eval

import argparse
import csv
import logging
import os
import pickle
import random
import json
import sys

import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from seqeval.metrics.sequence_labeling import internal_report

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

ud_list = ['', 'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB', 'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PRON', 'SCONJ',
           'PUNCT', 'SYM', 'X', 'PART']

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, ud=None):
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
        self.ud = ud


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ud_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ud_ids = ud_ids


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
        sentence = []
        label = []
    return data


def readfilesl(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''

    df = pd.read_csv(filename, sep='\t')
    # first_sentence_i = df['sentence_id']
    first_sentence_i = df['sentence_id'][0]
    # last_sentence_i = df['sentence_id'].tail(1)
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
            # if data['label'] == float('nan'):
            if not isinstance(data['label'], str) and math.isnan(data['label']):
                labels.append('O')
            else:
                labels.append(data['label'])
        output.append((sentence, labels))

    return output


def readfile_embeddia(filename, ud_translation):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''

    a = [x for x in os.walk(filename)]
    df = pd.DataFrame()
    names = next(os.walk(filename))
    for name in names[1]:
        if name != 'SLO':
            df_filename = os.path.join(filename, name, 'one_file/input_msd.tsv')
            if df.empty:
                df = pd.read_csv(df_filename, sep='\t')
            else:
                # df = df.merge(pd.read_csv(df_filename, sep='\t'))
                new_df = pd.read_csv(df_filename, sep='\t')
                test = df.loc[df.index[-1], 'sentence_id']
                new_df['sentence_id'] = new_df['sentence_id'] + df.loc[df.index[-1], 'sentence_id']
                df = pd.concat([df, new_df])
                df = df.reset_index(drop=True)


    # groups = [df for _, df in df.groupby('sentence_id')]
    # random.shuffle(groups)
    # df = pd.concat(groups).reset_index(drop=True)

    # df = pd.read_csv(filename, sep='\t')
    # first_sentence_i = df['sentence_id']
    first_sentence_i = df['sentence_id'][0]
    # last_sentence_i = df['sentence_id'].tail(1)
    last_sentence_i = df['sentence_id'].tail(1).iloc[0]

    sentence_order = list(range(first_sentence_i, last_sentence_i + 1))
    random.shuffle(sentence_order)

    output = []

    # for i in range(first_sentence_i, last_sentence_i):
    for i in sentence_order:
        df_sentence = df.loc[df['sentence_id'] == i]
        sentence = []
        labels = []
        ud = []
        for _, data in df_sentence.iterrows():
            if isinstance(data['word'], float):
                data['word'] = ''
            sentence.append(data['word'])
            if data['word'] == '"':
                continue
            # if 'msd' in data:
            #     ud.append(data['msd'])
            # if data['label'] == float('nan'):
            if not isinstance(data['label'], str) and math.isnan(data['label']):
                labels.append('O')
            else:
                labels.append(data['label'])
        output.append((sentence, labels, ud))

    return output


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
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

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            readfile_embeddia(data_dir, ud_translation), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            readfile_embeddia(data_dir, ud_translation), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            readfile_embeddia(data_dir, ud_translation), "test")

    def get_labels(self):
        # return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        # return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", 'B-DERIV-PER', 'I-DERIV-PER', "X", "[CLS]", "[SEP]"]
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def get_ud_tags(self):
        # return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        # return ["O", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", 'PART', 'PRON', "PROPN", "PUNCT",
        #         "SCONJ", "SYM", "VERB", "X"]
        return ud_list

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label, ud) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # if math.isnan(sentence):
            #     sentence = ''
            # if i == 1081:
            #     print(i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, ud=ud))
        return examples


class NerSlProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            readfilesl(os.path.join(data_dir, "train_msd.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            readfilesl(os.path.join(data_dir, "test_msd.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            readfilesl(os.path.join(data_dir, "test_msd.tsv")), "test")

    def get_labels(self):
        # return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        return ["O", "loc", "org", "per", "misc", "deriv-per", "X", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # if math.isnan(sentence):
            #     sentence = ''
            # if i == 1081:
            #     print(i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, use_ud=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        if textlist == ['']:
            continue
        labellist = example.label
        if use_ud:
            udlist = example.ud
        tokens = []
        labels = []
        if use_ud:
            uds = []
        # print(textlist)
        # print(labellist)
        parentheses_occurences = 0
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)

            # error hotfix because parentheses should be ignored
            if word == '"':
                parentheses_occurences += 1
                label_1 = 'O'
                if use_ud:
                    ud_1 = "UposTag=ADP|Case=Acc"
            else:
                i = i - parentheses_occurences
                if i >= len(labellist):
                    print('ERROR')
                label_1 = labellist[i]
                if use_ud:
                    ud_1 = udlist[i]
            for m in range(len(token)):
                if use_ud:
                    uds.append(ud_1)
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
        if use_ud:
            ud_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        if use_ud:
            ud_ids.append(0)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])

            if use_ud:
                split = uds[i].split('|')[0].split('=')
                if len(split) > 1:
                    ud_ids.append(ud_map[split[1]])
                else:
                    ud_ids.append(ud_map[split[0]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        if use_ud:
            ud_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)

        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            if use_ud:
                ud_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        if use_ud:
            assert len(ud_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            if use_ud:
                logger.info("ud_ids: %s" % " ".join([str(x) for x in ud_ids]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))


        if use_ud:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              ud_ids=ud_ids))
        else:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids))

    return features


def read_ud_translations(path):
    ud_translation = {}
    df = pd.read_csv(path, sep='\t')
    for index, row in df.iterrows():
        ud_translation[row['msd']] = [row['word_type'], row['ud']]

    return ud_translation


def main():
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
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--use_ud",
                        action='store_true',
                        help="Set this flag if you want to use uds as inputs as well.")
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

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # ud_translation = read_ud_translations(os.path.join(args.data_dir, 'sl_multext-ud'))

    processors = {"ner": NerProcessor,
                  "nersl": NerSlProcessor,
                  "nerembeddia": NerEmbeddiaProcessor}

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

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    ud_tags_list = processor.get_ud_tags()
    num_labels = len(label_list) + 1
    num_ud_dependencies = len(ud_tags_list) + 1

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    if args.use_ud:
        model = BertForTokenClassificationUdExpanded.from_pretrained(args.bert_model,
                                                           cache_dir=cache_dir,
                                                           num_labels=num_labels,
                                                           num_ud_dependencies=num_ud_dependencies)
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
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, use_ud=args.use_ud)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        if args.use_ud:
            all_ud_ids = torch.tensor([f.ud_ids for f in train_features], dtype=torch.long)
        if args.use_ud:
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ud_ids)
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
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                if args.use_ud:
                    input_ids, input_mask, segment_ids, label_ids, ud_ids = batch
                else:
                    input_ids, input_mask, segment_ids, label_ids = batch
                    ud_ids = None
                # loss = model(input_ids, segment_ids, input_mask, label_ids)
                if args.use_ud:
                    loss = model(input_ids, segment_ids, input_mask, label_ids, ud_ids, use_ud=args.use_ud)
                else:
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
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if args.do_eval_in_training:
                eval_examples = processor.get_dev_examples(args.data_dir)
                eval_features = convert_examples_to_features(
                    eval_examples, label_list, args.max_seq_length, tokenizer, use_ud=args.use_ud)
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                if args.use_ud:
                    all_ud_ids = torch.tensor([f.ud_ids for f in eval_features], dtype=torch.long)
                if args.use_ud:
                    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ud_ids)
                else:
                    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                y_true = []
                y_pred = []
                label_map = {i: label for i, label in enumerate(label_list, 1)}
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    if args.use_ud:
                        input_ids, input_mask, segment_ids, label_ids, ud_ids = batch
                    else:
                        input_ids, input_mask, segment_ids, label_ids = batch

                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    if args.use_ud:
                        ud_ids = ud_ids.to(device)

                    with torch.no_grad():
                        if args.use_ud:
                            logits = model(input_ids, segment_ids, input_mask, ud_ids=ud_ids, use_ud=True)
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
                epochs_specific.extend(specific_history)
                epochs_overall.extend(overall_accuracy)
                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    logger.info("\n%s", report)

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                        "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
                        "label_map": label_map}
        json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))


                # writer.write(report)

        # Load a trained model and config that you have fine-tuned
    else:
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        config = BertConfig(output_config_file)
        if args.use_ud:
            model = BertForTokenClassificationUdExpanded(config, num_labels=num_labels,
                                                           num_ud_dependencies=num_ud_dependencies)
        else:
            model = BertForTokenClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))

    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, use_ud=args.use_ud)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        if args.use_ud:
            all_ud_ids = torch.tensor([f.ud_ids for f in eval_features], dtype=torch.long)
        if args.use_ud:
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ud_ids)
        else:
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            if args.use_ud:
                input_ids, input_mask, segment_ids, label_ids, ud_ids = batch
            else:
                input_ids, input_mask, segment_ids, label_ids = batch

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            if args.use_ud:
                ud_ids = ud_ids.to(device)

            with torch.no_grad():
                if args.use_ud:
                    logits = model(input_ids, segment_ids, input_mask, ud_ids=ud_ids, use_ud=True)
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
                            # print(i)
                            # print(j)
                            # print(label_map[label_ids[i][j]])
                            # print(label_map[logits[i][j]])
                            temp_2.append(label_map[logits[i][j]])
                    else:
                        temp_1.pop()
                        temp_2.pop()
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
        report = classification_report(y_true, y_pred, digits=4)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

        # find errors:
        errors = []
        for i, (tru, pre) in enumerate(zip(y_true, y_pred)):
            if tru != pre:
                errors.append((i, eval_examples[i], tru, pre))

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)
        results_history_file = os.path.join(args.output_dir, "results_history.pkl")
        if args.do_eval_in_training:
            with open(results_history_file, "wb") as writer:
                mydict = {'specific_history': specific_history, 'overall_accuracy': overall_accuracy}
                output = open('myfile.pkl', 'wb')
                pickle.dump(mydict, writer)
                output.close()


# "/media/luka/Portable Disk/Datasets/named_entity_recognition/ENG/train_test/eval_pos"
if __name__ == "__main__":
    main()

# --do_train --num_train_epochs 5 --do_eval_in_training
