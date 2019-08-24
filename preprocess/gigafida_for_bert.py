# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import unicodedata
import six
import argparse
import os
import re
import os.path
import shutil

from tensor2tensor.clear_unimportant import clear_unimportant
from tensor2tensor.tensor2tensor.data_generators.text_encoder_build_subword import run_text_encoder


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.
    """
    # six_ensure_text is copied from https://github.com/benjaminp/six
    def six_ensure_text(s, encoding='utf-8', errors='strict'):
        if isinstance(s, six.binary_type):
            return s.decode(encoding, errors)
        elif isinstance(s, six.text_type):
            return s
        else:
            raise TypeError("not expecting type '%s'" % type(s))

    return six_ensure_text(text, encoding="utf-8", errors="ignore")


def run_strip_accents(text):
    """
    Strips accents from a piece of text.
    """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            if char == '̌':
                if output[-1] == 'c':
                    output[-1] = 'č'
                elif output[-1] == 's':
                    output[-1] = 'š'
                elif output[-1] == 'z':
                    output[-1] = 'ž'
            elif char == '́':
                if output[-1] == 'c':
                    output[-1] = 'ć'
            continue
        output.append(char)
    return "".join(output)

def lowercase_and_remove_accent(gigafidaroot, data_type, lowercase=True):
    source_path = os.path.join(gigafidaroot, 'txt_single/', data_type + '.txt')
    dest_path = os.path.join(gigafidaroot, 'txt_single/', data_type + '_lowercase.txt')

    if os.path.exists(dest_path):
        os.remove(dest_path)
        print('Previous file removed!')
    # first_line = True
    with open(dest_path, 'a') as dest_file:
        with open(source_path, 'r') as source_file:
            for line in source_file:
                if line == '\n':
                    dest_file.write(u'\n')
                    continue
                if lowercase:
                    line = convert_to_unicode(line.rstrip().lower())
                    line = run_strip_accents(line)
                    dest_file.write(u'%s\n' % line.lower())
                else:
                    line = convert_to_unicode(line.rstrip())
                    line = run_strip_accents(line)
                    dest_file.write(u'%s\n' % line)


# for line in sys.stdin:
#     line = convert_to_unicode(line.rstrip().lower())
#     line = run_strip_accents(line)
#     print(u'%s' % line.lower())


def copy_txt_data(gigafidaroot):


    for dirpath, dirnames, filenames in os.walk(gigafidaroot):
        for f in filenames:
            if f.endswith('dedup.xml'):
                filepath = os.path.join(dirpath, f)
                outpath = filepath.replace("/tei/", "/txt/")
                outpath = re.sub("GF\d\d/", "", outpath)
                infile = open(filepath, 'r')
                outfile = open(outpath[:-4] + '.txt', 'w')

                reading = False
                body = False
                sentence = ""
                paragraph = ""
                for line in infile:
                    if "<s" in line:
                        sentence = ""
                        reading = True
                    elif "<body>" in line:
                        body = True
                    elif "</body>" in line:
                        body = False
                    elif "</s>" in line:
                        paragraph += sentence[:-1] + '\n'
                        reading = False
                    elif "</p>" in line and body:
                        outfile.write(paragraph + '\n')
                        paragraph = ""
                        reading = False
                    elif "</w>" in line and reading:
                        word = re.search(">(.+?)</w>", line)
                        if args.annotate:
                            postag = re.search('ana="mte:(.+?)"', line)
                            sentence += word.group(1) + '###' + postag.group(1) + ' '
                        else:
                            sentence += word.group(1) + ' '
                    elif "</pc>" in line and reading:
                        punct = re.search(">(.+?)</pc>", line)
                        sentence += punct.group(1) + ' '
                outfile.close()
                infile.close()

def move_files(gigafidaroot):
    directory = os.path.join(gigafidaroot, 'txt/')

    for filename in os.listdir(directory):
        if filename.endswith(".txt") and (filename.startswith("GF95") or filename.startswith("GF96") or
                filename.startswith("GF97") or filename.startswith("GF98") or filename.startswith("GF99")):
            shutil.move(os.path.join(directory, filename), os.path.join(directory, 'eval/', filename))
        elif filename.endswith(".txt"):
            shutil.move(os.path.join(directory, filename), os.path.join(directory, 'train/', filename))

def combine_txt_files_to_one(gigafidaroot, data_type):

    source_path = os.path.join(gigafidaroot, 'txt/', data_type + '/')
    dest_path = os.path.join(gigafidaroot, 'txt_single/', data_type + '.txt')

    if os.path.exists(dest_path):
        os.remove(dest_path)
        print('Previous file removed!')
    # first_line = True
    with open(dest_path, 'a') as dest_file:
        for file in os.listdir(source_path):
            # if not first_line:
            #     r = '\n'
            # else:
            #     r = ''
            #     first_line = False

            filename = os.fsdecode(file)
            path = os.path.join(source_path, filename)
            with open(path) as source_file:
                # dest_file.write(r + source_file.read())
                dest_file.write(source_file.read())


def vocabulary_build_format_data(gigafidaroot, data_type):
    source_path = os.path.join(gigafidaroot, 'txt_single/', data_type + '_lowercase.txt')
    dest_path = os.path.join(gigafidaroot, 'txt_single/', data_type + '_vocab_build.txt')

    if os.path.exists(dest_path):
        os.remove(dest_path)
        print('Previous file removed!')
    # first_line = True
    with open(dest_path, 'a') as dest_file:
        with open(source_path, 'r') as source_file:
            for line in source_file:
                line = line.rstrip()
                for w in line.split():
                    dest_file.write(u'%s\n' % w)

def merge_files(input, input2, output):
    filenames = [input, input2]
    with open(output, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help="gigafida root folder")
    # parser.add_argument('--tokenize', action='store_true')
    parser.add_argument('--annotate', action='store_true')
    args = parser.parse_args()

    copy_txt_data(args.input)
    move_files(args.input)
    combine_txt_files_to_one(args.input, 'eval')
    combine_txt_files_to_one(args.input, 'train')
    lowercase_and_remove_accent(args.input, 'eval', lowercase=False)
    lowercase_and_remove_accent(args.input, 'train', lowercase=False)
    vocabulary_build_format_data(args.input, 'eval')
    vocabulary_build_format_data(args.input, 'train')

    print('Merging split files')
    merge_files(os.path.join(args.input, 'txt_single/eval_vocab_build.txt'),
                os.path.join(args.input, 'txt_single/train_vocab_build.txt'),
                os.path.join(args.input, 'txt_single/vocab_build.txt'))

    print('Running encoder')
    run_text_encoder(os.path.join(args.input, 'txt_single/vocab_build.txt'),
                     os.path.join(args.input, 'txt_single/vocab.txt'))

    print('Running cleanup')
    clear_unimportant(os.path.join(args.input, 'txt_single/vocab.txt'),
                      os.path.join(args.input, 'txt_single/vocab_cleaned.txt'))

    ### Add [UNUSED X], [MASK] etc.


