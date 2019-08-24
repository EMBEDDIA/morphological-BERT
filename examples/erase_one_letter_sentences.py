# coding=utf-8


from __future__ import absolute_import, division, print_function, unicode_literals

# with open('../../data/slwac-tokenized-annotated-corpus/slwac-token-notannot-upper-5M-cleaned.txt', 'w') as nf:
#     with open('../../data/slwac-tokenized-annotated-corpus/slwac-token-notannot-upper-5M.txt', 'r') as f:

repeat = False
with open('../../data/slwac-tokenized-annotated-corpus/slwac-token-annot-upper-5M-cleaned.txt', 'w') as nf:
    with open('../../data/slwac-tokenized-annotated-corpus/slwac-token-annot-lower-5M.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            text = []
            split_sent = l.split()
            for w in split_sent:
                fw = w.split('###')
                text.append(fw[0])
            t = ' '.join(text)
            if len(t) > 5 or len(l) == 1:
                if repeat and len(l) == 1:
                    continue
                nf.write(l)

                if len(l) == 1:
                    repeat = True
                else:
                    repeat = False
