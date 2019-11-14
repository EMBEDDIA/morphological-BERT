import pandas as pd

from pytorch_pretrained_bert import BasicTokenizer


def modify_df(df, tokenizer):
    df['Data'] = df.apply(lambda x: ' '.join(tokenizer.tokenize(x['Data'])), axis=1)
    return df


def clean_sentences():
    train_df = pd.read_csv('../../comments_sentiment/data/train.tsv', sep='\t', index_col=0)
    dev_df = pd.read_csv('../../comments_sentiment/data/dev.tsv', sep='\t', index_col=0)
    test_df = pd.read_csv('../../comments_sentiment/data/test.tsv', sep='\t', index_col=0)

    basic_tokenizer = BasicTokenizer(do_lower_case=False)

    train_df = modify_df(train_df, basic_tokenizer)
    dev_df = modify_df(dev_df, basic_tokenizer)
    test_df = modify_df(test_df, basic_tokenizer)

    train_df.to_csv('../../comments_sentiment/data/train_cleaned.tsv', sep='\t')
    dev_df.to_csv('../../comments_sentiment/data/dev_cleaned.tsv', sep='\t')
    test_df.to_csv('../../comments_sentiment/data/test_cleaned.tsv', sep='\t')

clean_sentences()
