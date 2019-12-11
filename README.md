# PyTorch Pretrained BERT: The Big & Extending Repository of pretrained Transformers
This repository is a clone of https://github.com/huggingface/transformers . The original code was modified for testing addition of morphological data to input of BERT.
The majority of functionalities are the same as in original repository, with some additions.

After installing all libraries from requirements.txt, you can test this algorithm via following command:

```
python run_ner_cross_validation.py --data_dir=<PATH TO DIR THAT CONTAINS train_msd.tsv AND test_msd.tsv> --bert_model=bert-base-multilingual-cased --task_name=nerembeddia --output_dir=<PATH TO RESULTS DIR THAT WILL BE CREATED> --max_seq_length=128 --do_eval --warmup_proportion=0.4  --num_train_epochs 10 --do_train --train_data_usage 1.0
```