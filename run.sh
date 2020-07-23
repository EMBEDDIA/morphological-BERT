#!/bin/bash
# languages = ("CRO" "ENG" "EST" "FIN" "LAT" "RUS" "SLO" "SWE")
languages=("SWE")

for i in "${languages[@]}"
do
	python run_ner_cross_validation.py --data_dir="/home/lkrsnik/Development/embeddia/data/$i/cross_validation/" --bert_model=bert-base-multilingual-cased --task_name=nerembeddia --output_dir="/home/lkrsnik/Development/embeddia/data/$i/cross_validation/bert_base" --max_seq_length=128 --do_eval --warmup_proportion=0.4  --num_train_epochs 10 --do_train --train_data_usage 1.0
done
