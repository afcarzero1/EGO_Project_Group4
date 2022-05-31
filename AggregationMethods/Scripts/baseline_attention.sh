#!/bin/bash

python3 train_model.py --verbose --epochs 60 --model TSM --temporal_aggregator MultiBaseLine --shift D3-D3 --modality Flow --transpose_input --batch_size 700 --learning_rate 0.001 --frequency_validation 60
python3 train_model.py --verbose --epochs 60 --model TSM --temporal_aggregator MultiBaseLine --shift D2-D2 --modality Flow --transpose_input --batch_size 700 --learning_rate 0.001 --frequency_validation 60
python3 train_model.py --verbose --epochs 60 --model TSM --temporal_aggregator MultiBaseLine --shift D1-D1 --modality Flow --transpose_input --batch_size 700 --learning_rate 0.001 --frequency_validation 60


