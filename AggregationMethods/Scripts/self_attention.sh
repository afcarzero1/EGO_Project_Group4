#!/bin/bash

echo "Self attention Script"
echo "D1"
python3 train_model.py --verbose --epochs 60 --model TSM --temporal_aggregator MultiAttention --shift D3-D3 --modality Flow --transpose_input --batch_size 700 --learning_rate 0.0001 --frequency_validation 60
echo "D2"
python3 train_model.py --verbose --epochs 60 --model TSM --temporal_aggregator MultiAttention --shift D2-D2 --modality Flow --transpose_input --batch_size 700 --learning_rate 0.0001 --frequency_validation 60
echo "D3"
python3 train_model.py --verbose --epochs 60 --model TSM --temporal_aggregator MultiAttention --shift D3-D3 --modality Flow --transpose_input --batch_size 700 --learning_rate 0.0001 --frequency_validation 60
