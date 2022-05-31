#!/bin/bash

echo "RGB modality"
python train_model.py --verbose --epochs 600 --model TSM --temporal_aggregator TRN --transpose_input --shift D1-D1 --modality RGB
python train_model.py --verbose --epochs 600 --model TSM --temporal_aggregator TRN --transpose_input --shift D2-D2 --modality RGB
python train_model.py --verbose --epochs 600 --model TSM --temporal_aggregator TRN --transpose_input --shift D2-D2 --modality RGB

python train_model.py --verbose --epochs 600 --model i3d --temporal_aggregator AvgPooling --shift D1-D1 --modality RGB
python train_model.py --verbose --epochs 600 --model i3d --temporal_aggregator AvgPooling --shift D2-D2 --modality RGB
python train_model.py --verbose --epochs 600 --model i3d --temporal_aggregator AvgPooling --shift D3-D3 --modality RGB
