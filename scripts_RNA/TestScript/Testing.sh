#!/bin/bash

python3 test.py --modality RGB --model i3d --num_frames_per_clip_train 16 --num_frames_per_clip_test 16 --resume_from  /home/andres/MLDL/Models --shift D1-D1
python3 test.py --modality RGB --model i3d --num_frames_per_clip_train 16 --num_frames_per_clip_test 16 --resume_from  /home/andres/MLDL/Models --shift D2-D2
python3 test.py --modality RGB --model i3d --num_frames_per_clip_train 16 --num_frames_per_clip_test 16 --resume_from  /home/andres/MLDL/Models --shift D3-D3

python3 test.py --modality Flow --model i3d --num_frames_per_clip_train 16 --num_frames_per_clip_test 16 --resume_from  /home/andres/MLDL/Models --shift D1-D1
python3 test.py --modality Flow --model i3d --num_frames_per_clip_train 16 --num_frames_per_clip_test 16 --resume_from  /home/andres/MLDL/Models --shift D2-D2
python3 test.py --modality Flow --model i3d --num_frames_per_clip_train 16 --num_frames_per_clip_test 16 --resume_from  /home/andres/MLDL/Models --shift D3-D3
python3 test.py --modality RGB --model i3d --num_frames_per_clip_train 16 --num_frames_per_clip_test 16 --resume_from  /home/andres/MLDL/Models --shift D1-D1
python3 test.py --modality RGB --model i3d --num_frames_per_clip_train 16 --num_frames_per_clip_test 16 --resume_from  /home/andres/MLDL/Models --shift D2-D2
python3 test.py --modality RGB --model i3d --num_frames_per_clip_train 16 --num_frames_per_clip_test 16 --resume_from  /home/andres/MLDL/Models --shift D3-D3

python3 test.py --modality Flow --model i3d --num_frames_per_clip_train 16 --num_frames_per_clip_test 16 --resume_from  /home/andres/MLDL/Models --shift D1-D1
python3 test.py --modality Flow --model i3d --num_frames_per_clip_train 16 --num_frames_per_clip_test 16 --resume_from  /home/andres/MLDL/Models --shift D2-D2
python3 test.py --modality Flow --model i3d --num_frames_per_clip_train 16 --num_frames_per_clip_test 16 --resume_from  /home/andres/MLDL/Models --shift D3-D3
