#!/usr/bin/bash
source activate ds-project
wandb login
python "/Users/roncaglionidaniele/Documents/CAS/Deep_Learning/scaling_mlps/train.py" --architecture B_12-Wi_1024 \
                                                 --dataset imagenette-160 \
                                                 --resolution 160  \
                                                 --crop_resolution 64 \
                                                 --batch_size 1024 \
                                                 --epochs 601 \
                                                 --save_freq 10 \
                                                 --lr 0.00006 \
                                                 --weight_decay 0.001            \
                                                 --optimizer lion                  \
                                                 --augment                        \
                                                 --smooth 0.3                     \
                                                 --wandb                          \
                                                 --wandb_project shape-vs-texture \
                                                 --calculate_stats 10  \
                                                 --dropout 0.3 \
                                                 --mixup 8.0 \
                                                 --rotation 20 \
#                                                --reload "/Users/roncaglionidaniele/Documents/CAS/Deep_Learning/scaling_mlps/train_checkpoints/imagenette-160/wandb_ybpul9ay__epoch_160__compute_85879351541760__B_6-Wi_512__imagenette-160__dropout_0.3__rotation_0.0__mixup_0.0__64"
