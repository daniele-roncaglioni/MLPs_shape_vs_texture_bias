#!/usr/bin/bash
source activate ds-project
wandb login
python "/Users/roncaglionidaniele/Documents/CAS/Deep_Learning/scaling_mlps/train.py" --architecture B_6-Wi_512 \
                                                 --dataset imagenette-160 \
                                                 --resolution 160  \
                                                 --crop_resolution 64 \
                                                 --batch_size 1024 \
                                                 --epochs 605 \
                                                 --save_freq 10 \
                                                 --lr 0.00004 \
                                                 --weight_decay 0.01            \
                                                 --optimizer lion                  \
                                                 --augment                        \
                                                 --smooth 0.3                     \
                                                 --wandb                          \
                                                 --wandb_project shape-vs-texture \
                                                 --calculate_stats 10  \
                                                 --dropout 0.3 \
                                                 --mixup 8.0 \
                                                 --rotation 0 \
                                                --reload "/Users/roncaglionidaniele/Documents/CAS/Deep_Learning/scaling_mlps/train_checkpoints/imagenette-160/wandb_zf1kn42w__epoch_0__compute_0__B_6-Wi_512__imagenette-160__dropout_0.3__rotation_0.0__mixup_8.0__64"
