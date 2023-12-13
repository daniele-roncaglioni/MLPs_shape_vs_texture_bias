#!/usr/bin/bash
#SBATCH -A deep_learning
#SBATCH -n 2
#SBATCH -t 120
#SBATCH -G 1
#SBATCH --mem-per-cpu=2048
#SBATCH --tmp=2048
#SBATCH --job-name=dl-finetune
#SBATCH --output=train_imagenette.out
#SBATCH --error=train_imagenette.err
#SBATCH --open-mode=truncate
#SBATCH --mail-user=danieron@student.ethz.ch
#SBATCH --mail-type=ALL
"${HOME}/scaling_mlps_mirror/.ds-venv/bin/wandb" login
"${HOME}/scaling_mlps_mirror/.ds-venv/bin/python" "${HOME}/scaling_mlps_mirror/train.py" --architecture B_6-Wi_512 \
                                                 --dataset imagenette-160 \
                                                 --resolution 160  \
                                                 --crop_resolution 64 \
                                                 --batch_size 512 \
                                                 --epochs 205 \
                                                 --save_freq 20 \
                                                 --lr 0.01 \
                                                 --weight_decay 0.0001            \
                                                 --optimizer sgd                  \
                                                 --augment                        \
                                                 --smooth 0.3                     \
                                                 --wandb                          \
                                                 --wandb_project shape-vs-texture \
                                                 --calculate_stats 20  \
                                                 --reload "${HOME}/scaling_mlps_mirror/train_checkpoints/imagenette-160/epoch_160_compute_85879351541760"
