#!/usr/bin/bash
#SBATCH -A deep_learning
#SBATCH -n 2
#SBATCH -t 120
#SBATCH -G 1
#SBATCH --mem-per-cpu=2048
#SBATCH --tmp=2048
#SBATCH --job-name=dl-finetune
#SBATCH --output=dl-finetune.out
#SBATCH --error=dl-approachX.err
#SBATCH --open-mode=truncate
#SBATCH --mail-user=danieron@student.ethz.ch
#SBATCH --mail-type=ALL
conda activate ds-project
python3 "${HOME}/scaling_mlps_mirror/finetune.py" --architecture B_6-Wi_512 \
                                                 --checkpoint res_64_in21k \
                                                 --dataset cifar10 \
                                                 --data_resolution 32 \
                                                 --batch_size 512 \
                                                 --epochs 400 \
                                                 --lr 0.01 \
                                                 --weight_decay 0.0001            \
                                                 --crop_scale 0.4 1.              \
                                                 --crop_ratio 1. 1.               \
                                                 --optimizer sgd                  \
                                                 --augment                        \
                                                 --mode finetune                  \
                                                 --smooth 0.3                     \
                                                 --skip_tta                       \
                                                 --wandb                          \
                                                 --wandb_project shape-vs-texture \
                                                 --calculate_stats 1
