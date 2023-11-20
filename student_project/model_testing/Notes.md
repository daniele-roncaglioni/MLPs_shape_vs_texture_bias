
srun --pty -A deep_learning -n 4 -G 1 -t 60 bash

python3 finetune.py --architecture B_6-Wi_512 --checkpoint res_64_in21k --dataset cifar10 --data_resolution 32 --batch_size 256 --epochs 10 --lr 0.01 --weight_decay 0.0001 --crop_scale 0.4 1. --crop_ratio 1. 1. --optimizer sgd --augment --mode finetune --smooth 0.3 --skip_tta --wandb --wandb_project shape-vs-texture

