import os
import json
from pathlib import Path

import torch
import wandb
from torch.nn import CrossEntropyLoss, Linear
from tqdm import tqdm

from data_utils.data_stats import *
from data_utils.dataloader import get_loader
from models.networks import get_model
from utils.parsers import get_finetune_parser
from utils.config import model_from_checkpoint
from utils.metrics import topk_acc, real_acc
from utils.optimizer import (
    get_optimizer,
    get_scheduler,
)
from train import train, test
from datetime import datetime
import random

now = datetime.now()
timestamp = now.strftime("%d-%m-%y, %H:%M")


def whiten_random_pixels(image, num_pixels=16):
    _, _, height, width = image.shape


    if height * width < num_pixels:
        raise ValueError("Das Bild ist zu klein für die Anzahl der geforderten Pixel.")

    pixels = [(random.randint(0, width - 1), random.randint(0, height - 1)) for _ in range(num_pixels)]

    white_pixel = torch.tensor([1.0, 1.0, 1.0])  # oder [255, 255, 255] für [0, 255] Bereich

    for x, y in pixels:
        image[:, :, y, x] = white_pixel

    return image

@torch.no_grad()
def test_time_aug(model, loader, num_augs, args):
    model.eval()
    if hasattr(loader, 'indices'):
        all_preds = torch.zeros(len(loader.indices), model.linear_out.out_features)
    else:
        all_preds = torch.zeros(len(loader) * loader.batch_size, model.linear_out.out_features)

    for _ in tqdm(range(num_augs)):
        
        targets = []
        cnt = 0
        count=0
        for ims, targs in loader:
            
            print(count)

            # 4 corners
            # white_pixel = torch.tensor([1.0, 1.0, 1.0])  # oder [255, 255, 255] für [0, 255] Bereich

            # # Erstellen Sie einen Tensor der richtigen Größe für die Ecken
            # white_area = white_pixel.view(1, 3, 1, 1).expand(-1, -1, 2, 2)

            # # Weisen Sie den weißen Bereich zu den Ecken zu
            # ims[:, :, 0:2, 0:2] = white_area
            # ims[:, :, -2:, 0:2] = white_area
            # ims[:, :, 0:2, -2:] = white_area
            # ims[:, :, -2:, -2:] = white_area
                    # center
            # white_pixel = torch.tensor([1.0, 1.0, 1.0])
            # white_area = white_pixel.view(1, 3, 1, 1).expand(-1, -1, 4, 4)
            # ims[:, :, 30:34, 30:34] = white_area

            # random pixels
            # ims=whiten_random_pixels(ims, num_pixels=16)
            ims = torch.reshape(ims, (ims.shape[0], -1))
            preds = model(ims)

            all_preds[cnt:cnt + ims.shape[0]] += torch.nn.functional.softmax(preds.detach().cpu(), dim=-1)
            targets.append(targs.detach().cpu())

            cnt += ims.shape[0]
            count+=1

    all_preds = all_preds / num_augs
    targets = torch.cat(targets)

    # if args.dataset != 'imagenet_real':
    acc, top5 = topk_acc(all_preds, targets, k=5, avg=True)
    # else:
    #     acc = real_acc(all_preds, targets, k=5, avg=True)
    #     top5 = 0.

    return 100 * acc, 100 * top5



def finetune(args):
    # Use mixed precision matrix multiplication
    torch.backends.cuda.matmul.allow_tf32 = True
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"RUNNING ON {device}")

    pretrained, crop_resolution, num_pretrain_classes = model_from_checkpoint(args.checkpoint)
    model = get_model(architecture=args.architecture, resolution=crop_resolution, num_classes=num_pretrain_classes,
                      checkpoint=pretrained, load_device=device_str)

    model.linear_out = Linear(model.linear_out.in_features, args.num_classes)
    model.to(device)

    if args.checkpoint_path:
        model.load_override(args.checkpoint_path)

    args.crop_resolution = crop_resolution

    # Get the dataloaders
    train_loader = get_loader(
        args.dataset,
        bs=args.batch_size,
        mode='train',
        augment=args.augment,
        dev=device,
        num_samples=args.n_train,
        mixup=args.mixup,
        data_path=args.data_path,
        data_resolution=args.data_resolution,
        crop_resolution=args.crop_resolution,
        crop_ratio=tuple(args.crop_ratio),
        crop_scale=tuple(args.crop_scale)
    )

    test_loader = get_loader(
        args.dataset,
        bs=args.batch_size,
        mode='test',
        augment=False,
        dev=device,
        data_path=args.data_path,
        data_resolution=args.data_resolution,
        crop_resolution=args.crop_resolution,
    )
    if not args.skip_tta:
        test_loader_aug = get_loader(
            args.dataset,
            bs=args.batch_size,
            mode='test',
            augment=True,
            dev=device,
            data_path=args.data_path,
            data_resolution=args.data_resolution,
            crop_resolution=args.crop_resolution,
            crop_ratio=tuple(args.crop_ratio),
            crop_scale=tuple(args.crop_scale)

        )

    param_groups = [
        {
            'params': [v for k, v in model.named_parameters() if 'linear_out' in k],
            'lr': args.lr,
        },
    ]

    if args.mode != "linear":
        param_groups.append(
            {
                'params': [
                    v for k, v in model.named_parameters() if 'linear_out' not in k
                ],
                'lr': args.lr * args.body_learning_rate_multiplier,
            },
        )
    else:
        # freeze the body
        for name, param in model.named_parameters():
            if 'linear_out' not in name:
                param.requires_grad = False

    # Create folder to store the checkpoints
    path = f'{Path(__file__).parent}/{os.path.join(args.checkpoint_folder, args.checkpoint + "_" + args.dataset)}'
    if not os.path.exists(path):
        os.makedirs(path)
        with open(path + '/config.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    opt = get_optimizer(args.optimizer)(param_groups, lr=args.lr)

    scheduler = get_scheduler(opt, args.scheduler, **args.__dict__)
    loss_fn = CrossEntropyLoss(label_smoothing=args.smooth).to(device)

    if args.wandb:
        # Add your wandb credentials and project name
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args.__dict__,
            tags=["finetune", timestamp, args.checkpoint, args.dataset, args.architecture, args.lr, args.weight_decay, args.optimizer],
            dir=f'{Path(__file__).parent}/wandb/'
        )
        wandb.run.name = f'finetune {args.dataset} {args.architecture}'

    for ep in range(args.epochs):
        train_acc, train_top5, train_loss, train_time = train(
            model, opt, scheduler, loss_fn, ep, train_loader, device, args
        )
        if args.wandb:
            wandb.log({"Training time": train_time, "Training loss": train_loss})

        if (ep + 1) % args.calculate_stats == 0:
            test_acc, test_top5, test_loss, test_time = test(
                model, test_loader, loss_fn, device, args
            )

            if args.wandb:
                wandb.log(
                    {
                        "Training accuracy": train_acc,
                        "Training Top 5 accuracy": train_top5,
                        "Test accuracy": test_acc,
                        "Test Top 5 accuracy": test_top5,
                        "Test loss": test_loss,
                        "Inference time": test_time,
                    }
                )

            # Print all the stats
            print('Epoch', ep, '       Time:', train_time)
            print('-------------- Training ----------------')
            print('Average Training Loss:       ', '{:.6f}'.format(train_loss))
            print('Average Training Accuracy:   ', '{:.4f}'.format(train_acc))
            print('Top 5 Training Accuracy:     ', '{:.4f}'.format(train_top5))
            print('---------------- Test ------------------')
            print('Test Accuracy        ', '{:.4f}'.format(test_acc))
            print('Top 5 Test Accuracy          ', '{:.4f}'.format(test_top5))
            print()

        if ep % args.save_freq == 0 and args.save:
            torch.save(
                model.state_dict(),
                path + "/epoch_" + str(ep),
            )

    if not args.skip_tta:
        print('-------- Test Time Augmentation Evaluation -------')

        num_augs = 100
        acc, top5 = test_time_aug(model, test_loader_aug, num_augs, args)
        print(num_augs, 'augmentations: Test accuracy:', acc)
        print(num_augs, 'augmentations: Test Top5 accuracy:', top5)


if __name__ == "__main__":
    parser = get_finetune_parser()
    args = parser.parse_args()

    args.num_classes = CLASS_DICT[args.dataset]

    if args.n_train is None:
        args.n_train = SAMPLE_DICT[args.dataset]

    finetune(args)
