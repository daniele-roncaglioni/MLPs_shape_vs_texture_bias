import os
import time
import json
from pathlib import Path

import torch
import wandb
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# from torchvision import datasets
# from torchvision import transforms

from models import get_architecture
from data_utils.data_stats import *
from data_utils.dataloader import get_loader
from utils.get_compute import get_compute
from utils.metrics import topk_acc, real_acc, AverageMeter
from utils.optimizer import get_optimizer, get_scheduler
from utils.parsers import get_training_parser
from datetime import datetime

now = datetime.now()
timestamp = now.strftime("%d-%m-%y, %H:%M")


# import matplotlib.pyplot as plt

def parse_checkpoint(path):
    split_checkpoint_path = path.split("__")
    checkpoint_data = {}
    for item_str in split_checkpoint_path:
        try:
            item = item_str.split("_")
            checkpoint_data[item[0]] = item[1]
        except:
            pass
    return checkpoint_data


def train(model, opt, scheduler, loss_fn, epoch, train_loader, device, args):
    start = time.time()
    model.train()

    total_acc, total_top5 = AverageMeter(), AverageMeter()
    total_loss = AverageMeter()

    for step, (ims, targs) in enumerate(tqdm(train_loader, desc="Training epoch: " + str(epoch))):
        targs = targs.to(device)
        ims = ims.to(device)

        # print images for debugging
        # idx = 20
        # test_img = ims[idx].clone().detach()
        # test_img = transforms.Normalize(torch.mean(ims[idx]), torch.std(ims[idx]))(test_img)
        # plt.imshow(test_img[2])
        # plt.show()

        ims = torch.reshape(ims, (ims.shape[0], -1))
        preds = model(ims)

        # if args.mixup > 0:
        #     targs_perm = targs[:, 1].long()
        #     weight = targs[0, 2].squeeze()
        #     targs = targs[:, 0].long()
        #     if weight != -1:
        #         loss = loss_fn(preds, targs) * weight + loss_fn(preds, targs_perm) * (
        #                 1 - weight
        #         )
        #     else:
        #         loss = loss_fn(preds, targs)
        #         targs_perm = None
        # else:
        #     loss = loss_fn(preds, targs)
        #     targs_perm = None
        loss = loss_fn(preds, targs)
        targs_perm = None

        acc, top5 = topk_acc(preds, targs, targs_perm, k=5, avg=True, mixup=args.mixup > 0.)
        total_acc.update(acc, ims.shape[0])
        total_top5.update(top5, ims.shape[0])

        loss = loss / args.accum_steps
        loss.backward()

        if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            opt.zero_grad()

        total_loss.update(loss.item() * args.accum_steps, ims.shape[0])

    end = time.time()

    scheduler.step()

    return (
        total_acc.get_avg(percentage=True),
        total_top5.get_avg(percentage=True),
        total_loss.get_avg(percentage=False),
        end - start,
    )


@torch.no_grad()
def test(model, loader, loss_fn, device, args):
    start = time.time()
    model.eval()
    total_acc, total_top5, total_loss = AverageMeter(), AverageMeter(), AverageMeter()

    for ims, targs in tqdm(loader, desc="Evaluation"):
        targs = targs.to(device)
        ims = ims.to(device)
        ims = torch.reshape(ims, (ims.shape[0], -1))
        preds = model(ims)

        if args.dataset != 'imagenet_real':
            acc, top5 = topk_acc(preds, targs, k=5, avg=True, mixup=False)
            loss = loss_fn(preds, targs).item()
        else:
            acc = real_acc(preds, targs, k=5, avg=True)
            top5 = 0
            loss = 0

        total_acc.update(acc, ims.shape[0])
        total_top5.update(top5, ims.shape[0])
        total_loss.update(loss)

    end = time.time()

    return (
        total_acc.get_avg(percentage=True),
        total_top5.get_avg(percentage=True),
        total_loss.get_avg(percentage=False),
        end - start,
    )


def main(args):
    # Use mixed precision matrix multiplication
    torch.backends.cuda.matmul.allow_tf32 = True
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"RUNNING ON {device}")

    model = get_architecture(**args.__dict__).to(device)

    # Count number of parameters for logging purposes
    args.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create unique identifier
    # name = config_to_name(args)
    # path = os.path.join(args.checkpoint_folder, name)

    # Create folder to store the checkpoints
    path = f'{Path(__file__).parent}/train_checkpoints/{args.dataset}/'
    if not os.path.exists(path):
        os.makedirs(path)
        with open(path + '/config.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # Get the dataloaders
    local_batch_size = args.batch_size // args.accum_steps

    train_loader = get_loader(
        args.dataset,
        bs=local_batch_size,
        mode="train",
        augment=args.augment,
        dev=device,
        num_samples=args.n_train,
        mixup=args.mixup,
        data_path=args.data_path,
        data_resolution=args.resolution,
        crop_resolution=args.crop_resolution,
        crop_ratio=tuple(args.crop_ratio),
        crop_scale=tuple(args.crop_scale)
    )

    test_loader = get_loader(
        args.dataset,
        bs=local_batch_size,
        mode="test",
        augment=False,
        dev=device,
        data_path=args.data_path,
        data_resolution=args.resolution,
        crop_resolution=args.crop_resolution
    )

    start_ep = 0
    opt = get_optimizer(args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(opt, args.scheduler, **args.__dict__)

    loss_fn = CrossEntropyLoss(label_smoothing=args.smooth).to(device)

    print("wandb", args.wandb)
    if args.wandb:
        common_kwargs = {
            'project': args.wandb_project,
            'entity': args.wandb_entity,
            'config': args.__dict__,
            'tags': ["pretrain", timestamp, args.dataset, args.architecture, str(args.lr), str(args.weight_decay), args.optimizer, str(args.dropout)],
            'dir': f'{Path(__file__).parent}/wandb/',
        }
        if args.reload:
            try:
                params = torch.load(args.reload) #, map_location=torch.device(device))
                model.load_state_dict(params['model'])
                opt.load_state_dict(params['optimizer'])
                scheduler.load_state_dict(params['lr_sched'])
                checkpoint_data = parse_checkpoint(os.path.split(args.reload)[1]) # args.reload.split("/")[-1])
                start_ep = int(checkpoint_data['epoch'])
                args.epochs = args.epochs + start_ep
                print(f"Reloaded {args.reload}, start epoch: {start_ep}")
            except:
                raise "No pretrained model found"
            wandb.init(
                **common_kwargs,
                id=checkpoint_data['wandb'],
                resume=True,
            )
        else:
            # Add your wandb credentials and project name
            wandb.init(
                **common_kwargs,
            )
        wandb.run.name = f'pretrain {args.dataset} {args.architecture} {args.dropout} rotations20 mixup'
        wandb_run_id = wandb.run.id
    else:
        wandb_run_id = 'NA'

    compute_per_epoch = get_compute(model, args.n_train, args.crop_resolution, device)

    for ep in range(start_ep, args.epochs+1):
        calc_stats = ((ep + 1) % args.calculate_stats == 0) or (ep == 0)

        current_compute = compute_per_epoch * ep

        train_acc, train_top5, train_loss, train_time = train(
            model, opt, scheduler, loss_fn, ep, train_loader, device, args
        )

        if args.wandb:
            wandb.log({"Training time": train_time, "Training loss": train_loss}, ep)

        if ep % args.save_freq == 0 and args.save:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
                'lr_sched': scheduler.state_dict()}
            torch.save(
                checkpoint,
                path + f"/wandb_{wandb_run_id}__epoch_{str(ep)}__compute_{str(current_compute)}__{args.architecture}__{args.dataset}__dropout_{args.dropout}__rotations20__mixup"
            )

        if calc_stats:
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
                    },
                    ep
                )

            # Print all the stats
            print("Epoch", ep, "       Time:", train_time)
            print("-------------- Training ----------------")
            print("Average Training Loss:       ", "{:.6f}".format(train_loss))
            print("Average Training Accuracy:   ", "{:.4f}".format(train_acc))
            print("Top 5 Training Accuracy:     ", "{:.4f}".format(train_top5))
            print("---------------- Test ------------------")
            print("Test Accuracy        ", "{:.4f}".format(test_acc))
            print("Top 5 Test Accuracy          ", "{:.4f}".format(test_top5))
            print()


if __name__ == "__main__":
    parser = get_training_parser()
    args = parser.parse_args()

    args.num_classes = CLASS_DICT[args.dataset]

    if args.n_train is None:
        args.n_train = SAMPLE_DICT[args.dataset]

    if args.crop_resolution is None:
        args.crop_resolution = args.resolution

    main(args)
