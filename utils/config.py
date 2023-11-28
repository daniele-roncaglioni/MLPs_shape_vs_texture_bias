import os
import json

from data_utils.data_stats import CLASS_DICT

def style_info():
    dataset_source_path = {"imagenette-160": "../data/imagenette-160/",
                           "tiny-imagenet": "../data/tiny-imagenet-200/"}
    dataset_target_path = {"imagenette-160": "../data/imagenette-160-stylized/",
                           "tiny-imagenet": "../data/tiny-imagenet-200-stylized/"}
    imgsize_target =  {"imagenette-160": 160,
                           "tiny-imagenet": 224}
    resize = {"imagenette-160": False,
                           "tiny-imagenet": True}
    return dataset_source_path, dataset_target_path, imgsize_target, resize

def config_to_name(args):
    return os.path.join(
        str(args.dataset) + '_res_' + str(args.crop_resolution),
        f"{args.model}_{args.architecture}_norm_{args.normalization}",
        f"batchsize_{args.batch_size}",
        f"{args.optimizer}_lr_{args.lr}_smooth_{args.smooth}"
        + f"_decay_{args.weight_decay}_augment_{args.augment}_mixup_{args.mixup}_droprate_{args.drop_rate}",
        f"ntrain_{args.n_train}",
    )


def model_from_config(path):
    """Return model class from checkpoint path."""
    path = os.path.dirname(path)
    with open(path + '/config.txt', 'r') as f:
        config = json.load(f)
    model = config["model"]
    architecture = config["architecture"]
    norm = config["normalization"]
    crop_resolution = int(config["crop_resolution"])

    return model, architecture, crop_resolution, norm


def model_from_checkpoint(checkpoint):
    res = int(checkpoint.split('_')[1])
    num_classes = CLASS_DICT[checkpoint.split('_')[-1]]
    if len(checkpoint.split('_')) == 4:
        pretrained_finetuned = checkpoint.split('_')[-2] + '_' +checkpoint.split('_')[-1]
    else:
        pretrained_finetuned = checkpoint.split('_')[-1]

    return pretrained_finetuned, res, num_classes
