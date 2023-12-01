import argparse
import collections
import warnings

import numpy as np
import torch

# import hw_asr.loss as module_loss
# import hw_asr.metric as module_metric
# import hw_asr.model as module_arch
# from hw_asr.trainer import Trainer
from utils import prepare_device
from utils.object_loading import get_dataloaders
from utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


from datasets import LJspeechDataset
from modules import Generator
from modules import MPD


def main(config):

    # dataset = LJspeechDataset("train", config_parser=config)
    # print(dataset[1])

    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)
    samples = next(iter(dataloaders["train"]))
    spec = samples['spectrogram'][..., :-1]
    wav = samples['audio'].unsqueeze(1)

    generator = Generator(**config["arch"]["args"]["Generator"])
    generated_wav = generator(spec)
    # print(wav.shape)
    # print(generated_wav.shape)
    mpd = MPD(**config["arch"]["args"]["MPD"])
    out_real = mpd(wav)
    print(out_real)
    out_generated = mpd(generated_wav)

    # build model architecture, then print to console
    # model = config.init_obj(config["arch"], module_arch, n_class=len(text_encoder))
    # logger.info(model)

    # # prepare for (multi-device) GPU training
    # device, device_ids = prepare_device(config["n_gpu"])
    # model = model.to(device)
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)

    # # get function handles of loss and metrics
    # loss_module = config.init_obj(config["loss"], module_loss).to(device)
    # metrics = {
    #     "train": [],
    #     "val": []
    # }
    # metrics["train"] = [
    #     config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
    #     for metric_dict in config["metrics"]["train"]
    # ]

    # metrics["val"] = [
    #     config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
    #     for metric_dict in config["metrics"]["val"]
    # ]



    # # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # # disabling scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    # lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    # trainer = Trainer(
    #     model,
    #     loss_module,
    #     metrics,
    #     optimizer,
    #     text_encoder=text_encoder,
    #     config=config,
    #     device=device,
    #     dataloaders=dataloaders,
    #     lr_scheduler=lr_scheduler,
    #     len_epoch=config["trainer"].get("len_epoch", None)
    # )

    # trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
