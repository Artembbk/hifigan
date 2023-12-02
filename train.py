import argparse
import collections
import warnings
import itertools

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

from modules import Generator
from modules import MPD
from modules import MSD
import loss as module_loss
from trainer import Trainer


def main(config):

    # dataset = LJspeechDataset("train", config_parser=config)
    # print(dataset[1])

    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    generator = Generator(**config["arch"]["args"]["Generator"])
    msd = MSD()
    mpd = MPD(**config["arch"]["args"]["MPD"])

    logger.info(generator)
    logger.info(msd)
    logger.info(mpd)

    device, device_ids = prepare_device(config["n_gpu"])
    generator = generator.to(device)
    msd = msd.to(device)
    mpd = mpd.to(device)
    if len(device_ids) > 1:
        generator = torch.nn.DataParallel(generator, device_ids=device_ids)
        msd = torch.nn.DataParallel(msd, device_ids=device_ids)
        mpd = torch.nn.DataParallel(mpd, device_ids=device_ids)

    generator_params = filter(lambda p: p.requires_grad, generator.parameters())
    optim_g = config.init_obj(config["optimizer"]["Generator"], torch.optim, generator_params)
    discriminator_params = filter(lambda p: p.requires_grad, itertools.chain(msd.parameters(), mpd.parameters()))
    optim_d = config.init_obj(config["optimizer"]["Discriminator"], torch.optim, discriminator_params)    

    loss_module = config.init_obj(config["loss"], module_loss)
    lr_scheduler_g = config.init_obj(config["lr_scheduler"]["Generator"], torch.optim.lr_scheduler, optim_g)
    lr_scheduler_d = config.init_obj(config["lr_scheduler"]["Discriminator"], torch.optim.lr_scheduler, optim_d)
    

    metrics = config["metrics"]

    trainer = Trainer(
        generator,
        msd,
        mpd,
        loss_module,
        metrics,
        optim_g,
        optim_d,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler_g=lr_scheduler_g,
        lr_scheduler_d=lr_scheduler_d,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


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
