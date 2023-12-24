import argparse
import collections
import warnings
import itertools

import numpy as np
import torch
import torchaudio

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
from melspec import *


def main(config):

    # dataset = LJspeechDataset("train", config_parser=config)
    # print(dataset[1])

    logger = config.get_logger("train")

    generator = Generator(**config["arch"]["args"]["Generator"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    generator.load_state_dict(state_dict)

    generator = generator.to(device)
    generator.eval()

    generator = generator.to(device)

    mel_specer = MelSpectrogram(MelSpectrogramConfig())
    for i in range(1, 4):
        audio_wave, sr = torchaudio.load(f"test/audio_{i}.wav")
        audio_wave = audio_wave[0:1, :] 
        target_sr = 22050
        if sr != target_sr:
            audio_wave = torchaudio.functional.resample(audio_wave, sr, target_sr)
        new_length = (audio_wave.shape[1] // 256) * 256
        audio_wave = audio_wave[:, :new_length].unsqueeze(1)
        mel = mel_specer(audio_wave)[..., :-1].squeeze(1).to(device)
        generated_audio = generator(mel)

        torchaudio.save(f"test/generated_{i}.wav", generated_audio.cpu().squeeze(1), 22050)


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
