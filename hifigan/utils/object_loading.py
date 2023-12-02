from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

import hifigan.augmentations
import hifigan.datasets
import hifigan.batch_sampler as batch_sampler_module
from hifigan.collate_fn.collate import collate_fn
from hifigan.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == 'train':
            wave_augs, spec_augs = hifigan.augmentations.from_configs(configs)
            drop_last = True
        else:
            wave_augs, spec_augs = None, None
            drop_last = False

        # create and join datasets
        dataset_list = []
        for ds in params["datasets"]:
            dataset_list.append(configs.init_obj(
                ds, hifigan.datasets, config_parser=configs,
                wave_augs=wave_augs, spec_augs=spec_augs))
        assert len(dataset_list)
        if len(dataset_list) > 1:
            dataset = ConcatDataset(dataset_list)
        else:
            dataset = dataset_list[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
            assert bs <= len(dataset), \
                f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        elif "batch_sampler" in params:
            batch_sampler = configs.init_obj(params["batch_sampler"], batch_sampler_module,
                                             data_source=dataset)
            bs, shuffle, drop_last = 1, False, False
            assert batch_sampler.batch_size <= len(dataset), \
                f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line

        # create dataloader
        print(bs, shuffle, drop_last)
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler, drop_last=drop_last
        )
        dataloaders[split] = dataloader
    return dataloaders
