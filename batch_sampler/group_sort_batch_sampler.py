from torch.utils.data import Sampler
import random


class GroupLengthBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_samples = len(data_source)
        self.indices = list(range(self.num_samples))
        self.first_epoch = True

    def __iter__(self):
        if self.first_epoch:
            self.first_epoch = False
            return iter(self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size))
        else:
            random.shuffle(self.indices)
            return iter(self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size))

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


