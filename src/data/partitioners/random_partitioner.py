import numpy as np
import torch

from data.partitioners.partitioner import Partitioner

class RandomPartitioner(Partitioner):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_partitions: int,
        seed: int = 0,
        shuffle: bool = True,
        **kwargs,
    ):
        self.dataset = dataset
        self.num_partitions = num_partitions
        self.partitioned_data = None
        self.seed = seed
        self.shuffle = shuffle
        self.extra_kwargs = kwargs

    def load_partitions(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            indices = indices[rng.permutation(len(indices))]

        if self.num_partitions > len(indices):
            print(f"[warn] {self.num_partitions} clients but only {len(indices)} "
                  f"rows -> some clients will be empty.")

        partitions = [
            torch.utils.data.Subset(self.dataset, np.sort(chunk).tolist())
            for chunk in np.array_split(indices, self.num_partitions)
        ]
        self.partitioned_data = partitions
        return partitions