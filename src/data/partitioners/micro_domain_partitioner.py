import numpy as np
import torch

import torch.utils.data as data

class DomainSubset(data.Subset):
    """A Subset that remembers which micro-domain values it holds."""
 
    def __init__(self, dataset, indices, values):
        super().__init__(dataset, indices)
        self.values = list(values)
 
class MicroDomainPartitioner:
    """Partition a dataset into micro-domains based on a specified attribute."""
 
    def __init__(
            self, 
            dataset: torch.utils.data.Dataset, 
            num_partitions: int,   
            micro_domain: str, 
            shuffle: bool = True, 
            seed: int = 0, 
            pre_partitioned_data: list = None, 
            **kwargs
        ):
       
        self.dataset = dataset
        self.num_partitions = num_partitions
        self.micro_domain = micro_domain
        self.pre_partitioned_data = pre_partitioned_data
        self.partitioned_data = None
        self.seed = seed
        self.shuffle = shuffle
 
    def get_unique_microdomains(self, micro_domain):
        return self.dataset.get_unique_microdomains(micro_domain)
 
    def get_samples_by_microdomain_value(self, micro_domain, value):
        return self.dataset.get_samples_by_microdomain_value(micro_domain, value)
 
    @staticmethod
    def _key(v):
        return v.item() if hasattr(v, "item") else v
 
    @staticmethod
    def _as_value_groups(pre):
        if isinstance(pre, dict):
            return [list(pre[k]) for k in sorted(pre)]
        return [list(p.values) if hasattr(p, "values") else list(p) for p in pre]
 
    def _make_subset(self, micro_domain, values):
        values = list(values)
        if len(values) == 0:
            return DomainSubset(self.dataset, [], [])
        rows = np.concatenate([
            self.get_samples_by_microdomain_value(micro_domain, v) for v in values
        ])
        return DomainSubset(self.dataset, np.sort(rows).tolist(), values)
 
    def load_partitions(self):
        present = np.asarray(self.get_unique_microdomains(self.micro_domain))
        pre = self.pre_partitioned_data
 
        if pre is not None:
            pre_groups = self._as_value_groups(pre)
 
            present_by_key = {self._key(v): v for v in present}   
            claimed = {self._key(v) for g in pre_groups for v in g}
 
            buckets = [
                [present_by_key[self._key(v)] for v in g if self._key(v) in present_by_key]
                for g in pre_groups
            ]
 
            remaining = np.asarray(
                [v for v in present if self._key(v) not in claimed], dtype=present.dtype
            )
            if self.shuffle and len(remaining):
                rng = np.random.default_rng(self.seed)
                remaining = remaining[rng.permutation(len(remaining))]
            for i, extra in enumerate(np.array_split(remaining, self.num_partitions)):
                buckets[i].extend(list(extra))
 
        else:
            values = present
            if self.shuffle:                   
                rng = np.random.default_rng(self.seed)
                values = values[rng.permutation(len(values))]
            if self.num_partitions > len(values):
                print(f"[warn] {self.num_partitions} clients but only {len(values)} "
                      f"'{self.micro_domain}' values -> some clients will be empty.")
            buckets = [list(g) for g in np.array_split(values, self.num_partitions)]
 
        partitions = [self._make_subset(self.micro_domain, b) for b in buckets]
        self.partitioned_data = partitions
        return partitions


