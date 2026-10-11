from abc import ABC, abstractmethod
from torch.utils.data.dataset import Dataset

class Partitioner(ABC):
    """
    Abstract base class.
    """

    def __init__(self):
        pass

    @property
    def partitions(self):
        self.load_partitions()