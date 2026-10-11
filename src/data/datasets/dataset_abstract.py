from torch.utils.data.dataset import Dataset
class DinoDataset(Dataset):
    """
    Abstract class.
    """

    def __init__(self, in_c):
        self.in_c = in_c

    @property
    def targets(self):
        return self.get_targets()

    @property
    def num_rows(self):
        return len(self)