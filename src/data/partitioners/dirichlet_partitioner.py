import typing
import numpy as np
import numpy.typing as npt
from torch.utils.data import Subset
import torch


class DirichletDataPartitioner:
    """
    As per https://github.com/adap/flower
    """

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            num_partitions: int,
            alpha: typing.Union[int, float, typing.List[float], np.ndarray],
            min_partition_size: int = 10,
            self_balancing: bool = False,
            shuffle: bool = True,
            seed: typing.Optional[int] = 42,
            custom_partitioner_function=None
    ) -> None:

        self.dataset = dataset
        self._num_partitions = num_partitions
        self._check_num_partitions_greater_than_zero()
        self._alpha = self._initialize_alpha(alpha)
        self._min_partition_size: int = min_partition_size
        self._self_balancing = self_balancing
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)

        self._avg_num_of_samples_per_partition: typing.Optional[float] = None
        self._unique_classes: typing.Optional[typing.Union[typing.List[int], typing.List[str]]] = None
        self._partition_id_to_indices: typing.Dict[int, typing.List[int]] = {}
        self._partition_id_to_indices_determined = False
        self.custom_partitioner_function = custom_partitioner_function

    @property
    def num_partitions(self) -> int:
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    def _check_num_partitions_greater_than_zero(self) -> None:
        if not self._num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")

    def _check_num_partitions_correctness_if_needed(self) -> None:
        if not self._partition_id_to_indices_determined:
            if self._num_partitions > self.dataset.num_rows:
                raise ValueError(
                    "The number of partitions needs to be smaller than the number of samples in the dataset.")

    def _initialize_alpha(self, alpha: typing.Union[int, float, typing.List[float], npt.NDArray[np.float64]]) -> \
    npt.NDArray[np.float64]:
        if isinstance(alpha, int):
            alpha = np.array([float(alpha)], dtype=float).repeat(self._num_partitions)
        elif isinstance(alpha, float):
            alpha = np.array([alpha], dtype=float).repeat(self._num_partitions)
        elif isinstance(alpha, typing.List):
            if len(alpha) != self._num_partitions:
                raise ValueError("If passing alpha as a List, it needs to be of length of equal to num_partitions.")
            alpha = np.asarray(alpha)
        elif isinstance(alpha, np.ndarray):
            if alpha.ndim == 1 and alpha.shape[0] != self._num_partitions:
                raise ValueError(
                    "If passing alpha as an NDArray, its length needs to be of length equal to num_partitions.")
            elif alpha.ndim == 2:
                alpha = alpha.flatten()
                if alpha.shape[0] != self._num_partitions:
                    raise ValueError(
                        "If passing alpha as an NDArray, its size needs to be of length equal to num_partitions.")
        else:
            raise ValueError("The given alpha format is not supported.")
        if not (alpha > 0).all():
            raise ValueError(f"Alpha values should be strictly greater than zero. Instead it'd be converted to {alpha}")
        return alpha

    def _determine_partition_id_to_indices_if_needed(self) -> None:
        if self._partition_id_to_indices_determined:
            return

        if self.custom_partitioner_function is not None:
            self._partition_id_to_indices = self.custom_partitioner_function(self._num_partitions, len(self.dataset),
                                                                             shuffle_indices=True)
        else:
            targets = np.asarray(self.dataset.targets)
            self._unique_classes = np.unique(targets).tolist()
            assert self._unique_classes is not None
            self._avg_num_of_samples_per_partition = len(self.dataset) / self._num_partitions

            sampling_try = 0
            while True:
                partition_id_to_indices: typing.Dict[int, typing.List[int]] = {nid: [] for nid in
                                                                               range(self._num_partitions)}
                for k in self._unique_classes:
                    indices_representing_class_k = np.where(targets == k)[0]
                    class_k_division_proportions = self._rng.dirichlet(self._alpha)
                    nid_to_proportion_of_k_samples = {nid: class_k_division_proportions[nid] for nid in
                                                      range(self._num_partitions)}
                    if self._self_balancing:
                        assert self._avg_num_of_samples_per_partition is not None
                        for nid in nid_to_proportion_of_k_samples.copy():
                            if len(partition_id_to_indices[nid]) > self._avg_num_of_samples_per_partition:
                                nid_to_proportion_of_k_samples[nid] = 0
                        sum_proportions = sum(nid_to_proportion_of_k_samples.values())
                        for nid in nid_to_proportion_of_k_samples:
                            nid_to_proportion_of_k_samples[nid] /= sum_proportions
                    cumsum_division_fractions = np.cumsum(list(nid_to_proportion_of_k_samples.values()))
                    cumsum_division_numbers = cumsum_division_fractions * len(indices_representing_class_k)
                    indices_on_which_split = cumsum_division_numbers.astype(int)[:-1]
                    split_indices = np.split(indices_representing_class_k, indices_on_which_split)
                    for nid in range(self._num_partitions):
                        partition_id_to_indices[nid].extend(split_indices[nid].tolist())
                min_sample_size_on_client = min(len(indices) for indices in partition_id_to_indices.values())
                if min_sample_size_on_client >= self._min_partition_size:
                    break
                sampling_try += 1
                if sampling_try == 10:
                    raise ValueError(
                        "The max number of attempts (10) was reached. Please update the values of alpha and try again.")

            if self._shuffle:
                for indices in partition_id_to_indices.values():
                    self._rng.shuffle(indices)
            self._partition_id_to_indices = partition_id_to_indices

        self._partition_id_to_indices_determined = True

    def load_partition(self, partition_id: int) -> torch.utils.data.Dataset:
        self._determine_partition_id_to_indices_if_needed()
        return Subset(self.dataset, self._partition_id_to_indices[partition_id])

    def load_partitions(self) -> typing.List[torch.utils.data.Dataset]:
        self._determine_partition_id_to_indices_if_needed()
        return [Subset(self.dataset, self._partition_id_to_indices[pid])
                for pid in range(self._num_partitions)]