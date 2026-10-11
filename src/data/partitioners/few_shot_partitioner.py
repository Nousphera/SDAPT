import typing

import numpy as np
from torch.utils.data import Subset


class FewShotDataPartitioner:
    """
    Greedily builds `num_partitions` few-shot subsets in which every class
    is represented (at least) `k_shot` times.
    """

    def __init__(
            self,
            dataset,
            num_partitions: int,
            k_shot: int,
            disjoint: bool = True,
            rare_first: bool = True,
            allow_partial: bool = True,
            seed: typing.Optional[int] = 0,
            num_classes: typing.Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self._num_partitions = num_partitions
        self._check_num_partitions_greater_than_zero()
        self._k_shot = k_shot
        self._disjoint = disjoint
        self._rare_first = rare_first
        self._allow_partial = allow_partial
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)
        self._num_classes = num_classes

        self._partition_id_to_indices: typing.Dict[int, typing.List[int]] = {}
        self._partition_id_to_indices_determined = False
        self._report: typing.Optional[typing.Dict[str, typing.List]] = None

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    @property
    def report(self) -> typing.Dict[str, typing.List]:
        """Per-partition achieved class counts and shortfalls."""
        self._determine_partition_id_to_indices_if_needed()
        return self._report

    def _check_num_partitions_greater_than_zero(self) -> None:
        if not self._num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")

    def _get_label_matrix(self) -> np.ndarray:
        Y = self.dataset.get_targets()
        if Y is None:
            raise ValueError(
                "Dataset has no labels (is_test=True); can't build few-shot sets."
            )
        Y = np.asarray(Y)
        if Y.ndim == 1 or (Y.ndim == 2 and Y.shape[1] == 1):
            # single-label: 1-D integer class vector -> one-hot
            labels = Y.reshape(-1).astype(np.int64)
            C = int(self._num_classes) if self._num_classes is not None else int(labels.max()) + 1
            Y = np.zeros((labels.shape[0], C), dtype=np.int64)
            Y[np.arange(labels.shape[0]), labels] = 1
        else:
            # multi-hot / one-hot matrix
            Y = (Y > 0.5).astype(np.int64)
        return Y

    def _determine_partition_id_to_indices_if_needed(self) -> None:
        if self._partition_id_to_indices_determined:
            return

        Y = self._get_label_matrix()
        n, C = Y.shape

        available = np.ones(n, dtype=bool)
        report: typing.Dict[str, typing.List] = {"partitions": [], "shortfalls": []}

        for p in range(self._num_partitions):
            chosen: typing.List[int] = []
            counts = np.zeros(C, dtype=np.int64)
            in_partition = np.zeros(n, dtype=bool)

            while True:
                need = counts < self._k_shot
                if not need.any():
                    break

                pool = np.where(available & ~in_partition)[0]
                if pool.size == 0:
                    break

                cand = pool[Y[pool][:, need].any(axis=1)]
                if cand.size == 0:
                    break

                if self._rare_first:
                    hungry = np.where(need)[0]
                    avail_per_class = Y[pool][:, hungry].sum(axis=0)
                    fillable = hungry[avail_per_class > 0]
                    target_class = fillable[np.argmin(avail_per_class[avail_per_class > 0])]
                    cand = cand[Y[cand, target_class] == 1]

                gain = Y[cand][:, need].sum(axis=1)
                best_cand = cand[gain == gain.max()]
                pick = int(self._rng.choice(best_cand))

                chosen.append(pick)
                counts += Y[pick]
                in_partition[pick] = True
                if self._disjoint:
                    available[pick] = False

            self._partition_id_to_indices[p] = chosen
            achieved = {c: int(counts[c]) for c in range(C)}
            short = {c: int(self._k_shot - counts[c]) for c in range(C) if counts[c] < self._k_shot}
            report["partitions"].append(achieved)
            report["shortfalls"].append(short)
            if short and not self._allow_partial:
                raise RuntimeError(
                    f"Partition {p}: classes {sorted(short)} could not reach "
                    f"k_shot={self._k_shot} (short by {short})."
                )

        self._report = report
        self._partition_id_to_indices_determined = True

    def load_partition(self, partition_id: int) -> Subset:
        self._determine_partition_id_to_indices_if_needed()
        return Subset(self.dataset, self._partition_id_to_indices[partition_id])

    def load_partitions(self) -> typing.List[Subset]:
        """Return all partitions in order as a list of Subsets."""
        self._determine_partition_id_to_indices_if_needed()
        return [
            Subset(self.dataset, self._partition_id_to_indices[pid])
            for pid in range(self._num_partitions)
        ]

    def load_partition_indices(self) -> typing.List[typing.List[int]]:
        """Return raw index lists instead of Subsets (replaces return_indices=True)."""
        self._determine_partition_id_to_indices_if_needed()
        return [self._partition_id_to_indices[pid] for pid in range(self._num_partitions)]