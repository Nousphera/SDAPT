import os

import numpy as np
from data.datasets.dataset_abstract import DinoDataset
from datasets import ClassLabel, load_dataset
CATEGORIES = [
    "airport",
    "airport_hangar",
    "airport_terminal",
    "amusement_park",
    "aquaculture",
    "archaeological_site",
    "barn",
    "border_checkpoint",
    "burial_site",
    "car_dealership",
    "construction_site",
    "crop_field",
    "dam",
    "debris_or_rubble",
    "educational_institution",
    "electric_substation",
    "factory_or_powerplant",
    "fire_station",
    "flooded_road",
    "fountain",
    "gas_station",
    "golf_course",
    "ground_transportation_station",
    "helipad",
    "hospital",
    "impoverished_settlement",
    "interchange",
    "lake_or_pond",
    "lighthouse",
    "military_facility",
    "multi-unit_residential",
    "nuclear_powerplant",
    "office_building",
    "oil_or_gas_facility",
    "park",
    "parking_lot_or_garage",
    "place_of_worship",
    "police_station",
    "port",
    "prison",
    "race_track",
    "railway_bridge",
    "recreational_facility",
    "road_bridge",
    "runway",
    "shipyard",
    "shopping_mall",
    "single-unit_residential",
    "smokestack",
    "solar_farm",
    "space_facility",
    "stadium",
    "storage_tank",
    "surface_mine",
    "swimming_pool",
    "toll_booth",
    "tower",
    "tunnel_opening",
    "waste_disposal",
    "water_treatment_facility",
    "wind_farm",
    "zoo",
]

class FMoWFromHuggingFace(DinoDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(
        self,
        csv_path,
        transform,
        target_transform=None,
        max_shots=0,
        subsample_fraction=1.0,
        train=True,
        custom_targets=None,
        path_prefix="",
        seed=0,
    ):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param csv_path: csv_path (string): path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=3)
        # Transforms
        self.transforms = transform
        # Load data from huggingface
        path_prefix = os.environ.get("TMPDIR")
        self.dataset = load_dataset("Staneman/DAPTSL_fMoW", cache_dir=path_prefix)
        if(not train):
            self.dataset = self.dataset["validation"]
        else:
            self.dataset = self.dataset["train"]

        # Calculate len
        if max_shots > 0:
            self._apply_few_shot(max_shots, seed)

        self.data_len = len(self.dataset)

        self.path_prefix = path_prefix

        self.label_to_idx = {label: i for i, label in enumerate(CATEGORIES)}

        # custom targets to replace CATEGORIES
        self.custom_targets = custom_targets

    def get_targets(self):
        if self.custom_targets is None:
            return np.array(
                [self.label_to_idx[label] for label in self.dataset["label"]],
                dtype=np.int64,
            )
        return np.array(self.custom_targets, dtype=np.int64)

    def _apply_few_shot(self, max_shots, seed):
        """
        Restrict the dataset to at most `max_shots` samples per class.
        Selection is deterministic given `seed`. Classes with fewer than
        `max_shots` samples keep all of their samples.
        """
        labels = self.dataset["label"]  # columnar access, fast in HF datasets

        # group example indices by class
        indices_by_class = {}
        for idx, label in enumerate(labels):
            indices_by_class.setdefault(label, []).append(idx)

        rng = np.random.RandomState(seed)

        selected = []
        for label, idxs in indices_by_class.items():
            idxs = np.array(idxs)
            if len(idxs) > max_shots:
                chosen = rng.choice(idxs, size=max_shots, replace=False)
            else:
                chosen = idxs
            selected.extend(chosen.tolist())

        selected.sort()  # keep a stable ordering

        self.dataset = self.dataset.select(selected)
        self.data_len = len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[int(index)]

        img = example["image"].convert("RGB")
        img_tensor = self.transforms(img)

        label = self.label_to_idx[example["label"]]
       
        return img_tensor, label

    def __len__(self):
        return self.data_len