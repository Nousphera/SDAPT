import csv
import os
import time
from collections import OrderedDict

import numpy as np
import torch

from evaluators.evaluator import Evaluator

# 2019 "Human Protein Atlas Image Classification" class order (index -> name).
# These index positions MUST line up with your model's 28 output columns and with
# HPADataset.PROTEIN_COLS. Kaggle wants the *integer* index in the submission.
PROTEIN_COLS = [
    "nucleoplasm", "nuclear membrane", "nucleoli", "nucleoli fibrillar center",
    "nuclear speckles", "nuclear bodies", "endoplasmic reticulum",
    "golgi apparatus", "peroxisomes", "endosomes", "lysosomes",
    "intermediate filaments", "actin filaments", "focal adhesion sites",
    "microtubules", "microtubule ends", "cytokinetic bridge",
    "mitotic spindle", "microtubule organizing center", "centrosome",
    "lipid droplets", "plasma membrane", "cell junctions", "mitochondria",
    "aggresome", "cytosol", "cytoplasmic bodies", "rods & rings",
]


class KaggleHPASubmissionEvaluator(Evaluator):
    """Evaluator for the *unlabeled* Kaggle test set.

    Unlike a normal metric evaluator there are no ground-truth labels, so
    compute() cannot score locally. Instead it:
      1. turns accumulated logits into multi-hot label sets (sigmoid + threshold),
      2. writes a Kaggle ``Id,Predicted`` submission CSV, and
      3. (optionally) submits it via the Kaggle API and polls the public score.

    Targets the 2019 image-level competition
    ``human-protein-atlas-image-classification`` (Id,Predicted with space-separated
    class ids 0-27). It does NOT work for the 2021
    ``hpa-single-cell-image-classification`` code competition, which needs per-cell
    RLE masks submitted from a Kaggle notebook.

    Interface note: for the test set your HPADataset yields ``(img, image_id)``, so
    the *labels* slot the trainer passes to ``update`` actually carries the string
    IDs. This class treats that argument as IDs.

    Thresholding: ``threshold`` may be a single float (same cutoff for every class)
    or a length-``num_classes`` array of per-class cutoffs, e.g. the vector returned
    by ``prevalence_thresholds()``. A per-class vector broadcasts elementwise against
    each row of sigmoid probabilities.
    """

    def __init__(
        self,
        competition="human-protein-atlas-image-classification",
        class_names=PROTEIN_COLS,
        threshold=0.35,           # sigmoid prob >= threshold => label present;
                                  # scalar OR (num_classes,) per-class vector
        min_labels=1,            # never emit an empty prediction; fall back to top-k
        output_dir="submissions",
        filename="submission.csv",
        template_csv=None,       # path to sample_submission.csv to guarantee full/ordered rows
        default_label="0",       # label used for template rows we never predicted on
        submit=False,            # OFF by default: Kaggle allows only ~5 subs/day
        message=None,
        wait_for_score=False,    # poll the API until the submission is scored
        poll_seconds=15,
        timeout_seconds=900,
    ):
        self.competition = competition
        self.class_names = list(class_names)

        # threshold: scalar (same cutoff for every class) OR a (num_classes,)
        # array of per-class thresholds, e.g. from prevalence_thresholds().
        self.threshold = np.asarray(threshold, dtype=np.float32)
        if self.threshold.ndim == 1 and len(self.threshold) != len(self.class_names):
            raise ValueError(
                f"per-class threshold has {len(self.threshold)} entries but there "
                f"are {len(self.class_names)} classes"
            )

        self.min_labels = min_labels
        self.output_dir = output_dir
        self.filename = filename
        self.template_csv = template_csv
        self.default_label = default_label
        self.submit = submit
        self.message = message
        self.wait_for_score = wait_for_score
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.reset()

    def reset(self):
        self._probs = []
        self._ids = []

    def update(self, logits, ids, client_id=None):
        # `ids` arrives in the trainer's "labels" position (see class docstring).
        self._probs.append(torch.sigmoid(logits.detach().float().cpu()))
        self._ids.extend(self._flatten_ids(ids))

    @staticmethod
    def _flatten_ids(ids):
        """Normalise whatever the dataloader collated into a flat list[str]."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, (str, bytes)):
            ids = [ids]
        out = []
        for i in ids:
            out.append(i.decode() if isinstance(i, bytes) else str(i))
        return out

    def _predict_labels(self, probs_row):
        # self.threshold is a scalar (0-d) or a (num_classes,) vector; both broadcast.
        idxs = np.where(probs_row >= self.threshold)[0]
        if len(idxs) < self.min_labels:
            # fall back to the highest-scoring class(es) so the field isn't empty
            idxs = np.argsort(probs_row)[::-1][: self.min_labels]
        return " ".join(str(int(i)) for i in sorted(int(x) for x in idxs))

    def compute(self):
        if not self._probs:
            return None

        probs = torch.cat(self._probs).numpy()
        if len(self._ids) != len(probs):
            raise ValueError(
                f"id/prediction count mismatch: {len(self._ids)} ids vs {len(probs)} rows"
            )

        # Dedupe (multiple clients iterating the same test set would repeat IDs).
        # Keep the first occurrence; Kaggle rejects duplicate Ids.
        preds = OrderedDict()
        for img_id, row in zip(self._ids, probs):
            if img_id not in preds:
                preds[img_id] = self._predict_labels(row)
        if len(preds) != len(self._ids):
            print(f"[Kaggle] warning: dropped {len(self._ids) - len(preds)} duplicate Ids")

        rows = self._align_to_template(preds)

        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, self.filename)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Id", "Predicted"])
            w.writerows(rows)

        print(f"[Kaggle] wrote {len(rows)} predictions -> {csv_path}")
        result = {"submission_path": csv_path, "num_rows": len(rows)}

        print(
            "[Kaggle] submit manually with:\n"
            f'  kaggle competitions submit -c {self.competition} '
            f'-f {csv_path} -m "message"'
        )
        return result

    def _align_to_template(self, preds):
        """Return list[[Id, Predicted]]. If a sample_submission.csv is given, emit
        exactly its Ids in its order (Kaggle needs every test Id, no extras)."""
        if not self.template_csv:
            return [[k, v] for k, v in preds.items()]

        rows, missing = [], 0
        with open(self.template_csv, newline="") as f:
            for r in csv.DictReader(f):
                img_id = r["Id"]
                if img_id in preds:
                    rows.append([img_id, preds[img_id]])
                else:
                    rows.append([img_id, self.default_label])
                    missing += 1
        if missing:
            print(f"[Kaggle] warning: {missing} template Ids had no prediction; "
                  f"filled with default_label={self.default_label!r}")
        return rows