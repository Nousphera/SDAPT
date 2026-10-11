from data.datasets.cheXpert_dataset import CHEXPERT_COMPETITION_TASKS, CHEXPERT_PATHOLOGY_TASKS
from evaluators.accuracy_evaluator import AccuracyEvaluator
from evaluators.auroc_evaluator import MultilabelAUROCEvaluator
from evaluators.hpa_evaluator import KaggleHPASubmissionEvaluator



def build_evaluator(global_args):
    kind = global_args["task_type"]
    if kind == "single_label_classification":
        return AccuracyEvaluator()
    if kind == "hpa":
        return KaggleHPASubmissionEvaluator()
    if kind == "chexpert_auroc":
        return MultilabelAUROCEvaluator(    
            task_names=CHEXPERT_PATHOLOGY_TASKS,
            mean_over=CHEXPERT_COMPETITION_TASKS,  # the official 5-task AUROC
        )
    raise ValueError(f"unknown task_type: {kind}")
