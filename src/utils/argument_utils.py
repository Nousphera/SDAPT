import argparse
import os
from distutils.util import strtobool
from typing import Any, Dict

import yaml


def _is_dict_like(arguments: Any) -> bool:
    return isinstance(arguments, dict)


def _get_argument_value(arguments: Any, key: str, default=None):
    if _is_dict_like(arguments):
        return arguments.get(key, default)
    return getattr(arguments, key, default)


def _set_argument_value(arguments: Any, key: str, value) -> None:
    if _is_dict_like(arguments):
        arguments[key] = value
        return
    setattr(arguments, key, value)


def _flatten_dict_values(data: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    flattened = {}
    for key, value in data.items():
        normalized_key = str(key).strip().lower()
        full_key = f"{parent_key}.{normalized_key}" if parent_key else normalized_key
        if isinstance(value, dict):
            flattened.update(_flatten_dict_values(value, parent_key=full_key))
        else:
            flattened[full_key] = value
    return flattened


def _load_yaml_config(config_file: str) -> Dict[str, Any]:
    config_path = os.path.abspath(os.path.expanduser(config_file))
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML dictionary at top-level: {config_path}")

    return loaded


def _get_config_section(config: Dict[str, Any], section_name: str) -> Dict[str, Any]:
    for key in (section_name, section_name.lower(), section_name.upper()):
        section = config.get(key)
        if isinstance(section, dict):
            return section
    return {}


def apply_yaml_config_to_arguments(arguments: Any) -> None:
    if _get_argument_value(arguments, "_yaml_config_loaded", False):
        return

    config_file = _get_argument_value(arguments, "config_file", None)
    if not config_file:
        _set_argument_value(arguments, "_yaml_config_loaded", True)
        return

    config_dict = _load_yaml_config(config_file)
    flat_values = _flatten_dict_values(config_dict)

    # Store raw config for model-side consumers that expect nested sections.
    _set_argument_value(arguments, "dino_config", config_dict)

    # Promote commonly used sections to top-level keys when no collision is expected.
    for section_name in ("train", "student", "teacher", "optim", "evaluation", "crops", "compute_precision", "lora", "dino", "ibot"):
        section_value = _get_config_section(config_dict, section_name)
        if section_value:
            existing = _get_argument_value(arguments, section_name, None)
            if existing is None or isinstance(existing, dict):
                _set_argument_value(arguments, section_name, section_value)

    # Keep MODEL section separate to avoid clobbering existing --model CLI arg.
    model_section = _get_config_section(config_dict, "model")
    if model_section:
        _set_argument_value(arguments, "dino_model", model_section)

    # Map relevant YAML keys to existing flat global args expected by the training code.
    mapped_keys = {
        "train.batch_size_per_gpu": "batch_size",
        "optim.epochs": "nr_of_epochs",
        "optim.base_lr": "start_lr",
        "optim.accum_iter": "accum_iter",
        "student.pretrained_weights": "student_pretrained_weights",
        "teacher.momentum_teacher": "momentum_teacher",
        "train.centering": "centering",
        "crops.local_crops_size": "local_crops_size",
    }

    for source_key, target_key in mapped_keys.items():
        if source_key in flat_values:
            _set_argument_value(arguments, target_key, flat_values[source_key])

    # Also expose every leaf as dino_<section>_<key> for easy lookup in model code.
    for source_key, value in flat_values.items():
        alias = f"dino_{source_key.replace('.', '_')}"
        _set_argument_value(arguments, alias, value)

    _set_argument_value(arguments, "_yaml_config_loaded", True)



def validate_base_argument_constraints(argument_parser):
    if argument_parser.vit_base != 'None':
        if argument_parser.vit_base == 'ViT-H/14' and not (argument_parser.dataset == 'coco-qa' or argument_parser.dataset == 't4sa'):
            raise NotImplementedError('ViT-H/14 base is currently only supported for image, text pairs (coco-qa & t4sa datasets)')

        if argument_parser.vit_base == 'ViT-Ti/16' or argument_parser.vit_base == 'ViT-S/16':
            # When using encoders that are smaller than ViT-B/16, we cannot use the CLIP encoder to encode text embeddings prior, as their dimension has a minimum of 512.
            # Thus, we need to manually override the chosen setting by disabling this altogether.
            argument_parser.use_clip_encoder_for_text_embeddings = False


def set_env_variables(arguments):
    apply_yaml_config_to_arguments(arguments)

    os.environ['TORCH_DATA_DIR'] = str(_get_argument_value(arguments, 'torch_data_dir', ''))
    os.environ['PRE_PROCESSORS_CACHE_DIR'] = str(_get_argument_value(arguments, 'pre_processors_cache_dir', ''))
    os.environ['TOKENIZER_WEIGHTS_CACHE_DIR'] = str(_get_argument_value(arguments, 'tokenizer_weights_cache_dir', ''))
    os.environ['MODEL_WEIGHTS_DIR'] = str(_get_argument_value(arguments, 'model_weights_dir', ''))
    os.environ['AE_WEIGHTS_DIR'] = str(_get_argument_value(arguments, 'ae_weights_dir', ''))
    os.environ['VIT_BASE'] = str(_get_argument_value(arguments, 'vit_base', 'None'))


def build_base_argument_parser():
    """
    Builds and returns an instance of ArgumentParser, with all shared parameters across the possible configurations (centralized, split-learning, and federated-learning with and without the adapter approach).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_layer_from_end', type=int, default=1, help='The number of feature layers to use from the end of the backbone.')

    # == Data directories ==
    parser.add_argument('--torch_data_dir', type=str, default='../../shared_data/datasets', help='The directory for the data of all datasets.')
    parser.add_argument('--pre_processors_cache_dir', type=str, default='../../shared_data/preprocessors', help='The directory for all pre-processor weights.')
    parser.add_argument('--pretrained_weight_path_server', type=str, default='../../shared_data/preprocessors', help='The directory for all pre-processor weights.')
    parser.add_argument('--logs_dir', type=str, default='../../shared_data/preprocessors', help='The directory for all pre-processor weights.')
    parser.add_argument('--checkpoint_key_server', type=str, default='../../shared_data/preprocessors', help='The directory for all pre-processor weights.')
    parser.add_argument('--tokenizer_weights_cache_dir', type=str, default='../../shared_data/tokenizers', help='The directory for all tokenizer weights.')
    parser.add_argument('--model_weights_dir', type=str, default='../../shared_data/model_checkpoints', help='The directory of all model weights. This is the directory in which the codebase expects the Meta-Transformer model weights to be present.')
    parser.add_argument('--config_file', type=str, default=None, help='Optional YAML config file whose values will be merged into global_args.')
    parser.add_argument('--eval_store_dir', type=str, default=None, help='Optional YAML config file whose values will be merged into global_args.')
    parser.add_argument('--pretrained_weight_folder', type=str, default=None, help='The directory containing the pretrained weights.')
    parser.add_argument('--nr_of_classes', type=int, default=62, help='The number of classes in the dataset.')
    parser.add_argument('--alpha', type=float, default=None, help='The alpha parameter for the DAPT approach.')
    parser.add_argument(
        '--freeze_server_backbone',
        action='store_true',
        default=False,
        help='Whether the server backbone should be frozen.'
    )
    parser.add_argument(
            '--dino_weight',
            action='store_true',
            default=False,
            help='Whether the DINO weights should be used for the server backbone.'
        )
    # == Dataset and model ==
    parser.add_argument('--dataset', type=str, help='The dataset that should be used.')
    parser.add_argument('--transforms', type=str, default='default', choices=['default', 'dino_pretrain'], help='Data transform pipeline to use. Set to dino_pretrain to enable DINO-style training augmentations when supported.')
    parser.add_argument('--val_mode', type=str, choices=['none', 'early_saving', 'early_stopping'], default='early_stopping', help='Whether a validation set should be used TODO full explanation')
    parser.add_argument('--val_split', type=float, default=0.1, help='The proportion of the dataset to use for validation.')

    parser.add_argument('--model', type=str, help='The model that should be used.', default='vit')
    parser.add_argument('--fusion_type', type=str, default='default', help='The type of fusion that should be used, if applicable. The \'default\' type has been used for the reported experiments. For available options, please refer to the model implementation of the chosen task.')
    parser.add_argument('--use_pre_layer_norm', dest='use_pre_layer_norm', type=lambda x: bool(strtobool(x)), default=True, help='Whether pre-layer normalization should be used.')
    parser.add_argument('--use_post_layer_norm', dest='use_post_layer_norm', type=lambda x: bool(strtobool(x)), default=False, help='Whether post-layer normalization should be used.')
    parser.add_argument('--use_large_encoder', dest='use_large_encoder', type=lambda x: bool(strtobool(x)), default=False, help='Whether the large encoder version of Meta-Transformer should be used.')

    parser.add_argument('--freeze_encoder', dest='freeze_encoder', type=lambda x: bool(strtobool(x)), default=False, help='Whether all blocks of the encoder should be frozen.')
    parser.add_argument('--nr_of_last_encoder_blocks_to_finetune', type=int, default=-1, help='If not freezing the entire encoder with freeze_encoder, we can optionally train the last X encoder blocks.')

    parser.add_argument('--use_clip_encoder_for_text_embeddings', dest='use_clip_encoder_for_text_embeddings', type=lambda x: bool(strtobool(x)), default=True, help='Whether the clip encoder should be used for encoding text, after text tokenization.')
    parser.add_argument('--vit_base', type=str, default='None', help='For manually overriding the ViT base that is used. All default experiments (those not in the ablation studies), are executed without touching this setting.')

    parser.add_argument('--split_layer', type=int, default=0, help='Where the model should be split and the AE inserted if applicable')
    parser.add_argument(
        '--split_depth',
        type=int,
        default=None,
        help='Number of ViT transformer blocks placed on the client in the DINO split path.',
    )
    # LoRA
    parser.add_argument('--use_lora', dest='use_lora', type=lambda x: bool(strtobool(x)), default=True, help='Whether LoRA should be used to finetune the model.')
    parser.add_argument('--lora_rank', type=int, default=16, help='The rank of the LoRA matrices.')
    parser.add_argument('--lora_alpha', type=int, default=32, help='The alpha scaling factor of LoRA.')

    # == Misc ==
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size on the server. This means that with a batch_size of 500 with 25 clients, each client would have a mini-batch size of 20.')
    parser.add_argument('--nr_of_epochs', type=int, default=15)
    parser.add_argument('--start_lr', type=float, default=1e-4)
    parser.add_argument('--scheduler_step_size', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=2024)
    parser.add_argument(
        '--partition_seed',
        type=int,
        default=None,
        help='Seed used to assign samples to clients. Defaults to random_seed when omitted.',
    )
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers provided to the Dataloader.')

    parser.add_argument('--save_model_after_each_epoch', dest='save_model_after_each_epoch', type=lambda x: bool(strtobool(x)), default=False, help='Whether model(s) weights should be saved after each training epoch.')
    parser.add_argument('--save_final_model', dest='save_final_model', type=lambda x: bool(strtobool(x)), default=True, help='Whether the final obtained model weights after training should be saved.')
    parser.add_argument('--save_file_name', type=str, default=None, help='The name of the file into which the model weights should be saved.')
    parser.add_argument('--gpu_id', type=str, default=None, help='If desired, this parameter can be used to explicitly define which CUDA gpu to use.')

    # Specifically for datasets with the audio modality
    parser.add_argument('--audio_time_dimension', type=str, default='10.24s', help='This parameter is explicitly used for datasets with the audio modality. It defines the duration/dimension of a single audio input sample. For available options, please refer to the model implementation of the chosen task.')

    # For testing
    parser.add_argument('--small_test_run', dest='small_test_run', type=lambda x: bool(strtobool(x)), default=False, help='Whether a small test run should be executed for rapid testing purposes.')

    return parser

def expand_argument_parser_with_distributed_learning_parameters(argument_parser):
    argument_parser.add_argument('--nr_of_clients', type=int, required=False, default=5, help='The number of clients to use during training.')
    argument_parser.add_argument('--dataset_split_type', type=str, required=False, default='nonidd', choices=['iid', 'noniid'], help='The type of data distribution that should be used for splitting the original dataset into all separate client-side datasets.')

    return argument_parser


def namespace_to_dict(namespace):
    """
    Converts an argparse.Namespace object to a dictionary.
    """
    return {key: value for key, value in vars(namespace).items() if not str(key).startswith("_")}