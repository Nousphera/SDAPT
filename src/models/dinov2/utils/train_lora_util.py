from functools import partial


def activate_lora(
    model,
    activate_layers=("attn",),
    lora_rank=8,
    include_attn_key=False,
    include_attn_proj=False,
):
    def recursive_activate_lora(layer, name):
        if name in activate_layers and hasattr(layer, "init_lora"):
            layer.init_lora(lora_rank=lora_rank, include_attn_key=include_attn_key, include_proj=include_attn_proj)
        for child_name, child_layer in layer.named_children():
            recursive_activate_lora(child_layer, child_name)

    recursive_activate_lora(model, "")


def deactivate_lora(model, activate_layers=("attn",), delete_separate_proj=False):
    def recursive_deactivate(layer, name):
        if name in activate_layers and hasattr(layer, "deinit_lora"):
            layer.deinit_lora(delete_separate_proj=delete_separate_proj)
        for child_name, child_layer in layer.named_children():
            recursive_deactivate(child_layer, child_name)

    recursive_deactivate(model, "")


# TODO: this is only if after deactivating lora you still want to delete the qkv layer
def delete_qkv(model, layers=("attn",)):
    def recursive_delete_qkv(layer, name):
        if name in layers and hasattr(layer, "qkv") and getattr(layer, "q_proj", None) is not None:
            del layer.qkv
            layer.qkv = None
        for child_name, child_layer in layer.named_children():
            recursive_delete_qkv(child_layer, child_name)

    recursive_delete_qkv(model, "")


def refresh_lora(model, key="lora_A", kwarg_key="only_lora"):  # key can be lora_A or monarch
    kwarg = {kwarg_key: True}

    def recursive_refresh(layer):
        if hasattr(layer, key):
            layer.reset_parameters(**kwarg)
        for child_name, child_layer in layer.named_children():
            recursive_refresh(child_layer)

    recursive_refresh(model)


def is_merged(model, key="lora_A"):
    def recursive_is_merged(layer):
        if hasattr(layer, key):
            return layer.merged
        result = True
        for child_name, child_layer in layer.named_children():
            result = result and recursive_is_merged(child_layer)
        return result

    return recursive_is_merged(model)


def lora_merge_all(model, unmerge=False, key="lora_A"):
    def recursive_merge(layer):
        if hasattr(layer, key):
            if unmerge:
                layer.unmerge()
            else:
                layer.merge()

        for child_name, child_layer in layer.named_children():
            recursive_merge(child_layer)

    recursive_merge(model)


def reset_merged_status(model, new_status=False, key="lora_A"):
    def recursive_reset(layer):
        if hasattr(layer, key):
            layer.merged = new_status

        for child_name, child_layer in layer.named_children():
            recursive_reset(child_layer)

    recursive_reset(model)


def merge_lora_refresh(model, key="lora_A"):
    lora_merge_all(model, unmerge=False, key=key)
    reset_merged_status(model, new_status=False, key=key)
