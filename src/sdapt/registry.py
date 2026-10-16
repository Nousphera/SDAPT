"""Pinned official timm DINO ViT artifacts and architecture geometry.

Revisions, byte sizes and LFS SHA-256 values were read from the official
Hugging Face timm repositories on 2026-10-16. Every artifact's remote
safetensors header was checked against timm 1.0.22 on the meta device;
all tensor keys and shapes matched. No large weights were downloaded for
that registry audit. The pinned revisions all contain one safetensors file.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL = "dinov2-vitb14"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    timm_name: str
    patch_size: int
    depth: int
    embed_dim: int
    num_prefix_tokens: int
    num_register_tokens: int
    family: str
    revision: str
    sha256: str
    size: int
    filename: str = "model.safetensors"

    @property
    def hf_repo(self):
        return "timm/" + self.timm_name

    @property
    def url(self):
        return f"https://huggingface.co/{self.hf_repo}/resolve/{self.revision}/{self.filename}"

    @property
    def default_checkpoint(self):
        # Preserve the paper baseline's original preparation command/path.
        basename = "dinov2-vit-b14" if self.name == DEFAULT_MODEL else self.name
        return f"checkpoints/{basename}.safetensors"

MODEL_SPECS = {
    'dino-vits8': ModelSpec(
        name='dino-vits8', timm_name='vit_small_patch8_224.dino',
        patch_size=8, depth=12, embed_dim=384,
        num_prefix_tokens=1, num_register_tokens=0, family='dino',
        revision='35030b73013bb9fcbf66a5b542b96a7e2fd9ec9e',
        sha256='e1b79616ce6d1b8dc3cc0d5952b6d0958356a189122882493e665d28f20654a0', size=86694768,
    ),
    'dino-vitb8': ModelSpec(
        name='dino-vitb8', timm_name='vit_base_patch8_224.dino',
        patch_size=8, depth=12, embed_dim=768,
        num_prefix_tokens=1, num_register_tokens=0, family='dino',
        revision='0d725ac467f73fd42e16bc0c86446db2d689d702',
        sha256='01be14b99784a4c0970a5703214513588daf27014e908ac7ea08967b7bd80f5f', size=343245410,
    ),
    'dino-vits16': ModelSpec(
        name='dino-vits16', timm_name='vit_small_patch16_224.dino',
        patch_size=16, depth=12, embed_dim=384,
        num_prefix_tokens=1, num_register_tokens=0, family='dino',
        revision='10e440b8a34dfd657a90f2fcaa988c6b6a2f0da4',
        sha256='8d4795b8e4e327c6d841e15a2e3e6d88c45346256a43f8cdae8c26d92c806f7a', size=86676340,
    ),
    'dino-vitb16': ModelSpec(
        name='dino-vitb16', timm_name='vit_base_patch16_224.dino',
        patch_size=16, depth=12, embed_dim=768,
        num_prefix_tokens=1, num_register_tokens=0, family='dino',
        revision='4735e392aa9f07d74d9aa296f7be800da73699df',
        sha256='93181583427d6de3e18ab0e6ee4df35736cfc51a3c1bf9ae0e9b53b1316598dc', size=343208550,
    ),
    'dinov2-vits14': ModelSpec(
        name='dinov2-vits14', timm_name='vit_small_patch14_dinov2.lvd142m',
        patch_size=14, depth=12, embed_dim=384,
        num_prefix_tokens=1, num_register_tokens=0, family='dinov2',
        revision='4610ca143709d58a633b6397a74412c2c3842454',
        sha256='04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081', size=88240510,
    ),
    'dinov2-vitb14': ModelSpec(
        name='dinov2-vitb14', timm_name='vit_base_patch14_dinov2.lvd142m',
        patch_size=14, depth=12, embed_dim=768,
        num_prefix_tokens=1, num_register_tokens=0, family='dinov2',
        revision='4685c99dabffe5affac90bd99dbffd25801ae58d',
        sha256='55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b', size=346334872,
    ),
    'dinov2-vitl14': ModelSpec(
        name='dinov2-vitl14', timm_name='vit_large_patch14_dinov2.lvd142m',
        patch_size=14, depth=24, embed_dim=1024,
        num_prefix_tokens=1, num_register_tokens=0, family='dinov2',
        revision='4741e1cafbf45415e77074bb0cb42dba76c8684a',
        sha256='0424a5d1b515278cba3c6640ccbeaacc41de59d3a93df0dd5e494285eea2b355', size=1217502758,
    ),
    'dinov2-vitg14': ModelSpec(
        name='dinov2-vitg14', timm_name='vit_giant_patch14_dinov2.lvd142m',
        patch_size=14, depth=40, embed_dim=1536,
        num_prefix_tokens=1, num_register_tokens=0, family='dinov2',
        revision='dd22bb6485eeb29d67273f76d983061298bdc7f0',
        sha256='949077749c7a414f3443a007cf021d085bc11deb0b08959a6e32e170b1e8e359', size=4545971222,
    ),
    'dinov2-vits14-reg4': ModelSpec(
        name='dinov2-vits14-reg4', timm_name='vit_small_patch14_reg4_dinov2.lvd142m',
        patch_size=14, depth=12, embed_dim=384,
        num_prefix_tokens=5, num_register_tokens=4, family='dinov2',
        revision='c04b5193082a8d5b0c4856c7937384a48136c5de',
        sha256='dca70548ecd7b03ffba6172c4db403014511b5ee6073f9fca72ba9e6e602a25d', size=88245200,
    ),
    'dinov2-vitb14-reg4': ModelSpec(
        name='dinov2-vitb14-reg4', timm_name='vit_base_patch14_reg4_dinov2.lvd142m',
        patch_size=14, depth=12, embed_dim=768,
        num_prefix_tokens=5, num_register_tokens=4, family='dinov2',
        revision='3b06466a5548c52b8b98822e1390987695fcbf82',
        sha256='c24ecfb4a1d8ca79193f6b9efcffc461872a09ddac43a0931357e4802931a006', size=346344168,
    ),
    'dinov2-vitl14-reg4': ModelSpec(
        name='dinov2-vitl14-reg4', timm_name='vit_large_patch14_reg4_dinov2.lvd142m',
        patch_size=14, depth=24, embed_dim=1024,
        num_prefix_tokens=5, num_register_tokens=4, family='dinov2',
        revision='f3c408e77602bb412aa65fb03dfa0d5f95cb3832',
        sha256='c893d72294d4c327e631ff92f428dbc14c4f93cb5581b6c5f9d89bb5d17def27', size=1217515128,
    ),
    'dinov2-vitg14-reg4': ModelSpec(
        name='dinov2-vitg14-reg4', timm_name='vit_giant_patch14_reg4_dinov2.lvd142m',
        patch_size=14, depth=40, embed_dim=1536,
        num_prefix_tokens=5, num_register_tokens=4, family='dinov2',
        revision='6994d74074a8f74e220de3f403dc49dd5d8ae704',
        sha256='6c038d1d0b91f20d551b052b85e66bf792e53a086a10ab2b054a8b1f185ed303', size=4545989736,
    ),
    'dinov3-vits16': ModelSpec(
        name='dinov3-vits16', timm_name='vit_small_patch16_dinov3.lvd1689m',
        patch_size=16, depth=12, embed_dim=384,
        num_prefix_tokens=5, num_register_tokens=4, family='dinov3',
        revision='3bf4720a82ec2066db88137180ff1f83a675cef0',
        sha256='2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040', size=86362376,
    ),
    'dinov3-vits16plus': ModelSpec(
        name='dinov3-vits16plus', timm_name='vit_small_plus_patch16_dinov3.lvd1689m',
        patch_size=16, depth=12, embed_dim=384,
        num_prefix_tokens=5, num_register_tokens=4, family='dinov3',
        revision='c35074f0e1a65242948168c9358ce26705c4deb2',
        sha256='423e7b4b1103de4100a5c19a436cc00f1a994d82835ed51623b209ccfd1e9615', size=114750056,
    ),
    'dinov3-vitb16': ModelSpec(
        name='dinov3-vitb16', timm_name='vit_base_patch16_dinov3.lvd1689m',
        patch_size=16, depth=12, embed_dim=768,
        num_prefix_tokens=5, num_register_tokens=4, family='dinov3',
        revision='c6a5fb7d12bbd3cf3b0079253141c3332aaed7da',
        sha256='1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332', size=342579728,
    ),
    'dinov3-vitl16': ModelSpec(
        name='dinov3-vitl16', timm_name='vit_large_patch16_dinov3.lvd1689m',
        patch_size=16, depth=24, embed_dim=1024,
        num_prefix_tokens=5, num_register_tokens=4, family='dinov3',
        revision='30c1109559f65dea34316b0d4842d35c5771fe11',
        sha256='45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941', size=1212347640,
    ),
    'dinov3-vith16plus': ModelSpec(
        name='dinov3-vith16plus', timm_name='vit_huge_plus_patch16_dinov3.lvd1689m',
        patch_size=16, depth=32, embed_dim=1280,
        num_prefix_tokens=5, num_register_tokens=4, family='dinov3',
        revision='dd0addc09788111fa893d24799a44744ba022eee',
        sha256='0d54131c846c91a78ec7641482f32e45b27cb4895f83653c038f305f18c7e61c', size=3362084520,
    ),
    'dinov3-vit7b16': ModelSpec(
        name='dinov3-vit7b16', timm_name='vit_7b_patch16_dinov3.lvd1689m',
        patch_size=16, depth=40, embed_dim=4096,
        num_prefix_tokens=5, num_register_tokens=4, family='dinov3',
        revision='b40f6c827cb02b13c7a23a76cf598489bb0f94e5',
        sha256='467c8d1d13fff6a2486dbd5fad1a47a1343fa68d4aa528dec373c441724fab56', size=26864183416,
    ),
    'dinov3-vitl16-sat493m': ModelSpec(
        name='dinov3-vitl16-sat493m', timm_name='vit_large_patch16_dinov3.sat493m',
        patch_size=16, depth=24, embed_dim=1024,
        num_prefix_tokens=5, num_register_tokens=4, family='dinov3',
        revision='ce148bd5ed3de1eb75ebdccafcb2c280bb3e8f7c',
        sha256='8d18337f766102f951ac0ca543a22536f58caeb6b6c33cfef73326f866791286', size=1212347640,
    ),
    'dinov3-vit7b16-sat493m': ModelSpec(
        name='dinov3-vit7b16-sat493m', timm_name='vit_7b_patch16_dinov3.sat493m',
        patch_size=16, depth=40, embed_dim=4096,
        num_prefix_tokens=5, num_register_tokens=4, family='dinov3',
        revision='781bbc782edbd43bf040d3336fc3d7dc207583e6',
        sha256='909f30e1b7b7f3f3461b8af18a12dc4b12a18e10172d38b12fd333f7fe5bf249', size=26864183416,
    ),
}

_ALIASES = {
    "dinov2-base": DEFAULT_MODEL,
    "dinov2-vit-b14": DEFAULT_MODEL,
    "dinov3-vitl16-sat": "dinov3-vitl16-sat493m",
    "dinov3-vit7b16-sat": "dinov3-vit7b16-sat493m",
}


def get_model_spec(name=DEFAULT_MODEL):
    if isinstance(name, ModelSpec):
        return name
    canonical = _ALIASES.get(name, name)
    try:
        return MODEL_SPECS[canonical]
    except KeyError as error:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODEL_SPECS)}") from error
