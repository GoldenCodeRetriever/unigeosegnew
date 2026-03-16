import logging
from typing import Tuple

import torch
from transformers import AutoTokenizer

from unigeoseg.mask_config.config import get_mask_config
from unigeoseg.model.language_model.llava_phi import LlavaConfig, UniGeoSeg


LOGGER = logging.getLogger(__name__)


def _resolve_vision_tower_path(model_args) -> str:
    return model_args.mm_vision_tower or model_args.vision_tower


def build_tokenizer(model_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model(model_args, training_args) -> Tuple[UniGeoSeg, object]:
    mask_cfg = get_mask_config(model_args.mask_config)
    mask_cfg.MODEL.MASK_FORMER.SEG_TASK = "instance"
    config = LlavaConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.use_cache = False
    config.mm_projector_type = model_args.mm_projector_type
    config.swin_type = model_args.swin_type
    config.with_norm = model_args.with_norm
    config.with_layernorm = model_args.with_layernorm

    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = UniGeoSeg.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        mask_decoder_cfg=mask_cfg,
        use_seg_query=model_args.use_seg_query,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    if model.get_vision_tower() is None:
        vision_tower_path = _resolve_vision_tower_path(model_args)
        if not vision_tower_path and not model_args.allow_random_init:
            raise ValueError(
                "The loaded checkpoint does not include a visual tower. "
                "Please provide --vision_tower/--mm_vision_tower or set --allow_random_init true."
            )
        model_args.mm_vision_tower = vision_tower_path
        model.get_model().initialize_vision_modules(model_args)
        LOGGER.info("Initialized Swin vision tower from %s", vision_tower_path or "random weights")

    if not getattr(model, "is_train_mask_decode", False) or not hasattr(model, "pixel_decoder"):
        if not model_args.pretrained_mask_decoder_path and not model_args.allow_random_init:
            raise ValueError(
                "The loaded checkpoint does not include the mask decoder. "
                "Please provide --pretrained_mask_decoder_path or set --allow_random_init true."
            )
        model.initial_mask_module(pretrained_path=model_args.pretrained_mask_decoder_path, model_args=model_args)
        LOGGER.info("Initialized mask decoder from %s", model_args.pretrained_mask_decoder_path or "random weights")

    if model_args.freeze_vision_tower and model.get_vision_tower() is not None:
        for parameter in model.get_vision_tower().parameters():
            parameter.requires_grad = False

    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model, mask_cfg


def count_parameters(model) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
