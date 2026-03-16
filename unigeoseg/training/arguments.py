from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "UniGeoSeg checkpoint or base language model path."})
    mask_config: str = field(
        default="./mask_config/maskformer2_swin_base_384_bs16_50ep.yaml",
        metadata={"help": "Mask2Former config used by UniGeoSeg."},
    )
    vision_tower: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained Swin-B weights when starting from a language-only checkpoint."},
    )
    mm_vision_tower: Optional[str] = field(
        default=None,
        metadata={"help": "Alias of vision_tower for compatibility with existing init code."},
    )
    pretrained_mask_decoder_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained Mask2Former weights used to initialize the pixel decoder and predictor."},
    )
    pretrain_mm_mlp_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "Optional pretrained multimodal projector checkpoint."},
    )
    model_map_name: str = field(default="unigeoseg")
    mm_projector_type: str = field(default="conv")
    swin_type: str = field(default="base")
    with_norm: bool = field(default=False)
    with_layernorm: bool = field(default=False)
    use_seg_query: bool = field(default=False)
    freeze_vision_tower: bool = field(default=True)
    trust_remote_code: bool = field(default=False)
    allow_random_init: bool = field(
        default=False,
        metadata={"help": "Allow random initialization of the visual tower or mask decoder when pretrained weights are unavailable."},
    )


@dataclass
class DataArguments:
    data_root: Optional[str] = field(
        default=None,
        metadata={"help": "Optional shared dataset root. Specific roots override this value."},
    )
    geoseg_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated JSON/JSONL manifests following the generic GeoSeg-1M-style format."},
    )
    rrsisd_root: Optional[str] = field(default=None)
    refsegrs_root: Optional[str] = field(default=None)
    earthreason_root: Optional[str] = field(default=None)
    rsrs_root: Optional[str] = field(default=None)
    rrsisd_split: str = field(default="train")
    refsegrs_split: str = field(default="train")
    earthreason_split: str = field(default="train")
    rsrs_split: str = field(default="train")
    image_size: int = field(default=512)
    samples_per_epoch: Optional[int] = field(
        default=None,
        metadata={"help": "Number of mixed-task samples drawn per epoch. Defaults to the sum of all task-group sizes."},
    )
    manifest_image_root: Optional[str] = field(default=None)
    manifest_mask_root: Optional[str] = field(default=None)


@dataclass
class UniGeoSegTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./outputs/unigeoseg")
    num_train_epochs: float = field(default=3.0)
    learning_rate: float = field(default=1e-4)
    warmup_ratio: float = field(default=0.03)
    weight_decay: float = field(default=0.0)
    lr_scheduler_type: str = field(default="cosine")
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    dataloader_num_workers: int = field(default=4)
    dataloader_persistent_workers: bool = field(default=False)
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="epoch")
    save_total_limit: int = field(default=2)
    report_to: str = field(default="none")
    pts_enabled: bool = field(default=True)
    interactive_weight_start: float = field(default=1.0)
    interactive_weight_end: float = field(default=0.7)
    referring_weight: float = field(default=1.0)
    reasoning_weight_start: float = field(default=0.0)
    reasoning_weight_end: float = field(default=0.3)
