# UniGeoSeg Training

This repository now includes a full training pipeline aligned to the paper `UniGeoSeg: Towards Unified Open-World Segmentation for Geospatial Scenes`.

## Paper-aligned defaults

- Language backbone: `Phi-1.5`
- Visual encoder: `Swin-B`
- Segmentation decoder: `Mask2Former`
- Image size: `512 x 512`
- Precision: `bf16`
- Optimizer: `AdamW`
- Learning rate: `1e-4`
- LR schedule: `cosine`
- Warmup ratio: `0.03`
- Weight decay: `0.0`
- Epochs: `3`
- Visual encoder: frozen during training
- Gradient checkpointing: enabled
- ZeRO: Stage 2 config provided in `training/deepspeed_zero2.json`
- PTS: interactive sampling decays to `0.7`, reasoning receives the remaining scheduled increase
- LKM defaults already match the paper ablation winner in the current model code: `N=4`, `lambda=0.2`

## Entrypoints

- Module entry: `python -m unigeoseg.training.train`
- Local wrapper: `python train.py`

## Supported datasets

The trainer can mix any subset of these sources:

- `RRSIS-D`
- `RefSegRS`
- `EarthReason`
- `RSRS`
- Generic GeoSeg-style manifests via `--geoseg_manifest`

The trainer groups samples into `interactive`, `referring`, and `reasoning`, then applies the paper's `PTS` resampling schedule epoch by epoch.

## Generic manifest format

Use JSON or JSONL. Each sample should include:

```json
{
  "image": "images/example.png",
  "mask": "masks/example.png",
  "task": "interactive",
  "instruction": "Please segment the region corresponding to the box x0,y0=[0.1,0.2], x1,y1=[0.4,0.5].",
  "bbox": [0.1, 0.2, 0.4, 0.5]
}
```

Reasoning samples can additionally include `answer` or `reason`.

## Example commands

### 1. Continue training from an existing UniGeoSeg checkpoint

```bash
python train.py ^
  --model_name_or_path D:\checkpoints\unigeoseg ^
  --rsrs_root D:\data\RSRS ^
  --earthreason_root D:\data\RSReasonSeg ^
  --rrsisd_root D:\data\RRSIS-D ^
  --refsegrs_root D:\data\RefSegRS ^
  --output_dir D:\runs\unigeoseg_train ^
  --deepspeed D:\cvpr\UniGeoSeg\unigeoseg\training\deepspeed_zero2.json
```

### 2. Start from a language-only Phi checkpoint

```bash
python train.py ^
  --model_name_or_path microsoft/phi-1_5 ^
  --vision_tower D:\weights\mask2former_swin_base.pkl ^
  --pretrained_mask_decoder_path D:\weights\mask2former_swin_base.pkl ^
  --rsrs_root D:\data\RSRS ^
  --earthreason_root D:\data\RSReasonSeg ^
  --output_dir D:\runs\unigeoseg_from_phi ^
  --deepspeed D:\cvpr\UniGeoSeg\unigeoseg\training\deepspeed_zero2.json
```

## Notes

- If your checkpoint does not already contain the visual tower or mask decoder, the trainer requires `--vision_tower` and `--pretrained_mask_decoder_path` unless you explicitly set `--allow_random_init true`.
- `RSRS` is automatically split into interactive, referring, and reasoning subsets according to its file layout.
- `EarthReason` is treated as reasoning segmentation.
- `RRSIS-D` and `RefSegRS` are treated as referring segmentation.
