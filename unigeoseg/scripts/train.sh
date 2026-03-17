#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/share/zhangyudong6-local/UniGeoSeg
export TORCH_LIB_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")/lib
export LD_LIBRARY_PATH=$TORCH_LIB_PATH:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NPROC_PER_NODE=8
MASTER_PORT=29501

MODEL_NAME_OR_PATH="/share/zhangyudong6-local/UniGeoSeg/pretrained/phi-1_5_dev"
VISION_TOWER="/share/zhangyudong6-local/UniGeoSeg/pretrained/Siwn-B Mask2former/model_final_54b88a.pkl"
PRETRAINED_MASK_DECODER_PATH="/share/zhangyudong6-local/UniGeoSeg/pretrained/Siwn-B Mask2former/model_final_54b88a.pkl"
MASK_CONFIG="/share/zhangyudong6-local/UniGeoSeg/unigeoseg/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"

RSRS_ROOT="/share/zhangyudong6-local/UniGeoSeg/UniGeoSeg_Intera_Dataset"
EARTHREASON_ROOT=""
RRSISD_ROOT=""
REFSEGRS_ROOT=""
GEOSEG_MANIFEST=""

OUTPUT_DIR="/share/zhangyudong6-local/UniGeoSeg/training"
DEEPSPEED_CONFIG="/share/zhangyudong6-local/UniGeoSeg/unigeoseg/training/deepspeed_zero2.json"


PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
NUM_TRAIN_EPOCHS=3
LEARNING_RATE=1e-4
DATALOADER_NUM_WORKERS=4

CMD=(
  torchrun
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  train.py
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --vision_tower "${VISION_TOWER}"
  --pretrained_mask_decoder_path "${PRETRAINED_MASK_DECODER_PATH}"
  --mask_config "${MASK_CONFIG}"
  --output_dir "${OUTPUT_DIR}"
  --deepspeed "${DEEPSPEED_CONFIG}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
)

if [[ -n "${RSRS_ROOT}" ]]; then
  CMD+=(--rsrs_root "${RSRS_ROOT}")
fi

if [[ -n "${EARTHREASON_ROOT}" ]]; then
  CMD+=(--earthreason_root "${EARTHREASON_ROOT}")
fi

if [[ -n "${RRSISD_ROOT}" ]]; then
  CMD+=(--rrsisd_root "${RRSISD_ROOT}")
fi

if [[ -n "${REFSEGRS_ROOT}" ]]; then
  CMD+=(--refsegrs_root "${REFSEGRS_ROOT}")
fi

if [[ -n "${GEOSEG_MANIFEST}" ]]; then
  CMD+=(--geoseg_manifest "${GEOSEG_MANIFEST}")
fi

CMD+=("$@")

printf ' %q' "${CMD[@]}"
echo

exec "${CMD[@]}"
