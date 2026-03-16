export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/share/zhangyudong6-local/UniGeoSeg
export TORCH_LIB_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")/lib
export LD_LIBRARY_PATH=$TORCH_LIB_PATH:$LD_LIBRARY_PATH
python unigeoseg/eval_and_test/eval.py \
  --model_path /share/zhangyudong6-local/UniGeoSeg/checkpoints/unigeoseg_model \
  --data_split "test" \
  --version "llava_phi" \
  --mask_config "unigeoseg/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml" \
  --dataset_type "RSRS" \
  --base_data_path "/share/zhangyudong6-local/UniGeoSeg/UniGeoSeg_Intera_Dataset
  " \
