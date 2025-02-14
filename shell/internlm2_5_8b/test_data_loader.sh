#!/usr/bin/env bash
set -x

NUM_GPUS_PER_NODE=1


MODEL_PATH="/map-vepfs/yuansheng/LongContext/trained_models/invervl2_5_8b_stage2_mammoth_baseline"
META_PATH="/map-vepfs/yuansheng/LongContext/V2PE/shell/data/annotation_train_debug.json"

OUTPUT_DIR="/map-vepfs/yuansheng/LongContext/V2PE/debug_data"

# 若目录不存在则创建
mkdir -p "$OUTPUT_DIR"

# torchrun \
#     --nproc_per_node "$NUM_GPUS_PER_NODE" \
#     --standalone \
#     internvl2_5/train/test_data.py \
#     \
#     --model_name_or_path "${MODEL_PATH}" \
#     --conv_style "internlm2-chat" \
#     --output_dir "${OUTPUT_DIR}" \
#     --meta_path "${META_PATH}" \
#     --overwrite_output_dir True \
#     \
#     --force_image_size 448 \
#     --down_sample_ratio 0.5 \
#     --pad2square False \
#     --dynamic_image_size True \
#     --dynamic_max_patch False \
#     --max_dynamic_patch 256 \
#     --use_thumbnail True \
#     --max_seq_length 32768 \
#     \
#     --dataloader_num_workers 1 \
#     --use_packed_ds False \
#     --group_by_length False \
#     --max_packed_tokens 32768 \
#     --max_buffer_size 20 \
#     --num_images_expected 160 \
#     --allow_overflow False \
#     --use_data_resampling False \
#     --replacement False \
#     --remove_unused_columns False \
#     \
#     --bf16 True \
#     --rope_pos_id_version 'v2pe_rnd' \
#     --strict_mode False \
#     --grad_checkpoint True \
#     --ps_version 'v2' \
#     --loss_reduction "square" \
#     --loss_reduction_all_gather True \
#     \
#     --error_output_dir "${OUTPUT_DIR}/validation_errors" \
#     --stop_on_error False \
#     2>&1 | tee -a "${OUTPUT_DIR}/check_data_log.txt"

python internvl2_5/train/test_data.py \
    --model_name_or_path "${MODEL_PATH}" \
    --conv_style "internlm2-chat" \
    --output_dir "${OUTPUT_DIR}" \
    --meta_path "${META_PATH}" \
    --overwrite_output_dir True \
    \
    --force_image_size 448 \
    --down_sample_ratio 0.5 \
    --pad2square False \
    --dynamic_image_size True \
    --dynamic_max_patch False \
    --max_dynamic_patch 256 \
    --use_thumbnail True \
    --max_seq_length 65536 \
    \
    --dataloader_num_workers 1 \
    --use_packed_ds False \
    --group_by_length False \
    --max_packed_tokens 65536 \
    --max_buffer_size 20 \
    --num_images_expected 256 \
    --allow_overflow False \
    --use_data_resampling False \
    --replacement False \
    --remove_unused_columns False \
    \
    --bf16 True \
    --rope_pos_id_version 'v2pe_rnd' \
    --strict_mode False \
    --grad_checkpoint True \
    --ps_version 'v2' \
    --loss_reduction "square" \
    --loss_reduction_all_gather True \
    --error_output_dir "${OUTPUT_DIR}/stage2_icl_errors" \
    --stop_on_error False \
    --num_workers 128
