#!/usr/bin/env bash
set -x        

export RANK=$MLP_ROLE_INDEX
export WORLD_SIZE=$(($MLP_WORKER_NUM * $MLP_WORKER_GPU))
export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export LOCAL_RANK=$(($RANK % $MLP_WORKER_GPU))


OUTPUT_DIR="/map-vepfs/yuansheng/LongContext/trained_models/invervl2_5_8b_stage1_100K"
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

files=(
  "internvl2_5/conversation.py"
  "internvl2_5/model/internlm2/configuration_internlm2.py"
  "internvl2_5/model/internlm2/modeling_internlm2.py"
  "internvl2_5/model/internlm2/tokenization_internlm2_fast.py"
  "internvl2_5/model/internlm2/tokenization_internlm2.py"
  "internvl2_5/model/internvl_chat/configuration_intern_vit.py"
  "internvl2_5/model/internvl_chat/configuration_internvl_chat.py"
  "internvl2_5/model/internvl_chat/modeling_intern_vit.py"
  "internvl2_5/model/internvl_chat/modeling_internvl_chat.py"
)

for file in "${files[@]}"; do
  cp "$file" "$OUTPUT_DIR"
done

MODEL_PATH="/map-vepfs/yuansheng/LongContext/V2PE/pretrained/InternVL2_5-8B"
META_PATH="/map-vepfs/yuansheng/LongContext/V2PE/shell/data/annotation_train_100K.json"

torchrun \
  --nproc_per_node "$MLP_WORKER_GPU" \
  --nnodes "$MLP_WORKER_NUM" \
  --node_rank "$MLP_ROLE_INDEX" \
  --master_addr "$MLP_WORKER_0_HOST" \
  --master_port "$MLP_WORKER_0_PORT" \
  internvl2_5/train/internvl2_5_chat_finetune.py \
  --model_name_or_path "${MODEL_PATH}" \
  --conv_style "internlm2-chat" \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir True \
  \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --drop_path_rate 0.1 \
  \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --pad2square False \
  --dynamic_image_size True \
  --dynamic_max_patch False \
  --max_dynamic_patch 512 \
  --use_thumbnail True \
  \
  --dataloader_num_workers 2 \
  --max_seq_length 131072 \
  --use_packed_ds True \
  --chunk_num 8 \
  --group_by_length False \
  --max_packed_tokens 131072 \
  --max_buffer_size 20 \
  --num_images_expected 512 \
  --allow_overflow False \
  --use_data_resampling False \
  --replacement False \
  --remove_unused_columns False \
  \
  --do_train True \
  --bf16 True \
  --num_train_epochs 1 \
  --max_steps 20000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --learning_rate 5e-6 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  \
  --logging_steps 1 \
  --save_strategy "steps" \
  --save_steps 2500 \
  --save_total_limit 10 \
  --report_to "tensorboard" \
  --log_freq 1000 \
  \
  --grad_checkpoint True \
  --ps_version 'v2' \
  --deepspeed "zero_stage3_config.json" \
  --strict_mode False \
  --rope_pos_id_version 'v2pe_rnd' \
  --attn_type 'ring' \
  \
  --loss_reduction "square" \
  --loss_reduction_all_gather True \
  \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
