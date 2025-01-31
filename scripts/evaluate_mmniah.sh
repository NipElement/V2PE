#!/bin/bash
# export MASTER_ADDR=localhost
export RANK=$MLP_ROLE_INDEX
export WORLD_SIZE=$(($MLP_WORKER_NUM * $MLP_WORKER_GPU))
export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export LOCAL_RANK=$(($RANK % $MLP_WORKER_GPU))
CHECKPOINT="/map-vepfs/yuansheng/LongContext/trained_models/invervl2_5_8b_stage1_mammoth/checkpoint-4375"
LOG_DIR=eval_logs/mmniah/invervl2_5_8b_stage1_mammoth-baseline-v2pe-stride64

mkdir -p $LOG_DIR  # Create log directory

SOURCE_DIR="/map-vepfs/yuansheng/LongContext/trained_models/invervl2_5_8b_stage1_mammoth"
files=(
    "configuration_intern_vit.py"
    "configuration_internlm2.py"
    "configuration_internvl_chat.py"
    "conversation.py"
    "modeling_intern_vit.py"
    "modeling_internlm2.py"
    "modeling_internvl_chat.py"
    "tokenization_internlm2_fast.py"
    "tokenization_internlm2.py"
)

for file in "${files[@]}"; do
    if [ ! -f "${CHECKPOINT}/${file}" ]; then
        echo "Copying $file to $CHECKPOINT"
        cp "${SOURCE_DIR}/${file}" "${CHECKPOINT}/"
    fi
done

declare -a tasks=( \
    'retrieval-text-test' \ 
    'retrieval-image-test' \ 
    'counting-text-test' \
    'counting-image-test' \
    'reasoning-text-test' \ 
    'reasoning-image-test' \ 
)

model_name="internvl2_5"

# Loop through each task and run evaluation
for task in "${tasks[@]}"; do
    echo "$(date) Starting task: ${task}"
    torchrun \
        --nproc_per_node "$MLP_WORKER_GPU" \
        --nnodes "$MLP_WORKER_NUM" \
        --node_rank "$MLP_ROLE_INDEX" \
        --master_port "$MLP_WORKER_0_PORT" \
        --master_addr "$MLP_WORKER_0_HOST" \
        eval/mm_niah/eval_mm_niah.py \
        --checkpoint $CHECKPOINT \
        --outputs-dir $LOG_DIR \
        --task $task \
        --num-gpus-per-rank=1 \
        --rope_pos_id_version v2pe_fix \
        --rope_pos_id_stride 64 \
        > "${LOG_DIR}/${task}.log" 2>&1
    wait
    echo "$(date) Finished task: ${task}"
done

echo "All tasks completed!"
