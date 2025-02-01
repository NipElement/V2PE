#!/bin/bash
NUM_GPUS_PER_NODE=4
# export MASTER_ADDR=localhost
# export WORLD_SIZE=1  # Total number of processes
# GPUS=$WORLD_SIZE
# export RANK=$MLP_ROLE_INDEX
# export WORLD_SIZE=$(($MLP_WORKER_NUM * $MLP_WORKER_GPU))
# export MASTER_ADDR=$MLP_WORKER_0_HOST
# export MASTER_PORT=$MLP_WORKER_0_PORT
# export LOCAL_RANK=$(($RANK % $MLP_WORKER_GPU))
CHECKPOINT="/map-vepfs/yuansheng/LongContext/V2PE/pretrained/InternVL2_5-8B"
LOG_DIR=eval_logs/milebench_shot/InternVL2_5-8B-5shot

# echo "Starting evaluation for MileBench tasks"
# echo "Distribution: $MLP_WORKER_NUM nodes, $MLP_WORKER_GPU GPUs per node, $WORLD_SIZE total processes, $RANK current process, $LOCAL_RANK local rank, $MASTER_ADDR master address, $MASTER_PORT master port".

mkdir -p $LOG_DIR  # Create log directory

# Task list
declare -a tasks=( \
    'ALFRED' \
)

model_name="invervl2_5_8b"

for ((j=0; j<${#tasks[@]}; j++)); do

    task="${tasks[j]}"

    
    echo "$(date) ${model_name}_${task}"

    # torchrun \
    #     --nproc_per_node "$MLP_WORKER_GPU" \
    #     --nnodes "$MLP_WORKER_NUM" \
    #     --node_rank "$MLP_ROLE_INDEX" \
    #     --master_port "$MLP_WORKER_0_PORT" \
    #     --master_addr "$MLP_WORKER_0_HOST" \
    torchrun \
        --nproc_per_node $NUM_GPUS_PER_NODE \
        --standalone \
        eval/milebench/eval_milebench_shot.py \
        --checkpoint $CHECKPOINT \
        --output_dir $LOG_DIR \
        --dataset_name $task \
        --num-gpus-per-rank 1 \
        --n-shot 5 \
        --example-seed 42 \
        --example-max-patch 4\
        > "${LOG_DIR}/${task}.log" 2>&1
    sleep 0.2
done

echo "All tasks completed!"


        # --rope_pos_id_version v2pe_fix \
        # --rope_pos_id_stride 2 \