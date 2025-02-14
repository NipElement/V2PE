#!/bin/bash
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi
NUM_GPUS_PER_NODE=8
# export MASTER_ADDR=localhost
# export WORLD_SIZE=1  # Total number of processes
# GPUS=$WORLD_SIZE
# export RANK=$MLP_ROLE_INDEX
# export WORLD_SIZE=$(($MLP_WORKER_NUM * $MLP_WORKER_GPU))
# export MASTER_ADDR=$MLP_WORKER_0_HOST
# export MASTER_PORT=$MLP_WORKER_0_PORT
# export LOCAL_RANK=$(($RANK % $MLP_WORKER_GPU))
CHECKPOINT="/map-vepfs/yuansheng/LongContext/trained_models/invervl2_5_8b_stage2_mammoth_baseline"
LOG_DIR=eval_logs/milebench/internvl2_5_8b_stage2_mammoth-baseline-v2pe-stride64

# echo "Starting evaluation for MileBench tasks"
# echo "Distribution: $MLP_WORKER_NUM nodes, $MLP_WORKER_GPU GPUs per node, $WORLD_SIZE total processes, $RANK current process, $LOCAL_RANK local rank, $MASTER_ADDR master address, $MASTER_PORT master port".

mkdir -p $LOG_DIR  # Create log directory

# Task list
declare -a tasks=( \
    'ALFRED' \
    'ActionLocalization' \
    'ActionPrediction' \
    'ActionSequence' \
    'CLEVR-Change' \
    'CharacterOrder' \
    'CounterfactualInference' \
    'DocVQA' \
    'EgocentricNavigation' \
    'GPR1200' \
    'IEdit' \
    'ImageNeedleInAHaystack' \
    'MMCoQA' \
    'MovingAttribute' \
    'MovingDirection' \
    'MultiModalQA' \
    'OCR-VQA' \
    'ObjectExistence' \
    'ObjectInteraction' \
    'ObjectShuffle' \
    'SceneTransition' \
    'SlideVQA' \
    'Spot-the-Diff' \
    'StateChange' \
    'TQA' \
    'TextNeedleInAHaystack' \
    'WebQA' \
    'WikiVQA' \
    'nuscenes' \
)

model_name="internvl2_5_8b_stage2_mammoth-baseline"

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
        eval/milebench/eval_milebench.py \
        --checkpoint $CHECKPOINT \
        --output_dir $LOG_DIR \
        --dataset_name $task \
        --num-gpus-per-rank 1 \
        --rope_pos_id_version v2pe_fix \
        --rope_pos_id_stride 64 \
        > "${LOG_DIR}/${task}.log" 2>&1
    sleep 0.2
done

echo "All tasks completed!"

