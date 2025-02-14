#!/bin/bash
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi
export RANK=$MLP_ROLE_INDEX
export WORLD_SIZE=$(($MLP_WORKER_NUM * $MLP_WORKER_GPU))
export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export LOCAL_RANK=$(($RANK % $MLP_WORKER_GPU))
CHECKPOINT="/map-vepfs/yuansheng/LongContext/trained_models/invervl2_5_8b_stage1_mammoth_baseline/checkpoint-2916"
BASE_LOG_DIR="eval_logs/milebench_1shot"

model_name="internvl2_5_8b_stage1_mammoth_baseline"

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

declare -a strides=(1 2 4 8 16 32 128 256)

for stride in "${strides[@]}"; do
    LOG_DIR="${BASE_LOG_DIR}/${model_name}-v2pe-stride${stride}"
    mkdir -p "$LOG_DIR"
    
    for task in "${tasks[@]}"; do
        echo "$(date) Running ${model_name}_${task} with rope_pos_id_stride=${stride}"       
            torchrun \
            --nproc_per_node "$MLP_WORKER_GPU" \
            --nnodes "$MLP_WORKER_NUM" \
            --node_rank "$MLP_ROLE_INDEX" \
            --master_port "$MLP_WORKER_0_PORT" \
            --master_addr "$MLP_WORKER_0_HOST" \
            eval/milebench/eval_milebench_shot.py \
            --checkpoint $CHECKPOINT \
            --output_dir $LOG_DIR \
            --dataset_name $task \
            --rope_pos_id_version v2pe_fix \
            --rope_pos_id_stride "$stride" \
            --num-gpus-per-rank 1 \
            --n-shot 1 \
            --example-seed 42 \
            --example-max-patch 6 \
            > "${LOG_DIR}/${task}.log" 2>&1
        sleep 0.2
    done
done