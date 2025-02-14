#!/bin/bash
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

NUM_GPUS_PER_NODE=8
CHECKPOINT="/map-vepfs/yuansheng/LongContext/V2PE/pretrained/InternVL2_5-8B"

model_name="internvl2_5_8b"

# 定义多个 n-shot 数值
declare -a shots=(1 2 3 5 10 20)

# Task list
declare -a tasks=( \
    # 'ALFRED' \
    # 'ActionLocalization' \
    # 'ActionPrediction' \
    # 'ActionSequence' \
    # 'CLEVR-Change' \
    # 'CharacterOrder' \
    # 'CounterfactualInference' \
    # 'DocVQA' \
    # 'EgocentricNavigation' \
    # 'GPR1200' \
    # 'IEdit' \
    # 'ImageNeedleInAHaystack' \
    # 'MMCoQA' \
    # 'MovingAttribute' \
    # 'MovingDirection' \
    # 'MultiModalQA' \
    # 'OCR-VQA' \
    # 'ObjectExistence' \
    # 'ObjectInteraction' \
    # 'ObjectShuffle' \
    # 'SceneTransition' \
    'SlideVQA' \
    # 'Spot-the-Diff' \
    # 'StateChange' \
    # 'TQA' \
    # 'TextNeedleInAHaystack' \
    # 'WebQA' \
    # 'WikiVQA' \
    # 'nuscenes' \
)

for shot in "${shots[@]}"; do
    LOG_DIR="eval_logs/milebench_${shot}shot/${model_name}"
    mkdir -p "$LOG_DIR"
    
    for task in "${tasks[@]}"; do
        echo "$(date) Running ${model_name}_${task} with n-shot=${shot}"
        
        torchrun \
            --nproc_per_node "$NUM_GPUS_PER_NODE" \
            --standalone \
            eval/milebench/eval_milebench_shot.py \
            --checkpoint "$CHECKPOINT" \
            --output_dir "$LOG_DIR" \
            --dataset_name "$task" \
            --num-gpus-per-rank 1 \
            --n-shot "$shot" \
            --example-seed 42 \
            --example-max-patch 6 \
            > "${LOG_DIR}/${task}.log" 2>&1
        
        sleep 0.2
    done
done
