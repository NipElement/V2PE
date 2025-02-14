# 文件: internvl2_5_data_validator_no_dist.py

import os
import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from internvl2_5.train.internvl2_5_chat_finetune import (
    ModelArguments,
    DataTrainingArguments,
    InternVLChatConfig,
    build_datasets,  # <-- 你原先的 build_datasets
    IMG_START_TOKEN,
    IMG_END_TOKEN,
    IMG_CONTEXT_TOKEN,
    QUAD_START_TOKEN,
    QUAD_END_TOKEN,
    REF_START_TOKEN,
    REF_END_TOKEN,
    BOX_START_TOKEN,
    BOX_END_TOKEN,
)
# 如果 LazySupervisedDataset 并不在上面的文件里，而是在别的地方定义
# 那么你需要自行导入，比如:
# from internvl2_5.train.dataset import LazySupervisedDataset
import torch.distributed as dist

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config

    has_tcs_loader = True
except ImportError as E:
    # print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

def fake_init_process_group(*args, **kwargs):
    pass

def fake_new_group(*args, **kwargs):
    pass

def fake_destroy_process_group(*args, **kwargs):
    pass

dist.init_process_group = fake_init_process_group
dist.new_group = fake_new_group
dist.destroy_process_group = fake_destroy_process_group
dist.is_initialized = lambda: True   # 骗过 dist.get_rank() 内部检查
dist.get_rank = lambda: 0
dist.get_world_size = lambda: 1

@dataclass
class ValidatorArguments:
    error_output_dir: str = field(
        default="./data_errors",
        metadata={'help': 'Directory to save error information'}
    )
    stop_on_error: bool = field(
        default=False,
        metadata={'help': 'Whether to stop when encountering the first error'}
    )
    max_errors: int = field(
        default=100,
        metadata={'help': 'Maximum number of errors to collect before stopping'}
    )
    num_workers: int = field(
        default=4,
        metadata={'help': 'Number of worker threads for data loading'}
    )

def find_dataset_index(concat_dataset, global_idx):
    """
    在ConcatDataset中找到对应的子数据集索引和在该子集内的索引。
    如果你构建的train_dataset并不是ConcatDataset，就无需该函数。
    """
    current_idx = 0
    for ds_idx, dataset in enumerate(concat_dataset.datasets):
        if global_idx - current_idx < len(dataset):
            return ds_idx, global_idx - current_idx
        current_idx += len(dataset)
    raise IndexError("Global index out of range")

def calculate_expected_end_tokens(data_item):
    """
    根据原始数据项计算预期的IMG_END_TOKEN出现次数。
    """
    # 图像数据
    if 'image' in data_item:
        if isinstance(data_item['image'], list):
            return len(data_item['image'])
        return 1 if data_item['image'] else 0

    # 视频数据（根据原始逻辑）
    if 'video' in data_item:
        return 1  # 视频统一认为会出现1次 <image>...<image>

    # 纯文本数据
    return 0

def main():
    # 使用与原训练脚本一致的参数结构
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, ValidatorArguments))
    model_args, data_args, training_args, validator_args = parser.parse_args_into_dataclasses()

    # 日志
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    set_seed(training_args.seed)

    # 1) 构建 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_path or model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True
    )
    # 补充特殊 token
    token_list = [
        IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
        REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN
    ]
    tokenizer.add_tokens(token_list, special_tokens=True)

    # 2) 加载模型配置，只是为了拿到vision_config.patch_size 等
    #    如不需要也可写死 patch_size 或从别处获取
    logger.info("Loading model config to get patch_size ...")
    config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
    patch_size = config.vision_config.patch_size

    # 计算 num_image_token
    num_image_token = int(
        (data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2)
    )
    if model_args.img_emb_down_sample_ratio is not None:
        num_image_token = int(num_image_token / model_args.img_emb_down_sample_ratio)
    logger.info(f"[Validator] num_image_token={num_image_token}, patch_size={patch_size}")

    # 3) 构建数据集
    #    这里不实际载入模型，而是弄个简单的FakeModel对象把 num_image_token 传进去即可
    FakeModel = type('FakeModel', (), {})
    fake_model = FakeModel()
    fake_model.num_image_token = num_image_token

    if os.environ.get('no_tcs', False):
        tcs_loader = None
    else:
        tcs_loader = TCSLoader('~/petreloss_zy.conf') if has_tcs_loader else None
    
    train_dataset = build_datasets(
        data_args=data_args,
        tokenizer=tokenizer,
        tcs_loader=tcs_loader,  # 如果不需要远端 petrel/s3 就传 None
        model=fake_model,
        group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size,
        dynamic_max_patch=data_args.dynamic_max_patch,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type,
        rope_pos_id_version=data_args.rope_pos_id_version,
        rope_pos_id_stride=data_args.rope_pos_id_stride
    )

    # 4) 创建 DataLoader 来多线程读取
    #    注意: 如果你的 Dataset 实现里本身带了随机数操作/分布式操作，可能需要做改动或强制关闭
    logger.info("Building DataLoader with num_workers=%d ...", validator_args.num_workers)
    data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=validator_args.num_workers,
        # 由于你的 Dataset.__getitem__ 返回的是一个字典，这里简单起见:
        collate_fn=lambda batch: batch[0]  # 每次batch=1，直接取出
    )

    # 5) 验证循环 & 记录错误
    error_log = []
    os.makedirs(validator_args.error_output_dir, exist_ok=True)
    output_path = os.path.join(validator_args.error_output_dir, "data_validation_errors.json")

    logger.info("Start validating data ... total samples = %d", len(train_dataset))

    # 注意: enumerate(data_loader) 时，下标 idx 不一定就是原先的 dataset index，
    #       如果 ConcatDataset, 需要 find_dataset_index 来反查。
    #       如果你构建的 train_dataset 不是 ConcatDataset，而是单一数据集，可酌情简化。
    for idx, sample in enumerate(tqdm(data_loader, total=len(train_dataset), desc="Validating")):
        try:
            # 在 ConcatDataset 场景下，需要 find_dataset_index 找回对应子集 & 样本下标
            ds_idx, sample_idx = find_dataset_index(train_dataset, idx)
            raw_dataset = train_dataset.datasets[ds_idx]
            raw_data = raw_dataset.raw_data[sample_idx]
            data_item = json.loads(raw_data)

            # 计算期望出现的 IMG_END_TOKEN 次数
            expected_end_tokens = calculate_expected_end_tokens(data_item)
            image_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

            # 统计实际出现次数
            actual_end_tokens = (sample['input_ids'] == image_end_token_id).sum().item()
            if actual_end_tokens != expected_end_tokens:
                raise AssertionError(
                    f"Image tokens mismatch. Expected {expected_end_tokens}, got {actual_end_tokens}. "
                    f"Dataset: {raw_dataset.ds_name}"
                )

        except Exception as e:
            error_info = {
                "index": idx,
                "error_type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            # 尝试放一些上下文信息
            if 'data_item' in locals():
                error_info["data_item"] = data_item
            if 'raw_dataset' in locals():
                error_info["dataset"] = raw_dataset.ds_name

            error_log.append(error_info)
            logger.error("Error processing sample %d: %s", idx, str(e))

            if validator_args.stop_on_error and len(error_log) >= validator_args.max_errors:
                logger.warning("Stopping early at %d errors", len(error_log))
                break

    # 6) 保存结果
    logger.info("Validation complete. Found %d errors.", len(error_log))
    logger.info("Writing error log to %s", output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(error_log, f, indent=2, ensure_ascii=False)

    logger.info("Done.")
    pass

if __name__ == '__main__':
    main()
