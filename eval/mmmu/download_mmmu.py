import os
import json
from datasets import load_dataset, concatenate_datasets
from data_utils import CAT_SHORT2LONG  # 确保这个模块可以导入

# 缓存路径
cache_dir = "/map-vepfs/yuansheng/tmp/MMMU"

# 数据集根目录
root = "MMMU/MMMU"
split = "validation"

# 遍历所有 subject（子数据集）
sub_dataset_list = []
print(f"Downloading and loading MMMU dataset from {root} (split: {split})...")

try:
    for subject in CAT_SHORT2LONG.values():
        print(f"Loading subject: {subject} ...")
        sub_dataset = load_dataset(root, subject, split=split, cache_dir=cache_dir)
        sub_dataset_list.append(sub_dataset)

    # 合并所有 subject 的数据集
    full_dataset = concatenate_datasets(sub_dataset_list)
    print(f"Successfully loaded dataset with {len(full_dataset)} samples.")

    # 输出部分数据示例
    print("Sample data:", json.dumps(full_dataset[0], indent=4))

except Exception as e:
    print(f"Error while downloading/loading dataset: {e}")
