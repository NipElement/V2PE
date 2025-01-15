import os
import json
import random
import torch
from PIL import Image
from internvl.train.dataset import dynamic_preprocess, build_transform

# --------------------------
# 配置参数
# --------------------------
DATA_PATH = "/map-vepfs/yuansheng/LongContext/codebase/V2PE/dataset/annotation/multiui_train_split.jsonl"
BASE_IMAGE_DIR = "/map-vepfs/yuansheng/LongContext/codebase/V2PE/dataset/image/multiui"
IMAGE_SIZE = 224  # 预处理时图像块的大小
MIN_DYNAMIC_PATCH = 1  # 最小切分块数量
MAX_DYNAMIC_PATCH = 6  # 最大切分块数量（你可根据测试结果调整）
USE_THUMBNAIL = False  # 是否使用缩略图
NORMALIZE_TYPE = 'imagenet'  # 图像归一化类型
NUM_SAMPLES = 10  # 随机抽取的样本数量

# --------------------------
# 加载数据
# --------------------------
def load_random_samples(data_path, num_samples):
    with open(data_path, 'r') as f:
        data = f.readlines()
    random_samples = random.sample(data, num_samples)
    return [json.loads(sample) for sample in random_samples]


# --------------------------
# 测试 Tile 分割逻辑
# --------------------------
def test_tile_splitting(samples):
    transform = build_transform(
        is_train=False,
        input_size=IMAGE_SIZE,
        pad2square=False,
        normalize_type=NORMALIZE_TYPE
    )
    
    results = []
    for idx, sample in enumerate(samples):
        image_path = sample['image']
        full_image_path = os.path.join(BASE_IMAGE_DIR, image_path)
        print(f"\n🖼️ Processing Sample {idx + 1}: {full_image_path}")
        
        try:
            image = Image.open(full_image_path).convert('RGB')
            orig_size = image.size
            
            # 使用 dynamic_preprocess 进行分割
            images, boxes = dynamic_preprocess(
                image,
                min_num=MIN_DYNAMIC_PATCH,
                max_num=MAX_DYNAMIC_PATCH,
                image_size=IMAGE_SIZE,
                use_thumbnail=USE_THUMBNAIL,
                return_box=True
            )
            
            num_tiles = len(images)
            print(f"✅ Original Image Size: {orig_size}")
            print(f"✅ Number of Tiles: {num_tiles}")
            print(f"✅ Tile Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
            
            results.append({
                "image_path": full_image_path,
                "original_size": orig_size,
                "num_tiles": num_tiles,
            })
        
        except Exception as e:
            print(f"❌ Error processing image {full_image_path}: {e}")
            results.append({
                "image_path": full_image_path,
                "error": str(e)
            })
    
    return results


# --------------------------
# 主测试流程
# --------------------------
def main():
    print("🚀 Loading random samples from dataset...")
    samples = load_random_samples(DATA_PATH, NUM_SAMPLES)
    
    print("🔍 Testing tile splitting logic on selected samples...")
    results = test_tile_splitting(samples)
    
    print("\n📊 Summary of Tile Splitting Test:")
    valid_results = [result for result in results if 'num_tiles' in result]
    if valid_results:
        max_tiles = max(result['num_tiles'] for result in valid_results)
        min_tiles = min(result['num_tiles'] for result in valid_results)
        avg_tiles = sum(result['num_tiles'] for result in valid_results) / len(valid_results)
        
        print(f"✅ Maximum Tiles: {max_tiles}")
        print(f"✅ Minimum Tiles: {min_tiles}")
        print(f"✅ Average Tiles: {avg_tiles:.2f}")
    else:
        print("❌ No valid results found. Please check errors above.")
    
    print("\n📄 Detailed Results:")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()