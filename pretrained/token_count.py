import os
import json
import threading
from queue import Queue
from transformers import AutoTokenizer
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import statistics
Image.MAX_IMAGE_PIXELS = None
# 📂 **路径配置**
TOKENIZER_PATH = "/map-vepfs/yuansheng/LongContext/codebase/V2PE/pretrained/InternVL2-2B"
MERGED_JSON_PATH = "/map-vepfs/yuansheng/LongContext/data/WebGUI/metadata.json"
IMAGE_BASE_PATH = "/map-vepfs/yuansheng/LongContext/data/WebGUI"
OUTPUT_TOKEN_STATS_PATH = "/map-vepfs/yuansheng/LongContext/data/WebGUI/token_stats.jsonl"
SUMMARY_STATS_PATH = "/map-vepfs/yuansheng/LongContext/data/WebGUI/token_summary.json"

# 🛠️ **图像预处理**
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def dynamic_preprocess(image, image_size=448):
    orig_width, orig_height = image.size
    num_tiles_x = orig_width // image_size
    num_tiles_y = orig_height // image_size
    total_tiles = num_tiles_x * num_tiles_y

    resized_img = image.resize((num_tiles_x * image_size, num_tiles_y * image_size))
    processed_images = []

    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            box = (
                x * image_size,
                y * image_size,
                (x + 1) * image_size,
                (y + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

    return processed_images

def load_image(image_file, image_size=448):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=image_size)
    images = dynamic_preprocess(image, image_size=image_size)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# 🔄 **线程处理函数**
def worker(queue, results, lock, progress_bar):
    """
    每个线程处理队列中的样本，并将结果存入 results 列表
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    
    while not queue.empty():
        try:
            sample_id, sample = queue.get_nowait()
        except Exception:
            break  # 队列为空，退出线程
        
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        image_path = os.path.join(IMAGE_BASE_PATH, sample.get("image_path", ""))

        # 处理文本 Token
        combined_text = f"{question}\n{answer}"
        text_tokens = tokenizer(combined_text, return_tensors="pt", add_special_tokens=True)
        text_token_count = text_tokens['input_ids'].shape[1]

        # 处理视觉 Token
        try:
            pixel_values = load_image(image_path, image_size=448)
            visual_token_count = pixel_values.shape[0]
        except Exception as e:
            print(f"❌ Failed to process image {image_path}: {e}")
            visual_token_count = 0

        total_tokens = text_token_count + visual_token_count

        sample_result = {
            "id": sample_id,
            "text_tokens": text_token_count,
            "visual_tokens": visual_token_count,
            "total_tokens": total_tokens
        }

        with lock:
            results.append(sample_result)
            with open(OUTPUT_TOKEN_STATS_PATH, "a") as output_file:
                output_file.write(json.dumps(sample_result) + "\n")
            progress_bar.update(1)

        queue.task_done()

# 📊 **主处理函数**
def process_all_samples(num_threads=4):
    """
    使用多线程处理所有样本，并展示实时进度条
    """
    with open(MERGED_JSON_PATH, "r") as f:
        metadata = json.load(f)

    sample_queue = Queue()
    for sample_id, sample in metadata.items():
        sample_queue.put((sample_id, sample))
    
    results = []
    lock = threading.Lock()
    threads = []

    # 启动进度条
    total_samples = len(metadata)
    with tqdm(total=total_samples, desc="Processing Samples", unit="sample") as progress_bar:
        # 启动线程
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(sample_queue, results, lock, progress_bar))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

    return results

# 📈 **统计整体信息**
def summarize_stats(results):
    token_counts = [res["total_tokens"] for res in results]
    
    summary_stats = {
        "max_tokens": max(token_counts),
        "min_tokens": min(token_counts),
        "avg_tokens": statistics.mean(token_counts),
        "median_tokens": statistics.median(token_counts),
        "total_samples": len(token_counts)
    }

    with open(SUMMARY_STATS_PATH, "w") as f:
        json.dump(summary_stats, f, indent=2)

    return summary_stats

# 🚀 **执行主程序**
if __name__ == "__main__":
    print("🔄 Starting multi-threaded token processing...")
    
    # 清空输出文件
    if os.path.exists(OUTPUT_TOKEN_STATS_PATH):
        os.remove(OUTPUT_TOKEN_STATS_PATH)
    
    results = process_all_samples(num_threads=64)

    print("📊 Summarizing results...")
    stats = summarize_stats(results)

    print("\n✅ **Token Statistics Completed**")
    print(f"🔢 Total Samples: {stats['total_samples']}")
    print(f"📊 Max Tokens: {stats['max_tokens']}")
    print(f"📉 Min Tokens: {stats['min_tokens']}")
    print(f"📈 Average Tokens: {stats['avg_tokens']:.2f}")
    print(f"⚖️ Median Tokens: {stats['median_tokens']}")
    print(f"💾 Stats saved to: {SUMMARY_STATS_PATH}")
    print(f"💾 Per-sample tokens saved to: {OUTPUT_TOKEN_STATS_PATH}")