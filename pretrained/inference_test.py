import os
import json
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# Step 1: 初始化 Tokenizer 和 模型
TOKENIZER_PATH = "/map-vepfs/yuansheng/LongContext/codebase/V2PE/pretrained/InternVL2-2B"
MODEL_PATH = "/map-vepfs/yuansheng/LongContext/codebase/V2PE/pretrained/InternVL2-2B"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().cuda()

# Step 2: 配置路径
MERGED_JSON_PATH = "/map-vepfs/yuansheng/LongContext/data/WebGUI/metadata.json"
IMAGE_BASE_PATH = "/map-vepfs/yuansheng/LongContext/data/WebGUI"

# Step 3: 图像预处理
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def preprocess_image(image_path, image_size=448):
    """
    加载并预处理图像，返回 pixel_values
    """
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=image_size)
    processed_image = transform(image)
    return processed_image.unsqueeze(0)  # 增加 batch 维度

# Step 4: 加载指定的样本
SPECIFIC_KEY = "fashion_0002_v4_50K_4_curated_v2_webqa_33938"

with open(MERGED_JSON_PATH, "r") as f:
    metadata = json.load(f)

# 检查指定的 key 是否存在
if SPECIFIC_KEY not in metadata:
    raise KeyError(f"❌ Key '{SPECIFIC_KEY}' not found in metadata.json")

# 获取指定样本
sample = metadata[SPECIFIC_KEY]
print(f"🔍 **Processing Sample:** {SPECIFIC_KEY}")

# Step 5: 提取 question, answer 和 image_path
question = sample.get("question", "")
answer = sample.get("answer", "")
image_path = os.path.join(IMAGE_BASE_PATH, sample.get("image_path", ""))

if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image not found: {image_path}")

# Step 6: 处理图像
pixel_values = preprocess_image(image_path).to(torch.bfloat16).cuda()

# Step 7: 替换 `<image>` 标签
if "<image>" in question:
    question = question.replace("<image>", "<image>\n")
else:
    print("⚠️ **No <image> tag found in question. Proceeding with the original question.**")

# Step 8: 模型推理
print("🤖 **Running inference...**")
generation_config = dict(max_new_tokens=1024, do_sample=True)

response = model.chat(
    tokenizer=tokenizer,
    pixel_values=pixel_values,
    question=question,
    generation_config=generation_config
)

# Step 9: 输出结果
print("\n🎯 **Inference Result:**")
print(f"📝 **Question:** {question}")
print(f"💬 **Answer:** {response}")

# ⚙️ **调试信息**
print("\n🛠️ Debug Info:")
print(f" - Image Path: {image_path}")
print(f" - Image Shape: {pixel_values.shape}")
print(f" - Question: {question}")