from datasets import load_dataset
from PIL import Image
import os

# Load one example
dataset = load_dataset("nlphuji/flickr30k", split="test")
sample = dataset[0]
image = sample["image"]
caption = sample["caption"]

# Save to file
os.makedirs("inference", exist_ok=True)
image_path = "inference/sample.jpg"
image.save(image_path)

print(f"✅ Saved sample image to {image_path}")
print("📝 Sample captions:")
for c in caption:
    print("•", c)
