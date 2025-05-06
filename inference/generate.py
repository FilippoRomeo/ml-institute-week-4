# inference/generate.py

import torch
from PIL import Image
from torchvision import transforms
from models.caption_model import CaptioningModel
from models.encoder_vit import ImagePatcher
from data.tokenizer import CaptionTokenizer
import argparse
import os

# 1. Load image and preprocess
def load_image(image_path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # [1, 3, H, W]

# 2. Inference (greedy decoding)
@torch.no_grad()
def generate_caption(model, image_tensor, tokenizer, image_patcher, max_len=30, device="cuda"):
    model.eval()
    print("DEBUG: image_tensor shape:", image_tensor.shape)

    # Start with <sos>
    generated = [tokenizer.sos_id]

    for _ in range(max_len):
        input_tensor = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
        logits = model(image_tensor.to(device), input_tensor)  # ✅ Pass raw image here
        next_token_logits = logits[0, -1]  # Last step
        next_token = next_token_logits.argmax(-1).item()
        if next_token == tokenizer.eos_id:
            break
        generated.append(next_token)

    return tokenizer.decode(generated[1:])  # skip <sos> in decoding

# 3. CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/caption_model_epoch5.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load components
    tokenizer = CaptionTokenizer()
    model = CaptioningModel(vocab_size=8000).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    image_patcher = ImagePatcher().to(device)

    # Load image
    image_tensor = load_image(args.image)

    # Generate caption
    caption = generate_caption(model, image_tensor, tokenizer, image_patcher, device=device)
    print("\n🖼️ Caption:", caption)

if __name__ == "__main__":
    main()
