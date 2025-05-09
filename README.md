# 🖼️ Image Captioning Transformer

A Transformer-based image captioning system implemented from scratch in PyTorch. Supports both **Vision Transformer (ViT)** and **CLIP** as encoders and allows you to choose between a custom **SentencePiece tokenizer** or a pretrained **Hugging Face tokenizer** (`EleutherAI/pythia-160m`).

---

## ✅ Requirements

Tested on **Ubuntu 22.04** with:

- Python 3.11+
- CUDA-enabled GPU (optional but recommended)
- `conda` (for environment management)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/FilippoRomeo/ml-institute-week-4.git
cd image_captioning_transformer
```

### 2. Create and Activate Conda Environment

```bash
conda create -n ai-lab python=3.11 -y
conda activate ai-lab
```

### 3. Install Required Packages

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets sentencepiece tqdm Pillow scikit-learn matplotlib
```

---

## 📚 Dataset Setup

The Flickr30k dataset is automatically downloaded from Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("nlphuji/flickr30k", split="test")
```

No manual download required.

---

## 🔤 Tokenizer Setup

### Option A: Use SentencePiece (default)

Generate captions.txt and train tokenizer:

```bash
python -c "from data.tokenizer import save_all_captions_to_txt, train_sentencepiece; save_all_captions_to_txt(); train_sentencepiece()"
```

This saves `spm.model` and `spm.vocab` in `data/tokenizer/`.

### Option B: Use Hugging Face Tokenizer

Switch to `HFTokenizerWrapper` in `data/tokenizer.py` and set model to `EleutherAI/pythia-160m`.

---

## 🏋️ Train the Model

To start training with ViT and SentencePiece:

```bash
python -m training.train
```

This will:

- Split Flickr30k into train/val/test (80/10/10)
- Train for 20 epochs
- Save model checkpoints to `checkpoints/`

To delete old checkpoints before retraining:

```bash
rm checkpoints/caption_model_epoch*.pt
```

---

## 🖼️ Generate Captions from Image

After training, run:

```bash
python -m inference.generate --image inference/sample.jpg --checkpoint checkpoints/caption_model_epoch10.pt
```

Expected output:

```
🖼️ Caption: A man in a red shirt is riding a bicycle.
```

---

## 🧠 Architecture Overview

- **Encoder**:
  - ViT patch-based image encoder (default)
  - Optional: use CLIP for pretrained image features
- **Decoder**:
  - Transformer-based decoder implemented from scratch
- **Tokenizers**:
  - SentencePiece (trained on Flickr30k)
  - Hugging Face model tokenizer

---

## 🗂️ Project Structure

```
image_captioning_transformer/
├── data/
│   ├── dataset.py
│   └── tokenizer.py
├── models/
│   ├── caption_model.py
│   ├── decoder_transformer.py
│   └── encoder_vit.py
├── training/
│   └── train.py
├── inference/
│   └── generate.py
├── checkpoints/
│   └── (saved model checkpoints)
└── README.md
```

---

## 🧹 Reset Local Code to Remote (if needed)

To reset your repo to match the latest version on GitHub:

```bash
git fetch origin
git reset --hard origin/main
```

---

## 📝 License

MIT License
