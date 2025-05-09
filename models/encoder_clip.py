# models/encoder_clip.py
import torch
import clip
from torch import nn

class CLIPResNetEncoder(nn.Module):
    def __init__(self, device='cuda', model_name='RN50'):
        super().__init__()
        self.model, _ = clip.load(model_name, device=device)
        self.encoder = self.model.visual
        self.embed_dim = self.encoder.output_dim

    def forward(self, images):
        return self.encoder(images)  # [B, D]
