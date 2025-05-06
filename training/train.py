# training/train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from datasets import load_dataset
from data.dataset import Flickr30kDataset
from models.caption_model import CaptioningModel

def train(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)                # [B, 3, H, W]
        caption_input = batch["caption_input"].to(device) # [B, T]
        caption_label = batch["caption_label"].to(device) # [B, T]

        optimizer.zero_grad()
        logits = model(images, caption_input)             # [B, T, vocab_size]
        
        loss = criterion(logits.view(-1, logits.size(-1)), caption_label.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            caption_input = batch["caption_input"].to(device)
            caption_label = batch["caption_label"].to(device)

            logits = model(images, caption_input)
            loss = criterion(logits.view(-1, logits.size(-1)), caption_label.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load original dataset (only split available is 'test')
    full_data = load_dataset("nlphuji/flickr30k", split="test")

    # Split into train (80%), val (10%), test (10%)
# 80% train, 10% val, 10% test
    train_size = int(0.8 * len(raw_dataset))
    val_size = int(0.1 * len(raw_dataset))
    test_size = len(raw_dataset) - train_size - val_size

    train_split, val_split, test_split = random_split(
        raw_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Wrap into Flickr30kDataset
    train_dataset = Flickr30kDataset(dataset_split=train_split)
    val_dataset = Flickr30kDataset(dataset_split=val_split)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Initialize model
    vocab_size = train_dataset.tokenizer.vocab_size
    model = CaptioningModel(vocab_size=vocab_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    pad_id = train_dataset.tokenizer.pad_id
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    for epoch in range(1, 21):
        print(f"\nEpoch {epoch}")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/caption_model_epoch{epoch}.pt")

if __name__ == "__main__":
    main()
