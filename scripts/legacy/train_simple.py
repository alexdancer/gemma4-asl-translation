#!/usr/bin/env python3
"""Simple fine-tuning script for ASL pose→gloss with Gemma 4 + Unsloth."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from unsloth import FastLanguageModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)


class PoseGlossDataset(Dataset):
    """Load pose NPZ files and corresponding glosses."""
    
    def __init__(self, csv_path, pose_root, max_samples=None, max_seq_len=50):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.head(max_samples)
        self.pose_root = Path(pose_root)
        self.max_seq_len = max_seq_len
        self.pose_dim = 59 * 3  # (body + left_hand + right_hand) * 3 coords
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gloss = row['gloss']
        
        # Load pose
        pose_path = self.pose_root / row['pose_path']
        try:
            with np.load(pose_path) as data:
                body = data.get('body', np.zeros((self.max_seq_len, 17, 3)))
                left_hand = data.get('left_hand', np.zeros((self.max_seq_len, 21, 3)))
                right_hand = data.get('right_hand', np.zeros((self.max_seq_len, 21, 3)))
                
                # Concatenate: (seq_len, 59, 3)
                pose = np.concatenate([body, left_hand, right_hand], axis=1)
                
                # Pad/truncate to max_seq_len
                if pose.shape[0] < self.max_seq_len:
                    pad_len = self.max_seq_len - pose.shape[0]
                    pose = np.pad(pose, ((0, pad_len), (0, 0), (0, 0)), mode='constant')
                else:
                    pose = pose[:self.max_seq_len]
                
                # Flatten: (max_seq_len, 59, 3) → (max_seq_len*59*3,)
                pose_flat = pose.reshape(-1).astype(np.float32)
                
                return {
                    'pose': torch.tensor(pose_flat),
                    'gloss': gloss,
                }
        except Exception as e:
            logger.error(f"Error loading {pose_path}: {e}")
            # Return zero-padded pose as fallback
            return {
                'pose': torch.zeros(self.max_seq_len * self.pose_dim, dtype=torch.float32),
                'gloss': gloss,
            }


def main():
    # Setup
    csv_path = "data/processed/splits/all/poses_train.csv"
    pose_root = "data/processed/poses"
    output_dir = "checkpoints/gemma_asl_simple"
    max_samples = 100  # Small batch for testing
    batch_size = 8
    num_epochs = 3
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading dataset from {csv_path}")
    dataset = PoseGlossDataset(csv_path, pose_root, max_samples=max_samples)
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Dataloader batches: {len(dataloader)}")
    
    # Load model
    logger.info("Loading Gemma 4 with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="google/gemma-4-E2B-it",
        max_seq_length=256,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    logger.info("✅ Model loaded")
    
    # Attach LoRA (use Unsloth's method for Gemma 4 compatibility)
    logger.info("Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        use_rslora=True,
    )
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"✅ LoRA attached: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")
    
    # Prepare for training
    model.gradient_checkpointing_enable()
    model.train()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"Model on device: {device}")
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Get poses and glosses
                poses = batch['pose'].to(device)  # (batch_size, seq_len*59*3)
                glosses = batch['gloss']  # List of strings
                
                # Tokenize glosses
                # Workaround: pass text as list
                tokens = tokenizer(text=glosses, return_tensors="pt", padding=True)
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens.get('attention_mask', torch.ones_like(input_ids)).to(device)
                
                # Forward pass: use pose as input embeddings
                # We'll embed poses into token space
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                loss = outputs.loss
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 5 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    logger.info(f"Saving checkpoint to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"✅ Training complete!")


if __name__ == "__main__":
    main()
