# dataset.py

import os
import logging
import pickle
import hashlib
import random
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional, Dict, Union, Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from tokenizer import MusicTokenizerWithStyle
from aria.data.midi import MidiDict
from tokenizer import STYLE_LABEL_MAP, IGNORE_LABEL_IDX

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Core Tokenization & Processing Functions ---
from utils import (
    get_style_labels,
    process_synthetic_pair,
    process_real_midi,
    load_data_parallel
)

# --- Default Augmentation Settings ---
DEFAULT_PITCH_AUG = 5         # ±5 semitones
DEFAULT_TEMPO_AUG = 0.2       # ±20%
DEFAULT_VELOCITY_AUG = 10     # ±10

# --- Dataset Classes ---

class BaseDataset(Dataset):
    """Base class for MIDI datasets."""
    def __init__(self, 
                 data_source: Union[str, List[Dict]], 
                 tokenizer: MusicTokenizerWithStyle,
                 max_len: int = 4096,
                 mode: str = None,
                 dataset_seq_limit: Optional[int] = None,
                 cache_dir: Optional[str] = None,
                 shuffle: bool = True,
                 skip_long_sequences: bool = True,
                 data_aug: bool = True,
                 pitch_aug: int = DEFAULT_PITCH_AUG,
                 velocity_aug: int = DEFAULT_VELOCITY_AUG,
                 tempo_aug: float = DEFAULT_TEMPO_AUG,
                 mixup: bool = False):
        """
        Args:
            data_source: Either a directory path or pre-processed data list
            tokenizer: MusicTokenizerWithStyle instance
            max_len: Maximum sequence length
            mode: "synthetic" or "real" (required if data_source is a path)
            dataset_seq_limit: Maximum number of sequences to include
            cache_dir: Directory for caching processed data
            shuffle: Whether to shuffle data before processing
            skip_long_sequences: Whether to skip sequences longer than max_len
            data_aug: If True, apply all enabled data augmentations
            pitch_aug: Max pitch augmentation (0 disables)
            velocity_aug: Max velocity augmentation steps (0 disables)
            tempo_aug: Max tempo augmentation (0 disables)
            mixup: Enable note mixup for tempo augmentation
        """
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_id
        self.data_aug = data_aug
        self.pitch_aug = pitch_aug
        self.velocity_aug = velocity_aug
        self.tempo_aug = tempo_aug
        self.mixup = mixup

        # Setup augmentation functions if data_aug is True
        self.data_aug_fns = []
        if self.data_aug:
            if self.pitch_aug > 0:
                self.data_aug_fns.append(self._get_pitch_aug_fn())
            if self.velocity_aug > 0:
                self.data_aug_fns.append(self._get_velocity_aug_fn())
            if self.tempo_aug > 0.0:
                self.data_aug_fns.append(self._get_tempo_aug_fn())

        # Load data if path provided, otherwise use provided data
        if isinstance(data_source, str):
            if not mode:
                raise ValueError("Mode must be specified when data_source is a path")
            self.data = load_data_parallel(
                data_dir=data_source,
                tokenizer=tokenizer,
                max_len=max_len,
                mode=mode,
                dataset_seq_limit=dataset_seq_limit,
                cache_dir=cache_dir,
                shuffle=shuffle,
                skip_long_sequences=skip_long_sequences
            )
        else:
            self.data = data_source
            
        logger.info(f"Dataset initialized with {len(self.data)} sequences.")

    def __len__(self):
        return len(self.data)

    def _get_pitch_aug_fn(self):
        return self.tokenizer._tokenizer.export_pitch_aug(self.pitch_aug)

    def _get_velocity_aug_fn(self):
        return self.tokenizer._tokenizer.export_velocity_aug(self.velocity_aug)

    def _get_tempo_aug_fn(self):
        return self.tokenizer._tokenizer.export_tempo_aug(self.tempo_aug, self.mixup)

    def apply_augmentation(self, input_ids: List[int]) -> List[int]:
        """Apply all augmentation functions in sequence to the input_ids."""
        if not self.data_aug_fns:
            return input_ids
        tokens = self.tokenizer.decode(input_ids)
        for aug_fn in self.data_aug_fns:
            tokens = aug_fn(tokens)
        aug_input_ids = self.tokenizer.encode(tokens)
        return aug_input_ids

class SyntheticMidiStyleDataset(BaseDataset):
    """Dataset for synthetic (MIDI IDs, Style Label Indices) pairs."""
    def __init__(self, 
                 data_source: Union[str, List[Dict]], 
                 tokenizer: MusicTokenizerWithStyle,
                 max_len: int = 4096,
                 dataset_seq_limit: Optional[int] = None,
                 cache_dir: Optional[str] = None,
                 shuffle: bool = True,
                 skip_long_sequences: bool = True,
                 data_aug: bool = True,
                 pitch_aug: int = DEFAULT_PITCH_AUG,
                 velocity_aug: int = DEFAULT_VELOCITY_AUG,
                 tempo_aug: float = DEFAULT_TEMPO_AUG,
                 mixup: bool = False):
        super().__init__(
            data_source=data_source,
            tokenizer=tokenizer,
            max_len=max_len,
            mode="synthetic" if isinstance(data_source, str) else None,
            dataset_seq_limit=dataset_seq_limit,
            cache_dir=cache_dir,
            shuffle=shuffle,
            skip_long_sequences=skip_long_sequences,
            data_aug=data_aug,
            pitch_aug=pitch_aug,
            velocity_aug=velocity_aug,
            tempo_aug=tempo_aug,
            mixup=mixup
        )

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        midi_ids = item['input_ids']
        style_label_indices = item['style_label_indices']
        file_path = item['file_path']

        # Data augmentation (if any)
        midi_ids_aug = self.apply_augmentation(midi_ids)
        
        attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in midi_ids_aug]
        
        return {
            'input_ids': torch.tensor(midi_ids_aug, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'style_label_indices': torch.tensor(style_label_indices, dtype=torch.long),
            'file_path': file_path
        }

class RealMidiDataset(BaseDataset):
    """Dataset for real (unlabeled) MIDI sequences."""
    def __init__(self, 
                 data_source: Union[str, List[Dict]], 
                 tokenizer: MusicTokenizerWithStyle,
                 max_len: int = 4096,
                 dataset_seq_limit: Optional[int] = None,
                 cache_dir: Optional[str] = None,
                 shuffle: bool = True,
                 skip_long_sequences: bool = True,
                 data_aug: bool = True,
                 pitch_aug: int = DEFAULT_PITCH_AUG,
                 velocity_aug: int = DEFAULT_VELOCITY_AUG,
                 tempo_aug: float = DEFAULT_TEMPO_AUG,
                 mixup: bool = False):
        super().__init__(
            data_source=data_source,
            tokenizer=tokenizer,
            max_len=max_len,
            mode="real" if isinstance(data_source, str) else None,
            dataset_seq_limit=dataset_seq_limit,
            cache_dir=cache_dir,
            shuffle=shuffle,
            skip_long_sequences=skip_long_sequences,
            data_aug=data_aug,
            pitch_aug=pitch_aug,
            velocity_aug=velocity_aug,
            tempo_aug=tempo_aug,
            mixup=mixup
        )

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item_dict = self.data[idx]
        input_ids = item_dict['input_ids']
        file_path = item_dict['file_path']

        # Data augmentation (if any)
        input_ids_aug = self.apply_augmentation(input_ids)
        
        attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in input_ids_aug]
        
        return {
            'input_ids': torch.tensor(input_ids_aug, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'file_path': file_path
        }

# --- Command Line Interface for Testing ---
def main():
    import argparse
    import torch.utils.data as data
    
    parser = argparse.ArgumentParser(description="Test MIDI dataset loading")
    parser.add_argument("--synthetic_data_dir", type=str, default="datasets/10k-various-synth-train-set", help="Path to synthetic data parent dir")
    parser.add_argument("--real_data_dir", type=str, default="datasets/aria-midi-v1-deduped-ext", help="Path to real MIDI files")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory for testing")
    parser.add_argument("--max_seq_len", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing")
    parser.add_argument("--dataset_seq_limit", type=int, default=100, help="Limit sequences for testing")
    parser.add_argument("--data_aug", action="store_true", default=False, help="Enable data augmentation (default: True, see pitch/velocity/tempo aug)")
    parser.add_argument("--pitch_aug", type=int, default=DEFAULT_PITCH_AUG, help="Max pitch augmentation (0 disables, default: ±5)")
    parser.add_argument("--velocity_aug", type=int, default=DEFAULT_VELOCITY_AUG, help="Max velocity augmentation steps (0 disables, default: ±10)")
    parser.add_argument("--tempo_aug", type=float, default=DEFAULT_TEMPO_AUG, help="Max tempo augmentation (0 disables, default: ±0.2)")
    parser.add_argument("--mixup", action="store_true", help="Enable note mixup for tempo augmentation")
    args = parser.parse_args()

    if "path/to/" in args.synthetic_data_dir or "path/to/" in args.real_data_dir:
         print("Please update placeholder paths for --synthetic_data_dir and --real_data_dir")
         return

    logger.info("--- Testing Dataset Loading ---")
    tokenizer = MusicTokenizerWithStyle()

    logger.info("Loading synthetic data...")
    synthetic_dataset = SyntheticMidiStyleDataset(
        data_source=args.synthetic_data_dir,
        tokenizer=tokenizer,
        max_len=args.max_seq_len,
        dataset_seq_limit=args.dataset_seq_limit,
        cache_dir=args.cache_dir,
        data_aug=True if not args.data_aug else args.data_aug,
        pitch_aug=args.pitch_aug,
        velocity_aug=args.velocity_aug,
        tempo_aug=args.tempo_aug,
        mixup=args.mixup
    )
    
    if len(synthetic_dataset) > 0:
        synthetic_loader = data.DataLoader(synthetic_dataset, batch_size=args.batch_size)
        logger.info("Testing synthetic batch...")
        try:
            batch = next(iter(synthetic_loader))
            logger.info(f"Synthetic Batch - Input IDs shape: {batch['input_ids'].shape}, Labels shape: {batch['style_label_indices'].shape}")
            logger.info(f"Example labels: {batch['style_label_indices'][0, :20]}...")
        except StopIteration:
            logger.warning("Synthetic loader empty.")
    else:
        logger.warning("No synthetic data loaded.")

    logger.info("Loading real data...")
    real_dataset = RealMidiDataset(
        data_source=args.real_data_dir,
        tokenizer=tokenizer,
        max_len=args.max_seq_len,
        dataset_seq_limit=args.dataset_seq_limit,
        cache_dir=args.cache_dir,
        shuffle=False,
        skip_long_sequences=True,
        data_aug=True if not args.data_aug else args.data_aug,
        pitch_aug=args.pitch_aug,
        velocity_aug=args.velocity_aug,
        tempo_aug=args.tempo_aug,
        mixup=args.mixup
    )
    
    if len(real_dataset) > 0:
        real_loader = data.DataLoader(real_dataset, batch_size=args.batch_size)
        logger.info("Testing real batch...")
        try:
            batch = next(iter(real_loader))
            logger.info(f"Real Batch - Input IDs shape: {batch['input_ids'].shape}, Mask shape: {batch['attention_mask'].shape}")
        except StopIteration:
            logger.warning("Real loader empty.")
    else:
        logger.warning("No real data loaded.")

    logger.info("--- Dataset Test Complete ---")

if __name__ == "__main__":
    main()