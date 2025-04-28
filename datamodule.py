# datamodule.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch
import os
from typing import Optional
from tokenizer import MusicTokenizer, IGNORE_LABEL_IDX
from dataset import SyntheticMidiStyleDataset, RealMidiDataset
import logging

logger = logging.getLogger(__name__)
class SyntheticMidiDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the SYNTHETIC MIDI+Style dataset."""
    def __init__(
        self,
        synthetic_data_dir: str,
        tokenizer: MusicTokenizer,
        batch_size: int = 32,
        max_len: int = 1024,
        val_split_ratio: float = 0.1,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        dataset_seq_limit: Optional[int] = None,
        cache_dir: Optional[str] = None,
        drop_last: bool = True,
        **kwargs
        ):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        
        # Store parameters
        self.data_dir = synthetic_data_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.val_split_ratio = val_split_ratio
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.dataset_seq_limit = dataset_seq_limit
        self.cache_dir = cache_dir
        self.drop_last = drop_last

        # Placeholders for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.full_dataset = None

        self.loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': True if self.num_workers > 0 else False,
            'prefetch_factor': 2 if self.num_workers > 0 else None,
            'drop_last': self.drop_last,
        }

    def prepare_data(self):
        """Prepares data. Called once per node in multi-GPU settings."""
        logger.info(f"--- Preparing Synthetic Data (Node: {int(os.environ.get('NODE_RANK', 0))}) ---")
        
        # Create the full dataset here so it gets cached once
        # This will be run on a single process, so other processes can read from cache
        SyntheticMidiStyleDataset(
            data_source=self.data_dir,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
            dataset_seq_limit=self.dataset_seq_limit,
            cache_dir=self.cache_dir
        )
        logger.info(f"Dataset created and cached at {self.cache_dir}")

    def setup(self, stage: str = None):
        """Sets up train/val datasets. Called on each GPU process."""
        logger.info(f"--- Setting up Synthetic DataModule (Stage: {stage}, Rank: {int(os.environ.get('LOCAL_RANK', 0))}) ---")

        if self.train_dataset is None or self.val_dataset is None:
            # Load the full dataset (will use cache if prepare_data has run)
            self.full_dataset = SyntheticMidiStyleDataset(
                data_source=self.data_dir,
                tokenizer=self.tokenizer,
                max_len=self.max_len,
                dataset_seq_limit=self.dataset_seq_limit,
                cache_dir=self.cache_dir
            )

            # Split into train and validation sets
            total_size = len(self.full_dataset)
            val_size = int(self.val_split_ratio * total_size)
            train_size = total_size - val_size

            if train_size == 0 or val_size == 0:
                raise ValueError(f"Train ({train_size}) or validation ({val_size}) split size is zero.")

            logger.info(f"Splitting into Train ({train_size}) and Validation ({val_size}) sets using seed {self.seed}...")
            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = random_split(
                self.full_dataset, [train_size, val_size], generator=generator
            )

            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            logger.info(f"Validation dataset size: {len(self.val_dataset)}")
        else:
            logger.info("Datasets already assigned. Skipping setup steps.")

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training set."""
        if not self.train_dataset:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self.loader_kwargs
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation set."""
        if not self.val_dataset:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self.loader_kwargs
        )

    def get_tokenizer(self) -> MusicTokenizer:
        """Returns the tokenizer instance."""
        return self.tokenizer

    def get_ignore_label_index(self) -> int:
        """Returns the ignore index used for padding style labels."""
        return IGNORE_LABEL_IDX

class RealMidiDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the REAL MIDI dataset."""
    def __init__(
        self,
        data_dir: str,
        tokenizer: MusicTokenizer,
        batch_size: int = 32,
        max_len: int = 1024,
        val_split_ratio: float = 0.1,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        dataset_seq_limit: Optional[int] = None,
        cache_dir: Optional[str] = None,
        drop_last: bool = True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        
        # Store parameters
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.val_split_ratio = val_split_ratio
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.dataset_seq_limit = dataset_seq_limit
        self.cache_dir = cache_dir
        self.drop_last = drop_last

        # Placeholders for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.full_dataset = None

        self.loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': True if self.num_workers > 0 else False,
            'prefetch_factor': 2 if self.num_workers > 0 else None,
            'drop_last': self.drop_last,
        }

    def prepare_data(self):
        """Prepares data. Called once per node in multi-GPU settings."""
        logger.info(f"--- Preparing Real Data (Node: {int(os.environ.get('NODE_RANK', 0))}) ---")
        
        # Create the full dataset here so it gets cached once
        # This will be run on a single process, so other processes can read from cache
        RealMidiDataset(
            data_source=self.data_dir,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
            dataset_seq_limit=self.dataset_seq_limit,
            cache_dir=self.cache_dir
        )
        logger.info(f"Dataset created and cached at {self.cache_dir}")

    def setup(self, stage: str = None):
        """Sets up train/val datasets. Called on each GPU process."""
        logger.info(f"--- Setting up Real DataModule (Stage: {stage}, Rank: {int(os.environ.get('LOCAL_RANK', 0))}) ---")

        if self.train_dataset is None or self.val_dataset is None:
            # Load the full dataset (will use cache if prepare_data has run)
            self.full_dataset = RealMidiDataset(
                data_source=self.data_dir,
                tokenizer=self.tokenizer,
                max_len=self.max_len,
                dataset_seq_limit=self.dataset_seq_limit,
                cache_dir=self.cache_dir
            )

            # Split into train and validation sets
            total_size = len(self.full_dataset)
            val_size = int(self.val_split_ratio * total_size)
            train_size = total_size - val_size

            if train_size == 0 or val_size == 0:
                raise ValueError(f"Train ({train_size}) or validation ({val_size}) split size is zero.")

            logger.info(f"Splitting into Train ({train_size}) and Validation ({val_size}) sets using seed {self.seed}...")
            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = random_split(
                self.full_dataset, [train_size, val_size], generator=generator
            )

            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            logger.info(f"Validation dataset size: {len(self.val_dataset)}")
        else:
            logger.info("Datasets already assigned. Skipping setup steps.")

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training set."""
        if not self.train_dataset:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self.loader_kwargs
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation set."""
        if not self.val_dataset:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self.loader_kwargs
        )

    def get_tokenizer(self) -> MusicTokenizer:
        """Returns the tokenizer instance."""
        return self.tokenizer


# --- Command Line Interface for Testing DataModule ---
def main():
    import torch
    from tokenizer import MusicTokenizer
    from dataset import IGNORE_LABEL_IDX
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)

    # Configuration parameters
    synthetic_data_dir = "datasets/10k-various-synth-train-set"
    real_data_dir = "datasets/aria-midi-v1-deduped-ext"
    cache_dir = "./cache"
    max_seq_len = 4096
    batch_size = 16
    dataset_seq_limit = 333
    num_workers = 128

    # Initialize tokenizer
    tokenizer = MusicTokenizer()
    
    logger.info("--- Testing SyntheticMidiDataModule ---")
    
    # Initialize Synthetic DataModule
    synthetic_data_module = SyntheticMidiDataModule(
        synthetic_data_dir=synthetic_data_dir,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_seq_len,
        num_workers=num_workers,
        pin_memory=True,
        cache_dir=cache_dir,
        dataset_seq_limit=dataset_seq_limit
    )
    
    # Prepare data and setup
    logger.info("Preparing synthetic data...")
    synthetic_data_module.prepare_data()
    
    logger.info("Setting up synthetic data module...")
    synthetic_data_module.setup(stage='fit')
    
    # Get train and validation dataloaders
    train_loader = synthetic_data_module.train_dataloader()
    val_loader = synthetic_data_module.val_dataloader()
    
    logger.info(f"Synthetic train dataset size: {len(synthetic_data_module.train_dataset)}")
    logger.info(f"Synthetic validation dataset size: {len(synthetic_data_module.val_dataset)}")
    
    # Test train batches
    logger.info("\n--- Testing Synthetic Train Batches ---")
    batch_count = 0
    for batch in train_loader:
        midi_ids, style_labels = batch['input_ids'], batch['style_label_indices']
        logger.info(f"Train Batch {batch_count+1}:")
        logger.info(f"  MIDI shape: {midi_ids.shape}, Labels shape: {style_labels.shape}")
        logger.info(f"  MIDI type: {type(midi_ids)}, dtype: {midi_ids.dtype}")
        logger.info(f"  Labels type: {type(style_labels)}, dtype: {style_labels.dtype}")
        logger.info(f"  MIDI tokens (first sample, first 20): {midi_ids[0, :20].tolist()}")
        logger.info(f"  Labels (first sample, first 20): {style_labels[0, :20].tolist()}")
        
        # Count label distribution (excluding padding)
        valid_labels = style_labels[style_labels != IGNORE_LABEL_IDX]
        if len(valid_labels) > 0:
            label_counts = torch.bincount(valid_labels)
            logger.info(f"  Label distribution: {label_counts.tolist()}")
        else:
            logger.info("  No valid labels in this batch (all padding)")
            
        batch_count += 1
        if batch_count >= 3:
            break
    
    # Test validation batches
    logger.info("\n--- Testing Synthetic Validation Batches ---")
    batch_count = 0
    for batch in val_loader:
        midi_ids, style_labels = batch['input_ids'], batch['style_label_indices']
        logger.info(f"Validation Batch {batch_count+1}:")
        logger.info(f"  MIDI shape: {midi_ids.shape}, Labels shape: {style_labels.shape}")
        logger.info(f"  MIDI type: {type(midi_ids)}, dtype: {midi_ids.dtype}")
        logger.info(f"  Labels type: {type(style_labels)}, dtype: {style_labels.dtype}")
        logger.info(f"  MIDI tokens (first sample, first 20): {midi_ids[0, :20].tolist()}")
        logger.info(f"  Labels (first sample, first 20): {style_labels[0, :20].tolist()}")
        
        # Count label distribution (excluding padding)
        valid_labels = style_labels[style_labels != IGNORE_LABEL_IDX]
        if len(valid_labels) > 0:
            label_counts = torch.bincount(valid_labels)
            logger.info(f"  Label distribution: {label_counts.tolist()}")
        else:
            logger.info("  No valid labels in this batch (all padding)")
            
        batch_count += 1
        if batch_count >= 3:
            break
    
    logger.info("--- Synthetic DataModule Test Complete ---")
    
    # Now test the RealMidiDataModule
    logger.info("\n--- Testing RealMidiDataModule ---")
    
    # Initialize Real DataModule
    real_data_module = RealMidiDataModule(
        data_dir=real_data_dir,
        tokenizer=tokenizer,
        max_len=max_seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_seq_limit=dataset_seq_limit,
        pin_memory=True,
        drop_last=True,
        cache_dir=cache_dir
    )
    
    # Prepare data and setup
    logger.info("Preparing real data...")
    real_data_module.prepare_data()
    
    logger.info("Setting up real data module...")
    real_data_module.setup(stage='fit')
    
    # Get train dataloader
    real_train_loader = real_data_module.train_dataloader()
    
    logger.info(f"Real train dataset size: {len(real_data_module.train_dataset)}")
    
    # Test real train batches
    logger.info("\n--- Testing Real Train Batches ---")
    batch_count = 0
    for batch in real_train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        logger.info(f"Real Train Batch {batch_count+1}:")
        logger.info(f"  Input IDs shape: {input_ids.shape}, Attention Mask shape: {attention_mask.shape}")
        logger.info(f"  Input IDs type: {type(input_ids)}, dtype: {input_ids.dtype}")
        logger.info(f"  Attention Mask type: {type(attention_mask)}, dtype: {attention_mask.dtype}")
        logger.info(f"  Input IDs (first sample, first 20): {input_ids[0, :20].tolist()}")
        logger.info(f"  Attention Mask (first sample, first 20): {attention_mask[0, :20].tolist()}")
        
        batch_count += 1
        if batch_count >= 3:
            break
    
    logger.info("--- Real DataModule Test Complete ---")

if __name__ == "__main__":
    main()