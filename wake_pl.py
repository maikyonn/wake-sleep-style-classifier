# train_generator_pl.py

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor #, StochasticWeightAveraging #, EarlyStopping
import socket
import time
import os
import datetime
import argparse
import json

# Import the LightningModule and DataModule
from PLGeneratorDM import GeneratorFinetuner
from datamodule import RealMidiDataModule
from tokenizer import MusicTokenizerWithStyle # Needed for DataModule

# Import ZClipLightningCallback for advanced gradient clipping
from zclip_lighting_callback import ZClipLightningCallback

# ==============================================================================
#                         CONFIGURATION (Moved Here)
# ==============================================================================

# --- Paths ---
BASE_INFERENCE_CHECKPOINT = "base-weights/vocab-mod-v2-ckpt-22-val_loss=0.27.ckpt"
BASE_GENERATIVE_WEIGHTS = "base-weights/vocab-mod-v2-ckpt-22-val_loss=0.27.ckpt"
BASE_GENERATIVE_CONFIG = "base-weights/config.json"

# --- Training Hyperparameters ---
NUM_EPOCHS = 30
LEARNING_RATE_GEN = 3e-4
BATCH_SIZE_PER_GPU = 1
GRAD_ACCUM_STEPS_GEN = 4 # Gradient accumulation factor

# --- Data Config ---
STYLE_VOCAB_SIZE = 4
MAX_SEQ_LEN = 3820
NUM_WORKERS = 32

# --- WandB Config ---
WANDB_PROJECT = 'aria-finetune-real-inferred-style-PL' # Specific project name
WANDB_ENTITY = None # Set to your entity

# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train generator with PyTorch Lightning')
    parser.add_argument('--data_dir', type=str, default="datasets/aria-midi-cycle1/data/",
                        help='Directory containing the MIDI dataset')
    parser.add_argument('--devices', type=int, default=4,
                        help='Number of GPUs to use')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='Number of nodes to use for distributed training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--file_limit', type=int, default=200000,
                        help='Maximum number of files to use from the dataset')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_GEN,
                        help='Learning rate for the generator')
    parser.add_argument('--batch_size_per_gpu', type=int, default=BATCH_SIZE_PER_GPU,
                        help='Batch size per GPU')
    parser.add_argument('--grad_accum_steps', type=int, default=GRAD_ACCUM_STEPS_GEN,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_seq_len', type=int, default=MAX_SEQ_LEN,
                        help='Maximum sequence length')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        help="WandB mode: 'online', 'offline', or 'disabled'")
    parser.add_argument('--style_vocab_size', type=int, default=STYLE_VOCAB_SIZE,
                        help='Style vocab size')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS,
                        help='Number of data loading workers')
    # NEW ------------------------------------------------------------------
    parser.add_argument('--strategy', type=str, default=None,
                        help="Lightning strategy, e.g. 'ddp', 'fsdp', 'auto'. "
                             "If omitted we keep the old heuristic.")
    args = parser.parse_args()

    # Compose a checkpoint directory name that reflects the training settings
    checkpoint_dir_name = (
        f"genfinetune_bs{args.batch_size_per_gpu * args.devices * args.num_nodes}"
        f"_lr{args.learning_rate:.0e}"
        f"_flim{args.file_limit}"
        f"_seq{args.max_seq_len}"
        f"_gacc{args.grad_accum_steps}"
        f"_{socket.gethostname()}"
        f"_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    args.checkpoint_dir = os.path.join("checkpoints", checkpoint_dir_name)

    # Compose a WandB run name that also reflects the settings
    args.wandb_run_name = (
        f"genfinetune_bs{args.batch_size_per_gpu * args.devices * args.num_nodes}"
        f"_lr{args.learning_rate:.0e}"
        f"_flim{args.file_limit}"
        f"_seq{args.max_seq_len}"
        f"_gacc{args.grad_accum_steps}"
        f"_{socket.gethostname()}"
        f"_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    args.wandb_mode = args.wandb_mode

    return args

def get_last_epoch_from_checkpoint(checkpoint_path):
    """Returns the last epoch from a PyTorch Lightning checkpoint file."""
    import torch
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # PyTorch Lightning saves epoch as 'epoch' in the checkpoint
        epoch = checkpoint.get("epoch", None)
        global_step = checkpoint.get("global_step", None)
        return epoch, global_step
    except Exception as e:
        print(f"Could not read epoch from checkpoint: {e}")
        return None, None

# --- Custom Progress Bar to show GPU memory usage ---
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar

class GPUMemoryProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        try:
            import torch
            if torch.cuda.is_available():
                # Show memory for all visible GPUs
                mems = []
                for i in range(torch.cuda.device_count()):
                    mem = torch.cuda.memory_allocated(i) / 1024**3
                    mems.append(f"{mem:.2f}GB")
                metrics["gpu_mem"] = "|".join(mems)
        except Exception as e:
            metrics["gpu_mem"] = "N/A"
        # Add learning rate to metrics if available
        try:
            # Try to get the current learning rate from the optimizer
            if hasattr(trainer, "optimizers") and trainer.optimizers:
                # If multiple optimizers, just show the first one
                optimizer = trainer.optimizers[0]
                lr = None
                for param_group in optimizer.param_groups:
                    lr = param_group.get("lr", None)
                    if lr is not None:
                        break
                if lr is not None:
                    metrics["lr"] = f"{lr:.2e}"
        except Exception as e:
            metrics["lr"] = "N/A"
        return metrics

def train():
    args = parse_args()
    pl.seed_everything(42, workers=True) # Set seed globally

    # --- Initialize Tokenizer (shared) ---
    tokenizer = MusicTokenizerWithStyle()

    # --- Setup DataModule ---
    # Total batch size across all GPUs
    total_batch_size = BATCH_SIZE_PER_GPU * args.devices * args.num_nodes
    print(f"Total batch size: {total_batch_size}")
    data_module = RealMidiDataModule(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE_PER_GPU,
        num_workers=NUM_WORKERS,
        cache_dir="./cache",
        dataset_seq_limit=args.file_limit,
        skip_long_sequences=True,
    )

    # --- Setup LightningModule ---
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = GeneratorFinetuner.load_from_checkpoint(
            args.checkpoint,
            generative_config_path=BASE_GENERATIVE_CONFIG,
            generative_weights_path=BASE_GENERATIVE_WEIGHTS,
            inference_checkpoint_path=BASE_INFERENCE_CHECKPOINT,
        )
        # Get last epoch and global step from checkpoint for logging/resume
        last_epoch, last_global_step = get_last_epoch_from_checkpoint(args.checkpoint)
        if last_epoch is not None:
            print(f"Resuming from epoch {last_epoch}, global step {last_global_step}")
        else:
            print("Could not determine last epoch from checkpoint.")
    else:
        print(f"Initializing model from base weights")
        model = GeneratorFinetuner(
            style_vocab_size=STYLE_VOCAB_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            learning_rate_gen=LEARNING_RATE_GEN,
            batch_size_per_gpu=BATCH_SIZE_PER_GPU,
            num_gpus=args.devices,
            grad_accum_steps_gen=GRAD_ACCUM_STEPS_GEN,
            generative_config_path=BASE_GENERATIVE_CONFIG,
            generative_weights_path=BASE_GENERATIVE_WEIGHTS,
            inference_checkpoint_path=BASE_INFERENCE_CHECKPOINT,
        )
        model._load_generative_model(load_weights=False)
        model._load_inference_model()
        model.inference_model.eval()
        last_epoch, last_global_step = None, None

    print(model)

    # --- Setup Logger ---
    if args.wandb_mode == "offline":
         os.makedirs("./wandb-local", exist_ok=True)

    # If resuming, try to keep the same WandB run name and resume the run
    wandb_logger_kwargs = dict(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        log_model=True, # Log checkpoints as artifacts
        save_dir="." if args.wandb_mode == "offline" else None
    )
    if args.checkpoint:
        # Resume the same WandB run if possible
        wandb_logger_kwargs["resume"] = "allow"
    wandb_logger = WandbLogger(**wandb_logger_kwargs)

    # --- Setup Callbacks ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f'gen-finetune-{{epoch:02d}}-{{step}}-{{val_loss:.4f}}-{timestamp}',
        monitor='val_loss', # Monitor validation loss
        mode='min',
        save_top_k=2,         # Save top 2 checkpoints based on monitored value
        save_last=True,       # Save the last checkpoint for resuming
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min") # Add if using validation

    # --- ZClipLightningCallback for advanced gradient clipping ---
    zclip_cb = ZClipLightningCallback(
        mode="zscore",
        alpha=0.90,
        z_thresh=2.0,
        clip_option="adaptive_scaling",
        max_grad_norm=1.0,
        clip_factor=0.5
    )

    # Add our custom progress bar callback
    gpu_mem_progress_bar = GPUMemoryProgressBar()

    # Remove Stochastic Weight Averaging callback

    callbacks = [checkpoint_callback, lr_monitor, zclip_cb, gpu_mem_progress_bar] # Add early_stop_callback if needed

    # --- Setup Trainer ---
    # Pick the strategy: cmd-line overrides the heuristic
    if args.strategy is not None:
        chosen_strategy = args.strategy
    else:
        chosen_strategy = (
            "fsdp" if args.devices > 1 or args.num_nodes > 1 else "auto"
        )

    trainer_kwargs = dict(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=chosen_strategy,
        max_epochs=NUM_EPOCHS,
        logger=wandb_logger,
        callbacks=callbacks,
        accumulate_grad_batches=GRAD_ACCUM_STEPS_GEN, # Handle gradient accumulation
        precision="bf16-mixed", # Optional: Use mixed precision for speed/memory
        log_every_n_steps=1 # How often to log metrics
        # Remove gradient_clip_val, since ZClipLightningCallback handles clipping
    )
    if args.checkpoint:
        # Resume from checkpoint, Lightning will restore epoch, optimizer, lr, etc.
        trainer_kwargs["resume_from_checkpoint"] = args.checkpoint

    # Implement the requested Trainer without SWA callback
    trainer = pl.Trainer(**trainer_kwargs)

    # --- Save Configuration to Text File ---
    config = {
        "timestamp": timestamp,
        "data_dir": args.data_dir,
        "devices": args.devices,
        "num_nodes": args.num_nodes,
        "checkpoint_dir": args.checkpoint_dir,
        "resume_checkpoint": args.checkpoint,
        "inference_checkpoint": BASE_INFERENCE_CHECKPOINT,
        "generative_weights": BASE_GENERATIVE_WEIGHTS,
        "generative_config": BASE_GENERATIVE_CONFIG,
        "num_epochs": NUM_EPOCHS,
        "learning_rate_gen": LEARNING_RATE_GEN,
        "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
        "grad_accum_steps_gen": GRAD_ACCUM_STEPS_GEN,
        "style_vocab_size": STYLE_VOCAB_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "file_limit": args.file_limit,
        "num_workers": NUM_WORKERS,
        "wandb_project": WANDB_PROJECT,
        "wandb_run_name": args.wandb_run_name,
        "wandb_mode": args.wandb_mode,
        "last_epoch": last_epoch,
        "last_global_step": last_global_step
    }
    
    config_path = os.path.join(args.checkpoint_dir, f"config_{timestamp}.txt")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to: {config_path}")

    # --- Start Training ---
    print("Starting PyTorch Lightning training...")
    trainer.fit(model, datamodule=data_module)
    print("Training finished.")

if __name__ == "__main__":
    train()