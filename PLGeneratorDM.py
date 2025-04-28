# generator_finetuner_pl.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os
from pathlib import Path
import logging # Keep logging for setup messages

# Assuming these imports exist and work as expected
from tokenizer import MusicTokenizerWithStyle # Assuming this is your tokenizer class
from aria_generative.config import load_model_config
from aria_generative.model import ModelConfig, TransformerLM
from aria_generative.utils import _load_weight
from MidiClassifierModel import MidiClassifier # Make sure this is compatible or adaptable
from utils import get_batch_prompts_from_midi_style_ids
from transformers import get_cosine_schedule_with_warmup
# Setup basic logging for initialization messages
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
class GeneratorFinetuner(pl.LightningModule):
    def __init__(
        self,
        style_vocab_size: int,
        max_seq_len: int, # Added for potential validation/generation later
        # Training HParams
        learning_rate_gen: float = 1e-5,
        batch_size_per_gpu: int = 1, # For info, effective batch size handled by Trainer
        num_gpus: int = 4, # For info
        grad_accum_steps_gen: int = 4, # Used by Trainer's accumulate_grad_batches
        # NEW: Add these for checkpoint compatibility
        generative_config_path: str = None,
        generative_weights_path: str = None,
        inference_checkpoint_path: str = None,
        **kwargs # Allow passthrough of other args if needed
    ):
        super().__init__()
        # Store hyperparameters automatically (access via self.hparams)
        # Note: paths might not be ideal hparams if they change, but convenient here
        self.save_hyperparameters()
        # Store paths for later use
        self.generative_config_path = generative_config_path
        self.generative_weights_path = generative_weights_path
        self.inference_checkpoint_path = inference_checkpoint_path

        logger.info("Initializing Tokenizer...")
        self.tokenizer = MusicTokenizerWithStyle()
        self.pad_token_id = self.tokenizer.pad_id
        self.vocab_size = self.tokenizer.vocab_size
        logger.info(f"Tokenizer vocab size: {self.tokenizer.vocab_size}, PAD ID: {self.pad_token_id}")

        # Loss function (can also use F.cross_entropy directly in step)
        # Using ignore_index with pad_token_id is crucial
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='none')


    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
        """
        Custom checkpoint loading method to handle both generative and inference models.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            map_location: Optional device mapping
            **kwargs: Additional arguments to pass to the model constructor
        
        Returns:
            Initialized model with loaded weights
        """
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint to CPU first for better memory management
        checkpoint = torch.load(checkpoint_path, map_location=map_location or 'cpu')
        
        # Get hyperparameters from checkpoint
        hparams = checkpoint.get('hyper_parameters', {})
        
        # Update with any provided kwargs
        hparams.update(kwargs)
        
        # Initialize model with hyperparameters
        model = cls(**hparams)
        
        # Initialize both models before loading weights
        logger.info("Initializing Generative Model (Aria)...")
        model._load_generative_model()
        
        logger.info("Initializing Inference Model (Frozen)...")
        model._load_inference_model()
        # Freeze the inference model immediately
        model.inference_model.eval()
        for param in model.inference_model.parameters():
            param.requires_grad = False
        logger.info("Inference model set to eval() and frozen.")
        
        # Get state dict from checkpoint
        state_dict = checkpoint['state_dict']
        
        # Create new state dict with proper key mapping
        new_state_dict = {}
        
        # Process state dict to handle the model prefixes
        for key, value in state_dict.items():
            # Handle generative model keys
            if key.startswith('generative_model.'):
                new_key = key
                new_state_dict[new_key] = value
            # Handle inference model keys
            elif key.startswith('inference_model.'):
                new_key = key
                new_state_dict[new_key] = value
            # Handle any other keys (like optimizer states, etc.)
            else:
                new_state_dict[key] = value
        
        # Load the processed state dict
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        
        logger.info("Checkpoint loaded successfully")
        return model
    
    def _load_generative_model(self, load_weights=True):
        """Loads the generative model config and weights. If weights are provided and load_weights is True, loads them."""
        try:
            model_config_dict = load_model_config(self.generative_config_path)
            model_config = ModelConfig(**model_config_dict)
            model_config.set_vocab_size(self.tokenizer.vocab_size)
            model_config.class_size = self.hparams.style_vocab_size
            model_config.grad_checkpoint = model_config_dict.get("grad_checkpoint", False)

            self.generative_model = TransformerLM(model_config).to('cpu')
            
            # Only load weights if load_weights is True and weights are provided
            if load_weights and self.generative_weights_path:
                logger.info(f"Loading generative weights from: {self.generative_weights_path}")
                checkpoint = torch.load(self.generative_weights_path, map_location='cpu')
                load_result = self.generative_model.load_state_dict(checkpoint, strict=False)
                logger.info(f"Generative model weight loading result: {load_result}")
            elif not load_weights:
                logger.info("load_weights is False. Using randomly initialized weights.")
            else:
                logger.info("No weights provided. Using randomly initialized weights.")
                
        except Exception as e:
            logger.error(f"Error initializing generative model: {e}", exc_info=True)
            raise

    def _load_inference_model(self):
        """Loads the inference model from checkpoint."""
        try:
            map_location = 'cpu' # Load to CPU
            logger.info(f"Loading inference checkpoint from: {self.inference_checkpoint_path}")
            self.inference_model = MidiClassifier.load_from_checkpoint(
                self.inference_checkpoint_path,
                map_location=map_location
            )
            # Don't move to device here, Lightning handles it
        except Exception as e:
            logger.error(f"Error initializing inference model: {e}", exc_info=True)
            raise

    def forward(self, input_tokens):
        """Defines the forward pass for the generative model."""
        return self.generative_model(input_tokens)

    def training_step(self, batch, batch_idx):
        """Performs the Wake Phase 2 training logic."""
        # 1. Get Data
        X_real = batch['input_ids'].to(self.device)

        # 2. Get inferred latent labels (X_predictions / Z_inf_real) from FROZEN Inference Model
        with torch.no_grad():  # Ensure no gradients are computed for inference model
            self.inference_model.eval()

            # Pass necessary inputs to inference model (check its forward signature)
            attention_mask = torch.ones_like(X_real)
            inf_logits_real = self.inference_model(X_real, attention_mask)
            Z_inf_real = torch.argmax(inf_logits_real, dim=-1)

            # Use the batch prompts function to create prompts from X_real and Z_inf_real
            # Set max_prompt_length=256
            batch_prompts, batch_prompt_tokens = get_batch_prompts_from_midi_style_ids(
                input_tokens_batch=X_real,
                style_ids_batch=Z_inf_real,
                tokenizer=self.tokenizer,
                max_prompt_length=256
            )

        # --- Build full sequences (prompt + X_real + EOS), pad, and build input/target ---
        batch_full_sequences = []
        prompt_lens = []
        eos_id = self.tokenizer.eos_id

        for i, prompt in enumerate(batch_prompts):
            prompt_tensor = torch.tensor(prompt, device=self.device)
            prompt_len = len(prompt_tensor)
            prompt_lens.append(prompt_len)
            # Only non-padding tokens from X_real
            x_real_non_padding = X_real[i][X_real[i] != self.pad_token_id]
            full_sequence = torch.cat([prompt_tensor, x_real_non_padding])
            # Ensure EOS at the end
            if full_sequence[-1] != eos_id:
                full_sequence = torch.cat([full_sequence, torch.tensor([eos_id], device=self.device)])
            batch_full_sequences.append(full_sequence)

        # Find max length for padding (after EOS added)
        max_len = max(seq.size(0) for seq in batch_full_sequences)
        padded_sequences = []
        for seq in batch_full_sequences:
            padding_needed = max_len - seq.size(0)
            if padding_needed > 0:
                pad = torch.full((padding_needed,), self.pad_token_id, device=self.device, dtype=seq.dtype)
                seq = torch.cat([seq, pad])
            padded_sequences.append(seq)

        # Stack to batch
        full_sequences = torch.stack(padded_sequences, dim=0)  # (B, max_len)

        # Prepare input/target for LM: input = all but last, target = all but first
        input_tokens_for_G = full_sequences[:, :-1]  # (B, max_len-1)
        target_tokens_for_loss = full_sequences[:, 1:]  # (B, max_len-1)

        # Run the generator model (self calls the forward method) - requires grads
        logits_gen_real = self(input_tokens_for_G)
        # logits_gen_real shape: (B, seq_len, vocab_size)

        # Flatten for loss computation
        logits_flat = logits_gen_real.reshape(-1, self.vocab_size)
        targets_flat = target_tokens_for_loss.reshape(-1)

        # Compute per-token loss (no reduction, ignore padding)
        per_token_loss_flat = self.criterion(logits_flat, targets_flat)
        # If ignore_index is hit, per_token_loss_flat will be 0 for those positions

        # Reshape back to (B, seq_len)
        per_token_loss = per_token_loss_flat.view(target_tokens_for_loss.shape)

        # Mask: only compute loss for tokens after the prompt (in the target, that's prompt_len-1 and onward)
        mask = torch.zeros_like(target_tokens_for_loss, dtype=torch.float, device=self.device)
        valid_token_count = 0
        for i, prompt_len in enumerate(prompt_lens):
            # In the target, the first (prompt_len-1) tokens are prompt (because of shift)
            # So, mask from (prompt_len-1) onward
            if prompt_len - 1 < mask.size(1):
                mask[i, prompt_len-1:] = 1.0
                valid_token_count += mask[i, prompt_len-1:].sum()
            # If prompt_len-1 >= seq_len, nothing is masked in (shouldn't happen)

        # Apply mask
        masked_loss = per_token_loss * mask

        # Compute final loss
        if valid_token_count > 0:
            loss = masked_loss.sum() / valid_token_count
        else:
            loss = torch.tensor(0.0, device=self.device)
            if self.trainer.global_rank == 0:
                logger.warning("Valid token count is zero in training step, setting loss to 0.")

        # Log the loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('lr_gen', self.hparams.learning_rate_gen, on_step=False, on_epoch=True, logger=True)

        return loss  # Return the loss tensor

    def configure_optimizers(self):
        """Configure the optimizer for the generative model."""
        optimizer = torch.optim.AdamW(
            self.generative_model.parameters(), # Only optimize generator
            lr=self.hparams.learning_rate_gen
        )
        
        # Get total steps for warmup calculation
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(10000, int(0.1 * total_steps))  # 10% of total steps or max 10k
        
        # Create warmup scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "train_loss"
            }
        }

    def validation_step(self, batch, batch_idx):
        """Validation step for the generative model. Mirrors training_step logic but without gradients."""
        with torch.no_grad():
            self.inference_model.eval()

            # 1. Get Data
            X_real = batch['input_ids'].to(self.device)
            # 2. Get inferred latent labels (Z_inf_real) from FROZEN Inference Model
            attention_mask = torch.ones_like(X_real)
            inf_logits_real = self.inference_model(X_real, attention_mask)
            Z_inf_real = torch.argmax(inf_logits_real, dim=-1)

            # 3. Use the batch prompts function to create prompts from X_real and Z_inf_real
            batch_prompts, batch_prompt_tokens = get_batch_prompts_from_midi_style_ids(
                input_tokens_batch=X_real,
                style_ids_batch=Z_inf_real,
                tokenizer=self.tokenizer,
                max_prompt_length=256
            )

            # 4. Convert batch_prompts to tensor and append X_real tokens, add EOS if needed
            batch_prompts_tensor = []
            prompt_lens = []
            eos_id = self.tokenizer.eos_id
            for i, prompt in enumerate(batch_prompts):
                prompt_tensor = torch.tensor(prompt, device=self.device)
                prompt_len = len(prompt_tensor)
                prompt_lens.append(prompt_len)
                x_real_non_padding = X_real[i][X_real[i] != self.pad_token_id]
                full_sequence = torch.cat([prompt_tensor, x_real_non_padding])
                # Add EOS token if not present at the end
                if full_sequence[-1] != eos_id:
                    full_sequence = torch.cat([full_sequence, torch.tensor([eos_id], device=self.device)])
                batch_prompts_tensor.append(full_sequence)

            # 5. Pad sequences to the same length
            max_len = max(len(seq) for seq in batch_prompts_tensor)
            padded_sequences = []
            for seq in batch_prompts_tensor:
                padding_needed = max_len - len(seq)
                padded_seq = torch.cat([seq, torch.full((padding_needed,), self.pad_token_id, device=self.device)])
                padded_sequences.append(padded_seq)

            # 6. Stack to create a batch tensor
            full_sequences = torch.stack(padded_sequences)

            # 7. Create input and target sequences for the generative model
            input_tokens_for_G = full_sequences[:, :-1]  # all tokens except the last one
            target_tokens_for_loss = full_sequences[:, 1:]  # all tokens except the first one

            # 8. Run the generator model
            logits_gen_real = self(input_tokens_for_G)  # (B, sequence_length, vocab_size)

            # --- Apply the same fixes from train_step to val_step ---

            # Flatten for loss computation
            logits_flat = logits_gen_real.reshape(-1, self.vocab_size)
            targets_flat = target_tokens_for_loss.reshape(-1)

            # Compute per-token loss (no reduction, ignore padding)
            per_token_loss_flat = self.criterion(logits_flat, targets_flat)
            # If ignore_index is hit, per_token_loss_flat will be 0 for those positions

            # Reshape back to (B, seq_len)
            per_token_loss = per_token_loss_flat.view(target_tokens_for_loss.shape)

            # Mask: only compute loss for tokens after the prompt (in the target, that's prompt_len-1 and onward)
            mask = torch.zeros_like(target_tokens_for_loss, dtype=torch.float, device=self.device)
            valid_token_count = 0
            for i, prompt_len in enumerate(prompt_lens):
                # In the target, the first (prompt_len-1) tokens are prompt (because of shift)
                # So, mask from (prompt_len-1) onward
                if prompt_len - 1 < mask.size(1):
                    mask[i, prompt_len-1:] = 1.0
                    valid_token_count += mask[i, prompt_len-1:].sum()
                # If prompt_len-1 >= seq_len, nothing is masked in (shouldn't happen)

            # Apply mask
            masked_loss = per_token_loss * mask

            # Compute final loss
            if valid_token_count > 0:
                val_loss = masked_loss.sum() / valid_token_count
            else:
                val_loss = torch.tensor(0.0, device=self.device)
                if self.trainer.global_rank == 0:
                    logger.warning("Valid token count is zero in validation step, setting loss to 0.")

            # Log the loss
            self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            return val_loss
        
    def generate_from_prompt(self, prompt, max_gen_tokens=512, beam_width=5, length_penalty=1.0):
        """
        Generate a sequence from a prompt using the generator model with beam search.
        Args:
            prompt (list[int]): List of token ids to start generation.
            max_gen_tokens (int): Maximum number of tokens to generate (not including prompt).
            beam_width (int): Number of beams to use for beam search.
            length_penalty (float): Exponent for length penalty (higher = shorter sequences).
        Returns:
            List[int]: The full generated sequence (prompt + generated tokens) for the best beam.
        """
        import heapq
        from tqdm import tqdm
        import torch

        self.eval()
        with torch.no_grad():
            # Each beam is a tuple: (score, [token_ids])
            beams = [(0.0, list(prompt))]
            prompt_len = len(prompt)
            completed_beams = []

            for _ in tqdm(range(max_gen_tokens), desc="Generating tokens (beam search)"):
                new_beams = []
                for score, seq in beams:
                    # If already ended with EOS, keep as completed
                    if hasattr(self, "eos_token_id") and seq[-1] == self.eos_token_id:
                        completed_beams.append((score, seq))
                        continue

                    input_tensor = torch.tensor(seq, device=self.device).unsqueeze(0).to("cuda")  # [1, seq_len]
                    logits = self(input_tensor)  # [1, seq_len, vocab_size]
                    next_token_logits = logits[0, -1, :]  # [vocab_size]
                    log_probs = torch.log_softmax(next_token_logits, dim=-1)

                    # Get top beam_width next tokens
                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                    for log_prob, idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                        new_seq = seq + [idx]
                        # Apply length penalty (optional)
                        new_score = score + log_prob / (len(new_seq) ** length_penalty)
                        new_beams.append((new_score, new_seq))

                # Keep only top beam_width beams
                if not new_beams:
                    break  # All beams completed
                # Use heapq.nlargest for efficiency
                beams = heapq.nlargest(beam_width, new_beams, key=lambda x: x[0])

            # Add any remaining beams to completed_beams
            completed_beams.extend(beams)
            # Sort completed beams by score
            completed_beams = sorted(completed_beams, key=lambda x: x[0], reverse=True)
            best_score, best_seq = completed_beams[0]

            # Remove any padding tokens from the output
            output = [t for t in best_seq if t != self.pad_token_id]
            # Also return output_no_prompt, which is output without the prompt at the start
            output_no_prompt = [t for t in best_seq[prompt_len:] if t != self.pad_token_id]
            return output, output_no_prompt
            

    # def test_step(self, batch, batch_idx):
    #     # ...
    #     # self.log('test_loss', loss, on_epoch=True, sync_dist=True)
    #     pass




if __name__ == "__main__":
    import argparse
    import os
    from tokenizer import MusicTokenizerWithStyle
    from datamodule import RealMidiDataModule
    from torch.utils.data import DataLoader
    
    # Configuration parameters
    REAL_DATA_DIR = "datasets/aria-midi-v1-deduped-ext/data/aa"
    CHECKPOINT_DIR = "ws_checkpoints_slurm_1024" # Checkpoint directory on SHARED FILESYSTEM
    INFERENCE_CHECKPOINT = "checkpoints_supervised/20250410_164054_len4096_lr3e-05_drop0.2_modelbert/vocab-mod-v1-ckpt-22-val_loss=0.27.ckpt"
    GENERATIVE_WEIGHTS = "aria_generative/TokenizerStyle-medium-e75.safetensors"
    GENERATIVE_CONFIG = "aria_generative/config.json"
    style_vocab_size = 4
    max_seq_len = 4096
    batch_size = 16
    dataset_seq_limit = 333
    num_workers = 128

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model from inference checkpoint: {INFERENCE_CHECKPOINT}")
    model = GeneratorFinetuner(
        generative_config_path=GENERATIVE_CONFIG,
        generative_weights_path=GENERATIVE_WEIGHTS,
        inference_checkpoint_path=INFERENCE_CHECKPOINT,
        style_vocab_size=style_vocab_size,
        max_seq_len=max_seq_len
    )
    model = model.to(device)
    model.eval()
    print(model)
    # Initialize tokenizer
    tokenizer = MusicTokenizerWithStyle()
    
    # Create a RealMidiDataModule
    print(f"Loading data from: {REAL_DATA_DIR}")
    data_module = RealMidiDataModule(
        data_dir=REAL_DATA_DIR,
        tokenizer=tokenizer,
        max_len=max_seq_len,
        batch_size=BATCH_SIZE_PER_GPU,
        num_workers=num_workers,
        dataset_seq_limit=dataset_seq_limit,
        pin_memory=True,
        drop_last=True,
        skip_long_sequences=True,
        cache_dir="./cache"
    )
    
    
    # Setup the data module
    data_module.setup()
    
    # Get a batch from the train dataloader
    print("Fetching a batch from the train dataloader...")
    train_dataloader = data_module.train_dataloader()
    batch = next(iter(train_dataloader))
    
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        loss = model.training_step(batch, batch_idx=0)
    
    print(f"Output loss: {loss}")
    
    print("Test completed successfully!")