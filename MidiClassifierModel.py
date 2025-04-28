import os
import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger
import wandb
import datetime
import logging
import shutil
import re

import torch.cuda as cuda
from lightning.pytorch.plugins.environments import SLURMEnvironment
from tqdm.auto import tqdm  # Import tqdm properly
from dataset import load_data_parallel
# --- Use the NEW DataModule ---
from datamodule import SyntheticMidiStyleDataset
# --- Use the NEW Tokenizer ---
from tokenizer import MusicTokenizerWithStyle # For type hints if needed later
# -----------------------------
import numpy as np

from transformers import BertModel, BertConfig
from transformers.optimization import get_cosine_schedule_with_warmup
from torchmetrics import Accuracy
from utils import filter_significant_styles, extract_style_change_timestamps, condense_style_sequence

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- WandB Setup --- (remains the same)
os.environ["WANDB_CACHE_DIR"] = "./wandb-cache"
os.environ["WANDB_DIR"] = "./wandb-local"
os.makedirs("./wandb-cache", exist_ok=True); os.makedirs("./wandb-local", exist_ok=True)
torch.set_float32_matmul_precision('medium')        


class MidiClassifier(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int, # Pass vocab size explicitly
        n_classes: int = 4,
        lr: float = 2e-5,
        max_length: int = 8192,
        dropout_rate: float = 0.3,
        model_type: str = "bert", # Only "bert" is supported
        pad_id: int = 0, # Pass pad_id explicitly
        checkpoint_path: str = None # Path to load checkpoint from
    ):
        super().__init__()
        # Store essential hparams manually if needed, or use save_hyperparameters carefully
        self.save_hyperparameters("vocab_size", "n_classes", "lr", "max_length", "dropout_rate", "model_type", "pad_id")

        self.pad_id = pad_id # Store pad_id
        self.model_type = "bert"  # Always use bert
        self.lr = lr
        self.idx_to_style = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        self.STYLE_LABEL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
        self.IGNORE_LABEL_IDX = -100 # Padding/WithStyleIgnore index for loss function
        self.tokenizer = MusicTokenizerWithStyle()
        self.hparams.vocab_size = self.tokenizer.vocab_size
        self.vocab_size = self.tokenizer.vocab_size



        config = BertConfig(
            vocab_size=self.hparams.vocab_size, # Use passed vocab_size
            hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
            intermediate_size=4096, max_position_embeddings=max_length,
            hidden_dropout_prob=dropout_rate, attention_probs_dropout_prob=dropout_rate
        )
        self.model = BertModel(config)
        hidden_size = config.hidden_size

        self.model.train()
        for param in self.model.parameters(): param.requires_grad = True

        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size, self.hparams.n_classes)
        )
        self.dropout2 = torch.nn.Dropout(dropout_rate) # Added dropout before final output

        # Metrics (Use hparams.n_classes)
        self.train_accuracy = Accuracy(task='multiclass', num_classes=self.hparams.n_classes, ignore_index=-100)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=self.hparams.n_classes, ignore_index=-100)
        logger.info(f"MidiClassifier initialized: vocab={self.hparams.vocab_size}, classes={self.hparams.n_classes}")

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading model state dict from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
            logger.info("Checkpoint loaded successfully")
        elif checkpoint_path:
            logger.warning(f"Checkpoint path provided but file not found: {checkpoint_path}")


    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        hidden_states = self.dropout1(hidden_states)
        logits = self.classifier(hidden_states)
        logits = self.dropout2(logits) # Apply final dropout
        return logits


    
    
    def generate_predictions_from_dataset(self, dataset, mode, max_length=4096, results_dir="./test_results", max_files=None, batch_size=16):
        """Analyze MIDI files using the provided dataset and generate style predictions."""
        # Create results directory if specified
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            # Create a subdirectory for MIDI files
            midi_output_dir = os.path.join(results_dir, 'midi_files')
            os.makedirs(midi_output_dir, exist_ok=True)
        
        # Create a DataLoader for batched inference
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=32,
            pin_memory=True
        )
        
        # For CSV output - one row per file
        csv_data = []
        # Print sample batch information for debugging
        print("Sample batch structure:")
        sample_batch = next(iter(dataloader))
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
        
        # Print a few token IDs from the first sequence
        if 'input_ids' in sample_batch:
            print(f"First 10 token IDs from first sequence: {sample_batch['input_ids'][0, :10].tolist()}")
        
        # Print style labels if available
        if 'style_label_indices' in sample_batch:
            print(f"First 10 style labels from first sequence: {sample_batch['style_label_indices'][0, :10].tolist()}")
        
        # Process each MIDI file's tokens
        success_count = 0
        total_tokens = 0
        skipped_count = 0
        
        # Get the original file paths from the processed data
        midi_files = []
        for i, item in enumerate(dataset):
            if isinstance(item, dict) and 'file_path' in item:
                midi_files.append(item['file_path'])
            else:
                midi_files.append(f"file_{i}")
        
        # Set model to evaluation mode
        self.eval()
        
        # Process batches
        file_idx = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Analyzing MIDI files in batches"):
                # Get input_ids from batch
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                file_paths = batch['file_path']  # Get file paths directly from batch
                
                # Check if we have ground truth style labels (synthetic data)
                has_ground_truth = 'style_label_indices' in batch
                if has_ground_truth:
                    style_labels = batch['style_label_indices'].to(self.device)
                
                # Get model predictions
                logits = self(input_ids, attention_mask)
                
                # Convert logits to predictions and confidence scores
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                batch_confidence = torch.softmax(logits, dim=-1).cpu().numpy()
                
                # Process each item in the batch
                for b in range(len(batch_predictions)):
                    if file_idx >= len(midi_files):
                        break
                        
                    midi_file = file_paths[b]  # Use file path from batch
                    
                    # Get the actual tokens (remove padding)
                    tokens = input_ids[b].cpu().tolist()
                    mask = attention_mask[b].cpu().tolist()
                    tokens = [t for t, m in zip(tokens, mask) if m == 1]
                    
                    # Get predictions for this file
                    predictions = batch_predictions[b][:len(tokens)]
                    confidence_scores = np.max(batch_confidence[b][:len(tokens)], axis=1)
                    
                    # Convert predictions to style tokens
                    style_tokens = [self.idx_to_style[idx] for idx in predictions]
                    
                    # Get ground truth if available (synthetic data)
                    ground_truth = None
                    if has_ground_truth:
                        gt_indices = style_labels[b][:len(tokens)].cpu().tolist()
                        # Filter out ignore indices (-100)
                        gt_indices = [idx for idx in gt_indices if idx != self.IGNORE_LABEL_IDX]
                        if gt_indices:
                            ground_truth = [self.idx_to_style[idx] for idx in gt_indices]
                            ground_truth_condensed = condense_style_sequence(ground_truth)
                    
                    # Get significant labels
                    significant_labels = filter_significant_styles(style_tokens)
                    significant_condensed = condense_style_sequence(significant_labels)
                    
                    # Get timestamps for style changes
                    
                    style_change_timestamps = extract_style_change_timestamps(tokens, style_tokens, tokenizer=self.tokenizer)
                    # Create condensed representation of predictions
                    pred_condensed = condense_style_sequence(style_tokens)
                    
                    # Calculate average confidence per style
                    style_confidences = {}
                    for style in self.idx_to_style.values():
                        style_indices = [i for i, s in enumerate(style_tokens) if s == style]
                        if style_indices:
                            avg_confidence = np.mean(confidence_scores[style_indices])
                            style_confidences[style] = f"{avg_confidence:.3f}"
                    
                    confidence_str = ", ".join([f"{style}:{conf}" for style, conf in style_confidences.items()])
                    
                    # Extract unique styles from significant_condensed for music_style column
                    music_style = ""
                    if significant_condensed:
                        # Extract just the style letters (A, B, C, D) from significant_condensed
                        styles = re.findall(r'([A-D])x\d+', significant_condensed)
                        music_style = "".join(styles)
                    
                    # Store data for CSV - one row per file
                    csv_row = {
                        'file_id': midi_file,
                        'music_style': music_style,
                        'prediction': pred_condensed,
                        'significant_prediction': significant_condensed,
                        'num_tokens': len(tokens),
                        'style_change_timestamps': "; ".join([f"{style}:{time}" for style, time in style_change_timestamps if time is not None]),
                        'confidence_scores': confidence_str
                    }
                    
                    # Add ground truth if available
                    if ground_truth is not None:
                        csv_row['ground_truth'] = ground_truth_condensed
                    
                    csv_data.append(csv_row)
                    
                    # Copy MIDI file to output directory if specified
                    if results_dir and os.path.exists(midi_file) and os.path.isfile(midi_file):
                        dest_path = os.path.join(midi_output_dir, os.path.basename(midi_file))
                        # Create parent directories if they don't exist
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.copy2(midi_file, dest_path)
                    
                    success_count += 1
                    total_tokens += len(tokens)
                    file_idx += 1
        
        if success_count == 0:
            print("No files were successfully processed.")
            return
        
        print(f"Skipped {skipped_count} files (processing errors)")
        
        # Save results if requested
        if results_dir:
            # Save text summary
            with open(os.path.join(results_dir, 'processing_summary.txt'), 'w') as f:
                f.write("===== PROCESSING SUMMARY =====\n")
                f.write(f"Total files processed: {success_count}\n")
                f.write(f"Total files skipped: {skipped_count}\n")
                f.write(f"Total tokens processed: {total_tokens}\n")
            
            # Save predictions to CSV
            import csv
            csv_path = os.path.join(results_dir, 'midi_predictions.csv')
            with open(csv_path, 'w', newline='') as csvfile:
                # Add ground_truth to fieldnames if we have synthetic data
                fieldnames = ['file_id', 'music_style', 'timestamps', 'significant_styles']
                if mode == "synthetic":
                    fieldnames.append('ground_truth')
                fieldnames.extend(['confidence', 'num_tokens', 'prediction'])
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in csv_data:
                    csv_row = {
                        'file_id': row['file_id'],
                        'music_style': row['music_style'],
                        'timestamps': row['style_change_timestamps'],
                        'significant_styles': row['significant_prediction'],
                        'confidence': row['confidence_scores'],
                        'num_tokens': row['num_tokens'],
                        'prediction': row['prediction']
                    }
                    
                    # Add ground truth if available
                    if 'ground_truth' in row:
                        csv_row['ground_truth'] = row['ground_truth']
                    
                    writer.writerow(csv_row)
            print(f"\nResults saved to {results_dir}")
            print(f"MIDI predictions saved to {csv_path}")
            print(f"MIDI files copied to {os.path.join(results_dir, 'midi_files')}")
        
        return {
            'total_files': success_count,
            'total_tokens': total_tokens,
            'skipped_files': skipped_count
        }
    
    def evaluate_sequence(self, midi_ids, attention_mask=None):
        """
        Evaluate a sequence of MIDI token IDs and return the style logits.
        
        Args:
            midi_ids (torch.Tensor): Tensor of shape [batch_size, seq_len] containing MIDI token IDs
            attention_mask (torch.Tensor, optional): Tensor of shape [batch_size, seq_len] indicating which tokens
                                                    to attend to (1) and which to ignore (0)
        
        Returns:
            dict: Dictionary containing:
                - 'logits': Raw logits from the model (torch.Tensor of shape [batch_size, seq_len, n_classes])
                - 'predictions': Predicted style indices (torch.Tensor of shape [batch_size, seq_len])
                - 'style_tokens': Predicted style tokens as strings (list of lists)
                - 'confidence': Confidence scores for predictions (torch.Tensor of shape [batch_size, seq_len])
        """
        # Ensure the input is on the correct device
        if not isinstance(midi_ids, torch.Tensor):
            midi_ids = torch.tensor(midi_ids, dtype=torch.long)
        
        midi_ids = midi_ids.to(self.device)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(midi_ids, dtype=torch.long)
        else:
            attention_mask = attention_mask.to(self.device)
        
        # Set model to evaluation mode
        self.eval()
        
        # Get model predictions
        with torch.no_grad():
            logits = self(midi_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            confidence = torch.softmax(logits, dim=-1).max(dim=-1).values
        
        # Convert predictions to style tokens
        style_tokens = []
        for batch_idx in range(midi_ids.size(0)):
            # Get the actual tokens (remove padding)
            mask = attention_mask[batch_idx].cpu().tolist()
            pred = predictions[batch_idx].cpu().tolist()
            
            # Filter out padding tokens
            valid_preds = [pred[i] for i in range(len(pred)) if mask[i] == 1]
            
            # Convert to style tokens
            batch_style_tokens = [self.idx_to_style[idx] for idx in valid_preds]
            style_tokens.append(batch_style_tokens)
        
        return {
            'logits': logits,
            'predictions': predictions,
            'style_tokens': style_tokens,
            'confidence': confidence
        }
    
    
    # def generate_further_statistics(self, results_dir):
    #     """Generate additional statistics and visualizations from the evaluation results."""
    #     print("\nGenerating additional statistics...")
        
    #     # Load the data
    #     df = pd.read_csv(os.path.join(results_dir, 'file_predictions.csv'))
        
    #     # Create a histogram of accuracies
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(df['accuracy'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    #     plt.title('Distribution of Prediction Accuracies')
    #     plt.xlabel('Accuracy')
    #     plt.ylabel('Frequency')
    #     plt.grid(axis='y', alpha=0.75)
    #     plt.savefig(os.path.join(results_dir, 'accuracy_histogram.png'))
    #     plt.close()
        
    #     # Function to categorize ground truth patterns
    #     def categorize_pattern(pattern_str):
    #         # Remove counts and just keep the sequence of styles
    #         sequence = re.findall(r'([A-Z])x\d+', pattern_str)
            
    #         # Join the sequence into a string
    #         pattern = ''.join(sequence)
            
    #         # Identify common patterns
    #         if pattern == 'ABA':
    #             return 'ABA'
    #         elif pattern == 'AB':
    #             return 'AB'
    #         elif pattern == 'ABC':
    #             return 'ABC'
    #         elif pattern == 'ABCBA':
    #             return 'ABCBA'
    #         elif pattern == 'ABAC':
    #             return 'ABAC'
    #         elif pattern == 'ABCA':
    #             return 'ABCA'
    #         elif pattern == 'ABABA':
    #             return 'ABABA'
    #         elif pattern == 'ABCABC':
    #             return 'ABCABC'
    #         elif pattern == 'ABCDE':
    #             return 'ABCDE'
    #         elif pattern == 'ABCD':
    #             return 'ABCD'
    #         elif pattern == 'A':
    #             return 'A'
    #         elif pattern == 'ABCDEFG':
    #             return 'ABCDEFG'
    #         elif pattern == 'ABCBCBA':
    #             return 'ABCBCBA'
    #         else:
    #             return 'Other'
        
    #     # Function to parse the condensed format and get token counts
    #     def parse_condensed_format(pattern_str):
    #         tokens = {}
    #         matches = re.findall(r'([A-Z])x(\d+)', pattern_str)
    #         for style, count in matches:
    #             if style in tokens:
    #                 tokens[style] += int(count)
    #             else:
    #                 tokens[style] = int(count)
    #         return tokens
        
    #     # Categorize each row and extract token counts
    #     df['pattern'] = df['ground_truth'].apply(categorize_pattern)
    #     df['token_counts'] = df['ground_truth'].apply(parse_condensed_format)
        
    #     # Calculate statistics for each pattern type
    #     pattern_stats = defaultdict(lambda: {'count': 0, 'avg_accuracy': 0, 'total_tokens': 0, 'style_tokens': defaultdict(int)})
        
    #     for _, row in df.iterrows():
    #         pattern = row['pattern']
    #         pattern_stats[pattern]['count'] += 1
    #         pattern_stats[pattern]['avg_accuracy'] += row['accuracy']
            
    #         # Add token counts for each style
    #         for style, count in row['token_counts'].items():
    #             pattern_stats[pattern]['style_tokens'][style] += count
    #             pattern_stats[pattern]['total_tokens'] += count
        
    #     # Calculate averages
    #     for pattern, stats in pattern_stats.items():
    #         if stats['count'] > 0:
    #             stats['avg_accuracy'] /= stats['count']
        
    #     # Create a DataFrame for the pattern statistics
    #     pattern_df = pd.DataFrame([
    #         {
    #             'Pattern': pattern,
    #             'Count': stats['count'],
    #             'Avg Accuracy': stats['avg_accuracy'],
    #             'Total Tokens': stats['total_tokens'],
    #             **{f'Style {style} Tokens': count for style, count in stats['style_tokens'].items()},
    #             **{f'Style {style} %': count/stats['total_tokens']*100 for style, count in stats['style_tokens'].items() if stats['total_tokens'] > 0}
    #         }
    #         for pattern, stats in pattern_stats.items()
    #     ])
        
    #     # Sort by count
    #     pattern_df = pattern_df.sort_values('Count', ascending=False)
        
    #     # Save the pattern statistics to CSV
    #     pattern_df.to_csv(os.path.join(results_dir, 'pattern_statistics.csv'), index=False)
        
    #     # Create a bar chart of average accuracies by pattern
    #     plt.figure(figsize=(10, 6))
    #     plt.bar(pattern_df['Pattern'], pattern_df['Avg Accuracy'], color='skyblue', edgecolor='black')
    #     plt.title('Average Accuracy by Pattern Type')
    #     plt.xlabel('Pattern')
    #     plt.ylabel('Average Accuracy')
    #     plt.ylim(0.8, 1.0)  # Adjust as needed
    #     plt.grid(axis='y', alpha=0.75)
    #     plt.savefig(os.path.join(results_dir, 'pattern_accuracy_chart.png'))
    #     plt.close()
        
    #     print(f"Additional analysis complete. Files saved to {results_dir} directory.")
    

    def common_step(self, batch):
        """Common logic for training and validation steps."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        style_label_indices = batch['style_label_indices']  # Get file paths directly from batch
        
        logits = self(input_ids=input_ids, attention_mask=attention_mask)

        # Reshape for loss calculation
        logits_flat = logits.view(-1, self.hparams.n_classes)
        labels_flat = style_label_indices.view(-1)

        # Calculate loss only on non-ignored labels
        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)

        return loss, logits_flat, labels_flat

    def training_step(self, batch, batch_idx):
        loss, logits_flat, labels_flat = self.common_step(batch)

        # Update accuracy (will ignore -100 automatically)
        self.train_accuracy(logits_flat, labels_flat)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log('train_loss_step', loss, on_step=True, prog_bar=False, sync_dist=True) # Can be verbose
        # self.log('train_loss_epoch', loss, on_epoch=True, prog_bar=True, sync_dist=True) # Redundant with train_loss
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log('step', self.global_step, on_step=True, on_epoch=False, prog_bar=False)
        # self.log('epoch', self.current_epoch, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits_flat, labels_flat = self.common_step(batch)

        # Update accuracy
        self.val_accuracy(logits_flat, labels_flat)

        # Log validation metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Returning loss is needed for ModelCheckpoint/EarlyStopping
        return loss

    def configure_optimizers(self):
        # AdamW is generally preferred for transformers
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr) # Use hparams

        # Calculate total steps (important for scheduler)
        # Handle cases where trainer might not be fully initialized yet
        try:
            total_steps = self.trainer.estimated_stepping_batches
            logger.info(f"Optimizer: Calculated total steps: {total_steps}")
        except (AttributeError, TypeError):
             # Fallback if trainer/estimated_stepping_batches isn't available
             logger.warning("Could not estimate total steps from trainer. Using fallback.")
             # Estimate based on epochs and a guess for steps per epoch
             # This guess should be large enough to ensure warmup completes reasonably.
             steps_per_epoch_guess = 500 # Adjust this based on dataset size / batch size
             total_steps = self.trainer.max_epochs * steps_per_epoch_guess

        warmup_steps = int(0.1 * total_steps) # 10% warmup
        logger.info(f"Optimizer: Warmup steps: {warmup_steps}")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return { "optimizer": optimizer,
                 "lr_scheduler": { "scheduler": scheduler, "interval": "step", "frequency": 1 }
               }