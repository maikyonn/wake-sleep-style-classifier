
⸻


# Midi Structure Classifier & Style-Aware MIDI Generator (Wake-Sleep)

A research prototype for **structure-aware music modeling**, combining:

1. A **token-level MIDI style classifier** (`A/B/C/D`)
2. A **generative LM** fine-tuned in a **wake–sleep loop** using style prompts inferred by the frozen classifier.

> Extends a standard MIDI tokenizer with structural tokens like `<A_SECTION>`, `<PROMPT_START>`, and form tags (`<ABA>`, `<ABAC>`, …).

---

## 📁 Repository Layout

.
├─ aria_generative/model.py     # Transformer blocks (LM / conditional / classifier heads)
├─ datamodule.py                # Lightning DataModules for synthetic + real MIDI
├─ dataset.py                   # Dataset loading, caching, augmentation
├─ MidiClassifierModel.py       # BERT-style token classifier (A/B/C/D)
├─ PLGeneratorDM.py             # Generator fine-tuning via frozen classifier
├─ tokenizer.py                 # Tokenizer with structural tokens
├─ utils.py                     # Prompt builders, parallel I/O, style helpers
├─ wake_pl.py                   # Training entrypoint
└─ train_ws_new.sh              # SLURM launcher

---

## 🧠 Workflow Overview

1. **Tokenization**  
   Adds section and form tokens (`<A_SECTION>`, `<PROMPT_START>`, `<ABA>`).

2. **Classifier (Optional)**  
   BERT-like encoder predicts per-token style `A–D`, trained on labeled synthetic MIDI.

3. **Wake–Sleep Fine-Tuning**  
   Freeze classifier → infer latent styles from real MIDI → build prompts → train LM to continue/generate from them.

---

## 🗂️ Data Layout

datasets//
├─ midi/.mid
└─ style/.txt      # per-MIDI A/B/C/D labels

datasets//data/**/*.mid

Data is cached under `./cache`.

---

## ⚙️ Installation

Tested with **Python ≥ 3.10** and **CUDA GPUs**.

```bash
pip install torch pytorch-lightning transformers wandb tqdm numpy

Additional internal deps:
	•	ariautils / aria — provides AbsTokenizer, MidiDict
	•	sageattention — custom attention kernel

⸻

🚀 Quickstart

Single machine:

python wake_pl.py \
  --data_dir datasets/aria-midi-cycle1/data \
  --devices 1 --num_nodes 1 \
  --file_limit 5000 --max_seq_len 3820 \
  --wandb_mode disabled

Multi-node (SLURM):

sbatch train_ws_new.sh

Train classifier (optional):
Use MidiClassifierModel.py + SyntheticMidiDataModule from datamodule.py.

⸻

🧩 Core Modules

Component	File	Purpose
Tokenizer	tokenizer.py	Adds structural tokens + helpers
Classifier	MidiClassifierModel.py	Per-token A–D style prediction
Generator	PLGeneratorDM.py, aria_generative/model.py	LM fine-tuning via inferred prompts
Data	datamodule.py, dataset.py	Load, augment, and cache MIDI datasets
Utils	utils.py	Prompt construction, style extraction, I/O helpers


⸻

📤 Outputs
	•	Checkpoints: ./checkpoints/<run>/
	•	Logs: WandB (or local)
	•	CSV summaries: classifier predictions

⸻

💻 Example

from MidiClassifierModel import MidiClassifier
from tokenizer import MusicTokenizerWithStyle
import torch

tok = MusicTokenizerWithStyle()
model = MidiClassifier(vocab_size=tok.vocab_size).eval()

out = model.evaluate_sequence(torch.randint(0, tok.vocab_size, (2,512)))
print(out['style_tokens'][0][:64])


⸻

⚠️ Notes
	•	Adjust base-weights/ paths in wake_pl.py.
	•	Remove optional ZClipLightningCallback if missing.
	•	Requires ariautils + sageattention.
	•	Long MIDI files truncated to max_len.

⸻

📜 License

TBD

⸻

📖 Citation

If you use this project, please cite appropriately (TBD).

---

Would you like me to make a **slightly more academic variant** next (e.g. with an “Abstract”, “Method”, and “Results” section for arXiv or GitHub research visibility)?
