⸻

Midi Structure Classifier & Style-Aware MIDI Generation

A research prototype for structure-aware music modeling that combines:
	1.	A token-level MIDI style classifier (A/B/C/D)
	2.	A generative LM fine-tuned in a wake–sleep loop using style prompts from the frozen classifier.

The tokenizer extends a standard MIDI tokenizer with structural tokens (<A_SECTION>, <PROMPT_START>, <ABA>, etc.).

⸻

Repository Layout

.
├─ aria_generative/model.py     # Transformer blocks for LM / conditional / classifier heads
├─ datamodule.py                # Lightning DataModules (synthetic + real MIDI)
├─ dataset.py                   # Dataset + parallel I/O, caching, augmentation
├─ MidiClassifierModel.py       # BERT-style token classifier (A/B/C/D)
├─ PLGeneratorDM.py             # Generator fine-tuner using frozen classifier
├─ tokenizer.py                 # Tokenizer with structural tokens
├─ utils.py                     # Prompt builders, helpers, parallel I/O
├─ wake_pl.py                   # Generator fine-tuning entrypoint
└─ train_ws_new.sh              # SLURM launcher


⸻

Workflow
	1.	Tokenization: Adds section and form tokens (<A_SECTION>, <PROMPT_START>, <ABA>).
	2.	Classifier: BERT-like encoder predicts per-token style A–D (trained on synthetic pairs).
	3.	Wake–Sleep Fine-Tuning:
Freeze classifier → infer styles from real MIDI → build prompts → train LM to continue from them.

⸻

Data Layout

datasets/<synthetic>/
  ├─ midi/*.mid
  └─ style/*.txt    # per-MIDI A/B/C/D labels

datasets/<real>/data/**/*.mid

Data is cached under ./cache.

⸻

Install

pip install torch pytorch-lightning transformers wandb tqdm numpy
# Plus:
# - ariautils / aria  (AbsTokenizer, MidiDict)
# - sageattention     (custom attention kernel)


⸻

Quickstart

Fine-tune generator:

python wake_pl.py \
  --data_dir datasets/aria-midi-cycle1/data \
  --devices 1 --num_nodes 1 --file_limit 5000 \
  --max_seq_len 3820 --wandb_mode disabled

Multi-node (SLURM):

sbatch train_ws_new.sh

Train classifier (optional): use MidiClassifierModel.py + SyntheticMidiDataModule.

⸻

Core Modules

Component	File	Purpose
Tokenizer	tokenizer.py	Adds structural tokens; encode/decode helpers
Classifier	MidiClassifierModel.py	Per-token A–D labels; BERT backbone
Generator	PLGeneratorDM.py, aria_generative/model.py	Fine-tunes LM via inferred prompts
Data	datamodule.py, dataset.py	Load synthetic/real MIDI, augment, cache
Utils	utils.py	Build prompts, extract timestamps, parallel I/O


⸻

Outputs
	•	Checkpoints → ./checkpoints/
	•	Logs → WandB or local
	•	CSV summaries → classifier predictions

⸻

Example

from MidiClassifierModel import MidiClassifier
from tokenizer import MusicTokenizerWithStyle
tok = MusicTokenizerWithStyle()
model = MidiClassifier(vocab_size=tok.vocab_size).eval()
out = model.evaluate_sequence(torch.randint(0, tok.vocab_size, (2,512)))
print(out['style_tokens'][0][:64])


⸻

Notes
	•	Adjust base-weights/ paths in wake_pl.py.
	•	Remove optional ZClipLightningCallback if missing.
	•	Requires ariautils + sageattention.
	•	Long MIDI truncated to max_len.

⸻


⸻

Would you like me to make it Markdown-styled for GitHub (with collapsible sections and badges) next? That would make it even cleaner for presentation.
