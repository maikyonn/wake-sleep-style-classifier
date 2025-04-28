# tokenizer.py

import logging
from typing import List, Optional

from ariautils.tokenizer.absolute import AbsTokenizer
from ariautils.midi import MidiDict

logger = logging.getLogger(__name__)

# Style mapping constants

IGNORE_LABEL_IDX = -100  # Padding/ignore index for loss function
STYLE_LABEL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
ID_TO_STYLE_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

class MusicTokenizerWithStyle:
    """
    A wrapper around aria's AbsTokenizer focused solely on MIDI tokenization.
    Adds section indicator tokens and structure form tokens (e.g., <AB>, <AAA>, etc.) to the vocabulary.
    """
    STRUCTURE_FORMS = [
        "A", "AB", "ABC", "ABA", "ABAB", "ABAC", "ABCA", "ABCB", "ABCD",
        "AAA", "AABB", "AABBAA", "ABCD", "ABACA", "ABBA", "ABABCB", "ABACABA"
    ]

    def __init__(self, vocab_path: Optional[str] = None):
        # Initialize the base tokenizer
        self._tokenizer = AbsTokenizer()

        self.PROMPT_START_INDICATOR = '<PROMPT_START>'
        self.PROMPT_END_INDICATOR = '<PROMPT_END>'
        self.A_INDICATOR = '<A_SECTION>'
        self.B_INDICATOR = '<B_SECTION>'
        self.C_INDICATOR = '<C_SECTION>'
        self.D_INDICATOR = '<D_SECTION>'
        self.idx_to_style = {0: "A", 1: "B", 2: "C", 3: "D"}
        self.style_to_idx = {v: k for k, v in self.idx_to_style.items()}

        # Add section indicators and prompt indicators
        special_tokens = [
            self.A_INDICATOR, self.B_INDICATOR, self.C_INDICATOR, self.D_INDICATOR,
            self.PROMPT_START_INDICATOR, self.PROMPT_END_INDICATOR
        ]

        # Add structure form tokens like <AB>, <AAA>, etc.
        self.STRUCTURE_FORM_TOKENS = [f"<{form}>" for form in self.STRUCTURE_FORMS]
        special_tokens.extend(self.STRUCTURE_FORM_TOKENS)

        self._tokenizer.add_tokens_to_vocab(special_tokens)

        logger.info(
            f"MusicTokenizerWithStyle initialized. Vocab size: {self.vocab_size}, "
            f"Pad ID: {self.pad_id}, BOS ID: {self.bos_id}, EOS ID: {self.eos_id} "
            f"A_INDICATOR ID: {self.A_INDICATOR_ID}, B_INDICATOR ID: {self.B_INDICATOR_ID}, "
            f"C_INDICATOR ID: {self.C_INDICATOR_ID}, D_INDICATOR ID: {self.D_INDICATOR_ID}, "
            f"Structure form tokens: {self.STRUCTURE_FORM_TOKENS}"
        )

    def tokenize(self, midi_dict: MidiDict) -> List[str]:
        """Tokenizes a MidiDict object into a list of token strings."""
        return self._tokenizer.tokenize(midi_dict)

    def encode(self, seq: List[str]) -> List[int]:
        """Encodes a sequence of token strings into integer IDs."""
        return self._tokenizer.encode(seq)

    def decode(self, seq: List[int]) -> List[str]:
        """Decodes a sequence of integer IDs back into token strings."""
        return self._tokenizer.decode(seq)

    def detokenize(self, *args, **kwargs):
        """Detokenizes - pass through to underlying tokenizer."""
        if hasattr(self._tokenizer, 'detokenize'):
            return self._tokenizer.detokenize(*args, **kwargs)
        else:
            raise NotImplementedError("Detokenize method not available in underlying tokenizer")

    def tokenize_from_file(self, midi_path: str) -> Optional[List[str]]:
        """Tokenizes a MIDI file into token strings/tuples."""
        try:
            midi_dict = MidiDict.from_midi(midi_path)
            tokens = self.tokenize(midi_dict)
            if not tokens:
                logger.warning(f"No tokens generated for MIDI file: {midi_path}")
                return None
            return tokens
        except FileNotFoundError:
            logger.error(f"MIDI file not found: {midi_path}")
            return None
        except Exception as e:
            logger.error(f"Error processing MIDI {midi_path}: {e}", exc_info=False)
            return None

    def calc_length_ms(self, tokens: List[str], onset: bool = True) -> int:
        """Calculate the length of a MIDI sequence in milliseconds."""
        return self._tokenizer.calc_length_ms(tokens, onset)

    def ids_to_file(self, ids: List[int], output_path: str):
        """Converts a list of integer IDs back into a MIDI file."""
        try:
            generated_abs_sequence = self.decode(ids)
            midi_dict = self._tokenizer.detokenize(generated_abs_sequence)
            midi_file = midi_dict.to_midi()
            midi_file.save(output_path)
            logger.info(f"Sample MIDI file saved to {output_path}")
        except Exception as e:
            logger.error(f"Error converting IDs to MIDI file: {e}", exc_info=False)
            return None

    # Properties

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def A_INDICATOR_ID(self) -> int:
        return self._tokenizer.tok_to_id[self.A_INDICATOR]

    @property
    def B_INDICATOR_ID(self) -> int:
        return self._tokenizer.tok_to_id[self.B_INDICATOR]

    @property
    def C_INDICATOR_ID(self) -> int:
        return self._tokenizer.tok_to_id[self.C_INDICATOR]

    @property
    def D_INDICATOR_ID(self) -> int:
        return self._tokenizer.tok_to_id[self.D_INDICATOR]

    @property
    def pad_id(self) -> int:
        return self._tokenizer.pad_id

    @property
    def bos_id(self) -> Optional[int]:
        if hasattr(self._tokenizer, 'bos_id'):
            return self._tokenizer.bos_id
        elif hasattr(self._tokenizer, 'bos_tok'):
            return self._tokenizer.tok_to_id.get(self._tokenizer.bos_tok, None)
        return None

    @property
    def eos_id(self) -> Optional[int]:
        if hasattr(self._tokenizer, 'eos_id'):
            return self._tokenizer.eos_id
        elif hasattr(self._tokenizer, 'eos_tok'):
            return self._tokenizer.tok_to_id.get(self._tokenizer.eos_tok, None)
        return None

    @property
    def vocab(self) -> dict:
        if hasattr(self._tokenizer, 'vocab'):
            return self._tokenizer.vocab
        elif hasattr(self._tokenizer, 'tok_to_id'):
            return self._tokenizer.tok_to_id
        else:
            raise AttributeError("Underlying tokenizer has no 'vocab' or 'tok_to_id'")

    @property
    def id_to_tok(self) -> dict:
        if hasattr(self._tokenizer, 'id_to_tok'):
            return self._tokenizer.id_to_tok
        else:
            try:
                return {v: k for k, v in self.vocab.items()}
            except AttributeError:
                raise AttributeError("Cannot get id_to_tok mapping")



class MusicTokenizer:
    """
    A wrapper around aria's AbsTokenizer focused solely on MIDI tokenization.
    Style tokens ('A', 'B', 'C', 'D') are NOT added to the vocabulary here.
    """
    def __init__(self, vocab_path: Optional[str] = None):
        # Initialize the base tokenizer
        self._tokenizer = AbsTokenizer()

        logger.info(f"MusicTokenizer initialized. Vocab size: {self.vocab_size}, "
                   f"Pad ID: {self.pad_id}, BOS ID: {self.bos_id}, EOS ID: {self.eos_id}")


    def tokenize(self, midi_dict: MidiDict) -> List[str]:
        """Tokenizes a MidiDict object into a list of token strings."""
        return self._tokenizer.tokenize(midi_dict)

    def encode(self, seq: List[str]) -> List[int]:
        """Encodes a sequence of token strings into integer IDs."""
        return self._tokenizer.encode(seq)

    def decode(self, seq: List[int]) -> List[str]:
        """Decodes a sequence of integer IDs back into token strings."""
        return self._tokenizer.decode(seq)

    def detokenize(self, *args, **kwargs):
        """Detokenizes - pass through to underlying tokenizer."""
        if hasattr(self._tokenizer, 'detokenize'):
            return self._tokenizer.detokenize(*args, **kwargs)
        else:
            raise NotImplementedError("Detokenize method not available in underlying tokenizer")

    def tokenize_from_file(self, midi_path: str) -> Optional[List[str]]:
        """Tokenizes a MIDI file into token strings/tuples."""
        try:
            midi_dict = MidiDict.from_midi(midi_path)
            tokens = self.tokenize(midi_dict)
            if not tokens:
                logger.warning(f"No tokens generated for MIDI file: {midi_path}")
                return None
            return tokens
        except FileNotFoundError:
            logger.error(f"MIDI file not found: {midi_path}")
            return None
        except Exception as e:
            logger.error(f"Error processing MIDI {midi_path}: {e}", exc_info=False)
            return None
        
    def calc_length_ms(self, tokens: List[str], onset: bool = True) -> int:
        """Calculate the length of a MIDI sequence in milliseconds."""
        return self._tokenizer.calc_length_ms(tokens, onset)
    
    
    def ids_to_file(self, ids: List[int], output_path: str):
        """Converts a list of integer IDs back into a MIDI file."""
        try:
            generated_abs_sequence = self.decode(ids)
            midi_dict = self._tokenizer.detokenize(generated_abs_sequence)
            midi_file = midi_dict.to_midi()
            midi_file.save(output_path)
            logger.info(f"Sample MIDI file saved to {output_path}")
        except Exception as e:
            logger.error(f"Error converting IDs to MIDI file: {e}", exc_info=False)
            return None

    # Properties

    @property
    def vocab_size(self) -> int: return self._tokenizer.vocab_size
    
    @property
    def A_INDICATOR_ID(self) -> int: return self._tokenizer.tok_to_id[self.A_INDICATOR]
    
    @property
    def B_INDICATOR_ID(self) -> int: return self._tokenizer.tok_to_id[self.B_INDICATOR]
    
    @property
    def C_INDICATOR_ID(self) -> int: return self._tokenizer.tok_to_id[self.C_INDICATOR]
    
    @property
    def D_INDICATOR_ID(self) -> int: return self._tokenizer.tok_to_id[self.D_INDICATOR]

    @property
    def pad_id(self) -> int: return self._tokenizer.pad_id
    
    @property
    def bos_id(self) -> Optional[int]:
        if hasattr(self._tokenizer, 'bos_id'): return self._tokenizer.bos_id
        elif hasattr(self._tokenizer, 'bos_tok'): return self._tokenizer.tok_to_id.get(self._tokenizer.bos_tok, None)
        return None
    
    @property
    def eos_id(self) -> Optional[int]:
        if hasattr(self._tokenizer, 'eos_id'): return self._tokenizer.eos_id
        elif hasattr(self._tokenizer, 'eos_tok'): return self._tokenizer.tok_to_id.get(self._tokenizer.eos_tok, None)
        return None
    
    @property
    def vocab(self) -> dict:
        if hasattr(self._tokenizer, 'vocab'): return self._tokenizer.vocab
        elif hasattr(self._tokenizer, 'tok_to_id'): return self._tokenizer.tok_to_id
        else: raise AttributeError("Underlying tokenizer has no 'vocab' or 'tok_to_id'")
    
    @property
    def id_to_tok(self) -> dict:
        if hasattr(self._tokenizer, 'id_to_tok'): return self._tokenizer.id_to_tok
        else:
            try: return {v: k for k, v in self.vocab.items()}
            except AttributeError: raise AttributeError("Cannot get id_to_tok mapping")
