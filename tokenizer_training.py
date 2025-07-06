import os
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE, WordPiece
from tokenizers.trainers import WordLevelTrainer, BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

# Constants
DATA_PATH = 'data/positive_data.csv'
OUTPUT_DIR = 'tokenizers'
VOCAB_SIZE = 30000  # Adjust as needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the data
data = pd.read_csv(DATA_PATH)
sequences = data['seq'].tolist()

def create_fixed_length_tokens(sequence, token_length=6, overlap=False):
    """Create k-mers with optional overlapping."""
    if overlap:
        return [sequence[i:i+token_length] for i in range(len(sequence) - token_length + 1)]
    else:
        return [sequence[i:i+token_length] for i in range(0, len(sequence), token_length)]

def train_kmer_tokenizer(sequences, vocab_size, token_length=6, overlap=False):
    """Train a k-mer tokenizer."""
    tokenized_sequences = [' '.join(create_fixed_length_tokens(seq, token_length, overlap)) for seq in sequences]
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    trainer = WordLevelTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(tokenized_sequences, trainer=trainer)
    return tokenizer

def train_bpe_tokenizer(sequences, vocab_size):
    """Train a BPE tokenizer."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(sequences, trainer=trainer)
    return tokenizer

def train_wpc_tokenizer(sequences, vocab_size):
    """Train a WordPiece tokenizer."""
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(sequences, trainer=trainer)
    return tokenizer

# Train and save non-overlapping K-MER tokenizer
print("Training non-overlapping K-MER tokenizer...")
non_overlapping_tokenizer = train_kmer_tokenizer(sequences, VOCAB_SIZE, overlap=False)
non_overlapping_tokenizer.save(os.path.join(OUTPUT_DIR, "kmer_non_overlapping_tokenizer.json"))

# Train and save overlapping K-MER tokenizer
print("Training overlapping K-MER tokenizer...")
overlapping_tokenizer = train_kmer_tokenizer(sequences, VOCAB_SIZE, overlap=True)
overlapping_tokenizer.save(os.path.join(OUTPUT_DIR, "kmer_overlapping_tokenizer.json"))

# Train and save BPE tokenizer
print("Training BPE tokenizer...")
bpe_tokenizer = train_bpe_tokenizer(sequences, VOCAB_SIZE)
bpe_tokenizer.save(os.path.join(OUTPUT_DIR, "bpe_tokenizer.json"))

# Train and save WordPiece tokenizer
print("Training WordPiece tokenizer...")
wpc_tokenizer = train_wpc_tokenizer(sequences, VOCAB_SIZE)
wpc_tokenizer.save(os.path.join(OUTPUT_DIR, "wpc_tokenizer.json"))

print("All tokenizers trained and saved.")
