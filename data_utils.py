import os
import tempfile
import subprocess
import pandas as pd
from Bio import SeqIO
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit

def get_dog_data_filtered(dog_fasta_path, train_csv_path, cdhit_cmd="cd-hit-est"):
    """
    Filters dog promoters:
    1. Removes sequences with 'N'.
    2. Runs CD-HIT-EST against the training set (positive_data.csv).
    3. Returns only dog sequences that form single-member clusters (no overlap with train).
    """
    print(f"Processing Dog data from {dog_fasta_path}...")

    # 1. Load Dog sequences and filter 'N'
    dog_seqs = []
    with open(dog_fasta_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_str = str(record.seq).upper()
            if 'N' not in seq_str:
                dog_seqs.append({'id': record.id, 'seq': seq_str, 'type': 'dog'})

    print(f"Dog sequences after removing 'N': {len(dog_seqs)}")

    # 2. Load Training sequences
    train_df = pd.read_csv(train_csv_path)
    train_seqs = []
    for _, row in train_df.iterrows():
        train_seqs.append({'id': f"train_{row['ID']}", 'seq': row['seq'], 'type': 'train'})

    # 3. Create temp FASTA for CD-HIT (Dog + Train)
    all_seqs = dog_seqs + train_seqs

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_in, \
            tempfile.NamedTemporaryFile(mode='w', suffix='.clstr', delete=False) as tmp_out:

        input_fasta = tmp_in.name
        output_result = tmp_out.name  # CD-HIT output prefix

        # Write combined FASTA
        for item in all_seqs:
            tmp_in.write(f">{item['id']}\n{item['seq']}\n")
        tmp_in.close()
        tmp_out.close() # Close to allow CD-HIT to write

        # Run CD-HIT-EST
        # -c 0.8: 80% identity
        # -n 5: word size for 0.8 threshold
        # -M 16000: Memory
        cmd = [cdhit_cmd, "-i", input_fasta, "-o", output_result, "-c", "0.8", "-n", "5", "-M", "16000", "-T", "0"]

        print("Running CD-HIT-EST (this may take a few minutes)...")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            raise FileNotFoundError("cd-hit-est command not found. Please ensure it is installed in your environment.")
        except subprocess.CalledProcessError as e:
            print(f"Error running CD-HIT: {e}")
            raise

        # 4. Parse CD-HIT .clstr output
        # We need clusters that contain ONLY ONE sequence, and that sequence must be a 'dog' sequence.

        clstr_file = output_result + ".clstr"
        final_dog_seqs = []

        current_cluster_ids = []

        with open(clstr_file, 'r') as f:
            for line in f:
                if line.startswith(">Cluster"):
                    # Process previous cluster
                    if len(current_cluster_ids) == 1:
                        seq_id = current_cluster_ids[0]
                        # If the single member is a dog sequence (not starting with 'train_')
                        if not seq_id.startswith("train_"):
                            # Find the sequence content
                            # (Inefficient search, but okay for this scale. Optimization: Dict lookup)
                            original_seq = next(s['seq'] for s in dog_seqs if s['id'] == seq_id)
                            final_dog_seqs.append(original_seq)

                    current_cluster_ids = []
                else:
                    # Line format: 0	601nt, >seq_id... *
                    # Extract ID between > and ...
                    parts = line.split('>')
                    if len(parts) > 1:
                        seq_id = parts[1].split('.')[0]
                        current_cluster_ids.append(seq_id)

            # Process last cluster
            if len(current_cluster_ids) == 1:
                seq_id = current_cluster_ids[0]
                if not seq_id.startswith("train_"):
                    original_seq = next(s['seq'] for s in dog_seqs if s['id'] == seq_id)
                    final_dog_seqs.append(original_seq)

        # Cleanup temp files
        os.remove(input_fasta)
        os.remove(output_result)
        os.remove(clstr_file)

        print(f"Final retained Dog sequences (singletons): {len(final_dog_seqs)}")

        # Create a DataFrame structure compatible with the pipeline
        return pd.DataFrame({'seq': final_dog_seqs, 'organism': 'dog', 'True_label': 1}) # All are positive promoters

def train_character_tokenizer(sequences, vocab_size=50): # Small vocab for chars
    """
    Trains a character-level tokenizer (WordLevel treating each char as a word).
    Since we need ACGTN, we effectively split by character.
    """
    print("Training Character-level tokenizer...")
    # Trick: Insert spaces between characters to treat them as words for WordLevel trainer
    spaced_seqs = [" ".join(list(seq)) for seq in sequences]

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit() # Split by the spaces we added

    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    tokenizer.train_from_iterator(spaced_seqs, trainer=trainer)
    return tokenizer