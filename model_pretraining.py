import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaConfig, RobertaForMaskedLM, AdamW
from tokenizers import Tokenizer
import pandas as pd
from tqdm import tqdm
import logging

# Constants
DATA_PATH = 'data/positive_data.csv'
TOKENIZERS_DIR = 'tokenizers'
OUTPUT_DIR = 'pretrain_models'
LOGS_DIR = 'logs'
BATCH_SIZE = 8
NUM_EPOCHS = 2
LEARNING_RATE = 1e-5
MAX_SEQ_LENGTH = 600  # Adjust as needed

# Ensure output and log directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, 'pretraining.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load data
data = pd.read_csv(DATA_PATH)
sequences = data['seq'].tolist()

class DNADataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=MAX_SEQ_LENGTH, overlap=False):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap = overlap

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # Generate tokens with or without overlap
        if self.overlap:
            tokens = [sequence[i:i+6] for i in range(len(sequence) - 6 + 1)]
        else:
            tokens = [sequence[i:i+6] for i in range(0, len(sequence), 6)]

        tokenized_sequence = ' '.join(tokens)
        encoded = self.tokenizer.encode(tokenized_sequence)
        input_ids = encoded.ids[:self.max_length]
        attention_mask = [1] * len(input_ids)

        # Padding
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.token_to_id("[PAD]")] * padding_length
        attention_mask += [0] * padding_length

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(input_ids, dtype=torch.long),
        }

# Function to train a model
def train_model(tokenizer_name, tokenizer_path, output_model_path, overlap=False):
    logging.info(f"Starting pretraining for {tokenizer_name} tokenizer...")
    print(f"Starting pretraining for {tokenizer_name} tokenizer...")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Prepare dataset and dataloader
    dataset = DNADataset(sequences, tokenizer, max_length=MAX_SEQ_LENGTH, overlap=overlap)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Configure model
    config = RobertaConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_position_embeddings=600,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
        pad_token_id=tokenizer.token_to_id("[PAD]"),
        bos_token_id=tokenizer.token_to_id("[CLS]"),
        eos_token_id=tokenizer.token_to_id("[SEP]")
    )

    model = RobertaForMaskedLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.train()
    train_history = []
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        train_history.append(avg_loss)
        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} completed. Average Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    # Save the pretrained model
    model.save_pretrained(output_model_path)
    logging.info(f"Model saved to {output_model_path}")
    print(f"Model saved to {output_model_path}")

    # Log training history
    logging.info(f"Training history for {tokenizer_name} tokenizer: {train_history}")

# Pretrain models
#train_model("non-overlapping K-MER", os.path.join(TOKENIZERS_DIR, "kmer_non_overlapping_tokenizer.json"), os.path.join(OUTPUT_DIR, "kmer_non_overlapping_pretrained"), overlap=False)
train_model("overlapping K-MER", os.path.join(TOKENIZERS_DIR, "kmer_overlapping_tokenizer.json"), os.path.join(OUTPUT_DIR, "kmer_overlapping_pretrained"), overlap=True)
train_model("BPE", os.path.join(TOKENIZERS_DIR, "bpe_tokenizer.json"), os.path.join(OUTPUT_DIR, "bpe_pretrained"))
train_model("WPC", os.path.join(TOKENIZERS_DIR, "wpc_tokenizer.json"), os.path.join(OUTPUT_DIR, "wpc_pretrained"))

logging.info("Pretraining completed for all models.")
print("Pretraining completed for all models.")
