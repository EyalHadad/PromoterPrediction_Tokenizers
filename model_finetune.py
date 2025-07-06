import os
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from transformers import RobertaConfig, RobertaForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from tqdm import tqdm
import pandas as pd
import logging

# Constants
DATA_DIRS = ['Genome','Substitution']
PRETRAIN_MODELS_DIR = 'pretrain_models'
TOKENIZERS_DIR = 'tokenizers'
OUTPUT_DIR = 'finetune_models'
LOGS_DIR = 'logs'
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5

# Ensure output and log directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, 'finetuning.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Dataset class
class DNADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=600, overlap=False):
        self.sequences = sequences
        self.labels = labels
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
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Function to fine-tune a model
def fine_tune_model(pretrained_model_path, tokenizer, train_data, val_data, organism, output_model_path, overlap=False):
    logging.info(f"Starting fine-tuning for {os.path.basename(pretrained_model_path)} on {organism}...")
    print(f"Starting fine-tuning for {os.path.basename(pretrained_model_path)} on {organism}...")

    # Prepare datasets and dataloaders
    train_sequences, train_labels = train_data
    val_sequences, val_labels = val_data

    train_dataset = DNADataset(train_sequences, train_labels, tokenizer, overlap=overlap)
    val_dataset = DNADataset(val_sequences, val_labels, tokenizer, overlap=overlap)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    config = RobertaConfig.from_pretrained(pretrained_model_path)
    model = RobertaForSequenceClassification.from_pretrained(pretrained_model_path, config=config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_accuracy = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['labels'].to(model.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_pretrained(output_model_path)
            logging.info(f"New best model saved at {output_model_path} with accuracy {val_accuracy:.4f}")

        logging.info(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Main fine-tuning loop
for data_dir in DATA_DIRS:
    for model_name in ['kmer_non_overlapping', 'kmer_overlapping', 'bpe', 'wpc']:
        model_path = os.path.join(PRETRAIN_MODELS_DIR, f"{model_name}_pretrained")

        for organism in ['human', 'musculus', 'norvegicus', 'rerio', 'melanogaster', 'celegans', 'gallus', 'mulatta']:
            # Load train and validation sets
            file_dir = os.path.join('data', data_dir)
            train_files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.startswith('train')]
            val_files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.startswith('val')]

            train_data = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
            val_data = pd.concat([pd.read_csv(f) for f in val_files], ignore_index=True)

            train_sequences = train_data[train_data['organism'] == organism]['seq'].tolist()
            train_labels = train_data[train_data['organism'] == organism]['True_label'].tolist()

            val_sequences = val_data[val_data['organism'] == organism]['seq'].tolist()
            val_labels = val_data[val_data['organism'] == organism]['True_label'].tolist()

            if not train_sequences or not val_sequences:
                continue

            # Load tokenizer
            tokenizer_path = os.path.join(TOKENIZERS_DIR, f"{model_name}_tokenizer.json")
            tokenizer = Tokenizer.from_file(tokenizer_path)

            # Define overlap for kmer models
            overlap = (model_name == 'kmer_overlapping')

            # Define output model path
            output_model_path = os.path.join(OUTPUT_DIR, f"{data_dir}_{model_name}_{organism}_finetuned")
            os.makedirs(output_model_path, exist_ok=True)

            # Fine-tune
            fine_tune_model(
                model_path,
                tokenizer,
                (train_sequences, train_labels),
                (val_sequences, val_labels),
                organism,
                output_model_path,
                overlap=overlap
            )

logging.info("Fine-tuning completed for all models.")
print("Fine-tuning completed for all models.")