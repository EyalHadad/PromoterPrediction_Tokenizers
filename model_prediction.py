import os
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from transformers import RobertaForSequenceClassification
import pandas as pd
from tqdm import tqdm

# Constants
DATA_DIRS = ['Genome','Substitution']
FINETUNE_MODELS_DIR = 'finetune_models'
TOKENIZERS_DIR = 'tokenizers'
RESULTS_DIR = 'results'
BATCH_SIZE = 8

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset class
class DNADataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=128, overlap=False):
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
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

# Function to predict using a fine-tuned model
def predict_model(finetune_model_path, tokenizer, test_data, output_path, overlap=False):
    print(f"Predicting using model {finetune_model_path}...")

    # Load model
    model = RobertaForSequenceClassification.from_pretrained(finetune_model_path, local_files_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    model.eval()

    # Prepare dataset and dataloader
    test_sequences = test_data['seq'].tolist()
    dataset = DNADataset(test_sequences, tokenizer, overlap=overlap)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Predictions
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            predictions.extend(probs)

    # Save predictions
    test_data['Prediction'] = predictions
    test_data.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

# Main prediction loop
for data_dir in DATA_DIRS:
    for model_name in ['kmer_non_overlapping', 'kmer_overlapping', 'bpe', 'wpc']:
        for organism in ['human', 'musculus', 'norvegicus', 'rerio', 'melanogaster', 'celegans', 'gallus', 'mulatta']:
            model_path = os.path.join(FINETUNE_MODELS_DIR, f"{data_dir}_{model_name}_{organism}_finetuned")
            # Load test set
            file_dir = os.path.join('data', data_dir)
            test_files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.startswith('test')]
            test_data = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)
            test_data = test_data[test_data['organism'] == organism]

            if test_data.empty:
                continue

            # Load tokenizer
            tokenizer_path = os.path.join(TOKENIZERS_DIR, f"{model_name}_tokenizer.json")
            tokenizer = Tokenizer.from_file(tokenizer_path)

            # Define overlap for kmer models
            overlap = (model_name == 'kmer_overlapping')

            # Define output path
            output_file = f"{data_dir}_{model_name}_{organism}_predictions.csv"
            output_path = os.path.join(RESULTS_DIR, output_file)

            # Predict
            predict_model(model_path, tokenizer, test_data, output_path, overlap=overlap)

print("Predictions completed for all models.")
