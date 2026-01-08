import os
import torch
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification, AdamW
from tokenizers import Tokenizer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


import data_utils

# Constants
DATA_PATH = 'data/positive_data.csv'
DOG_FASTA = 'data/C.familiaris.txt'
OUTPUT_DIR = 'advanced_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AdvancedDNADataset(Dataset):
    """
    Enhanced Dataset class that handles:
    1. Character tokenization (splitting by char)
    2. Short promoter slicing (center crop)
    """
    def __init__(self, sequences, labels=None, tokenizer=None, max_length=600,
                 overlap=False, is_char=False, target_seq_len=600):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap = overlap
        self.is_char = is_char
        self.target_seq_len = target_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # Handle Short Promoter (Center Crop)
        if self.target_seq_len < 600:
            center = len(sequence) // 2
            start = center - (self.target_seq_len // 2)
            end = start + self.target_seq_len
            sequence = sequence[start:end]

        # Tokenization Logic
        if self.is_char:
            tokenized_sequence = ' '.join(list(sequence))
        else:
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

        item = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

def run_pretraining_in_memory(sequences, tokenizer, is_char=False, device='cuda'):
    """
    Runs pretraining and returns the MODEL OBJECT (not saving to disk),
    as requested for the Char experiment.
    """
    print("Running Pretraining (In-Memory)...")
    dataset = AdvancedDNADataset(sequences, tokenizer=tokenizer, is_char=is_char)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    config = RobertaConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_position_embeddings=600 + 2, # +2 for specials
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
        pad_token_id=tokenizer.token_to_id("[PAD]"),
        bos_token_id=tokenizer.token_to_id("[CLS]"),
        eos_token_id=tokenizer.token_to_id("[SEP]")
    )

    model = RobertaForMaskedLM(config)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model.train()

    for epoch in range(2):
        loop = tqdm(dataloader, desc=f"Pretrain Epoch {epoch+1}")
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone() # MLM labels

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    print("Pretraining finished.")
    return model

def run_finetuning_custom(model, tokenizer, train_seqs, train_lbls, val_seqs, val_lbls,
                          is_char=False, seq_len=600, device='cuda', no_pretrain=False):
    """
    Custom Finetuning loop that supports:
    - In-memory pretrained model
    - 'No Pretrain' initialization
    - Short sequences
    """
    print(f"Starting Finetuning (Seq Len: {seq_len}, Char: {is_char}, NoPretrain: {no_pretrain})")

    # Dataset Preparation
    train_ds = AdvancedDNADataset(train_seqs, train_lbls, tokenizer, is_char=is_char, target_seq_len=seq_len)
    val_ds = AdvancedDNADataset(val_seqs, val_lbls, tokenizer, is_char=is_char, target_seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Model Initialization
    if no_pretrain:
        # Initialize from scratch
        config = RobertaConfig(
            vocab_size=tokenizer.get_vocab_size(),
            max_position_embeddings=600 + 2,
            num_attention_heads=12,
            num_hidden_layers=6,
            num_labels=2
        )
        clf_model = RobertaForSequenceClassification(config)
    else:
        # Use the pretrained weights (transfer from MaskedLM to SeqClass)
        # Note: 'model' passed here is RobertaForMaskedLM. We need to extract base or load state.
        # Quickest way: Save config, create new Class model, load compatible weights.
        # Ideally, we used RobertaForSequenceClassification.from_pretrained...
        # But since we have the object in memory, let's just copy the encoder.
        config = model.config
        config.num_labels = 2
        clf_model = RobertaForSequenceClassification(config)
        # Copy weights from roberta encoder
        clf_model.roberta.load_state_dict(model.roberta.state_dict())

    clf_model.to(device)
    optimizer = AdamW(clf_model.parameters(), lr=1e-5)

    # Training Loop
    for epoch in range(1): # Reduced for demo, set to 10
        clf_model.train()
        for batch in tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = clf_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    clf_model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = clf_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds.extend(probs)
            truths.extend(batch['labels'].cpu().numpy())

    auc = roc_auc_score(truths, preds)
    print(f"Final AUC: {auc:.4f}")
    return clf_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['dog', 'char', 'no_pretrain', 'short'],
                        help='Which advanced experiment to run')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load General Data
    pos_data = pd.read_csv(DATA_PATH)
    all_pos_seqs = pos_data['seq'].tolist()

    # ------------------------------------------------------------------
    # EXPERIMENT 1: Character Tokenizer
    # ------------------------------------------------------------------
    if args.experiment == 'char':
        # 1. Train Tokenizer
        tokenizer = data_utils.train_character_tokenizer(all_pos_seqs)

        # 2. Pretrain (In Memory)
        pretrained_model = run_pretraining_in_memory(all_pos_seqs, tokenizer, is_char=True, device=device)

        # 3. Finetune (Example on Human)
        # Load human data (example path, adjust to your structure)
        human_train = pd.read_csv('data/Substitution/train_human.csv')
        human_val = pd.read_csv('data/Substitution/val_human.csv')

        run_finetuning_custom(
            pretrained_model, tokenizer,
            human_train['seq'].tolist(), human_train['True_label'].tolist(),
            human_val['seq'].tolist(), human_val['True_label'].tolist(),
            is_char=True, device=device
        )

    # ------------------------------------------------------------------
    # EXPERIMENT 2: Dog External Validation
    # ------------------------------------------------------------------
    elif args.experiment == 'dog':
        # 1. Get filtered Dog data
        dog_df = data_utils.get_dog_data_filtered(DOG_FASTA, DATA_PATH)

        # 2. Load a Pretrained/Finetuned model (Human model for example)
        # Assuming you want to test the BEST model (Non-overlapping)
        tokenizer = Tokenizer.from_file("tokenizers/kmer_non_overlapping_tokenizer.json")

        # Load one of your saved models
        model_path = "finetune_models/Substitution_kmer_non_overlapping_human_finetuned"
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}. Run finetuning first.")
            return

        model = RobertaForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()

        # 3. Predict on Dog
        dataset = AdvancedDNADataset(dog_df['seq'].tolist(), tokenizer=tokenizer, overlap=False)
        loader = DataLoader(dataset, batch_size=8)

        preds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting Dog"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
                preds.extend(probs)

        print(f"Predictions on {len(preds)} dog sequences completed.")
        # Save results...

    # ------------------------------------------------------------------
    # EXPERIMENT 3: No Pretrain
    # ------------------------------------------------------------------
    elif args.experiment == 'no_pretrain':
        tokenizer = Tokenizer.from_file("tokenizers/kmer_non_overlapping_tokenizer.json")
        human_train = pd.read_csv('data/Substitution/train_human.csv')
        human_val = pd.read_csv('data/Substitution/val_human.csv')

        run_finetuning_custom(
            None, tokenizer, # Model is None -> triggers scratch init
            human_train['seq'].tolist(), human_train['True_label'].tolist(),
            human_val['seq'].tolist(), human_val['True_label'].tolist(),
            is_char=False, device=device, no_pretrain=True
        )

    # ------------------------------------------------------------------
    # EXPERIMENT 4: Short Promoter (300bp)
    # ------------------------------------------------------------------
    elif args.experiment == 'short':
        # Same as standard but with seq_len=300
        # Need to load pretrained model first
        tokenizer = Tokenizer.from_file("tokenizers/kmer_non_overlapping_tokenizer.json")

        # Mocking a loaded pretrained model object (in reality load from disk)
        config = RobertaConfig(vocab_size=tokenizer.get_vocab_size())
        pretrained_model = RobertaForMaskedLM(config) # Placeholder

        human_train = pd.read_csv('data/Substitution/train_human.csv')
        human_val = pd.read_csv('data/Substitution/val_human.csv')

        run_finetuning_custom(
            pretrained_model, tokenizer,
            human_train['seq'].tolist(), human_train['True_label'].tolist(),
            human_val['seq'].tolist(), human_val['True_label'].tolist(),
            is_char=False, seq_len=300, device=device
        )

if __name__ == "__main__":
    main()