import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set up directories
BASE_DIR = 'shap_plots'
OUTPUT_DIR = os.path.join(BASE_DIR, 'shap_importance_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
NUM_TOKENS = 100  # assuming 100 tokens per sequence
SEQ_LENGTH = 600  # 600 nucleotides
METHODS = ['Substitution', 'Genome']  # Changed order

# Helper function to aggregate importance per position
def aggregate_importance_by_position(files):
    position_values = {i: [] for i in range(NUM_TOKENS)}
    for f in files:
        df = pd.read_csv(os.path.join(BASE_DIR, f))
        df = df.head(NUM_TOKENS)
        for i, val in enumerate(df['importance'].values):
            position_values[i].append(val)

    return [sum(position_values[i])/len(position_values[i]) for i in range(NUM_TOKENS)]

# 1. Line plot comparing organisms within Genome method
plt.figure(figsize=(12, 6))
for organism in sorted(set([f.split('_')[4] for f in os.listdir(BASE_DIR) if f.startswith('Genome')])):
    files = [f for f in os.listdir(BASE_DIR) if f.startswith('Genome') and f"_{organism}_" in f]
    mean_values = aggregate_importance_by_position(files)
    plt.plot(range(1, SEQ_LENGTH + 1, 6), mean_values, label=organism, linewidth=2.5)
plt.title('SHAP Importance by Token Position - Genome')
plt.xlabel('Nucleotide Position')
plt.ylabel('Average Importance')
plt.xlim([1, SEQ_LENGTH])
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'genome_organism_comparison.png'))
plt.close()

# 2. Line plot comparing organisms within Substitution method
plt.figure(figsize=(12, 6))
for organism in sorted(set([f.split('_')[4] for f in os.listdir(BASE_DIR) if f.startswith('Substitution')])):
    files = [f for f in os.listdir(BASE_DIR) if f.startswith('Substitution') and f"_{organism}_" in f]
    mean_values = aggregate_importance_by_position(files)
    plt.plot(range(1, SEQ_LENGTH + 1, 6), mean_values, label=organism, linewidth=2.5)
plt.title('SHAP Importance by Token Position - Substitution')
plt.xlabel('Nucleotide Position')
plt.ylabel('Average Importance')
plt.xlim([1, SEQ_LENGTH])
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'substitution_organism_comparison.png'))
plt.close()

# 3. Comparison of Genome and Substitution (average over all organisms)
plt.figure(figsize=(12, 6))
for method in METHODS:
    files = [f for f in os.listdir(BASE_DIR) if f.startswith(method)]
    mean_values = aggregate_importance_by_position(files)
    plt.plot(range(1, SEQ_LENGTH + 1, 6), mean_values, label=method, linewidth=2.5)
plt.title('SHAP Importance by Token Position - Genome vs Substitution')
plt.xlabel('Nucleotide Position')
plt.ylabel('Average Importance')
plt.xlim([1, SEQ_LENGTH])
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'genome_vs_substitution.png'))
plt.close()

print("SHAP importance plots generated.")
