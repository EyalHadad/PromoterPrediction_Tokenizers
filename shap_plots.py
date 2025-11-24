import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up directories
BASE_DIR = 'shap_plots'
OUTPUT_DIR = os.path.join(BASE_DIR, 'shap_importance_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
NUM_TOKENS = 100
SEQ_LENGTH = 600
METHODS = ['Substitution', 'Genome']


# Helper function
def aggregate_importance_by_position(files):
    position_values = {i: [] for i in range(NUM_TOKENS)}
    token_tracker = {i: [] for i in range(NUM_TOKENS)}
    for f in files:
        df = pd.read_csv(os.path.join(BASE_DIR, f)).head(NUM_TOKENS)
        for i, row in df.iterrows():
            position_values[i].append(row['importance'])
            token_tracker[i].append((row['Token_id'], row['importance']))
    avg_importance = [sum(position_values[i])/len(position_values[i]) for i in range(NUM_TOKENS)]
    top_tokens = {i: max(token_tracker[i], key=lambda x: abs(x[1]))[0] for i in token_tracker if token_tracker[i]}
    return avg_importance, top_tokens

# 2 for bpe\wpc. 4 for non_overlapping
def plot_organism_comparison(method):
    plt.figure(figsize=(12, 6))

    organisms = sorted(set([f.split('_')[2] for f in os.listdir(BASE_DIR) if f.startswith(method)]))

    for organism in organisms:
        files = [f for f in os.listdir(BASE_DIR) if f.startswith(method) and f"_{organism}_" in f]
        all_positions = []
        all_importances = []

        for file in files:
            df = pd.read_csv(os.path.join(BASE_DIR, file))

            nucleotide_importances = [[] for _ in range(SEQ_LENGTH)]
            current_pos = 0
            for _, row in df.iterrows():
                token = str(row['Token_id'])
                importance = row['importance']
                token_length = len(token)
                for i in range(token_length):
                    if current_pos + i < SEQ_LENGTH:
                        nucleotide_importances[current_pos + i].append(importance)
                current_pos += token_length

        # Average importance at each nucleotide
        avg_importances = []
        for imp_list in nucleotide_importances:
            if imp_list:
                avg_importances.append(np.mean(imp_list))
            else:
                avg_importances.append(0)

        plt.plot(range(1, SEQ_LENGTH + 1), avg_importances, label=organism, linewidth=2.5)

    plt.title(f'SHAP Importance by Token Position - {method}')
    plt.xlabel('Nucleotide Position')
    plt.ylabel('Average Importance')
    plt.xlim([1, SEQ_LENGTH])
    plt.ylim([-0.5, 0.5])
    plt.xticks(np.arange(0, SEQ_LENGTH + 1, 50))
    plt.legend(loc='lower left')

    for x in range(25, SEQ_LENGTH, 25):
        plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{method.lower()}_organism_comparison.png'))
    plt.close()


def plot_genome_vs_substitution():
    for method in METHODS:
        files = [f for f in os.listdir(BASE_DIR) if f.startswith(method)]
        position_values = {i: [] for i in range(NUM_TOKENS)}

        for f in files:
            df = pd.read_csv(os.path.join(BASE_DIR, f)).head(NUM_TOKENS)
            for i, val in enumerate(df['importance'].tolist()):
                position_values[i].append(val)

        records = []
        max_points = []
        for pos in range(NUM_TOKENS):
            values = position_values[pos]
            value_counts = pd.Series(values).value_counts()
            max_val = value_counts.idxmax()
            for val in values:
                is_max = val == max_val
                records.append({
                    'Position': pos,
                    'Importance': val,
                    'Highlight': is_max
                })

        df_plot = pd.DataFrame(records)

        plt.figure(figsize=(14, 6))
        # Plot regular points
        sns.scatterplot(
            data=df_plot[df_plot['Highlight'] == False],
            x='Position',
            y='Importance',
            color='gray',
            alpha=0.4,
            s=20,
            edgecolor=None
        )
        # Plot highlighted points
        sns.scatterplot(
            data=df_plot[df_plot['Highlight'] == True],
            x='Position',
            y='Importance',
            color='red',
            s=80,
            edgecolor='black',
            linewidth=0.5,
            zorder=3
        )
        plt.title(f'SHAP Importance Distribution by Position - {method}')
        plt.xlabel('Token Position Index')
        plt.ylabel('SHAP Importance')
        plt.xticks([])
        plt.ylim([-1, 2])
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{method.lower()}_importance_scatter.png'))
        plt.close()

def plot_individual_top_sequences():
    individual_files = [
        'Genome_kmer_non_overlapping_human_finetuned_seq1.csv',
        'Genome_kmer_non_overlapping_human_finetuned_seq2.csv',
        'Genome_kmer_non_overlapping_human_finetuned_seq3.csv',
        'Genome_kmer_non_overlapping_human_finetuned_seq4.csv',
        'Genome_kmer_non_overlapping_human_finetuned_seq5.csv',
        'Substitution_kmer_non_overlapping_human_finetuned_seq6.csv',
        'Substitution_kmer_non_overlapping_human_finetuned_seq2.csv',
        'Substitution_kmer_non_overlapping_human_finetuned_seq3.csv',
        'Substitution_kmer_non_overlapping_human_finetuned_seq4.csv',
        'Substitution_kmer_non_overlapping_human_finetuned_seq5.csv'
    ]

    all_y_vals = []
    data_by_file = {}

    for fname in individual_files:
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            continue

        df = pd.read_csv(fpath).head(NUM_TOKENS)
        y_vals = df['importance'].tolist()
        all_y_vals.extend(y_vals)
        data_by_file[fname] = {
            'x_vals': list(range(1, SEQ_LENGTH + 1, 6)),
            'y_vals': y_vals,
            'tokens': df['Token_id'].tolist()
        }

    global_min = min(all_y_vals)
    global_max = max(all_y_vals)

    for fname, data in data_by_file.items():
        x_vals = data['x_vals']
        y_vals = data['y_vals']
        tokens = data['tokens']

        sorted_idxs = sorted(range(len(y_vals)), key=lambda i: abs(y_vals[i]), reverse=True)
        selected = []
        for idx in sorted_idxs:
            nucleotide_pos = x_vals[idx]
            if nucleotide_pos > 20 and all(abs(nucleotide_pos - x_vals[sel]) >= 25 for sel in selected):
                selected.append(idx)
            if len(selected) == 3:
                break

        annotation_text = "\n".join([f"{tokens[i]},{x_vals[i]}" for i in selected])

        plt.figure(figsize=(12, 6))
        plt.plot(x_vals, y_vals, linewidth=2.5)
        for idx in selected:
            plt.scatter(x_vals[idx], y_vals[idx], color='red', s=80, zorder=5)

        plt.figtext(1.02, 0.5, annotation_text, ha="left", va="center", fontsize=10, color="red", weight="bold")
        plt.title(fname.replace('.csv', ''))
        plt.xlabel('Nucleotide Position')
        plt.ylabel('Token Importance')
        plt.xlim([1, SEQ_LENGTH])
        plt.ylim([global_min, global_max])
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        outname = fname.replace('.csv', '_annotated.png')
        plt.savefig(os.path.join(OUTPUT_DIR, outname), bbox_inches='tight')
        plt.close()


# argument 64 is excel 66
def analyze_token_nucleotide_distribution(position_to_check):
    files = [f for f in os.listdir(BASE_DIR) if f.endswith('.csv') and 'mer' in f]

    position_counts = [{nuc: 0 for nuc in 'ACGT'} for _ in range(6)]
    total_counts = [0 for _ in range(6)]
    token_counter = {}

    for file in files:
        path = os.path.join(BASE_DIR, file)
        df = pd.read_csv(path)
        if len(df) <= position_to_check:
            continue
        token = str(df.iloc[position_to_check]['Token_id'])
        if len(token) != 6:
            continue

        # Count nucleotide distribution
        for i, nuc in enumerate(token):
            if nuc in 'ACGT':
                position_counts[i][nuc] += 1
                total_counts[i] += 1

        # Count token occurrences
        if token not in token_counter:
            token_counter[token] = 0
        token_counter[token] += 1

    # --- Plot 1: Nucleotide distribution ---
    plot_data = []
    for i in range(6):
        for nuc in 'ACGT':
            percentage = (position_counts[i][nuc] / total_counts[i]) * 100 if total_counts[i] else 0
            plot_data.append({'Position': i + 1, 'Nucleotide': nuc, 'Percentage': percentage})

    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='Position', y='Percentage', hue='Nucleotide')
    for i in range(len(df_plot)):
        row = df_plot.iloc[i]
        plt.text(row['Position'] - 1 + (i % 4) * 0.2 - 0.3, row['Percentage'] + 1, f"{row['Percentage']:.1f}%",
                 color='black', ha='center', va='bottom', fontsize=8)
    plt.title(f'Nucleotide Distribution at {position_to_check+1}th Token (6-mers)')
    plt.xlabel('Token Nucleotide Position')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)
    plt.legend(title='Nucleotide')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'nucleotide_distribution_token_{position_to_check+1}.png'))
    plt.close()

    # --- Plot 2: Top 10 most common tokens ---
    if token_counter:
        top_tokens = sorted(token_counter.items(), key=lambda x: (-x[1], x[0]))[:10]
        tokens, counts = zip(*top_tokens)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(tokens), y=list(counts), color='skyblue')
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(count), color='black', ha='center', va='bottom', fontsize=9)
        plt.title(f'Top 10 Most Common Tokens at {position_to_check+1}th Token')
        plt.xlabel('Token')
        plt.ylabel('Count')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'top_tokens_token_{position_to_check+1}.png'))
        plt.close()



def main():
    # plot_organism_comparison('Genome')
    # plot_organism_comparison('Substitution')
    # plot_genome_vs_substitution()
    plot_individual_top_sequences()
    # analyze_token_nucleotide_distribution(63)
    print("SHAP importance plots generated.")

if __name__ == '__main__':
    main()
