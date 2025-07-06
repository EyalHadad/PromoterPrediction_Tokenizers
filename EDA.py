import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import chi2_contingency

# Constants
DATA_DIR = os.path.join('data', 'Substitution')
PLOT_DIR = 'plots'
RESULTS_DIR = 'results'
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

matplotlib.use('TkAgg')

def plot_train_val_test_proportions(data_dir, plot_dir):
    # Load all CSV files
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    dataframes = [pd.read_csv(f) for f in files]
    data = pd.concat(dataframes, ignore_index=True)

    # Add a column indicating the dataset split (train/val/test) based on filename
    def get_split(filename):
        if filename.startswith('train'):
            return 'train'
        elif filename.startswith('val'):
            return 'val'
        elif filename.startswith('test'):
            return 'test'
        return None

    data['split'] = data['source_file'] = [os.path.basename(f) for f in files for _ in range(len(pd.read_csv(f)))]
    data['split'] = data['source_file'].apply(get_split)

    # Calculate proportions
    def calculate_proportions(df):
        proportions = df['split'].value_counts(normalize=True).to_dict()
        return {split: proportions.get(split, 0) for split in ['train', 'val', 'test']}

    # Calculate proportions for all and each organism
    all_proportions = calculate_proportions(data)
    organism_proportions = data.groupby('organism').apply(calculate_proportions)

    # Prepare data for plotting
    categories = ['all'] + list(organism_proportions.index)
    proportions = [all_proportions] + list(organism_proportions)

    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.5

    train_values = [p['train'] for p in proportions]
    val_values = [p['val'] for p in proportions]
    test_values = [p['test'] for p in proportions]

    train_bottom = [0] * len(categories)
    val_bottom = train_values
    test_bottom = [t + v for t, v in zip(train_values, val_values)]

    bars_train = ax.bar(categories, train_values, bar_width, label='Train', color='skyblue')
    bars_val = ax.bar(categories, val_values, bar_width, label='Validation', color='orange', bottom=val_bottom)
    bars_test = ax.bar(categories, test_values, bar_width, label='Test', color='lightgreen', bottom=test_bottom)

    # Add percentage labels inside bars
    for bars, values, bottoms in zip([bars_train, bars_val, bars_test], [train_values, val_values, test_values], [train_bottom, val_bottom, test_bottom]):
        for bar, value, bottom in zip(bars, values, bottoms):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + height / 2,
                    f"{value * 100:.1f}%",
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='black'
                )

    # Add labels and legend
    ax.set_title('Proportion of Train-Val-Test by Organism')
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Organism')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(plot_dir, 'train_val_test_proportions.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")

    # Chi-squared test
    chi2_results = []
    for i, (cat1, prop1) in enumerate(zip(categories, proportions)):
        for j, (cat2, prop2) in enumerate(zip(categories, proportions)):
            if i >= j:  # Skip redundant comparisons
                continue

            observed = [prop1['train'], prop1['val'], prop1['test']]
            expected = [prop2['train'], prop2['val'], prop2['test']]
            chi2, p, _, _ = chi2_contingency([observed, expected])

            chi2_results.append({
                'Comparison': f"{cat1} vs {cat2}",
                'Chi2': chi2,
                'P-Value': p
            })

    # Save Chi-squared results
    chi2_output_path = os.path.join(RESULTS_DIR, 'chi_squared_pairwise_results.csv')
    pd.DataFrame(chi2_results).to_csv(chi2_output_path, index=False)
    print(f"Chi-squared pairwise results saved to {chi2_output_path}")

# Call the function
plot_train_val_test_proportions(DATA_DIR, PLOT_DIR)
