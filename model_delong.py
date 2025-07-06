import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

# Constants
RESULTS_DIR = 'results'
DELONG_OUTPUT = 'delong_comparisons'
PLOTS_DIR = 'delong_comparisons/plots'
METHODS = ['Genome', 'Substitution']

# Ensure output directories exist
os.makedirs(DELONG_OUTPUT, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# DeLong test implementation
def delong_roc_test(pred1, pred2, true_labels):
    auc1 = roc_auc_score(true_labels, pred1)
    auc2 = roc_auc_score(true_labels, pred2)

    var1 = auc1 * (1 - auc1) / len(true_labels)
    var2 = auc2 * (1 - auc2) / len(true_labels)
    variance = var1 + var2

    if variance == 0:
        print(f"Warning: Variance is zero for AUC1={auc1}, AUC2={auc2}")
        return float('nan'), float('nan'), float('nan')

    z = (auc1 - auc2) / np.sqrt(variance)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    log_p_value = -np.log10(p_value) if p_value > 0 else 20  # Cap at 20
    return auc1, auc2, z, p_value, log_p_value

# Process each method
all_comparison_results = []
organisms_list = set()

for method in METHODS:
    prediction_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith(method)]
    comparison_results = []

    # Extract unique organisms
    organisms = set(f.split('_')[2] for f in prediction_files)
    organisms_list.update(organisms)
    models_set = set()

    for organism in organisms:
        organism_files = [f for f in prediction_files if f.split('_')[2] == organism]

        # Load model predictions
        models = {}
        for file in organism_files:
            model_type = file.split('_')[1]
            file_path = os.path.join(RESULTS_DIR, file)
            df = pd.read_csv(file_path)
            models[model_type] = df[['True_label', 'Prediction']]
            models_set.add(model_type)

        model_names = list(models.keys())

        # Perform pairwise DeLong comparisons
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]

                true_labels = models[model1]['True_label'].values
                pred1 = models[model1]['Prediction'].values
                pred2 = models[model2]['Prediction'].values

                auc1, auc2, z, p, log_p = delong_roc_test(pred1, pred2, true_labels)

                if np.isnan(z) or np.isnan(p):
                    print(f"Warning: DeLong test returned NaN for {model1} vs {model2} in {organism} ({method})")

                comparison_results.append({
                    'Organism': organism,
                    'Model 1': model1,
                    'Model 2': model2,
                    'AUC 1': auc1,
                    'AUC 2': auc2,
                    'Z-Score': z,
                    'P-Value': p,
                    '-log10(P-Value)': log_p,
                    'Method': method
                })

    all_comparison_results.extend(comparison_results)
    output_file = os.path.join(DELONG_OUTPUT, f"{method}_delong_comparisons.csv")
    df_results = pd.DataFrame(comparison_results)
    df_results.to_csv(output_file, index=False)
    print(f"DeLong comparison results saved to {output_file}")

# Create a combined 8x2 heatmap figure
fig, axes = plt.subplots(len(organisms_list), 2, figsize=(12, len(organisms_list) * 6))
fig.suptitle("DeLong Comparison Heatmaps for All Organisms", fontsize=16)

for idx, organism in enumerate(sorted(organisms_list)):
    for col_idx, method in enumerate(METHODS):
        sub_df = pd.DataFrame(all_comparison_results)
        sub_df = sub_df[(sub_df['Organism'] == organism) & (sub_df['Method'] == method)]

        if sub_df.empty:
            axes[idx, col_idx].axis('off')
            continue

        heatmap_data = sub_df.pivot(index='Model 1', columns='Model 2', values='-log10(P-Value)')

        sns.heatmap(heatmap_data, ax=axes[idx, col_idx], cmap='coolwarm', annot=False, linewidths=0.5, vmin=0, vmax=20)
        axes[idx, col_idx].set_title(f"{organism} ({method})")
        axes[idx, col_idx].set_xticklabels(axes[idx, col_idx].get_xticklabels(), rotation=45, ha='right')
        axes[idx, col_idx].set_yticklabels(axes[idx, col_idx].get_yticklabels(), rotation=0)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plot_file = os.path.join(PLOTS_DIR, "all_organisms_delong_comparisons.png")
plt.savefig(plot_file)
plt.close()
print(f"DeLong combined heatmap saved to {plot_file}")

print("DeLong comparisons completed for all datasets.")
