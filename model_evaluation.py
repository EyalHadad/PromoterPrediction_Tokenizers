import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

# Constants
RESULTS_DIR = 'results'
EVALUATION_OUTPUT = 'evaluation'
DATA_DIRS = ['Genome', 'Substitution']

# Ensure evaluation directory exists
os.makedirs(EVALUATION_OUTPUT, exist_ok=True)

# Metrics calculation function
def calculate_metrics(true_labels, predictions):
    acc = accuracy_score(true_labels, predictions > 0.5)
    f1 = f1_score(true_labels, predictions > 0.5)
    auc = roc_auc_score(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, predictions > 0.5)

    tn, fp, fn, tp = confusion_matrix(true_labels, predictions > 0.5).ravel()
    tp_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        'ACC': acc,
        'F1': f1,
        'AUC': auc,
        'MCC': mcc,
        'TP Rate': tp_rate,
        'FP Rate': fp_rate
    }

# Main evaluation loop
for data_dir in DATA_DIRS:
    evaluation_results = []

    for result_file in os.listdir(RESULTS_DIR):
        if not result_file.startswith(data_dir.replace(' ', '_')):
            continue

        # Load the predictions
        file_path = os.path.join(RESULTS_DIR, result_file)
        data = pd.read_csv(file_path)

        # Extract labels and predictions
        true_labels = data['True_label']
        predictions = data['Prediction']

        # Calculate metrics
        metrics = calculate_metrics(true_labels, predictions)
        metrics['Model'] = result_file
        evaluation_results.append(metrics)

    # Save evaluation results for the dataset
    output_file = os.path.join(EVALUATION_OUTPUT, f"{data_dir.replace(' ', '_')}_evaluation.csv")
    pd.DataFrame(evaluation_results).to_csv(output_file, index=False)
    print(f"Evaluation results saved to {output_file}")

print("Evaluation completed for all datasets.")
