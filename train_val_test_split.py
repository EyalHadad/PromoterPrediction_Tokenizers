import os
import pandas as pd

def read_clstr_file(file_path):
    clusters = []
    with open(file_path, 'r') as file:
        current_cluster = []
        for line in file:
            if line.startswith(">Cluster"):
                if current_cluster:
                    clusters.append(current_cluster)
                # the first index of each cluster is the cluster id:
                current_cluster = [line[9:].strip()]
            else:
                current_cluster.append(line.strip())

        if current_cluster:
            clusters.append(current_cluster)

    return clusters

def save_dict_to_csv(data_dict, output_filename):
    if not data_dict:
        print(f"Warning: No data for {output_filename}")
        return

    max_length = max(len(arr) for arr in data_dict.values()) if data_dict else 0
    padded_data = {key: value + [''] * (max_length - len(value)) for key, value in data_dict.items()}

    df = pd.DataFrame(padded_data)
    df.to_csv(output_filename, index=False)
    print(f"Saved {output_filename}")

clusters_dir = "raw_data/cd_hit80"

final_train_ids = {}
final_val_ids = {}
final_test_ids = {}


TEST_RATIO = 0.10
VAL_RATIO = 0.15


print("Processing clusters...")

for filename in os.listdir(clusters_dir):
    if not filename.endswith(".clstr"):
        pass

    clusters_path = os.path.join(clusters_dir, filename)
    org = filename[4:-6]


    clusters = read_clstr_file(clusters_path)
    sorted_clusters = sorted(clusters, key=lambda x: len(x))
    total_sequences = sum(len(c) - 1 for c in sorted_clusters)

    target_test_count = int(total_sequences * TEST_RATIO)
    target_val_count = int(total_sequences * VAL_RATIO)

    org_test_seqs = []
    org_val_seqs = []
    org_train_seqs = []

    current_test_count = 0
    current_val_count = 0

    for cluster in sorted_clusters:
        seq_lines = cluster[1:]
        cluster_size = len(seq_lines)

        ids_in_cluster = [line[-13:-5] for line in seq_lines]


        if current_test_count < target_test_count:
            org_test_seqs.extend(ids_in_cluster)
            current_test_count += cluster_size


        elif current_val_count < target_val_count:
            org_val_seqs.extend(ids_in_cluster)
            current_val_count += cluster_size


        else:
            org_train_seqs.extend(ids_in_cluster)


    final_test_ids[org] = org_test_seqs
    final_val_ids[org] = org_val_seqs
    final_train_ids[org] = org_train_seqs

    print(f"Processed {org}: Total={total_sequences}, Test={len(org_test_seqs)}, Val={len(org_val_seqs)}, Train={len(org_train_seqs)}")


print("\nSaving files")
save_dict_to_csv(final_test_ids, "test_ids.csv")
save_dict_to_csv(final_val_ids, "val_ids.csv")
save_dict_to_csv(final_train_ids, "train_ids.csv")

print("Done.")