import os
import csv


def fasta_to_csv(fasta_path):
    output_path = os.path.splitext(fasta_path)[0] + '.csv'
    with open(fasta_path, 'r') as f_in, open(output_path, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['ID', 'length', 'seq'])

        current_id = None
        current_seq = []

        for line in f_in:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    full_seq = ''.join(current_seq)
                    writer.writerow([current_id, len(full_seq), full_seq])
                current_id = line.split()[0][1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)

        # Write last sequence
        if current_id:
            full_seq = ''.join(current_seq)
            writer.writerow([current_id, len(full_seq), full_seq])

    print(f"CSV file saved to: {output_path}")

