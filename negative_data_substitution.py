import pandas as pd
import random
from pathlib import Path


def exchange_segments(dna_sequence, segment_length=24, num_exchanges=8):
    """
    Randomly exchanges and shuffles segments within a DNA sequence.

    Args:
        dna_sequence (str): Original DNA sequence.
        segment_length (int): Length of each segment.
        num_exchanges (int): Number of segments to exchange and shuffle.

    Returns:
        str: Modified DNA sequence.
    """
    segments = [dna_sequence[i:i + segment_length] for i in range(0, len(dna_sequence), segment_length)]
    if len(segments) < num_exchanges:
        raise ValueError("Not enough segments to perform the exchange.")

    # Select and shuffle segments
    indices = random.sample(range(len(segments)), num_exchanges)
    shuffled_segments = [segments[i] for i in indices]
    random.shuffle(shuffled_segments)

    for i, idx in enumerate(indices):
        segments[idx] = ''.join(random.sample(shuffled_segments[i], len(shuffled_segments[i])))

    return ''.join(segments)


def generate_negative_data(df, output_path, segment_length=24, num_exchanges=8):
    """
    Generates a negative dataset by modifying sequences.

    Args:
        df (pd.DataFrame): Input DataFrame with 'ID', 'seq', 'organism'.
        output_path (str or Path): Where to save the new CSV file.
    """
    output_data = []

    for _, row in df.iterrows():
        new_seq = exchange_segments(row['seq'], segment_length, num_exchanges)
        output_data.append({
            'ID': row['ID'],
            'lengths': 600,
            'seq': new_seq,
            'organism': row['organism']
        })

    pd.DataFrame(output_data).to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

