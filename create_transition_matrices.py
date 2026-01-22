import pandas as pd
import numpy as np
import spacy
import os

nlp = spacy.load('en_core_web_sm')

# List of POS tags to consider
pos_tags_list = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ',
    'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'SCONJ',
    'SYM', 'VERB', 'PUNCT', 'X', 'SPACE'
]

pos_to_idx = {pos: idx for idx, pos in enumerate(pos_tags_list)}
num_pos_tags = len(pos_tags_list)

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
TRANSITION_DIR = os.path.join(OUTPUT_DIR, "transition_matrices")

def get_pos_tags(text):
    """Extract POS tags from the text using spaCy."""
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    return pos_tags

def build_transition_matrix(pos_tags, pos_to_idx, num_pos_tags):
    """Build a transition matrix from the list of POS tags."""
    transition_matrix = np.zeros((num_pos_tags, num_pos_tags))
    for i in range(len(pos_tags) - 1):
        current_pos = pos_tags[i]
        next_pos = pos_tags[i + 1]
        if current_pos in pos_to_idx and next_pos in pos_to_idx:
            current_idx = pos_to_idx[current_pos]
            next_idx = pos_to_idx[next_pos]
            transition_matrix[current_idx, next_idx] += 1
    return transition_matrix

def print_matrix(matrix):
    """Print the matrix in a formatted way."""
    for row in matrix:
        print("[", "  ".join(f"{elem:.3f}" for elem in row), "]")

def main(text_type, N='full', model='gpt4o'):

    df_human = pd.read_parquet(os.path.join(DATA_DIR, 'hape-text_human-chunk-2.parquet'))
    
    if model == 'gpt4o':
        df_ai = pd.read_parquet(os.path.join(DATA_DIR, 'hape-text_gpt-4o-2024-08-06.parquet'))
    elif model == 'gpt4o-mini':
        df_ai = pd.read_parquet(os.path.join(DATA_DIR, 'hape-text_gpt-4o-mini-2024-07-18.parquet'))
    elif model == 'llama-instruct':
        df_ai = pd.read_parquet(os.path.join(DATA_DIR, 'hape-text_llama-3-70B-Instruct.parquet'))
    elif model == 'llama':
        df_ai = pd.read_parquet(os.path.join(DATA_DIR, 'hape-text_llama-3-70B.parquet'))

    # Filter the datasets based on the text type
    filtered_df_human = df_human[df_human['doc_id'].str.contains(text_type)].copy()
    filtered_df_ai = df_ai[df_ai['doc_id'].str.contains(text_type)].copy()

    # Limit to N documents if specified
    if N != 'full':
        filtered_df_human = filtered_df_human.head(N)
        filtered_df_ai = filtered_df_ai.head(N)

    # Initialize lists to hold all POS tags
    all_pos_tags_human = []
    all_pos_tags_ai = []

    # Process human texts
    for idx, row in filtered_df_human.iterrows():
        text_human = row['text']
        pos_tags_human = get_pos_tags(text_human)
        all_pos_tags_human.extend(pos_tags_human)

    # Build transition matrix for human texts
    transition_matrix_human = build_transition_matrix(
        all_pos_tags_human, pos_to_idx, num_pos_tags
    )

    # Apply Laplace smoothing
    transition_matrix_human += 1 / num_pos_tags

    row_sums_human = transition_matrix_human.sum(axis=1, keepdims=True)
    transition_matrix_human_prob = transition_matrix_human / row_sums_human

    # Print the count matrix and the probability matrix for human texts
    print("Transition matrix counts for human text:")
    print_matrix(transition_matrix_human)
    print("\nTransition matrix probabilities for human text:")
    print_matrix(transition_matrix_human_prob)
    row_sums_human = transition_matrix_human_prob.sum(axis=1, keepdims=True)
    print(row_sums_human)

    os.makedirs(os.path.join(TRANSITION_DIR, 'human', text_type), exist_ok=True)
    np.save(
        os.path.join(TRANSITION_DIR, 'human', text_type, 'human_counts.npy'),
        transition_matrix_human
    )
    np.save(
        os.path.join(TRANSITION_DIR, 'human', text_type, 'human_prob.npy'),
        transition_matrix_human_prob
    )

    for idx, row in filtered_df_ai.iterrows():
        text_ai = row['text']
        pos_tags_ai = get_pos_tags(text_ai)
        all_pos_tags_ai.extend(pos_tags_ai)

    transition_matrix_ai = build_transition_matrix(
        all_pos_tags_ai, pos_to_idx, num_pos_tags
    )

    # Apply Laplace smoothing
    transition_matrix_ai += 1 / num_pos_tags

    # Compute the probability matrix
    row_sums_ai = transition_matrix_ai.sum(axis=1, keepdims=True)
    transition_matrix_ai_prob = transition_matrix_ai / row_sums_ai

    # Print the count matrix and the probability matrix for AI texts
    print("\nTransition matrix counts for AI text:")
    print_matrix(transition_matrix_ai)
    print("\nTransition matrix probabilities for AI text:")
    print_matrix(transition_matrix_ai_prob)
    row_sums_ai = transition_matrix_ai_prob.sum(axis=1, keepdims=True)
    print(row_sums_ai)

    os.makedirs(os.path.join(TRANSITION_DIR, model, text_type), exist_ok=True)
    np.save(
        os.path.join(TRANSITION_DIR, model, text_type, 'ai_counts.npy'),
        transition_matrix_ai
    )
    np.save(
        os.path.join(TRANSITION_DIR, model, text_type, 'ai_prob.npy'),
        transition_matrix_ai_prob
    )


if __name__ == "__main__":
    # Example usage:
    # for text_type in ['news', 'tvm']:
    #     main(text_type, N=750, model='gpt4o-mini')
    pass

