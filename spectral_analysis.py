import pandas as pd
import numpy as np
import spacy
import os

nlp = spacy.load('en_core_web_sm')

# taken from spacy documentation
pos_tags_list = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ',
    'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'SCONJ',
    'SYM', 'VERB', 'PUNCT', 'X', 'SPACE'
]

# label each pos
pos_to_idx = {pos: idx for idx, pos in enumerate(pos_tags_list)}

print(pos_to_idx)
num_pos_tags = len(pos_tags_list)

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
SPECTRAL_GAPS_DIR = os.path.join(OUTPUT_DIR, "spectral_gaps")

def get_pos_tags(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    return pos_tags

def build_transition_matrix(pos_tags, pos_to_idx, num_pos_tags):
    # print(len(pos_tags))
    # build numPosTags by numPosTags matrix
    transition_matrix = np.zeros((num_pos_tags, num_pos_tags))

    for i in range(len(pos_tags) - 1):
        current_pos = pos_tags[i]
        next_pos = pos_tags[i + 1]
        if (current_pos in pos_to_idx) and (next_pos in pos_to_idx):
            current_idx = pos_to_idx[current_pos]
            next_idx = pos_to_idx[next_pos]
            transition_matrix[current_idx, next_idx] += 1

    print("Initial transition matrix without probabilities")
    print_matrix(transition_matrix)

    transition_matrix = transition_matrix + 1/len(pos_tags_list)
    print("Initial transition matrix without probabilities after Laplace Smoothing")
    print_matrix(transition_matrix)

    print(len(pos_tags))
    print(np.sum(transition_matrix))
    row_sums = transition_matrix.sum(axis=1, keepdims=True)


    print(row_sums)
    # making sure that we never divide by 0
    row_sums[row_sums == 0] = 1


    transition_matrix = transition_matrix / row_sums

    # print("transition matrix \n", transition_matrix, "\n")
    return transition_matrix

def compute_spectral_gap(transition_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # print(eigenvalues)
    # print(eigenvectors)

    spectral_gap = np.abs(eigenvalues[0]) - np.abs(eigenvalues[1])
    second_eigenvector = eigenvectors[:, 1].real
    return spectral_gap, second_eigenvector.tolist()

def extract_common_id(doc_id):
    return doc_id.split('@')[0]  

def main(text_type, N, ai_model):
    
    df_human = pd.read_parquet(os.path.join(DATA_DIR, 'hape-text_human-chunk-2.parquet'))

    if ai_model == 'gpt4o':
        df_ai = pd.read_parquet(os.path.join(DATA_DIR, 'hape-text_gpt-4o-2024-08-06.parquet'))
    elif ai_model == 'llama-instruct':
        df_ai = pd.read_parquet(os.path.join(DATA_DIR, 'hape-text_llama-3-70B-Instruct.parquet'))
    elif ai_model == 'llama':
        df_ai = pd.read_parquet(os.path.join(DATA_DIR, 'hape-text_llama-3-70B.parquet'))
    elif ai_model == 'gpt4o-mini':
        df_ai = pd.read_parquet(os.path.join(DATA_DIR, 'hape-text_gpt-4o-mini-2024-07-18.parquet'))

    filtered_df_human = df_human[df_human['doc_id'].str.contains(text_type)].copy()
    filtered_df_ai = df_ai[df_ai['doc_id'].str.contains(text_type)].copy()

    filtered_df_human['common_id'] = filtered_df_human['doc_id'].apply(extract_common_id)
    filtered_df_ai['common_id'] = filtered_df_ai['doc_id'].apply(extract_common_id)

    # merge the datasets for direct comparison
    merged_df = pd.merge(
        filtered_df_human[['common_id', 'text']],
        filtered_df_ai[['common_id', 'text']],
        on='common_id',
        suffixes=('_human', '_ai')
    )
    
    if N != 'full':
        merged_df = merged_df.head(N)
    
    print(merged_df.shape)

    spectral_gaps_human = []
    spectral_gaps_ai = []
    line_numbers = []

    for idx, row in merged_df.iterrows():
        line_number = idx
        # Human stuff
        text_human = row['text_human']
        pos_tags_human = get_pos_tags(text_human)
        transition_matrix_human = build_transition_matrix(pos_tags_human, pos_to_idx, num_pos_tags)
        spectral_gap_human = compute_spectral_gap(transition_matrix_human)
        spectral_gaps_human.append(spectral_gap_human)

        # AI stuff
        text_ai = row['text_ai']
        pos_tags_ai = get_pos_tags(text_ai)
        transition_matrix_ai = build_transition_matrix(pos_tags_ai, pos_to_idx, num_pos_tags)
        spectral_gap_ai = compute_spectral_gap(transition_matrix_ai)
        spectral_gaps_ai.append(spectral_gap_ai)

        line_numbers.append(line_number)

    data = pd.DataFrame({
        'Line Number': line_numbers,
        'Spectral Gap Human': spectral_gaps_human,
        'Spectral Gap AI': spectral_gaps_ai
    })


    os.makedirs(os.path.join(SPECTRAL_GAPS_DIR, ai_model), exist_ok=True)
    data.to_csv(
        os.path.join(SPECTRAL_GAPS_DIR, ai_model, f'Spectral_Gaps_{text_type}.csv'),
        index=False
    )

def print_matrix(matrix):
    for row in matrix:
        print("[", "  ".join(f"{elem:.3f}" for elem in row), "]")

N = 750 # adjust for more rows or full for full data set

# This can be slow on the full dataset; adjust N for quick iteration.
for text_type in ['acad', 'blog', 'tvm', 'news']:
    for model in ['gpt4o', 'gpt4o-mini', 'llama', 'llama-instruct']:
        ai_model = model
        main(text_type, N, ai_model) 
