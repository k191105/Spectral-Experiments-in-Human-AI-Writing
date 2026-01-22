import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# List of POS tags to consider
pos_tags_list = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ',
    'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'SCONJ',
    'SYM', 'VERB', 'PUNCT', 'X', 'SPACE'
]

def print_matrix(matrix):
    """Print the matrix in a formatted way."""
    for row in matrix:
        print("[", "  ".join(f"{elem:.3f}" for elem in row), "]")

def print_vector(vector):
    # Print the vector elements with 3 decimal precision
    print("[", "  ".join(f"{elem:.3f}" for elem in vector), "]")

def load_matrices(model, text_type):
    transition_matrix_human_counts = np.load(
        os.path.join(TRANSITION_DIR, 'human', text_type, 'human_counts.npy')
    )
    transition_matrix_human_prob = np.load(
        os.path.join(TRANSITION_DIR, 'human', text_type, 'human_prob.npy')
    )

    transition_matrix_ai_counts = np.load(
        os.path.join(TRANSITION_DIR, model, text_type, 'ai_counts.npy')
    )
    transition_matrix_ai_prob = np.load(
        os.path.join(TRANSITION_DIR, model, text_type, 'ai_prob.npy')
    )

    return transition_matrix_human_counts, transition_matrix_human_prob, transition_matrix_ai_counts, transition_matrix_ai_prob

def showloading():
    print("Loaded transition matrix counts for human text:")
    print_matrix(transition_matrix_human_counts)

    print("\nLoaded transition matrix probabilities for human text:")
    print_matrix(transition_matrix_human_prob)

    # Similarly for AI texts
    print("\nLoaded transition matrix counts for AI text:")
    print_matrix(transition_matrix_ai_counts)

    print("\nLoaded transition matrix probabilities for AI text:")
    print_matrix(transition_matrix_ai_prob)

def power_iteration(matrix, epsilon=1e-6, max_iterations=1000):
    """Compute PageRank using the power iteration method."""
    n = matrix.shape[0]
    pagerank = np.ones(n) / n  # Initial uniform distribution

    for _ in range(max_iterations):
        new_pagerank = np.dot(matrix.T, pagerank)
        if np.linalg.norm(new_pagerank - pagerank, 1) < epsilon:
            break
        pagerank = new_pagerank

    return pagerank

def stationary_distribution(transition_matrix):
    """Compute the stationary distribution of a Markov chain."""
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    # Find the eigenvector corresponding to eigenvalue 1
    stationary_vector = eigenvectors[:, np.isclose(eigenvalues, 1)]
    # Normalize the vector
    stationary_vector = stationary_vector[:, 0]
    stationary_distribution = stationary_vector / stationary_vector.sum()
    stationary_distribution = stationary_distribution.real  # In case of complex numbers
    return stationary_distribution

def plot_vector_heatmap(vector, labels, title='Vector Heatmap', cmap='coolwarm', ax=None):

    # Ensure vector is a numpy array
    vector = np.array(vector)

    # Convert vector to a DataFrame for better labeling
    df = pd.DataFrame(vector.reshape(-1, 1), index=labels, columns=['Value'])

    # Create the heatmap
    if ax is None:
        plt.figure(figsize=(2, len(labels) / 2))
        ax = plt.gca()
    sns.heatmap(df, annot=True, fmt=".3f", cmap=cmap, cbar=True, linewidths=0.5, ax=ax)
    ax.set_title(title)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels([''], rotation=0)  # Hide x-axis label (since it's just 'Value')
    plt.tight_layout()

def plot_three_heatmaps(vector1, vector2, difference_vector, labels, titles, cmaps, output_folder_main, textType, model):
    output_folder = os.path.join(output_folder_main, textType)

    os.makedirs(output_folder, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, len(labels) / 2 + 1))
    plt.subplots_adjust(wspace=0.5)

    # Plot the first vector
    plot_vector_heatmap(
        vector=vector1,
        labels=labels,
        title=titles[0],
        cmap=cmaps[0],
        ax=axes[0]
    )

    plot_vector_heatmap(
        vector=vector2,
        labels=labels,
        title=titles[1],
        cmap=cmaps[1],
        ax=axes[1]
    )

    plot_vector_heatmap(
        vector=difference_vector,
        labels=labels,
        title=titles[2],
        cmap=cmaps[2],
        ax=axes[2]
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{model}_vs_human.png"))
    plt.close()



OUTPUT_DIR = "outputs"
TRANSITION_DIR = os.path.join(OUTPUT_DIR, "transition_matrices")
STATIONARY_DIR = os.path.join(OUTPUT_DIR, "stationary_distributions")

output_folder_main = STATIONARY_DIR

for textType in ['acad', 'blog', 'tvm', 'news']:
    for model in ['gpt4o', 'gpt4o-mini', 'llama', 'llama-instruct']:

        transition_matrix_human_counts = load_matrices(model = model, text_type = textType)[0]
        transition_matrix_human_prob = load_matrices(model = model, text_type = textType)[1]
        transition_matrix_ai_counts = load_matrices(model = model, text_type = textType)[2]
        transition_matrix_ai_prob = load_matrices(model = model, text_type = textType)[3]


        human_stationary = stationary_distribution(transition_matrix_human_prob)
        ai_stationary = stationary_distribution(transition_matrix_ai_prob)
        difference_stationary = abs(human_stationary - ai_stationary)


        titles = ['Human Stationary Distribution/PageRank', 'AI Stationary Distribution/PageRank', 'Difference PageRank abs(Human - AI)']
        cmaps = ['viridis', 'viridis', 'bwr']

        plot_three_heatmaps(
            vector1=human_stationary,
            vector2=ai_stationary,
            difference_vector=difference_stationary,
            labels=pos_tags_list,
            titles=titles,
            cmaps=cmaps,
            output_folder_main = output_folder_main,
            textType = textType,
            model=model
        )
