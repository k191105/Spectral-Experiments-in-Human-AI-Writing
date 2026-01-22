import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os



OUTPUT_DIR = "outputs"
SPECTRAL_GAPS_DIR = os.path.join(OUTPUT_DIR, "spectral_gaps")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")


def main(data, output_folder_main, textType, model):

    output_folder = os.path.join(output_folder_main, textType, model)

    os.makedirs(output_folder, exist_ok=True)
    data_melted = data.melt(
        id_vars=['Line Number'],
        value_vars=['Spectral Gap Human', 'Spectral Gap AI'],
        var_name='Type',
        value_name='Spectral Gap'
    )

    data_melted['Type'] = data_melted['Type'].map({
        'Spectral Gap Human': 'Human',
        'Spectral Gap AI': 'AI'
    })

    # Visualization 2: Histogram of Spectral Gaps
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=data_melted,
        x='Spectral Gap',
        hue='Type',
        kde=True,
        bins=20,
        palette='viridis',
        alpha=0.5
    )
    plt.title('Distribution of Spectral Gaps for Human and AI Texts')
    plt.xlabel('Spectral Gap')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/histogram_spectral_gaps.png")
    plt.close()

    # Visualization 3: Boxplot of Spectral Gaps
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x='Type',
        y='Spectral Gap',
        data=data_melted,
        palette='viridis'
    )
    plt.title('Boxplot of Spectral Gaps for Human and AI Texts')
    plt.xlabel('Text Type')
    plt.ylabel('Spectral Gap')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/boxplot_spectral_gaps.png")
    plt.close()


    # Visualization 6: Density Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=data_melted,
        x='Spectral Gap',
        hue='Type',
        fill=True,
        common_norm=False,
        palette='viridis',
        alpha=0.5
    )
    plt.title('Density Plot of Spectral Gaps for Human and AI Texts')
    plt.xlabel('Spectral Gap')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/densityplot_spectral_gaps.png")
    plt.close()


    # Visualization 9: Average Spectral Gap Comparison
    avg_spectral_gap_human = data['Spectral Gap Human'].mean()
    avg_spectral_gap_ai = data['Spectral Gap AI'].mean()

    avg_data = pd.DataFrame({
        'Text Type': ['Human', 'AI'],
        'Average Spectral Gap': [avg_spectral_gap_human, avg_spectral_gap_ai]
    })

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x='Text Type',
        y='Average Spectral Gap',
        data=avg_data,
        palette='viridis'
    )
    plt.title('Average Spectral Gap for Human and AI Texts')
    plt.xlabel('Text Type')
    plt.ylabel('Average Spectral Gap')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/average_spectral_gaps.png")
    plt.close()


    # Calculate the number of times spectral gap of human text is greater than AI text
    count_human_gt_ai = sum(data['Spectral Gap Human'] > data['Spectral Gap AI'])
    count_ai_gt_human = sum(data['Spectral Gap AI'] > data['Spectral Gap Human'])

    count_data = pd.DataFrame({
        'Comparison': ['Human > AI', 'AI > Human'],
        'Count': [count_human_gt_ai, count_ai_gt_human]
    })

    print("Counts of Spectral Gap Comparisons:")
    print(count_data)

    # Plot the counts as a bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x='Comparison',
        y='Count',
        data=count_data,
        palette='viridis'
    )
    plt.title('Number of Times Spectral Gap Comparison Holds')
    plt.xlabel('Spectral Gap Comparison')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/counts_comparison_spectral_gaps.png")
    plt.close()

output_folder_main = VISUALIZATION_DIR

for textType in ['acad', 'blog', 'tvm', 'news']:
    for model in ['gpt4o', 'gpt4o-mini', 'llama', 'llama-instruct']:
        data = pd.read_csv(
            os.path.join(SPECTRAL_GAPS_DIR, model, f'Spectral_Gaps_{textType}.csv')
        )
        main(data, output_folder_main, textType, model)


