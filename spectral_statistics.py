import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy.stats import ttest_rel

OUTPUT_DIR = "outputs"
SPECTRAL_GAPS_DIR = os.path.join(OUTPUT_DIR, "spectral_gaps")


textType = 'news'
model = 'gpt4o'
data = pd.read_csv(
    os.path.join(SPECTRAL_GAPS_DIR, model, f'Spectral_Gaps_{textType}.csv')
)


def chi_squared(data):
    count_human_gt_ai = sum(data['Spectral Gap Human'] > data['Spectral Gap AI'])
    count_ai_gt_human = sum(data['Spectral Gap AI'] > data['Spectral Gap Human'])
    total_counts = count_human_gt_ai + count_ai_gt_human

    count_data = pd.DataFrame({
        'Comparison': ['Human > AI', 'AI > Human'],
        'Observed': [count_human_gt_ai, count_ai_gt_human]
    })

    expected_counts = [total_counts / 2, total_counts / 2]

    chi2_statistic, p_value = chisquare(f_obs=count_data['Observed'], f_exp=expected_counts)

    print("Counts of Spectral Gap Comparisons")
    print(count_data)
    print(f"\nChi-Squared Statistic: {chi2_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x='Comparison',
        y='Observed',
        data=count_data,
        palette='viridis'
    )
    plt.title('Number of Times Spectral Gap Comparison Holds (Excluding Ties)')
    plt.xlabel('Spectral Gap Comparison')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def paired_t_test(data):
    spectralGap_human = data['Spectral Gap Human']
    spectralGap_ai = data['Spectral Gap AI']

    t_statistic, p_value = ttest_rel(spectralGap_human, spectralGap_ai)

    print(f"Paired T-Test Statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(spectralGap_human, label='Spectral Gap Human', color='blue', alpha=0.7)
    plt.plot(spectralGap_ai, label='Spectral Gap AI', color='red', alpha=0.7)
    plt.title('Spectral Gap Values: Human vs AI')
    plt.xlabel('Instance')
    plt.ylabel('Spectral Gap')
    plt.legend()
    plt.tight_layout()
    plt.show()


chi_squared(data)
paired_t_test(data)