This project studies whether AI-written text exhibits different syntactic dynamics from human text by treating part-of-speech sequences as Markov chains. It uses spaCy to tag text, builds POS transition matrices with Laplace smoothing, and compares human and model-generated corpora across multiple genres (academic, blog, news, and TV/movie). The core signal is the spectral gap of the transition matrix, which is derived from eigenvalues to quantify how quickly a chain mixes; the repo also computes stationary distributions to visualize long-run POS behavior and runs paired statistical tests to compare human versus AI outputs.

## File overview

Generated with Claude in Cursor:

The pipeline is organized around reproducible data, artifacts, and plots. Raw parquet datasets live in `data/`, derived matrices and CSVs are written to `outputs/transition_matrices/` and `outputs/spectral_gaps/`, and visual summaries (histograms, boxplots, density plots, and aggregate comparisons) are saved under `outputs/visualizations/` and `outputs/stationary_distributions/`. Running `create_transition_matrices.py` builds the POS transition matrices, `spectral_analysis.py` computes spectral gaps for human/AI pairs, `spectral_visualisation.py` generates charts, `stationarydistributions.py` plots stationary distributions, and `spectral_statistics.py` performs chi-square and paired t-tests for a selected model and genre.

Key ideas:
- POS tagging and sequence modeling with spaCy and pandas over large text corpora.
- Markov transition matrix construction, Laplace smoothing, and eigenvalue-based spectral gap analysis using NumPy.
- Stationary distribution and PageRank-style power iteration intuition for linguistic dynamics.
- Statistical testing (paired t-test, chi-square) to validate differences between human and AI text.
- Visualization of distributions and comparative summaries with seaborn and matplotlib.

The project is designed to be extensible: you can swap in new models or genres by adding data files, and the analysis scripts will generate comparable outputs in a consistent directory structure.
