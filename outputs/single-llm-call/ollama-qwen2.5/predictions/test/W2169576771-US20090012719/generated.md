- The study aimed to identify synergistic gene pairs in prostate cancer using an information-theoretic approach on a microarray dataset of 102 samples (52 tumors, 50 healthy). Researchers developed algorithms to evaluate entropy and synergy for all possible gene pairs.

- They used the Robust Multi-array Average method to normalize raw probe data from Affymetrix Human Genome U95Av2 microarrays. Two types of permutation tests were implemented: permuting sample class labels or shuffling each gene's expression independently within classes.

- The highest-synergy pair (RBP1 with EEF1B2) had a P-value < 10^-15 for both permutations. A cut-off threshold was defined using false discovery rate adjustment to control for multiple comparisons, identifying 473 significant pairs under permutation B at α=0.05.

- The algorithm utilized exhaustive search and partitioned the search space across a 200-node computing cluster, completing in about one hour. Software for evaluating entropy and synergy was implemented in MATLAB and made available as supplementary material.

- Authors included biologists and computer scientists who developed algorithms, performed computational implementation, provided statistical validation, and wrote corresponding manuscript sections. All authors reviewed and approved the final version of the manuscript.
