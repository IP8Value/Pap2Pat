In this document, detailed hyperparameters are provided for various models used in active learning experiments. For ULMFiT (ULTRA-LM Fine-tuning), the document specifies default settings with Adam optimizer and dropout rates tailored for both language model and classifier stages. This setup ensures reproducibility and consistency across different datasets.

The Multinomial Naive Bayes (MNB) model uses scikit-learn's implementation with default hyperparameters, including a smoothing parameter α = 1.0, fit prior set to True, and class prior set to None. The TFIDF Vectorizer is configured with a maximum feature threshold of 50,000, removal of English stop words, and sublinear tf set to True.

For the SVM model, ThunderSVM is utilized for its GPU acceleration capabilities. The SVC configuration includes a linear kernel, auto gamma, C = 1.0, tolerance of 0.001, and other default settings. This setup ensures efficient computation of support vectors on large datasets.

The document provides comprehensive results for class bias experiments across different iterations using the entropy query strategy. These results are tabulated in Table 15, showing how class distribution evolves over multiple active learning cycles, which is crucial for understanding model behavior and dataset characteristics.

Several metrics measuring uncertainty are provided in Table 13, including negative log-likelihood (NLL), Brier Score Loss (BrierL), expected calibration error (ECE), variation ratio (VarR), entropy (ENT), and standard deviation (STD). These metrics help assess the model's confidence and calibration across different datasets after multiple queries.

Accuracy plots for FastText and NaiveBayes models are presented in Figure 3, showing performance over 4, 9, 19, and 39 iterations using the entropy query strategy compared to random sampling. These plots provide insights into how active learning strategies improve model accuracy more efficiently than random selection.