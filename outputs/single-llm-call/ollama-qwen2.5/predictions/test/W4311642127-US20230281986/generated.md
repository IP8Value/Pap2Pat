- We present zxVAD, a method for cross-domain video anomaly detection (xVAD) that leverages task-irrelevant data when domain VAD training data are available or not. Results show the potential of using such data to address xVAD effectively.

- We analyze the relevancy between TI and target domain VAD data, confirming TI is less relevant. This justifies our approach of using TI data for learning relative normalcy in the absence of VAD-specific data.

- Our experiments demonstrate that zxVAD achieves near SOTA cross-domain performance with as little as 0.5% to 8% of HMDB or UCF101 data, highlighting its efficiency and effectiveness even with limited TI data.

- We evaluate zxVAD's sensitivity to loss ratios in the proposed normalcy classifier. Results show that while varying these ratios impacts performance, all configurations outperform SOTA MPN [2], indicating the robustness of our approach to hyperparameter tuning.

- Additional qualitative results, including pseudo-abnormal frames and difference maps from various datasets, validate zxVAD's ability to detect anomalies accurately in a cross-domain setting. These visualizations provide insights into how zxVAD identifies and localizes anomalies effectively.

- Our method outperforms SOTA unsupervised xVAD methods under the same-dataset setting, demonstrating its effectiveness not only in cross-domain scenarios but also when trained and tested on the same dataset. This further validates the robustness of our approach across different settings.