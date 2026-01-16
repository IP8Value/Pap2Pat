**PATENT APPLICATION**  

**DESCRIPTION**  

### **BACKGROUND**  

The field of anomaly detection encompasses a wide range of applications, including manufacturing defect identification, network security threat detection, and financial fraud analysis. Traditional approaches to anomaly detection can be broadly categorized into three settings: fully supervised, unsupervised, and semi-supervised.  

In the fully supervised setting, labeled data for both normal and anomalous samples are available, enabling the use of specialized techniques such as weighted loss functions or resampling methods to address data imbalance. However, this approach requires extensive manual labeling, which is often impractical due to high costs and time constraints.  

Conversely, unsupervised anomaly detection methods operate without labeled data, relying instead on statistical or machine learning techniques to identify deviations from normal patterns. While these methods eliminate labeling costs, they frequently suffer from reduced accuracy compared to supervised approaches, limiting their applicability in real-world scenarios.  

Semi-supervised anomaly detection seeks to bridge the gap between these extremes by leveraging a limited set of labeled data alongside a larger pool of unlabeled data. However, existing semi-supervised methods typically assume that labeled and unlabeled data originate from the same distribution, an assumption that often fails in practice. For instance, labeled data may contain only certain anomaly types, while unlabeled data includes new or evolving anomalies. Similarly, human labelers may prioritize easy-to-classify samples, leading to distributional discrepancies between labeled and unlabeled datasets.  

Current semi-supervised anomaly detection frameworks, including positive-unlabeled (PU) learning and one-class classifiers (OCCs), struggle to adapt to distribution mismatches. These methods either fail to utilize all available labeled data effectively or produce biased models due to over-reliance on small labeled subsets. Additionally, conventional hyperparameter tuning methods require validation data, further reducing the already limited labeled training set.  

Thus, there exists a critical need for a robust semi-supervised anomaly detection framework capable of handling distribution mismatches while minimizing reliance on labeled data.  

### **BRIEF SUMMARY**  

The present invention introduces **SPADE (Semi-supervised Pseudo-labeler Anomaly Detection with Ensembling)**, a novel framework designed to address the challenges of semi-supervised anomaly detection under distribution mismatch. SPADE integrates multiple innovations to achieve high detection accuracy while reducing dependence on labeled data.  

Key components of SPADE include:  

1. **Pseudo-Labeling via Ensemble OCCs**: Unlike conventional semi-supervised methods that rely on a single binary classifier for pseudo-labeling, SPADE employs an ensemble of one-class classifiers (OCCs) to generate robust pseudo-labels. This approach minimizes bias from small labeled datasets and enhances generalization to unseen anomalies.  

2. **Partial Matching for Hyperparameter Selection**: SPADE eliminates the need for a validation set by using a partial matching technique to determine optimal thresholds for pseudo-labeling. This innovation is particularly valuable in real-world applications where labeled data is scarce.  

3. **Integration of Supervised and Self-Supervised Learning**: SPADE combines supervised learning on labeled and pseudo-labeled data with self-supervised representation learning, improving encoder quality and overall detection performance.  

4. **Handling Distribution Mismatch**: SPADE is explicitly designed to operate effectively when labeled and unlabeled data originate from different distributions, a common scenario in real-world applications such as fraud detection and manufacturing defect analysis.  

Experimental results demonstrate that SPADE significantly outperforms existing methods, achieving improvements of up to **10.6% in AUC on tabular data** and **3.6% on image data** across various distribution mismatch scenarios.  

### **DETAILED DESCRIPTION**  

**1. System Architecture**  

SPADE comprises four primary components:  

- **Encoder (h)**: A neural network (e.g., MLP for tabular data, CNN for images) that maps input features **x** to latent representations **r = h(x)**.  
- **Predictor (q)**: A binary classifier that outputs anomaly scores **q(r)** based on the latent representations.  
- **Pseudo-Labeler (v)**: An ensemble of K OCCs that assigns pseudo-labels (**1**, **0**, or **-1**) to unlabeled data based on consensus among the OCCs.  
- **Projection Head (g)**: A module facilitating self-supervised learning (e.g., contrastive learning or reconstruction tasks) to enhance encoder training.  

**2. Pseudo-Labeling Mechanism**  

The pseudo-labeler **v** assigns labels as follows:  

- **Anomalous (1)**: If all OCCs in the ensemble agree that the sample is anomalous.  
- **Normal (0)**: If all OCCs agree the sample is normal.  
- **Unlabeled (-1)**: If no consensus is reached.  

This unanimous voting scheme ensures high-confidence pseudo-labels, reducing contamination from uncertain predictions.  

**3. Threshold Determination via Partial Matching**  

SPADE employs a Wasserstein distance-based partial matching method to set thresholds (**η_p**, **η_n**) without requiring validation data:  

- **η_p**: Matches the anomaly score distribution of labeled positive samples to unlabeled data.  
- **η_n**: Similarly derived from labeled negative samples.  

In PU or NU settings where only one class is labeled, Otsu’s method is used to determine the missing threshold.  

**4. Loss Functions and Optimization**  

SPADE optimizes the following objectives:  

- **Supervised Loss (L_l)**: Binary cross-entropy (BCE) on labeled data.  
- **Pseudo-Label Loss (L_u)**: BCE on high-confidence pseudo-labeled data.  
- **Self-Supervised Loss (L_ssl)**: Auxiliary task (e.g., reconstruction or contrastive learning) applied to all data.  

The total loss is:  
**L_total = L_l + αL_u + βL_ssl**  

where **α** and **β** are weighting hyperparameters (default: 1.0).  

**5. Experimental Validation**  

SPADE was evaluated on multiple datasets, including:  

- **Tabular Data**: Thyroid, Covertype, and Drug datasets.  
- **Image Data**: MVTec and Magnetic Tile datasets.  
- **Real-World Fraud Detection**: Kaggle Credit and Xente datasets.  

Key findings include:  

- **New Anomaly Types**: SPADE outperformed baselines by up to **10.6% AUC** when unlabeled data contained unseen anomalies.  
- **Labeling Bias**: SPADE maintained high accuracy even when labeled data consisted only of easy-to-classify samples.  
- **PU Learning**: Achieved superior performance in positive-unlabeled settings without sacrificing generalization.  

**6. Ablation Studies**  

Ablation experiments confirmed the contributions of each SPADE component:  

- Removing self-supervised learning reduced AUC by **1.8%**.  
- Replacing unanimous voting with majority voting decreased performance by **2.4%**.  

**7. Industrial Applications**  

SPADE is particularly suited for:  

- **Fraud Detection**: Adapts to evolving fraud patterns without frequent relabeling.  
- **Manufacturing Quality Control**: Detects novel defects not present in labeled training data.  
- **Cybersecurity**: Identifies emerging threats with minimal labeled examples.  

**Conclusion**  

SPADE represents a significant advancement in semi-supervised anomaly detection, offering robust performance under real-world distribution mismatches. By integrating ensemble-based pseudo-labeling, partial matching, and hybrid learning, SPADE achieves state-of-the-art results while minimizing reliance on costly labeled data.  

---  
*This concludes the detailed description of the invention.*