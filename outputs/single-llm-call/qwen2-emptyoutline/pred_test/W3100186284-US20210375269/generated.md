# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of natural language processing (NLP) and, more specifically, to methods and systems for improving the performance of dialog act (DA) tagging in task-oriented dialog systems. The invention addresses the challenge of domain adaptation, particularly in scenarios where labeled data in the target domain is scarce but unlabeled data is abundant. By leveraging pre-trained language models and a novel data augmentation technique, the invention aims to enhance the generalization capabilities of DA taggers across different domains.

## BACKGROUND

Dialog act (DA) tagging is a crucial component in modern task-oriented dialog systems, aimed at capturing the speaker's intention behind each utterance in a dialogue. Over the years, various schemas and taxonomies have been proposed to standardize DA tagging (Core and Allen, 1997; Stolcke et al., 2000; Bunt et al., 2010; Mezza et al., 2018). Recent advancements have focused on human-human social conversations (Godfrey et al., 1992; Jurafsky et al., 1997), which, however, are less applicable to task-oriented settings.

The availability of task-oriented dialogue datasets (Shah et al., 2018; Henderson et al., 2014; Budzianowski et al., 2018) has facilitated research in this area. However, the discrepancy in annotation schemas across these datasets hinders the development of DA taggers that can generalize across domains. To address this issue, Paul et al. (2019) proposed a universal schema for DAs by aligning annotations from multiple corpora. Another valuable resource is the Schema-guided dialogues (SGD) dataset (Rastogi et al., 2020), which covers 20 domains under a unified DA annotation schema.

Obtaining large amounts of labeled dialogue data in the target domain is often challenging and costly. Unlabeled dialogue corpora, however, can be easily curated from past conversation logs or collected via crowdsourcing (Byrne et al., 2019; Budzianowski et al., 2018) at a more reasonable cost. Pre-trained language models, such as BERT (Devlin et al., 2019), have shown remarkable success in various NLP tasks, including dialogue systems (Wolf et al., 2019; Zhang et al., 2019; Bao et al., 2020; Henderson et al., 2019; Wu et al., 2020). However, the domain adaptation capabilities of these models in the context of goal-oriented dialogues remain underexplored.

The present invention introduces a novel method for leveraging pre-trained masked language models to improve the performance of DA taggers in unseen domains. Specifically, the invention utilizes the MASK token of the BERT model to define a data augmentation technique called MASKAUGMENT. This technique stochastically augments text input by randomly replacing tokens with the MASK token. The invention further adopts a consistency regularization approach to implement an unsupervised teacher-student learning scheme, thereby enhancing the model's ability to generalize across domains.

## DETAILED DESCRIPTION

### Overview

The present invention provides a method and system for improving the performance of dialog act (DA) tagging in task-oriented dialog systems, particularly in scenarios where labeled data in the target domain is limited. The invention leverages pre-trained language models, such as BERT, and introduces a novel data augmentation technique called MASKAUGMENT. This technique involves stochastically augmenting text input by randomly replacing tokens with the MASK token. The invention further employs a consistency regularization approach to implement an unsupervised teacher-student learning scheme, which helps the model generalize better across different domains.

The method includes the following steps:
1. **Formalizing the DA Tagging Task**: The DA tagging task is formulated as a multi-label classification problem, where the goal is to determine a subset of predefined DAs that apply to the current turn in a dialogue, given the conversation history.
2. **Model Architecture**: The invention uses a pre-trained language model, such as BERT, to process the dialogue history and predict the DA labels. The model is fine-tuned using a combination of supervised and unsupervised objectives.
3. **Data Augmentation with MASKAUGMENT**: The MASKAUGMENT technique is used to generate augmented versions of the input dialogue history by randomly replacing tokens with the MASK token. This helps the model learn more robust and generalizable representations.
4. **Unsupervised Teacher-Student Learning**: The invention employs a consistency regularization approach to implement an unsupervised teacher-student learning scheme. The teacher model generates predictions based on less perturbed input, while the student model learns from more perturbed input. The disagreement between the teacher and student predictions is minimized to improve the model's performance.

### Example Performance

The effectiveness of the proposed method has been evaluated on two benchmark datasets: GSIM (Shah et al., 2018) and SGD (Rastogi et al., 2020). The GSIM dataset consists of machine-machine task-oriented dialogues in two domains: buying a movie ticket (GMov) and reserving a restaurant table (GRes). The SGD dataset includes 22,825 schema-guided single/multi-domain dialogues across 20 domains.

**Training and Implementation Details**:
- The final loss function is a combination of the supervised tagging loss (STL), masked tagging loss (MTL), disagreement loss (DAL), and masked language model loss (MLM). The MLM loss is weighted by a factor of 0.1 when active.
- The disagreement loss (DAL) is activated after one epoch of training with the other objectives.
- Hyperparameters such as the masking probabilities for the teacher and student models (t and s) are tuned within specific ranges.
- The model is optimized using the AdamW optimizer (Loshchilov and Hutter, 2017) with a learning rate tuned within the range of [10^-5, 5 × 10^-5].
- The batch size is set to 16 examples, and the maximum sequence length is 128 tokens.

**Results and Discussion**:
- The proposed method significantly improves the cross-domain generalization performance of DA taggers. When compared to the baseline BERT model, the proposed method achieves up to a 3% improvement in F1 score on the target domain when the full source domain data is used.
- In low-resource settings, where only a small portion of the source domain data is available, the proposed method demonstrates even more substantial improvements, with up to a 10% increase in F1 score.
- The method consistently improves both precision and recall, with notable gains in closing the recall gap between the scratch-BERT and pre-BERT models.
- The use of the masked language model (MLM) loss as an unsupervised fine-tuning objective further enhances the model's performance on the target domain.

In conclusion, the present invention provides a robust and effective method for improving the performance of DA taggers in task-oriented dialog systems, particularly in scenarios with limited labeled data in the target domain. By leveraging pre-trained language models and a novel data augmentation technique, the invention significantly enhances the model's ability to generalize across different domains. Future work will explore the application of MASKAUGMENT to other NLP tasks and further refine the method to achieve even better performance.