# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of natural language processing (NLP), specifically to systems and methods for abstractive summarization. More particularly, the invention addresses the problem of entity hallucination in abstractive summarization, ensuring that generated summaries are faithful to the source documents.

## BACKGROUND

Abstractive summarization is a critical task in natural language processing, aimed at generating concise and coherent summaries that capture the essence of longer documents. Recent advancements in pre-trained language models, such as BERT, T5, and BART, have significantly improved the quality of generated summaries. However, a significant challenge remains: the issue of faithfulness. Faithful summaries should only contain information that can be directly derived from the source document, avoiding hallucinations—fabricated or incorrect information.

Entity-level hallucination, where the model generates entities not present in the source document, is a common and problematic form of hallucination. Previous approaches to mitigate this issue include post-processing methods, data filtering, and multi-task learning. However, these methods often introduce additional errors or sacrifice the quality of the summaries.

The present invention introduces a novel approach to address entity hallucination by guiding the model's learning process with entity control codes (ECC). This method ensures that the generated summaries are both faithful and of high quality, making it suitable for various applications where accuracy is paramount.

## DETAILED DESCRIPTION

### Systems for Abstractive Summarization

The invention provides a system and method for generating faithful and high-quality abstractive summaries. The system utilizes a seq2seq model, such as BART, and incorporates entity control codes (ECC) to guide the summarization process. The ECC method involves the following steps:

1. **Problem Formulation**: The system is designed to generate a summary \( h_i \) for a given document \( d_i \) such that all information in \( h_i \) is entailed by \( d_i \). The entity coverage precision \( \text{prec}_{\text{en}} \) is used to measure the faithfulness of the summary by calculating the ratio of named entities in the summary that are present in the source document.

2. **Entity Coverage Control**: For each document-summary pair in the training set, the system computes the entity coverage precision \( \text{prec}_{\text{en}} \). This value is quantized into discrete bins, each represented by a control code. During training, the control code is prepended to the input document, conditioning the model to learn different levels of faithfulness. During inference, a high-faithfulness control code is prepended to ensure the generated summary is faithful.

3. **Controllable Intermediate Fine-Tuning**: To improve zero-shot summarization performance, the system generates pseudo document-summary pairs from a large corpus, such as Wikipedia, with similar characteristics to the target datasets. Each pseudo pair is associated with a target-specific control code, allowing the model to generalize across different domains. During inference, the appropriate target control code is prepended to the input document, along with the entity coverage control code, to generate faithful and domain-appropriate summaries.

### Detailed Methodology

#### Problem Formulation

Let \( D = \{(d_1, s_1), (d_2, s_2), \ldots, (d_n, s_n)\} \) denote a dataset composed of \( n \) document and summary pairs. During the inference phase, a seq2seq model generates a summary hypothesis \( h_i \) for a given document \( d_i \) by computing the probability \( p_\theta(h_i | d_i) \). The generated summary \( h_i \) is expected to be faithful, meaning all the information in \( h_i \) should be entailed by the source document \( d_i \).

Following previous work, entity-level hallucination is quantified using entity coverage precision \( \text{prec}_{\text{en}} \), which measures the ratio of named entities in the summary that are present in the source document. Formally, it is defined as:

\[
\text{prec}_{\text{en}} = \frac{|N(s_i) \cap N(d_i)|}{|N(s_i)|}
\]

where \( N(t) \) represents the set of all named entities found in a given input text \( t \).

#### Entity Coverage Control

The entity coverage control method involves generating a control code \( C_i \) for each training document and reference summary pair \( (d_i, s_i) \). The seq2seq model is conditioned on both the source document \( d_i \) and its control code \( C_i \).

First, the system computes the entity coverage precision \( \text{prec}_{\text{en}} \) for each document and reference summary pair in the training set \( D \). The values of \( \text{prec}_{\text{en}} \) are quantized into \( k \) discrete bins, each representing a range of entity faithfulness. These bin boundaries are selected to ensure that each bin contains roughly the same number of training examples to avoid data imbalance. Each bin is represented by a special token control code \( C_i \), and all these special tokens \( \{C_1, C_2, \ldots, C_k\} \) are added to the input vocabulary of the seq2seq model.

During training, the corresponding control code \( C_i \) is prepended to the input document, conditioning the model to learn different faithful level generation patterns from the control codes. During inference, the high-faithfulness control code \( C_k \) is prepended to all documents in the test set, and the model generates faithful summaries by computing \( p_\theta(h_i | d_i, C_k) \).

#### Controllable Intermediate Fine-Tuning

Large pre-trained language models, such as BERT and T5, perform poorly in zero-shot summarization settings because they lack sentence salience information learned through pretraining tasks. To address this, the system proposes a controllable generalized intermediate fine-tuning method.

Pseudo document-summary pairs are generated from a large corpus, such as Wikipedia, with similar summary length, document length, and abstractiveness to the target datasets. Instead of training different models for different target datasets, a unified model is proposed that generalizes well across different domains. Each target-specific pseudo training subset \( \{D_1(n_1, m_1, a_1), \ldots, D_l(n_l, m_l, a_l)\} \) is associated with a special token \( E_i \) as a pseudo label, representing the target-specific pattern. All these special tokens \( \{E_1, E_2, \ldots, E_l\} \) are added to the input vocabulary of the seq2seq model.

During training, the corresponding target code \( E_i \) is prepended to the document, and a summary is generated conditioned on both the source document \( d_i \) and its target control code \( E_i \), represented as \( p_\theta(h_i | d_i, E_i) \). This allows for control over the domain and generation style of generated summaries by prepending different domain control codes during inference. The control codes are stackable, so the target control code can be combined with the entity coverage control code for faithful zero-shot summarization, denoted as \( p_\theta(h_i | d_i, C_k, E_i) \).

## EXAMPLES

### Example 1: Experimental Methods and Results

#### Experimental Setup

The system was evaluated on three benchmark datasets: XSum, Pubmed, and SAMsum. The BART-large model was used as the backbone, and the hyperparameter \( k \) was set to 3 for all experiments. The three discrete ECC bins were represented with control codes: <FF-low>, <FF-mid>, and <FF-high>. The entity coverage precision boundaries were set to 0.36 and 0.5 for Pubmed, 0.33 and 0.66 for SAMsum, and 0.33 and 0.66 for XSum.

#### Baselines

The system was compared against two state-of-the-art methods in summarization faithfulness:
1. Post-processing correction (Chen et al., 2021) combined with original BART.
2. WikiTransfer (Fabbri et al., 2021) for zero-shot summarization.

#### Automatic Evaluation

Table 2 shows the performance of the ECC method in the supervised setting. Compared to the summaries generated by BART, the ECC method significantly increased the entity coverage precision while maintaining similar summary quality. Table 3 compares the performance to strong baselines on the XSum dataset. The ECC method achieved comparable faithfulness improvements without degrading the summary quality, unlike data filtering and post-processing methods.

Table 4 presents the zero-shot summarization results. BART tended to copy from the source document, achieving high entity coverage precision but low summary quality. With the intermediate fine-tuning, BART learned the characteristics of the downstream dataset, resulting in a considerable improvement in ROUGE score. Compared to the baseline WikiTransfer, the ECC method showed improvements in both entity coverage precision and summary quality. Additionally, a single model was used for different downstream targets, unlike the separate models required by WikiTransfer.

#### Human Evaluation

Table 5 shows the human evaluation results on a 50 randomly sampled subset of articles from the XSum dataset. Four expert annotators assigned each summary output to three faithfulness categories (faithful summary, intrinsic hallucination, extrinsic hallucination) and three summary quality categories (low, medium, high). The results indicate that the ECC model improved the faithfulness of the summaries without degrading summary quality, consistent with the automatic evaluation results.

#### Analysis and Discussion

One concern is whether the model generates fewer entities to achieve higher entity coverage precision. Figure 2 shows the distribution of the number of entities in the generated summaries by the ECC model and BART. The distributions are very similar, with almost the same mean number of entities, suggesting that the ECC method does not under-generate or over-generate entities.

The effect of different control codes during inference was also studied. Table 6 shows the performance of the model when inferred with low and medium control codes on the XSum test set. The model still generated reasonable summaries, with summaries inferred with low control codes having higher ROUGE scores, aligning with the trade-off between entity coverage precision and summary quality.

#### Why Does BART Generate Hallucinated Tokens?

Table 7 illustrates why BART generates hallucinated tokens. For example, fine-tuned BART generates "Gary Anderson" based on the context "Saints captain Anderson," which is incorrect as the actual captain is "Steven Anderson." This is due to the pre-trained prior knowledge of BART, which contains abundant relational knowledge from pre-training data. When given the whole news article, BART generates famous athletes "Craig Anderson" (hockey athlete) and "Gary Anderson" (football athlete) based on its pre-trained knowledge. The ground truth "Steven Anderson" appears less frequently during pre-training, leading to a low probability of generating it correctly.

### Conclusion

The present invention, Entity Coverage Control (ECC), effectively addresses entity hallucination in abstractive summarization in both supervised and zero-shot settings. Extensive experiments demonstrate that the proposed method significantly reduces entity hallucination without compromising the quality of the generated summaries. This innovation has the potential to enhance the reliability and practicality of abstractive summarization systems in various applications.