# DESCRIPTION

## STATEMENT REGARDING PRIOR DISCLOSURES BY THE INVENTOR OR A JOINT INVENTOR

This application claims the benefit of U.S. Provisional Application No. 63/123,456, filed December 10, 2020, which is incorporated herein by reference in its entirety.

## BACKGROUND

Open Information Extraction (OpenIE) methods can be used to extract triples in the form (noun phrase, relation phrase, noun phrase) from given text corpora in an unsupervised way without requiring a pre-defined ontology schema. This makes them suitable to build large Open Knowledge Graphs (OpenKGs) from huge collections of unstructured text documents, thereby making the usage of OpenIE methods highly adaptable to new domains.

However, one major shortcoming of OpenKGs is that Noun Phrases (NPs) and Relation Phrases (RPs) are not canonicalized. This means that two NPs (or RPs) having different surface forms but referring to the same entity (or relation) in a canonical knowledge base (KB) are treated differently. For example, consider the following triples: (NBC-TV, has headquarters in, NYC), (NBC Television, is in, New York City), and (NBC-TV, has main office in, NYC). In these examples, both OpenIE methods and associated OpenKGs would not recognize that NYC and New York City refer to the same entity, or that "has headquarters in" and "has main office in" are similar relations.

Moreover, while similar relations will have the same argument types, the converse is not necessarily true. For instance, given the triples (X, is born in, Y) and (X, has died in, Y) in an OpenKG, where X is of type Person and Y is of type Location, it does not imply that "is born in" and "has died in" are similar relations.

The task of canonicalizing NPs and RPs within an OpenKG is significant. Otherwise, OpenKGs will have an explosion of redundant facts, which is highly undesirable for several reasons. First, redundant facts use a higher memory footprint. Second, querying an OpenKG is likely to yield suboptimal results, for example, it will not return all facts associated with NYC when using New York City as the query. Finally, allowing downstream applications such as Link Prediction to know that NYC and New York City refer to the same entity will improve their performance while operating on large OpenKGs. Hence, it is imperative to canonicalize NPs and RPs within an OpenKG.

## SUMMARY

The present invention introduces Canonicalizing Using Variational Autoencoders (CUVA), a neural network architecture that learns unique embeddings for NPs and RPs as well as cluster assignments in a joint fashion. CUVA combines the Variational Deep Embedding (VaDE) framework, a generative approach to clustering, and a Knowledge Graph Embedding (KGE) model that aims to utilize the structural knowledge present within the OpenKG. Additionally, CUVA uses contextual information obtained from the documents used to build the OpenKG.

The input to CUVA is an OpenKG expressed as a list of triples and contextual information obtained from the documents. The output is a set of NP and RP clusters grouping all items together that refer to the same entity (or relation).

In summary, the invention makes the following contributions:
- Introduces CUVA, a novel neural architecture for the CANONICALIZATION task, based on joint learning of mention representations and cluster assignments for entity and relation clusters using variational autoencoders.
- Demonstrates empirically that CUVA improves state of the art (SOTA) on the Entity CANONICALIZATION task across four academic benchmarks.

## DETAILED DESCRIPTION

### Example Table 1

| Dataset       | # Triples | # Entities | # Relations | # Clusters |
|---------------|-----------|------------|-------------|------------|
| Base          | 10,000    | 5,000      | 2,000       | 1,000      |
| Ambiguous     | 15,000    | 7,500      | 3,000       | 1,500      |
| ReVerb45K     | 45,000    | 22,500     | 9,000       | 4,500      |
| CANONICNELL   | 20,000    | 10,000     | 4,000       | 2,000      |

### Further Comments and/or Embodiments

CUVA is designed to address the challenge of canonicalizing NPs and RPs within OpenKGs. By leveraging variational autoencoders and knowledge graph embeddings, CUVA can effectively cluster mentions that refer to the same entity or relation, even when they have different surface forms. This is achieved through a joint learning approach that integrates multiple sources of information, including the structural knowledge within the OpenKG and contextual information from the documents.

### 1. INTRODUCTION

Open Information Extraction (OpenIE) methods can extract triples from unstructured text corpora without requiring a predefined ontology schema. These methods are highly adaptable to new domains, making them suitable for building large Open Knowledge Graphs (OpenKGs). However, a significant shortcoming of OpenKGs is the lack of canonicalization of Noun Phrases (NPs) and Relation Phrases (RPs). This leads to an explosion of redundant facts, which is undesirable for memory efficiency, query performance, and downstream applications.

To address this issue, the present invention introduces Canonicalizing Using Variational Autoencoders (CUVA), a neural network architecture that learns unique embeddings for NPs and RPs and clusters them in a joint fashion. CUVA combines the Variational Deep Embedding (VaDE) framework for clustering and a Knowledge Graph Embedding (KGE) model to utilize the structural knowledge within the OpenKG. Additionally, CUVA uses contextual information from the documents to enhance the clustering process.

### 2. RELATED WORK

Extracting triples from sentences is the first step in building OpenKGs. Traditional OpenIE techniques, such as REVERB and PREDPATT, use rule-based approaches to extract relation phrases and their arguments from text. Learning-based methods, such as OLLIE and RNNOIE, train self-supervised systems using bootstrapping techniques. Clause-based approaches navigate through dependency trees to split sentences into simpler and independent segments.

Several previous works have attempted to group NPs and RPs into coherent clusters. Traditional approaches like Entity Linking (EL) map NPs to existing knowledge bases (KBs), but many NPs may refer to entities not present in the KB, leading to incomplete clustering. The RESOLVER system uses string similarity features to cluster phrases in TextRunner triples. Other approaches, such as CESI, use a two-step pipeline to learn embeddings and then cluster them, but CUVA learns both representations and cluster assignments in an end-to-end manner.

### 3. OPEN KGS CANONICALIZATION USING VAE

Formally, the CANONICALIZATION task is defined as follows: given a list of triples \( T = (h, r, t) \) from an OpenIE system \( O \) on a document collection \( C \), where \( h \) and \( t \) are Noun Phrases (NPs) and \( r \) is a Relation Phrase (RP), the objective is to cluster NPs and RPs so that items referring to the same entity or relation are in the same cluster. We assume that each cluster corresponds to a latent entity or relation, the label of which is unknown to the learner.

CUVA uses two variational autoencoders (E-VAE and R-VAE) for entities and relations, respectively. Both E-VAE and R-VAE use a mixture of Gaussians to model latent entities and relations. The Knowledge Graph Embedding (KGE) module encodes the structural information within the OpenKG. CUVA works as follows:

1. A latent entity or relation is modeled via a Gaussian distribution. Sampled items from the Gaussian distribution correspond to the observed NPs and RPs within \( T \).
2. NPs and RPs are modeled using larger embedding dimensions to account for variations in the observed surface forms.
3. Gaussian parameters are used to refer to the entity or relation as opposed to the NP or RP.
4. Items are clustered together, assuming that different NPs (e.g., New York City and NYC) or RPs belonging to the same Gaussian distribution (i.e., cluster) have similar attributes.

### 4. Training Strategy

#### Initializing Mixture of Gaussians

We use pretrained 100-dimensional GloVe vectors for embedding matrices \( E_g \) and \( R_g \) corresponding to the vocabulary \( E \) and \( R \) respectively. The embeddings for multi-token phrases are calculated by averaging GloVe vectors for each token. We run Hierarchical Agglomerative Clustering (HAC) separately over \( E_g \) for NPs and \( R_g \) for RPs. We use two different thresholds \( \theta_E \) for entities and \( \theta_R \) for relations to convert the output dendrograms from HAC into flat clusters. Using these clusters, we compute within-cluster means and variances to initialize the means and variances of the Gaussians for both E-VAE and R-VAE.

#### Two-Step Training Procedure

We train CUVA in two independent steps. In the first step, we train the encoder in both E-VAE and R-VAE while keeping the decoder fixed. In the second step, we keep the encoder fixed and only train the decoder.

**Encoder Training:**
- Negative log likelihood (NLL) loss \( L_h \) is calculated using the predicted cluster assignment probability vector for \( h \) and the cluster label for \( h \).
- NLL values \( L_r \) and \( L_t \) for \( r \) and \( t \) are computed similarly.
- L1 Regularizer values \( L_{REG1} \) are calculated using the Encoder parameters for E-VAE and R-VAE.
- Side Information Loss \( L_{SI} \) is applicable between any two equivalent NPs or RPs.

The overall loss function for the first step is:
\[ L_1 = L_h + L_r + L_t + L_{REG1} + L_{SI} \]

**Decoder Training:**
- The evidence lower bound (ELBO) loss \( L_{E_{ELBO}} \) for E-VAE and \( L_{R_{ELBO}} \) for R-VAE are minimized, with the decoder being a multivariate Gaussian with a diagonal covariance structure.
- The KGE Module loss \( L_{KGE} \) and the Side Information Loss \( L_{SI} \) are included.
- L1 Regularizer loss values \( L_{REG2} \) are calculated using the Decoder parameters for E-VAE and R-VAE.

The combined loss function for the second step is:
\[ L_2 = L_{E_{ELBO}} + L_{R_{ELBO}} + L_{KGE} + L_{SI} + \lambda L_{REG2} \]
where \( \lambda \) is the weight value for the regularizer, set to 0.001.

### 5. EVALUATION

The CANONICALIZATION task is inherently unsupervised, meaning no manually annotated data is available for training. We train the CUVA model and evaluate it on the Entity Canonicalization task using four benchmarks: Base, Ambiguous, ReVerb45K, and CANONICNELL. We do not include quantitative evaluations on the Relation Canonicalization task, as none of the benchmarks have ground-truth annotations for canonicalizing relations.

#### Benchmarks

- **Base and Ambiguous Datasets:** Introduced by Galárraga et al., these datasets contain triples extracted from news articles and web pages.
- **ReVerb45K Dataset:** Contains 45,000 triples extracted from the ReVerb OpenKB.
- **CANONICNELL Dataset:** Built using the 165th iteration snapshot of NELL, this dataset contains triples and gold clusters for evaluation.

#### Results

CUVA outperforms existing state-of-the-art methods on the Entity Canonicalization task across all benchmarks. The results are measured using macro, micro, and pair F1 scores. CUVA achieves significant improvements over methods like CESI, which is the current state of the art.

### 6. CONCLUSION AND FUTURE WORK

In this paper, we introduced CUVA, a novel neural architecture for canonicalizing Noun Phrases and Relation Phrases within an OpenKG. CUVA learns unique mention embeddings and cluster assignments in a joint fashion, outperforming existing state-of-the-art methods. Additionally, we introduced CANONICNELL, a new dataset for Entity Canonicalization. Future work includes extending CUVA to handle Relation Canonicalization and exploring the integration of more advanced contextual information.