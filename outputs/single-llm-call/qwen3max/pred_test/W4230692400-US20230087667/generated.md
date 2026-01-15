# DESCRIPTION

## STATEMENT REGARDING PRIOR DISCLOSURES BY THE INVENTOR OR A JOINT INVENTOR

The present invention builds upon foundational concepts in machine learning, knowledge representation, and natural language processing that have been disclosed in prior scientific literature. Specifically, Open Information Extraction (OpenIE) techniques, as introduced by Banko et al. (2007) and subsequently refined by Fader et al. (2011a), Stanovsky et al. (2018), and others, constitute prior art relevant to the extraction of structured triples from unstructured text. Additionally, the use of Variational Autoencoders (VAEs) for generative modeling and clustering, notably the Variational Deep Embedding (VaDE) framework proposed by Jiang et al. (2017a), represents established methodology in unsupervised representation learning. Knowledge Graph Embedding (KGE) models such as HolE (Nickel et al., 2016) are also well-documented in the literature for encoding relational structures. Furthermore, prior approaches to canonicalization—including RESOLVER (Yates and Etzioni, 2009), Concept Resolver (Krishnamurthy and Mitchell, 2011), KB-Unify (Delli Bovi et al., 2015), and CESI (Vashishth et al., 2018a)—have addressed aspects of noun phrase and relation phrase clustering using pipeline architectures combining embedding learning with hierarchical agglomerative clustering. The integration of side information from sources such as Entity Linking, PPDB, IDF token overlap, and morphological normalization has also been previously employed in canonicalization tasks. None of these prior disclosures, individually or in combination, teach or suggest the specific computer-implemented method described herein, which jointly learns mention embeddings and cluster assignments through a unified neural architecture incorporating variational autoencoders, knowledge graph structural constraints, and side information within a two-step training regimen.

## BACKGROUND

Machine learning refers to a class of computational methods that enable systems to automatically improve performance on a task through experience, typically by identifying patterns in data without explicit programming. Within this domain, neural networks—particularly deep neural networks—have demonstrated remarkable success in modeling complex, high-dimensional data such as images, audio, and text. A critical aspect of deploying neural networks effectively involves the selection and tuning of hyperparameters, which are configuration variables that govern the learning process itself, including learning rate, network depth, hidden layer dimensions, regularization coefficients, and optimization algorithms. These hyperparameters are not learned from data but must be set prior to training and significantly influence model performance.

Neural networks are motivated by their ability to approximate arbitrary functions through compositions of nonlinear transformations, enabling them to capture intricate dependencies in input data. A typical neural network architecture consists of an input layer, one or more hidden layers, and an output layer, with each layer comprising multiple neurons that apply weighted linear combinations followed by activation functions. Learning in neural networks occurs via backpropagation, wherein gradients of a loss function with respect to model parameters are computed and used to iteratively adjust weights to minimize prediction error. Applications of neural networks span diverse fields, including computer vision, speech recognition, natural language understanding, and knowledge base construction.

In the context of knowledge representation, canonicalizing extracted phrases from open-domain text presents a unique challenge. Traditional knowledge graphs rely on predefined ontologies, but Open Knowledge Graphs (OpenKGs) derived via OpenIE lack such structure, resulting in surface-form variations for semantically equivalent entities or relations. To address this, recent work has explored canonicalization using embedding-based clustering. However, existing methods often decouple representation learning from clustering, leading to suboptimal performance. The present invention introduces a novel approach—Canonicalizing Using Variational Autoencoders (CUVA)—that integrates variational inference, mixture-of-Gaussians priors, and structural knowledge from the OpenKG into a single end-to-end trainable framework, thereby overcoming limitations of prior pipeline-based solutions.

## SUMMARY

The present invention provides a computer-implemented method for canonicalizing noun phrases and relation phrases within an Open Knowledge Graph (OpenKG). The method comprises receiving, by a computing system, a set of triples extracted from unstructured text via an Open Information Extraction (OpenIE) system, wherein each triple includes a head noun phrase, a relation phrase, and a tail noun phrase. The computing system further receives side information comprising pairs of equivalent mentions derived from contextual cues such as entity linking, paraphrase databases, inverse document frequency (IDF) token overlap, and morphological normalization. The method then initializes a mixture of Gaussian distributions over latent entity and relation clusters using hierarchical agglomerative clustering applied to pretrained word embeddings of the noun and relation phrases. Subsequently, the computing system trains a neural network architecture comprising an Entity Variational Autoencoder (E-VAE), a Relation Variational Autoencoder (R-VAE), and a Knowledge Graph Embedding (KGE) module in two sequential steps: first, training the encoder sections of E-VAE and R-VAE using weak supervision from initial cluster labels and side information; second, training the decoder sections along with the KGE module using an evidence lower bound (ELBO) loss, a KGE-specific loss, and a constraint-based side information loss. During inference, the trained model assigns each noun phrase and relation phrase to a cluster by computing soft cluster probabilities and selecting the most probable cluster via a differentiable soft-argmax operation. The output is a canonicalized OpenKG wherein redundant surface forms are grouped into coherent semantic clusters representing latent entities and relations.

## DETAILED DESCRIPTION

Knowledge graphs represent structured knowledge as collections of subject-predicate-object triples, enabling reasoning, question answering, and data integration. Open Knowledge Graphs (OpenKGs), constructed via Open Information Extraction (OpenIE) from large text corpora, offer high adaptability across domains due to their ontology-free design. However, a significant limitation of OpenKGs is the absence of canonicalization: semantically identical entities or relations may appear under multiple surface forms (e.g., “New York City” vs. “NYC”), leading to fragmented knowledge, increased storage overhead, and degraded query performance. This redundancy impedes downstream applications such as link prediction and entity resolution.

Existing solutions to this adaptability problem include mapping noun phrases to external knowledge bases (Entity Linking), applying string similarity heuristics, or using embedding-based clustering. For instance, the CESI system employs a two-stage pipeline: first learning embeddings via HolE, then clustering via Hierarchical Agglomerative Clustering (HAC). While effective, such approaches suffer from error propagation—embedding errors cannot be corrected during clustering—and fail to leverage structural and contextual signals jointly. Moreover, they do not model uncertainty in cluster membership, which is crucial for polysemous phrases.

To overcome these deficiencies, the present invention introduces Canonicalizing Using Variational Autoencoders (CUVA), a unified neural architecture that performs joint learning of mention representations and cluster assignments. CUVA comprises two variational autoencoders: E-VAE for noun phrases (entities) and R-VAE for relation phrases. Each VAE assumes a mixture-of-Gaussians prior over latent clusters, enabling soft clustering that accommodates ambiguity. The architecture further incorporates a Knowledge Graph Embedding (KGE) module—implemented via HolE—that enforces structural consistency by ensuring that triples (h, r, t) satisfy relational constraints in the embedding space. Additionally, side information from external sources is encoded as a constraint-based loss that penalizes dissimilar embeddings for known equivalent mentions.

The CUVA architecture features distinct encoder and decoder sections. The encoder maps input phrase embeddings to parameters of a variational posterior distribution over latent codes, using fully connected layers with tanh activations. The decoder reconstructs the input from sampled latent codes and also interfaces with the KGE module. A key innovation is the constraint loss, which computes a weighted mean squared error between embeddings of equivalent mention pairs, with weights derived from plausibility scores of the side information sources.

Training proceeds in two steps to avoid posterior collapse. First, the encoders are trained using negative log-likelihood (NLL) loss against pseudo-labels from HAC-initialized clusters, augmented with side information loss and L1 regularization. Second, the decoders and KGE module are trained using ELBO loss (comprising reconstruction and KL divergence terms), KGE loss, and side information loss, while encoder weights remain fixed. This staged approach ensures that latent representations meaningfully encode cluster structure before being used for reconstruction and relational modeling.

Side information is encoded by generating equivalent mention pairs from five sources: Entity Linking (via Stanford CoreNLP), PPDB paraphrase clusters, IDF token overlap scores, WordNet synsets with word-sense disambiguation, and morphological normalization rules. Each pair is assigned a plausibility score based on source reliability and cluster ambiguity, which modulates its contribution to the side information loss.

Operational steps for training include: (1) preprocessing OpenKG triples and side information; (2) initializing Gaussian mixture components via HAC on GloVe embeddings; (3) performing Step 1 encoder training; (4) performing Step 2 decoder and KGE training; and (5) assigning final cluster labels via maximum a posteriori estimation. The benefits of CUVA include improved canonicalization accuracy, robustness to surface-form variation, end-to-end trainability, and compatibility with diverse side information sources.

### Example Table 1

Example Table 1 illustrates the core components and operational flow of the CUVA system. The table describes how noun phrases (NPs), relation phrases (RPs), and their associated triples are processed through the architecture. Clustering begins by grouping NPs and RPs into initial clusters using hierarchical agglomerative clustering (HAC) on pretrained GloVe embeddings, which serves to initialize the means and variances of the Gaussian mixture components in the E-VAE and R-VAE, respectively. The E-VAE is specifically designed to model entity mentions, taking NP embeddings as input and outputting parameters of a multivariate Gaussian distribution in the latent space. Similarly, the R-VAE processes RP embeddings to produce latent relation representations.

A dedicated module for knowledge base completion is integrated via the Knowledge Graph Embedding (KGE) component, which uses the HolE algorithm to enforce that reconstructed triples adhere to the compositional structure of valid facts. The entire neural network architecture—comprising E-VAE, R-VAE, and KGE—is trained end-to-end using a composite loss function that balances reconstruction fidelity, clustering coherence, structural validity, and side information constraints.

The hierarchical agglomerative cluster model is built during initialization by applying HAC separately to entity and relation GloVe embeddings, using dataset-specific distance thresholds to determine the number of initial clusters (K_E and K_R). The encoder section of each VAE is trained in Step 1 to predict cluster assignment probabilities, using NLL loss against HAC-derived pseudo-labels and incorporating side information loss to correct erroneous groupings. The decoder section, trained in Step 2, reconstructs input embeddings from latent samples and updates Gaussian parameters to better reflect true cluster structure.

The total loss value combines multiple objectives: in Step 1, it includes NLL losses for head, relation, and tail predictions, L1 regularization on encoder weights, and side information loss; in Step 2, it comprises ELBO losses for both VAEs, KGE loss based on HolE’s circular correlation scoring, side information loss, and L1 regularization on decoder weights. FIG. 2A depicts the encoder block, showing input embedding projection through fully connected layers to produce μ and σ vectors. FIG. 2B illustrates the decoder block, which maps latent samples back to reconstructed embeddings. FIG. 2C shows the cluster assignment mechanism using soft-argmax over posterior probabilities.

The core structure of CUVA integrates E-VAE, R-VAE, and KGE into a single computational graph. The encoder block uses three hidden layers (768 → 384 → 100 dimensions) with tanh activation to compress input embeddings into latent distribution parameters. The decoder block mirrors this with reverse dimensions (100 → 384 → 768). Side information is encoded by looking up embeddings of equivalent pairs and computing weighted MSE, as shown in FIG. 3, which diagrams the side information loss computation. FIG. 4 visualizes an example equivalent pair (“NYC”, “New York City”) and its impact on embedding alignment. FIG. 5 outlines the two-step training workflow, while FIG. 6 details the inference pipeline for assigning new mentions to clusters.

During operation, the system receives OpenKG triples and side information, dynamically clusters mentions by refining initial HAC clusters through neural training, and initializes cluster means and variances from within-cluster statistics of GloVe embeddings. The neural network is dynamically trained using two distinct loss functions across two phases, with an additional constraint-based loss enforcing equivalence from side information. Structural knowledge from the OpenKG is encoded via the KGE module, which couples entity and relation representations through HolE’s interaction function.

### Further Comments and/or Embodiments

The CUVA architecture offers significant benefits over prior art by unifying representation learning and clustering into a single probabilistic framework. Unlike pipeline methods such as CESI, CUVA avoids error propagation and leverages gradient-based optimization to jointly refine embeddings and cluster assignments. Current state-of-the-art approaches fail to model cluster uncertainty or integrate structural and contextual signals cohesively, leading to brittle performance on ambiguous or sparse mentions. CUVA addresses these gaps through its variational formulation and multi-objective loss design.

To facilitate empirical validation, the inventors introduce the CANONICNELL dataset, derived from the 165th iteration of the Never-Ending Language Learner (NELL) system. CANONICNELL provides gold-standard entity clusters based on co-reference annotations filtered by soft-truth scores, offering a novel benchmark independent of ReVerb-based OpenKGs. This dataset enables rigorous evaluation of canonicalization systems in a setting with inherent type constraints and unique relation phrases, complementing existing benchmarks like ReVerb45K, Base, and Ambiguous.

### 1. INTRODUCTION

Knowledge graphs encode real-world facts as structured triples, supporting applications in search, recommendation, and reasoning. Open Knowledge Graphs (OpenKGs), built via Open Information Extraction (OpenIE) from raw text, bypass the need for predefined schemas, enabling rapid adaptation to new domains. However, OpenKGs suffer from a critical flaw: the same entity or relation may be expressed through multiple surface forms (e.g., “NBC-TV” vs. “NBC Television”), leading to redundant, fragmented knowledge. This lack of canonicalization inflates storage requirements, complicates querying, and degrades performance in downstream tasks like link prediction.

OpenIE methods extract triples (noun phrase, relation phrase, noun phrase) without ontological constraints, yielding highly adaptable but noisy knowledge bases. The resulting OpenKGs contain numerous near-duplicates that refer to identical semantic concepts, yet are treated as distinct due to lexical variation. Canonicalization—the process of grouping such variants into coherent clusters—is thus essential for producing compact, queryable, and semantically consistent knowledge graphs.

To solve this, the present invention introduces Canonicalizing Using Variational Autoencoders (CUVA), a neural architecture that jointly learns embeddings and cluster assignments for noun and relation phrases. CUVA integrates variational inference with knowledge graph structural priors and external side information, enabling end-to-end optimization. The key contributions include: (1) a unified model for entity and relation canonicalization using mixture-of-Gaussians VAEs; (2) a two-step training strategy that prevents posterior collapse; (3) incorporation of structural knowledge via a KGE module; and (4) empirical demonstration of state-of-the-art performance across four benchmarks, including the newly introduced CANONICNELL dataset.

### 2. RELATED WORK

Open Information Extraction (OpenIE) was pioneered by Banko et al. (2007) to extract relational triples from text without domain-specific rules. Subsequent systems improved precision and recall through rule-based approaches like REVERB (Fader et al., 2011a) and PREDPATT (White et al., 2016), which use syntactic patterns, and learning-based methods such as OLLIE (Mausam et al., 2012) and RNNOIE (Stanovsky et al., 2018), which employ bootstrapping or neural sequence labeling. Clause-based systems (Angeli et al., 2015) further decompose sentences for finer-grained extraction.

For canonicalization, traditional Entity Linking (EL) maps noun phrases to entries in curated knowledge bases like Wikidata (Lin et al., 2012; Ceccarelli et al., 2014). However, EL fails for out-of-vocabulary entities. Alternative approaches include RESOLVER (Yates and Etzioni, 2009), which uses string similarity; Galárraga et al. (2014a), which combines manual features with AMIE for relation clustering; and Concept Resolver (Krishnamurthy and Mitchell, 2011), which assumes “one sense per category” in NELL. KB-Unify (Delli Bovi et al., 2015) merges multiple KGs but requires sense inventories.

The state-of-the-art CESI (Vashishth et al., 2018a) uses a pipeline: HolE embeddings followed by HAC clustering, augmented with side information. However, its decoupled design limits joint optimization. CUVA overcomes this by integrating embedding and clustering into a single variational framework, enabling mutual refinement of representations and assignments.

### 3. OPEN KGS CANONICALIZATION USING VAE

The CANONICALIZATION task is formally defined as: given a set of OpenIE triples T = {(h_i, r_i, t_i)}, cluster all noun phrases (NPs) and relation phrases (RPs) such that those referring to the same latent entity or relation are grouped together. The label of each latent concept is unknown, making this an unsupervised clustering problem.

CUVA addresses this via two Variational Autoencoders: E-VAE for NPs and R-VAE for RPs. Each assumes a mixture of K Gaussians as the prior over latent clusters. Observed phrases are modeled as samples from these Gaussians, with high-dimensional embeddings capturing surface-form variation. The Gaussian parameters (mean and variance) represent the canonical entity or relation, while individual mentions are noisy observations.

The Variational AutoEncoder follows the VaDE generative process: (1) sample a cluster c from categorical prior π; (2) sample latent code z from Gaussian N(μ_c, σ²_c I); (3) decode z to observation x via neural network f_θ; (4) sample x from N(μ_x, σ²_x I). Inference uses mean-field approximation q(z,c|x) = q(z|x)q(c|x).

The Encoder block computes q(z|h) for head NP h via:
μ_h = W_2 tanh(W_1 h + b_1) + b_2  
σ_h = softplus(W_3 h + b_3)  
z = μ_h + σ_h ⊙ ε, ε ~ N(0,I)

The Decoder block reconstructs h from z and computes cluster probabilities:
q(c|h) ∝ π_c N(z; μ_c, σ²_c I)

Cluster assignment uses soft-argmax with temperature τ=1e5 to approximate argmax differentiably.

The KGE module enforces structural validity. Given cluster assignment probabilities for (h,r,t), it computes soft cluster indicators v_h, v_r, v_t via:
v_α = softmax(τ c_α)

Entity/relation representations are then:
e_h = M_E v_h, e_r = M_R v_r

where M_E, M_R are matrices of Gaussian means. These are fed into HolE, which scores triples via circular correlation.

Side Information Loss L_SI penalizes distance between equivalent mentions:
L_SI = Σ_{(p,q)∈S} s_{pq} ||e_p - e_q||²

where S is the set of equivalent pairs and s_{pq} is their plausibility score.

### 4. Training Strategy

Training begins by initializing the Gaussian mixture. Pretrained 100D GloVe embeddings are averaged for multi-token phrases. Hierarchical Agglomerative Clustering (HAC) is applied separately to entity and relation embeddings using complete linkage. Distance thresholds θ_E and θ_R determine the number of clusters K_E and K_R. Within-cluster means and variances initialize the Gaussian parameters.

A two-step training strategy is employed. Step 1 trains encoders using weak supervision from HAC labels. For each triple (h,r,t), Negative Log Likelihood (NLL) losses L_h, L_r, L_t are computed against pseudo-labels. Total Step 1 loss:
L_step1 = L_h + L_r + L_t + λ L_REG1 + L_SI

Step 2 trains decoders and KGE module. Evidence Lower Bound (ELBO) losses for E-VAE and R-VAE include reconstruction and KL terms. KGE loss L_KGE uses HolE’s margin-based ranking. Total Step 2 loss:
L_step2 = L_E_ELBO + L_R_ELBO + L_KGE + L_SI + λ L_REG2

This staged approach prevents the decoder from ignoring latent codes—a common issue in VAEs—by fixing encoder weights during Step 2.

### 5. Evaluation

Evaluation focuses on Entity Canonicalization, as relation ground truth is unavailable in standard benchmarks. Datasets include Base and Ambiguous (Galárraga et al., 2014a), ReVerb45K (Vashishth et al., 2018a), and the newly introduced CANONICNELL, built from NELL165 co-reference data. CANONICNELL uses DFS-connected components of high-confidence entity pairs as gold clusters.

Metrics are Macro-F1, Micro-F1, and Pairwise-F1. Hyperparameters are tuned on validation sets via grid search over HAC thresholds, latent dimensions {50,100,200}, and side information cutoffs. CUVA uses 768→384→100 encoder and reverse decoder dimensions, tanh activations, Adam optimizer (lr=1e-3/1e-4), batch size 50, and 20 negative samples for HolE.

Results show CUVA matches or exceeds CESI on Base, Ambiguous, and ReVerb45K, and sets new SOTA on CANONICNELL. Ablation studies confirm the necessity of the KGE module (+8.4% Pair-F1) and joint learning (vs. VAE+HAC pipeline). CUVA also outperforms BERT, RoBERTa, and ERNIE+HAC by 4–12% in average F1, demonstrating the advantage of task-specific architecture over generic language models.

Qualitative analysis reveals correct clustering of “kodagu”/“coorg” but errors in disambiguating same-name persons (e.g., “bill cosby”/“bill nye”). Relation clusters sometimes conflate antonyms (“rank below”/“rank above”), indicating need for negation handling.

### 6. CONCLUSION AND FUTURE WORK

CUVA presents a novel neural architecture for OpenKG canonicalization, jointly learning mention embeddings and cluster assignments via variational autoencoders, structural constraints, and side information. It achieves state-of-the-art results across four benchmarks, including the new CANONICNELL dataset. Future work includes extending to relation canonicalization, incorporating type information for disambiguation, and exploring hypernymy detection and link prediction as downstream applications.

The invention may be implemented in a computing environment comprising one or more computer systems. Each computer system includes a communications fabric connecting processor(s), cache, memory (RAM), persistent storage (HDD/SSD), communications unit (NIC), I/O interface, and external devices (keyboard, display). Program instructions for CUVA are stored on a computer-readable storage medium (e.g., SSD) and loaded into memory for execution by the processor. Instructions may be downloaded via network adapter from remote servers. Execution follows flowchart logic for training and inference. Block diagrams illustrate architectural components. The described embodiments thus provide a complete, operable system for scalable, accurate knowledge graph canonicalization.