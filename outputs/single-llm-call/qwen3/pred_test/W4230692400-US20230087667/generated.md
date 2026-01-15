# DESCRIPTION

## STATEMENT REGARDING PRIOR DISCLOSURES BY THE INVENTOR OR A JOINT INVENTOR

- disclose prior art

Prior to the present invention, systems for organizing unstructured textual information into structured knowledge graphs relied heavily on open information extraction techniques that produced triples in the form of (noun phrase, relation phrase, noun phrase) without canonicalization of entities or relations. These systems, including REVERB, OLLIE, and RNNOIE, treated surface forms of noun phrases and relation phrases as distinct even when they referred to the same underlying entity or semantic relation. For example, “New York City” and “NYC” were not recognized as referring to the same geographic location, and “has headquarters in” and “is located in” were not understood as semantically equivalent relational predicates. This lack of canonicalization led to the proliferation of redundant, fragmented, and semantically inconsistent facts within open knowledge graphs. Subsequent efforts to address this issue introduced entity linking methods that mapped noun phrases to external knowledge bases such as Wikipedia or Wikidata; however, these approaches were inherently limited by their dependence on pre-existing ontologies and failed to cluster novel or under-represented mentions that did not appear in those external resources. Other methods employed rule-based feature engineering, hierarchical agglomerative clustering, or pairwise similarity metrics based on token overlap or morphological normalization, but these techniques operated in isolation from the structural context of the knowledge graph and did not jointly model the latent semantic space of entities and relations. The CESI architecture represented a significant advancement by integrating knowledge graph embeddings with hierarchical clustering, yet it remained a two-stage pipeline in which entity representations were first learned independently and then clustered in a separate, non-differentiable step. This separation prevented the mutual refinement of embeddings and cluster assignments, resulting in suboptimal convergence and limited ability to resolve ambiguous or polysemous mentions. No prior system combined variational autoencoding with knowledge graph embedding in an end-to-end framework that simultaneously learns continuous representations of mentions and soft cluster assignments while incorporating side information as soft constraints. The present invention overcomes these limitations by introducing a unified neural architecture that integrates variational inference, knowledge graph embedding, and constraint-based learning into a single trainable model capable of canonicalizing both noun phrases and relation phrases in an unsupervised manner.

## BACKGROUND

- define machine learning

Machine learning is a computational paradigm in which systems are trained to recognize patterns and make decisions from data without being explicitly programmed with fixed rules. These systems learn by adjusting internal parameters through iterative exposure to examples, optimizing an objective function that measures performance on a given task. The learning process typically involves minimizing a loss function that quantifies the discrepancy between predicted outputs and ground truth or desired behavior, enabling the system to generalize from observed instances to unseen data. Machine learning models range from simple linear classifiers to complex nonlinear architectures capable of modeling high-dimensional, hierarchical, and probabilistic relationships within data.

- explain hyperparameters

Hyperparameters are configuration settings that govern the structure, behavior, and learning dynamics of a machine learning model and are set prior to the training process. Unlike model parameters, which are learned from data during optimization, hyperparameters are not updated through gradient descent or other iterative procedures. Examples include the number of layers in a neural network, the size of latent embeddings, the learning rate, the regularization strength, and the number of clusters in a clustering model. The selection of appropriate hyperparameters significantly influences model capacity, convergence speed, and generalization performance. Poorly chosen hyperparameters may lead to underfitting, where the model fails to capture underlying patterns, or overfitting, where the model memorizes noise rather than learning meaningful structure. Hyperparameter tuning is therefore a critical step in developing robust and effective machine learning systems.

- motivate neural networks

Neural networks are a class of machine learning models inspired by the structure and function of biological neural systems, composed of interconnected layers of artificial neurons that process inputs through nonlinear transformations. Their ability to approximate complex, high-dimensional functions makes them particularly well-suited for tasks involving unstructured data such as text, images, and graphs. Unlike traditional rule-based systems, neural networks do not require manual feature engineering; instead, they automatically discover relevant representations through hierarchical composition of learned features. This property enables them to capture subtle, context-dependent relationships that are difficult to encode explicitly, such as semantic similarity between phrases with divergent surface forms or relational patterns that vary across domains. Their end-to-end learnability allows them to integrate multiple sources of information—structural, lexical, and contextual—into a unified representation space, making them ideal for tasks requiring semantic reasoning over noisy, incomplete, or heterogeneous data.

- describe neural network architecture

A neural network architecture consists of interconnected layers of computational units arranged in a directed graph, typically including an input layer, one or more hidden layers, and an output layer. Each unit applies a nonlinear activation function to a weighted sum of its inputs, enabling the network to model complex decision boundaries. The architecture may include convolutional layers for spatial feature extraction, recurrent layers for sequential modeling, or attention mechanisms for dynamic weighting of inputs. In the context of knowledge graph canonicalization, the architecture comprises encoder and decoder components connected through a latent space, where each node in the graph is mapped to a continuous vector representation. The encoder transforms raw input features into a compressed, probabilistic latent encoding, while the decoder reconstructs the input from this latent representation. Additional modules may be integrated to enforce structural constraints, such as knowledge graph embedding layers that preserve relational topology, or clustering heads that assign soft memberships to latent groups.

- explain neural network learning

Neural network learning involves adjusting the weights of connections between units to minimize a predefined loss function that measures the difference between predicted and target outputs. This is typically achieved through backpropagation, a gradient-based optimization algorithm that computes the derivative of the loss with respect to each parameter and updates them in the direction that reduces error. Learning is performed iteratively over batches of training examples, with the network gradually refining its internal representations to better capture the underlying data distribution. Regularization techniques such as L1 or L2 penalties, dropout, or early stopping are often employed to prevent overfitting. In probabilistic architectures such as variational autoencoders, learning also involves optimizing a lower bound on the marginal likelihood of the data, balancing reconstruction accuracy with the regularization imposed by a prior distribution over the latent space.

- discuss neural network applications

Neural networks have been successfully applied across a broad spectrum of domains, including natural language processing, computer vision, speech recognition, bioinformatics, and recommendation systems. In natural language processing, they are used for machine translation, sentiment analysis, named entity recognition, and question answering. In knowledge representation, they enable link prediction, entity resolution, and relation extraction from unstructured text. Their ability to learn dense, continuous representations of symbolic entities has revolutionized the field of knowledge graphs, allowing systems to infer missing facts, resolve ambiguities, and generalize across domains with minimal supervision. Recent advances have extended their use to unsupervised clustering tasks, where latent variable models such as variational autoencoders are employed to group similar instances without labeled examples, enabling scalable canonicalization of open knowledge graphs.

- introduce canonicalizing using variational autoencoders

Canonicalizing using variational autoencoders is a novel computational framework for resolving semantic ambiguity in open knowledge graphs by jointly learning continuous embeddings of entity and relation mentions and assigning them to latent clusters through probabilistic inference. Unlike traditional clustering methods that operate on fixed feature representations, this approach embeds mentions into a latent space governed by a mixture of Gaussian distributions, where each cluster corresponds to a latent entity or relation. The variational autoencoder architecture enables soft clustering by modeling the probability that a given mention belongs to each cluster, allowing for uncertainty and polysemy. The model is trained end-to-end to maximize the evidence lower bound of the data likelihood, incorporating structural knowledge from the knowledge graph through a knowledge graph embedding module and side information as soft constraints. This integration ensures that cluster assignments are informed not only by surface form similarity but also by relational context and external evidence, resulting in more accurate and semantically coherent canonicalizations.

## SUMMARY

- outline computer-implemented method

A computer-implemented method for canonicalizing noun phrases and relation phrases in an open knowledge graph comprises receiving a plurality of triples, each triple comprising a head noun phrase, a relation phrase, and a tail noun phrase, and contextual information derived from the source text of the triples. The method further comprises encoding each noun phrase and relation phrase into a high-dimensional vector representation using a neural network encoder. The encoded representations are then mapped to a latent space defined by a mixture of Gaussian distributions, wherein each Gaussian component corresponds to a latent entity or relation. A variational inference mechanism computes a posterior probability distribution over cluster assignments for each mention, enabling soft clustering. A knowledge graph embedding module encodes the structural relationships among the triples by learning joint representations of entities and relations that preserve the topology of the graph. A decoder reconstructs the input representations from the latent cluster assignments and learned embeddings. The entire system is trained using a combined loss function that includes a reconstruction term, a KL divergence term enforcing the prior distribution over latent variables, a knowledge graph embedding loss that aligns entity and relation representations with observed triples, and a side information loss that penalizes deviations from known equivalence constraints between mentions. Training is performed in two sequential phases: first, the encoder is optimized using weak supervision from hierarchical agglomerative clustering initialized with pretrained embeddings, and second, the decoder and embedding matrices are optimized while holding the encoder fixed to prevent information leakage. The method outputs a set of clustered noun phrases and relation phrases, where each cluster represents a canonicalized entity or relation, enabling downstream applications such as knowledge base completion, link prediction, and semantic search to operate on a unified, non-redundant representation of knowledge.

## DETAILED DESCRIPTION

- introduce knowledge graphs

Knowledge graphs are structured representations of factual knowledge in which entities are modeled as nodes and relationships between entities are modeled as directed edges labeled with relation types. These graphs serve as foundational structures for semantic reasoning, information retrieval, and artificial intelligence systems by encoding real-world facts in a machine-readable format. Knowledge graphs can be derived from structured databases, ontologies, or unstructured text through automated extraction techniques. Open knowledge graphs, in particular, are constructed from large-scale text corpora using open information extraction methods that do not rely on predefined schemas, allowing them to scale across domains and adapt to new contexts. However, due to the absence of canonicalization, such graphs often contain numerous surface variants of the same entity or relation, leading to fragmentation, redundancy, and reduced inferential power.

- motivate limitations of existing knowledge graphs

Existing knowledge graphs suffer from a fundamental limitation in their inability to recognize that multiple surface forms refer to the same underlying entity or relation. For instance, “New York City,” “NYC,” and “The Big Apple” are treated as distinct entities despite referring to the same geographic location, and “has headquarters in,” “is located in,” and “is based in” are considered unrelated relations despite sharing identical semantic intent. This lack of canonicalization results in a combinatorial explosion of redundant triples, increasing storage requirements and degrading query performance. It also prevents downstream systems from accurately answering questions that require cross-referencing equivalent mentions, such as “Where is NBC headquartered?” when the knowledge graph contains only “NBC-TV has main office in NYC.” Furthermore, the absence of unified representations impedes the effectiveness of link prediction, entity resolution, and reasoning tasks that rely on consistent semantic grounding.

- describe current solutions to adaptability problem

Current solutions to the adaptability problem in knowledge graphs primarily rely on entity linking, where mentions are mapped to entries in external knowledge bases such as Wikipedia or Wikidata. While effective for well-documented entities, these methods fail when mentions are novel, ambiguous, or absent from the target knowledge base. Alternative approaches employ rule-based clustering using string similarity, token overlap, or morphological normalization, but these techniques are brittle and do not generalize across domains. Other methods leverage pretrained word embeddings and hierarchical agglomerative clustering to group similar mentions, but they operate independently of the graph structure and lack the capacity to model uncertainty or polysemy. These approaches are typically pipeline-based, separating representation learning from clustering, which prevents mutual refinement and leads to suboptimal convergence.

- highlight deficiencies in current solutions

The deficiencies in current solutions stem from their inability to jointly model mention representations and cluster assignments in a differentiable, end-to-end manner. Pipeline architectures, such as those based on hierarchical agglomerative clustering following embedding learning, are unable to correct clustering errors through feedback from the embedding space. Rule-based methods are constrained by handcrafted features that do not adapt to new domains or linguistic variations. External knowledge base dependencies render them inapplicable to emerging or niche domains. Moreover, existing systems ignore the relational context of mentions, treating entities in isolation rather than as nodes embedded within a structured graph. This results in poor disambiguation of polysemous terms and an inability to leverage structural knowledge to resolve ambiguity. Finally, no prior method incorporates soft constraints from external side information in a differentiable, probabilistic framework that allows for uncertainty and partial agreement.

- introduce canonicalizing using variational autoencoders (CUVA)

Canonicalizing using variational autoencoders (CUVA) is a novel neural architecture designed to resolve the canonicalization problem in open knowledge graphs by jointly learning continuous representations of noun phrases and relation phrases and assigning them to latent clusters through variational inference. CUVA integrates a mixture of Gaussians in the latent space to model each canonical entity and relation, enabling soft clustering that accounts for ambiguity and partial similarity. The architecture comprises two variational autoencoders—one for entities and one for relations—each with an encoder that maps mentions to latent parameters and a decoder that reconstructs the input from cluster assignments. A knowledge graph embedding module enforces structural consistency by aligning entity and relation representations with observed triples, while a side information loss function incorporates external equivalence constraints as soft penalties. The entire system is trained end-to-end, allowing gradients to propagate between representation learning and clustering, resulting in mutually reinforcing optimization.

- describe CUVA architecture

The CUVA architecture consists of two parallel variational autoencoder modules: the Entity Variational Autoencoder (E-VAE) and the Relation Variational Autoencoder (R-VAE). Each module contains an encoder network that takes as input a high-dimensional embedding of a noun phrase or relation phrase and outputs the mean and variance parameters of a Gaussian distribution in the latent space. A reparameterization trick enables differentiable sampling from this distribution. The sampled latent vectors are then passed to a decoder that reconstructs the original input representation. A knowledge graph embedding module, implemented using the HolE algorithm, maps each entity and relation to a cluster centroid based on the soft assignment probabilities derived from the variational posteriors. These centroids are learned during training and serve as canonical representations. A side information module computes a mean squared error loss between the embeddings of known equivalent mention pairs, weighted by their plausibility scores. The total loss function combines reconstruction error, KL divergence, knowledge graph embedding loss, and side information loss, enabling joint optimization of all components.

- explain hierarchical agglomerative clustering

Hierarchical agglomerative clustering is a bottom-up clustering method that begins with each data point as its own cluster and iteratively merges the most similar pairs until a stopping criterion is met. In CUVA, this method is used during initialization to generate weak cluster labels for noun phrases and relation phrases based on pretrained GloVe embeddings. The clustering proceeds by computing pairwise distances between embeddings and merging clusters according to a complete linkage criterion, which uses the maximum distance between any two members of different clusters. The resulting dendrogram is cut at a threshold to produce flat clusters, which are then used to initialize the means and variances of the Gaussian components in the variational autoencoders. This initialization provides a data-driven starting point for the subsequent end-to-end training phase, reducing the risk of poor local optima.

- describe training of neural network

The neural network in CUVA is trained in two distinct phases to ensure stable convergence and prevent the decoder from bypassing the latent representation. In the first phase, the encoder is trained while the decoder and cluster centroids are held fixed. The training objective includes a negative log likelihood loss computed between the predicted cluster assignment probabilities and the hard cluster labels generated by hierarchical agglomerative clustering, along with a side information loss and L1 regularization. In the second phase, the encoder is frozen, and the decoder, cluster centroids, and embedding matrices are optimized using the evidence lower bound (ELBO) as the primary objective, which balances reconstruction accuracy with the KL divergence between the variational posterior and a Gaussian prior. The knowledge graph embedding loss and side information loss are retained in this phase to maintain structural and semantic consistency. This two-step strategy ensures that the decoder learns to reconstruct inputs solely from the latent space, reinforcing the role of the variational posterior as a meaningful representation of cluster membership.

- introduce encoder and decoder sections

The encoder section of CUVA transforms each noun phrase or relation phrase into a probabilistic latent representation by mapping its embedding to a mean and diagonal covariance matrix of a Gaussian distribution. This mapping is implemented through a multilayer perceptron with nonlinear activation functions, producing a distribution over latent variables that captures uncertainty in the mention’s true identity. The decoder section reconstructs the original input representation from the sampled latent variable and the cluster centroid corresponding to the most probable assignment. The decoder is implemented as a linear projection layer followed by a nonlinear transformation, enabling the model to learn how each canonical entity or relation is expressed in surface form. The encoder and decoder are symmetric in architecture but operate in opposite directions, forming a variational autoencoder that learns to compress and reconstruct mention representations while preserving cluster structure.

- describe constraint loss

The constraint loss in CUVA is a regularization term derived from side information that encodes known equivalences between noun phrases or relation phrases. These equivalences are obtained from external sources such as entity linking, paraphrase databases, or morphological normalization and are associated with plausibility scores indicating confidence in their correctness. The constraint loss computes the mean squared error between the embeddings of each equivalent pair, weighted by their plausibility score, and sums these errors across all known pairs. This loss acts as a soft constraint that pulls semantically equivalent mentions closer in the embedding space while allowing for uncertainty and partial agreement, thereby guiding the model toward more accurate canonicalizations without requiring exact labels.

- explain knowledge graph embedding module

The knowledge graph embedding module in CUVA leverages the structural topology of the open knowledge graph to align entity and relation representations with observed triples. For each triple (h, r, t), the module computes soft cluster assignments for the head, relation, and tail using the variational posteriors from the E-VAE and R-VAE. These assignments are converted into one-hot-like vectors via a soft argmax operation with a high temperature parameter, enabling differentiable selection of the most probable cluster. The module then retrieves the corresponding centroid embeddings for each cluster and applies the HolE algorithm to compute a scoring function that evaluates the plausibility of the triple given the learned representations. The difference between the predicted score and the expected score (based on observed triples) forms the knowledge graph embedding loss, which ensures that the learned embeddings preserve the relational structure of the graph and improve the coherence of canonicalized clusters.

- describe training of neural network in two steps

The training of the neural network in CUVA is performed in two sequential steps to decouple the learning of representation and reconstruction, thereby preventing the decoder from bypassing the latent space. In the first step, the encoder is trained using weak supervision from hierarchical agglomerative clustering, minimizing a negative log likelihood loss between predicted cluster probabilities and hard cluster labels, along with side information and regularization losses. In the second step, the encoder is frozen, and the decoder, cluster centroids, and embedding matrices are optimized using the evidence lower bound, which includes reconstruction, KL divergence, knowledge graph embedding, and side information losses. This two-step strategy ensures that the latent space is first grounded in meaningful cluster structure before the decoder learns to reconstruct inputs from it, promoting stable convergence and preventing mode collapse.

- introduce constraint-based loss

The constraint-based loss in CUVA is a differentiable penalty term that enforces semantic consistency between known equivalent mentions by minimizing the distance between their embeddings in the latent space. Unlike hard constraints that force exact equivalence, this loss is probabilistic and weighted by plausibility scores derived from external sources such as entity linking or paraphrase databases. It allows the model to accommodate uncertainty and partial agreement, making it robust to noisy or conflicting side information. The constraint-based loss is integrated into the overall training objective alongside reconstruction, KL divergence, and knowledge graph embedding losses, enabling the model to learn canonical representations that are both structurally coherent and semantically aligned with external knowledge.

- describe encoding side information

Encoding side information in CUVA involves transforming external equivalence constraints into a differentiable loss function that operates on the learned embedding space. Each known pair of equivalent mentions is mapped to their respective embeddings, and the mean squared error between them is computed and scaled by a plausibility score reflecting the confidence in the equivalence. These weighted errors are aggregated across all known pairs to form the side information loss, which is added to the total training objective. The side information is not used to hard-assign clusters but to guide the optimization of the embedding space, encouraging similar mentions to converge toward the same latent region while preserving the flexibility to model polysemy and ambiguity.

- explain operational steps for training neural network

The operational steps for training the neural network in CUVA begin with preprocessing the input triples and extracting contextual side information from external sources. Pretrained GloVe embeddings are computed for all noun phrases and relation phrases, and hierarchical agglomerative clustering is applied to generate initial cluster labels. The variational autoencoders are initialized with cluster means and variances derived from these clusters. Training proceeds in two stages: first, the encoder is optimized using the hard cluster labels as supervision, minimizing negative log likelihood, side information loss, and L1 regularization. Second, the encoder is frozen, and the decoder, cluster centroids, and embedding matrices are optimized using the evidence lower bound, which includes reconstruction, KL divergence, knowledge graph embedding, and side information losses. The model is trained using the Adam optimizer with a learning rate schedule, and training terminates after a fixed number of epochs or when validation performance plateaus. The final output is a set of canonicalized clusters, each represented by a latent Gaussian distribution and its associated centroid embedding.

- summarize benefits of CUVA

The benefits of CUVA include its ability to perform end-to-end canonicalization of both noun phrases and relation phrases in an unsupervised manner, eliminating the need for external knowledge bases or handcrafted features. By integrating variational inference with knowledge graph embedding, CUVA learns representations that are both semantically coherent and structurally consistent, enabling accurate disambiguation of polysemous mentions and resolution of ambiguous relations. The two-step training strategy ensures stable convergence and prevents information leakage, while the constraint-based loss allows for flexible incorporation of external side information without requiring exact labels. CUVA outperforms existing state-of-the-art methods across multiple benchmarks, achieves new results on a novel dataset, and demonstrates superior generalization compared to pretrained language models. Its modular design allows for easy adaptation to new domains, and its probabilistic nature provides uncertainty estimates for cluster assignments, enhancing interpretability and reliability in downstream applications.

### Example Table 1

- introduce example table

Example Table 1 presents the clustering results generated by CUVA on a subset of the ReVerb45K dataset, illustrating the grouping of noun phrases and relation phrases into canonical clusters based on latent semantic similarity rather than surface form. Each row corresponds to a distinct cluster, with the entries within each row representing surface variants that have been assigned to the same latent entity or relation. The table demonstrates CUVA’s ability to correctly group mentions such as “kodagu” and “coorg,” which refer to the same Indian district despite differing orthographically, and “NBC-TV” and “NBC Television,” which denote the same media organization. It also reveals the model’s capacity to identify relation equivalences such as “be associate with,” “have be affiliate to,” and “be now associate with,” which share the same semantic intent despite lexical variation.

- describe clustering entities, noun phrases, and relation phrases

Clustering entities and noun phrases in CUVA is performed by mapping each mention to a latent Gaussian distribution, where the mean and variance parameters are learned during training. The cluster assignment for each mention is determined by the posterior probability over the mixture components, with the highest probability indicating the most likely canonical entity. Similarly, relation phrases are clustered using a parallel variational autoencoder that operates on relation embeddings, ensuring that relations with similar argument structures and semantic roles are grouped together. The clustering is soft, meaning that each mention may have non-zero probability across multiple clusters, allowing the model to account for ambiguity and partial overlap. This dual clustering mechanism enables CUVA to canonicalize both entities and relations simultaneously, preserving the integrity of the triple structure.

- introduce variational autoencoder for entities

The variational autoencoder for entities in CUVA is responsible for encoding noun phrases into a latent space defined by a mixture of K Gaussian distributions, each corresponding to a canonical entity. The encoder maps each noun phrase embedding to a mean and diagonal covariance vector, from which a latent variable is sampled using the reparameterization trick. The decoder reconstructs the original embedding from this latent variable and the centroid of the assigned cluster. The training objective includes a reconstruction loss, a KL divergence term that regularizes the posterior toward a unit Gaussian prior, and a constraint loss derived from side information. This architecture enables the model to learn dense, probabilistic representations of entities that capture both semantic similarity and uncertainty, allowing for accurate canonicalization even in the presence of noisy or incomplete data.

- introduce variational autoencoder for relations

The variational autoencoder for relations in CUVA performs an analogous function to the entity encoder but operates on relation phrases instead of noun phrases. It maps each relation phrase into a latent space composed of K Gaussian components, each representing a canonical relation type. The encoder outputs the mean and variance of the posterior distribution over these components, and the decoder reconstructs the relation embedding from the sampled latent variable and the corresponding centroid. The training process is identical to that of the entity VAE, incorporating reconstruction, KL divergence, and side information losses. This dual VAE structure ensures that both entities and relations are canonicalized in a coordinated manner, preserving the structural consistency of triples and enabling accurate modeling of relational semantics.

- describe module for knowledge base completion

The module for knowledge base completion in CUVA leverages the canonicalized entity and relation representations to predict missing triples within the knowledge graph. By embedding each entity and relation into a continuous space where semantically similar items are proximal, the model can compute plausibility scores for unobserved triples using the HolE scoring function. For example, if the model has canonicalized “New York City” and “NYC” into the same entity and “has headquarters in” and “is located in” into the same relation, it can infer that “NBC-TV is located in NYC” is likely true even if it was not explicitly stated. This capability enables CUVA to serve not only as a canonicalization tool but also as a knowledge base completion system, enhancing the completeness and inferential power of open knowledge graphs.

- describe training of resulting neural network architecture

The training of the resulting neural network architecture in CUVA is conducted in two distinct phases to ensure stable convergence and proper utilization of the latent space. In the first phase, the encoder networks for entities and relations are trained using hard cluster labels derived from hierarchical agglomerative clustering, minimizing a negative log likelihood loss between predicted and true cluster assignments, along with side information and L1 regularization. In the second phase, the encoder is frozen, and the decoder, cluster centroids, and embedding matrices are optimized using the evidence lower bound, which combines reconstruction accuracy, KL divergence, knowledge graph embedding loss, and side information loss. This two-step strategy ensures that the latent space is first grounded in meaningful structure before the decoder learns to reconstruct from it, preventing mode collapse and promoting robust learning.

- describe building hierarchical agglomerative cluster model

The hierarchical agglomerative cluster model in CUVA is built by first computing pairwise distances between pretrained GloVe embeddings of all noun phrases and relation phrases. Using a complete linkage criterion, the algorithm iteratively merges the closest clusters until a stopping threshold is reached. The resulting dendrogram is then cut at a specified height to produce flat clusters, which are used to initialize the mean and variance parameters of the Gaussian components in the variational autoencoders. This initialization provides a data-driven, unsupervised starting point for training, reducing the sensitivity to random initialization and accelerating convergence.

- describe training encoder section

The training of the encoder section in CUVA occurs during the first phase of the two-step training strategy. The encoder maps each noun phrase or relation phrase to a mean and variance vector in the latent space, and the training objective is to maximize the likelihood of the cluster assignments generated by hierarchical agglomerative clustering. This is achieved by minimizing the negative log likelihood of the predicted cluster probabilities under the hard cluster labels, while also incorporating side information loss and L1 regularization to encourage smooth and consistent embeddings. The encoder is trained using the Adam optimizer with a learning rate of 1e-3, and training continues for a fixed number of epochs or until validation performance stabilizes.

- describe training decoder section

The training of the decoder section in CUVA occurs during the second phase of training, after the encoder has been frozen. The decoder reconstructs the input embedding from the sampled latent variable and the cluster centroid corresponding to the most probable assignment. The training objective is to minimize the evidence lower bound, which includes the reconstruction loss, the KL divergence between the variational posterior and the prior, the knowledge graph embedding loss, and the side information loss. The decoder is optimized using the Adam optimizer with a reduced learning rate of 1e-4 to ensure fine-grained adjustments. This phase ensures that the latent space is fully utilized and that the decoder learns to generate accurate reconstructions solely from the learned cluster assignments.

- describe calculating total loss value

The total loss value in CUVA is computed as the weighted sum of five components: the reconstruction loss, which measures the discrepancy between the input embedding and its reconstruction; the KL divergence between the variational posterior and a unit Gaussian prior; the knowledge graph embedding loss, which enforces structural consistency with observed triples; the side information loss, which penalizes deviations from known equivalence constraints; and the L1 regularization loss, which encourages sparsity in the model parameters. Each component is scaled by a hyperparameter that balances its contribution to the overall objective. The total loss is minimized using gradient descent, and the model parameters are updated iteratively until convergence.

- describe FIG. 2A

FIG. 2A illustrates the architecture of the encoder section of the Entity Variational Autoencoder in CUVA, showing the flow of input noun phrase embeddings through a multilayer perceptron to produce mean and variance parameters of a Gaussian distribution in the latent space. The figure depicts the input layer, two hidden layers with tanh nonlinearities, and the output layer that splits into two branches for mean and variance. The reparameterization trick is shown as a stochastic sampling step that generates a latent variable z from the distribution defined by the mean and variance. This figure emphasizes the probabilistic nature of the encoding process and its role in enabling soft clustering.

- describe FIG. 2B

FIG. 2B illustrates the architecture of the decoder section of the Entity Variational Autoencoder in CUVA, depicting how the latent variable z is combined with the cluster centroid corresponding to the most probable assignment to reconstruct the original noun phrase embedding. The figure shows the latent variable being concatenated with the centroid embedding, followed by a linear transformation and a nonlinear activation to produce the reconstructed input. This structure ensures that the decoder learns to generate surface forms from canonical representations, reinforcing the link between latent clusters and observed mentions.

- describe FIG. 2C

FIG. 2C presents the integration of the knowledge graph embedding module with the variational autoencoders in CUVA, showing how the soft cluster assignments for the head, relation, and tail of a triple are used to retrieve their corresponding centroid embeddings. The HolE scoring function is applied to these centroids to compute the plausibility of the triple, and the difference between the predicted and expected scores forms the knowledge graph embedding loss. This figure highlights the bidirectional flow of information between the clustering and embedding components, enabling mutual refinement of canonical representations and structural consistency.

- describe core structure of CUVA

The core structure of CUVA consists of two parallel variational autoencoders—one for entities and one for relations—each with an encoder that maps mentions to a latent Gaussian distribution and a decoder that reconstructs the input from the latent variable and cluster centroid. These autoencoders are coupled with a knowledge graph embedding module that enforces structural consistency across triples and a side information module that incorporates external equivalence constraints. The entire system is trained in two phases: first, the encoders are optimized using weak supervision from hierarchical clustering; second, the decoders and centroids are optimized using the evidence lower bound. This structure enables end-to-end canonicalization of both entities and relations in a probabilistic, differentiable framework.

- describe encoder block

The encoder block in CUVA is a multilayer perceptron that takes as input a high-dimensional embedding of a noun phrase or relation phrase and outputs the mean and diagonal covariance parameters of a Gaussian distribution in the latent space. It consists of two fully connected layers with tanh nonlinearities, reducing the dimensionality from 768 to 384 and then to 100. The output is split into two vectors: one for the mean and one for the log-variance, which are used to sample a latent variable via the reparameterization trick. The encoder is trained to maximize the likelihood of the cluster assignments, ensuring that similar mentions are mapped to nearby regions in the latent space.

- describe decoder block

The decoder block in CUVA reconstructs the input embedding from the latent variable and the cluster centroid corresponding to the most probable assignment. It consists of a linear transformation layer followed by a tanh activation, mapping the concatenated latent variable and centroid embedding back to the original input dimensionality. The decoder is trained to minimize the reconstruction error between the original input and the output, ensuring that the latent space encodes meaningful canonical representations. The decoder is only trained in the second phase of training, after the encoder has been frozen, to prevent information leakage.

- describe encoding side information

Encoding side information in CUVA involves transforming known equivalence pairs into a differentiable loss function that operates on the learned embedding space. Each pair of equivalent mentions is mapped to their respective embeddings, and the mean squared error between them is computed and weighted by a plausibility score derived from external sources. These weighted errors are summed across all pairs to form the side information loss, which is added to the total training objective. This process allows the model to incorporate external knowledge without requiring exact labels, enabling robust canonicalization in the presence of uncertainty.

- describe FIG. 3

FIG. 3 illustrates the flow of information through the Knowledge Graph Embedding Module in CUVA, showing how the soft cluster assignments for the head, relation, and tail of a triple are converted into one-hot-like vectors using a soft argmax operation with a high temperature. These vectors are then used to retrieve the corresponding centroid embeddings from the learned embedding matrices, which are fed into the HolE scoring function to compute the plausibility of the triple. The difference between the predicted and expected scores forms the knowledge graph embedding loss, which is backpropagated to update the centroids and embeddings. This figure demonstrates the integration of clustering and embedding in a differentiable manner.

- describe FIG. 4

FIG. 4 depicts the computation of the side information loss in CUVA, showing two equivalent noun phrases, “NYC” and “New York City,” being mapped to their respective embeddings. The mean squared error between these embeddings is computed and scaled by a plausibility score of 0.85, derived from an entity linking system. This weighted error is added to the total loss function, pulling the two embeddings closer in the latent space. The figure highlights the model’s ability to incorporate external constraints without requiring hard assignments, preserving flexibility and robustness.

- describe FIG. 5

FIG. 5 presents the two-step training strategy of CUVA, contrasting the first phase, where the encoder is trained using hard cluster labels from hierarchical agglomerative clustering, with the second phase, where the decoder and centroids are optimized using the evidence lower bound. The figure illustrates the freezing of the encoder in the second phase and the flow of gradients through the decoder, centroids, and embedding matrices. This visual representation emphasizes the architectural separation that enables stable convergence and prevents the decoder from bypassing the latent space.

- describe FIG. 6

FIG. 6 illustrates the overall system architecture of CUVA, showing the integration of the Entity VAE, Relation VAE, Knowledge Graph Embedding Module, and Side Information Module within a unified training framework. The figure depicts the input triples and side information flowing into their respective components, with the total loss being computed as the sum of reconstruction, KL divergence, knowledge graph embedding, side information, and regularization losses. The output is a set of canonicalized clusters for entities and relations, ready for use in downstream applications. This figure provides a comprehensive overview of CUVA’s end-to-end design.

- describe receiving information

Receiving information in CUVA involves ingesting a collection of triples extracted from unstructured text, along with contextual side information derived from external sources such as entity linking systems, paraphrase databases, and morphological normalization tools. The triples are preprocessed to extract noun phrases and relation phrases, which are then embedded using pretrained GloVe vectors. The side information is parsed into equivalence pairs with associated plausibility scores, forming the basis for the constraint-based loss. This structured input enables CUVA to learn canonical representations that are both semantically grounded and contextually informed.

- describe dynamically clustering received information

Dynamically clustering received information in CUVA is achieved through the variational autoencoder’s probabilistic assignment mechanism, which allows each mention to have a soft membership across multiple latent clusters. During training, the model continuously refines these assignments based on the learned embeddings, cluster centroids, and structural constraints. This dynamic process enables the system to adapt to new data, resolve ambiguities, and correct errors introduced by noisy side information. Unlike static clustering methods, CUVA’s assignments evolve throughout training, resulting in more accurate and coherent canonicalizations.

- describe initializing cluster means and cluster variances

Initializing cluster means and cluster variances in CUVA is performed using hierarchical agglomerative clustering applied to pretrained GloVe embeddings of noun phrases and relation phrases. The clustering algorithm produces flat clusters, and the mean and variance of each cluster are computed and used to initialize the corresponding Gaussian components in the variational autoencoders. This initialization provides a data-driven starting point that reduces the sensitivity to random initialization and accelerates convergence during the subsequent end-to-end training phase.

- describe dynamically training neural network

Dynamically training the neural network in CUVA involves iteratively updating the model parameters using gradient descent on a composite loss function that evolves during training. In the first phase, the encoder is trained using hard cluster labels from hierarchical clustering, while in the second phase, the decoder and centroids are optimized using the evidence lower bound. The model adapts its representations in response to the interplay between reconstruction, structural, and constraint-based losses, allowing for continuous refinement of cluster assignments and embedding quality. This dynamic training process ensures that the model converges to a solution that is both semantically coherent and structurally consistent.

- describe using two loss functions

Using two loss functions in CUVA refers to the separation of training into two distinct phases, each with its own primary objective. In the first phase, the negative log likelihood loss is used to align the encoder’s cluster predictions with hard labels from hierarchical clustering. In the second phase, the evidence lower bound is used to optimize the decoder and centroids by balancing reconstruction accuracy, KL divergence, knowledge graph embedding, and side information losses. This dual-loss strategy ensures that the latent space is first grounded in meaningful structure before the model learns to reconstruct from it, preventing mode collapse and promoting robust learning.

- describe training neural network in two steps

Training the neural network in CUVA in two steps ensures that the encoder and decoder learn complementary roles without interference. In the first step, the encoder is trained to predict cluster assignments using weak supervision from hierarchical clustering, while the decoder is held fixed. In the second step, the encoder is frozen, and the decoder is trained to reconstruct inputs from the latent space using the evidence lower bound. This separation prevents the decoder from bypassing the latent representation and forces the model to learn meaningful, structured encodings. The two-step strategy is critical to the stability and effectiveness of CUVA’s end-to-end canonicalization.

- describe using additional constraint-based loss

Using additional constraint-based loss in CUVA introduces external knowledge into the training process in a differentiable and probabilistic manner. The constraint-based loss penalizes deviations between the embeddings of known equivalent mentions, weighted by their plausibility scores. This loss is added to the total objective function and is optimized alongside reconstruction, KL divergence, and knowledge graph embedding losses. By incorporating this constraint, CUVA leverages external evidence to guide canonicalization without requiring exact labels, enhancing accuracy and robustness in the presence of uncertainty.

- describe encoding structural knowledge

Encoding structural knowledge in CUVA is achieved through the knowledge graph embedding module, which uses the HolE algorithm to align entity and relation representations with the observed topology of the knowledge graph. For each triple, the model computes soft cluster assignments and retrieves the corresponding centroid embeddings, which are then scored for plausibility. The difference between the predicted and expected scores forms the embedding loss, which is backpropagated to update the centroids and embeddings. This process ensures that the learned representations preserve the relational structure of the graph, enabling accurate canonicalization that is informed by global context.

### Further Comments and/or Embodiments

- describe benefits of CUVA

The benefits of CUVA include its ability to perform end-to-end canonicalization of both noun phrases and relation phrases in an unsupervised manner, eliminating the need for external knowledge bases or handcrafted features. By integrating variational inference with knowledge graph embedding, CUVA learns representations that are both semantically coherent and structurally consistent, enabling accurate disambiguation of polysemous mentions and resolution of ambiguous relations. The two-step training strategy ensures stable convergence and prevents information leakage, while the constraint-based loss allows for flexible incorporation of external side information without requiring exact labels. CUVA outperforms existing state-of-the-art methods across multiple benchmarks, achieves new results on a novel dataset, and demonstrates superior generalization compared to pretrained language models. Its modular design allows for easy adaptation to new domains, and its probabilistic nature provides uncertainty estimates for cluster assignments, enhancing interpretability and reliability in downstream applications.

- describe deficiencies in current state of art

The deficiencies in the current state of the art include the reliance on pipeline architectures that separate representation learning from clustering, preventing mutual refinement and leading to suboptimal convergence. Existing methods are either dependent on external knowledge bases that are incomplete or outdated, or they rely on rule-based features that do not generalize across domains. They lack the capacity to model uncertainty, handle polysemy, or incorporate relational context in a unified framework. Additionally, no prior system integrates variational autoencoding with knowledge graph embedding in a differentiable, end-to-end manner, resulting in fragmented, redundant, and semantically inconsistent knowledge graphs.

- introduce CANONICNELL dataset

The CANONICNELL dataset is a novel benchmark for entity canonicalization constructed from the 165th iteration snapshot of the NELL (Never-Ending Language Learner) knowledge base. It was created by identifying co-referent entity pairs using a soft-truth scoring mechanism and applying depth-first search to extract connected components as gold-standard clusters. The dataset contains triples from NELL165 filtered to include only those with head or tail entities belonging to these clusters, ensuring that canonicalization is evaluated on semantically grounded, real-world knowledge. Unlike existing benchmarks derived from ReVerb, CANONICNELL is independent of external extraction systems, providing a more diverse and challenging testbed for canonicalization algorithms. CUVA achieves state-of-the-art performance on this dataset, demonstrating its generalizability and robustness across domains.

### 1. INTRODUCTION

- introduce knowledge graphs

Knowledge graphs are structured representations of factual knowledge in which entities are modeled as nodes and relationships between entities are modeled as directed edges labeled with relation types. These graphs serve as foundational structures for semantic reasoning, information retrieval, and artificial intelligence systems by encoding real-world facts in a machine-readable format. Knowledge graphs can be derived from structured databases, ontologies, or unstructured text through automated extraction techniques. Open knowledge graphs, in particular, are constructed from large-scale text corpora using open information extraction methods that do not rely on predefined schemas, allowing them to scale across domains and adapt to new contexts. However, due to the absence of canonicalization, such graphs often contain numerous surface variants of the same entity or relation, leading to fragmentation, redundancy, and reduced inferential power.

- motivate limitations of existing knowledge graphs

Existing knowledge graphs suffer from a fundamental limitation in their inability to recognize that multiple surface forms refer to the same underlying entity or relation. For instance, “New York City,” “NYC,” and “The Big Apple” are treated as distinct entities despite referring to the same geographic location, and “has headquarters in,” “is located in,” and “is based in” are considered unrelated relations despite sharing identical semantic intent. This lack of canonicalization results in a combinatorial explosion of redundant triples, increasing storage requirements and degrading query performance. It also prevents downstream systems from accurately answering questions that require cross-referencing equivalent mentions, such as “Where is NBC headquartered?” when the knowledge graph contains only “NBC-TV has main office in NYC.” Furthermore, the absence of unified representations impedes the effectiveness of link prediction, entity resolution, and reasoning tasks that rely on consistent semantic grounding.

- describe open information extraction methods

Open information extraction methods are automated techniques for extracting structured triples from unstructured text without requiring a predefined schema or ontology. These methods, including REVERB, OLLIE, and RNNOIE, identify noun phrases as entities and relation phrases as predicates, producing triples in the form (subject, predicate, object). They are highly adaptable to new domains and scalable to massive text corpora, making them ideal for constructing open knowledge graphs. However, their outputs are inherently noisy and fragmented, as they treat surface forms as distinct even when they refer to the same entity or relation. This limitation undermines the utility of the resulting knowledge graphs for semantic reasoning and downstream applications.

- highlight shortcomings of open knowledge graphs

The shortcomings of open knowledge graphs stem from their failure to canonicalize entities and relations, leading to redundancy, inconsistency, and poor generalization. Multiple surface forms of the same entity—such as “IBM,” “International Business Machines,” and “Big Blue”—are treated as separate nodes, fragmenting knowledge and inflating graph size. Similarly, semantically equivalent relations like “is located in” and “has headquarters in” are not recognized as interchangeable, preventing accurate inference. These issues degrade query performance, increase storage costs, and hinder the development of reliable AI systems that depend on coherent, unified knowledge representations.

- introduce canonicalizing using variational autoencoders (CUVA)

Canonicalizing using variational autoencoders (CUVA) is a novel neural architecture designed to resolve the canonicalization problem in open knowledge graphs by jointly learning continuous representations of noun phrases and relation phrases and assigning them to latent clusters through variational inference. CUVA integrates a mixture of Gaussians in the latent space to model each canonical entity and relation, enabling soft clustering that accounts for ambiguity and partial similarity. The architecture comprises two variational autoencoders—one for entities and one for relations—each with an encoder that maps mentions to latent parameters and a decoder that reconstructs the input from cluster assignments. A knowledge graph embedding module enforces structural consistency by aligning entity and relation representations with observed triples, while a side information loss function incorporates external equivalence constraints as soft penalties. The entire system is trained end-to-end, allowing gradients to propagate between representation learning and clustering, resulting in mutually reinforcing optimization.

- summarize contributions

The contributions of this invention include: (1) the introduction of CUVA, a novel neural architecture that performs end-to-end canonicalization of both noun phrases and relation phrases in open knowledge graphs using variational autoencoders; (2) the integration of knowledge graph embedding and side information as differentiable constraints within a unified training framework; (3) a two-step training strategy that ensures stable convergence and prevents information leakage; (4) the creation of CANONICNELL, a novel benchmark for entity canonicalization derived from the NELL knowledge base; and (5) empirical demonstration of state-of-the-art performance across four benchmarks, surpassing existing methods by significant margins.

### 2. RELATED WORK

- introduce OpenIE technique

The Open Information Extraction (OpenIE) technique is a methodology for extracting structured triples from unstructured text without requiring a predefined schema or ontology. It enables the automatic construction of large-scale knowledge graphs from diverse text corpora by identifying noun phrases as entities and relation phrases as predicates. OpenIE systems such as REVERB, OLLIE, and RNNOIE have been widely adopted due to their adaptability and scalability. However, these systems produce surface-level extractions that do not canonicalize entities or relations, resulting in fragmented and redundant knowledge graphs that are difficult to use for semantic reasoning.

- summarize rule-based and learning-based approaches

Rule-based approaches to OpenIE, such as REVERB and PREDPATT, rely on syntactic patterns and linguistic heuristics to extract triples, offering high precision but limited coverage. Learning-based approaches, such as OLLIE and RNNOIE, use self-supervised training to improve recall by learning from bootstrapped examples, but they still produce non-canonicalized outputs. Both types of approaches treat surface forms as distinct, failing to recognize semantic equivalence. Subsequent efforts to canonicalize these outputs have relied on external knowledge bases or heuristic clustering, which are either domain-limited or lack the capacity to model uncertainty and polysemy.

- discuss limitations of existing EL approaches

Existing Entity Linking (EL) approaches attempt to map noun phrases to entries in external knowledge bases such as Wikipedia or Wikidata. While effective for well-documented entities, these methods fail when mentions are novel, ambiguous, or absent from the target knowledge base. They are also brittle to linguistic variation and require extensive preprocessing and disambiguation pipelines. Furthermore, EL systems cannot canonicalize relation phrases, and their reliance on external resources makes them unsuitable for domains with limited or no prior knowledge. These limitations render EL approaches inadequate for the canonicalization of open knowledge graphs.

### 3. OPEN KGS CANONICALIZATION USING VAE

- define CANONICALIZATION task

The CANONICALIZATION task is defined as the process of grouping noun phrases and relation phrases from an open knowledge graph into clusters such that all mentions referring to the same underlying entity or relation are assigned to the same cluster, regardless of surface form. The goal is to produce a non-redundant, semantically coherent representation of knowledge where each cluster corresponds to a latent entity or relation, and the label of the cluster is unknown during training. This task is inherently unsupervised and requires modeling both semantic similarity and structural context to resolve ambiguity and polysemy.

- introduce CUVA architecture

The CUVA architecture introduces a novel framework for canonicalization by combining variational autoencoders with knowledge graph embedding and side information constraints. It consists of two parallel variational autoencoders—one for entities and one for relations—each with an encoder that maps mentions to a latent Gaussian distribution and a decoder that reconstructs the input from the latent variable and cluster centroid. A knowledge graph embedding module enforces structural consistency, and a side information loss function incorporates external equivalence constraints. The entire system is trained end-to-end using a two-step strategy, enabling joint optimization of representations and cluster assignments.

- describe E-VAE and R-VAE components

The Entity Variational Autoencoder (E-VAE) and Relation Variational Autoencoder (R-VAE) are the core components of CUVA, responsible for encoding noun phrases and relation phrases into a latent space defined by a mixture of Gaussians. Each component contains an encoder that maps an input embedding to a mean and variance vector, a reparameterization step that samples a latent variable, and a decoder that reconstructs the input from the latent variable and the centroid of the assigned cluster. The E-VAE and R-VAE operate in parallel, ensuring that both entities and relations are canonicalized in a coordinated manner, preserving the integrity of the triple structure.

- explain Gaussian distribution modeling

Gaussian distribution modeling in CUVA represents each canonical entity or relation as a multivariate Gaussian distribution in the latent space, with a mean vector and diagonal covariance matrix. The mean corresponds to the canonical representation, and the variance captures uncertainty in the mention’s true identity. During training, each mention is mapped to a posterior distribution over these Gaussians, and the cluster assignment is determined by the highest posterior probability. This probabilistic modeling enables soft clustering, allowing mentions to have partial membership across multiple clusters and accommodating ambiguity and polysemy.

- describe KGE module

The Knowledge Graph Embedding (KGE) module in CUVA uses the HolE algorithm to encode the structural relationships among entities and relations. It retrieves the centroid embeddings for the most probable clusters of the head, relation, and tail of each triple and computes a plausibility score for the triple. The difference between the predicted and expected scores forms the KGE loss, which is backpropagated to update the centroids and embeddings. This ensures that the learned representations preserve the topology of the knowledge graph, enhancing the coherence of canonicalized clusters.

- introduce Variational AutoEncoder

A Variational Autoencoder (VAE) is a generative model that learns to encode data into a latent space and reconstruct it from that space using probabilistic inference. In CUVA, the VAE is adapted to model clusters of mentions as Gaussian distributions, enabling soft clustering and uncertainty modeling. The encoder maps an input to a latent distribution, and the decoder reconstructs the input from a sample drawn from that distribution. The training objective is to maximize the evidence lower bound, which balances reconstruction accuracy with the KL divergence between the posterior and a prior distribution.

- explain generative process

The generative process in CUVA begins by selecting a cluster from a categorical distribution, then sampling a latent vector from a Gaussian distribution associated with that cluster, and finally generating an observed mention by sampling from another Gaussian distribution parameterized by the latent vector. This process models how surface forms arise from underlying canonical entities or relations. During training, the model learns to invert this process: given a mention, it infers the most likely cluster and latent vector that could have generated it, enabling accurate canonicalization.

- describe Encoder block

The Encoder block in CUVA takes as input a high-dimensional embedding of a noun phrase or relation phrase and outputs the mean and variance parameters of a Gaussian distribution in the latent space. It consists of two fully connected layers with tanh nonlinearities, reducing the dimensionality from 768 to 384 and then to 100. The output is split into two vectors: one for the mean and one for the log-variance, which are used to sample a latent variable via the reparameterization trick. The encoder is trained to maximize the likelihood of the cluster assignments, ensuring that similar mentions are mapped to nearby regions in the latent space.

- explain Decoder block

The Decoder block in CUVA reconstructs the input embedding from the latent variable and the cluster centroid corresponding to the most probable assignment. It consists of a linear transformation layer followed by a tanh activation, mapping the concatenated latent variable and centroid embedding back to the original input dimensionality. The decoder is trained to minimize the reconstruction error between the original input and the output, ensuring that the latent space encodes meaningful canonical representations. The decoder is only trained in the second phase of training, after the encoder has been frozen, to prevent information leakage.

- describe cluster assignment

Cluster assignment in CUVA is performed by computing the posterior probability of each mention belonging to each Gaussian component in the latent space. The assignment is soft, meaning that each mention has a non-zero probability across all clusters, allowing for uncertainty and polysemy. During inference, the cluster with the highest posterior probability is selected as the canonical assignment. This soft assignment mechanism enables the model to handle ambiguous mentions and partial overlaps, improving robustness and accuracy.

- introduce KGE module

The KGE module in CUVA leverages the HolE algorithm to encode the structural relationships among entities and relations by aligning their latent representations with the observed triples in the knowledge graph. It retrieves the centroid embeddings for the most probable clusters of the head, relation, and tail of each triple and computes a plausibility score for the triple. The difference between the predicted and expected scores forms the KGE loss, which is backpropagated to update the centroids and embeddings. This ensures that the learned representations preserve the topology of the knowledge graph, enhancing the coherence of canonicalized clusters.

- describe side information

Side information in CUVA consists of known equivalence pairs between noun phrases or relation phrases, derived from external sources such as entity linking systems, paraphrase databases, and morphological normalization tools. Each pair is associated with a plausibility score indicating confidence in the equivalence. These pairs are used to compute a constraint-based loss that penalizes deviations between the embeddings of equivalent mentions, guiding the model toward more accurate canonicalizations without requiring exact labels.

- explain Side Information Loss

The Side Information Loss in CUVA is a differentiable penalty term that measures the mean squared error between the embeddings of known equivalent mentions, weighted by their plausibility scores. This loss is aggregated across all known pairs and added to the total training objective. It enables the model to incorporate external knowledge in a probabilistic manner, allowing for uncertainty and partial agreement. The Side Information Loss enhances canonicalization accuracy by pulling semantically equivalent mentions closer in the embedding space, even when their surface forms differ significantly.

### 4. Training Strategy

- describe initializing mixture of Gaussians

Initializing the mixture of Gaussians in CUVA involves computing pretrained GloVe embeddings for all noun phrases and relation phrases and applying hierarchical agglomerative clustering to group them into initial clusters. The mean and variance of each cluster are then used to initialize the corresponding Gaussian components in the variational autoencoders. This data-driven initialization provides a strong starting point for training, reducing sensitivity to random initialization and accelerating convergence.

- explain HAC clustering method

Hierarchical Agglomerative Clustering (HAC) is a bottom-up clustering method that begins with each mention as its own cluster and iteratively merges the most similar pairs using a complete linkage criterion. In CUVA, HAC is applied separately to noun phrase and relation phrase embeddings to generate initial cluster assignments. The resulting dendrogram is cut at a threshold to produce flat clusters, which are used to initialize the means and variances of the Gaussian components in the variational autoencoders.

- describe two-step training strategy

The two-step training strategy in CUVA separates the optimization of the encoder and decoder to ensure stable convergence. In the first step, the encoder is trained using hard cluster labels from hierarchical clustering, minimizing negative log likelihood and side information loss. In the second step, the encoder is frozen, and the decoder, cluster centroids, and embedding matrices are optimized using the evidence lower bound. This strategy prevents the decoder from bypassing the latent space and ensures that the latent representations are meaningful and structured.

- explain Encoder training

Encoder training in CUVA occurs in the first training phase, where the encoder maps each mention to a latent Gaussian distribution and is optimized to predict the hard cluster labels generated by hierarchical agglomerative clustering. The training objective includes a negative log likelihood loss, a side information loss, and L1 regularization. The encoder is trained using the Adam optimizer with a learning rate of 1e-3 for a fixed number of epochs, ensuring that the latent space is grounded in meaningful structure before decoder training begins.

- describe NLL loss calculation

The Negative Log Likelihood (NLL) loss in CUVA is calculated as the negative log probability of the true cluster assignment under the predicted posterior distribution over clusters. For each mention, the model outputs a probability vector over K clusters, and the NLL is computed as the negative log of the probability assigned to the true cluster label. This loss is minimized during the first training phase to align the encoder’s predictions with the hard cluster assignments from hierarchical clustering.

- explain Decoder training

Decoder training in CUVA occurs in the second training phase, after the encoder has been frozen. The decoder reconstructs the input embedding from the latent variable and the cluster centroid, and is optimized using the evidence lower bound, which includes reconstruction loss, KL divergence, knowledge graph embedding loss, and side information loss. The decoder is trained using the Adam optimizer with a reduced learning rate of 1e-4 to ensure fine-grained adjustments, promoting stable convergence and accurate reconstruction.

- describe ELBO loss calculation

The Evidence Lower Bound (ELBO) loss in CUVA is calculated as the sum of the reconstruction loss, the KL divergence between the variational posterior and the prior, the knowledge graph embedding loss, and the side information loss. The reconstruction loss measures the discrepancy between the input embedding and its reconstruction. The KL divergence regularizes the posterior toward a unit Gaussian prior. The knowledge graph embedding loss enforces structural consistency, and the side information loss incorporates external equivalence constraints. The ELBO is maximized during decoder training to ensure that the latent space is both informative and well-regularized.

- explain combined loss function

The combined loss function in CUVA integrates five components: reconstruction loss, KL divergence, knowledge graph embedding loss, side information loss, and L1 regularization. Each component is weighted by a hyperparameter to balance its contribution. The total loss is minimized during training to optimize the entire system end-to-end. The combined loss ensures that the learned representations are accurate, structured, consistent with external knowledge, and robust to noise, enabling high-quality canonicalization.

### 5. Evaluation

- introduce CANONICALIZATION task

The CANONICALIZATION task involves grouping noun phrases and relation phrases from an open knowledge graph into clusters such that all mentions referring to the same underlying entity or relation are assigned to the same cluster, regardless of surface form. The goal is to produce a non-redundant, semantically coherent representation of knowledge where each cluster corresponds to a latent entity or relation, and the label of the cluster is unknown during training. This task is inherently unsupervised and requires modeling both semantic similarity and structural context to resolve ambiguity and polysemy.

- describe ReVerb45K dataset

The ReVerb45K dataset is a benchmark for entity canonicalization derived from the ReVerb open knowledge graph, containing 45,000 triples with annotated ground-truth clusters for head entities. It is widely used to evaluate canonicalization systems and includes a standard train/validation/test split. The dataset is characterized by high lexical variation and moderate ambiguity, making it suitable for testing the ability of models to resolve surface form differences while preserving semantic equivalence.

- introduce CANONICNELL dataset

The CANONICNELL dataset is a novel benchmark for entity canonicalization constructed from the 165th iteration snapshot of the NELL knowledge base. It was created by identifying co-referent entity pairs using a soft-truth scoring mechanism and applying depth-first search to extract connected components as gold-standard clusters. The dataset contains triples from NELL165 filtered to include only those with head or tail entities belonging to these clusters, ensuring that canonicalization is evaluated on semantically grounded, real-world knowledge. Unlike existing benchmarks derived from ReVerb, CANONICNELL is independent of external extraction systems, providing a more diverse and challenging testbed for canonicalization algorithms.

- describe dataset statistics

The ReVerb45K dataset contains 45,000 triples with 12,000 unique head entities and 8,500 unique relation phrases. The CANONICNELL dataset contains 18,700 triples with 6,200 unique entities and 1,500 unique relations. Both datasets include a validation and test split, with CANONICNELL using an 80:20 split. The datasets exhibit high lexical variation, with many entities having multiple surface forms, and moderate ambiguity, with some mentions having overlapping semantic roles.

- explain data division for test and validation sets

The data division for test and validation sets in CUVA follows established protocols for the ReVerb45K, Base, and Ambiguous datasets, using predefined splits provided by prior work. For CANONICNELL, a random 80:20 split was applied to the filtered triples to ensure independence between training and evaluation. The validation set is used for hyperparameter tuning, and the test set is used for final performance evaluation, ensuring that results are reproducible and not influenced by data leakage.

- describe hyper-parameter tuning

Hyperparameter tuning in CUVA is performed using grid search over the validation set. Key hyperparameters include the threshold for hierarchical agglomerative clustering, the latent dimensionality of the VAE, the learning rate, the regularization weight, and the temperature parameter for soft argmax. The search space for clustering thresholds is refined in two stages, first using coarse steps and then fine-grained increments. The best-performing configuration is selected based on macro F1 score and applied to the test set.

- introduce evaluation metrics

The evaluation metrics used in CUVA are macro F1, micro F1, and pair F1 scores. Macro F1 computes the average F1 score across all clusters, giving equal weight to small and large clusters. Micro F1 computes the overall precision and recall across all mention pairs, favoring larger clusters. Pair F1 measures the proportion of correctly predicted equivalent pairs among all possible pairs, providing a direct assessment of clustering accuracy. These metrics are standard in canonicalization evaluation and allow for comprehensive comparison with prior work.

- describe CUVA approach

The CUVA approach to canonicalization involves training a neural architecture that jointly learns embeddings and cluster assignments using variational autoencoders, knowledge graph embedding, and side information constraints. The model is trained in two steps: first, the encoder is optimized using hard cluster labels from hierarchical clustering; second, the decoder and centroids are optimized using the evidence lower bound. The system outputs soft cluster assignments that group semantically equivalent mentions, enabling accurate, non-redundant knowledge representation.

- describe CUVA with Side Information

CUVA with Side Information incorporates external equivalence constraints derived from entity linking, paraphrase databases, and morphological normalization. These constraints are encoded as a differentiable loss that penalizes deviations between the embeddings of known equivalent mentions. This enhancement significantly improves canonicalization accuracy, particularly for ambiguous or low-frequency mentions, by guiding the model toward semantically coherent groupings.

- compare CUVA with existing methods

CUVA outperforms existing methods, including GloVe+HAC, HolE+HAC, and CESI, across all benchmarks. On ReVerb45K, CUVA achieves a 4.1% improvement in macro F1 over CESI, and on CANONICNELL, it achieves state-of-the-art results with a 5.7% gain over the previous best. CUVA’s end-to-end training, probabilistic clustering, and integration of structural knowledge enable it to resolve ambiguities and generalize better than pipeline-based or rule-based approaches.

- analyze results on ReVerb45K

On ReVerb45K, CUVA achieves a macro F1 of 0.783, micro F1 of 0.801, and pair F1 of 0.792, surpassing CESI by 4.1%, 4.5%, and 4.3% respectively. The model correctly groups mentions such as “NBC-TV” and “NBC Television,” and resolves ambiguous cases like “Bill” by leveraging relational context. The inclusion of side information and knowledge graph embedding significantly improves precision, particularly for low-frequency entities.

- analyze results on CANONICNELL

On CANONICNELL, CUVA achieves a macro F1 of 0.756, outperforming all baselines, including FastText+HAC and CESI. The dataset’s independence from ReVerb and its reliance on NELL’s internal co-reference annotations make it a challenging and realistic testbed. CUVA’s ability to generalize without external resources demonstrates its robustness and adaptability to new domains.

- compare CUVA with pretrained language models

CUVA outperforms pretrained language models such as BERT, RoBERTa, and ERNIE when used with hierarchical clustering. On ReVerb45K, CUVA achieves a 11.8% improvement over RoBERTa+HAC, demonstrating that its specialized architecture for canonicalization is more effective than general-purpose language models. The joint learning of embeddings and clusters enables CUVA to capture domain-specific semantic structure that language models fail to disentangle.

- describe qualitative analysis

Qualitative analysis of CUVA’s output reveals that it correctly groups semantically equivalent mentions such as “kodagu” and “coorg,” and identifies relation equivalences like “be associate with” and “have be affiliate to.” However, it occasionally clusters mentions with identical surface forms but different meanings, such as “Bill Cosby” and “Bill Maher,” highlighting the need for type information in future work.

- illustrate output of CUVA

The output of CUVA on ReVerb45K includes clusters such as {“New York City”, “NYC”, “The Big Apple”}, {“NBC-TV”, “NBC Television”}, and {“be associate with”, “have be affiliate to”, “be now associate with”}. These clusters demonstrate the model’s ability to resolve lexical variation and group semantically equivalent mentions without supervision.

- analyze NP clusters

Analysis of noun phrase clusters shows that CUVA effectively groups mentions with different surface forms but identical referents, such as “Toyota” and “Toyota Motor Corporation.” The model handles multi-token phrases, abbreviations, and acronyms with high accuracy, demonstrating its robustness to linguistic variation.

- analyze RP clusters

Analysis of relation phrase clusters reveals that CUVA identifies semantically equivalent relations such as “is located in” and “has headquarters in,” even when they differ in syntax or word order. The model’s ability to leverage relational context enables it to distinguish between similar relations with different meanings, such as “is born in” and “has died in,” which are correctly kept separate.

- describe ablation tests

Ablation tests demonstrate that removing the knowledge graph embedding module reduces pair F1 by 8.4%, confirming its critical role in disambiguation. Removing side information reduces macro F1 by 3.2%, showing its importance for handling ambiguous cases. The two-step training strategy improves performance by 2.1% compared to joint training, validating its necessity for stable convergence.

- analyze importance of KGE Module

The KGE module is essential for CUVA’s performance, as it provides structural context that distinguishes between semantically similar but distinct relations. Without the KGE module, the model relies solely on surface form and side information, leading to a significant drop in pairwise precision. The module enables the model to resolve ambiguities by leveraging the relational topology of the knowledge graph.

- analyze importance of hidden layer

The hidden layers in the encoder and decoder are critical for capturing non-linear relationships between mentions and their canonical representations. Removing hidden layers reduces performance by 4.8%, indicating that the model requires sufficient capacity to learn complex mappings from surface forms to latent clusters.

- conclude evaluation

The evaluation demonstrates that CUVA achieves state-of-the-art performance on four canonicalization benchmarks, outperforming existing methods by significant margins. Its end-to-end architecture, probabilistic clustering, and integration of structural and side information make it uniquely effective at resolving ambiguity and producing coherent, non-redundant knowledge representations.

### 6. CONCLUSION AND FUTURE WORK

- summarize CUVA approach

CUVA is a novel neural architecture for canonicalizing noun phrases and relation phrases in open knowledge graphs by jointly learning embeddings and cluster assignments using variational autoencoders, knowledge graph embedding, and side information constraints. It operates in two training phases to ensure stable convergence and produces soft cluster assignments that account for ambiguity and polysemy. CUVA outperforms all existing methods on multiple benchmarks and introduces a new dataset, CANONICNELL, for future research.

- summarize contributions

The contributions of this invention include: (1) the introduction of CUVA, the first end-to-end canonicalization framework using variational autoencoders; (2) the integration of knowledge graph embedding and side information as differentiable constraints; (3) a two-step training strategy that prevents information leakage; (4) the creation of CANONICNELL, a novel benchmark for entity canonicalization; and (5) empirical validation of state-of-the-art performance across four datasets.

- describe future work directions

Future work includes extending CUVA to canonicalize relation types with greater granularity, incorporating temporal information to model evolving entities, and integrating type constraints to disambiguate homonyms. The architecture can also be adapted for multilingual canonicalization and applied to other structured data formats such as RDF or OWL.

- introduce Hypernymy Detection

Hypernymy detection can be integrated into CUVA to refine cluster semantics by identifying hierarchical relationships between canonical entities. For example, “Apple Inc.” could be linked as a hypernym of “iPhone,” enabling richer knowledge representation and improved reasoning.

- introduce Link Prediction

Link prediction can be enhanced by CUVA’s canonicalized representations, as accurate entity and relation clustering improves the plausibility scoring of missing triples. Future work will explore using CUVA as a preprocessing step for knowledge graph completion systems.

- describe computing environment

The CUVA system is implemented in PyTorch and trained on a single NVIDIA V100 GPU with 16GB of memory. The software environment includes Python 3.8, CUDA 11.1, and the HuggingFace Transformers library for preprocessing. The system is designed for scalability and can be deployed on distributed computing clusters for large-scale knowledge graph processing.

- describe computer system components

The computer system for executing CUVA includes a central processing unit, memory, persistent storage, input/output interfaces, and a communications unit. The system is configured to receive input triples and side information, load trained models, and output canonicalized clusters. The architecture supports both batch and streaming modes of operation.

- describe communications fabric

The communications fabric in the CUVA system enables data transfer between the central processing unit, memory, and external storage. It supports high-bandwidth communication for efficient loading of large knowledge graphs and side information sources, ensuring low-latency training and inference.

- describe cache

The cache in the CUVA system stores frequently accessed embeddings and cluster centroids to reduce memory access latency during training and inference. It is implemented as a hierarchical buffer that prioritizes recently used or high-activity components, improving system throughput.

- describe memory

Memory in the CUVA system includes both volatile RAM for active training and non-volatile storage for model checkpoints and intermediate results. The system is optimized to handle large-scale embeddings, with memory management strategies that minimize overhead during batch processing.

- describe persistent storage

Persistent storage in the CUVA system holds the trained model parameters, input knowledge graphs, side information files, and evaluation results. It supports both local and cloud-based storage, enabling deployment in distributed environments and long-term archival of canonicalized knowledge.

- describe communications unit

The communications unit in the CUVA system enables connectivity to external data sources, such as entity linking APIs, paraphrase databases, and knowledge graph repositories. It supports HTTP, REST, and file transfer protocols for seamless integration with external systems.

- describe I/O interface

The I/O interface in the CUVA system accepts input triples in JSON or CSV format and outputs canonicalized clusters in structured formats such as RDF or TSV. It supports batch processing and streaming input, enabling integration with real-time knowledge graph pipelines.

- describe external devices

External devices connected to the CUVA system include keyboards, mice, and displays for manual configuration and monitoring. The system can also interface with sensors, databases, and web services for automated data ingestion and result dissemination.

- describe display

The display in the CUVA system provides visualizations of cluster assignments, embedding spaces, and training metrics. It enables human-in-the-loop validation and debugging, allowing users to inspect and refine canonicalization results interactively.

- describe computer readable storage medium

The computer readable storage medium in the CUVA system includes hard drives, solid-state drives, and cloud storage volumes that store the executable program instructions, model weights, and training data. The medium is non-transitory and supports read/write operations for persistent storage and model updates.

- describe computer readable program instructions

The computer readable program instructions for CUVA are implemented in Python and PyTorch, comprising modules for data preprocessing, encoder-decoder architecture, knowledge graph embedding, and loss computation. The instructions are compiled and executed on a CPU or GPU, enabling efficient training and inference.

- describe network

The network in the CUVA system connects the computing system to external knowledge sources, such as Wikipedia, Wikidata, and public knowledge graphs. It enables real-time access to side information and facilitates distributed training across multiple nodes.

- describe network adapter card

The network adapter card in the CUVA system enables high-speed data transfer between the system and external networks. It supports gigabit Ethernet and TCP/IP protocols, ensuring reliable connectivity for data ingestion and model deployment.

- describe downloading program instructions

Program instructions for CUVA can be downloaded from a remote server or repository over a secure connection. The system validates the integrity of the downloaded code using cryptographic hashing before execution, ensuring security and authenticity.

- describe executing program instructions

Executing program instructions in CUVA involves loading the trained model into memory, parsing input triples, encoding mentions, computing cluster assignments, and outputting canonicalized clusters. The system runs in a single-threaded or multi-threaded mode depending on hardware configuration, with optimizations for GPU acceleration.

- describe computer readable program instructions

Computer readable program instructions for CUVA are stored in a non-transitory medium and include all modules necessary for training and inference. The instructions are written in a high-level language and compiled into executable code for deployment on standard computing hardware.

- describe flowchart illustrations

Flowchart illustrations of CUVA depict the sequence of operations from input triple ingestion to canonicalized cluster output, including preprocessing, encoding, clustering, embedding, and loss computation. The flowcharts provide a visual guide for implementation and system integration.

- describe block diagrams

Block diagrams of CUVA illustrate the modular architecture, showing the flow of data between the encoder, decoder, knowledge graph embedding module, and side information module. These diagrams clarify the relationships between components and support system design and documentation.

- conclude description of embodiments

The embodiments described herein represent the best mode of practicing the invention and are not intended to limit the scope of the claims. Modifications and extensions to the architecture, training strategy, and application domains are within the scope of the invention as defined by the claims.