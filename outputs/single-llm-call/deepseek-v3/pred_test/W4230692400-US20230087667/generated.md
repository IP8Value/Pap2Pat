Here is the patent application drafted according to the provided outline and research paper content:

# DESCRIPTION  

## STATEMENT REGARDING PRIOR DISCLOSURES BY THE INVENTOR OR A JOINT INVENTOR  

The present invention relates to improvements in knowledge graph canonicalization through machine learning techniques. Prior art includes traditional approaches to canonicalizing noun phrases and relation phrases in open knowledge graphs, such as rule-based systems like REVERB and learning-based methods like OLLIE. Entity linking techniques that map phrases to existing knowledge bases represent additional prior art. The CESI architecture, which uses a two-step pipeline combining HolE embeddings with hierarchical agglomerative clustering, constitutes the closest prior art. These existing approaches suffer from limitations in handling polysemy, inability to jointly learn embeddings and cluster assignments, and suboptimal utilization of structural knowledge within knowledge graphs.  

## BACKGROUND  

Machine learning refers to computational systems that improve their performance on tasks through experience without being explicitly programmed. In the context of knowledge processing, machine learning models utilize hyperparameters - configuration variables that govern the learning process and model architecture. Neural networks represent a particularly powerful class of machine learning models inspired by biological neural systems. These networks consist of interconnected processing units organized in layers that transform input data through successive nonlinear transformations.  

The architecture of a neural network determines its information processing capabilities through the arrangement of layers and connections between units. Neural network learning occurs through optimization algorithms that adjust connection weights to minimize error on training tasks. Backpropagation represents the standard learning algorithm that computes gradients of the error with respect to network parameters. Neural networks find wide application in knowledge processing tasks including information extraction, knowledge base completion, and semantic similarity measurement.  

Canonicalization using variational autoencoders constitutes a novel approach to organizing knowledge graphs. Variational autoencoders are generative neural network models that learn compact latent representations of input data while preserving essential features. These models combine the representational power of deep neural networks with probabilistic graphical models to enable efficient inference and generation. When applied to knowledge graph canonicalization, variational autoencoders can simultaneously learn embeddings and cluster assignments while incorporating structural knowledge from the graph.  

## SUMMARY  

The present invention provides a computer-implemented method for canonicalizing open knowledge graphs using variational autoencoders. The method receives a knowledge graph comprising triples of noun phrases and relation phrases extracted from text corpora. A neural network architecture processes these triples through entity and relation variational autoencoders that model latent clusters as Gaussian distributions. The architecture incorporates a knowledge graph embedding module to utilize structural information and side information constraints to improve canonicalization. Training occurs through a two-step procedure that first optimizes encoder components before refining decoder elements. The system outputs clustered noun phrases and relation phrases where surface forms referring to the same underlying entities or relations are grouped together.  

## DETAILED DESCRIPTION  

Knowledge graphs represent structured networks of entities and their relationships, typically stored as subject-predicate-object triples. Existing knowledge graphs suffer from limitations in adaptability and canonicalization, particularly when constructed through open information extraction methods. Current solutions employ pipeline approaches that first learn embeddings then perform clustering, or utilize external knowledge bases for entity linking. These approaches exhibit deficiencies including inability to handle novel entities, suboptimal use of structural knowledge, and separation of representation learning from clustering.  

The present invention introduces Canonicalizing Using Variational Autoencoders (CUVA), a neural architecture that jointly learns mention representations and cluster assignments. CUVA combines variational deep embedding for clustering with knowledge graph embedding to utilize structural information. The architecture comprises entity and relation variational autoencoders (E-VAE and R-VAE) that model latent entities and relations as Gaussian mixtures. Each autoencoder includes encoder and decoder sections that transform between input space and latent representations.  

Hierarchical agglomerative clustering initializes the Gaussian mixture parameters before neural network training. The training procedure employs two steps - first optimizing encoder components using negative log likelihood loss, then refining decoders through evidence lower bound optimization. A constraint-based loss incorporates side information about equivalent mentions to guide the learning process. The knowledge graph embedding module encodes structural knowledge by modeling relationships between latent cluster assignments.  

Operational steps include receiving knowledge graph triples and contextual information, dynamically clustering mentions, initializing cluster parameters, and training the neural network through alternating optimization of encoder and decoder components. The system calculates total loss incorporating reconstruction error, cluster assignment likelihood, knowledge graph embedding score, and side information constraints. Benefits include improved canonicalization accuracy, joint learning of embeddings and clusters, and effective utilization of both structural and contextual information.  

### Example Table 1  

The example table illustrates clustering of entities, noun phrases, and relation phrases through the CUVA system. The variational autoencoder for entities processes noun phrases by encoding them into latent Gaussian distributions then reconstructing the inputs. Similarly, the relation variational autoencoder handles relation phrases through the same probabilistic framework. The knowledge base completion module predicts missing relationships between clustered entities.  

Training the resulting neural network architecture involves building hierarchical agglomerative cluster models to initialize Gaussian parameters. The encoder section transforms input phrases into latent variables while the decoder reconstructs inputs from these representations. Total loss value calculation combines reconstruction error, cluster assignment likelihood, knowledge graph embedding score, and side information constraints.  

FIG. 2A depicts the core structure of CUVA showing interconnected encoder and decoder blocks. The encoder block processes input phrases through successive neural layers to produce latent variables. The decoder block reverses this process to reconstruct inputs from latent representations. FIG. 2B illustrates encoding of side information through constraint losses that penalize deviations between known equivalent mentions. FIG. 2C shows the knowledge graph embedding module that models relationships between latent cluster assignments.  

FIG. 3 presents the complete CUVA architecture integrating entity and relation variational autoencoders with the knowledge graph embedding module. FIG. 4 demonstrates dynamic clustering of received information through the hierarchical agglomerative process. FIG. 5 details initialization of cluster means and variances based on the clustering output. FIG. 6 depicts the two-step training procedure that alternately optimizes encoder and decoder components.  

The system operates by receiving information from knowledge graph triples and contextual documents. It dynamically clusters received information through hierarchical agglomerative methods before initializing neural network parameters. Training proceeds in two steps using both reconstruction and constraint-based loss functions. The first step trains encoder components while fixing decoders, then reverses this for the second optimization phase. Additional constraint-based loss encodes structural knowledge from the graph topology and side information about equivalent mentions.  

### Further Comments and/or Embodiments  

The CUVA system provides significant benefits over current approaches to knowledge graph canonicalization. Deficiencies in the state of art include pipeline architectures that separate representation learning from clustering, inability to handle novel entities absent from reference knowledge bases, and suboptimal use of structural information. The CANONICNELL dataset provides a new benchmark for evaluating canonicalization systems, constructed from NELL knowledge base iterations with automatically identified coreferent entities.  

### 1. INTRODUCTION  

Knowledge graphs organize world knowledge as networks of interconnected entities and relationships. Existing knowledge graphs face limitations in adaptability and canonicalization, particularly when constructed through open information extraction methods. Open information extraction techniques extract triples from text without predefined schemas but produce uncanonicalized outputs where varied surface forms refer to identical entities or relations.  

Current solutions employ rule-based patterns or learning-based systems to extract relations and arguments. These approaches generate open knowledge graphs with redundant facts due to lack of canonicalization, increasing memory requirements and impairing query performance. The present invention introduces canonicalizing using variational autoencoders (CUVA) to address these limitations through joint learning of mention representations and cluster assignments. Key contributions include a novel neural architecture combining variational deep embedding with knowledge graph modeling, and demonstration of state-of-the-art performance on entity canonicalization benchmarks.  

### 2. RELATED WORK  

Open information extraction techniques originated with systems like TextRunner and evolved through rule-based approaches such as REVERB and learning-based methods like RNNOIE. Entity linking represents a traditional canonicalization approach that maps phrases to existing knowledge base entries, but fails for novel entities. The RESOLVER system used string similarity features while later work incorporated manually defined features and pruning techniques.  

The CESI architecture currently represents state of the art through a two-step pipeline combining HolE embeddings with hierarchical agglomerative clustering. Unlike these approaches, CUVA learns embeddings and cluster assignments end-to-end within a single model while incorporating structural knowledge through a dedicated graph embedding module.  

### 3. OPEN KGS CANONICALIZATION USING VAE  

The canonicalization task involves clustering noun phrases and relation phrases from open knowledge graphs so that mentions referring to the same underlying entities or relations group together. CUVA implements this through entity and relation variational autoencoders (E-VAE and R-VAE) that model latent clusters as Gaussian distributions. The generative process assumes observed phrases are sampled from these latent distributions after nonlinear transformations.  

The knowledge graph embedding module encodes structural information by modeling relationships between latent cluster assignments. Variational autoencoder components include encoder blocks that approximate posterior distributions over latent variables and decoder blocks that reconstruct inputs from these representations. Cluster assignment occurs through a differentiable approximation of argmax that selects the most probable Gaussian component for each mention.  

Side information from entity linking, paraphrase databases, and morphological normalization provides additional constraints through a weighted mean squared error loss. This supplementary information helps correct errors introduced during initial clustering while maintaining differentiability throughout the architecture.  

### 4. Training Strategy  

Training begins by initializing the mixture of Gaussians through hierarchical agglomerative clustering of pretrained GloVe vectors. The two-step training strategy first optimizes encoder components using negative log likelihood loss while fixing decoders, then reverses this configuration to refine decoder elements through evidence lower bound optimization.  

Encoder training incorporates weak supervision from initial cluster assignments along with side information constraints. Decoder training focuses on reconstruction quality while maintaining the knowledge graph embedding objective. The combined loss function balances these components through weighting hyperparameters that control their relative influence during optimization.  

### 5. Evaluation  

Evaluation of the canonicalization task employs multiple benchmarks including the ReVerb45K dataset and newly introduced CANONICNELL dataset constructed from NELL knowledge base iterations. Dataset statistics demonstrate coverage of diverse entities and relations with standard splits for validation and testing. Hyperparameter tuning occurs through grid search over latent dimensions, learning rates, and regularization strengths.  

Metrics include macro, micro and pairwise F1 scores that measure clustering quality at different granularities. Results show CUVA outperforming existing methods across all benchmarks, with particular gains from incorporating side information. Comparative analysis demonstrates advantages over pretrained language models like BERT and ablation studies confirm the importance of the knowledge graph embedding module.  

Qualitative analysis illustrates correct clustering of diverse surface forms like "kodagu" and "coorg" while revealing challenges with ambiguous names and antonym relations. Further examination through ablation tests quantifies the contribution of individual components, showing significant performance drops when removing the knowledge graph embedding module or switching to pipeline training.  

### 6. CONCLUSION AND FUTURE WORK  

The CUVA system advances knowledge graph canonicalization through joint learning of mention representations and cluster assignments within a variational autoencoder framework. Key innovations include Gaussian mixture modeling of latent entities and relations, incorporation of structural knowledge through graph embeddings, and constraint-based utilization of side information.  

Future work directions include extending the approach to hypernymy detection and link prediction tasks. The system operates within standard computing environments comprising processors, memory, storage, and input/output interfaces. Implementation occurs through computer-readable program instructions that perform the described methods when executed on appropriate hardware. Flowcharts and block diagrams in the figures illustrate exemplary embodiments of the inventive concepts.  

The complete system enables improved knowledge graph organization through accurate canonicalization of diverse surface forms to their underlying entities and relations. By combining neural representation learning with probabilistic clustering and structural knowledge utilization, the invention provides significant advances over existing approaches to knowledge graph construction and maintenance.