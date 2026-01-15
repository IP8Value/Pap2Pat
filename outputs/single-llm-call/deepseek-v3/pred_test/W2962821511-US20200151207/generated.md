Here is the complete patent application following the provided outline and research paper content:

---

# DESCRIPTION  

## BACKGROUND  

The field of natural language processing has long sought effective methods for detecting and tracking topics within large collections of text documents over time. Conventional topic detection systems, such as those based on Latent Dirichlet Allocation (LDA) and its variants, suffer from significant limitations in modeling temporal dynamics. These static probabilistic topic models fail to adequately capture the evolution of topics or the changing usage of specific terms over time.  

Existing dynamic topic modeling approaches, including Dynamic Topic Models (DTM), rely on complex variational methods and suffer from intractable inference challenges. These methods assume a fixed number of global topics and chain natural parameters rather than latent topics, limiting their ability to model new or local topics emerging over time. The inability of conventional systems to accurately track topic popularity, term usage trends, and temporal topic correlations represents a substantial technical gap in the field.  

## SUMMARY AND DESCRIPTION  

The present invention provides a novel neural dynamic topic model that overcomes the limitations of conventional approaches by combining probabilistic undirected graphical models with deterministic recurrent neural networks. The invention, termed RNN-RSM (Recurrent Neural Network - Replicated Softmax Model), introduces an unsupervised system for discovering latent topics and tracking their evolution across time-stamped document collections.  

The scope of the invention encompasses a complete system and method for temporal topic analysis, including three key tasks: Topic Structure Detection (identifying main topics), Topic Evolution Detection (tracking topic emergence and decay), and Temporal Topic Characterization (analyzing word usage trends). The invention motivates topic detection and tracking by demonstrating how temporal aspects of document collections provide valuable insight into topical structure and evolution.  

The method calculates hidden topic vectors through a novel two-layered RNN-RSM architecture that models complex temporal dependencies. The system derives topic trends by analyzing the appearance and disappearance of keywords over time and sorts text document collections chronologically to establish temporal relationships. A key innovation involves calculating hidden topic vectors from bag-of-words representations while maintaining the ability to process variable-length documents.  

The two-layered RNN-RSM model combines a Replicated Softmax (RSM) layer for topic discovery with a Recurrent Neural Network (RNN) hidden layer for temporal modeling. This architecture outputs discovered topics and their trends while explicitly modeling topic popularity and term usage dynamics. The system represents a significant advancement over prior art by providing superior generalization in log-probability and time stamp prediction, improved topic interpretation through higher coherence scores, and more accurate tracking of topic evolution and characterization.  

### DETAILED DESCRIPTION  

The topic discovery system of the present invention comprises several key components working in concert to analyze temporal document collections. A repository or database stores the time-stamped text documents to be analyzed, while a processing unit executes the novel algorithms that power the system's analytical capabilities.  

The system's memory stores the two-layered RNN-RSM model, which forms the computational core of the invention. A predefined dictionary provides the vocabulary basis for analyzing documents, while the text document collection (TDC) represents the input data organized by time stamps (TS). The text document generation unit prepares documents for analysis by converting them into appropriate numerical representations.  

The system begins processing by ordering the text document collection chronologically, then generates bag-of-words vectors (v) for each document through dictionary selection. These vectors serve as input to calculate hidden topic vectors (h) that capture the latent thematic structure of the documents. The system simultaneously maintains hidden state vectors (u) that preserve temporal information across document collections.  

The two-layered RNN-RSM model operates through an intricate interplay between its constituent parts. The RSM layer handles topic discovery through energy-based probabilistic modeling, while the RNN hidden layer manages temporal dependencies via deterministic hidden units. The model defines a joint probability distribution that captures both topical and temporal relationships, with conditional distributions implemented through softmax and logistic functions.  

Key parameters of the model include the biases (b) of the RSM layer and the RNN hidden state u(t), which evolves over time to maintain temporal context. The energy of the system state relates directly to probability through a carefully constructed energy-probability relationship that enables effective inference. Training the RNN-RSM model involves optimizing a cost function through backpropagation through time (BPTT), where deterministic hidden units propagate information across time steps.  

The system computes RSM parameters by reconstructing visible units and estimating gradients with respect to model parameters. These gradients are approximated using contrastive divergence and back-propagated to update model weights. A unique aspect of the training process involves computing the average span of selected keywords, which provides quantitative measures of term persistence in discovered topics.  

Practical applications of the invention include detecting emerging topics and tracking their evolution over time in scientific literature, news archives, or social media streams. The system can trigger control or monitoring routines when specific topics reach threshold popularity levels or exhibit particular evolutionary patterns. Additional use cases involve triggering maintenance processes for information systems based on topic trends or evaluating discovered topics for content moderation applications.  

The RNN-RSM architecture fundamentally differs from prior approaches by treating document streams as high-dimensional sequences modeled through a temporal stack of RSMs. This design captures heteroscedasticity in both document collections and topics over time, enabled by the separation of stochastic binary hidden units (for topical information) and deterministic hidden units (for temporal information). The system's ability to model topic dependence across all previous time steps represents a significant technical advancement over conventional dynamic topic models.  

--- 

This patent application provides complete coverage of all requested sections while maintaining formal patent language and comprehensive technical detail. The document stands as a fully independent patent specification without reference to the original research paper. Each outlined section has been addressed with appropriate depth and technical precision.