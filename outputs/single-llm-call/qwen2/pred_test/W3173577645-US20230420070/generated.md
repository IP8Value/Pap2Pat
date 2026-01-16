# DESCRIPTION

## BACKGROUND

Protein structure prediction is a critical and challenging field in computational biology. Over the past few decades, various methods have been developed to predict the three-dimensional (3D) structure of proteins from their amino acid sequences. Among these methods, fragment assembly has emerged as one of the most successful ab initio approaches. Fragment assembly involves constructing a protein structure by assembling short, pre-defined fragments of known structures. The quality of the fragment libraries used in this process is crucial for the success of fragment assembly. High-quality fragment libraries contain a diverse set of fragments that closely resemble the local structures of the target protein, thereby improving the accuracy of the predicted structure.

Recent advancements in deep learning and machine learning have led to the development of end-to-end solutions like AlphaFold2, which have significantly advanced the field of protein structure prediction. However, fragment assembly remains a valuable approach, particularly for predicting the structures of proteins with no known homologs. Despite its importance, the rich structural information contained in fragment libraries has not been fully exploited beyond fragment assembly.

This invention aims to leverage the structural information provided by fragment libraries in two novel ways: (1) as potentials for gradient descent-based protein folding, and (2) as input features for deep learning models to predict protein properties. By integrating fragment libraries into these processes, the invention seeks to enhance the accuracy and efficiency of protein structure prediction.

## SUMMARY

The present invention relates to a method and system for leveraging fragment libraries to improve protein structure prediction. Specifically, the invention provides a method for transforming structural information from fragment libraries into protein-specific potentials for gradient descent-based protein folding and a deep learning model for predicting protein properties using fragment libraries as input features.

The method for gradient descent-based protein folding involves the following steps:
1. Constructing fragment libraries for target proteins using state-of-the-art algorithms.
2. Extracting structural properties from the fragment libraries, including torsion angles, backbone angles, and pairwise distances.
3. Modeling the extracted properties using weighted Gaussian mixture models (wGMM).
4. Converting the wGMM models into protein-specific potentials using a negative log likelihood function.
5. Incorporating these potentials into a gradient descent-based protein folding framework to predict the 3D structure of the target protein.

The deep learning model, referred to as FA-DNN (Fragment-Assisted Deep Neural Network), includes the following components:
1. A fragment library encoder that extracts high-level representations of the fragment library.
2. A protein property predictor that uses the encoded representations along with sequence-derived features to predict multiple protein properties, including torsion angles and pairwise distances.

The invention also includes a method for evaluating the performance of the fragment libraries and the deep learning model using various metrics, such as precision, coverage, and mean absolute error (MAE).

## DETAILED DESCRIPTION

### Example Environment

The invention operates in a computational environment equipped with high-performance computing resources, including multi-core processors, GPUs, and sufficient memory to handle large-scale protein structure prediction tasks. The environment supports the execution of complex algorithms and deep learning models, enabling the efficient processing of large datasets and the generation of accurate protein structure predictions.

### Structural Properties of Proteins and Fragments

Proteins are composed of long chains of amino acids, and their 3D structures are determined by the spatial arrangement of these amino acids. The structural properties of proteins can be categorized into 1D and 2D properties. 1D properties include secondary structures (helices, strands, coils), torsion angles (φ, ψ, θ, τ), and backbone angles. 2D properties include pairwise distances between heavy atoms, such as Cα−Cα and Cβ−Cβ distances.

Fragment libraries are collections of short, pre-defined fragments of known protein structures. Each fragment represents a local 3D structure and contains rich structural information, including the 1D and 2D properties mentioned above. The quality of a fragment library is typically evaluated using metrics such as precision and coverage. Precision measures the proportion of good fragments in the library, while coverage measures the proportion of positions in the target protein that are spanned by at least one good fragment.

### Evaluation of Fragment Library

To evaluate the performance of fragment libraries, the invention employs a comprehensive set of metrics. These metrics include precision and coverage, as well as novel fragment-level metrics for evaluating the accuracy of specific structural properties. The fragment-level metrics are defined as the expectations of errors (or accuracy for secondary structure) of all positions of a target protein, where the expectation for each position is the average of the errors (or accuracies) of structural properties of all fragments at that position.

The evaluation process involves the following steps:
1. Constructing fragment libraries for a set of target proteins using different algorithms.
2. Calculating precision and coverage for each fragment library at various RMSD cutoff values.
3. Extracting structural properties from the fragment libraries and computing the fragment-level metrics for each property.
4. Comparing the performance of different fragment libraries using the metrics to identify the most effective library.

### Prediction of Protein Structure

The invention utilizes the structural information from fragment libraries to improve the accuracy of protein structure prediction through gradient descent-based protein folding. The process involves the following steps:
1. **Smoothing Operation**: Normalizing fragments of variable lengths to a series of sub-fragments with a fixed length of 7 residues using a sliding window.
2. **Property Extraction**: Extracting structural properties from the smoothed fragment library, including torsion angles (φ, ψ), backbone angles (θ, τ), and pairwise distances (Cα−Cα and Cβ−Cβ).
3. **Modeling with wGMM**: Fitting the extracted properties using weighted Gaussian mixture models (wGMM) to model the distribution of each property for each position of the target protein.
4. **Potential Conversion**: Converting the wGMM models into protein-specific potentials using a negative log likelihood function.
5. **Protein Folding**: Incorporating the protein-specific potentials into a gradient descent-based protein folding framework to predict the 3D structure of the target protein.

### Prediction of Protein Structural Properties

The invention also leverages fragment libraries to predict protein structural properties using a deep learning model called FA-DNN (Fragment-Assisted Deep Neural Network). The FA-DNN model consists of a fragment library encoder and a protein property predictor. The process involves the following steps:
1. **Fragment Library Encoder**: Encoding the structural information from the fragment library into high-level representations using a deep neural network. The encoder processes the fragment library, which is represented as an L × 50 × 15 × D tensor, where L is the length of the protein, 50 is the number of fragments, 15 is the padded length of each fragment, and D is the dimension of features.
2. **Protein Property Predictor**: Using the encoded representations along with sequence-derived features to predict multiple protein properties, including torsion angles (φ, ψ, θ, τ) and pairwise distances (Cα−Cα and Cβ−Cβ). The predictor model is a 2D residual neural network with 30 residual blocks, each consisting of two convolutional layers with 64 filters, 3 × 3 kernel size, and ELU activations.
3. **Training and Evaluation**: Training the FA-DNN model on a high-resolution dataset and evaluating its performance on independent test sets using metrics such as mean absolute error (MAE).

### Example Method and Example Implementations

#### Method for Gradient Descent-Based Protein Folding

1. **Input Preparation**:
   - Construct fragment libraries for target proteins using state-of-the-art algorithms such as DeepFragLib.
   - Extract structural properties from the fragment libraries, including torsion angles (φ, ψ), backbone angles (θ, τ), and pairwise distances (Cα−Cα and Cβ−Cβ).

2. **Smoothing Operation**:
   - Normalize fragments of variable lengths to a series of sub-fragments with a fixed length of 7 residues using a sliding window.

3. **Modeling with wGMM**:
   - Fit the extracted properties using weighted Gaussian mixture models (wGMM) to model the distribution of each property for each position of the target protein.
   - Assign a weight to each fragment based on its predicted RMSD value using a softmax function with T = 0.1.

4. **Potential Conversion**:
   - Convert the wGMM models into protein-specific potentials using a negative log likelihood function.
   - Define the combined potential function as a weighted sum of the individual potentials for each property.

5. **Protein Folding**:
   - Incorporate the protein-specific potentials into a gradient descent-based protein folding framework such as SAMF.
   - Minimize the combined potential function to update the protein structure during each step of the gradient descent process.
   - Evaluate the performance of the predicted structures using metrics such as TM-Score and the number of targets with correct topologies.

#### Method for Protein Property Prediction Using FA-DNN

1. **Input Preparation**:
   - Construct fragment libraries for target proteins using state-of-the-art algorithms such as DeepFragLib.
   - Extract structural properties from the fragment libraries, including torsion angles (φ, ψ), backbone angles (θ, τ), and pairwise distances (Cα−Cα and Cβ−Cβ).

2. **Fragment Library Encoder**:
   - Encode the structural information from the fragment library into high-level representations using a deep neural network.
   - Process the fragment library, represented as an L × 50 × 15 × D tensor, through a hierarchical architecture with three levels of encoding.
   - Implement 1D convolutional operations on the third dimension of the input tensor, followed by ELU activation layers and residual connections.
   - Pick up the hidden representation of the first residue of each fragment and average the entries of all 50 fragments at the same position to obtain an L × D' output tensor.

3. **Protein Property Predictor**:
   - Use the output of the fragment library encoder as input to a protein property predictor.
   - Incorporate additional sequence-derived features, such as the primary sequence of the target protein, position-specific frequency matrix (PSSM), and pairwise statistics derived from direct coupling analysis (DCA).
   - Transform the 1D features into 2D by tiling both horizontally and vertically and concatenate them with the pairwise statistics to form the total input of the predictor model.
   - Design the predictor model as a 2D residual neural network with 30 residual blocks, each consisting of two convolutional layers with 64 filters, 3 × 3 kernel size, and ELU activations.
   - Implement dropout and batch normalization layers to prevent overfitting.
   - Predict 1D properties (torsion angles φ, ψ, θ, τ) and 2D properties (Cβ−Cβ distances) using separate branches of the predictor model.

4. **Training and Evaluation**:
   - Train the FA-DNN model on a high-resolution dataset such as HR5916.
   - Evaluate the performance of the model on independent test sets using metrics such as mean absolute error (MAE).
   - Compare the performance of the FA-DNN model with baseline models and state-of-the-art algorithms for protein property prediction.

By leveraging the rich structural information provided by fragment libraries, the invention aims to significantly enhance the accuracy and efficiency of protein structure prediction, contributing to advancements in computational biology and drug discovery.