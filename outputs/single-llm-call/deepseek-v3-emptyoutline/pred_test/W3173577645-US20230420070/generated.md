Here is the complete patent application following the provided outline:

---

# DESCRIPTION  

## BACKGROUND  

Protein structure prediction remains one of the most complex challenges in computational biology, with significant implications for drug discovery, biotechnology, and biomedical research. Traditional approaches, such as fragment assembly, have demonstrated success in predicting protein structures by assembling short structural fragments derived from known protein templates. However, despite the rich structural information embedded within fragment libraries—including one-dimensional (1D) properties like secondary structure and torsion angles, as well as two-dimensional (2D) properties such as inter-residue distances—this information has not been systematically leveraged beyond fragment assembly.  

Recent advancements in gradient-descent-based protein folding, exemplified by systems such as AlphaFold and trRosetta, rely heavily on predicted protein properties derived from sequence information alone. While these methods have achieved notable success, their accuracy is inherently limited by the absence of complementary structural constraints from experimentally determined or computationally predicted fragments. Additionally, existing protein property prediction pipelines predominantly utilize sequential features, such as multiple sequence alignments (MSA) and position-specific scoring matrices (PSSM), neglecting the potential benefits of incorporating structural information from fragment libraries.  

There exists a critical need for improved computational methods that integrate fragment-derived structural information into protein property prediction and structure determination pipelines. Such methods would enhance the accuracy of predicted protein properties, refine gradient-descent-based folding algorithms, and ultimately yield higher-quality protein models for downstream applications.  

## SUMMARY  

The present invention provides novel computational methods and systems for leveraging the structural information embedded within protein fragment libraries to improve protein property prediction and structure determination. The disclosed technology encompasses:  

1. **Comprehensive Fragment Library Analysis:** A systematic evaluation of fragment libraries using novel fragment-level metrics to quantify the accuracy of structural properties, including secondary structure, torsion angles, backbone angles, and inter-residue distances.  
2. **Fragment-Assisted Gradient-Descent Folding:** A method for transforming fragment-derived structural properties into protein-specific potentials using weighted Gaussian Mixture Models (wGMM), which are incorporated into gradient-descent-based folding algorithms to enhance structure prediction accuracy.  
3. **Fragment-Assisted Deep Neural Network (FA-DNN):** A deep learning framework that encodes fragment libraries into high-dimensional feature representations and predicts multiple protein properties, including torsion angles, backbone angles, and inter-residue distances, with superior accuracy compared to existing sequence-based predictors.  

Experimental validation on benchmark datasets, including CASP13 FM, CASP13 TBM, CAMEO, and CASP14 FM, demonstrates that the integration of fragment-derived structural information significantly improves both protein property prediction and structure determination. The disclosed methods represent a transformative advancement in computational biology, enabling more accurate and efficient protein modeling for applications in drug design, protein engineering, and biomedical research.  

## DETAILED DESCRIPTION  

### Example Environment  

The disclosed methods are implemented in a computational environment comprising high-performance computing resources, including multi-core processors, graphical processing units (GPUs), and distributed memory systems. Software implementations may utilize programming languages such as Python, C++, or CUDA, with deep learning frameworks such as TensorFlow or PyTorch for neural network training and inference.  

Fragment libraries are constructed using state-of-the-art algorithms such as DeepFragLib, NNMake, or Flib-Coevo, which recruit template fragments from protein structure databases (e.g., PDB) based on sequence and structural similarity to the target protein. The resulting fragment libraries are stored in standardized formats (e.g., NNMake format) for subsequent analysis and processing.  

### Structural Properties of Proteins and Fragments  

Fragment libraries contain rich structural information that can be categorized into:  

1. **1D Structural Properties:**  
   - Secondary structure (helix, strand, coil)  
   - Torsion angles (ϕ, ψ)  
   - Backbone angles (θ, τ)  

2. **2D Structural Properties:**  
   - Inter-residue distances (Cα–Cα, Cβ–Cβ)  

Novel fragment-level metrics are introduced to evaluate these properties, including:  
- **Fragment Secondary Structure Accuracy (ACCSS):** The proportion of fragments correctly matching the native secondary structure.  
- **Angle Error (ERRang):** The mean absolute error of torsion and backbone angles relative to the native structure.  
- **Distance Error (ERRdist):** The mean absolute error of inter-residue distances within fragments.  

These metrics enable quantitative assessment of fragment library quality and guide the selection of optimal fragments for downstream applications.  

### Evaluation of Fragment Library  

Fragment libraries constructed by DeepFragLib, NNMake, and Flib-Coevo are evaluated on benchmark datasets (CASP13 FM, CASP13 TBM, CAMEO) using precision and coverage metrics at varying RMSD thresholds (0.1–2.0 Å). DeepFragLib demonstrates superior performance, achieving ~90% coverage at 2.0 Å cutoff and higher accuracy in structural property predictions.  

Weighted fragment libraries are generated by assigning confidence scores to fragments based on predicted RMSD values, enhancing the accuracy of extracted structural information. Comparative analysis with Rosetta’s AbInitioRelax confirms that DeepFragLib-derived fragments yield higher-quality structural models.  

### Prediction of Protein Structure  

Structural information from fragment libraries is transformed into protein-specific potentials using weighted Gaussian Mixture Models (wGMM). Key steps include:  

1. **Fragment Smoothing:** Normalizing variable-length fragments into fixed-length (7-residue) sub-fragments via sliding window operations.  
2. **wGMM Modeling:** Fitting fragment-derived properties (ϕ, ψ, θ, τ, Cα–Cα, Cβ–Cβ distances) to four-component wGMMs.  
3. **Potential Derivation:** Converting wGMMs into energy potentials via negative log-likelihood functions.  

These potentials are integrated into gradient-descent-based folding systems (e.g., SAMF) alongside distance constraints from trRosetta. Benchmarking on CASP13 FM, CASP13 TBM, and CAMEO demonstrates improved TM-Scores and topology prediction accuracy, with significant reductions in the gap between best and top1 decoys.  

### Prediction of Protein Structural Properties  

The Fragment-Assisted Deep Neural Network (FA-DNN) comprises:  

1. **Fragment Library Encoder:** A hierarchical neural network that processes fragment libraries into high-dimensional embeddings using residual convolutional blocks.  
2. **Protein Property Predictor:** A 2D residual network that integrates fragment embeddings with sequence-derived features (PSSM, DCA) to predict torsion angles, backbone angles, and Cβ–Cβ distances.  

FA-DNN outperforms state-of-the-art predictors (SPOT-1D, Spider3, trRosetta) on independent test sets, achieving lower mean absolute errors (MAE) in property predictions. Notably, FA-DNN exhibits robustness on targets with limited MSAs, where sequence-based methods underperform.  

### Example Method and Example Implementations  

**Example 1: Protein Folding with Fragment-Derived Potentials**  
1. Construct fragment library for target protein using DeepFragLib.  
2. Extract and smooth structural properties (ϕ, ψ, θ, τ, Cα–Cα, Cβ–Cβ distances).  
3. Fit wGMMs to smoothed properties and derive energy potentials.  
4. Integrate potentials into SAMF alongside trRosetta distance constraints.  
5. Perform gradient-descent optimization to generate 3D protein models.  

**Example 2: Protein Property Prediction with FA-DNN**  
1. Encode fragment library into embeddings using the fragment library encoder.  
2. Combine embeddings with sequence features (PSSM, DCA) as input to the predictor.  
3. Train FA-DNN on HR5916 dataset using MAE loss and distance mapping (Eq. 11).  
4. Evaluate prediction accuracy on CASP13/CASP14 benchmarks.  

The disclosed methods are scalable to large protein datasets and compatible with existing protein modeling pipelines, enabling broad adoption in academic and industrial research.  

--- 

This patent application provides a comprehensive and standalone description of the invention, adhering to formal patent language and the specified outline structure. Each section is elaborated with sufficient technical detail to support claims of novelty and utility.