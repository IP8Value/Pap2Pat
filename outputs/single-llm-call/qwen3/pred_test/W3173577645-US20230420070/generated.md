# DESCRIPTION

## BACKGROUND

- motivate protein structure prediction

Protein structure prediction remains one of the most enduring and computationally demanding challenges in molecular biology, with profound implications for drug discovery, enzyme design, and understanding disease mechanisms at the atomic level. For decades, researchers have sought to infer the three-dimensional conformation of a protein solely from its amino acid sequence, a problem often described as the “protein folding problem.” While advances in experimental techniques such as cryo-electron microscopy and X-ray crystallography have expanded the structural database, the vast majority of known protein sequences lack experimentally determined structures, creating a critical gap between sequence availability and structural knowledge. Traditional methods relied on homology modeling, which requires evolutionarily related templates, but this approach fails for proteins with no detectable homologs of known structure—so-called “orphan” proteins. Ab initio methods emerged as alternatives, attempting to predict structure from physical principles and statistical potentials without reliance on templates. Among these, fragment assembly has proven particularly effective, leveraging short, locally conserved structural motifs extracted from known protein structures to guide the folding process. These fragments, typically 3 to 15 residues in length, are assembled in a combinatorial fashion to build plausible full-length models. However, despite their widespread use in systems such as Rosetta and Quark, the rich structural information embedded within fragment libraries has historically been treated as a black box—used for sampling but not systematically analyzed or repurposed for other predictive tasks. Recent breakthroughs in deep learning, notably AlphaFold2, have demonstrated that end-to-end neural networks can achieve remarkable accuracy by learning complex patterns from co-evolutionary signals and sequence alignments. Yet, even these state-of-the-art approaches rely primarily on sequence-derived features such as multiple sequence alignments and residue co-variation statistics, leaving untapped the wealth of geometric and conformational data contained within experimentally derived fragment libraries. These libraries, constructed from high-resolution structures in the Protein Data Bank, encode not only secondary structure propensities but also precise inter-residue distances, torsional angles, and backbone geometries that reflect the local physical constraints governing protein folding. The failure to integrate this structural information into modern prediction pipelines represents a significant missed opportunity to enhance both the accuracy and robustness of computational models, particularly for targets with sparse sequence information or low evolutionary conservation. This disclosure addresses this gap by introducing novel methodologies that explicitly extract, quantify, and leverage structural properties from fragment libraries—not merely as sampling candidates, but as foundational inputs for both potential functions in energy minimization and as feature representations in deep learning architectures, thereby enabling a more comprehensive and physically informed approach to protein structure prediction.

## SUMMARY

- introduce protein structure prediction solution
- outline prediction process
- highlight benefits of solution

This disclosure presents a novel, integrated solution for protein structure prediction that fundamentally redefines the role of fragment libraries by transforming them from passive sampling pools into active, information-rich sources of structural constraints and feature inputs. The solution operates through two complementary, synergistic pathways: first, by converting the structural properties of fragments into weighted Gaussian mixture models that serve as protein-specific potential functions within a gradient descent-based folding framework; and second, by encoding fragment libraries into high-dimensional feature representations using a deep neural network architecture that predicts both one-dimensional torsion angles and two-dimensional inter-residue distances with superior accuracy. The prediction process begins with the construction of a fragment library for a target protein sequence, wherein thousands of short structural templates are assembled for each residue position based on sequence similarity and structural compatibility. From this library, a comprehensive set of structural properties—including backbone torsion angles φ and ψ, backbone angles θ and τ, and pairwise Cα–Cα and Cβ–Cβ distances—are extracted and statistically modeled using weighted Gaussian mixture models that account for the confidence of each fragment, as determined by predicted RMSD values. These models are then converted into differentiable potential functions that are incorporated into an energy landscape alongside other constraints, enabling the refinement of candidate structures through gradient-based optimization. Simultaneously, the same fragment library is processed by a fragment library encoder—a hierarchical convolutional neural network—that transforms variable-length fragments into fixed-dimensional embeddings, which are combined with sequence-derived features such as position-specific scoring matrices and co-evolutionary couplings to predict structural properties using a symmetrized residual neural network. The benefits of this dual-pathway approach are manifold. First, it enables the explicit incorporation of local, fragment-derived geometric constraints that are inherently absent from sequence-only models, thereby improving the physical plausibility of predicted structures. Second, it significantly enhances the accuracy of structural property predictions, particularly for torsion angles and Cβ–Cβ distances, outperforming existing state-of-the-art methods on multiple independent benchmarks including CASP13 and CASP14. Third, by leveraging fragment confidence scores and weighted statistical modeling, the method reduces noise and amplifies signal from high-quality fragments, leading to more reliable predictions even for targets with limited evolutionary information. Finally, the approach is modular and generalizable, allowing for seamless integration with existing folding pipelines and deep learning frameworks, offering a scalable and computationally efficient pathway to high-fidelity protein structure prediction without requiring massive structural databases or prohibitively expensive sampling strategies.

## DETAILED DESCRIPTION

- introduce example implementations

This disclosure encompasses multiple example implementations of a protein structure prediction system that leverages fragment libraries as both potential functions and feature inputs. In one implementation, a computing system receives as input a target amino acid sequence and retrieves a precomputed fragment library generated by a high-performance algorithm such as DeepFragLib, wherein each residue position is associated with a set of structural fragments drawn from experimentally determined protein structures. The system then initiates a dual-pathway processing pipeline: one path constructs weighted Gaussian mixture models from the fragments to generate protein-specific potential functions, while the other path encodes the fragments into a tensor representation for input into a deep neural network. Both pathways operate in parallel and may be executed on the same or separate computational resources. The resulting predictions—either a full three-dimensional structure or a set of predicted structural properties—are then combined to produce a final output. In another implementation, the system is deployed as a cloud-based service accessible via application programming interface, allowing users to submit protein sequences and receive predicted structures or structural property maps within minutes. In yet another implementation, the system is embedded within a laboratory information management system used in pharmaceutical research, where predicted structures are used to prioritize targets for experimental validation or to guide the design of protein-binding molecules. Each implementation is configured to handle proteins of varying lengths, from small peptides to multi-domain proteins exceeding 1000 residues, and is optimized for scalability across heterogeneous computing environments.

- define terms used in the disclosure

For the purposes of this disclosure, the term “fragment library” refers to a collection of short, overlapping structural segments, each corresponding to a contiguous region of a target protein sequence, wherein each segment is derived from a known protein structure and represents a plausible local conformation. A “fragment” is defined as a contiguous sequence of amino acid residues with an associated three-dimensional coordinate set, typically ranging from 3 to 15 residues in length. The term “structural property” encompasses any measurable geometric or conformational characteristic of a protein fragment, including but not limited to backbone torsion angles (φ, ψ), backbone angles (θ, τ), and pairwise distances between heavy atoms such as Cα–Cα and Cβ–Cβ. A “weighted Gaussian mixture model” is a probabilistic model that represents the distribution of a structural property across multiple fragments at a given residue position as a linear combination of Gaussian distributions, each assigned a weight proportional to the confidence of its corresponding fragment. The term “fragment library encoder” refers to a deep neural network component that transforms a tensor representation of a fragment library into a latent feature space that captures high-level structural patterns. The term “property predictor” refers to a neural network module that takes as input the encoded fragment features and sequence-derived features to predict structural properties of the target protein. The term “potential function” refers to a differentiable energy term that penalizes deviations from desired structural properties during structure optimization. The term “residue position” denotes the sequential index of an amino acid within a protein chain, and “template fragment” refers to a fragment retrieved from a structural database that serves as a candidate for inclusion in the fragment library.

- describe structure of a protein

A protein is a linear polymer composed of amino acid residues linked by peptide bonds, which fold into a unique three-dimensional conformation dictated by the sequence of its constituent residues. The primary structure is the linear sequence of amino acids, while the secondary structure consists of locally stabilized motifs such as α-helices and β-strands, formed by hydrogen bonding between backbone atoms. Tertiary structure refers to the overall three-dimensional arrangement of all atoms in the polypeptide chain, stabilized by a combination of hydrophobic interactions, van der Waals forces, hydrogen bonds, salt bridges, and disulfide linkages. The geometry of a protein is defined by a set of dihedral and bond angles, including the backbone torsion angles φ (phi) and ψ (psi), which describe rotations around the N–Cα and Cα–C bonds, respectively, and the backbone angles θ and τ, which describe planar and dihedral angles involving successive Cα atoms. The spatial relationships between atoms are further characterized by inter-residue distances, such as the distance between Cα atoms of residues i and j, or between Cβ atoms, which reflect the compactness and packing of the protein core. These geometric parameters are highly constrained by steric and energetic considerations, and their collective values determine the stability and function of the folded protein.

- describe fragment assembly for protein structure prediction

Fragment assembly is a computational method for predicting protein structure by assembling short, locally conserved structural motifs—known as fragments—into a full-length model. Fragments are typically extracted from known protein structures in the Protein Data Bank and selected based on sequence similarity to the target protein. For each residue position in the target sequence, a library of fragments is compiled, each representing a possible local conformation. These fragments are then combined in a combinatorial fashion, often using Monte Carlo or simulated annealing techniques, to generate candidate structures that satisfy both local structural preferences and global folding constraints. The process relies on the assumption that local structural elements are largely independent of long-range interactions and can be reliably predicted from sequence context. The assembled model is refined through energy minimization, where scoring functions evaluate the compatibility of the structure with physical principles and statistical preferences derived from known structures. Fragment assembly has been successfully implemented in systems such as Rosetta and Quark, and remains a cornerstone of ab initio structure prediction, particularly for targets lacking detectable homologs.

- describe limitations of current protein structure prediction methods

Current protein structure prediction methods exhibit several critical limitations. Homology modeling, while accurate for proteins with close evolutionary relatives, fails entirely for orphan proteins with no detectable templates. Sequence-based deep learning models, such as those relying on multiple sequence alignments and co-evolutionary signals, perform well on many targets but degrade significantly when sequence coverage is sparse or when evolutionary information is insufficient, as is common in metagenomic or synthetic protein datasets. Furthermore, these models often treat structural properties as independent predictions, neglecting the interdependence between torsion angles, distances, and secondary structure. Many methods also rely on rigid, predefined energy functions that do not adapt to the unique structural context of individual proteins, leading to suboptimal sampling and poor discrimination between native-like and non-native conformations. Additionally, fragment-based methods, despite their empirical success, typically treat fragments as discrete sampling units without quantifying or weighting their structural reliability, resulting in the inclusion of low-confidence, high-error fragments that introduce noise into the prediction process. These limitations collectively constrain the accuracy, robustness, and generalizability of current approaches, particularly for challenging targets with low sequence complexity or limited evolutionary information.

- introduce solution for protein structure prediction

The solution introduced herein overcomes these limitations by treating fragment libraries not merely as sources of candidate structures, but as rich, quantifiable sources of structural information that can be systematically encoded into both energy functions and deep learning features. By extracting and modeling the statistical distributions of structural properties directly from fragments, the method generates protein-specific potentials that guide structure optimization toward physically plausible conformations. Simultaneously, by encoding fragment libraries into high-dimensional feature representations using a neural network, the method enables the prediction of structural properties with greater accuracy than methods relying solely on sequence information. This dual-pathway approach integrates the strengths of physics-based modeling and data-driven learning, resulting in a more comprehensive, adaptive, and accurate framework for protein structure prediction.

- describe example environment for implementing the solution

The solution may be implemented in a variety of computing environments, ranging from high-performance workstations to distributed cloud infrastructures. In one embodiment, the system is deployed on a server cluster equipped with multiple graphics processing units (GPUs) to accelerate neural network inference and gradient-based optimization. The system is configured to receive protein sequences via a web interface or application programming interface, process them through the fragment assembly and prediction pipelines, and return predicted structures or structural property maps in standard file formats such as PDB or CIF. The environment supports batch processing of multiple targets and includes automated quality assessment modules to evaluate the reliability of predictions. Data storage is managed through a scalable database system that caches precomputed fragment libraries and model parameters, reducing computational overhead for frequently requested sequences.

- describe components of a computing device

The computing device used to implement the solution comprises a central processing unit (CPU), one or more graphics processing units (GPUs), system memory (RAM), non-volatile storage (such as solid-state drives), input/output interfaces, and a communication unit for network connectivity. The CPU executes control logic and coordinates data flow between components, while the GPUs accelerate matrix operations critical to neural network inference and gradient descent. System memory stores active data and intermediate results during computation, and non-volatile storage retains the fragment libraries, model weights, and input sequences. Input devices such as keyboards or touchscreens allow user interaction, while output devices such as displays present prediction results. The communication unit enables secure data transfer to and from remote clients or cloud services.

- describe functions of components of the computing device

The central processing unit orchestrates the execution of software modules, including fragment library retrieval, feature encoding, potential function generation, and structure optimization. The graphics processing units perform parallel computations required for convolutional operations in the fragment library encoder and residual neural network, as well as for iterative minimization of energy functions. System memory holds the target sequence, fragment tensor, and intermediate feature maps during processing. Non-volatile storage maintains persistent copies of trained models, fragment databases, and user inputs. The communication unit facilitates remote access, enabling cloud-based deployment and integration with laboratory information systems. Input devices allow users to upload sequences or adjust prediction parameters, while output devices visualize predicted structures, confidence scores, and error metrics.

- describe cloud computing architecture

The cloud computing architecture comprises a distributed network of virtualized computing resources, including compute instances, storage services, and networking infrastructure, accessible over the internet. Compute instances are dynamically allocated to handle incoming prediction requests, with autoscaling mechanisms ensuring efficient resource utilization during peak demand. Storage services host fragment libraries and model checkpoints, with redundancy and access controls ensuring data integrity and security. Networking components enable low-latency communication between client applications and backend services. The architecture supports containerized deployment using Docker or Kubernetes, allowing for rapid scaling, version control, and fault tolerance.

- describe use of computing device for protein structure prediction

The computing device is configured to execute a software pipeline that accepts a protein sequence as input, retrieves a precomputed fragment library, extracts structural properties, generates weighted Gaussian mixture models, encodes fragments into feature representations, predicts structural properties using a deep neural network, and refines a three-dimensional structure through gradient-based optimization. The entire process is automated and may be initiated via command line, web interface, or API. Results are returned in standard formats and may be visualized using molecular graphics software. The system is optimized for throughput and accuracy, enabling high-volume prediction for applications in drug discovery, protein engineering, and structural genomics.

### Example Environment

- describe computing device

The computing device is a specialized workstation or server configured for high-throughput computational biology tasks, equipped with multiple high-performance GPUs, large-capacity memory, and fast storage subsystems. It is designed to run complex deep learning and physics-based simulation workflows simultaneously, with dedicated cooling and power management to sustain prolonged computational loads. The device is connected to a local or remote database containing precomputed fragment libraries and model parameters, enabling rapid access to structural information without redundant computation.

- describe components of computing device

The computing device includes a multi-core central processing unit, one or more tensor processing units optimized for deep learning, random-access memory exceeding 256 gigabytes, solid-state storage exceeding 10 terabytes, high-bandwidth network interfaces, and input/output ports for peripheral devices. The system runs a Linux-based operating system with optimized drivers for GPU acceleration and supports containerized execution environments for reproducibility.

- describe processing unit

The processing unit comprises one or more central processing units and graphics processing units that execute the software modules responsible for fragment extraction, feature encoding, potential function generation, and structure optimization. The processing unit is capable of parallel computation across thousands of threads, enabling real-time processing of large fragment libraries and rapid convergence during gradient descent.

- describe memory

The memory includes volatile random-access memory used to store active data structures such as fragment tensors, feature maps, and intermediate energy gradients during computation. Memory is allocated dynamically based on the size of the target protein and the complexity of the fragment library, with garbage collection and memory pooling mechanisms to optimize performance.

- describe storage device

The storage device is a non-volatile solid-state drive array that retains the fragment library databases, trained neural network weights, model configuration files, and user input sequences. The storage is organized hierarchically, with frequently accessed fragments cached in high-speed memory and archival data stored in compressed, indexed formats for efficient retrieval.

- describe communication unit

The communication unit is a high-speed network interface that enables secure data transfer between the computing device and external clients, cloud services, or laboratory information systems. It supports encrypted protocols such as HTTPS and SSH, and is configured to handle concurrent connections from multiple users or automated pipelines.

- describe input device

The input device includes a keyboard, mouse, or touchscreen interface that allows users to submit protein sequences, select prediction parameters, or initiate batch processing. In automated environments, the input device may be replaced by programmatic interfaces such as REST APIs or command-line tools.

- describe output device

The output device includes a high-resolution display, printer, or data export interface that presents predicted protein structures, confidence scores, and structural property maps in visual or file-based formats. Output may also be transmitted to downstream analysis tools for further interpretation or experimental validation.

- describe computing device in a networked environment

In a networked environment, the computing device is connected to a local area network or wide area network, enabling remote access by multiple users or integration into larger computational workflows. The device may act as a server for a cluster of client machines, or as a node in a distributed computing grid, sharing computational load and data resources across multiple locations.

- describe external devices

External devices include laboratory instruments such as mass spectrometers or cryo-electron microscopes that provide experimental data to validate or refine computational predictions. These devices may interface with the computing system via standardized data formats or APIs, enabling closed-loop feedback between prediction and experimentation.

- describe cloud computing architecture

The cloud computing architecture comprises a scalable, distributed infrastructure of virtual machines, storage buckets, and message queues that dynamically allocate resources based on demand. Predictions are queued and processed asynchronously, with results stored in secure, version-controlled repositories accessible via web portals or programmatic interfaces.

- describe components of cloud computing architecture

Components include compute instances for running prediction pipelines, object storage for fragment libraries and model weights, load balancers for distributing requests, container orchestration systems for managing deployments, and monitoring tools for tracking performance and errors. All components are interconnected via secure, low-latency networks and are protected by authentication and access control mechanisms.

- describe services provided by cloud computing

Services include on-demand compute capacity, automated scaling, data persistence, version control for models, user authentication, and result delivery via web interfaces or APIs. Additional services include quality assessment, visualization tools, and integration with external databases such as UniProt or PDB.

- describe data storage in cloud computing

Data is stored in redundant, geographically distributed object storage systems with automatic backup and encryption. Fragment libraries and model parameters are stored in compressed, indexed formats to minimize storage costs and maximize retrieval speed. Access is controlled via role-based permissions and audit logs.

- describe computing resources in cloud computing

Computing resources include virtual machines with GPU acceleration, high-memory instances for large fragment libraries, and batch processing queues for handling large-scale prediction jobs. Resources are allocated on-demand and billed based on usage, enabling cost-effective scaling for academic and industrial users.

- describe use of cloud computing for protein structure prediction

Cloud computing enables high-throughput, on-demand protein structure prediction for users without access to local high-performance computing infrastructure. Predictions are submitted via web interface or API, processed in parallel across hundreds of compute instances, and returned within minutes. The system supports batch processing, versioned model updates, and integration with downstream analysis tools.

- describe input information for protein structure prediction

Input information consists of a protein amino acid sequence in FASTA format, optionally accompanied by metadata such as organism, domain boundaries, or known functional motifs. In some implementations, additional inputs include precomputed multiple sequence alignments or co-evolutionary contact maps.

- describe amino acid sequence of target protein

The amino acid sequence of the target protein is a linear string of single-letter codes representing the sequence of residues, such as “MKTIAAFVLV...”. This sequence is the sole mandatory input and is used to retrieve a fragment library, generate sequence-derived features, and guide the prediction process.

- describe fragment library for target protein

The fragment library for the target protein is a collection of structural fragments, each corresponding to a contiguous region of the protein sequence, where each fragment is derived from a known protein structure and represents a plausible local conformation. Each fragment includes a set of three-dimensional coordinates and associated confidence scores based on predicted RMSD.

- describe residue position of target protein

Each residue position refers to the sequential index of an amino acid within the target protein, ranging from 1 to the total length of the sequence. For each position, a set of fragments is assembled, and structural properties are extracted and modeled independently.

- describe template fragment

A template fragment is a structural segment retrieved from a database of known protein structures and selected for inclusion in the fragment library based on sequence similarity and structural compatibility with the target protein.

### Structural Properties of Proteins and Fragments

- describe structural properties of proteins

Structural properties of proteins encompass the geometric and conformational characteristics that define their three-dimensional shape, including torsion angles, inter-residue distances, secondary structure elements, and backbone orientations. These properties are determined by the chemical nature of amino acids and the physical constraints of peptide bonding and steric exclusion.

- describe inter-residue distances

Inter-residue distances refer to the spatial separation between atoms in non-adjacent residues, such as the distance between Cα atoms or Cβ atoms of residues i and j. These distances reflect the compactness of the protein core and are critical for defining tertiary structure.

- describe Cα-Cα distance

The Cα–Cα distance is the Euclidean distance between the alpha carbon atoms of two amino acid residues. This distance is a key indicator of backbone conformation and is used to assess local folding and packing.

- describe Cβ-Cβ distance

The Cβ–Cβ distance is the spatial separation between the beta carbon atoms of two residues, which reflects side-chain packing and is particularly informative for hydrophobic core formation and protein stability.

- describe inter-residue orientations

Inter-residue orientations describe the relative spatial alignment of residues, including the dihedral angles and vector directions between atoms. These orientations determine the relative positioning of secondary structure elements and are critical for predicting domain interfaces.

- describe torsion angles φ and ω

The torsion angle φ (phi) is the rotation around the N–Cα bond, and ω (omega) is the rotation around the C–N peptide bond. These angles define the backbone conformation and are highly constrained in folded proteins, with φ typically ranging from –180° to 180° and ω nearly fixed at 180° in trans peptide bonds.

- describe backbone angles θ and τ

The backbone angle θ is the planar angle formed by three successive Cα atoms, and τ is the dihedral angle formed by four successive Cα atoms. These angles capture the curvature and twist of the polypeptide chain and are informative for helical and strand geometries.

- describe other orientations between atoms

Other orientations include the angles between side-chain atoms, such as chi angles, and the relative orientations of hydrogen bond donors and acceptors, which are critical for stabilizing secondary and tertiary structures.

- describe bond lengths and bond angles

Bond lengths refer to the fixed distances between covalently bonded atoms, such as C–N or C–C, while bond angles describe the angles between adjacent bonds, such as the Cα–C–N angle. These parameters are governed by chemical bonding rules and are relatively invariant across proteins.

- describe secondary structure of a fragment

The secondary structure of a fragment is classified into helical, strand, or coil conformations based on hydrogen bonding patterns and backbone torsion angles. A fragment is assigned a secondary structure type if a majority of its residues adopt a consistent conformation, such as α-helix or β-sheet.

### Evaluation of Fragment Library

- introduce fragment library evaluation

Fragment library evaluation involves quantifying the accuracy and completeness of structural information contained within a set of fragments assembled for a target protein. This evaluation determines the suitability of the library for downstream prediction tasks.

- define evaluation metrics

Evaluation metrics include precision, coverage, and fragment-level measures of structural property accuracy, such as mean absolute error in torsion angles and inter-residue distances.

- describe precision and coverage metrics

Precision is the proportion of fragments in the library that closely match the native structure, as measured by root-mean-square deviation below a threshold. Coverage is the proportion of residue positions in the target protein that are represented by at least one high-quality fragment.

- introduce structural property metrics

Structural property metrics quantify the fidelity of geometric features such as torsion angles, backbone angles, and pairwise distances within the fragments relative to the native structure.

- define accuracy of secondary structure

The accuracy of secondary structure is the proportion of fragments whose assigned secondary structure matches that of the corresponding region in the native protein.

- define error of angles φ, ψ, ω, θ and τ

The error of torsion and backbone angles is the mean absolute difference between the angle values in the fragments and those in the native structure, averaged across all fragments and positions.

- define error of Cα-Cα and Cβ-Cβ distances

The error of Cα–Cα and Cβ–Cβ distances is the mean absolute deviation between the distances measured in the fragments and those in the native structure, averaged over all fragment-residue pairs.

- describe evaluation of fragment libraries built by different algorithms

Fragment libraries generated by different algorithms—such as DeepFragLib, NNMake, and Flib-Coevo—are evaluated using the same set of metrics to compare their performance in capturing structural information.

- select algorithm based on evaluation metrics

The algorithm that achieves the highest precision, coverage, and lowest structural error across multiple test sets is selected as the preferred method for fragment library construction.

- describe process of selecting algorithm

The selection process involves computing evaluation metrics for each algorithm on a benchmark dataset, ranking them by aggregate performance, and choosing the algorithm with the highest mean score across all metrics.

- calculate evaluation metrics for each fragment library

For each fragment library, the accuracy of secondary structure, error of torsion angles, and error of inter-residue distances are computed using mathematical expectations over all fragments and residue positions.

- compare evaluation metrics among fragment libraries

Evaluation metrics are compared using statistical tests to determine whether differences in performance are significant, with p-values and confidence intervals used to assess reliability.

- select algorithm with best performance

The algorithm demonstrating the most consistent and superior performance across all metrics and test sets is selected for use in downstream prediction tasks.

- describe advantages of using evaluation metrics

Using comprehensive evaluation metrics enables objective comparison of fragment libraries, identifies sources of error, and guides the selection of optimal algorithms and confidence thresholds.

- summarize evaluation metrics

The evaluation metrics collectively provide a multidimensional assessment of fragment library quality, combining global measures such as precision and coverage with local, geometric fidelity measures to ensure high structural accuracy.

- conclude evaluation of fragment library

The evaluation confirms that fragment libraries constructed by DeepFragLib exhibit superior structural accuracy and coverage compared to other methods, making them ideal for use in advanced prediction pipelines.

### Prediction of Protein Structure

- introduce protein structure prediction

Protein structure prediction is the computational inference of the three-dimensional conformation of a protein from its amino acid sequence. This disclosure presents a novel method that integrates fragment-derived potentials with gradient-based optimization to generate high-accuracy models.

- describe prediction module

The prediction module is a software component that takes as input a fragment library and a target sequence, generates weighted Gaussian mixture models of structural properties, converts them into potential functions, and minimizes the combined energy landscape to produce a predicted structure.

- extract structural properties from fragment library

Structural properties including φ, ψ, θ, τ, Cα–Cα, and Cβ–Cβ distances are extracted from each fragment in the library and aggregated by residue position.

- determine feature representation of structural properties

Each structural property is represented as a probability distribution modeled by a weighted Gaussian mixture model, parameterized by mean, variance, and weight for each component.

- describe process of predicting protein structure

The process begins with the initialization of a random or extended conformation, followed by iterative refinement through gradient descent, where the energy function is composed of fragment-derived potentials and geometric constraints.

- extract fragments from initial fragment library

Fragments are selected based on their confidence scores, and variable-length fragments are smoothed into fixed-length sub-fragments using a sliding window.

- process initial fragment library

The fragment library is normalized, weighted, and transformed into a tensor representation where each position contains a set of structural property values for each fragment.

- generate fragments with same length

All fragments are cut into 7-residue sub-fragments using a sliding window to ensure uniformity for statistical modeling.

- determine probability distribution of structural properties

For each residue position and each structural property, a weighted Gaussian mixture model is fitted to the distribution of values across all fragments.

- describe Gaussian mixture models

A Gaussian mixture model is a probabilistic model that represents a distribution as a weighted sum of multiple Gaussian distributions, each characterized by a mean, variance, and weight.

- assign weights to fragments

Weights are assigned to fragments using a softmax function applied to predicted RMSD values, with lower RMSD fragments receiving higher weights.

- build weighted Gaussian mixture models

Weighted Gaussian mixture models are constructed for each structural property at each residue position, with four components selected based on Bayesian Information Criterion.

- determine feature representation of structural properties

The feature representation consists of the parameters of the weighted Gaussian mixture models: mean, variance, and weight for each of the four components per property per position.

- generate potential function from Gaussian distribution

A negative log-likelihood function is derived from each Gaussian mixture model, converting the probability distribution into an energy penalty for deviations from the expected structural property.

- describe potential functions for different structural properties

Potential functions are defined for each of the six structural properties: φ, ψ, θ, τ, Cα–Cα, and Cβ–Cβ, each penalizing deviations from the fragment-derived distribution.

- combine potential functions

The individual potential functions are summed into a composite energy function, with weights manually tuned on a reference dataset to maximize prediction accuracy.

- tune weights on reference dataset

Weights are optimized on the CASP12FM dataset by maximizing the mean TM-score of predicted structures, using grid search and gradient-based optimization.

- describe target function for structure prediction model

The target function is the total energy of the predicted structure, defined as the sum of fragment-derived potentials, geometric penalties, and contact constraints.

- minimize target function

The target function is minimized using gradient descent, iteratively adjusting atomic coordinates to reduce the total energy until convergence.

- generate predicted structure of target protein

The final set of atomic coordinates after convergence constitutes the predicted three-dimensional structure of the target protein.

- describe advantages of using potential functions

Fragment-derived potential functions provide physically grounded, protein-specific constraints that improve the accuracy and reliability of predicted structures, particularly for targets with limited evolutionary information.

- compare with other methods

Compared to methods relying solely on sequence-derived constraints, the inclusion of fragment-derived potentials results in significant improvements in TM-score and better discrimination between native-like and non-native decoys.

- summarize protein structure prediction

The prediction process integrates fragment libraries as probabilistic constraints, enabling accurate, physics-informed structure generation without reliance on homology or co-evolutionary signals.

- conclude protein structure prediction

The method produces high-quality protein structures across diverse target classes, demonstrating robustness and generalizability across multiple benchmark datasets.

- finalize protein structure prediction

The final predicted structure is validated using quality assessment metrics and formatted for downstream applications in structural biology and drug design.

### Prediction of Protein Structural Properties

- introduce protein structural properties prediction

Protein structural properties prediction involves estimating the geometric features of a protein, such as torsion angles and inter-residue distances, from its amino acid sequence. This disclosure presents a deep learning method that leverages fragment libraries to enhance prediction accuracy.

- describe fragment library property set extraction

Structural properties are extracted from each fragment in the library, including secondary structure, torsion angles, and inter-residue distances, and organized into a tensor representation.

- extract structural properties for each residue position

For each residue position, the structural properties of all fragments are collected and aligned to the corresponding position in the target sequence.

- pad fragments to have a length of R residues

Fragments of variable length are padded or truncated to a uniform length of 15 residues to enable batch processing in neural networks.

- represent fragment library property set as L×F×R×D tensor

The fragment library is represented as a four-dimensional tensor with dimensions corresponding to protein length (L), number of fragments per position (F), fragment length (R), and feature dimension (D).

- input fragment library property set to feature encoder

The tensor is fed into a fragment library encoder composed of stacked convolutional layers with residual connections to extract high-level structural features.

- generate fragment library feature set by encoding

The encoder outputs a reduced-dimensional feature tensor that captures latent patterns in the fragment library, preserving positional and structural information.

- obtain structural feature at each residue position

The feature tensor is averaged across the fragment dimension to produce a single feature vector per residue position, representing the collective structural context.

- describe feature encoder architecture

The feature encoder consists of eight residual blocks, each containing two one-dimensional convolutional layers with 64 filters, ELU activation, and batch normalization.

- perform convolution process

Convolutional operations are applied along the fragment length dimension to capture local interactions between residues within each fragment.

- select implicit representation of one residue

The hidden state of the first residue in each fragment is selected to maintain alignment with the target sequence position.

- average all F fragments at the same residue position

The feature vectors from all fragments at a given position are averaged to produce a single, robust representation per residue.

- input fragment library feature set to property predictor

The encoded fragment features are concatenated with sequence-derived features and fed into a two-dimensional residual neural network.

- input sequence feature set to property predictor

Sequence features include one-hot encoding of the amino acid sequence, position-specific scoring matrices, and co-evolutionary coupling scores.

- predict structural properties of target protein

The property predictor outputs predicted values for torsion angles φ, ψ, θ, τ and Cβ–Cβ inter-residue distances.

- describe property predictor architecture

The property predictor is a 30-layer residual neural network with 64 filters per convolutional layer, ELU activation, batch normalization, and dropout to prevent overfitting.

- perform pre-processing on input features

Input features are tiled into two dimensions and concatenated to form a square matrix representing residue-residue interactions.

- use two-dimensional residual neural network

The network applies convolutional filters across both sequence dimensions to capture long-range dependencies and symmetries in structural properties.

- perform symmetrization operation

The output of the final layer is symmetrized to enforce reciprocity in distance predictions, ensuring that the predicted distance from residue i to j equals that from j to i.

- predict different structural properties

Separate output branches predict 1D torsion angles and 2D inter-residue distances, with the latter transformed using a hyperbolic tangent mapping to improve gradient flow.

### Example Method and Example Implementations

- describe method for protein structure prediction

The method comprises determining fragments for each residue position, generating a first feature representation of structural properties, and predicting the protein structure by minimizing a target function derived from weighted Gaussian mixture models.

- determine fragments for each residue position

Fragments are retrieved from a precomputed library based on sequence similarity and structural compatibility with the target protein.

- generate first feature representation of structures

Structural properties are extracted and modeled as weighted Gaussian mixture models, producing a probabilistic representation of each property at each position.

- determine prediction of structure or structural property

The prediction is either a full three-dimensional structure or a set of predicted structural properties, depending on the output mode of the system.

- describe generating first feature representation

The first feature representation is generated by fitting a weighted Gaussian mixture model to the distribution of structural property values across fragments at each position.

- determine property value of structural property

The property value is the mean or mode of the fitted Gaussian mixture model, representing the most probable value for that property at that position.

- determine probability distribution of structural property

The probability distribution is defined by the parameters of the weighted Gaussian mixture model: means, variances, and weights for each component.

- describe determining prediction of structure

The prediction of structure is determined by minimizing a target energy function composed of fragment-derived potentials and geometric constraints using gradient descent.

- generate potential function

The potential function is generated by computing the negative log-likelihood of the observed structural property values under the fitted Gaussian mixture model.

- determine target function of structure prediction model

The target function is the sum of all fragment-derived potentials, contact constraints, and stereochemical penalties.

- determine prediction of structure by minimizing target function

The target function is minimized iteratively using gradient descent, adjusting atomic coordinates until convergence to a low-energy conformation.

- describe determining plurality of fragments

A plurality of fragments is determined by selecting the top N fragments with the lowest predicted RMSD values for each residue position.

- determine initial fragments

Initial fragments are retrieved from a database of known structures using sequence alignment and structural clustering algorithms.

- generate fragments with predetermined number of residues

Fragments are cut into uniform lengths of 7 or 15 residues using a sliding window to ensure compatibility with modeling and encoding pipelines.

- describe structural property

A structural property is a measurable geometric characteristic of a protein fragment, such as a torsion angle, backbone angle, or inter-residue distance.

- describe generating first feature representation

The first feature representation is generated by modeling the distribution of each structural property across fragments as a weighted Gaussian mixture model.

- determine plurality of structural properties

A plurality of structural properties includes φ, ψ, θ, τ, Cα–Cα, and Cβ–Cβ distances, each modeled independently.

- encode structural properties according to feature encoder

Structural properties are encoded into a latent feature space using a hierarchical convolutional neural network that captures local and global structural patterns.

- describe determining prediction of structural property

The prediction of structural property is determined by feeding encoded fragment features and sequence features into a residual neural network that outputs real-valued predictions.

- determine second feature representation of amino acid sequence

The second feature representation includes one-hot encoding of the amino acid sequence, position-specific scoring matrices, and direct coupling analysis scores.

- determine prediction of structural property

The prediction is computed as the output of the property predictor network, which maps the combined feature representation to predicted torsion angles and distances.

- describe selecting target algorithm

The target algorithm is selected based on its performance across evaluation metrics, with DeepFragLib chosen for its superior precision and structural accuracy.

- determine reference property values

Reference property values are obtained from experimentally determined structures in the Protein Data Bank.

- determine true property value

The true property value is the measured value of a structural property in the native structure of the protein.

- determine difference between reference and true property values

The difference is computed as the mean absolute error between predicted and true values, used to evaluate algorithm performance.

- select target algorithm based on differences

The algorithm with the smallest mean absolute error across all structural properties and test sets is selected as the target algorithm.

- describe electronic device

The electronic device is a computing system configured to execute the prediction pipeline, comprising a processor, memory, storage, and communication interfaces.

- describe processing unit and memory

The processing unit executes the prediction software, while the memory stores active data such as fragment tensors, feature maps, and model parameters during computation.

- describe instructions stored in memory

Instructions stored in memory include code for fragment retrieval, feature encoding, potential function generation, gradient descent optimization, and result output.

- describe computer program product

The computer program product comprises a non-transitory computer-readable medium storing executable instructions that, when executed by a processor, cause the system to perform protein structure prediction.

- describe computer-readable medium

The computer-readable medium is a tangible storage device such as a solid-state drive, optical disc, or flash memory containing the software instructions for implementing the prediction method.

- describe hardware logic components

Hardware logic components include field-programmable gate arrays or application-specific integrated circuits designed to accelerate specific operations such as convolution or gradient computation.

- describe program code

Program code is written in Python and C++ and compiled for execution on CPUs and GPUs, with libraries such as PyTorch and TensorFlow used for neural network operations.

- describe machine-readable medium

The machine-readable medium is a physical storage medium encoded with instructions that can be read and executed by a computing device to perform the disclosed method.