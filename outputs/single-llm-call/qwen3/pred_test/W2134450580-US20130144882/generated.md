# DESCRIPTION

## BACKGROUND OF THE INVENTION

- define field of invention  
The present invention resides in the field of computational biology and diagnostic oncology, specifically in the development of advanced systems and methods for the analysis of gene expression patterns in human epithelial cancers through integrative network-based profiling. This field encompasses the application of bioinformatics, systems biology, and machine learning to identify subtle but biologically significant alterations in gene expression that are not discernible through conventional single-gene analyses. The invention is particularly directed toward the transformation of high-dimensional gene expression data into clinically actionable signatures by leveraging the topological structure of protein-protein interaction networks, thereby enabling more accurate classification, prognosis, and therapeutic decision-making in epithelial malignancies.

- describe limitations of current approaches  
Current methodologies for analyzing gene expression in cancer rely predominantly on statistical comparisons of individual gene expression levels between diseased and healthy tissues. These approaches often disregard the functional relationships among genes, treating each transcript as an independent variable despite well-established biological evidence that gene function is inherently contextual and network-dependent. As a result, many genes with modest but coordinated expression changes—particularly those involved in regulatory cascades, signaling modules, or metabolic pathways—are dismissed as noise or filtered out due to insufficient fold-change thresholds. Furthermore, existing clustering and ranking techniques fail to preserve the topological integrity of biological networks during data transformation, leading to loss of spatial and relational information critical for capturing disease-specific molecular phenotypes. The inability to integrate local gene connectivity with global network architecture limits the sensitivity and specificity of current diagnostic models, especially in heterogeneous cancers where no single gene serves as a universal biomarker.

- motivate need for new approach  
There exists a critical and unmet need for a method that can synthesize gene expression data with the structural organization of molecular interaction networks to reveal hidden patterns of dysregulation that emerge only through collective behavior. Such a method must account for the fact that disease states are not defined by isolated gene aberrations but by perturbations in interconnected systems. A novel approach is required that transforms raw expression profiles into integrated signatures reflecting the spatial organization of functionally related genes, thereby enhancing classification accuracy, reducing false negatives, and uncovering novel candidate biomarkers that are individually weak but collectively decisive. This innovation must be scalable, reproducible, and applicable across diverse microarray and sequencing platforms to facilitate clinical translation and broad adoption in diagnostic laboratories.

## SUMMARY OF THE INVENTION

- introduce MIXP approach  
The present invention introduces a novel method termed MIXP—Multiscale Integrative eXpression Profiling—which transforms gene expression data into a topologically informed, network-derived signature capable of significantly improving the classification of epithelial cancers. Unlike conventional approaches that analyze genes in isolation, MIXP integrates gene expression values with the structural topology of protein-protein interaction networks to generate a unified, ordered representation of molecular activity. This transformation enables the amplification of subtle, coordinated expression signals that are otherwise undetectable, thereby enhancing diagnostic precision and uncovering novel biological insights into disease progression.

- describe network-based gene expression analysis  
MIXP operates on the principle that genes participating in shared biological functions are physically and functionally connected within protein interaction networks. By mapping gene expression profiles onto these networks, the method captures not only the magnitude of expression change but also its spatial context relative to neighboring genes. This network-based analysis reveals emergent patterns of dysregulation that arise from the collective behavior of functionally related gene clusters, providing a systems-level view of disease pathology that transcends the limitations of single-gene analyses.

- motivate iterative weighing of network nodes  
The invention is grounded in the recognition that the relative importance of network nodes cannot be adequately captured by static metrics such as degree centrality or simple ranking. Instead, iterative, dynamic reweighting of nodes based on their position within the network and their expression profile enables the emergence of biologically meaningful orderings that reflect functional modules and hierarchical organization. This iterative process, driven by stochastic optimization, allows the system to converge upon an optimal arrangement that maximizes the coherence of expression patterns across topologically proximal nodes.

- describe MIXP method  
The MIXP method comprises four sequential steps: seed gene selection, network reconstruction, network reordering, and expression integration. Seed genes associated with epithelial cancer are identified from curated databases and used as anchors for network expansion. A disease-specific protein-protein interaction network is then constructed using nearest-neighbor expansion within a comprehensive human interactome database. The adjacency matrix of this network is reordered using an ant colony optimization algorithm to reveal fractal-like topological patterns indicative of functional modularity. Finally, gene expression values are mapped onto the reordered gene list and integrated using a Gaussian influence function, producing a one-dimensional MIXP profile that encapsulates both local expression intensity and global network topology.

- outline network analysis algorithms  
The core algorithmic framework of MIXP employs the ant colony optimization reordering (ACOR) algorithm, a bio-inspired computational technique that simulates the behavior of artificial ants traversing the network to deposit pheromone-like weights on edges and nodes. Through iterative refinement, this process identifies an optimal node ordering that clusters functionally related genes while preserving their hierarchical relationships. The resulting reordered adjacency matrix exhibits a fractal-like structure, demonstrating that the network’s organization is not random but reflects an underlying biological hierarchy.

- describe database creation method  
The invention utilizes a curated, high-confidence human protein-protein interaction database derived from multiple experimental and computational sources, including HAPPI, STRING, and BioGRID. This database is filtered for tissue-specific relevance to epithelial tissues and validated for reproducibility across independent datasets. The resulting network is annotated with gene ontology terms, pathway memberships, and expression confidence scores to ensure biological fidelity.

- outline gene-expression profile mapping  
Gene expression profiles from microarray or RNA-Seq platforms are normalized and aligned to the gene identifiers in the reordered network. Each gene’s expression value is assigned to its corresponding position in the ordered list, ensuring that the spatial arrangement of genes reflects their functional relationships rather than arbitrary indexing.

- describe system for determining health situation propensity  
The MIXP profile is used as an input feature vector for supervised machine learning classifiers, such as support vector machines, to determine the propensity of a biological sample to be classified as healthy, pre-malignant, or malignant. The system outputs a quantitative score reflecting the likelihood of epithelial cancer presence, enabling risk stratification and early detection.

- provide disclaimer for drawings  
The accompanying figures are provided for illustrative purposes only and are not intended to limit the scope of the invention. The exact visual representation of network structures, heat maps, or integrated profiles may vary depending on data source, normalization method, or algorithmic parameters without departing from the essential principles of the invention.

## DESCRIPTION OF EMBODIMENTS OF THE PRESENT INVENTION

- introduce gene expression profiling  
Gene expression profiling refers to the measurement of the activity levels of thousands of genes simultaneously in a biological sample, providing a molecular snapshot of cellular state. In the context of epithelial cancers, this technique enables the identification of transcriptional programs associated with tumor initiation, progression, and metastasis.

- describe DNA Microarray technology  
DNA microarray technology facilitates high-throughput gene expression analysis by hybridizing labeled mRNA from a sample to complementary DNA probes immobilized on a solid substrate. The intensity of fluorescence at each probe spot correlates with the abundance of the corresponding transcript, enabling quantitative comparison across multiple samples.

- explain sequence-based techniques  
Sequence-based techniques, including SuperSAGE and RNA-Seq, offer higher resolution and broader dynamic range than microarrays by directly sequencing cDNA fragments derived from mRNA. These methods enable the detection of novel transcripts, splice variants, and low-abundance genes, enhancing the comprehensiveness of expression profiling.

- introduce SuperSAGE  
SuperSAGE is a tag-based sequencing method that generates short, unique sequence tags from the 3′ end of transcripts, allowing for precise quantification and cross-platform compatibility. It is particularly useful in samples with limited RNA quantity and provides a robust alternative to traditional microarray platforms.

- describe RNA-Seq  
RNA-Seq utilizes next-generation sequencing to capture the entire transcriptome, offering unparalleled sensitivity and the ability to detect non-coding RNAs, fusion transcripts, and allele-specific expression. Its application in cancer research has revealed complex regulatory landscapes previously inaccessible to array-based methods.

- motivate expression profiling  
Expression profiling is motivated by the need to move beyond single-gene biomarkers toward systems-level understanding of disease. In epithelial cancers, where heterogeneity and plasticity are hallmarks, profiling enables the identification of coherent molecular signatures that reflect underlying biological processes rather than isolated anomalies.

- explain gene regulation  
Gene regulation in epithelial tissues is governed by intricate networks of transcription factors, epigenetic modifiers, and signaling cascades that respond to environmental cues, genetic mutations, and cellular stress. Dysregulation of these networks underlies the transition from normal epithelium to invasive carcinoma.

- describe factors affecting gene expression  
Factors influencing gene expression in epithelial cancers include somatic mutations, copy number alterations, promoter methylation, microRNA activity, and stromal interactions. These factors act synergistically to reprogram transcriptional output, resulting in disease-specific expression signatures.

- introduce expression profiling experiments  
Expression profiling experiments are designed to compare gene expression across defined biological conditions, such as tumor versus adjacent normal tissue, or pre-treatment versus post-treatment states. These experiments generate large, multidimensional datasets requiring sophisticated analytical frameworks for interpretation.

- describe measuring mRNA levels  
mRNA levels are measured through hybridization or sequencing, followed by normalization to account for technical variability such as RNA input, labeling efficiency, and batch effects. Accurate quantification is essential for reliable downstream analysis.

- explain contrasting healthy and diseased states  
Contrasting healthy and diseased epithelial states reveals consistent patterns of upregulated oncogenic pathways and downregulated differentiation markers. These patterns form the basis for developing diagnostic classifiers that distinguish benign from malignant tissue.

- introduce gene signatures  
Gene signatures are sets of co-regulated genes whose combined expression pattern correlates with a specific biological condition. In epithelial cancers, such signatures can predict aggressiveness, therapeutic response, or recurrence risk.

- describe GSEA  
Gene Set Enrichment Analysis (GSEA) is a computational method that determines whether predefined sets of genes show statistically significant, concordant differences between two biological states. While useful, GSEA does not account for topological relationships among genes within interaction networks.

- introduce pathway models  
Pathway models represent biological processes as ordered sequences of molecular interactions. These models are valuable for hypothesis generation but often lack the dynamic, context-dependent flexibility required for heterogeneous cancers.

- describe protein-protein interactions  
Protein-protein interactions form the backbone of cellular signaling, structural organization, and metabolic coordination. In epithelial cancers, disruptions in these interactions drive aberrant proliferation, evasion of apoptosis, and metastatic dissemination.

- explain gene regulatory networks  
Gene regulatory networks consist of transcription factors and their target genes, forming hierarchical control systems that dictate cell identity and function. Their perturbation is a hallmark of epithelial malignancies.

- introduce metabolic networks  
Metabolic networks describe the enzymatic conversion of substrates into products and are frequently rewired in cancer to support biosynthetic demands. Their integration with expression data reveals metabolic dependencies that can be therapeutically targeted.

- describe signaling networks  
Signaling networks transmit extracellular cues into intracellular responses through cascades of phosphorylation and protein assembly. In epithelial cancers, these networks are commonly hijacked by oncogenic mutations.

- introduce MAPK/ERK pathway  
The MAPK/ERK pathway is a central signaling cascade frequently activated in epithelial cancers through mutations in RAS, RAF, or EGFR. Its dysregulation leads to sustained proliferation and resistance to apoptosis.

- describe algorithmic representations  
Algorithmic representations of biological networks encode nodes as genes or proteins and edges as interactions, enabling computational manipulation and analysis. These representations are the foundation of network-based diagnostics.

- explain computer operations  
Computer operations in this invention include data loading, normalization, matrix manipulation, iterative optimization, and classification. These operations are executed by specialized software modules running on high-performance computing systems.

- introduce data structures  
Data structures such as adjacency matrices, linked lists, and sparse arrays are employed to efficiently store and manipulate large-scale interaction networks and expression datasets.

- describe machine operations  
Machine operations involve the execution of algorithms on hardware platforms, including parallel processing of network reordering and simultaneous classification of multiple samples.

- explain software modules  
Software modules are discrete, reusable components that perform specific functions such as network construction, ACOR optimization, Gaussian integration, and SVM classification. These modules are interoperable and configurable.

- introduce apparatus for performing operations  
An apparatus for performing MIXP operations includes a computing system with sufficient memory and processing power, coupled with input devices for data upload and output devices for result visualization and reporting.

- describe object-oriented software  
Object-oriented software organizes code into classes and objects that encapsulate data and behavior, facilitating modularity, inheritance, and extensibility in the implementation of the MIXP system.

- explain object-oriented operating system  
An object-oriented operating system manages hardware resources and provides an environment in which object-oriented software modules can interact through well-defined interfaces and message passing.

- introduce messages and events  
Messages and events are used to coordinate operations between software components, such as triggering network reordering upon completion of gene expression normalization.

- describe inheritance in object-oriented systems  
Inheritance allows new classes to derive properties and methods from existing ones, enabling the creation of specialized versions of the MIXP algorithm tailored to different cancer types or data modalities.

- explain object-oriented programming  
Object-oriented programming principles are applied to ensure that the MIXP system is maintainable, scalable, and adaptable to future biological discoveries and technological advancements.

- define key terms  
Key terms such as “seed gene,” “adjacency matrix,” “Gaussian influence function,” and “ACOR algorithm” are defined with precision to ensure consistent interpretation throughout the disclosure.

- explain windowing environment  
A windowing environment provides a graphical interface for users to visualize network structures, integrated expression profiles, and classification outcomes, enabling interactive exploration of results.

- describe network and server  
The invention may be implemented on a networked server system that receives expression data from remote clinical sites, processes it using MIXP, and returns diagnostic scores via secure web portals.

- define process and agent  
A process refers to an executing instance of the MIXP algorithm, while an agent is a software entity that autonomously performs tasks such as data preprocessing or result validation.

- explain module  
A module is a self-contained software unit that performs a specific function within the MIXP system, such as network reconstruction or expression integration.

- describe desktop and API  
A desktop application provides a user-friendly interface for local analysis, while an application programming interface (API) enables integration with electronic health records and laboratory information systems.

- introduce browser  
A web browser allows clinicians to access MIXP results remotely through secure, password-protected portals without requiring local software installation.

- explain SGML and HTML  
Standard Generalized Markup Language (SGML) and Hypertext Markup Language (HTML) are used to structure and display diagnostic reports and network visualizations in standardized formats.

- describe XML file  
XML files are employed to store and exchange gene expression data, network topologies, and classification parameters in a structured, machine-readable format.

- define PDA and WWAN  
Personal digital assistants (PDAs) and wireless wide area networks (WWANs) enable mobile access to diagnostic results in clinical settings where connectivity is limited.

- explain synchronization  
Synchronization ensures that data uploaded from remote devices is consistent with central databases, preventing duplication or version conflicts.

- describe wireless communication  
Wireless communication protocols such as Wi-Fi, Bluetooth, and cellular networks facilitate the transmission of expression data from biopsy sites to analysis servers.

- introduce WAP  
Wireless Application Protocol (WAP) enables secure, low-bandwidth access to MIXP diagnostic services on mobile devices.

- define mobile software and mobile apps  
Mobile software and applications allow clinicians to view classification results, interpret network visualizations, and receive alerts regarding high-risk samples directly on handheld devices.

- explain PACS  
Picture Archiving and Communication Systems (PACS) may be integrated with MIXP to correlate molecular diagnostics with histopathological images, enabling multimodal analysis.

- describe computing environment  
The computing environment for MIXP includes high-performance servers, distributed storage systems, and secure data transmission protocols compliant with HIPAA and GDPR standards.

- illustrate server and clients  
The invention operates in a client-server architecture, where multiple clinical clients submit expression data to a central server that performs MIXP analysis and returns diagnostic outputs.

- describe computer system  
The computer system comprises a central processing unit, memory, storage, input/output controllers, and network interfaces, all interconnected via a system bus to execute the MIXP algorithm.

- explain bus and central processor  
The system bus facilitates data transfer between the central processor and other components, while the central processor executes the computational steps of network reordering and expression integration.

- describe input/output controller  
The input/output controller manages communication between the computer system and external devices such as microarrays, sequencers, and user interfaces.

- explain storage interface  
The storage interface connects the system to persistent storage devices containing gene expression datasets, interaction networks, and classification models.

- describe network interface  
The network interface enables secure, encrypted transmission of data between the MIXP server and remote clinical sites.

- introduce modem  
A modem may be employed to establish connectivity in environments where broadband infrastructure is unavailable.

- explain signal transmission  
Signal transmission is performed using digital encoding protocols to ensure data integrity during transfer across wired and wireless networks.

- describe microarray  
A microarray is a solid substrate bearing thousands of immobilized DNA probes used to measure gene expression levels through hybridization with labeled RNA.

- explain gene expression analysis  
Gene expression analysis involves the quantification of transcript abundance across a genome-wide set of genes, followed by statistical and computational interpretation to identify disease-associated patterns.

- describe heat maps  
Heat maps are graphical representations of gene expression data in which color intensity corresponds to expression level, enabling visual identification of clusters and trends.

- explain network structure comparison  
Network structure comparison involves evaluating the topological similarity between networks derived from different samples or conditions to assess disease-specific perturbations.

- describe iterative sampling and analysis  
Iterative sampling and analysis involve repeated application of the ACOR algorithm to refine node ordering and optimize the integrated expression profile until convergence is achieved.

- explain statistical treatment  
Statistical treatment includes normalization, transformation, and validation of expression data to minimize technical noise and ensure biological relevance.

- describe microarray data sets  
Microarray data sets consist of intensity values measured across hundreds of thousands of probes, representing the expression levels of thousands of genes in a given sample.

- explain normalization methods  
Normalization methods such as quantile normalization, loess correction, and RMA adjustment are applied to correct for systematic biases introduced during sample processing.

- describe probe and mRNA relation  
Each probe on a microarray is designed to hybridize specifically to a complementary mRNA sequence, enabling the indirect measurement of transcript abundance.

- explain amplification bias  
Amplification bias arises during RNA amplification steps and can distort the relative abundance of transcripts, necessitating correction algorithms to preserve biological accuracy.

- describe genomic EST information  
Expressed Sequence Tag (EST) information provides evidence of gene transcription and is used to validate the presence of transcripts in the interaction network.

- introduce MIXP concept  
The MIXP concept is the foundational innovation of this invention, representing a paradigm shift from single-gene to network-integrated expression profiling that enhances diagnostic accuracy and uncovers novel biomarkers.

- explain feature transformation approach  
The MIXP method serves as a knowledge-supervised feature transformation approach, converting raw gene expression data into a topologically informed signature that captures both local and global network properties.

- describe gene discovery  
Through MIXP, previously overlooked genes with low individual differential expression but high network connectivity are identified as potential biomarkers, expanding the repertoire of detectable disease signatures.

- describe MIXP approach  
The MIXP approach integrates gene expression with protein interaction topology using iterative network reordering and Gaussian-based signal integration to generate a unified diagnostic profile.

- introduce network reordering algorithm  
The network reordering algorithm employed in MIXP is the ant colony optimization algorithm, which simulates the behavior of artificial ants to discover optimal node arrangements that reveal functional modularity.

- outline four steps of MIXP modeling  
The four steps of MIXP modeling are: (1) selection of seed genes associated with epithelial cancer; (2) reconstruction of a disease-specific protein-protein interaction network; (3) reordering of the network using the ACOR algorithm; and (4) integration of gene expression values into a one-dimensional profile using a Gaussian influence function.

- describe seed molecule selection  
Seed molecules are selected from curated databases of genes known to be associated with epithelial cancers, including those implicated in proliferation, invasion, and apoptosis.

- describe network reconstruction  
Network reconstruction involves expanding the seed gene set by including their direct interaction partners, resulting in a context-specific subnetwork enriched for disease-relevant biology.

- describe network reordering  
Network reordering is performed using the ACOR algorithm, which iteratively rearranges nodes to maximize the spatial coherence of expression values, producing a fractal-like adjacency matrix.

- describe expression integrating  
Expression integrating involves convolving the ordered gene list with a Gaussian kernel to aggregate the influence of neighboring genes, producing a smooth, continuous profile that reflects collective dysregulation.

- illustrate MIXP approach with Alzheimer's disease example  
Although the invention is directed toward epithelial cancers, the MIXP approach was initially validated using Alzheimer’s disease data, demonstrating its generalizability to complex, multifactorial diseases.

- describe gene expression profiles  
Gene expression profiles in epithelial cancers exhibit distinct patterns of upregulation in oncogenic pathways and downregulation in differentiation markers, which are amplified and clarified by MIXP.

- describe seed gene selection  
Seed gene selection is performed using evidence from published literature, genomic databases, and functional annotations to ensure biological relevance and reproducibility.

- describe network construction  
Network construction employs a nearest-neighbor expansion strategy to build a disease-specific interaction subnetwork from a comprehensive human interactome database.

- describe node-weighted edge-scored PPI network  
The protein-protein interaction network is weighted by node degree and edge confidence scores to reflect the reliability and centrality of interactions.

- describe average differential expression values  
Average differential expression values are calculated by comparing expression levels in cancerous tissue to matched normal tissue, providing a quantitative measure of dysregulation.

- describe network reordering using ACOR algorithm  
The ACOR algorithm reorders the network by simulating ant colonies that traverse paths and deposit pheromones, leading to convergence on an optimal node arrangement that clusters functionally related genes.

- describe ant colony optimization reordering  
Ant colony optimization reordering is a stochastic, population-based method that mimics the collective behavior of ants to solve combinatorial optimization problems, such as node ordering in complex networks.

- describe iteration process of ACOR algorithm  
The iteration process involves repeated traversal of the network by artificial ants, with pheromone updates guiding subsequent paths toward increasingly coherent node arrangements.

- describe reordered adjacency matrix  
The reordered adjacency matrix displays a fractal-like pattern, indicating that the network contains self-similar, hierarchical modules that reflect biological organization.

- describe fractal-like pattern in reordered adjacency matrix  
The fractal-like pattern observed in the reordered adjacency matrix is a signature of functional modularity and hierarchical organization, confirming that the ACOR algorithm has successfully uncovered biologically meaningful structure.

- describe relative position of proteins in reordered network  
The relative position of proteins in the reordered network reflects their functional relatedness, with seed genes and their interactors occupying distinct topological regions that correlate with disease severity.

- describe expression integrating using 1-D Gaussian function  
Expression integrating is performed using a one-dimensional Gaussian function that assigns influence to neighboring genes based on their distance in the reordered list, amplifying weak but coordinated signals.

- describe integrated expression profile MIXP(t)  
The integrated expression profile MIXP(t) is a continuous, one-dimensional vector representing the cumulative expression influence across the reordered network, serving as the primary input for classification.

- describe sample classification using SVM  
Sample classification is performed using a support vector machine trained on MIXP profiles from known cancer and non-cancer samples, achieving high accuracy in blind validation.

- describe input feature scaling  
Input feature scaling normalizes MIXP profiles to a standard range to ensure equal weighting of features during classification and prevent bias from magnitude differences.

- describe classification results for ACOR-based MIXP approach  
Classification results demonstrate that the ACOR-based MIXP approach achieves significantly higher accuracy than conventional methods, including ranking, clustering, and random-ordering variants.

- compare classification results with other approaches  
Compared to other approaches, the ACOR-based MIXP method outperforms them in sensitivity, specificity, and area under the receiver operating characteristic curve, particularly in heterogeneous datasets.

- describe knowledge-supervised feature transformation  
Knowledge-supervised feature transformation refers to the use of prior biological knowledge—in the form of protein interaction networks—to guide the transformation of raw expression data into optimized diagnostic features.

- describe local and global network topology information  
Local topology refers to the immediate neighborhood of a gene, while global topology refers to its position within the overall network structure. Both are captured by MIXP to enhance classification performance.

- describe references used in development of present invention  
References include peer-reviewed publications on protein interaction networks, ant colony optimization, gene expression analysis, and machine learning in biomedical informatics.

- describe HAPPI database  
The HAPPI database is a comprehensive, manually curated repository of high-confidence human protein-protein interactions, used as the foundation for network reconstruction in this invention.

- describe mining Alzheimer disease relevant proteins  
Alzheimer’s disease-relevant proteins were mined from literature and databases to validate the generalizability of the MIXP approach before application to epithelial cancers.

- describe ant algorithms and stigmergy  
Ant algorithms and stigmergy refer to the principles of indirect communication through environmental modification, which underpin the ACOR algorithm’s ability to discover optimal solutions.

- describe walking the interactome for prioritization of candidate disease genes  
Walking the interactome involves traversing the network from seed genes to identify candidate disease genes based on proximity and connectivity, a strategy employed in seed expansion.

- describe efficient algorithm for large-scale detection of protein families  
An efficient algorithm for detecting protein families was adapted to ensure scalability of the network reconstruction step across thousands of genes.

- describe whole-proteome prediction of protein function  
Whole-proteome prediction of protein function was used to annotate uncharacterized genes in the network, enhancing the biological interpretability of the results.

- describe exploiting indirect neighbours and topological weight  
Indirect neighbors and topological weights are exploited to capture higher-order relationships beyond direct interactions, increasing the sensitivity of the method.

- describe finding fractal patterns in molecular interaction networks  
Fractal patterns in molecular interaction networks indicate self-similarity across scales, a property leveraged by MIXP to reveal hierarchical organization of disease biology.