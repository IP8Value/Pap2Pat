Here is the complete patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

The present invention relates generally to computational virology and more specifically to systems and methods for automated strain identification of single-stranded RNA (ssRNA) viruses based on capsid protein sequence analysis. Viral outbreaks caused by ssRNA viruses such as caliciviruses and picornaviruses present significant public health challenges due to their rapid mutation rates and resulting genetic diversity. Traditional methods for viral strain identification rely on homology-based sequence comparisons using complete capsid sequences or other genomic regions. These conventional approaches seek to identify clusters of similar sequences that comprise major groups (genogroups or genera) and sub-groups (species and serotypes). However, such methods suffer from several limitations including exponential computation time dependence on sequence lengths, sensitivity to parameter selection, and inability to reliably handle partial sequence data.  

Current strain recognition techniques using sequence identity scores have proven difficult to automate reliably across ssRNA virus families. Existing methods typically use sequence similarity cut-off values derived from homology-based comparisons between target sequences and known reference sequences. While recent reports indicate reliable estimation of such cut-off values for distinguishing norovirus genogroups, no uniform criteria exist to accurately estimate these values for other caliciviruses. These difficulties are compounded when analyzing different virus genera or families together, or when only partial sequences from smaller, more conserved regions are available. Even for complete capsid sequences, homology-based similarity scores present limitations in determining viral strains due to computational bottlenecks and potential biases introduced by parameter selection.  

Alternative approaches that align sliding windows of target virus sequences against reference sequence databases still depend critically on parameters such as window sizes and reference sequence selection. Incorrect parameter choices may introduce error-inducing biases while significantly increasing computation time due to repetitive runs with different trial parameter values. Therefore, a need exists for more robust, automated methods of viral strain identification that can overcome these limitations while providing reliable results across complete and partial capsid sequences from diverse ssRNA viruses.  

## SUMMARY OF THE INVENTION  

The present invention provides a novel computational system and method for automated strain identification of ssRNA viruses, particularly caliciviruses and picornaviruses, based on partitioned phylogenetic analysis of capsid protein sequences. The invention implements a residue-wise comparison approach to automate strain predictions using both complete and partial amino acid capsid sequences.  

Key aspects of the invention include:  

1. Construction of partitioned phylogenetic trees for viral genera using capsid protein sequences, wherein each partition contains groups of similar sequences emanating from nodes within that partition.  

2. Identification of characteristic residues that are conserved within each sequence group but not among different groups of a partition.  

3. Creation of comprehensive databases storing genus tree information including partitions, sequence groups, and characteristic residues.  

4. Implementation of an efficient tree-based search algorithm that compares target sequence residues with database characteristic residues in a partition-wise manner to identify the closest matching strain.  

5. Capability to handle both complete and partial capsid protein sequences while maintaining computational efficiency.  

6. Optional detection of potential recombination events and spontaneous mutations through partition-wise residue comparisons.  

The system demonstrates particular advantages in processing speed, typically requiring only about 5 seconds per sequence for strain identification, while maintaining high accuracy across diverse viral strains. The method is robust to sequence variations and can reliably identify strains even from partial sequence data.  

## DETAILED DESCRIPTION OF THE INVENTION  

The present invention provides a comprehensive system and method for automated strain identification of ssRNA viruses, with particular application to caliciviruses and picornaviruses. The detailed implementation involves several key components and processes as described below.  

### Phylogenetic Tree Construction  

The system begins by constructing partitioned phylogenetic trees for each viral genus of interest. For caliciviruses, trees are built for the four known genera (noroviruses, sapoviruses, lagoviruses and vesiviruses). For picornaviruses, trees are constructed for the nine known genera (apthoviruses, cardioviruses, enteroviruses, erboviruses, hepatoviruses, kobuviruses, parechoviruses, rhinoviruses and teschoviruses).  

Virus sequences are obtained from public databases and aligned using standard alignment tools such as ClustalW with optimized parameters including the Gonnet-250 distance matrix model and specific gap penalty settings. The Fitch-Margoliash distance matrix is calculated for each set of aligned sequences, and phylogenetic trees are constructed using neighbor-joining algorithms such as the Kitsch algorithm from the PHYLIP package. The trees are displayed with branches horizontal, the root on the left, and tips on the right.  

Topology validation is performed by comparing trees generated using different neighbor-joining methods (UPGMA, minimum evolution with Poisson distance models) and confirming that capsid sequence trees maintain consistent topology regardless of construction method. This validation step confirms the phylogenetic robustness of capsid sequences for strain identification purposes.  

### Tree Partitioning and Characteristic Residue Identification  

Each constructed phylogenetic tree is divided into ten equally spaced partitions (P01-P10), with inter-partition spacing calculated from maximum evolutionary distances within the trees. Ten partitions have been determined to be optimal, as no significant changes in node distributions occur with additional partitions.  

Within each partition, sequence groups are identified as clusters of similar sequences emanating from specific nodes. For each group in a partition, consensus residues are identified that are conserved within the group but not among different groups of the same partition. These "characteristic residues" serve as unique identifiers for each sequence group.  

### Database Creation  

All information about the genus trees including partitions, sequence groups, and characteristic residues is stored in structured databases. These databases use two-dimensional arrays to efficiently organize the information for rapid retrieval during strain identification operations. The databases contain complete phylogenetic information for each viral genus, enabling comprehensive strain identification capabilities.  

### Strain Identification Process  

The strain identification process involves several key steps:  

1. **Input Processing**: The system accepts target sequences in FASTA format, either as pasted text or file uploads. A graphical user interface provides options to specify the viral genus (if known) and select reference sequences for alignment.  

2. **Sequence Alignment**: Target sequences are aligned with appropriate reference sequences from the database. Location numbers from the reference sequence are assigned to corresponding aligned locations in the target sequence.  

3. **Partition-wise Residue Comparison**: Beginning with partition P02 (the first partition after the root), the system compares target sequence residues with characteristic residues of each group in the partition. The target sequence is assigned to the group showing the maximum number of matches.  

4. **Tree-traversal Optimization**: In subsequent partitions, the system only examines groups that are directly tree-linked to the most recently accepted group, significantly reducing search time. This constrained traversal maintains computational efficiency while ensuring accurate strain identification.  

5. **Residue Flagging**: Matched residues are flagged and excluded from subsequent partition comparisons, except in cases of ambiguity where they may be reconsidered in later partitions.  

6. **Ambiguity Resolution**: When multiple groups show equal numbers of matches, or when no matches occur in a partition, the system proceeds to subsequent partitions without flagging residues, allowing potential resolution of ambiguities through additional comparisons.  

7. **Strain Determination**: The process continues through all partitions until the target sequence's closest matching strain is identified based on cumulative residue matches across partitions.  

For target sequences of unknown genus, the system first compares the sequence with representative reference sequences from each genus tree to determine the most likely genus before proceeding with detailed strain identification.  

### Recombination and Mutation Detection  

The system provides optional capabilities for detecting potential recombination events and spontaneous mutations through partition-wise residue comparisons. Recombination is indicated by abrupt changes in phylogenetic sequence groupings among trees constructed from different genomic regions. Spontaneous mutations are suggested when characteristic residue changes occur without corresponding changes in tree topology.  

### Implementation Details  

The system is implemented in Perl programming language with a Java-based graphical user interface, supporting both Windows and Linux operating environments. The software architecture includes modules for sequence input processing, database management, phylogenetic analysis, strain identification, and results presentation.  

## CONCLUSIONS  

The present invention provides a novel, efficient, and accurate system for automated strain identification of ssRNA viruses, particularly caliciviruses and picornaviruses. Key advantages over conventional methods include:  

1. Robust strain identification from both complete and partial capsid protein sequences  
2. Computational efficiency with average processing times of approximately 5 seconds per sequence  
3. Reduced sensitivity to parameter selection compared to sliding window approaches  
4. Capability to handle sequence variations and potential recombination events  
5. Comprehensive coverage of diverse viral genera through partitioned phylogenetic databases  

The system has been successfully validated using over 300 complete and partial capsid sequences from various caliciviruses and picornaviruses, demonstrating consistent and reliable strain identification capabilities. The method represents a significant advancement in computational virology and has important applications in viral outbreak monitoring, vaccine development, and epidemiological research.  

## AVAILABILITY AND REQUIREMENTS  

The RECOVIR software system implementing the present invention has the following specifications:  

- **Project Home Page**: [To be determined by patent applicant]  
- **Operating Systems**: Windows XP and Linux platforms  
- **Programming Languages**: Perl and Java implementations  
- **Other Requirements**: X-Windows support (such as Cygwin) required for remote Linux operation  
- **License**: [To be determined by patent applicant]  

## AUTHOR CONTRIBUTIONS  

The inventors of the present patent application have made the following contributions to the development of the RECOVIR system:  

- **DZ**: Primary software developer responsible for coding the core functionality and designing the graphical user interface  
- **SC**: Concept originator who developed the foundational algorithm, designed the implementation strategy, and created the initial viral sequence databases  
- **SC and DZ**: Jointly performed extensive software testing and troubleshooting using both synthetic and real viral sequence data  
- **SC and GEF**: Primary authors of the technical documentation and patent application  
- **All inventors**: Contributed to manuscript preparation and approved the final version of the patent application  

The inventors declare no competing interests related to this invention.