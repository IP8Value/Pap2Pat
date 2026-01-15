Here is the complete patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

The present invention relates to the field of computational virology, specifically addressing the challenges in distinguishing among single-stranded RNA (ssRNA) viruses. These viruses represent a significant cause of non-bacterial epidemic outbreaks worldwide and exhibit rapid genetic mutations that result in extensive population diversity. This diversity manifests as different virus strains capable of utilizing multiple hosts, making accurate strain identification critical for diagnostics and treatment.  

Conventional methods for distinguishing among ssRNA viruses rely on immuno-based assays and reverse transcription polymerase chain reaction (RT-PCR) based techniques. However, these approaches suffer from limitations in specificity and sensitivity, particularly when dealing with closely related strains or emerging variants. Homology comparison-based methods have been employed to infer relationships among virus strains by analyzing complete capsid sequences or other genomic regions. These methods aim to identify clusters of similar sequences that comprise major groups such as genogroups or genera, as well as subgroups including species and serotypes.  

Despite their widespread use, existing homology-based methods face significant challenges in accurately distinguishing subtle sequence variations among virus strains. A primary limitation stems from the difficulty in establishing uniform sequence similarity cut-off values that can reliably differentiate between genogroups, clusters, and strains. This challenge is particularly pronounced when analyzing partial sequences from conserved genomic regions or when comparing viruses across different genera or families.  

The computational complexity of comparing large numbers of sequences presents another substantial obstacle. Conventional methods exhibit exponential dependence on sequence lengths and the number of sequences being analyzed, leading to increased processing times and reduced prediction accuracy. Recent approaches that employ sliding window alignments of target sequences against reference databases attempt to address these computational limitations. However, these methods remain sensitive to parameter selection, including window sizes and reference sequence choices, which can introduce error-inducing biases and require repetitive computational runs.  

Within the ssRNA virus families, caliciviruses and picornaviruses represent two of the most highly divergent groups. The calicivirus family comprises four genera (noroviruses, sapoviruses, lagoviruses, and vesiviruses), while the picornavirus family includes nine genera (apthoviruses, cardioviruses, enteroviruses, erboviruses, hepatoviruses, kobuviruses, parechoviruses, rhinoviruses, and teschoviruses). These genera further divide into numerous species and serotypes, reflecting the extensive sequence diversity within these virus families.  

The coat proteins of these viruses demonstrate particular utility for strain characterization studies. Caliciviruses possess a single coat protein subunit, while picornaviruses have four subunits (VP1-VP4). Structural analyses reveal that the exposed regions of picornavirus VP1 subunits contain most neutralization sites, making them immunodominant regions. Similarly, calicivirus coat proteins exhibit the highest variability among genomic regions and show antigenic correlations, despite challenges in culturing human caliciviruses for traditional antigenic characterization.  

Recent advances in computational techniques have enabled strain prediction methods based on capsid sequence analysis. These methods typically employ sequence similarity cut-off values derived from homology-based comparisons between target sequences and known references. While such approaches have shown promise in distinguishing norovirus genogroups and clusters, they lack uniform criteria for accurate cut-off value estimation across different calicivirus genera or when analyzing multiple virus families simultaneously.  

The present invention addresses these limitations through an innovative residue-wise comparison approach that automates strain predictions using both complete and partial amino acid capsid sequences of caliciviruses and picornaviruses. This method builds upon earlier analyses of noroviruses but extends the capability to encompass broader ssRNA virus classification while overcoming the computational bottlenecks inherent in conventional homology-based methods.  

## SUMMARY OF THE INVENTION  

The present invention discloses a novel software product, RECOVIR, designed to accurately characterize strains of highly divergent caliciviruses and picornaviruses. The invention employs a unique approach utilizing characterizing residues that distinguish among reference sequences in phylogeny-based databases. Unlike conventional methods that rely on sequence similarity scores, the present invention implements a residue-wise comparison technique that enables rapid and precise strain identification for both complete and partial viral sequences.  

The software product operates by creating comprehensive databases of capsid residues that uniquely identify reference sequences across phylogenetic branches. Through systematic residue-wise comparisons between input target sequences and these databases, the invention determines the phylogenetic branches whose reference sequences most closely resemble the target sequences. These branches subsequently yield the genogroup and other classification characteristics of the target sequences, thereby identifying their strains with high accuracy.  

Key features of the invention include its ability to handle partial sequences, resolve ambiguities through partition-wise comparisons, and detect potential recombination events and spontaneous mutations. The software demonstrates particular advantages in processing speed, typically requiring approximately five seconds per sequence for strain identification, including input/output operations. This efficiency represents a significant improvement over conventional methods while maintaining robust prediction accuracy.  

The computer program product embodies the invention as a practical tool for virology research and diagnostics. Implemented in Perl programming language with a Java-based graphical user interface, the software supports Windows XP and Linux operating systems. The invention's modular design allows for regular database updates to accommodate emerging virus strains while maintaining statistical significance in residue comparisons.  

## DETAILED DESCRIPTION OF THE INVENTION  

The present invention provides a comprehensive software solution for strain identification of caliciviruses and picornaviruses through an innovative residue-wise comparison approach. The software product, RECOVIR, represents a significant advancement over conventional homology-based methods by employing partitioned phylogenetic trees and characteristic residue databases to achieve rapid and accurate strain predictions.  

The invention's application encompasses analysis of both complete and partial amino acid capsid sequences from calicivirus and picornavirus isolates. The method demonstrates particular utility for noroviruses and enteroviruses, representing two highly diverse groups within these virus families. The software's implementation addresses critical challenges in viral genotyping, including processing speed, prediction accuracy, and ability to handle partial sequence data.  

An embodiment of the present invention involves the construction of partitioned phylogenetic trees for each calicivirus and picornavirus genus. These trees serve as the foundation for creating comprehensive databases of characteristic residues that uniquely distinguish among reference sequences. The tree construction process utilizes publicly available virus sequences aligned using ClustalW with optimized parameters, including the Gonnet-250 distance matrix model and carefully calibrated gap penalties.  

The method's demonstration begins with processing input target sequences through systematic comparisons with the characteristic residue databases. For sequences of known genus, the software directly accesses the appropriate genus database. When the genus is unknown, the program first compares the target sequence with representative references from each genus tree to determine the most likely classification before proceeding with detailed strain identification.  

### Capsid Embodiment of the Invention  

The capsid embodiment of the invention focuses on the structural proteins of caliciviruses and picornaviruses as the basis for strain identification. This embodiment defines capsid residues as those amino acid positions within the viral coat proteins that demonstrate phylogenetic significance across virus strains. The invention creates specialized databases of these capsid residues through meticulous analysis of reference sequences and their phylogenetic relationships.  

Residue-wise comparisons form the core of the strain identification process. The software aligns target sequences with appropriate reference sequences and systematically compares residues at characteristic locations across successive partitions of the phylogenetic trees. This partition-based approach enables efficient tree searching by progressively narrowing the possible strain classifications while maintaining high accuracy.  

The method identifies the closest reference sequences through iterative residue matching across partitions. Beginning with partition two (immediately following the root partition), the software calculates match scores between target sequence residues and characteristic residues of each partition group. The target sequence is assigned to the group showing the maximum number of matches, and the process continues through subsequent partitions along the connected phylogenetic branches.  

This embodiment yields genogroup and classification characteristics by leveraging the evolutionary context embedded in the partitioned trees. Each partition represents a distinct level of phylogenetic resolution, enabling the software to progressively refine strain predictions from broad genogroup classifications to specific strain identifications. The method's accuracy stems from its use of multiple characteristic residues across partitions rather than relying on single sequence similarities.  

The invention constructs partitioned phylogenetic trees through rigorous analysis of sequence relationships. Each tree divides into ten equally spaced partitions (P1-P10) based on maximum evolutionary distances within the tree. These partitions act as similarity filters by creating sequence groups containing similar sequences emanating from specific nodes. The partition spacing optimizes node distribution while maintaining computational efficiency.  

Sequence group comparisons reveal characteristic residues conserved within groups but not among different groups of the same partition. These residues represent key differentiators for strain identification and form the basis of the invention's databases. The software stores comprehensive information about genus trees, including partitions, sequence groups, and characteristic residues, in multidimensional arrays that constitute the calicivirus and picornavirus databases.  

The capsid embodiment illustrates sequence groups through representative examples. A hypothetical phylogenetic tree demonstrates how partitions create distinct sequence groups at various evolutionary levels. The root partition (P1) contains all aligned sequences, while subsequent partitions progressively divide these sequences into increasingly specific groups based on characteristic residue patterns.  

Characteristic residue generation occurs partition-wise for each tree. The invention identifies these residues by comparing aligned sequences from different groups within each partition and determining consensus residues that show conservation within groups but variation between groups. This process creates a comprehensive map of phylogenetically significant residues across all partitions and groups.  

Database creation represents a critical component of the capsid embodiment. The invention stores complete information about genus trees in two-dimensional arrays that form the foundation for strain identification. These databases include partition structures, sequence group compositions, and complete sets of characteristic residues organized by group and partition.  

Strain identification proceeds through systematic partition-wise comparisons. The software matches target sequence residues with characteristic residues of database groups, beginning with partition two and progressing through successive partitions. This approach significantly reduces computational complexity by limiting comparisons to tree-connected groups in subsequent partitions once initial matches are established.  

The invention incorporates sophisticated handling of matching residues and ambiguities. Residues showing matches in a partition are flagged and excluded from subsequent comparisons, except in cases of ambiguity where equal matches occur across groups. This flagging system optimizes processing speed while maintaining accuracy through careful ambiguity resolution protocols.  

An example demonstrates the method's application. A target sequence comparison with characteristic residues in partition two shows maximum matches with a specific group, directing subsequent searches along connected branches. The process continues through partitions, resolving ambiguities when they arise by carrying unmatched residues forward for additional comparisons in later partitions.  

The capsid embodiment includes capabilities for detecting recombination and spontaneous mutations. Partition-wise comparisons can identify abrupt changes in phylogenetic groupings that may indicate recombination events between sequences. Similarly, the method detects spontaneous mutations through residue variations that maintain overall tree topology while showing localized changes.  

Program testing and validation confirm the invention's accuracy. Initial validation using known norovirus and enterovirus strains demonstrated precise strain identification for both complete and partial capsid sequences. Subsequent testing with over 300 calicivirus and picornavirus sequences, including many partial sequences, confirmed the method's robustness across diverse virus groups.  

The software implementation of the capsid embodiment utilizes Perl programming language for core functionality and Java for graphical user interface development. The GUI provides intuitive access to all program features through organized input sections for sequences, databases, and output controls. The software supports batch processing of multiple sequence files in FASTA format without practical limits on file numbers or sizes.  

Input processing begins with sequence submission through pasting or file browsing. The GUI's database section allows specification of known sequence genus or automatic genus determination for unknown sequences. Output options range from summary results to detailed partition-wise match information, providing flexibility for different analysis needs.  

The capsid embodiment demonstrates particular effectiveness in norovirus strain identification. The norovirus genus tree partitions effectively separate known genogroups (GI and GII) and their subgroups through characteristic residue patterns. Detailed analysis of complete norovirus capsid sequences shows unambiguous strain identification through progressive partition comparisons.  

Characteristic residue comparisons enable precise strain identification for complete sequences. For example, analysis of the norovirus "Seacroft" sequence shows partition two matches with GII-specific residues, directing subsequent searches along the GII branch. Progressive partition analyses refine the strain prediction to specific clusters within the GII genogroup.  

The method shows remarkable robustness in handling partial sequences. Testing with norovirus partial sequences from various capsid regions demonstrates accurate strain identification despite limited sequence data. The partition-wise approach resolves ambiguities that might arise from partial sequence coverage by utilizing matches across multiple partitions.  

Enterovirus strain identification follows similar principles. The capsid embodiment successfully identifies strains for complete and partial enterovirus VP1 sequences, including poliovirus, simian enterovirus, echovirus, and Coxsackievirus strains. The method remains accurate regardless of reference sequence choice, demonstrating its robustness.  

Analysis of enterovirus sequences proceeds through systematic partition comparisons. For example, a poliovirus strain identification begins with partition two matches to enterovirus group characteristics. Subsequent partitions progressively narrow the classification to specific poliovirus serotypes through characteristic residue matching.  

The invention's ability to detect recombination and spontaneous mutations provides valuable additional functionality. Partition-wise residue comparisons can reveal topological inconsistencies suggesting recombination events. Similarly, localized residue variations that maintain overall tree topology may indicate spontaneous mutations.  

Processing times represent a significant advantage of the capsid embodiment. Typical runs demonstrate real-time processing (including I/O) of approximately five seconds per sequence. This efficiency enables rapid analysis of outbreak sequences and large datasets, a critical capability for public health applications.  

The software's advantages include database robustness, efficient tree-structured searching, and enhanced classification capabilities. Databases built from complete capsid sequences show remarkable consistency across different tree construction methods, ensuring reliable strain identification results. The partitioned structure enables biologically meaningful residue comparisons that surpass conventional homology-based methods.  

The invention provides particular benefits in classifying tentatively grouped sequences like bovine and alphatron noroviruses. Characteristic residue comparisons and evolutionary distance analyses support classification decisions and provide insights into sequence relationships that may inform future taxonomic revisions.  

While offering significant advantages, the capsid embodiment acknowledges certain limitations. The need for regular database updates to accommodate emerging strains represents an ongoing requirement. However, the invention incorporates criteria for statistically significant changes to ensure database reliability during updates.  

## CONCLUSIONS  

The present invention introduces the RECOVIR software package as a powerful tool for accurate strain characterization of caliciviruses and picornaviruses. This innovative solution overcomes limitations of conventional homology-based methods through implementation of a residue-wise comparison approach that leverages partitioned phylogenetic trees and characteristic residue databases.  

The method's advantages include rapid processing times, robust performance with partial sequences, and ability to detect potential recombination events and spontaneous mutations. The software's efficient tree-structured searching and biologically meaningful residue comparisons provide significant improvements over existing strain identification techniques.  

RECOVIR demonstrates particular effectiveness in analyzing norovirus and enterovirus sequences, with applications extending to all calicivirus and picornavirus genera. The invention's general methodology shows promise for adaptation to nucleotide sequence analysis and potential application to other ssRNA virus families.  

## Availability and Requirements  

The RECOVIR software package requires the following system specifications for optimal operation:  
- Operating systems: Windows XP or Linux platforms  
- Programming language support: Perl and Java runtime environments  
- For Linux operation: X-Windows support (such as Cygwin) for remote program execution  

## Author Contributions  

The invention represents collaborative development with distinct author contributions:  
- DZ implemented the software coding and designed the graphical user interface  
- SC conceived the core concept, developed the algorithm and its implementation, and created the initial databases  
- SC and DZ conducted extensive testing and troubleshooting using both synthetic and real sequence data  
- SC and GEF prepared the patent documentation  
- All authors participated in manuscript refinement and approved the final version