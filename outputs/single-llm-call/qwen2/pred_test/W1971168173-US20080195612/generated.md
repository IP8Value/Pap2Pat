# DESCRIPTION

## BACKGROUND OF THE INVENTION

The present invention relates to a method and software for identifying strains of highly divergent single-stranded RNA (ssRNA) viruses, particularly caliciviruses and picornaviruses, using capsid sequences. Most non-bacterial epidemic outbreaks are caused by ssRNA viruses, which undergo rapid genetic mutations leading to a large and dynamic population diversity. These viruses are typically classified into genera, species, and serotypes based on their genetic and antigenic relationships. However, traditional methods of strain identification, which rely on sequence similarity scores, are limited by computational bottlenecks and the lack of uniform criteria for estimating sequence similarity cut-offs. The present invention addresses these limitations by implementing a residue-wise comparison approach that uses partitioned phylogenetic trees of capsid sequences to automate strain predictions.

## SUMMARY OF THE INVENTION

The invention provides a method and software for identifying strains of caliciviruses and picornaviruses using capsid sequences. The method involves constructing partitioned phylogenetic trees for each virus genus, creating databases of characteristic residues that distinguish among the reference sequences, and comparing input target sequences with these databases to identify the closest strains. The software, named RECOVIR, efficiently and accurately characterizes strains of these highly divergent viruses, even for partial capsid sequences. The method is general enough to be applicable to both amino acid and nucleotide sequences of calicivirus and picornavirus capsids, providing a powerful alternative to current strain determination and classification techniques.

## DETAILED DESCRIPTION OF THE INVENTION

### Capsid Embodiment of the Invention

The invention is embodied in a software tool, RECOVIR, which automates the process of strain identification for caliciviruses and picornaviruses using capsid sequences. The method involves several key steps:

1. **Phylogenetic Tree Construction**: Partitioned phylogenetic trees are constructed for each of the 4 calicivirus genera (noroviruses, sapoviruses, lagoviruses, and vesiviruses) and the 9 picornavirus genera (apthoviruses, cardioviruses, enteroviruses, erboviruses, hepatoviruses, kobuviruses, parechoviruses, rhinoviruses, and teschoviruses). These trees are based on evolutionary trace approaches and are constructed using aligned sequences obtained from public databases. The Fitch-Margoliash distance matrix and the neighbor-joining Kitsch algorithm are used to generate the trees, ensuring that the topology of the trees is robust and consistent.

2. **Partitioning of Phylogenetic Trees**: The trees are divided into ten equally spaced partitions, each acting as a "similarity filter" to create different sequence groups. Each partition contains similar sequences emanating from a given node within the partition. The sequences used to construct the trees are called "reference" sequences, and the corresponding trees are called "reference" or "genus" trees.

3. **Database Creation**: For each tree, sequences belonging to different groups of a given partition are aligned, and consensus residues are identified as "characteristic residues" that are conserved within each group but not among different groups. These characteristic residues are stored in multiple 2-dimensional arrays that form the calicivirus and picornavirus databases.

4. **Strain Identification**: To identify the strain of an input target sequence, the program aligns the target sequence with a database reference sequence and compares the target residues with the characteristic residues of each group in the appropriate genus database. The program starts with the second partition from the root and proceeds through subsequent partitions, testing only those groups that are directly tree-linked with the most recently accepted group. This ensures an optimal tree search time, making the program computationally efficient.

5. **Handling Ambiguities**: If all groups in a partition show an identical number of characteristic residue matches, an ambiguity is declared, and no match is flagged. This ensures that all such matched residues of the input sequence are available for matching purposes in subsequent partitions. Ambiguities may also occur when all groups within a given partition show no matches with the input sequence, or if two successive partitions show identical numbers of characteristic residue matches. In these cases, the program ignores the ambiguous partition(s) and proceeds to the next one without marking any residue.

6. **Genus Determination**: When the genus of the target sequence is not known, the sequence is first compared with groups of representative reference sequences from each of the genus trees in the database using ClustalW. Alignment scores are computed for each group, and the highest alignment score is used to select the most appropriate genus tree from the databases. Detailed strain identification is then conducted as described earlier.

7. **Recombination and Spontaneous Mutation Detection**: Partition-wise comparisons allow RECOVIR to detect abrupt changes in phylogenetic sequence groupings among trees constructed using sequences from different genomic parts. These abrupt changes or incongruities indicate nodes that may contain recombination sites. Changes in some of the strain-distinguishing residues of a given region without changes in the tree topology indicate possible spontaneous mutations. Detection of possible recombination or spontaneous mutations is currently done manually, but an automated version of this feature is planned for future development.

## CONCLUSIONS

The invention provides a robust and efficient method and software for identifying strains of caliciviruses and picornaviruses using capsid sequences. The method, implemented in the software RECOVIR, overcomes the limitations of traditional homology-based strain identification techniques by using partitioned phylogenetic trees and characteristic residue databases. The software can accurately and rapidly identify strains of both complete and partial capsid sequences, making it a valuable tool for viral diagnostics and epidemiology.

## Availability and Requirements

- **Project Home Pages**: The software RECOVIR is available for download from the project home pages.
- **Operating Systems**: The software is compatible with Windows-XP and Linux operating systems.
- **Programming Language**: The software is written in Perl and Java.
- **Other Requirements**: X-Windows support (such as Cygwin) is needed for remotely running the program under a Linux environment.

## Author Contributions

- **DZ**: Coded the software and designed the graphical user interface (GUI).
- **SC**: Developed the concept, designed the algorithm and its implementation, and created the initial databases.
- **SC and DZ**: Performed extensive troubleshooting with both synthetic and real data.
- **SC and GEF**: Wrote the manuscript.
- **All Authors**: Suggested improvements at different stages of manuscript preparation and read and approved the final version of the manuscript.