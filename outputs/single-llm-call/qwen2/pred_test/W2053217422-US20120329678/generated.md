# DESCRIPTION

## BACKGROUND OF THE INVENTION

The field of genomics has seen rapid advancements with the advent of next-generation sequencing (NGS) technologies. These technologies generate vast amounts of short read data, which are essential for various applications such as de novo genome assembly, transcriptome analysis, and the identification of structural variations. However, the short read length of NGS platforms poses significant challenges, particularly in assembling complex genomes with repetitive sequences. Mate pair reads, which provide long-range information, are crucial for overcoming these challenges. They help in joining contigs flanking repetitive sequences and are vital for the discovery of structural variations such as insertions, deletions, and inversions.

Several methods for constructing genomic DNA (gDNA) mate pair libraries have been developed, each with its own advantages and limitations. Sanger paired-end sequencing, while generating long reads of high quality, is costly, labor-intensive, and time-consuming. The genomic DNA di-tag method, derived from Serial Analysis of Gene Expression (SAGE), produces short reads that may map to multiple locations in complex genomes. The Illumina 40 kb jumping library, made by cloning 40 kb gDNA in a modified fosmid vector, has limited library complexity and commercially unavailable vectors. Commercial kits for making mate pair libraries on NGS platforms, such as the Illumina Mate Pair Library Prep Kit and the Roche 454 Jump Recombi Paired-End Library Preparation Kit, have their own constraints, including limited insert size and issues with chimeric reads.

This invention introduces a novel in vitro method that utilizes the Cre-LoxP recombination system and inverse PCR to create long insert mate-pair libraries. This method addresses the limitations of existing techniques by providing a robust, high-quality, and versatile approach for generating mate pair libraries suitable for various NGS platforms.

## BRIEF SUMMARY OF THE INVENTION

The present invention relates to a method for generating long insert mate-pair libraries using the Cre-LoxP recombination system and inverse PCR. The method involves the following steps: (1) shearing genomic DNA to a desired size, (2) repairing the ends of the DNA fragments, (3) ligating adapters containing LoxP and Illumina P1 or P2 PCR priming sequences to the ends of the DNA fragments, (4) circularizing the DNA fragments using Cre recombinase, (5) enzymatically fragmenting the circularized DNA, (6) self-ligating the fragmented DNA, (7) selectively amplifying the DNA fragments containing the P1-LoxP-P2 sequences using PCR, and (8) preparing the amplified products for sequencing. The resulting mate-pair libraries are fully compatible with Illumina's sequencing platform and can be used for de novo genome assembly, structural variation detection, and other genomic analyses.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS

### Definitions

- **Mate Pair Library**: A library of DNA fragments where the ends of each fragment are sequenced to provide long-range information.
- **Cre-LoxP System**: A site-specific recombination system used to mediate the circularization of DNA fragments.
- **Inverse PCR**: A technique used to amplify specific regions of circular DNA.
- **Non-Redundant Pairs**: Read pairs that have unambiguous mapping coordinates and are only counted once if they were duplicated.
- **Chimeric Pairs**: Read pairs that map to different chromosomes or in the wrong orientation.

### I. Introduction

The ability to generate high-quality mate pair libraries is essential for advancing genomic research. Existing methods, while useful, have limitations that hinder their effectiveness in certain applications. The present invention provides a novel method that overcomes these limitations by utilizing the Cre-LoxP recombination system and inverse PCR to create long insert mate-pair libraries. This method ensures high ligation efficiency, reduces the occurrence of chimeric reads, and is compatible with various NGS platforms, particularly Illumina's sequencing technology.

### II. Generation of Mate-Pair Libraries

The method for generating mate-pair libraries involves several key steps:

1. **Shearing Genomic DNA**: Genomic DNA is sheared to a desired size using a hydrodynamic shearing device. The size of the DNA fragments can range from 5 kb to 22 kb, depending on the application.

2. **End Repair**: The ends of the DNA fragments are repaired using a mixture of T4 DNA polymerase, Klenow enzyme, and T4 Polynucleotide Kinase. This step ensures that the DNA fragments have blunt ends, which are necessary for subsequent ligation reactions.

3. **Adapter Ligation**: Adapters containing LoxP and Illumina P1 or P2 PCR priming sequences are ligated to the ends of the DNA fragments. These adapters facilitate the circularization of the DNA fragments and provide priming sites for PCR amplification.

4. **Circularization**: The DNA fragments are circularized using Cre recombinase, which mediates the intra-molecular recombination of the LoxP sites. This step brings the ends of the DNA fragments together, forming circular molecules.

5. **Enzymatic Fragmentation**: The circularized DNA is enzymatically fragmented using a restriction enzyme. This step generates smaller DNA fragments that are suitable for PCR amplification.

6. **Self-Ligation**: The fragmented DNA is self-ligated to form linear DNA molecules. This step ensures that only the desired DNA fragments are amplified in the subsequent PCR step.

7. **PCR Amplification**: The DNA fragments containing the P1-LoxP-P2 sequences are selectively amplified using PCR with Illumina P1 and P2 primers. This step enriches the library for the desired mate-pair reads.

8. **Library Preparation**: The amplified products are purified and prepared for sequencing on an Illumina platform.

### III. Software and Data Analysis

The quality and utility of the mate-pair libraries generated by this method are assessed using various software tools and data analysis techniques:

1. **Read Trimming**: To reduce the probability of a read crossing the junction point where the two distant ends of the original DNA fragment were joined during circularization, reads are trimmed to a specific length. For CLIP-PE libraries, bases after the restriction enzyme recognition site are trimmed, resulting in an average read length of 70 bp.

2. **Alignment**: The trimmed reads are aligned to the reference genome using the BWA aligner. This step helps in identifying the mapping coordinates of the read pairs and assessing the quality of the library.

3. **Quality Metrics**: Several metrics are used to evaluate the quality of the mate-pair libraries, including the percentage of non-redundant pairs, the percentage of chimeric pairs, and the clone coverage of the genome. These metrics provide insights into the performance and reliability of the libraries.

4. **Assembly**: The mate-pair libraries are combined with short insert reads to improve genome assembly. The N50 scaffold size, the number of scaffolds and contigs, and the number of mis-assemblies are used to assess the quality of the assemblies.

### IV. Kits

The invention also encompasses kits for generating mate-pair libraries using the described method. These kits include:

- **Reagents**: Enzymes (T4 DNA polymerase, Klenow enzyme, T4 Polynucleotide Kinase, Cre recombinase, restriction enzymes, T4 ligase, Bst DNA polymerase, and Plasmid-Safe™ ATP-Dependent DNase), adapters, primers, and buffers.
- **Consumables**: Hydrodynamic shearing devices, gel electrophoresis equipment, and PCR amplification kits.
- **Protocols**: Detailed instructions for each step of the library preparation process, including shearing, end repair, adapter ligation, circularization, enzymatic fragmentation, self-ligation, PCR amplification, and library preparation for sequencing.

## EXAMPLES

### CLIP-PE Method can Consistently Generate High Quality Mate Pair Libraries

To demonstrate the effectiveness of the CLIP-PE method, we created 5 kb, 12 kb, and 22 kb mate-pair libraries from *Haloterrigena turkmenica* and *Saccharomyces cerevisiae* genomic DNA. The libraries were sequenced using Illumina's Genome Analyzer IIx, and the quality of the libraries was assessed using various metrics.

For the 5 kb *H. turkmenica* library, the CLIP-PE method yielded 20.6% non-redundant pairs with the expected insert size, compared to 8.7% from the Illumina jumping library. The average clone coverage of the CLIP-PE library was 4,746×, while the Illumina jumping library had an average clone coverage of 18×. The CLIP-PE library also had fewer uncovered gaps (7 gaps) and a lower chimeric rate (2.3%) compared to the Illumina jumping library (767 gaps and 9.2% chimeric rate).

For the 12 kb *S. cerevisiae* libraries, the CLIP-PE method consistently produced high-quality libraries with an average of 59% non-redundant pairs and low chimeric rates (5–7%). The 22 kb *S. cerevisiae* libraries also showed high quality, with 11.1% non-redundant pairs and low chimeric rates (1.5–1.7%).

### Ligation Efficiency Affects the Productivity and Quality of CLIP-PE Libraries

To investigate the impact of ligation efficiency on the quality of CLIP-PE libraries, we compared the use of restriction digestion and random shearing for the secondary fragmentation step. Three 22 kb *S. cerevisiae* libraries were created using NlaIII (4 bp overhang), HpyCH4IV (2 bp overhang), and random shearing (blunt end). The NlaIII library had the highest proportion of non-redundant pairs (11.1%), followed by the HpyCH4IV library (4.0%) and the randomly sheared library (2.5%). This result is consistent with the higher ligation efficiency of 4 bp overhangs compared to 2 bp overhangs and blunt ends.

### Discussion

The CLIP-PE method offers several advantages over existing methods for generating mate-pair libraries. By utilizing the Cre-LoxP recombination system and inverse PCR, the method ensures high ligation efficiency, reduces the occurrence of chimeric reads, and is compatible with various NGS platforms. The introduction of a recognizable junction site between read pairs helps in avoiding chimeric reads and facilitates downstream data analysis and assembly.

The CLIP-PE method has been successfully used to generate high-quality mate-pair libraries with insert sizes ranging from 5 kb to 22 kb. These libraries have been shown to improve genome assembly and structural variation detection, particularly in complex genomes with repetitive sequences. The method is versatile and can be adapted for use with other NGS platforms, making it a valuable tool for genomic research.

## Methods

### Illumina Library Preparation

Illumina standard shotgun libraries were created using the commercial Illumina Pair-end kit with 1 μg of genomic DNA without PCR amplification. Illumina jumping libraries were created using the commercial Illumina's Mate-pair library preparation kit V2 with 5 μg of genomic DNA.

### CLIP-PE Library Preparation

1. **Shearing Genomic DNA**: 5, 15, or 30 μg of genomic DNA in 150 μl of EB buffer was sheared to a desired size (5 kb, 12 kb, or 22 kb) using a hydrodynamic shearing device.
2. **End Repair**: 5 μl each of T4 DNA polymerase, Klenow enzyme, and T4 Polynucleotide Kinase, along with dNTP and BSA, were used to repair the ends of the DNA fragments in a 200 μl volume of 1× TNK buffer for 20 minutes at 25°C.
3. **Adapter Ligation**: 2.5 μl of 20 μM each LoxP-P1 and LoxP-P2 integrated adapters were ligated to the ends of the DNA fragments using Quick ligase for 15 minutes at 25°C.
4. **Size Selection**: Adapter-ligated DNA was size-selected through gel electrophoresis for 5 and 12 kb libraries or pulse-field gel electrophoresis for 22 kb libraries.
5. **Fill-In Reaction**: DNA was filled-in with Bst DNA polymerase and dNTP for 15 minutes at 50°C.
6. **Circularization**: 1–4 LoxP-Cre reactions were set up for 45 minutes at 37°C, followed by linear DNA digestion using Plasmid-Safe™ ATP-Dependent DNase.
7. **Enzymatic Fragmentation**: Circularized DNA was digested using NlaIII for 1–2 hours at 37°C.
8. **Self-Ligation**: ATP, T4 ligase buffer, and T4 ligase were added to self-ligate the fragmented DNA.
9. **PCR Amplification**: Inverse PCR with Illumina pair-end library primers and Phusion DNA Polymerase was used to amplify the molecules containing the mate-pair ends.
10. **Purification**: The PCR products were purified using gel electrophoresis and a Wizard SV column.

### Illumina Sequencing

Sequencing was performed on an Illumina Genome Analyzer IIx according to the manufacturer's recommended protocols. For standard Illumina PE libraries, a sequencing run was 2×100 cycles, and the data was trimmed to 76 bp for assembly analysis. All other sequencing runs were performed at 2×76 cycles.

### Post-Sequencing Analysis

Reads were trimmed to 35 bp for Illumina jumping libraries and 70 bp for CLIP-PE libraries. The trimmed reads were aligned to the reference genome using the BWA aligner. Assembly accuracy was evaluated using dnadiff to compare the assembly to the reference.

### Data Simulations and Genome Assembly

Simulated reads were generated from the reference using wgsim with a read length of 76 bp and an error rate of 1%. Datasets were assembled using Velvet, and various parameters were tested to optimize the assembly. Assembly accuracy was evaluated using dnadiff to compare the assembly to the reference.