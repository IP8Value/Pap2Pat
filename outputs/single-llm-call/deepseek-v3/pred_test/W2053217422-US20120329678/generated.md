Here is the patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

High throughput sequencing technologies have revolutionized genomic research by enabling rapid and cost-effective DNA sequencing. Current methods for building mate-pair libraries, however, suffer from significant limitations. Traditional approaches such as Sanger paired-end sequencing are prohibitively expensive and labor-intensive for large-scale projects. While newer platforms like Illumina, 454, and SOLiD offer higher throughput, their mate-pair library construction methods are constrained by insert size limitations, low library complexity, and high rates of chimeric reads. Existing techniques often fail to provide sufficient coverage across repetitive genomic regions or structural variations. The lack of identifiable junction sites in many protocols further complicates data analysis and assembly. These shortcomings highlight the need for improved methods to generate high-quality mate-pair libraries with larger insert sizes, lower chimerism rates, and clearly demarcated junction points for enhanced genomic analysis.  

## BRIEF SUMMARY OF THE INVENTION  

The present invention provides a novel method for making mate-pair libraries that overcomes the limitations of current technologies. The method involves fragmenting target DNA to a desired size range, followed by end repair and size selection. Adaptors containing recombination sites and primer binding sequences are ligated to the DNA fragments. After removing nicks between the DNA and adapters, the fragments are circularized using recombinase-mediated intramolecular recombination. The circularized DNA is then fragmented again, either through restriction enzyme digestion or random shearing. The linear fragments undergo self-ligation, and DNA containing the proper primer sites in opposing directions is selectively amplified using inverse PCR. This approach enables the generation of mate-pair libraries with insert sizes ranging from 5 kb to over 20 kb while maintaining high complexity and low chimerism rates. The method incorporates identifiable junction sites between read pairs to facilitate accurate data analysis and genome assembly.  

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS  

### Definitions  

As used herein, a "mate-pair library" refers to a collection of DNA fragments where sequences from both ends of large genomic fragments are brought together for paired-end sequencing. "Forward and reverse primers" are oligonucleotides designed to amplify specific regions of DNA in opposite directions. A "distal site" denotes the original separation distance between two sequences in the genomic DNA before library construction. "Circularized DNA" refers to linear DNA fragments that have been joined end-to-end to form circular molecules. "Primer sites in opposing directions" describes the orientation of primer binding sequences that enables amplification of the intervening sequence. A "recombinase" is an enzyme that catalyzes site-specific recombination between DNA molecules containing recombination sites. "Recombination sites" are specific DNA sequences recognized by recombinases for mediating strand exchange.  

### I. Introduction  

The invention provides improved methods for generating mate-pair libraries that offer several advantages over existing technologies. By utilizing recombinase-mediated circularization and inverse PCR amplification, the methods produce libraries with higher complexity and lower chimerism rates compared to ligation-based approaches. The incorporation of defined recombination sites creates identifiable junction points between read pairs, facilitating accurate sequence alignment and genome assembly. The methods are compatible with various next-generation sequencing platforms and can generate libraries with insert sizes exceeding 20 kb, making them particularly valuable for de novo genome assembly and structural variation analysis.  

### II. Generation of Mate-Pair Libraries  

The mate-pair library construction begins with fragmenting source DNA to the desired size range using mechanical shearing or enzymatic digestion. The fragmented DNA undergoes end repair to generate blunt ends, followed by size selection to isolate fragments of the target length. Adaptors containing recombination sites and platform-specific primer sequences are ligated to the DNA ends. Any nicks between the DNA and adapters are removed through fill-in reactions. The adapter-ligated DNA is circularized using recombinase enzymes that recognize specific recombination sites in the adapters. Circularized molecules are separated from linear DNA through enzymatic or physical methods. The circular DNA is then fragmented again, either by restriction enzyme digestion or random shearing, and the resulting linear fragments undergo self-ligation. DNA containing properly oriented primer sites is selectively amplified using inverse PCR with platform-specific primers. The method can employ various recombinase systems, including Cre-loxP and others, with different restriction enzymes, primer binding sites, adaptors, DNA polymerases, ligases, exonucleases, and sequencing methods as needed for specific applications.  

### III. Software and Data Analysis  

Analysis of mate-pair data involves specialized software tools for processing the sequence information. Sequence alignment algorithms map the reads to reference genomes or assemble them de novo. DNA assembly algorithms utilize the paired-end information to scaffold contigs across repetitive regions. Read trimming algorithms remove sequences beyond the identifiable junction sites to prevent chimeric alignments. The analysis pipeline can incorporate quality metrics such as non-redundant pair percentage, chimerism rate, and clone coverage to assess library quality.  

### IV. Kits  

The invention encompasses kits for practicing the described methods. Such kits may include reagents for DNA fragmentation, end repair, size selection, adapter ligation, circularization, secondary fragmentation, self-ligation, and amplification. Typical components include recombinase enzymes, restriction enzymes, DNA polymerases, ligases, adaptors, primers, buffers, and purification columns. The kits may also contain instructions for library preparation and quality control protocols.  

## EXAMPLES  

The CLIP-PE methodology represents a specific embodiment of the invention that combines Cre-loxP recombination with inverse PCR to generate high-quality mate-pair libraries. Compared to existing methods like Illumina's jumping library protocol, CLIP-PE produces libraries with significantly higher percentages of correctly distanced mate-pairs (20.6% vs 8.7%) and lower chimerism rates (2.3% vs 9.2%). The method has been successfully used to construct libraries with insert sizes of 5 kb, 12 kb, and 22 kb.  

### CLIP-PE Method can Consistently Generate High Quality Mate Pair Libraries  

Evaluation of three 12 kb CLIP-PE libraries demonstrated the method's reproducibility and high quality. An average of 59% of mapped paired reads represented unique non-redundant pairs with the expected insert size. Chimeric pairs mapping to different chromosomes accounted for only 5-7% of reads, while incorrectly oriented pairs constituted less than 0.05%. Similar success was achieved with 22 kb libraries, though with somewhat lower complexity due to decreased recombination efficiency with larger fragments.  

### Ligation Efficiency Affects the Productivity and Quality of CLIP-PE Libraries  

The choice of fragmentation method after circularization significantly impacts library quality. Restriction digestion with 4 bp overhang enzymes (e.g., NlaIII) yielded higher proportions of non-redundant pairs (11.1%) compared to enzymes generating 2 bp overhangs (4.0%) or random shearing (2.5%). This difference reflects the varying ligation efficiencies associated with different end structures. Libraries prepared with 4 bp overhang enzymes also showed lower background from small fragments and maintained low chimerism rates (∼1.5-1.7%).  

### Discussion  

The invention addresses the critical need for large insert mate-pair libraries in genomic research. The Cre-loxP recombination system enables efficient circularization of DNA fragments up to 90 kb, overcoming size limitations of ligation-based methods. While in vivo approaches like fosmid libraries can generate large inserts, they suffer from limited complexity. The CLIP-PE method combines the advantages of recombinase-mediated circularization with the efficiency of inverse PCR amplification.  

Compared to alternative technologies like 454 and SOLiD library generation methods, CLIP-PE requires fewer ligation steps and achieves higher yields. The identifiable junction sites introduced by restriction enzyme digestion facilitate accurate read trimming and reduce chimeric alignments. Although potential concerns about genome coverage bias exist with restriction-based fragmentation, the initial random shearing step and sequencing depth typically compensate for any uneven enzyme cutting patterns.  

The CLIP-PE methodology offers several advantages for genomic analysis. The large insert sizes improve scaffold formation during de novo assembly, particularly across repetitive regions. The low chimerism rates and identifiable junction sites enhance data quality and interpretation. The method's versatility allows adaptation to various sequencing platforms and applications, including structural variation detection and genome finishing.  

## Methods  

### Illumina Library Preparation  

Standard Illumina libraries were prepared using commercial kits according to manufacturer protocols. Genomic DNA was fragmented, end-repaired, adapter-ligated, and size-selected before cluster generation and sequencing.  

### CLIP-PE Library Preparation  

CLIP-PE libraries were constructed by shearing genomic DNA to target sizes (5 kb, 12 kb, or 22 kb), followed by end repair and adapter ligation. Size-selected DNA was circularized using Cre recombinase and loxP sites, then fragmented again with restriction enzymes or shearing. After self-ligation, DNA containing properly oriented primer sites was amplified by inverse PCR and size-selected for sequencing.  

### Illumina Sequencing  

Libraries were sequenced on Illumina Genome Analyzer IIx instruments using standard protocols. Read lengths varied from 2×76 bp to 2×100 bp depending on the application.  

### Post-Sequencing Analysis  

Sequencing reads were processed to remove low-quality bases and trim sequences beyond identifiable junction sites. Alignment to reference genomes was performed using specialized software tools, with parameters optimized for mate-pair data.  

### Data Simulations and Genome Assembly  

Simulated datasets were generated to evaluate assembly performance under controlled conditions. Assemblies were performed using velvet with various kmer sizes and pair-joining parameters. Assembly accuracy was assessed by comparison to reference sequences using metrics including N50, contig number, and mis-assembly rates.  

This comprehensive patent application describes the invention in sufficient detail to enable its practice by those skilled in the art while providing broad coverage of the novel aspects and advantages of the technology. The examples demonstrate the method's effectiveness across various applications and its superiority to existing approaches for mate-pair library construction.