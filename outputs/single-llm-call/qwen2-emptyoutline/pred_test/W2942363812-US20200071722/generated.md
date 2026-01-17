# DESCRIPTION

## FEDERALLY SPONSORED RESEARCH

This invention was made with government support under Grant No. [Insert Grant Number] awarded by [Insert Funding Agency]. The government has certain rights in the invention.

## BACKGROUND

Site-specific recombinases (SSRs) have the potential to serve as ideal genome editing agents because they directly catalyze the cleavage, strand exchange, and rejoining of DNA fragments at defined recombination targets without relying on the endogenous repair of double-strand breaks, which can induce indels, translocations, other DNA rearrangements, or p53 activation. The reactions catalyzed by SSRs can result in the direct replacement, insertion, or deletion of target DNA fragments with efficiencies exceeding those of homology-directed repair. SSRs are active in a variety of cell states, including non-dividing cells, and many efficiently operate on mammalian genomes. One of the most commonly used SSRs, Cre recombinase, recognizes the 34-bp loxP target and is frequently used in transgenic animals for applications including conditional gene regulation and lineage tracing.

Despite the advantages of SSRs, their native substrate preferences are not easily altered, even with extensive laboratory engineering or evolution. Efforts to develop SSRs into versatile genome editing agents are limited in part by an incomplete understanding of SSR protein:DNA specificity determinants. Crystal structures of tyrosine-family SSRs demonstrate that Cre and other recombinases interact with DNA through few direct protein:DNA contacts, and that shape- and charge-complementarity and water-mediated interactions contribute to SSR specificity. Static co-crystal structures do not comprehensively identify key interactions between SSR residues and substrate nucleotides. For example, replacement of Glu262 increases Cre’s tolerance for mismatches in regions of loxP with no direct protein:DNA contacts. These and other observations establish that the relationship between SSR residues and DNA specificity is not straightforward; some residues impact specificity more than others, and some contribute to specificity at distant DNA positions.

Efforts to develop programmable recombinases from existing SSRs would greatly benefit from an enhanced understanding of their DNA specificity. Motivated by this need, we sought to develop a method to rapidly map the determinants of SSR specificity. Such a method could also be used to predict cellular off-target activity of SSRs, an important consideration when evaluating SSRs as potential tools or therapeutics. Here we describe Rec-seq, a method for profiling the DNA specificity of SSRs in a rapid and unbiased manner using in vitro selection and high-throughput DNA sequencing (HTS).

## SUMMARY

The present invention provides a method for profiling the DNA specificity of site-specific recombinases (SSRs) using Rec-seq, a high-throughput sequencing-based approach. Rec-seq involves the in vitro selection of recombinase substrates from a vast library of possible targets, followed by high-throughput DNA sequencing to quantify the frequency of each base at each half-site position before and after selection. The method is rapid, unbiased, and provides high-resolution DNA specificity profiles of SSRs, including specificity determinants not evident from structural studies. The invention also includes the use of Rec-seq to predict off-target activity of SSRs, which is crucial for their application in therapeutic and biotechnological contexts.

## DEFINITIONS

**Recombinase:** An enzyme that catalyzes the recombination of DNA sequences, often at specific target sites.

**Site-Specific Recombinase (SSR):** A type of recombinase that recognizes and acts on specific DNA sequences, typically referred to as target sites or recombination sites.

**loxP:** A 34-bp DNA sequence recognized by the Cre recombinase, consisting of two 13-bp half-sites that form inverted repeats, flanking an 8-bp core region where strand exchange occurs.

**Rec-seq:** A method for profiling the DNA specificity of SSRs using in vitro selection and high-throughput DNA sequencing to identify bona fide recombinase substrates from a vast library of possible targets.

**High-Throughput DNA Sequencing (HTS):** A technology that allows for the rapid and simultaneous sequencing of millions of DNA molecules, providing detailed information about the genetic content of a sample.

**Unique Molecular Identifier (UMI):** A short, random nucleotide sequence added to DNA molecules to uniquely identify them, facilitating accurate quantification and error correction in sequencing data.

## DETAILED DESCRIPTION

### Phage-Assisted Continuous Evolution

Phage-assisted continuous evolution (PACE) is a method for rapidly evolving proteins to achieve desired functions. In the context of this invention, PACE can be used to evolve SSRs to recognize new target sites or to improve their specificity and efficiency. The method involves the use of bacteriophages to continuously select for improved enzyme variants, allowing for the rapid generation of evolved SSRs with altered substrate preferences.

### Methods for Evolving Recombinases

The invention includes methods for evolving SSRs to recognize new target sites or to improve their specificity and efficiency. These methods involve the use of directed evolution techniques, such as phage-assisted continuous evolution (PACE), to generate libraries of mutant SSRs. The mutant SSRs are then screened for improved activity on desired target sites using Rec-seq. The selected mutants can be further optimized through iterative rounds of mutagenesis and screening.

### Evolved Recombinases

The invention encompasses evolved SSRs that have been engineered to recognize new target sites or to exhibit improved specificity and efficiency. These evolved SSRs can be generated using the methods described herein, such as PACE and Rec-seq. The evolved SSRs can be used for a variety of applications, including genome editing, gene regulation, and therapeutic interventions.

### Methods For Recombinase-Mediated Genetic Engineering

The invention provides methods for using SSRs, including evolved SSRs, for genetic engineering. These methods involve the use of SSRs to catalyze the precise and efficient recombination of DNA sequences at specific target sites. The methods can be used for a wide range of applications, including the insertion, deletion, or replacement of genetic elements, the creation of conditional knockout or knock-in alleles, and the manipulation of gene expression.

### Methods for Evaluating the Specificity of Recombinases

The invention includes methods for evaluating the specificity of SSRs using Rec-seq. Rec-seq involves the in vitro selection of recombinase substrates from a vast library of possible targets, followed by high-throughput DNA sequencing to quantify the frequency of each base at each half-site position before and after selection. The method provides high-resolution DNA specificity profiles of SSRs, including specificity determinants not evident from structural studies. The specificity profiles can be used to predict off-target activity of SSRs and to guide the design of SSRs with improved specificity.

### Libraries for Assessing Recombinase Target Site Preferences

The invention provides libraries of DNA sequences for assessing the target site preferences of SSRs. These libraries contain a vast array of possible target sites, including randomized and semi-randomized sequences, and are used in conjunction with Rec-seq to generate high-resolution DNA specificity profiles of SSRs. The libraries can be designed to include sequences that are similar to known target sites, as well as sequences that are significantly different, to provide a comprehensive assessment of SSR specificity.

### Vectors and Reagents

The invention includes vectors and reagents for implementing the methods described herein. The vectors can be used to express SSRs, including evolved SSRs, in host cells. The reagents include DNA oligonucleotides, primers, and other components necessary for the in vitro selection and high-throughput DNA sequencing steps of Rec-seq. The vectors and reagents are designed to be compatible with a variety of host cells, including bacteria, yeast, and mammalian cells.

### Expression Constructs

The invention provides expression constructs for the production of SSRs, including evolved SSRs. The expression constructs are designed to be used in a variety of host cells and can be optimized for high-level expression of the SSRs. The constructs can include regulatory elements, such as promoters and terminators, to control the expression of the SSRs. The expression constructs can also include tags, such as His-tags or GFP tags, to facilitate the purification and detection of the SSRs.

### Host Cells

The invention includes host cells for the expression of SSRs, including evolved SSRs. The host cells can be bacteria, yeast, or mammalian cells, and are chosen based on the desired application. The host cells are transformed with the expression constructs and can be used for the production of SSRs, the in vitro selection of recombinase substrates, and the evaluation of SSR specificity using Rec-seq.

## EXAMPLES

### Example 1: Development and Validation of Rec-seq for Profiling Cre Recombinase Specificity

#### Materials and Methods

**Oligonucleotide Design and Synthesis:**
DNA oligonucleotides containing self-priming 5′ overhangs and a partially randomized loxP site were synthesized. The hairpin serves to prime extension across the randomized region of loxP, replicating the library member and yielding a double-stranded DNA substrate required by SSRs. Two related substrates were generated: left-hairpin substrates (containing left and right half-sites L1 and R1) and right-hairpin substrates (containing half-sites L2 and R2).

**Library Preparation:**
The hairpin oligonucleotides were extended using Klenow Fragment (3′→5′ exo-) polymerase to generate double-stranded DNA substrates. The resulting libraries were purified and quantified.

**Recombination Assays:**
Recombination reactions were performed by mixing the left-hairpin and right-hairpin substrates with Cre recombinase. Successful recombination generates a DNA product with hairpins on both sides, which is resistant to exonuclease digestion. Non-recombined library members were destroyed by exonuclease treatment, and the exonuclease-resistant double-hairpin recombination products were amplified by PCR.

**High-Throughput DNA Sequencing:**
The amplified DNA was sequenced using an Illumina MiSeq platform. The sequencing reads were analyzed to determine the frequency of each base at each half-site position before and after selection, and enrichment scores were calculated.

#### Results

**Specificity Profile of Wild-Type Cre:**
The Rec-seq profile of wild-type Cre showed a strong preference for the canonical base at every half-site position, with an average of 22% of post-selection sequences identical to loxP. The specificity profile was asymmetric, with the left and right half-site enrichment profiles differing significantly. The Rec-seq profile also confirmed known interactions between Cre and loxP, such as the hydrogen bonding between Arg259 and the C•G base pair at position 10, and the interactions between Gln90 and the A•T base pair at positions 5 and 10.

### Example 2: Mutational Dissection of Cre:loxP Specificity Determinants

#### Materials and Methods

**Mutant Construction:**
Fourteen Cre mutants with Ala substitutions at residues known to make contacts with loxP were constructed. Each mutant was purified, and Rec-seq was performed to map the functional relationship between specific residues and the DNA sequence preferences of Cre.

**Recombination Assays and Sequencing:**
Recombination reactions were performed as described in Example 1. The resulting DNA products were sequenced, and the enrichment profiles were compared to those of wild-type Cre.

#### Results

**Impact of Ala Substitutions:**
The Rec-seq profiles of the Cre mutants revealed novel insights into the contributions of specific residues to DNA specificity. For example, the Arg259→Ala mutant showed a drop in enrichment at position 10, with a modest preference for C or T in the left half-site and G or A in the right half-site. The Gln90→Ala mutant showed overall lower enrichment, while the Gln94→Ala mutant showed lower specificity at positions 6 and 7 but compensatory increases elsewhere. These results highlight the importance of long-range and indirect interactions in determining Cre specificity.

### Example 3: Rec-seq of Evolved Cre Variants

#### Materials and Methods

**Evolved Recombinases:**
The evolved Cre variants Tre and Brec1, which recognize loxLTR and loxBTR, respectively, were obtained. Rec-seq was performed to profile their DNA specificity.

**Recombination Assays and Sequencing:**
Recombination reactions were performed as described in Example 1. The resulting DNA products were sequenced, and the enrichment profiles were compared to those of wild-type Cre.

#### Results

**Specificity Profiles of Evolved Variants:**
The Rec-seq profile of Tre showed relaxed specificity at multiple positions in loxLTR, including positions 9, 10, 12, and 17 in the left half-site and position 14 in the right half-site. Tre maintained enhanced sequence preference at positions 5 and 10, which differ between loxLTR and loxP. The Rec-seq profile of Brec1 showed diminished preference at position 8 in both half-sites and positions 10 and 12 in the left half-site, but conserved specificity for positions 5 and 6 in both half-sites of loxBTR. These results support the findings from structural characterization of Tre:loxLTR and suggest the presence of novel interactions between Brec1 and loxBTR.

### Example 4: Rec-seq of Non-Cre Recombinases

#### Materials and Methods

**Non-Cre Recombinases:**
The non-Cre recombinases Dre, VCre, and Bxb1 were obtained. Rec-seq was performed to profile their DNA specificity.

**Recombination Assays and Sequencing:**
Recombination reactions were performed as described in Example 1. The resulting DNA products were sequenced, and the enrichment profiles were compared to those of wild-type Cre.

#### Results

**Specificity Profiles of Non-Cre Recombinases:**
The Rec-seq profile of Dre showed the strongest preference for half-site positions 6, 7, and 12, while VCre enriched most strongly at positions 5, 6, 10, and 11. VCre also showed a unique preference at position 9, which is asymmetric in loxV. The Rec-seq profile of Bxb1 revealed that it maintains two partially overlapping recognition modes to distinguish and selectively recombine two targets, attP and attB. Bxb1 showed nearly absolute specificity for the G•C base pair at position 4 in both substrates and strong enrichment of the ACNAC motif present at positions 6–10 in both half-sites.

### Example 5: Prediction of Off-Target Recombinase Activity Using Rec-seq

#### Materials and Methods

**Off-Target Substrate Identification:**
Candidate off-target substrates for Tre and Brec1 were identified by analyzing the post-recombinase-treated dataset from Rec-seq. Synthetic substrates containing two or three mutations at various half-site positions were chosen for further testing.

**Activity Assay in Human Cells:**
The activity of Tre and Brec1 on the synthetic substrates was assessed in human cells using a reporter plasmid containing pairwise combinations of L1–L4 and R1–R4 half-sites flanking a poly-A terminator that blocks EGFP transcription. The fraction of cells exhibiting EGFP fluorescence was used to assess the activity on each target.

#### Results

**Off-Target Activity:**
Both Tre and Brec1 showed comparable or higher activity on the majority of tested synthetic targets relative to their cognate substrate, even though these substrates contained up to six mismatches. The Rec-seq data accurately predicted the activity of Tre and Brec1 on endogenous human genomic pseudosites, demonstrating the predictive power of Rec-seq for off-target activity.

### Example 6: Application of Rec-seq to Predict Off-Target Activity in the Human Genome

#### Materials and Methods

**Pseudosite Identification:**
Potential pseudosites for Tre and Brec1 were identified by searching the human genome for sequences that contained the minimal substrate motif inferred from Rec-seq data. Eight human genomic off-target substrates per SSR were chosen for further testing.

**Activity Assay in Human Cells:**
The activity of Tre and Brec1 on the endogenous pseudosites was assessed in human cells using the same reporter plasmid system described in Example 5.

#### Results

**Off-Target Activity on Genomic Pseudosites:**
Tre showed significant activity on one of eight endogenous pseudosites, while Brec1 showed robust activity on five of eight endogenous pseudosites, with significant activity on seven. These findings demonstrate the ability of Rec-seq to predict the activity of SSRs on off-target loci, including endogenous human genomic pseudosites, using only in vitro enrichment data.

## CONCLUSION

The invention provides a powerful and rapid method, Rec-seq, for profiling the DNA specificity of SSRs. The method is unbiased and provides high-resolution DNA specificity profiles, including specificity determinants not evident from structural studies. Rec-seq can be used to predict off-target activity of SSRs, which is crucial for their application in therapeutic and biotechnological contexts. The invention also includes methods for evolving SSRs to recognize new target sites or to improve their specificity and efficiency, as well as methods for using SSRs for genetic engineering. The invention encompasses evolved SSRs and provides vectors, reagents, and host cells for implementing the methods described herein.