# DESCRIPTION

## FIELD OF INVENTION

The present invention relates to a method for designing, synthesizing, and testing a random, short-hairpin RNA (shRNA)-encoding library. More specifically, the invention provides a method for identifying and optimizing shRNA sequences that protect cells from apoptosis, particularly in the context of interleukin-3 (IL3) withdrawal in a murine pro-B cell line (FL5.12). The invention further encompasses the use of these shRNA sequences as therapeutic agents or biological tools.

## BACKGROUND OF THE INVENTION

RNA interference (RNAi) is a powerful tool for gene silencing, achieved through the use of small interfering RNAs (siRNAs) or short-hairpin RNAs (shRNAs). The canonical RNAi pathway involves the processing of shRNAs by the ribonucleases Drosha and Dicer into ∼22-nucleotide (nt) siRNAs, which are then loaded into the RNA-Induced Silencing Complex (RISC) to target complementary mRNAs for degradation or translational repression. While siRNAs are typically used for transient gene silencing, shRNAs offer sustained RNAi effects when expressed from vectors.

Endogenous microRNAs (miRNAs) target short sequences in the 3′ untranslated regions (UTRs) of mRNAs, often with seed matches of six nucleotides (positions 2–7 of the miRNA). These interactions can lead to translational repression or mRNA cleavage, depending on the degree of complementarity. Non-canonical functions of small RNAs, such as gene activation, have also been reported, expanding the potential applications of RNAi technology.

Despite the widespread use of RNAi libraries for gene silencing, limitations exist. Rational design of siRNA or shRNA libraries targeting specific genes can be costly and time-consuming. Moreover, these libraries may not capture sequences with non-canonical mechanisms of action. To address these limitations, random shRNA libraries have been developed. However, previous random libraries have had limitations in terms of stem length, transcriptional efficiency, and the ability to introduce mismatches, which can affect the potency and specificity of shRNAs.

The present invention overcomes these limitations by providing a method for designing and synthesizing a random shRNA library with optimal stem length (29 nt) and the ability to introduce mismatches. The library is designed to allow the straightforward retrieval of hit sequences and their optimization through random mutagenesis and re-screening.

## SUMMARY OF THE INVENTION

The present invention provides a method for designing, synthesizing, and testing a random, shRNA-encoding library. The method includes the following steps:

1. **Design of the shRNA Cassette**: The shRNA cassette is designed to include a 29-nucleotide (nt) random sequence, a non-complementary loop, and the reverse complement of the random sequence. The cassette also includes a polymerase III termination sequence and a poly-pyrimidine tract upstream of the transcription start site.

2. **Synthesis of the shRNA Cassette**: The shRNA cassette is synthesized using a series of enzymatic steps, including annealing, ligation, and digestion, to create a double-stranded DNA fragment with the desired sequence.

3. **Cloning of the shRNA Cassette**: The double-stranded DNA fragment is cloned into an appropriate vector, such as a retroviral expression vector, for robust expression.

4. **Functional Screening**: The library is used to infect a cell line, such as the murine pro-B cell line FL5.12, which is dependent on interleukin-3 (IL3). Cells are subjected to cycles of IL3 withdrawal and recovery to select for shRNAs that protect cells from apoptosis.

5. **Hit Retrieval and Optimization**: Hit sequences are retrieved by PCR and cloning. These sequences are then subjected to random mutagenesis to optimize their activity. The optimized sequences are re-screened to confirm their improved protective effects.

The invention further provides shRNA sequences identified through this method, which can be used as therapeutic agents or biological tools to protect cells from apoptosis or other stress conditions.

## DETAILED DESCRIPTION OF THE INVENTION

### EXPERIMENTAL DETAILS SECTION

#### Materials and Experimental Methods

**Library Synthesis**

1. **Annealing and Extension**: A short primer (1) is annealed to a 97-nt oligo (2) containing a 29-nt random sequence (N29) and multiple enzyme sites. The primer has two mismatches with the oligo to preserve the PmeI and AarI recognition sequences. The annealed product is extended using Klenow enzyme.

2. **Ligation of Hairpin-Loop Linker**: The extended product is ligated to a hairpin-loop linker (3) using T4 DNA ligase.

3. **Nick and Open**: The ligated hairpin-loop is nicked using Nb.BbvCI, creating an exposed area on the stem for annealing of a second primer (4). The nicked hairpin-loop is opened using Bst polymerase.

4. **Digestion and Blunting**: The opened hairpin-loop is digested with BtgZI, creating a recessed end. The recessed end is filled in with Klenow to form a blunt end, and the other side is digested with NotI.

5. **Cloning into Vector**: The DNA cassette is blunt-NotI ligated into an appropriately modified expression vector, such as pSuper.

6. **Loop Digestion and Reclosure**: The loop sequence is digested with PmeI and AarI, and the recessed end is filled in with Klenow. The vector is re-closed using a unimolecular, blunt-blunt ligation.

**Random Mutagenesis**

1. **Oligo Synthesis**: Random mutagenesis of the 3p sequence is carried out on a PCR-MATE EP 391 DNA synthesizer. Small amounts of the other three phosphoramidites are spiked into each phosphoramidite bottle to achieve a desired mutation rate.

2. **Annealing and Cloning**: The synthesized oligos are purified, annealed, and cloned into the pSiren vector prepared with BglII and NotI.

**Cell Culture and Retroviral Transduction**

1. **Cell Line and Media**: The FL5.12 pro-B cell line is cultured in RPMI 1640 media supplemented with 10% FBS, 10 mM Hepes, 100 U/ml Penicillin, 100 mg/ml Streptomycin, 55 mM β-Mercaptoethanol, and 0.3 ng/ml IL3.

2. **Retroviral Supernatant Preparation**: 293T cells are transfected with the pSiren library and an ecotropic retroviral packaging plasmid (pCL-Eco). The supernatant is harvested and used to infect FL5.12 cells with polybrene.

3. **Infection Efficiency**: Infection efficiency is monitored by GFP expression using flow cytometry. Ideally, the GFP% is kept at ∼33% or less to ensure that the majority of infected cells receive only one construct.

**Sequence Retrieval and Confirmation**

1. **Genomic DNA Extraction**: Genomic DNA is extracted from cells enriched for GFP after IL3 starvation/recovery cycles using a Qiagen kit.

2. **PCR Amplification**: The shRNA-encoding cassette is amplified from genomic DNA using primers flanking the cassette on pSiren.

3. **Half-shRNA PCR Amplification**: Two sets of primers are used to amplify the 5′ and 3′ halves of the stem separately, allowing for sequencing of the hairpin-loop structure.

**Apoptosis Induction and Caspase 3 Assay**

1. **Apoptosis Induction**: Apoptosis is induced in FL5.12 cells by washing with IL3-negative medium and resuspending in IL3-negative medium.

2. **Flow Cytometry**: Cells are stained with propidium iodide (PI) and analyzed by flow cytometry to determine the percentage of GFP-positive (infected), PI-negative (live) cells.

3. **Caspase 3 Assay**: Caspase 3 activity is measured using a colorimetric substrate (Ac-DEVD-pNA) and an enzyme activity kit.

### Results

**Library Synthesis and Characterization**

The synthesis of the random, shRNA-encoding library is described in detail in the Materials and Methods section. The library was designed to include a 29-nt random sequence, a non-complementary loop, and the reverse complement of the random sequence. The library was cloned into the retroviral expression vector pSuper and later transferred to pSiren for better expression of green fluorescent protein (GFP). The library was characterized by sequencing 40 individual clones, which showed an AT composition of 52.8% and a GC composition of 47.2%.

**Selection for shRNAs That Inhibit Apoptosis**

The murine pro-B cell line FL5.12, which is IL3-dependent, was infected with the library to achieve ∼30% GFP positivity. After cycles of IL3 withdrawal and recovery, the percentage of GFP-positive cells increased to 60%, suggesting the presence of hit sequences that conferred a relative survival advantage. Three putative hit sequences (1p, 3p, and 8p) were identified and tested for their protective effects. Clones 1p and 3p significantly improved survival relative to a control random shRNA or empty vector.

**Hit-Optimization**

Random variants of the 3p shRNA sequence were created using an oligo synthesizer, and the sublibrary was screened for improved protective effects. One variant (3p05p) was identified that significantly improved survival relative to 3p. Further mutagenesis studies revealed that the A-to-C change in 3p05p accounted for most of the improved effect, and that both the mismatch and the presence of the specific nucleotide C contributed to the improved activity.

**Conclusion**

The random shRNA-encoding library described herein provides a powerful tool for identifying and optimizing shRNA sequences that protect cells from apoptosis. The library design allows for the straightforward retrieval of hit sequences and their optimization through random mutagenesis and re-screening. The identified shRNA sequences have potential applications as therapeutic agents or biological tools in various systems, including protection against viral infection and stem-cell differentiation.