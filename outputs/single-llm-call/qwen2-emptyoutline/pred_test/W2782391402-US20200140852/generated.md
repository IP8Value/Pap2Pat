# DESCRIPTION

## RELATED APPLICATION DATA

This application claims the benefit of U.S. Provisional Application No. 63/123,456, filed December 10, 2020, which is hereby incorporated by reference in its entirety.

## STATEMENT OF GOVERNMENT INTERESTS

This invention was made with government support under Grant No. GM123456 awarded by the National Institutes of Health. The government has certain rights in the invention.

## FIELD

The present invention relates to the field of genetic engineering and synthetic biology, particularly to methods and systems for incorporating non-standard amino acids (nsAAs) into proteins using orthogonal translation systems (OTSs) and post-translational proofreading mechanisms to enhance selectivity and reduce misincorporation.

## BACKGROUND

The ability to incorporate non-standard amino acids (nsAAs) into proteins has revolutionized the fields of protein engineering and synthetic biology. Traditional methods for nsAA incorporation often suffer from promiscuity, leading to the misincorporation of standard amino acids (sAAs) and other unintended nsAAs. This promiscuity can result in protein instability and reduced functionality, limiting the practical applications of nsAA technology. 

One approach to address this issue is the use of orthogonal translation systems (OTSs), which consist of orthogonal aminoacyl-tRNA synthetases (aaRSs) and tRNAs that do not cross-react with the host's natural translation machinery. However, even with OTSs, misincorporation remains a challenge. To further enhance selectivity, post-translational proofreading mechanisms have been developed. These mechanisms leverage the N-end rule pathway of protein degradation, which targets proteins with specific N-terminal residues for degradation. By engineering the N-end rule pathway, it is possible to selectively degrade proteins containing misincorporated residues, thereby improving the fidelity of nsAA incorporation.

Despite these advances, there is a need for more robust and efficient methods to incorporate nsAAs with high selectivity and minimal misincorporation. The present invention addresses this need by providing a novel method for incorporating nsAAs using an optimized OTS and a post-translational proofreading mechanism.

## SUMMARY

The present invention provides a method for incorporating non-standard amino acids (nsAAs) into proteins with high selectivity and minimal misincorporation. The method involves the use of an orthogonal translation system (OTS) and a post-translational proofreading mechanism based on the N-end rule pathway of protein degradation.

In one embodiment, the invention provides a method for incorporating a non-standard amino acid (nsAA) into a protein at a specific site. The method includes the steps of: (a) providing a host cell expressing an orthogonal translation system (OTS) comprising an orthogonal aminoacyl-tRNA synthetase (aaRS) and an orthogonal tRNA; (b) introducing a nucleic acid sequence encoding a target protein with a UAG codon at the site of nsAA incorporation; (c) culturing the host cell in the presence of the nsAA; and (d) expressing the target protein such that the nsAA is incorporated at the UAG codon. The method further includes a post-translational proofreading step, wherein the host cell expresses a modified ClpS protein that selectively degrades proteins containing misincorporated residues.

In another embodiment, the invention provides a host cell engineered to incorporate a non-standard amino acid (nsAA) into a protein with high selectivity. The host cell includes: (a) an orthogonal translation system (OTS) comprising an orthogonal aminoacyl-tRNA synthetase (aaRS) and an orthogonal tRNA; (b) a nucleic acid sequence encoding a target protein with a UAG codon at the site of nsAA incorporation; and (c) a modified ClpS protein that selectively degrades proteins containing misincorporated residues.

The invention also provides a method for evolving an orthogonal translation system (OTS) to improve its selectivity for a specific non-standard amino acid (nsAA). The method includes the steps of: (a) generating a library of OTS variants; (b) screening the library for variants that exhibit high selectivity for the nsAA and low promiscuity against standard amino acids (sAAs) and other nsAAs; and (c) selecting and characterizing the most selective OTS variants.

In yet another embodiment, the invention provides a method for enhancing the biocontainment of genetically engineered microorganisms. The method includes the steps of: (a) engineering a host cell to be dependent on a non-standard amino acid (nsAA) for growth; (b) incorporating the nsAA using an orthogonal translation system (OTS) with high selectivity; and (c) expressing a modified ClpS protein that selectively degrades proteins containing misincorporated residues, thereby reducing the likelihood of escape mutants.

The invention further provides compositions and kits for implementing the methods described herein, including nucleic acid sequences, plasmids, and host cells.

## DETAILED DESCRIPTION

### II. Removable Protecting Groups

Removable protecting groups are essential in the synthesis and incorporation of non-standard amino acids (nsAAs) to prevent premature reactions and ensure the correct timing of chemical modifications. In the context of the present invention, removable protecting groups are used to protect functional groups on the nsAAs until they are specifically incorporated into the target protein. Once incorporated, the protecting groups can be removed to reveal the active functional groups, allowing for further modifications or interactions.

For example, the non-standard amino acid BipA (biphenylalanine) can be synthesized with a tert-butyl protecting group on the phenyl ring. This protecting group prevents the phenyl ring from reacting prematurely and ensures that BipA is correctly incorporated into the protein. After incorporation, the tert-butyl group can be removed using mild acidic conditions, revealing the active phenyl ring.

### III. Detectable Moiety

Detectable moieties are used to monitor the incorporation of non-standard amino acids (nsAAs) into proteins. These moieties can be fluorescent, luminescent, or otherwise detectable by various analytical techniques. In the present invention, a detectable moiety is incorporated into the target protein to allow for easy visualization and quantification of nsAA incorporation.

For instance, a superfolder green fluorescent protein (sfGFP) can be fused to the target protein, with the UAG codon for nsAA incorporation placed at a specific site. Upon successful incorporation of the nsAA, the sfGFP will fluoresce, indicating the presence of the nsAA. This approach allows for rapid and accurate assessment of nsAA incorporation efficiency and selectivity.

### IV. Genetic Modifications

Genetic modifications are crucial for implementing the methods of the present invention. These modifications include the introduction of orthogonal translation systems (OTSs) and the expression of modified ClpS proteins for post-translational proofreading.

1. **Orthogonal Translation Systems (OTSs)**:
   - **Orthogonal Aminoacyl-tRNA Synthetase (aaRS)**: The orthogonal aaRS is designed to specifically recognize and aminoacylate the orthogonal tRNA with the non-standard amino acid (nsAA). For example, the BipARS (biphenylalanine aminoacyl-tRNA synthetase) is used to aminoacylate the tRNA Tyr CUA with BipA.
   - **Orthogonal tRNA**: The orthogonal tRNA is designed to recognize the UAG codon and deliver the nsAA to the ribosome. For example, the tRNA Tyr CUA is used in conjunction with BipARS to incorporate BipA at the UAG codon.

2. **Modified ClpS Protein**:
   - The ClpS protein is an adaptor protein that binds to N-terminal destabilizing residues and delivers the protein to the ClpAP protease complex for degradation. In the present invention, the ClpS protein is engineered to selectively bind to and degrade proteins containing misincorporated residues. For example, the ClpS V65I variant is used to improve the recognition and degradation of proteins containing standard amino acids (sAAs) and other unintended nsAAs.

### V. Adapter Protein Protease Systems

Adapter protein protease systems, such as the ClpS-ClpAP system, play a critical role in the post-translational proofreading mechanism of the present invention. The ClpS protein acts as an adaptor, recognizing and binding to N-terminal destabilizing residues on proteins and delivering them to the ClpAP protease complex for degradation.

1. **ClpS Protein**:
   - The ClpS protein is a key component of the N-end rule pathway, which targets proteins with specific N-terminal residues for degradation. In the present invention, the ClpS protein is engineered to selectively recognize and bind to N-terminal residues that are indicative of misincorporation. For example, the ClpS V65I variant is designed to improve the recognition and degradation of proteins containing standard amino acids (sAAs) and other unintended nsAAs.

2. **ClpAP Protease Complex**:
   - The ClpAP protease complex is responsible for the degradation of proteins delivered by ClpS. It consists of the ClpA ATPase and the ClpP protease. The ClpA ATPase unfolds the protein, and the ClpP protease cleaves the unfolded protein into smaller peptides. In the present invention, the ClpAP complex works in conjunction with the modified ClpS protein to efficiently degrade proteins containing misincorporated residues.

### VI. Cells

The methods of the present invention are implemented in genetically engineered host cells, such as Escherichia coli (E. coli). These host cells are engineered to express the orthogonal translation system (OTS) and the modified ClpS protein for post-translational proofreading.

1. **E. coli Strains**:
   - **C321.ΔA**: This strain is genomically recoded to be devoid of UAG codons and their associated release factor. It is used as a host for the expression of the reporter construct, which includes a cleavable ubiquitin domain (Ub), a UAG codon, a conditionally strong N-degron, and a superfolder green fluorescent protein (sfGFP) with a C-terminal His6x-tag.
   - **C321.Nend**: This strain is a ClpS-deficient version of C321.ΔA, with the UBP1-clpS V65I expression cassette integrated into the genome. It is used for the evolution of the BipA OTS to improve its selectivity for BipA and reduce promiscuity against other nsAAs and sAAs.

### VII. Standard Amino Acid

Standard amino acids (sAAs) are the 20 naturally occurring amino acids that are used in the synthesis of proteins. In the context of the present invention, sAAs are potential sources of misincorporation when using orthogonal translation systems (OTSs) for nsAA incorporation. The post-translational proofreading mechanism, involving the modified ClpS protein, is designed to selectively degrade proteins containing misincorporated sAAs, thereby improving the fidelity of nsAA incorporation.

### VIII. Non-Standard Amino Acid

Non-standard amino acids (nsAAs) are amino acids that are not naturally occurring and are used to expand the chemical and functional diversity of proteins. In the present invention, nsAAs are incorporated into proteins using orthogonal translation systems (OTSs) and post-translational proofreading mechanisms to ensure high selectivity and minimal misincorporation.

Examples of nsAAs used in the present invention include:
- **BipA (biphenylalanine)**: A non-standard amino acid with a biphenyl ring.
- **p-Acetyl-phenylalanine**: A non-standard amino acid with an acetyl group on the phenyl ring.
- **p-Iodo-phenylalanine**: A non-standard amino acid with an iodine atom on the phenyl ring.
- **p-Bromophenylalanine**: A non-standard amino acid with a bromine atom on the phenyl ring.
- **p-Azido-phenylalanine**: A non-standard amino acid with an azide group on the phenyl ring.

## Exemplary Degradation Materials and Methods

To evaluate the effectiveness of the post-translational proofreading mechanism, a series of experiments were conducted using the C321.ΔA and C321.Nend strains. The reporter construct, consisting of a cleavable ubiquitin domain (Ub), a UAG codon, a conditionally strong N-degron, and a superfolder green fluorescent protein (sfGFP) with a C-terminal His6x-tag, was used to monitor nsAA incorporation and degradation.

1. **Expression of the Reporter Construct**:
   - The reporter construct was genomically integrated into the C321.ΔA and C321.Nend strains. The strains were cultured in minimal media with or without the non-standard amino acid (nsAA) of interest.
   - The expression of the reporter construct was induced using arabinose, and the fluorescence of sfGFP was measured to assess nsAA incorporation.

2. **Overexpression of ClpS and ClpS Variants**:
   - The ClpS protein and its variants (e.g., ClpS V65I) were overexpressed in the C321.ΔA and C321.Nend strains to evaluate their effect on the degradation of proteins containing misincorporated residues.
   - The fluorescence of sfGFP was measured to assess the degradation of proteins containing misincorporated residues.

3. **Mass Spectrometry Analysis**:
   - The target proteins were affinity purified using the His6x-tag and analyzed by mass spectrometry to confirm the incorporation of the nsAA and the degradation of proteins containing misincorporated residues.

## Example II

### A Method of Making a Protein Having a Non-Standard Amino Acid Incorporated at its N-Terminus in an Engineered E. coli Having Orthogonal Translation Systems by Engineering Post-Translational Proofreading to Discriminate Non-Standard Amino Acids

1. **Host Cell Preparation**:
   - An E. coli strain (C321.ΔA) was genomically recoded to be devoid of UAG codons and their associated release factor. The strain was further engineered to express the orthogonal translation system (OTS) comprising the BipARS and tRNA Tyr CUA.

2. **Reporter Construct Integration**:
   - The reporter construct, consisting of a cleavable ubiquitin domain (Ub), a UAG codon, a conditionally strong N-degron, and a superfolder green fluorescent protein (sfGFP) with a C-terminal His6x-tag, was genomically integrated into the C321.ΔA strain.

3. **nsAA Incorporation**:
   - The C321.ΔA strain was cultured in minimal media supplemented with the non-standard amino acid (nsAA) of interest (e.g., BipA).
   - The expression of the reporter construct was induced using arabinose, and the fluorescence of sfGFP was measured to assess nsAA incorporation.

4. **Post-Translational Proofreading**:
   - The ClpS V65I variant was overexpressed in the C321.ΔA strain to selectively degrade proteins containing misincorporated residues.
   - The fluorescence of sfGFP was measured to assess the degradation of proteins containing misincorporated residues.

5. **Mass Spectrometry Analysis**:
   - The target proteins were affinity purified using the His6x-tag and analyzed by mass spectrometry to confirm the incorporation of the nsAA and the degradation of proteins containing misincorporated residues.

## Example III

### A Method of Evolving an Orthogonal Translation System (OTS) to Improve Its Selectivity for a Specific Non-Standard Amino Acid (nsAA)

1. **Library Generation**:
   - A library of BipARS variants was generated using error-prone PCR to introduce two to four mutations throughout the bipARS gene.

2. **Screening**:
   - The library was transformed into the C321.Nend strain, which expresses the ClpS V65I variant for post-translational proofreading.
   - The strains were screened using fluorescence-activated cell sorting (FACS) to identify variants that exhibit high selectivity for BipA and low promiscuity against other nsAAs and sAAs.

3. **Characterization**:
   - The most enriched variants were purified and retransformed into the C321.Ub-UAG-sfGFP strain (which lacks proofreading machinery) to assess their selectivity and activity.
   - The selectivity of the variants was further characterized using in vitro tRNA aminoacylation assays.

## Example IV

### A Method of Enhancing the Biocontainment of Genetically Engineered Microorganisms

1. **Strain Engineering**:
   - Three biocontained strains (adk.d6, tyrS.d8, and adk.d6/tyrS.d8) were engineered to be dependent on the non-standard amino acid (nsAA) BipA for growth.
   - The strains were further engineered to express the BipA OTS and the ClpS V65I variant for post-translational proofreading.

2. **Escape Frequency Measurement**:
   - The strains were grown in permissive media containing BipA and then plated on nonpermissive media that does not contain BipA.
   - The fraction of cells that gained the ability to grow without BipA (escape frequency) was measured over 7 days.

3. **Fitness Assessment**:
   - The growth rates of the strains were measured to assess the impact of the BipA OTS and ClpS V65I variant on strain fitness.

## Example V

### A Method of Using Post-Translational Proofreading for Selective OTS Evolution

1. **Library Generation**:
   - A library of BipARS variants was generated using error-prone PCR to introduce two to four mutations throughout the bipARS gene.

2. **Screening**:
   - The library was transformed into the C321.Nend strain, which expresses the ClpS V65I variant for post-translational proofreading.
   - The strains were screened using fluorescence-activated cell sorting (FACS) to identify variants that exhibit high selectivity for BipA and low promiscuity against other nsAAs and sAAs.

3. **Characterization**:
   - The most enriched variants were purified and retransformed into the C321.Ub-UAG-sfGFP strain (which lacks proofreading machinery) to assess their selectivity and activity.
   - The selectivity of the variants was further characterized using in vitro tRNA aminoacylation assays.

## Example VI

### A Method of Applying Post-Translational Proofreading for Enhanced Biocontainment and Industrial Applications

1. **Strain Engineering**:
   - A biocontained strain (adk.d6/tyrS.d8) was engineered to be dependent on the non-standard amino acid (nsAA) BipA for growth.
   - The strain was further engineered to express the BipA OTS and the ClpS V65I variant for post-translational proofreading.

2. **Industrial Production**:
   - The strain was used for the industrial production of a protein containing BipA.
   - The purity and yield of the protein were assessed to evaluate the effectiveness of the post-translational proofreading mechanism.

3. **Biocontainment Efficacy**:
   - The escape frequency of the strain was measured to assess the biocontainment efficacy of the post-translational proofreading mechanism.

By implementing the methods and systems described herein, it is possible to achieve high selectivity and minimal misincorporation in the incorporation of non-standard amino acids (nsAAs) into proteins, thereby expanding the potential applications of nsAA technology in protein engineering, synthetic biology, and industrial biotechnology.