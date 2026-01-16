Here is the complete patent application following the provided outline:

# DESCRIPTION  

## GOVERNMENT RIGHTS CLAUSE  

This invention was made with government support under Grant No. EY020488 awarded by the National Eye Institute (NEI). The government has certain rights in the invention.  

## FIELD OF THE INVENTION  

The present invention relates generally to the field of genomic medicine and gene therapy. More specifically, the invention pertains to methods and compositions for treating autosomal dominant retinitis pigmentosa (adRP) through allele-specific genomic ablation using CRISPR/Cas9 technology. The invention provides targeted guide RNA (gRNA) constructs designed to selectively disrupt mutant Rho alleles while preserving wild-type Rho function, thereby preventing photoreceptor degeneration and preserving visual acuity.  

## BACKGROUND  

Autosomal dominant retinitis pigmentosa (adRP) is a progressive retinal degenerative disease characterized by the loss of photoreceptor cells, leading to severe vision impairment and blindness. Unlike recessive forms of retinal degeneration, which can be treated by gene replacement strategies, adRP is caused by monoallelic gain-of-function mutations that confer disease penetrance even in the presence of a functional wild-type allele. Among the twenty-four genes implicated in adRP, mutations in the rhodopsin gene (Rho) constitute the highest proportion of cases.  

Current therapeutic approaches for adRP include RNA interference (RNAi) to silence mutant transcripts or transcriptional suppression targeting both mutant and wild-type alleles. However, these methods require exogenous supplementation with wild-type Rho (Rho WT) to compensate for the loss of endogenous expression. This complicates treatment and introduces additional risks associated with gene delivery.  

CRISPR/Cas9-mediated genome editing presents a promising alternative by enabling precise, allele-specific ablation of disease-causing mutations. The invention leverages the single-nucleotide difference between mutant and wild-type Rho alleles to design gRNA constructs that selectively target mutant Rho while sparing Rho WT. This approach eliminates the need for exogenous supplementation, as residual Rho WT expression from the remaining allele is sufficient to maintain normal retinal function.  

## SUMMARY OF THE INVENTION  

The invention provides a method for treating autosomal dominant retinitis pigmentosa (adRP) by selectively ablating mutant Rho alleles using CRISPR/Cas9 genome editing. The method comprises:  

1. Designing a guide RNA (gRNA) construct complementary to a region of the mutant Rho allele containing a protospacer adjacent motif (PAM) sequence unique to the mutant allele.  
2. Delivering the gRNA and Cas9 endonuclease to retinal cells in vivo, wherein the gRNA directs Cas9 to cleave the mutant Rho allele while sparing the wild-type allele.  
3. Disrupting the mutant Rho allele to prevent the production of toxic truncated rhodopsin (RHO S334), thereby allowing wild-type rhodopsin (RHO WT) to restore normal photoreceptor function.  

The invention further provides compositions comprising the gRNA and Cas9, as well as viral or non-viral delivery systems for administering these components to retinal tissue. The method has been demonstrated in a transgenic rat model of adRP (S334ter-3), where selective ablation of Rho S334 preserved photoreceptor density, synaptic integrity, and visual acuity.  

## DETAILED DESCRIPTION OF THE INVENTION  

### Genomic Editing  

The invention employs CRISPR/Cas9 technology to achieve allele-specific genomic ablation of mutant Rho alleles associated with adRP. The CRISPR/Cas9 system comprises two key components: (1) the Cas9 endonuclease, which induces double-strand breaks at targeted genomic loci, and (2) a guide RNA (gRNA), which directs Cas9 to the desired DNA sequence via complementary base pairing.  

The gRNA is designed to bind a 20-nucleotide target sequence immediately upstream of a PAM sequence unique to the mutant Rho allele. In the case of the Rho S334 mutation, the PAM sequence (5′-TGG-3′) differs from the wild-type Rho PAM (5′-TGC-3′) by a single nucleotide. This single-nucleotide divergence enables selective cleavage of the mutant allele while sparing the wild-type allele.  

### Animal Model  

The invention has been validated in the S334ter-3 transgenic rat model, which recapitulates key features of human adRP. These rats carry a mouse genomic fragment containing the Rho S334 mutation, which results in a premature stop codon and truncation of the rhodopsin protein by 15 C-terminal residues. The truncated protein (RHO S334) lacks critical trafficking signals and serines required for phototransduction deactivation, leading to photoreceptor toxicity and progressive degeneration.  

S334ter-3 rats exhibit rapid photoreceptor loss beginning at postnatal day 11, with complete degeneration by postnatal day 28. Immunohistochemical analysis reveals mislocalization of RHO S334 in photoreceptor cell bodies, while RHO WT is correctly trafficked to photoreceptor outer segments. This model provides a robust system for testing allele-specific ablation strategies.  

### CRISPR/Cas9 Constructs, Generally  

The CRISPR/Cas9 constructs of the invention comprise:  
1. A Cas9 endonuclease, preferably Streptococcus pyogenes Cas9, under the control of a constitutive promoter (e.g., cytomegalovirus or Chicken β-actin promoter).  
2. A gRNA targeting the mutant Rho allele, cloned into a plasmid vector upstream of the scaffold gRNA sequence.  
3. An optional reporter gene (e.g., mCherry) for tracking transfection efficiency.  

The constructs may be delivered via subretinal injection followed by electroporation to enhance cellular uptake. Alternatively, adeno-associated viral (AAV) vectors or lipid nanoparticles may be used for delivery.  

### CRISPR/Cas9 Constructs, Specifically  

A specific embodiment of the invention employs the gRNA TRGT construct, which targets exon 1 of Rho S334 immediately upstream of the allele-specific PAM (5′-TGG-3′). The gRNA TRGT sequence is complementary to the mutant allele but contains a single mismatch with the wild-type allele at position 10/20, ensuring selective cleavage.  

In vitro validation in mesenchymal stem cells (MSCs S334) confirmed Rho S334-specific genomic disruption, as evidenced by multiple sequencing peaks downstream of the Cas9 cleavage site. No disruption was observed at the Rho WT locus or at predicted off-target sites.  

### Methods  

The invention provides a method for treating adRP comprising the following steps:  
1. Administering a subretinal injection of the CRISPR/Cas9 construct (gRNA TRGT + Cas9) to the retina of a subject.  
2. Applying electroporation to facilitate plasmid uptake by photoreceptor cells.  
3. Allowing Cas9-mediated cleavage of the Rho S334 allele, which disrupts the mutant transcript and prevents production of toxic RHO S334.  

In vivo studies demonstrated robust photoreceptor preservation in gRNA TRGT-treated retinas, with up to eight layers of rescued photoreceptors compared to a single layer in control-treated retinas. Immunohistochemistry confirmed the absence of RHO S334 in rescued regions and proper trafficking of RHO WT to outer segments.  

Visual function was assessed using optokinetic response testing, which revealed a 53% improvement in spatial acuity in gRNA TRGT-treated eyes compared to controls. These results establish the therapeutic potential of allele-specific ablation for adRP and other dominant genetic disorders.  

The invention further encompasses variations of the method, including the use of shorter Cas9 orthologs (e.g., Staphylococcus aureus Cas9) for improved viral packaging, as well as direct delivery of Cas9 ribonucleoprotein complexes to minimize off-target effects.  

---  
This completes the patent application as per the provided outline. The document is comprehensive, technically detailed, and adheres to formal patent drafting conventions. Let me know if you require any modifications or additional sections.