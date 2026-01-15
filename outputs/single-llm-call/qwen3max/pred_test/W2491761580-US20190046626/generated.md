# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to the field of recombinant protein expression and vaccine development, specifically directed to the production of recombinant Chlamydia major outer membrane protein (MOMP) for use in prophylactic and therapeutic compositions against Chlamydia infections. More particularly, the invention provides methods for the optimized expression of MOMP in the outer membrane of Escherichia coli (E. coli) through codon harmonization, selection of appropriate expression vectors and promoters, use of heterologous leader sequences, and precise control of culture conditions. The resulting recombinant MOMP retains structural and immunogenic features comparable to native MOMP purified from Chlamydia elementary bodies, thereby serving as a viable and scalable antigen for vaccine formulations targeting Chlamydia trachomatis and related species.

## BACKGROUND OF THE INVENTION

Chlamydia trachomatis is an obligate intracellular Gram-negative bacterium that causes a wide spectrum of human diseases. Serovars A, B, Ba, and C are associated with ocular trachoma, a leading cause of preventable blindness in endemic regions. Serovars L1, L2, and L3 cause lymphogranuloma venereum, a sexually transmitted infection characterized by invasive lymphadenopathy and systemic complications. Serovars D through K are responsible for oculogenital infections, including urethritis, cervicitis, pelvic inflammatory disease, tubal infertility, ectopic pregnancy, and neonatal pneumonia or conjunctivitis. Given its high prevalence and significant public health burden, Chlamydia trachomatis is one of the most common bacterial sexually transmitted infections worldwide, underscoring the urgent need for an effective vaccine.

Historically, formalin-inactivated whole-cell Chlamydia vaccines provided only transient and incomplete protection, often exacerbating disease upon natural reinfection. Consequently, subunit vaccine strategies have gained prominence, with the major outer membrane protein (MOMP) emerging as a leading candidate. MOMP constitutes approximately 60% of the outer membrane mass of the infectious elementary body (EB) form of Chlamydia and contains both B-cell epitopes capable of eliciting neutralizing antibodies and T-cell epitopes that drive cellular immune responses. Native MOMP (nMOMP) purified from EBs has demonstrated protective efficacy in murine challenge models, validating its potential as a vaccine antigen. However, large-scale production of nMOMP is inherently limited by the obligate intracellular nature of Chlamydia, which necessitates growth in eukaryotic cell cultures—a process that is costly, labor-intensive, and difficult to standardize for commercial manufacturing.

Recombinant expression of MOMP offers a scalable alternative. However, early attempts to express MOMP in E. coli resulted in cytoplasmic inclusion bodies, and refolded protein exhibited diminished immunogenicity and protective capacity compared to native MOMP. This loss of efficacy is attributed to improper folding and conformational distortion, as MOMP is a cysteine-rich, 16-stranded β-barrel porin whose native structure is critical for antigenic integrity. Subsequent efforts to target MOMP to the E. coli outer membrane—where it might adopt a more native-like conformation—were hampered by host toxicity, low expression yields, and poor surface localization. While some studies reported improved surface expression using engineered leader sequences, comprehensive evaluation of expression parameters—including codon usage, promoter strength, vector copy number, and culture conditions—was lacking, and crucially, the immunogenicity of such outer membrane-expressed recombinant MOMP (rMOMP) in animal models remained unverified.

Thus, there remains a critical need for a robust, reproducible, and scalable method to produce recombinant MOMP that faithfully mimics the structural and immunological properties of native MOMP. The present invention addresses this unmet need by integrating codon harmonization, optimized expression vector design, heterologous leader sequences, and precisely controlled fermentation conditions to achieve high-yield, outer membrane-localized rMOMP that elicits protective immune responses comparable to nMOMP.

## SUMMARY OF THE INVENTION

The present invention provides a method for producing recombinant Chlamydia major outer membrane protein (rMOMP) comprising transforming a host cell with an expression vector encoding a nucleic acid molecule that includes a leader sequence operably linked to a codon-harmonized MOMP gene derived from Chlamydia trachomatis or Chlamydia muridarum; culturing the transformed host cell under conditions that promote expression of rMOMP in the outer membrane of the host cell; and optionally purifying the expressed rMOMP. The host cell is preferably E. coli, and the expression vector is selected to have a low copy number and a promoter of moderate strength, such as the λPL promoter, to avoid transcriptional overload and host toxicity. The method further comprises inducing expression at mid-log phase (OD600 ~0.5) with IPTG at 30°C for 4 hours in a defined medium such as Cinnabar medium to maximize outer membrane localization while maintaining cell viability.

As used herein, “recombinant” refers to a polypeptide produced by expression of a cloned gene in a heterologous host system. “Isolated” or “purified” means a substance that has been separated from at least one component with which it was naturally associated, preferably to a purity of at least 70% by mass. “Homology” and “sequence identity” refer to the degree of similarity between two nucleic acid or amino acid sequences, calculated using standard alignment algorithms such as BLAST or ClustalW, with identity percentages based on exact matches over the aligned region.

The expression cassette comprises a promoter, a leader sequence, and the MOMP coding sequence. Suitable promoters include λPL, tac, or trc, with moderate strength preferred. The leader sequence may be derived from Shigella flexneri SopA, Salmonella enterica PgtE, Yersinia pestis Pla, E. coli OmpA or OmpP, or Erwinia carotovora PelB, all of which facilitate translocation across the inner membrane and enhance outer membrane insertion. The term “MAA” refers to muramyl dipeptide analogs, though not used herein, and “ISCOM-type adjuvant” denotes immune-stimulating complexes containing saponins, cholesterol, and phospholipids.

“Derivative” encompasses MOMP variants with conservative amino acid substitutions, deletions, or additions that retain immunogenicity. Conservative substitutions involve replacement with residues of similar physicochemical properties (e.g., leucine for isoleucine). Chemical modifications such as PEGylation are also contemplated. “Cell,” “cell line,” and “cell culture” refer to prokaryotic or eukaryotic systems used for propagation or expression. “Treatment” includes both prophylaxis and therapy. A “therapeutically effective amount” is that sufficient to ameliorate symptoms, while an “immunologically effective amount” induces a measurable immune response, defined as detectable antibody titers or T-cell activation via ELISA, ELISPOT, or cytotoxicity assays.

The invention further provides pharmaceutical compositions comprising rMOMP and a pharmaceutically acceptable carrier, optionally with adjuvants such as CpG oligonucleotides, Montanide ISA 720, aluminum salts, or saponin-based formulations. These compositions are useful for the prevention or treatment of Chlamydia infection in a patient, which includes humans and non-human primates. Animal models, particularly C57BL/6 mice challenged intravaginally with C. trachomatis serovar D, are used to evaluate efficacy. Immune sera or T cells from immunized subjects are assessed for neutralizing capacity, interferon-gamma production (via ELISA or ELISPOT), and cytotoxic activity (via 51Chromium release assay).

## DETAILED DESCRIPTION OF THE INVENTION

MOMP is a well-established vaccine target due to its abundance, surface exposure, and immunodominance. However, purification of native MOMP from Chlamydia-infected cells is impractical for large-scale vaccine production. Recombinant expression in E. coli offers a solution, but prior attempts yielded misfolded protein in inclusion bodies. The present invention overcomes this by expressing rMOMP directly in the E. coli outer membrane, where it can adopt a near-native conformation. This is achieved through a multi-faceted optimization strategy.

First, the MOMP gene is codon-harmonized rather than conventionally optimized. Codon harmonization preserves the relative codon usage frequency of the native Chlamydia host, including rare codons that may introduce translational pauses necessary for proper co-translational folding. This contrasts with codon optimization, which replaces all codons with the most frequent E. coli equivalents, often accelerating translation and causing misfolding. Synthetic genes encoding full-length MOMP, including its native signal sequence or a heterologous leader, are designed using codon usage tables from both Chlamydia and E. coli.

Second, expression vectors are selected to minimize transcriptional burden. High-copy plasmids and strong promoters (e.g., T7) lead to excessive mRNA and protein accumulation, overwhelming the Sec translocon and β-barrel assembly machinery (BAM), resulting in aggregation and toxicity. The pAVE029 vector, a low-copy plasmid with a moderate-strength λPL promoter, provides optimal expression levels. Induction is tightly controlled with IPTG at mid-log phase (OD600 ~0.5) and reduced temperature (30°C) to balance protein synthesis with folding capacity.

Third, leader sequences are engineered to enhance secretion and outer membrane targeting. Native Chlamydia MOMP leaders function poorly in E. coli. In contrast, leaders from E. coli OmpA, OmpP, or PelB, or from pathogenic bacteria like Shigella SopA, significantly improve surface expression by efficiently engaging the Sec pathway and BAM complex. The PelB leader, in particular, enables complete cleavage and proper N-terminus formation.

The resulting rMOMP is localized to the outer membrane, as confirmed by flow cytometry and protease accessibility assays. It exhibits β-barrel secondary structure by circular dichroism, migrates similarly to nMOMP on SDS-PAGE, and reacts with conformation-sensitive antibodies. Purification involves membrane fractionation, selective detergent extraction (e.g., sarkosyl), and chromatography (size exclusion and ion exchange), yielding ~6 mg/L of >70% pure protein.

### Pharmaceutical Compositions

Pharmaceutical compositions of the invention comprise rMOMP formulated with a pharmaceutically acceptable carrier, which includes any excipient, diluent, or stabilizer compatible with administration to a subject. Examples include sterile water for injection, saline, phosphate-buffered saline (PBS), and buffers such as Tris, HEPES, or acetate, typically adjusted to pH 6.0–7.5. Formulations may be sterile injectable solutions, dispersions, or lyophilized powders reconstituted prior to use. Prolonged absorption may be achieved with depot formulations or biodegradable polymers.

Toxicity and therapeutic efficacy are determined by standard pharmaceutical procedures in cell cultures or animal models to establish a therapeutic index. Dosage ranges vary but typically deliver 1–100 μg of rMOMP per dose. Compositions may include additional rMOMP antigens from different serovars or non-MOMP Chlamydia antigens (e.g., CPAF, PmpG) for broad protection. Adjuvants are essential and include saponin-based (e.g., Quil A, ISCOMs), aluminum salts (e.g., Alhydrogel), TLR agonists (e.g., CpG, MPLA), or oil-in-water emulsions (e.g., Montanide ISA 720). Combinations, such as CpG + Montanide, synergistically enhance Th1-biased responses critical for Chlamydia clearance.

### Methods of Use

The rMOMP and pharmaceutical compositions are used for the prophylaxis or treatment of Chlamydia infection. Administration routes include subcutaneous, intramuscular, intranasal, or intravaginal, delivered via syringes, microneedles, or mucosal sprays. The compositions stimulate humoral and cellular immunity, reducing bacterial load upon challenge.

### Codon-Harmonized MOMP Nucleotide Sequences

The invention includes codon-harmonized nucleic acid molecules encoding MOMP from C. trachomatis serovars D or E, or C. muridarum, as set forth in SEQ ID NOs:1–3. Derivatives include variants with up to 10% amino acid substitutions, deletions, or additions that retain immunogenicity. Leader sequences such as PelB (SEQ ID NO:4) or OmpA (SEQ ID NO:5) are operably linked. All publications cited herein are incorporated by reference.

## EXAMPLE 1

### Expression of Recombinant Chlamydia MOMP.

Codon harmonization was motivated by the need to mimic native translational kinetics. Evaluation showed harmonized genes yielded 2-fold higher surface expression than optimized genes. Expression vector screening revealed that only low-copy, moderate-promoter systems (e.g., pAVE029) supported high rMOMP display. Leader sequence testing identified PelB and OmpA as superior. Condition optimization established 4h induction at 30°C in Cinnabar medium at OD590 0.5 as optimal.

## EXAMPLE 2

### Purification of Recombinant Chlamydia MOMP.

Cells were disrupted by microfluidization; membranes isolated by ultracentrifugation. Sequential washes with high salt, Triton X-100, and β-octyl-glucoside removed contaminants. Sarkosyl extraction solubilized rMOMP, which was purified by size exclusion and ion exchange chromatography, yielding ~6 mg/L of >70% pure protein.

## EXAMPLE 3

### Mouse Immunogenicity and Challenge Study

Mice immunized with rMOMP or nMOMP plus CpG/Montanide developed comparable IgG titers and IgG1/IgG2c ratios, indicating similar Th1/Th2 bias. Upon intravaginal challenge with CtD, both groups showed significant reduction in bacterial shedding versus controls, with no statistical difference between rMOMP and nMOMP.

### EXAMPLE 4

MOMP gene sequences from C. muridarum and C. trachomatis serovars D and E were retrieved from genomic databases. Codon harmonization was performed using native and E. coli codon usage tables. Restriction sites (NdeI, XhoI) were removed for cloning.

### EXAMPLE 5

Harmonized genes were synthesized, cloned into pAVE029, and transformed into E. coli W25113. Expression induced with IPTG yielded OM-localized rMOMP, confirmed by flow cytometry and Western blot.

### EXAMPLE 6

Whole cell flow cytometry using anti-Chlamydia EB sera confirmed surface expression of rMOMP, with geomean fluorescence intensity correlating with expression levels.

### EXAMPLE 7

SDS-PAGE and Western blot showed rMOMP migrated similarly to nMOMP, with minor differences in oligomerization. Anti-E. coli sera detected ~30% host contaminants.

### EXAMPLE 8

Membrane fractions were isolated by ultracentrifugation. rMOMP was extracted with sarkosyl and purified by Sephacryl S300 size exclusion and Hitrap Q ion exchange chromatography.

### EXAMPLE 9

Chlamydiae were propagated in HeLa 229 cells in EMEM with 10% FBS, harvested after 72h, and purified by Renograffin gradient centrifugation.

### EXAMPLE 10

Mice immunized s.c. with rMOMP + CpG/Montanide developed high-titer serum antibodies against CtD EBs, measured by ELISA.

### EXAMPLE 11

Mice were challenged intravaginally with 1x105 CtD EBs 14 days post-final immunization.

### EXAMPLE 12

Genital swab DNA was extracted and analyzed by real-time PCR using 16S rRNA primers/probe to quantify Chlamydia load.

### EXAMPLE 13

ELISA plates coated with CtD EBs detected serum IgG. HRP-conjugated secondary antibodies and TMB substrate enabled quantification of antibody titers.