# DESCRIPTION

## STATEMENT AS TO RIGHTS TO INVENTIONS MADE UNDER FEDERALLY SPONSORED RESEARCH AND DEVELOPMENT

This invention was made with government support under Grant No. [REDACTED FOR PATENT DRAFTING PURPOSES] awarded by the National Institutes of Health. The government has certain rights in the invention.

## BACKGROUND OF THE INVENTION

The field of human reproductive biology and regenerative medicine has long been hindered by the ethical and practical limitations associated with accessing and studying early human embryonic development, particularly during the critical window of weeks 2–3 post-fertilization when primordial germ cells (PGCs)—the precursors to sperm and oocytes—first emerge. PGCs represent the sole cellular lineage capable of transmitting genetic and epigenetic information across generations, and their proper specification is fundamental to fertility, inheritance, and the prevention of transgenerational disease. Despite their biological significance, the precise developmental origins, molecular regulators, and signaling dynamics governing human PGC specification remain poorly understood due to the inaccessibility of human post-implantation embryos at these stages. Consequently, researchers have turned to in vitro models using human pluripotent stem cells (hPSCs), including both embryonic stem cells (hESCs) and induced pluripotent stem cells (hiPSCs), to recapitulate early germline development through the generation of PGC-like cells (PGCLCs).

Prior art methods for differentiating hPSCs into PGCLCs have largely relied on complex three-dimensional (3D) aggregate cultures that mimic aspects of embryonic organization. These protocols typically involve a two-phase process: first, a brief induction of primitive streak (PS)-like intermediates using high concentrations of WNT and TGFβ signaling agonists (e.g., CHIR99021 and Activin A) for 12–60 hours; second, aggregation of these intermediates into 3D clusters followed by prolonged exposure (4–7 days) to a cocktail of growth factors including bone morphogenetic protein 4 (BMP4), stem cell factor (SCF), epidermal growth factor (EGF), leukemia inhibitory factor (LIF), and a ROCK inhibitor (e.g., Y-27632). While such approaches have demonstrated proof-of-concept for generating human PGCLCs expressing canonical markers such as NANOS3, SOX17, PRDM1 (BLIMP1), and TFAP2C, they suffer from several critical limitations that impede their utility for mechanistic studies, clinical translation, and scalable applications. First, the 3D architecture introduces significant heterogeneity due to gradients in nutrient availability, oxygen tension, and morphogen diffusion, leading to inconsistent differentiation outcomes and low reproducibility across cell lines and laboratories. Second, the requirement for high-dose BMP4 (often 200–500 ng/mL) is not only cost-prohibitive but also physiologically implausible, suggesting that inefficient signal delivery in aggregates necessitates supraphysiological ligand concentrations. Third, existing protocols yield highly heterogeneous populations in which PGCLCs are intermixed with off-target somatic lineages—particularly mesodermal derivatives—making it difficult to isolate pure PGCLCs for downstream functional or molecular analyses. Fourth, while fluorescent reporter lines (e.g., NANOS3-mCherry or SOX17-GFP) have enabled enrichment of PGCLCs in research settings, such genetically modified lines are unsuitable for clinical applications and do not address the need for universal, non-genetic purification strategies applicable to any wild-type hPSC line.

Moreover, the molecular trajectory of human PGCLC specification remains contentious. A prevailing model, largely derived from murine studies, posits that pluripotency factors such as OCT4 (POU5F1) and NANOG are transiently downregulated during early differentiation and then re-expressed specifically in the germline lineage, thereby “reacquiring” a pluripotent state. However, recent evidence from non-human primates suggests an alternative model wherein pluripotency factors are continuously expressed from the pluripotent epiblast through to specified PGCs without an intermediate silencing phase. Resolving this question in humans has been impossible due to the lack of access to appropriate embryonic tissues, and prior in vitro models have lacked the temporal resolution and purity to definitively test either hypothesis. Additionally, the identity of the immediate precursor to human PGCs—whether it arises from the posterior epiblast, the primitive streak, or the amnion—remains debated, with conflicting data from mouse, pig, and cynomolgus monkey models. This uncertainty has led to suboptimal differentiation protocols that fail to precisely recapitulate the in vivo signaling sequence.

Compounding these challenges is the absence of robust, specific cell-surface markers for human PGCLCs. Existing markers such as EpCAM, ITGA6, PDPN, and alkaline phosphatase activity are also expressed on undifferentiated hPSCs, making it impossible to distinguish true PGCLCs from residual pluripotent cells or to purify PGCLCs without genetic reporters. This lack of surface phenotype has severely limited the ability to isolate viable, functional PGCLCs for transplantation, omics profiling, or drug screening. Furthermore, while bulk RNA sequencing has suggested transcriptional similarity between in vitro-derived PGCLCs and fetal PGCs, single-cell resolution comparisons have been lacking, leaving open questions about the fidelity and homogeneity of current PGCLC models.

In light of these gaps, there exists a pressing need for a simplified, robust, and scalable platform for generating human PGCLCs that overcomes the limitations of 3D culture, enables precise temporal control of signaling pathways, provides a universal method for PGCLC purification without genetic modification, and yields cells that faithfully recapitulate the molecular identity of in vivo human PGCs. Such a platform would not only advance basic understanding of human germline development but also accelerate the development of infertility treatments, in vitro gametogenesis, and models for studying germ cell tumors and epigenetic inheritance.

## BRIEF SUMMARY OF THE INVENTION

The present invention provides a novel, simplified, and highly efficient two-dimensional (2D) monolayer method for the differentiation of human pluripotent stem cells (hPSCs) into primordial germ cell-like cells (PGCLCs) with unprecedented speed, reproducibility, and purity. Central to this invention is the discovery that temporally dynamic modulation of WNT signaling—specifically, a brief 12-hour activation followed by sustained inhibition—is critical for directing hPSCs toward the germline fate while suppressing alternative mesodermal lineages. This insight has enabled the development of a serum-free, chemically defined protocol that generates PGCLCs within 3.5 days of differentiation, significantly faster than prior 3D-based approaches.

A key innovation of the present invention is the identification of a unique cell-surface marker profile—CXCR4⁺PDGFRα⁻GARP⁻—that specifically identifies and enables the purification of human PGCLCs from heterogeneous differentiation cultures. Unlike previously reported markers, this tripartite signature effectively distinguishes PGCLCs from both undifferentiated hPSCs and off-target somatic cells, including mesodermal derivatives, without requiring genetic modification of the starting cell line. This surface phenotype is conserved across diverse hESC and hiPSC lines, regardless of sex or genetic background, and facilitates the isolation of >97% pure PGCLC populations via fluorescence-activated cell sorting (FACS).

Furthermore, the invention provides definitive molecular evidence that pluripotency transcription factors, particularly NANOG and OCT4, are continuously expressed throughout the transition from pluripotent epiblast to posterior epiblast intermediate to specified PGCLC, without an intermediate phase of downregulation. This finding resolves a longstanding controversy in the field and establishes a direct molecular continuum between pluripotency and germline identity in humans. Single-cell RNA sequencing (scRNA-seq) analyses confirm that the PGCLCs generated by this method exhibit a transcriptional profile nearly identical to that of early human fetal PGCs (specifically, the FGC1 population isolated from week 5–7 human fetuses), validating their physiological relevance.

The invention thus comprises: (1) a 2D monolayer differentiation protocol for generating human PGCLCs in 3.5 days using temporally controlled WNT activation (12 hours with CHIR99021) followed by WNT inhibition (with XAV939) and optimized concentrations of BMP4, SCF, and EGF; (2) a universal cell-surface marker signature (CXCR4⁺PDGFRα⁻GARP⁻) for the identification and purification of human PGCLCs; (3) the use of this purified PGCLC population for modeling human germline development, studying infertility, and developing reproductive technologies; and (4) the application of this platform to dissect the signaling dynamics and transcriptional networks underlying human PGC specification. The method is robust, scalable, and applicable to any wild-type hPSC line, making it suitable for both research and potential clinical applications.

## DETAILED DESCRIPTION OF THE INVENTION

### Definitions

As used herein, the term “human pluripotent stem cells” or “hPSCs” refers to cells derived from the human inner cell mass or reprogrammed somatic cells that possess the capacity for unlimited self-renewal and the potential to differentiate into all three germ layers (endoderm, mesoderm, and ectoderm). This includes both human embryonic stem cells (hESCs) and human induced pluripotent stem cells (hiPSCs), whether maintained in feeder-dependent or feeder-free conditions, and irrespective of their sex chromosome complement (XX or XY).

The term “primordial germ cell-like cells” or “PGCLCs” refers to in vitro-differentiated cells derived from hPSCs that molecularly and functionally resemble bona fide human primordial germ cells (PGCs) as found in early human embryos. PGCLCs are characterized by the expression of a core set of germline-specific transcription factors and markers, including but not limited to POU5F1 (OCT4), NANOG, PRDM1 (BLIMP1), TFAP2C (AP2γ), NANOS3, SOX17, and TFCP2L1, and by the absence of markers associated with somatic lineages such as endoderm (FOXA2, HHEX), mesoderm (BRACHYURY, HAND1, PDGFRα), and trophoblast (CDX2). PGCLCs generated by the methods of the present invention are further defined by their unique cell-surface marker profile CXCR4⁺PDGFRα⁻GARP⁻.

The term “posterior epiblast” refers to a transient intermediate cell state generated during the first 12 hours of hPSC differentiation under the influence of WNT and TGFβ signaling. This state is characterized by the co-expression of pluripotency factors (OCT4, NANOG) and early primitive streak/posterior epiblast markers (MIXL1, BRACHYURY, NODAL, FGF8), but at lower levels than those observed in fully committed primitive streak cells. The posterior epiblast is distinct from the primitive streak in that it retains germline competence and can be directed toward PGCLC fate upon subsequent WNT inhibition.

The term “temporally dynamic WNT signaling” refers to a precisely timed sequence of WNT pathway activation followed by inhibition. In the context of the present invention, this involves an initial 12-hour exposure to a WNT agonist (e.g., CHIR99021, a GSK3β inhibitor) to induce posterior epiblast formation, followed by sustained inhibition of WNT signaling (e.g., with XAV939, an AXIN stabilizer that promotes β-catenin degradation) for the remainder of the differentiation period to suppress mesodermal differentiation and promote PGCLC specification.

The term “CXCR4⁺PDGFRα⁻GARP⁻” refers to a specific cell-surface immunophenotype wherein cells express the chemokine receptor CXCR4 (CD184) and lack detectable expression of platelet-derived growth factor receptor alpha (PDGFRα, CD140a) and glycoprotein A repetitions predominant (GARP, LRRC32). This signature is used to identify and isolate human PGCLCs from mixed differentiation cultures via flow cytometry or FACS.

The term “monolayer culture” refers to a two-dimensional (2D) cell culture system in which cells are grown as a single layer on a flat, coated substrate (e.g., Matrigel-coated tissue culture plastic), as opposed to three-dimensional (3D) aggregates or organoids. Monolayer culture ensures uniform exposure to extracellular signals and facilitates precise temporal control of differentiation cues.

### Methods

The present invention provides a method for differentiating human pluripotent stem cells (hPSCs) into primordial germ cell-like cells (PGCLCs) in a two-dimensional (2D) monolayer culture system. The method comprises two sequential phases of differentiation, each defined by specific signaling modulators and time intervals.

In the first phase, undifferentiated hPSCs are exposed to a combination of a WNT pathway activator and a TGFβ pathway activator for a duration of approximately 12 hours. In a preferred embodiment, the WNT activator is CHIR99021, a selective inhibitor of glycogen synthase kinase-3 beta (GSK3β), administered at a concentration of 3 μM. The TGFβ activator is Activin A, administered at a concentration of 100 ng/mL. Additionally, a Rho-associated kinase (ROCK) inhibitor, such as Y-27632, is included at a concentration of 10 μM to enhance cell survival during the initial differentiation step. This treatment is carried out in a chemically defined basal medium, such as aRB27 medium, which consists of Advanced RPMI 1640 supplemented with 1% B27 supplement, 0.1 mM non-essential amino acids, 100 U/mL penicillin, 0.1 mg/mL streptomycin, and 2 mM L-glutamine. The 12-hour duration is critical; extension of this phase to 24 hours or longer results in the formation of primitive streak cells that are incompetent for subsequent PGCLC differentiation. At the end of this phase, the cells adopt a “posterior epiblast” identity, characterized by the co-expression of pluripotency factors (OCT4, NANOG) and early mesendodermal markers (MIXL1, BRACHYURY, NODAL), but at levels insufficient to commit to somatic lineages.

Following the first phase, the cells are washed to remove the initial signaling molecules and transitioned into the second phase of differentiation. This phase spans approximately 84 hours (3.5 days total differentiation time) and is divided into three sub-stages to optimize PGCLC yield. In the first 24 hours of the second phase (days 1–2), cells are treated with bone morphogenetic protein 4 (BMP4) at a concentration of 40 ng/mL, a WNT inhibitor such as XAV939 at 1 μM, and the ROCK inhibitor Y-27632 at 10 μM. In the next 24 hours (days 2–3), BMP4 is omitted, and cells are treated with stem cell factor (SCF) at 100 ng/mL, epidermal growth factor (EGF) at 50 ng/mL, XAV939 at 1 μM, and Y-27632 at 10 μM. In the final 24 hours (days 3–3.5), cells are treated with BMP4 (40 ng/mL), SCF (100 ng/mL), EGF (50 ng/mL), XAV939 (1 μM), and Y-27632 (10 μM). Notably, leukemia inhibitory factor (LIF), commonly used in prior protocols, is dispensable in this system and may be omitted without loss of efficiency. The use of XAV939 to actively inhibit WNT signaling is essential; merely withholding exogenous WNT agonists is insufficient, as differentiating hPSCs produce endogenous WNT ligands that would otherwise suppress PGCLC formation. The concentration of BMP4 used (40 ng/mL) is significantly lower than that required in 3D systems (200–500 ng/mL), reflecting the improved signal accessibility in monolayer culture.

This method consistently yields 20–30% PGCLCs in unsorted cultures, with peak efficiencies reaching up to 73.2% across diverse hPSC lines. The entire differentiation process is completed within 3.5 days, substantially faster than conventional 3D protocols that require 4–7 days.

### Cell Compositions

The present invention also provides novel cell compositions comprising purified human PGCLCs identified by the cell-surface marker profile CXCR4⁺PDGFRα⁻GARP⁻. These cells are generated by the aforementioned monolayer differentiation method and subsequently isolated via fluorescence-activated cell sorting (FACS) or magnetic-activated cell sorting (MACS) using antibodies against CXCR4, PDGFRα, and GARP.

The CXCR4⁺PDGFRα⁻GARP⁻ PGCLCs are characterized by the following molecular features: (1) high expression of germline-specific transcription factors including POU5F1 (OCT4), NANOG, PRDM1 (BLIMP1), TFAP2C (AP2γ), NANOS3, SOX17, and TFCP2L1; (2) absence of somatic lineage markers such as endodermal markers (FOXA2, HHEX, SOX7), mesodermal markers (BRACHYURY, HAND1, PDGFRα, MYL4, ACTC1), and extraembryonic markers (CDX2); (3) continuous expression of pluripotency factors from the undifferentiated hPSC state through the posterior epiblast intermediate to the specified PGCLC, without an intermediate downregulation phase; and (4) a transcriptional profile that closely matches that of early human fetal PGCs (FGC1 population) as determined by single-cell RNA sequencing.

These purified PGCLCs are >97% homogeneous, as confirmed by scRNA-seq of FACS-sorted populations, and are viable for downstream applications including transplantation, cryopreservation, omics analysis, and further differentiation into gamete precursors. The cell composition is stable for at least 72 hours post-sorting under appropriate culture conditions and maintains its germline identity without spontaneous differentiation into somatic lineages.

### METHODS OF USE

The PGCLCs and differentiation methods of the present invention have numerous applications in research, diagnostics, and therapeutics. In basic research, the monolayer platform enables high-resolution dissection of the signaling dynamics and transcriptional networks governing human PGC specification, including the role of WNT, BMP, and other pathways. The ability to generate synchronized, homogeneous PGCLC populations facilitates genome-wide studies such as ChIP-seq, ATAC-seq, and CRISPR screens to identify novel regulators of germline development.

In disease modeling, the method can be applied to hPSCs derived from patients with infertility or genetic disorders affecting germ cell development (e.g., Turner syndrome, Klinefelter syndrome, or mutations in NANOS3 or DAZL) to study the cellular and molecular basis of these conditions. Drug screening platforms can be developed using patient-derived PGCLCs to identify compounds that rescue defective germline specification.

For reproductive technologies, the purified PGCLCs serve as a starting material for in vitro gametogenesis—the generation of functional oocytes or spermatozoa in vitro. Although complete meiosis has not yet been achieved in human systems, the high fidelity and purity of the PGCLCs provided by this invention represent a critical advance toward this goal. Additionally, the CXCR4⁺PDGFRα⁻GARP⁻ signature allows for the quality control of PGCLC preparations intended for clinical use, ensuring the absence of residual pluripotent cells that could pose teratoma risk.

The method is also useful for toxicology studies, as germ cells are particularly sensitive to environmental toxins and endocrine disruptors. PGCLCs generated by this protocol can be exposed to candidate compounds to assess their impact on germline integrity and epigenetic programming.

### EXAMPLES

**Example 1: Optimization of WNT Signaling Dynamics**

Undifferentiated H1 hESCs were differentiated using a 12-hour pulse of 100 ng/mL Activin A and 3 μM CHIR99021, followed by 3.5 days of culture in the presence or absence of the WNT inhibitor XAV939 (1 μM). PGCLC formation was quantified using a NANOS3-mCherry reporter line. Cultures treated with XAV939 yielded 28.5 ± 3.2% NANOS3⁺ cells, whereas control cultures without XAV939 yielded only 9.7 ± 1.5% NANOS3⁺ cells (p < 0.001, n = 6). Continued exposure to CHIR99021 in the second phase completely abolished PGCLC formation, confirming the necessity of WNT inhibition after the initial pulse.

**Example 2: Identification of CXCR4⁺PDGFRα⁻GARP⁻ Signature**

SOX17-GFP hESCs were differentiated into PGCLCs using the optimized monolayer protocol. On day 3.5, cells were stained with a panel of 369 cell-surface antibodies and analyzed by high-throughput FACS. CXCR4 was the most specific positive marker for SOX17-GFP⁺ cells, while PDGFRα and GARP were highly expressed on SOX17-GFP⁻ non-PGCLCs. Sorting based on CXCR4⁺PDGFRα⁻GARP⁻ yielded a population that was 97.2% SOX17-GFP⁺ and 95.8% NANOS3⁺ by qPCR, compared to 32.1% in unsorted cultures.

**Example 3: Cross-Line Validation**

The monolayer protocol was applied to five independent hPSC lines (three hESCs and two hiPSCs, including both male and female lines). All lines generated CXCR4⁺PDGFRα⁻GARP⁻ PGCLCs with efficiencies ranging from 15.3% to 73.2% (mean 36.7 ± 22.6%). Sorted cells from all lines expressed canonical PGCLC markers and lacked somatic contamination, demonstrating the robustness of the method.

**Example 4: scRNA-seq Validation Against Fetal PGCs**

FACS-sorted CXCR4⁺PDGFRα⁻GARP⁻ PGCLCs were subjected to scRNA-seq and compared to published data from human fetal germ cells (Li et al., 2017). Hierarchical clustering and Pearson correlation analysis showed that in vitro PGCLCs clustered most closely with the FGC1 population (r = 0.89), which represents early migratory PGCs, and not with later-stage germ cells (FGC2–FGC4, r < 0.65).

## P EMBODIMENTS

### EMBODIMENTS

1. A method for differentiating human pluripotent stem cells (hPSCs) into primordial germ cell-like cells (PGCLCs), comprising:
   a) culturing hPSCs in a monolayer in a basal medium comprising a WNT activator and a TGFβ activator for 12 hours to generate posterior epiblast cells;
   b) replacing the medium with a second medium comprising a WNT inhibitor, BMP4, SCF, and EGF for 3.5 days to generate PGCLCs;
   wherein the WNT activator is CHIR99021 at 3 μM, the TGFβ activator is Activin A at 100 ng/mL, and the WNT inhibitor is XAV939 at 1 μM.

2. The method of embodiment 1, wherein the basal medium is aRB27 medium.

3. The method of embodiment 1, further comprising adding a ROCK inhibitor to both the first and second media.

4. The method of embodiment 1, wherein the concentration of BMP4 is 40 ng/mL.

5. The method of embodiment 1, wherein the PGCLCs are generated in 3.5 days.

6. A cell composition comprising human PGCLCs purified by fluorescence-activated cell sorting based on the cell-surface marker profile CXCR4⁺PDGFRα⁻GARP⁻.

7. The cell composition of embodiment 6, wherein the PGCLCs express POU5F1, NANOG, PRDM1, TFAP2C, NANOS3, and SOX17.

8. The cell composition of embodiment 6, wherein the PGCLCs do not express FOXA2, BRACHYURY, or CDX2.

9. The cell composition of embodiment 6, wherein the PGCLCs are >95% pure.

10. A method for treating infertility, comprising administering to a subject in need thereof the cell composition of embodiment 6.

11. A method for modeling human germ cell development, comprising differentiating hPSCs using the method of embodiment 1 and analyzing the resulting PGCLCs.

12. A kit for differentiating hPSCs into PGCLCs, comprising:
    a) a first medium containing CHIR99021, Activin A, and a ROCK inhibitor;
    b) a second medium containing XAV939, BMP4, SCF, EGF, and a ROCK inhibitor;
    c) instructions for use according to the method of embodiment 1.

13. The method of embodiment 1, wherein the hPSCs are selected from the group consisting of hESCs and hiPSCs.

14. The method of embodiment 1, wherein the posterior epiblast cells co-express OCT4 and BRACHYURY.

15. The cell composition of embodiment 6, wherein the PGCLCs are transcriptionally similar to human fetal PGCs of the FGC1 subtype.