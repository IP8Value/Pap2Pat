Below is the drafted patent application following the provided outline and research paper content. The application maintains formal patent language and adheres strictly to the specified structure:

---

# DESCRIPTION  

## BACKGROUND  

Lipids constitute a fundamental class of biomolecules that serve critical roles as energy storage compounds and structural components of cellular membranes. These hydrophobic molecules, comprising fatty acids, isoprenoids, and related hydrocarbon structures, have substantial commercial value as precursors for biofuels, specialty chemicals, and nutritional supplements. The biotechnological production of lipids through microbial fermentation presents a sustainable alternative to traditional plant- and animal-derived sources, particularly when utilizing low-cost carbon feedstocks such as agricultural residues or lignocellulosic biomass.  

Recent advances in metabolic engineering have demonstrated that lipid yields can be substantially increased through targeted genetic modifications that regulate key enzymes in lipid biosynthesis pathways. Among these, diacylglycerol acyltransferase proteins (DGATs) have emerged as pivotal catalysts in triacylglycerol (TAG) synthesis. Specifically, the DGA1 protein (classified as DGAT type 2) performs the terminal acylation step in TAG formation by transferring an acyl group from acyl-CoA to diacylglycerol. This enzymatic activity not only drives lipid accumulation but also facilitates the packaging of TAGs into lipid droplets for cellular storage.  

The DGA1 enzyme operates in coordination with other DGAT isoforms, including DGA2 (DGAT type 1) and DGA3 (DGAT type 3), which exhibit distinct substrate preferences and subcellular localizations. While DGA1 predominates on lipid body membranes, DGA2 localizes to the endoplasmic reticulum and contributes to nascent lipid droplet formation. The complementary functions of these enzymes create a synergistic system for TAG biosynthesis, though their relative contributions vary across species.  

Additional genetic factors influence lipid production, including triacylglycerol lipases (TGLs) that catalyze TAG degradation. In Yarrowia lipolytica, the TGL3 lipase regulates lipid mobilization during nutrient limitation, making its suppression beneficial for maintaining high lipid content during fermentation. Other enzymes in the lipid pathway—such as malic enzyme (ME), ATP citrate lyase (ACL), and glycerol-3-phosphate dehydrogenase (GPD1)—also modulate carbon flux toward lipid accumulation.  

## SUMMARY  

The present invention provides a transformed microbial cell comprising genetic modifications that significantly enhance lipid production. In one embodiment, the cell contains: (1) a first genetic modification that increases the activity of a DGA1 protein; (2) a second genetic modification that increases the activity of a DGA2 protein; and optionally (3) a third genetic modification that decreases the activity of a triacylglycerol lipase, such as TGL3.  

The first genetic modification involves introducing an exogenous nucleotide sequence encoding a DGA1 protein, preferably derived from Rhodosporidium toruloides or Lipomyces starkeyi, under the control of a constitutive promoter (e.g., GPD1). The second genetic modification comprises introducing a nucleotide sequence encoding a DGA2 protein, preferably from Claviceps purpurea or Chaetomium globosum. The optional third modification may include a knockout mutation of the TGL3 gene or replacement of its native promoter with a less active variant.  

The invention further provides methods for increasing triacylglycerol content in a microbial cell. One such method comprises: (a) transforming a host cell with a first nucleotide sequence encoding a DGA1 protein and a second nucleotide sequence encoding a DGA2 protein; (b) culturing the transformed cell under conditions suitable for lipid accumulation; and (c) optionally recovering the triacylglycerols. The DGA1 and DGA2 proteins may be derived from different species, and their coding sequences may be codon-optimized for expression in the host cell (e.g., Yarrowia lipolytica or Arxula adeninivorans).  

In alternative embodiments, the method may utilize a single genetic modification—such as overexpression of DGA1 or DGA2 alone—or combine DGA1/DGA2 overexpression with suppression of TAG degradation pathways. The resulting engineered strains achieve lipid contents exceeding 70% of dry cell weight, with fed-batch productivities surpassing 0.7 g/L/h.  

## DETAILED DESCRIPTION  

### Overview  

The invention capitalizes on coordinated overexpression of diacylglycerol acyltransferases (DGA1 and DGA2) to maximize triacylglycerol (TAG) production in microbial hosts. DGA1 overexpression creates a driving force for lipid droplet expansion by enhancing the final acylation step in TAG synthesis. Concurrent DGA2 expression ensures efficient initiation of lipid droplet formation in the endoplasmic reticulum. To further augment lipid retention, the invention optionally incorporates suppression of TGL3, a lipase that mobilizes stored TAGs during nutrient limitation.  

### Definitions  

As used herein:  
- "Activity" refers to the catalytic function of a protein, such as the acyltransferase activity of DGA1.  
- "Biologically-active portion" denotes a fragment of a protein (e.g., DGA1) that retains at least 80% of the reference protein's activity.  
- "Diacylglycerol acyltransferase" (DGA) encompasses enzymes classified as DGAT1 (DGA2), DGAT2 (DGA1), or DGAT3 (DGA3).  
- "Dry cell weight" (DCW) is the biomass weight after removal of all water content.  
- "Exogenous gene" refers to a nucleic acid introduced into a host cell via genetic engineering.  
- "Inducible promoter" drives transcription in response to a specific stimulus (e.g., chemical or thermal).  
- "Knockout mutation" renders a gene nonfunctional through deletion or disruption.  

### Microbe Engineering  

The invention employs standard microbial engineering techniques to construct high-lipid strains. Suitable host cells include oleaginous yeasts (e.g., Yarrowia lipolytica, Arxula adeninivorans) and fungi. Expression cassettes for DGA1/DGA2 genes are assembled using modular vectors containing constitutive promoters (e.g., GPD1, TEF1), terminators, and selectable markers (e.g., nourseothricin resistance). Homologous recombination integrates these cassettes into the host genome, though autonomous plasmids may also be used.  

For TGL3 suppression, knockout constructs are designed with flanking homology regions targeting the TGL3 locus. Transformation efficiency is enhanced by hydroxyurea treatment, which synchronizes cells in the S phase. Alternatively, CRISPR/Cas9 systems enable precise gene editing.  

### Exemplary Nucleic Acids, Cells, and Methods  

**DGA1 Constructs**:  
The R. toruloides DGA1 gene (SEQ ID NO: 1) and its codon-optimized variant (SEQ ID NO: 2) are cloned into expression vector pNC243 under the GPD1 promoter. Substantially identical sequences (>90% amino acid identity) from L. starkeyi (SEQ ID NO: 3) or A. limacinum (SEQ ID NO: 4) may also be used.  

**DGA2 Constructs**:  
The C. purpurea DGA2 gene (SEQ ID NO: 5) is inserted into vector pNC327. Homologs from C. globosum (SEQ ID NO: 6) sharing ≥85% identity retain comparable activity.  

**TGL3 Knockout**:  
A deletion cassette replaces the TGL3 coding sequence with a hygromycin resistance marker (SEQ ID NO: 7).  

**Transformed Cells**:  
Engineered Y. lipolytica strain NS432 (genotype: RtDGA1_CpDGA2_Δtgl3) accumulates 85 g/L lipids in fed-batch fermentation. Arxula adeninivorans strains expressing DGA1/DGA2 show similar improvements.  

### Methods  

1. **Dual Overexpression**:  
   - Transform Y. lipolytica with pNC243 (DGA1) and pNC327 (DGA2).  
   - Select transformants on nourseothricin/zeocin media.  
   - Cultivate in nitrogen-limited glucose medium (pH 5.5) at 30°C.  

2. **TGL3 Knockout**:  
   - Co-transform with TGL3 deletion cassette and DGA1/DGA2 vectors.  
   - Verify knockouts via PCR (primers SEQ ID NOs: 8–9).  

3. **Lipid Recovery**:  
   - Harvest cells by centrifugation.  
   - Extract lipids using chloroform/methanol (2:1 v/v).  
   - Transesterify to fatty acid methyl esters for GC analysis.  

## EXEMPLIFICATION  

### Example 1: DGA1 Overexpression in Y. lipolytica  

Strain NS281 (RtDGA1) showed a 3-fold lipid increase versus wild-type (77% DCW). Fluorescence assays (ex. 486 nm/em. 510 nm) confirmed enhanced Bodipy staining.  

### Example 2: Combined DGA1/DGA2 Expression  

Strain NS432 (RtDGA1_CpDGA2_Δtgl3) achieved 85 g/L lipid titer in fed-batch fermentation (0.73 g/L/h productivity). GC analysis revealed predominant C16–C18 fatty acids.  

### Example 3: TGL3 Knockout Effect  

NS377 (RtDGA1_Δtgl3) retained 15% more lipids than NS281 after 140 h cultivation, demonstrating reduced TAG degradation.  

## EQUIVALENTS  

The invention encompasses variants of exemplified sequences, including DGA1/DGA2 homologs with ≥80% identity and alternative lipase suppression strategies (e.g., siRNA).  

--- 

This application provides comprehensive coverage of the engineered strains, methods, and industrial applications while adhering to patent drafting conventions. Let me know if you'd like any modifications or expansions.