Here is the complete patent application following the provided outline and research paper:

# DESCRIPTION  

## TECHNICAL DOMAIN  

The present invention relates to the field of industrial enzyme production, specifically to genetically modified strains of the filamentous fungus Trichoderma reesei and processes for the industrial-scale production of cellulolytic enzymes. More particularly, the invention concerns a recombinant T. reesei strain engineered through targeted genetic modifications to achieve superior cellulase production capabilities when cultivated on low-cost substrates, along with an optimized fermentation process utilizing sugarcane molasses as the primary carbon source.  

The invention addresses several critical challenges in industrial enzyme manufacturing, including: (1) the high production costs associated with conventional inducers and medium components; (2) limitations in enzyme titers and productivity from existing production strains; (3) deficiencies in β-glucosidase activity in native T. reesei enzyme cocktails; and (4) the inability of standard production strains to efficiently utilize sucrose-rich feedstocks. The disclosed invention provides solutions to these challenges through a combination of strategic genetic modifications and process optimizations, resulting in unprecedented enzyme production levels exceeding 80 g/L with significantly improved saccharification efficiency.  

## PRIOR ART  

Industrial production of cellulases has traditionally relied on strains of Trichoderma reesei, particularly derivatives of the QM6a isolate developed through multiple rounds of mutagenesis and selection. The RUT-C30 strain represents one such publicly available hyperproducing variant, featuring an 85-kb chromosomal deletion and a truncated cre1 gene that partially relieves carbon catabolite repression. While RUT-C30 has served as a workhorse for both academic research and industrial applications, its enzyme production capacity remains substantially below that of proprietary industrial strains, which have been reported to achieve titers exceeding 100 g/L through undisclosed genetic modifications and optimized fermentation processes.  

Previous attempts to enhance cellulase production in T. reesei have focused on various genetic targets, including:  

1. Modification of transcription factors such as XYR1, the major activator of cellulase gene expression, where constitutive expression and specific point mutations (V821F, A824V) have been shown to increase enzyme production and relieve glucose-mediated repression.  

2. Deletion of transcriptional repressors including ACE1, whose silencing has been demonstrated to further boost cellulase expression when combined with XYR1 overexpression.  

3. Removal of extracellular proteases such as SLP1 (subtilisin protease) and PEP1 (aspartic protease), which has been associated with increased protein secretion stability.  

4. Heterologous expression of complementary enzymatic activities, particularly β-glucosidases to address the well-documented deficiency in T. reesei secretomes.  

5. Introduction of invertase activity to enable sucrose utilization, through expression of genes such as suc1 from Aspergillus niger.  

Despite these individual advances, the combined effects of multiple genetic modifications have rarely been investigated due to technical challenges in fungal genetic engineering. Moreover, even when beneficial modifications have been identified, their implementation in industrial strains has typically resulted in only incremental improvements in enzyme productivity, with reported titers for engineered strains generally remaining below 40 g/L in published studies.  

Concurrently, significant efforts have been directed toward reducing production costs through the use of alternative carbon sources. Sugarcane molasses represents an attractive low-cost substrate due to its high sugar content (approximately 740 g/L total sugars) and abundance of vitamins and minerals. However, native T. reesei strains cannot utilize the sucrose component (approximately 35% w/w) of molasses due to their lack of invertase activity, limiting the economic viability of molasses-based processes.  

The present invention overcomes these limitations through the rational combination of multiple genetic modifications in a publicly available strain background, coupled with an optimized fermentation process, to achieve enzyme production levels that surpass all previously reported values in the scientific literature while utilizing cost-effective feedstocks.  

## SUMMARY OF THE INVENTION  

The present invention provides a genetically modified strain of Trichoderma reesei, derived from the publicly available RUT-C30 strain, that exhibits dramatically improved cellulase production capabilities when cultivated on low-cost substrates. The engineered strain, designated BR_TrR03, incorporates six strategic genetic modifications implemented through three rounds of CRISPR/Cas9-mediated engineering:  

1. Replacement of the slp1 gene (encoding a subtilisin protease) with a heterologous β-glucosidase gene (cel3a from Talaromyces emersonii) under control of the xyn1 promoter;  

2. Replacement of the ace1 gene (encoding a transcriptional repressor) with a constitutively expressed, mutated version of xyr1 (V821F allele) under control of the pdc1 promoter;  

3. Replacement of the pep1 gene (encoding an aspartic protease) with the suc1 gene from Aspergillus niger, including its native regulatory sequences, to confer invertase activity.  

These modifications collectively address four key aspects of cellulase production: (i) enabling utilization of sucrose-rich feedstocks; (ii) increasing transcription of genes encoding secreted enzymes; (iii) reducing proteolytic degradation of secreted proteins; and (iv) enhancing the β-glucosidase activity of the enzyme cocktail.  

The invention further provides an optimized fermentation process utilizing sugarcane molasses as the primary carbon source and molasses-grown yeast cells as an organic nitrogen source. This molasses-based medium supports extracellular protein titers exceeding 80 g/L (0.24 g/L/h) in fed-batch bioreactor cultivations - the highest experimentally demonstrated titer reported for T. reesei in the scientific literature.  

The enzyme cocktail produced by the BR_TrR03 strain exhibits significantly improved specific activities compared to the parental RUT-C30 strain, particularly for β-glucosidase (72-fold increase) and xylanase (42-fold increase). Saccharification assays with industrially pretreated biomass demonstrate that the BR_TrR03 cocktail performs equivalently to commercial enzyme preparations, while avoiding the cellobiose accumulation characteristic of native T. reesei enzymes.  

## DESCRIPTION OF THE EMBODIMENTS  

The present invention encompasses genetically modified strains of Trichoderma reesei exhibiting enhanced cellulase production capabilities, methods for constructing such strains, and processes for cultivating the strains to produce cellulolytic enzyme compositions.  

### Strain Construction  

The BR_TrR03 strain was derived from T. reesei RUT-C30 (IHEM_5652/ATCC_56765) through three sequential rounds of genetic modification using a single-plasmid CRISPR/Cas9 system and markerless donor cassettes.  

The CRISPR/Cas9 system employed a Streptococcus pyogenes Cas9 gene codon-optimized for expression in T. reesei, under control of the pdc1 promoter. Guide RNAs were expressed using ribozyme-mediated cassettes with hammerhead (5') and hepatitis delta virus (3') ribozyme sequences. Target-specific plasmids were constructed by inserting 20-nt protospacer sequences designed to direct Cas9 cleavage to specific genomic loci.  

For each modification round, linear donor DNA cassettes containing 1 kb homology arms flanking the targeted genomic regions were generated. The cassettes were assembled in Saccharomyces cerevisiae through in vivo homologous recombination of PCR-amplified fragments, including:  

1. For the first modification: The cel3a β-glucosidase gene from Talaromyces emersonii, under control of the xyn1 promoter, flanked by sequences homologous to regions upstream and downstream of the slp1 locus.  

2. For the second modification: The xyr1-V821F allele under control of the pdc1 promoter, flanked by sequences homologous to regions upstream and downstream of the ace1 locus.  

3. For the third modification: The suc1 gene from Aspergillus niger including its native regulatory sequences, flanked by sequences homologous to regions upstream and downstream of the pep1 locus.  

Protoplast transformations were performed by co-delivery of the appropriate CRISPR/Cas9 vector and donor cassette. Transformants were selected on hygromycin-containing media and subjected to multiple rounds of isolation and PCR verification to obtain stable, homokaryotic, marker-free strains.  

### Strain Characteristics  

The completed BR_TrR03 strain exhibits the following genotypic and phenotypic characteristics:  

1. Deletion of the native slp1 gene and insertion of the heterologous cel3a β-glucosidase gene under control of the xyn1 promoter, resulting in a 4-fold increase in specific β-glucosidase activity compared to RUT-C30 (0.74 vs 0.18 IU/mg in lactose medium).  

2. Deletion of the native ace1 gene and insertion of the constitutively expressed xyr1-V821F allele, leading to a 3-fold increase in extracellular protein production (11.3 vs 3.6 g/L in lactose medium) and a 26-fold increase in specific β-glucosidase activity (4.6 vs 0.18 IU/mg) compared to RUT-C30.  

3. Deletion of the native pep1 gene and insertion of the A. niger suc1 gene, enabling growth on sucrose-containing media and further improving protein secretion in bioreactor cultivations.  

The strain maintains all beneficial mutations present in the parental RUT-C30 strain, including the 85-kb chromosomal deletion and truncated cre1 allele that partially relieves carbon catabolite repression.  

### Fermentation Process  

The invention further provides an optimized fermentation process for cellulase production using the BR_TrR03 strain. Key aspects of the process include:  

1. **Medium Composition**:  
   - Carbon source: Sugarcane molasses containing approximately 740 g/L total sugars (35% w/w sucrose)  
   - Nitrogen sources:  
     - Inorganic: Ammonium sulfate (20-30 g/L)  
     - Organic: Molasses-grown whole yeast cells (equivalent to 20 g/L yeast extract)  
   - Minerals: KH2PO4 (15 g/L), MgSO4 (0.59 g/L), CaCl2 (0.45 g/L), trace metals  
   - Antifoam: 1 mL/L J647 antifoam  

2. **Fermentation Parameters**:  
   - Mode: Fed-batch  
   - Temperature: 28°C  
   - pH: Controlled at 4.8 using ammonia and phosphoric acid  
   - Aeration: Maintained to keep dissolved oxygen above 30% saturation  
   - Agitation: Adjusted to maintain oxygen transfer  
   - Feed: Sugarcane molasses delivered at 1.0 g total sugars/L/h from 44-336 h  

3. **Performance Metrics**:  
   - Extracellular protein titer: 80.6 g/L in 336 h  
   - Productivity: 0.24 g/L/h  
   - Specific enzymatic activities (compared to RUT-C30):  
     - β-glucosidase: 72-fold increase  
     - Xylanase: 42-fold increase  
     - Cellobiohydrolase: 5-fold increase  
     - Endoglucanase: 4-fold increase  

### Enzyme Cocktail Characteristics  

The cellulolytic enzyme cocktail produced by BR_TrR03 exhibits the following properties:  

1. **Composition**: Contains all major cellulase and hemicellulase activities, including:  
   - Cellobiohydrolases (CBHI/II)  
   - Endoglucanases  
   - β-glucosidases (native and heterologous Cel3a)  
   - Xylanases  
   - β-xylosidases  

2. **Performance**:  
   - Saccharification efficiency equivalent to commercial CTec2 preparation  
   - No cellobiose accumulation during biomass hydrolysis  
   - Effective at high solids loading (20% w/w)  
   - Maintains activity when used as whole fermentation broth  

3. **Applications**: Suitable for industrial processes including:  
   - Lignocellulosic biomass hydrolysis for biofuel production  
   - Textile and pulp/paper processing  
   - Food and feed industries  
   - Waste treatment  

### Examples  

**Example 1: Strain Construction and Verification**  

The BR_TrR03 strain was constructed through three sequential rounds of CRISPR/Cas9-mediated engineering of T. reesei RUT-C30:  

1. **First Round**: The slp1 locus was replaced with the T. emersonii cel3a gene under control of the xyn1 promoter, generating strain BR_TrR01. PCR verification confirmed correct integration using primers flanking the slp1 locus and internal to the cel3a gene.  

2. **Second Round**: The ace1 locus was replaced with the xyr1-V821F allele under control of the pdc1 promoter in BR_TrR01, generating BR_TrR02. Integration was verified by PCR across the modified locus and sequencing of the xyr1-V821F allele.  

3. **Third Round**: The pep1 locus was replaced with the A. niger suc1 gene including native regulatory sequences in BR_TrR02, generating the final BR_TrR03 strain. Correct integration was confirmed by PCR and sequencing.  

All intermediate and final strains were purified to homokaryotic state through multiple rounds of spore isolation and verified to be free of CRISPR/Cas9 plasmids by growth on hygromycin-containing media.  

**Example 2: Shake Flask Performance Evaluation**  

The parental RUT-C30 and engineered strains were cultivated in shake flasks containing medium with 50 g/L of lactose, glucose or sucrose:  

1. **Protein Production**:  
   - BR_TrR03 produced 11.2 g/L protein in lactose medium (vs 3.6 g/L for BR_TrR01 and 1.2 g/L for RUT-C30)  
   - Similar titers were achieved in glucose (9.5 g/L) and sucrose (10.8 g/L) media, demonstrating CCR relief and sucrose utilization capabilities  

2. **β-glucosidase Activity**:  
   - Specific activity in lactose medium:  
     - RUT-C30: 0.18 IU/mg  
     - BR_TrR01: 0.74 IU/mg  
     - BR_TrR02: 4.6 IU/mg  
     - BR_TrR03: 4.5 IU/mg  

**Example 3: Bioreactor Cultivations with Inducer-Rich Medium**  

Fed-batch bioreactor cultivations were conducted with RUT-C30 and BR_TrR03 using an inducer-rich medium containing Avicel and lactose:  

1. **RUT-C30**: Reached 12.5 g/L protein in 168 h (0.07 g/L/h)  
2. **BR_TrR03**: Achieved 34.9 g/L protein in 168 h (0.21 g/L/h) - a 2.8-fold increase  

**Example 4: Molasses-Based Process Development**  

Fed-batch bioreactor cultivations with BR_TrR03 using sugarcane molasses-based media:  

1. **Molasses + Ammonium Sulfate**: 25.0 g/L protein in 168 h (0.15 g/L/h)  
2. **Molasses + Yeast Extract**: 46.8 g/L in 168 h (0.28 g/L/h)  
3. **Molasses + Molasses-Grown Yeast (MMGY)**:  
   - 42.9 g/L in 168 h (0.26 g/L/h)  
   - Extended to 336 h: 80.6 g/L final titer (0.24 g/L/h)  

Comparative experiments with BR_TrR02 (lacking suc1) using acid-inverted molasses showed lower productivity (59.2 g/L in 310 h, 0.19 g/L/h), demonstrating the advantage of the complete BR_TrR03 genotype.  

**Example 5: Enzyme Cocktail Characterization**  

Enzyme cocktails from BR_TrR03 and RUT-C30 were compared for specific activities and saccharification efficiency:  

1. **Specific Activities (BR_TrR03 MMGY vs RUT-C30)**:  
   - β-glucosidase: 72-fold higher  
   - Xylanase: 42-fold higher  
   - Cellobiohydrolase: 5-fold higher  
   - Endoglucanase: 4-fold higher  
   - Filter paper activity: 3-fold higher  

2. **Saccharification of Pretreated Sugarcane Straw (20% solids)**:  
   - BR_TrR03 cocktail released significantly more glucose and total sugars than RUT-C30  
   - Performance matched commercial CTec2 preparation  
   - No cellobiose accumulation observed with BR_TrR03  
   - Whole broth performed equivalently to processed cocktail  

These examples demonstrate the superior performance of the BR_TrR03 strain and associated fermentation process across multiple metrics relevant to industrial enzyme production.