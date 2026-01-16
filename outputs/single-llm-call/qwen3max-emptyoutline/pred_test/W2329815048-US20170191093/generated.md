# DESCRIPTION

## BACKGROUND

The production of lipids through microbial fermentation has emerged as a promising alternative to conventional plant- and animal-based sources, offering the potential for sustainable, scalable, and economically viable routes to biofuels, specialty chemicals, and bulk industrial feedstocks. Among microbial hosts, oleaginous yeasts—organisms capable of accumulating lipids at levels exceeding 20% of their dry cell weight—have garnered significant attention due to their metabolic versatility, robustness under industrial fermentation conditions, and amenability to genetic manipulation. Of particular interest is *Yarrowia lipolytica*, a well-characterized yeast that natively accumulates triacylglycerols (TAGs) up to 36% of its dry weight when grown on glucose and can reach lipid contents of 50–60% when supplemented with exogenous fatty acids. This organism not only serves as an efficient platform for lipid biosynthesis but also possesses the capacity to secrete valuable metabolites such as citric acid and polyols, and to express extracellular enzymes under controlled fermentation regimes.

Lipids in oleaginous microorganisms primarily consist of fatty acids esterified into neutral storage forms, most notably TAGs, which are sequestered within intracellular organelles known as lipid droplets. The biosynthesis of TAGs involves a multi-step enzymatic cascade that channels carbon from central metabolism—particularly from acetyl-CoA derived from glycolysis or alternative carbon sources—into fatty acid synthesis and subsequent esterification onto a glycerol backbone. Key regulatory nodes in this pathway include malic enzyme (ME), ATP citrate lyase (ACL), acetyl-CoA carboxylase (ACC), glycerol-3-phosphate acyltransferases (such as SCT and SLC1), stearoyl-CoA desaturase (SCD), and phospholipid:diacylglycerol acyltransferase (LRO). However, the terminal and rate-determining step in TAG assembly is catalyzed by diacylglycerol acyltransferases (DGATs), which transfer a third fatty acyl group from acyl-CoA onto diacylglycerol (DAG) to form TAG. In *Y. lipolytica*, two distinct DGAT enzymes have been identified: DGA1 (a type 2 DGAT, DGAT2 family) and DGA2 (a type 1 DGAT, DGAT1 family). These enzymes differ not only in sequence homology and subcellular localization but also in functional roles—DGA1 is associated with the expansion of pre-existing lipid droplets, while DGA2 is implicated in the de novo formation of nascent lipid bodies in the endoplasmic reticulum.

Genetic engineering efforts aimed at enhancing lipid titers in *Y. lipolytica* have consistently targeted the overexpression of DGA1 and DGA2, with numerous studies demonstrating that elevated DGAT activity correlates strongly with increased lipid accumulation. For instance, overexpression of native *DGA1* has been shown to significantly boost TAG content, and deletion of *DGA2* impairs lipid synthesis, underscoring the non-redundant and complementary functions of these enzymes. Moreover, heterologous expression of DGAT genes from other high-lipid-producing organisms—such as *Rhodosporidium toruloides* and *Lipomyces starkeyi*—has further expanded the toolkit for metabolic optimization. Concurrently, strategies to minimize lipid catabolism have focused on disrupting genes encoding triacylglycerol lipases, particularly *TGL3* and *TGL4*, which mediate the hydrolysis of stored TAGs during nutrient limitation or stationary phase. While early-stage fermentations may not reveal the full impact of lipase deletions, late-phase analyses demonstrate that abrogating TAG degradation pathways preserves accumulated lipids, thereby increasing final product yield.

Despite these advances, achieving industrially relevant lipid titers, yields, and productivities requires a systems-level approach that integrates multiple genetic modifications—simultaneous overexpression of synergistic biosynthetic enzymes, strategic deletion of catabolic genes, and fine-tuning of expression through promoter selection and codon optimization. Prior art has largely addressed these elements in isolation or in limited combinations, often failing to achieve the theoretical maximum lipid yield predicted by stoichiometric models (~0.276 g lipid per g glucose in *Y. lipolytica*). Furthermore, the performance of heterologous DGAT enzymes in *Y. lipolytica* remains underexplored, with insufficient data on how enzyme origin, gene structure (e.g., presence of introns), and codon usage influence functional expression and catalytic efficiency. There exists, therefore, a critical need for a comprehensive strain engineering framework that systematically evaluates endogenous and heterologous genetic targets, identifies optimal combinatorial configurations, and validates performance under scalable bioreactor conditions. The present invention addresses this unmet need by providing engineered microbial cells—particularly *Y. lipolytica* strains—that exhibit unprecedented lipid accumulation through the coordinated enhancement of DGA1 and DGA2 activities coupled with the suppression of TAG degradation via *TGL3* knockout, resulting in strains capable of producing lipids at titers exceeding 85 g/L with volumetric productivities surpassing 0.7 g/L/h.

## SUMMARY

The present invention provides genetically modified microbial cells, methods for their construction, and processes for the high-yield production of triacylglycerols (TAGs) using these cells. Specifically, the invention centers on the synergistic overexpression of diacylglycerol acyltransferase (DGAT) enzymes—namely DGA1 and DGA2—from both endogenous and heterologous sources, combined with the targeted deletion of triacylglycerol lipase genes, particularly *TGL3*, to minimize lipid catabolism during late-stage fermentation. The engineered cells, exemplified by *Yarrowia lipolytica* strain NS432, demonstrate markedly enhanced lipid accumulation, achieving lipid contents of up to 77% of dry cell weight in batch culture and producing 85 g/L of lipids in fed-batch bioreactors with a productivity of 0.73 g/L/h and a yield of 0.20–0.21 g lipid per g glucose consumed.

In one aspect, the invention provides a recombinant microbial cell comprising: (i) a heterologous nucleic acid encoding a DGA1 polypeptide derived from *Rhodosporidium toruloides*, operably linked to a strong constitutive promoter; (ii) a heterologous nucleic acid encoding a DGA2 polypeptide derived from *Claviceps purpurea*, operably linked to a strong constitutive promoter; and (iii) a deletion of the endogenous *TGL3* gene, wherein the cell exhibits increased triacylglycerol accumulation compared to a parental strain lacking said genetic modifications. The DGA1 and DGA2 polypeptides may be encoded by intron-free synthetic genes, optionally codon-optimized for expression in the host microorganism, though codon optimization is not required for enhanced function.

In another aspect, the invention encompasses methods for producing lipids by culturing such engineered cells in a fermentation medium containing a carbon source, such as glucose, under conditions conducive to lipid biosynthesis—typically nitrogen-limited environments that trigger the oleaginous phenotype. The fermentation may be conducted in batch or fed-batch mode, with fed-batch processes enabling higher cell densities and lipid titers. Lipids are recovered from the biomass via standard extraction and transesterification procedures, yielding fatty acid methyl esters (FAMEs) suitable for use as biodiesel or chemical feedstocks.

The invention further includes isolated nucleic acids encoding the disclosed DGA1 and DGA2 polypeptides, expression vectors containing these nucleic acids under the control of yeast-compatible promoters (e.g., *GPD1*, *TEF1*, or *EXP1*), and methods for transforming oleaginous yeasts—such as *Y. lipolytica*, *Arxula adeninivorans*, or *R. toruloides*—with these constructs. Additionally, the invention provides screening assays based on fluorescence detection of neutral lipids (e.g., using BODIPY 493/503 dye) to rapidly identify high-performing transformants, as well as gas chromatography protocols for quantitative lipid analysis.

Critically, the invention demonstrates that the combination of *R. toruloides* DGA1 and *C. purpurea* DGA2 overexpression, together with *TGL3* deletion, produces a non-additive, synergistic increase in lipid accumulation that exceeds the sum of individual modifications. This synergy arises from the complementary roles of DGA1 and DGA2 in lipid droplet biogenesis and expansion, coupled with the prevention of late-stage TAG mobilization. The resulting strains represent a significant advance over prior art, achieving among the highest reported lipid titers, yields, and productivities in *Y. lipolytica*, and establishing a new benchmark for microbial lipid production platforms.

## DETAILED DESCRIPTION

### Overview

The present invention is directed to engineered microbial cells optimized for the high-efficiency production of triacylglycerols (TAGs), with a focus on oleaginous yeasts, particularly *Yarrowia lipolytica*. The core innovation lies in a multi-pronged genetic strategy that simultaneously enhances the biosynthetic capacity for TAG assembly and suppresses its degradation. This is accomplished through the coordinated overexpression of two diacylglycerol acyltransferase (DGAT) enzymes—DGA1 and DGA2—derived from heterologous, high-lipid-producing organisms, in conjunction with the targeted disruption of the *TGL3* gene, which encodes a key triacylglycerol lipase involved in lipid mobilization during nutrient stress. The resulting strains exhibit dramatically elevated lipid contents, high volumetric productivities, and improved yields, making them suitable for industrial-scale production of lipids for biofuels, oleochemicals, and nutraceuticals.

The engineering approach begins with the identification of superior DGAT variants through systematic screening of endogenous and heterologous genes. Native *Y. lipolytica* DGA1 overexpression serves as a baseline, but heterologous DGA1 genes from organisms such as *Rhodosporidium toruloides* and *Lipomyces starkeyi*—which naturally accumulate lipids at >50% of dry weight—confer even greater enhancements. Similarly, among DGA2 candidates, *Claviceps purpurea* DGA2 emerges as the most effective when co-expressed with a potent DGA1. The deletion of *TGL3* does not significantly affect lipid levels during active growth but becomes crucial in the stationary phase, where it prevents the catabolism of stored TAGs, thereby preserving final lipid yield. The integration of these three modifications—heterologous DGA1, heterologous DGA2, and *TGL3* knockout—into a single strain yields a biocatalyst (e.g., NS432) that outperforms all intermediate constructs and sets new performance records in *Y. lipolytica*.

The invention is not limited to *Y. lipolytica*; the same principles apply to other oleaginous microbes, including *Arxula adeninivorans*, as demonstrated herein. The genetic constructs are modular, allowing for promoter swapping, copy number variation, and combinatorial testing. Cultivation is performed under nitrogen-limited, carbon-rich conditions to induce the oleaginous state, with fed-batch strategies employed to achieve high cell densities and lipid titers. Lipid quantification is validated through both high-throughput fluorescence assays and rigorous gas chromatography, ensuring robust correlation between screening and production metrics.

### Definitions

As used herein, the term “diacylglycerol acyltransferase” or “DGAT” refers to any enzyme that catalyzes the transfer of a fatty acyl group from acyl-CoA to diacylglycerol to form triacylglycerol. DGAT enzymes are classified into two evolutionarily distinct families: DGAT1 (type 1) and DGAT2 (type 2). In *Yarrowia lipolytica*, DGA2 is a DGAT1-family enzyme, while DGA1 belongs to the DGAT2 family.

The term “DGA1” denotes a DGAT2-family polypeptide, whether of endogenous or heterologous origin, that functions in the final step of TAG biosynthesis. Similarly, “DGA2” refers to a DGAT1-family polypeptide. The terms encompass full-length proteins, functional fragments, and variants exhibiting at least 70% amino acid sequence identity to the reference sequences described herein (e.g., *R. toruloides* DGA1 or *C. purpurea* DGA2).

“Heterologous nucleic acid” means a DNA or RNA sequence that is not native to the host cell or is present in a non-native genomic context. This includes synthetic genes, codon-optimized sequences, and genes derived from other species.

“Operably linked” describes a configuration wherein a promoter is positioned relative to a coding sequence such that transcription of the sequence is under the control of the promoter.

“Triacylglycerol lipase” or “TGL” refers to enzymes that hydrolyze TAGs into diacylglycerol and free fatty acids. *TGL3* in *Y. lipolytica* is a lipid droplet-associated lipase that, while potentially lacking canonical lipase motifs, regulates TAG turnover, especially during late fermentation.

“Oleaginous microorganism” is a microbe capable of accumulating lipids to at least 20% of its dry cell weight under appropriate culture conditions.

“Lipid content” is expressed as the percentage of total lipids (typically measured as fatty acid methyl esters, FAMEs) per unit dry cell weight (DCW).

“Productivity” refers to the rate of lipid production, expressed as grams of lipid per liter per hour (g/L/h).

“Yield” is the mass of lipid produced per mass of carbon source consumed (e.g., g lipid/g glucose).

### Microbe Engineering

The microbial hosts of the invention are engineered through recombinant DNA techniques to introduce specific genetic modifications that enhance lipid biosynthesis and reduce degradation. The primary host is *Yarrowia lipolytica*, selected for its native oleaginous capacity, well-developed genetic tools, and industrial robustness. However, the methods are equally applicable to other oleaginous yeasts and fungi, such as *Arxula adeninivorans*, *Rhodosporidium toruloides*, and *Lipomyces starkeyi*.

Engineering is achieved by integrating expression cassettes into the host genome via homologous recombination or random integration. Each cassette consists of a strong constitutive or inducible promoter (e.g., *GPD1*, *TEF1*, or *EXP1* from *Y. lipolytica*), a coding sequence for the target gene (e.g., *DGA1* or *DGA2*), and a transcriptional terminator. Selectable markers (e.g., *nat1* for nourseothricin resistance, *ble* for zeocin resistance, or *hph* for hygromycin resistance) are included to facilitate transformant selection.

Gene deletions, such as *Δtgl3*, are constructed using split-marker or PCR-based homologous recombination strategies. Flanking sequences (~50 bp) homologous to regions upstream and downstream of the target gene are fused to a selectable marker, and the resulting linear DNA is transformed into competent cells. Hydroxyurea treatment is optionally used to synchronize cells in S-phase, enhancing homologous recombination efficiency.

Multiple modifications are stacked sequentially or in parallel using different selection markers. For example, a strain may first be engineered to overexpress *R. toruloides DGA1* (selected with nourseothricin), then transformed with *C. purpurea DGA2* (selected with zeocin), and finally subjected to *TGL3* deletion (selected with hygromycin), yielding a triple-modified strain like NS432.

### Exemplary Nucleic Acids, Cells, and Methods

The invention provides isolated nucleic acids encoding DGA1 and DGA2 polypeptides from specific donor organisms. Exemplary DGA1 sequences include:
- NG15: native *Y. lipolytica DGA1* (GenBank accession or sequence provided in supplementary materials)
- NG66: synthetic intron-free *R. toruloides DGA1* cDNA
- NG68: synthetic intron-free *L. starkeyi DGA1* cDNA

Exemplary DGA2 sequences include:
- NG16: native *Y. lipolytica DGA2*
- NG112: synthetic *C. purpurea DGA2*

These nucleic acids are codon-optimized or left in native form, as optimization does not consistently improve function. They are cloned into expression vectors under control of *Y. lipolytica* promoters and transformed into host cells via lithium acetate/polyethylene glycol-mediated transformation.

Cells are cultivated in defined media with high carbon-to-nitrogen ratios (e.g., 100 g/L glucose, 0.5 g/L urea) to induce lipid accumulation. Fluorescence-based screening using BODIPY 493/503 allows rapid assessment of neutral lipid content in microtiter plates. High-performing clones are scaled to shake flasks and bioreactors for validation by gas chromatography.

### Species of Transformed Cell

The invention encompasses transformed cells of various oleaginous species. Primary examples include:
- *Yarrowia lipolytica* strains such as NS18 (wild-type), NS297 (*YlDGA1* overexpression), NS281 (*RtDGA1* overexpression), NS432 (*RtDGA1* + *CpDGA2* + *Δtgl3*)
- *Arxula adeninivorans* engineered to overexpress DGA1, DGA2, or DGA3 (a third DGAT isoform)

Each transformed cell exhibits increased TAG accumulation relative to its unmodified counterpart, with *Y. lipolytica* NS432 representing the pinnacle of performance.

### Products

The primary product of the invention is microbial biomass enriched in triacylglycerols, which can be processed into fatty acid methyl esters (biodiesel), free fatty acids, or other lipid derivatives. Secondary products include the engineered cells themselves, which serve as proprietary biocatalysts, and the nucleic acids/vectors used to construct them, which are valuable intellectual property assets.

### Methods Related to DGA1 and DGA2

Methods include transforming a host cell with nucleic acids encoding DGA1 and DGA2, culturing the cell under lipid-accumulating conditions, and recovering lipids. The DGA1 and DGA2 may be from the same or different species, with optimal combinations identified empirically (e.g., *R. toruloides* DGA1 + *C. purpurea* DGA2).

### Methods Related to DGA2

Specific methods focus on DGA2 overexpression, particularly using *C. purpurea* DGA2 in a DGA1-overexpressing background. This enhances lipid content beyond DGA1 alone, confirming DGA2’s role as a limiting factor when DGA1 is abundant.

### Methods Related to DGA3

Although not extensively characterized in the research paper, DGA3—a putative third DGAT—is mentioned in the context of *Arxula adeninivorans*. Methods include cloning and expressing *DGA3* homologs to assess their contribution to lipid synthesis, expanding the DGAT engineering toolkit.

## EXEMPLIFICATION

### Example 1: Method to Increase the Activity of a DGA1 Protein (DGAT2 Gene)

To enhance DGA1 activity in *Yarrowia lipolytica*, the native *DGA1* gene (NG15) was placed under the control of the strong constitutive *GPD1* promoter and integrated randomly into the genome of strain NS18, yielding strain NS297. Transformants were screened in 96-well plates using a BODIPY-based fluorescence assay, with top performers validated in shake flasks and bioreactors. NS297 exhibited a two-fold increase in lipid content over wild-type, confirming DGA1 overexpression as a potent strategy for lipid enhancement.

### Example 2: Lipid Assay

A high-throughput lipid assay was developed using BODIPY 493/503 dye. Cells grown in nitrogen-limited media were mixed with ethanol, then combined with a master mix containing potassium iodide, BODIPY, DMSO, PEG, and water. Fluorescence (ex 486 nm, em 510 nm) was normalized to OD600, providing a rapid proxy for lipid content that correlated strongly with GC-measured FAMEs.

### Example 3: Analysis and Screening of Y. lipolytica Strains that Express DGA1

Nine DGA1 genes—from *Y. lipolytica*, *R. toruloides*, *L. starkeyi*, *A. limacinum*, *A. terreus*, and *C. purpurea*—were expressed under the *GPD1* promoter. Strains expressing *R. toruloides* (NG66, NG67) and *L. starkeyi* (NG68) DGA1 showed the highest lipid accumulation (~3-fold over wild-type), surpassing native *YlDGA1*. Intron-containing *RtDGA1* (NG49) performed worse, suggesting inefficient splicing in *Y. lipolytica*.

### Example 4: Method to Knockout Triacylglycerol Lipase Knockout Gene in Y. lipolytica

The *TGL3* gene was deleted using a split-marker approach. Flanking homology arms were fused to the *hph* marker, and the construct was transformed into hydroxyurea-synchronized cells. PCR confirmed gene replacement. While *Δtgl3* did not increase lipids at 96 h, it significantly preserved lipids at 140 h in bioreactors, demonstrating its role in late-stage TAG degradation.

### Example 5: Cells that Overexpress Both DGA1 and DGA2 and That Contain a TGl3 Deletion Accumulate More TAGs Than Cells That Do Not Overexpress DGA2

Strain NS125 (*YlDGA1* overexpression) was further engineered to overexpress *YlDGA2*, yielding a 65% lipid increase. Combining *RtDGA1*, *CpDGA2*, and *Δtgl3* in NS432 resulted in superior performance: 77% lipid content in batch and 85 g/L titer in fed-batch, outperforming strains lacking any one component.

### Example 6: Increasing the Activity of DGA1, DGA2, or DGA3 in Arxula adeninivorans

*Arxula adeninivorans* was transformed with expression cassettes for DGA1, DGA2, or DGA3 homologs. Fluorescence assays confirmed increased lipid accumulation, demonstrating the portability of the DGAT engineering strategy beyond *Y. lipolytica*.

### Example 7: Increasing the Activity of DGA1, DGA2, or DGA3 in Yarrowia lipolytica

Comprehensive screening in *Y. lipolytica* identified *RtDGA1* and *CpDGA2* as optimal. Their co-expression with *Δtgl3* yielded strain NS432, which achieved 0.21 g/g yield and 0.73 g/L/h productivity—the highest reported values for this organism.

## EQUIVALENTS

The present invention is not limited to the specific strains, genes, or methods described herein. Equivalent embodiments include the use of functionally similar DGAT enzymes from other species, alternative promoters, different selection markers, and other oleaginous hosts. Modifications that do not alter the core inventive concept—synergistic DGAT overexpression coupled with lipase deletion—are considered within the scope of the claims.