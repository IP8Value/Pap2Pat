# DESCRIPTION

## BACKGROUND

Lipids are a diverse class of hydrophobic or amphipathic molecules that play essential roles in biological systems, including energy storage, membrane structure, and signaling. In industrial biotechnology, microbial lipids—particularly triacylglycerols (TAGs)—have attracted significant interest as renewable feedstocks for biofuels, oleochemicals, and specialty fats. Unlike plant- and animal-derived oils, microbial lipids can be produced from low-cost, non-food carbon sources such as agricultural residues, lignocellulosic biomass, and waste streams, offering a sustainable alternative to conventional oil production. Among microbial hosts, oleaginous yeasts—those capable of accumulating lipids exceeding 20% of their dry cell weight—are particularly promising due to their high lipid yields, genetic tractability, and ability to grow on diverse substrates. Of these, *Yarrowia lipolytica* has emerged as a model organism owing to its robust metabolism, well-characterized genetics, and capacity to accumulate up to 60% lipid by dry weight under optimized conditions.

Efforts to enhance microbial lipid production have increasingly focused on metabolic engineering of the TAG biosynthesis pathway. A central strategy involves modulating the expression of key enzymes that catalyze rate-limiting steps in lipid assembly. One such approach is the targeted upregulation of genes encoding diacylglycerol acyltransferases (DGATs), which catalyze the final and committed step in TAG synthesis: the acylation of diacylglycerol (DAG) with a fatty acyl-CoA to form triacylglycerol. In *Y. lipolytica*, two primary DGAT isoforms—DGA1 (a type 2 DGAT) and DGA2 (a type 1 DGAT)—have been identified as critical determinants of lipid accumulation. Overexpression of either enzyme has been shown to significantly increase intracellular TAG content, suggesting that DGAT activity is a bottleneck in native lipid biosynthesis. Moreover, studies in other oleaginous organisms, including *Rhodosporidium toruloides* and *Lipomyces starkeyi*, corroborate the pivotal role of DGATs in driving high lipid yields.

Beyond DGAT overexpression, lipid titers can be further enhanced by downregulating competing pathways, particularly those involved in TAG degradation. In *Y. lipolytica*, intracellular lipases such as TGL3 and TGL4 mediate the hydrolysis of stored TAGs during nutrient limitation or stationary phase, thereby reducing net lipid accumulation. Genetic deletion of these lipases has been demonstrated to stabilize lipid droplets and improve final lipid content, especially in prolonged fermentation processes. Additionally, other genes in the lipid biosynthetic cascade—such as those encoding glycerol-3-phosphate dehydrogenase (GPD1), acyltransferases (SLC1, LRO1), and stearoyl-CoA desaturase (SCD)—have been explored as secondary targets for combinatorial engineering. However, the synergistic effects of heterologous DGAT expression, endogenous pathway enhancement, and lipase suppression remain underexplored, particularly when applied systematically across multiple genetic backgrounds and fermentation regimes.

The protein DGA1, specifically, is a diacylglycerol acyltransferase of type 2 (DGAT2) that localizes to lipid droplets and is responsible for both TAG synthesis and droplet expansion. In *Y. lipolytica*, DGA1 overexpression alone can double lipid content, and its activity is considered a major driver of oleaginicity. DGA1 homologs from other high-lipid-producing organisms often exhibit superior catalytic efficiency or expression characteristics when expressed heterologously, making them attractive candidates for strain engineering. Similarly, DGA2—a type 1 DGAT (DGAT1) located in the endoplasmic reticulum—is implicated in the formation of nascent lipid droplets and may become limiting only when DGA1 activity is elevated. The interplay between DGA1 and DGA2 thus represents a tunable node for metabolic optimization.

Other genes influencing lipid production include those involved in precursor supply (e.g., ATP citrate lyase, acetyl-CoA carboxylase), redox balance (malic enzyme), and fatty acid modification (desaturases). While individually beneficial, their impact is often context-dependent and less pronounced than direct manipulation of the terminal TAG assembly and degradation machinery. Therefore, a rational engineering strategy prioritizing DGAT overexpression and lipase knockout offers a streamlined path to high-yield lipid production.

## SUMMARY

The present invention provides a genetically modified microbial cell engineered to exhibit enhanced triacylglycerol (TAG) content through specific genetic modifications. In one embodiment, the transformed cell comprises a first genetic modification comprising an exogenous nucleotide sequence encoding a type 1 diacylglycerol acyltransferase (DGA1 or DGAT2) operably linked to a promoter that drives its expression in the host cell. This first modification results in increased activity of the DGA1 protein, thereby enhancing the conversion of diacylglycerol and acyl-CoA into triacylglycerol.

The transformed cell further comprises a second genetic modification comprising an exogenous nucleotide sequence encoding a type 2 diacylglycerol acyltransferase (DGA2 or DGAT1) operably linked to a promoter that drives its expression. This second modification increases DGA2 activity, which complements DGA1 function by promoting the formation of new lipid droplets and supporting sustained TAG synthesis, particularly under high-flux conditions.

Optionally, the transformed cell includes a third genetic modification that reduces or eliminates the activity of a triacylglycerol lipase, such as TGL3 or TGL4. This is achieved through gene knockout, promoter replacement with a less active variant, or RNA interference, thereby minimizing TAG degradation during late-stage fermentation when carbon sources are depleted.

The invention also encompasses a method for increasing lipid content in a microbial host, comprising introducing into the host cell a first nucleotide sequence encoding a DGA1 protein and a second nucleotide sequence encoding a DGA2 protein, each under the control of a functional promoter. In a further embodiment, the method includes a third nucleotide sequence designed to disrupt or silence a triacylglycerol lipase gene, such as TGL3. Alternatively, the method may employ a single nucleotide sequence encoding a chimeric or fusion protein that combines functional domains of both DGA1 and DGA2, though this is less preferred due to potential folding or localization issues.

A specific method for increasing triacylglycerol content involves transforming a host cell—such as *Yarrowia lipolytica* or *Arxula adeninivorans*—with expression constructs containing codon-optimized DGA1 and DGA2 genes derived from high-lipid-producing donors like *Rhodosporidium toruloides* or *Claviceps purpurea*. The transformed cells are then cultured under nitrogen-limited, carbon-rich conditions to induce lipid accumulation. Optionally, the triacylglycerol is recovered from the cells via extraction, transesterification, or secretion, yielding a product suitable for use as biodiesel, lubricants, or food-grade oils.

## DETAILED DESCRIPTION

### Overview

The invention is directed toward increasing triacylglycerol (TAG) content in microbial cells through coordinated genetic engineering of the lipid biosynthesis and degradation pathways. Central to this strategy is the overexpression of diacylglycerol acyltransferase enzymes, which catalyze the final step of TAG assembly. Specifically, DGA1 (a type 2 DGAT) is overexpressed to drive high-level TAG synthesis and lipid droplet expansion, while DGA2 (a type 1 DGAT) is co-expressed to support the formation of new lipid droplets and prevent metabolic bottlenecks. Concurrently, the activity of triacylglycerol lipases—particularly TGL3—is reduced or eliminated to prevent catabolism of stored TAGs during stationary phase. The combined effect of these modifications results in microbial strains that accumulate TAGs at levels exceeding 70% of dry cell weight, with improved yield and productivity in both batch and fed-batch fermentations.

### Definitions

As used herein, the articles “a” and “an” are intended to mean one or more unless the context clearly indicates otherwise. “Activity” refers to the enzymatic or functional capacity of a protein, such as the ability of a diacylglycerol acyltransferase to catalyze TAG formation. Genetic modification can increase or decrease activity by altering gene expression, protein stability, or catalytic efficiency. A “biologically-active portion” of a protein is a fragment that retains the functional activity of the full-length protein; for DGA1, this includes the conserved acyltransferase domain spanning residues X to Y, and for DGA2, the transmembrane and catalytic domains. “DGAT1,” “DGAT2,” and “DGAT3” refer to type 1, type 2, and type 3 diacylglycerol acyltransferases, respectively, with DGA2 and DGA1 being synonymous with DGAT2 and DGAT1 in yeast nomenclature. “Diacylglyceride,” “diacylglycerol,” and “diglyceride” are used interchangeably to denote DAG, the substrate for DGATs. “Diacylglycerol acyltransferase” (DGA) is an enzyme that transfers an acyl group from acyl-CoA to DAG to form TAG. A “domain” is a structurally and functionally distinct region of a protein. A “drug” includes any therapeutic or diagnostic agent. “Dry weight” or “dry cell weight” is the mass of cells after removal of water. “Encode” means a nucleic acid specifies the amino acid sequence of a protein. “Exogenous” refers to a molecule or sequence not native to the host cell; an “exogenous nucleic acid” or “exogenous gene” is introduced via transformation. “Expression” is the process by which a gene is transcribed and translated into protein; “increased expression” denotes higher mRNA or protein levels relative to a control. A “gene” is a DNA segment encoding a functional product. “Genetic modification” includes any alteration to the genome, such as insertion, deletion, or mutation. A “homolog” is a gene or protein sharing evolutionary ancestry and sequence similarity. An “inducible promoter” activates transcription in response to a signal. “Integrated” means stably incorporated into the host genome. “In operable linkage” indicates a promoter is positioned to drive transcription of a downstream gene. A “knockout mutation” abolishes gene function. “Native” refers to endogenous sequences. A “nucleic acid” is DNA or RNA. A “parent cell” is the unmodified host. A “plasmid” is a circular DNA vector. A “portion” is a segment of a larger molecule. A “promoter” initiates transcription. “Recombinant” describes cells or molecules altered by genetic engineering. A “regulatory region” controls gene expression. “Transformation” is the introduction of foreign DNA into a cell, yielding a “transformed cell.” “Triacylglyceride” and “triacylglycerol” (TAG) are synonymous. A “triacylglycerol lipase” hydrolyzes TAG into DAG and free fatty acids. A “vector” is a DNA molecule used to deliver genetic material.

### Microbe Engineering

Microbial host cells are genetically modified using standard recombinant DNA techniques. Suitable hosts include prokaryotes (e.g., *E. coli*) and eukaryotes, particularly oleaginous yeasts such as *Yarrowia lipolytica*, *Arxula adeninivorans*, *Rhodosporidium toruloides*, and *Lipomyces starkeyi*. Fungal and yeast systems are preferred due to their native lipid metabolism and compartmentalization. Expression systems involve chimeric genes comprising heterologous coding sequences under control of strong promoters (e.g., *Y. lipolytica* GPD1 or TEF1). Transformation is achieved via electroporation, biolistics, or chemical methods (e.g., lithium acetate/PEG). Plasmids are constructed with selectable markers (e.g., hygromycin, zeocin resistance) and expression cassettes containing promoters, coding sequences, and terminators. Homologous recombination is used for targeted integration, facilitated by flanking sequences homologous to genomic loci. Vector design includes optimized codon usage, removal of introns, and inclusion of regulatory elements. Co-transformation with multiple constructs enables stacking of modifications. Selectable markers allow isolation of transformants, and homologous recombination ensures stable inheritance.

### Exemplary Nucleic Acids, Cells, and Methods

The invention utilizes nucleic acid molecules encoding DGA1, DGA2, and DGA3 proteins. DGA1 genes are derived from organisms such as *R. toruloides* (SEQ ID NO:1 for gene, SEQ ID NO:2 for protein), *Y. lipolytica* (NG15), *L. starkeyi*, *A. limacinum*, *A. terreus*, and *C. purpurea*. Substantially identical variants share at least 80% amino acid identity. DGA2 genes include those from *C. purpurea* (NG112, SEQ ID NO:3/4), *Y. lipolytica* (NG16), *R. toruloides*, *L. starkeyi*, *A. terreus*, and *C. globosum*, with ≥80% identity. DGA3 genes are less common but include homologs from plants and fungi. Conservative substitutions (e.g., leucine for isoleucine) are permitted. Percent identity is determined by algorithms such as BLAST or ClustalW. Coding sequences are optimized for host expression. Vectors like pNC243 (for DGA1) and pNC327 (for DGA2) contain strong promoters and markers. Triacylglycerol lipase knockouts target TGL3 (e.g., deletion cassette with hygromycin marker) or TGL4. Transformed cells include *Y. lipolytica* NS18 derivatives and *A. adeninivorans*. Modifications include DGA1/DGA2 overexpression via constitutive promoters, TGL3 knockout via homologous recombination, and integration into the genome for stable inheritance. Cells are grown on glucose or alternative carbon sources, and products include intracellular TAGs recoverable by extraction.

### Species of Transformed Cell

Transformed cell species include *Yarrowia lipolytica*, *Arxula adeninivorans*, *Rhodosporidium toruloides*, *Lipomyces starkeyi*, *Aurantiochytrium limacinum*, *Aspergillus terreus*, *Claviceps purpurea*, and *Chaetomium globosum*.

### Products

Products derived from the cells include triacylglycerols, fatty acid methyl esters (biodiesel), and oleochemicals, useful in fuels, lubricants, cosmetics, and food.

### Methods Related to DGA1 and DGA2

A method for increasing TAG content involves: (1) introducing a first nucleotide sequence encoding a DGA1 protein (e.g., from *R. toruloides*, SEQ ID NO:1, ≥80% identity, biologically-active portion); (2) introducing a second nucleotide sequence encoding a DGA2 protein (e.g., from *C. purpurea*, SEQ ID NO:3); and (3) optionally introducing a third nucleotide sequence to disrupt TGL3 via homologous recombination. The host is *Y. lipolytica* or *A. adeninivorans*, and a drug resistance marker (e.g., hygromycin) aids selection. TAG is optionally recovered post-fermentation.

### Methods Related to DGA2

A method comprises transforming a cell with a nucleotide sequence encoding DGA2 (e.g., SEQ ID NO:3, ≥80% identity to *C. purpurea* DGA2), growing under nitrogen limitation, and optionally recovering TAG.

### Methods Related to DGA3

Similarly, a method uses a DGA3-encoding sequence (e.g., from a plant or fungus, ≥80% identity), with analogous growth and recovery steps.

## EXEMPLIFICATION

### Example 1: Method to Increase the Activity of a DGA1 Protein (DGAT2 Gene)

The expression construct pNC243 was assembled containing the *R. toruloides* DGA1 gene (NG66) under the *Y. lipolytica* GPD1 promoter and a nourseothricin resistance marker. *Y. lipolytica* strain NS18 was transformed via PEG-mediated transformation, and integrants were selected on nourseothricin plates. Fluorescence assay showed a 3-fold increase in lipid content versus wild-type, confirmed by GC analysis.

### Example 2: Lipid Assay

Cells were grown in nitrogen-limited media (100 g/L glucose) in 96-well plates for 96 h at 30°C. Bodipy 493/503 staining was performed, and fluorescence (ex 486 nm, em 510 nm) was normalized to OD600. Data correlated with GC-measured FAMEs.

### Example 3: Analysis and Screening of Y. lipolytica Strains that Express DGA1

Strains expressing *R. toruloides* DGA1 (NG66) or *Y. lipolytica* DGA1 (NG15) were screened. NG66 transformants (e.g., NS281) showed 3-fold higher lipid content than NS18. HPLC confirmed similar glucose consumption, indicating enhanced conversion efficiency.

### Example 4: Method to Knockout Triacylglycerol Lipase Knockout Gene in Y. lipolytica

A TGL3 deletion cassette with 50-bp homology arms and hygromycin resistance was co-transformed into NS18 and NS281. PCR-confirmed knockouts (e.g., NS377) showed increased lipid content after 140 h in bioreactors.

### Example 5: Cells that Overexpress Both DGA1 and DGA2 and That Contain a TGl3 Deletion Accumulate More TAGs Than Cells That Do Not Overexpress DGA2

Strain NS377 (RtDGA1, tgl3Δ) was transformed with *C. purpurea* DGA2 (NG112) to generate NS432. NS432 exhibited 77% lipid content in batch fermentation, outperforming strains lacking DGA2 or TGL3 knockout.

### Example 6: Increasing the Activity of DGA1, DGA2, or DGA3 in Arxula adeninivorans

Twenty-nine DGA genes were expressed in *A. adeninivorans*. Top performers included *Y. lipolytica* and *C. globosum* DGA2, which increased lipid content by >2-fold in fluorescence assays.

### Example 7: Increasing the Activity of DGA1, DGA2, or DGA3 in Yarrowia lipolytica

Eighteen DGA genes were screened in *Y. lipolytica* NS598. *R. toruloides* DGA1 and *C. purpurea* DGA2 yielded the highest lipid accumulation, consistent with prior results.

## EQUIVALENTS

The invention encompasses all equivalents of the disclosed embodiments, including variants with conservative amino acid substitutions, alternative promoters, different host species, and functionally similar genetic modifications that achieve enhanced triacylglycerol production.