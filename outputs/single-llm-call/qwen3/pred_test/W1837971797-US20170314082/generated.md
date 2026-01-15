# DESCRIPTION

- relate to recombinant protein production  
The present invention relates to methods for improving the efficiency and reliability of recombinant protein production in mammalian cell systems, particularly through the early identification of cell clones capable of sustained high-level expression of recombinant proteins over extended cultivation periods. Recombinant protein production in mammalian hosts is a cornerstone of modern biopharmaceutical manufacturing, enabling the synthesis of complex therapeutic molecules such as monoclonal antibodies, fusion proteins, and glycosylated enzymes that require precise post-translational modifications for biological activity. Despite the widespread adoption of these systems, a significant proportion of generated cell clones exhibit declining productivity over time, leading to substantial losses in yield, increased development timelines, and elevated manufacturing costs. The invention addresses this persistent challenge by providing a predictive, gene expression-based method for selecting cell clones with inherent stability prior to extensive downstream characterization and expansion, thereby reducing the reliance on time-consuming and resource-intensive long-term stability assays.

- introduce CHO cells  
Chinese hamster ovary (CHO) cells are the most widely utilized host system for the production of recombinant therapeutic proteins due to their ability to perform human-like post-translational modifications, their well-characterized biology, and their established regulatory acceptance. These cells are particularly suited for the expression of complex biologics such as immunoglobulins and other glycoproteins requiring proper folding, disulfide bond formation, and glycosylation patterns. However, CHO cells possess a highly dynamic and unstable genome, which renders them susceptible to epigenetic silencing, chromosomal rearrangements, and loss of amplified transgene copies during prolonged culture, especially in the absence of selective pressure. This genomic instability is a primary cause of productivity decline in recombinant CHO cell lines, often manifesting after weeks or months of cultivation and rendering previously high-producing clones unsuitable for commercial manufacturing. The invention leverages the unique transcriptional signatures of CHO cells to predict this instability at an early stage, enabling the selection of clones with superior long-term performance.

- describe gene amplification procedure  
Gene amplification in CHO cell lines is typically achieved through the use of dihydrofolate reductase (DHFR)-deficient host cells and selective pressure with methotrexate (MTX), which induces copy number increases in the co-transfected recombinant gene linked to the DHFR gene. This process allows for the generation of high-producing clones by enriching for cells that have integrated multiple copies of the expression cassette. However, the amplified regions are often prone to recombination, deletion, or silencing, particularly when MTX is removed or its concentration is reduced. The resulting heterogeneity within clonal populations leads to unpredictable productivity profiles, making it difficult to identify clones that will maintain high output over the duration of a manufacturing campaign. The invention circumvents this uncertainty by identifying transcriptional markers that correlate with genomic and epigenetic stability, independent of the amplification method employed.

- explain unstable recombinant protein production  
Unstable recombinant protein production is characterized by a progressive decline in the titer of the target protein over successive generations of cell culture, even under conditions that initially supported high productivity. This phenomenon is frequently observed in CHO cell lines following the removal of selective agents such as MTX, and it is not always correlated with a proportional loss of transgene copy number, suggesting that transcriptional silencing, chromatin remodeling, or metabolic stress may also contribute. The instability manifests as increased variability in productivity among clones, delayed onset of decline, and inconsistent performance across bioreactor scales, all of which complicate process development and regulatory compliance. The invention provides a solution by detecting molecular signatures that precede and predict this decline, enabling early elimination of unstable clones before significant resources are invested in their propagation.

- motivate need for early identification of stable clones  
The conventional approach to clone selection relies on screening hundreds of clones over extended periods—often 10 to 16 weeks—to identify those with stable productivity. This process is labor-intensive, costly, and delays the timeline for clinical and commercial development. The ability to identify stable clones within the first few weeks of culture would dramatically accelerate cell line development, reduce operational expenditures, and improve the predictability of manufacturing outcomes. The present invention fulfills this need by demonstrating that the expression levels of a defined set of endogenous genes, measurable shortly after transfection and before amplification, are predictive of long-term stability, allowing for the prioritization of high-performing clones at the earliest feasible stage.

- summarize previous publications on instability  
Prior studies have explored the relationship between productivity and gene expression using transcriptomic and proteomic profiling, but these efforts have largely focused on identifying markers associated with high productivity rather than long-term stability. Some investigations have linked apoptosis, metabolic stress, or endoplasmic reticulum burden to instability, while others have examined the role of promoter methylation or chromatin structure. However, none have established a robust, multi-gene signature capable of distinguishing stable from unstable clones prior to the onset of productivity decline, nor have they validated such signatures across multiple clones under identical cultivation conditions. The present invention overcomes these limitations by identifying a specific combination of genes whose expression patterns, when analyzed collectively, reliably classify clones as stable or unstable based on data collected during the initial phase of culture.

- state problem to be solved  
The problem to be solved is the inability to accurately predict, at an early stage of cell line development, which recombinant CHO cell clones will maintain consistent and high-level production of a recombinant protein over extended periods of cultivation without selective pressure. Current methods rely on prolonged culture and repeated productivity measurements, which are inefficient and delay the selection of optimal clones. There is a critical need for a rapid, scalable, and molecularly grounded method to identify clones with inherent stability, thereby reducing the time, cost, and risk associated with biopharmaceutical manufacturing.

- introduce method for selecting suitable candidate cell clones  
The invention introduces a method for selecting suitable candidate cell clones for recombinant protein production by determining the expression levels of a panel of endogenous genes in early-stage clones, correlating these levels with known stability outcomes, and using the resulting expression profile to classify clones as suitable or unsuitable for further development. This method enables the exclusion of unstable clones prior to extensive amplification and expansion, thereby streamlining the cell line development workflow and improving the success rate of commercial manufacturing.

- describe step a) of method  
Step a) of the method comprises determining the expression level of at least two endogenous genes in a plurality of candidate cell clones derived from a mammalian host cell line, wherein the expression level is measured prior to or shortly after the introduction of a recombinant gene encoding the target protein. The determination is performed under standardized culture conditions, using a quantitative method capable of detecting transcript abundance with high precision and reproducibility.

- define genes for which expression level is determined  
The genes for which expression level is determined include, but are not limited to, Fgfr2, BX842664.2/Hist1h3c, Ptpre, Cspg4, and E130203B14. These genes were identified through comparative transcriptome analysis as being differentially expressed between clones exhibiting stable and unstable recombinant protein production, and their expression levels are not significantly influenced by the presence or absence of methotrexate.

- explain optional Vsnl1 gene  
In certain embodiments, the expression level of the Vsnl1 gene may be included in the analysis, although it has been determined that its expression does not statistically differentiate between stable and unstable clones and therefore is not required for classification. Its inclusion may be used for verification purposes or as part of a broader gene panel, but its absence does not compromise the predictive accuracy of the method.

- describe determining expression level of RNA  
The expression level of each gene is determined by measuring the abundance of its corresponding messenger RNA transcript using a quantitative nucleic acid amplification technique, such as quantitative reverse transcription polymerase chain reaction (RT-qPCR). Total RNA is extracted from cells during the early log phase of growth, reverse transcribed into complementary DNA, and amplified using gene-specific primers and probes under conditions optimized for sensitivity and specificity.

- clarify determining expression level of endogenous genes  
The expression levels determined are those of endogenous cellular genes, not the recombinant gene encoding the target protein. The method does not rely on the expression level of the recombinant transgene, which may vary widely due to integration site effects, copy number, or epigenetic silencing, but instead utilizes the transcriptional state of the host cell as a predictor of its long-term stability.

- explain minimum requirement of two cell clones  
The method requires the analysis of at least two candidate cell clones to enable comparison and classification. The relative expression levels between clones are used to identify patterns associated with stability, and the method becomes increasingly robust as the number of clones analyzed increases.

- describe possibility of analysing many more clones  
The method is scalable and may be applied to hundreds of clones in parallel, enabling high-throughput screening of clonal populations generated during cell line development. Automation of RNA extraction, cDNA synthesis, and RT-qPCR setup allows for the simultaneous analysis of dozens to hundreds of clones in a single experiment.

- explain method can be carried out prior to transfection  
The method may be carried out prior to transfection by analyzing the baseline expression profile of untransfected host cells, thereby establishing a reference for normal cellular transcriptional activity. This baseline can be used to normalize or contextualize expression data from transfected clones, enhancing the accuracy of classification.

- describe analysing expression level after transfection  
Alternatively, the expression level may be analyzed after transfection and before amplification, allowing for the assessment of the host cell’s transcriptional response to the introduction of the recombinant gene. This time point is optimal for early prediction, as it precedes the onset of instability and avoids confounding effects of methotrexate selection.

- clarify type of recombinant protein is not relevant  
The type of recombinant protein expressed is not relevant to the method, as the predictive power derives from the transcriptional state of the host cell rather than the nature of the transgene. The method is applicable to the production of antibodies, enzymes, fusion proteins, or any other recombinant molecule expressed in CHO cells.

- prefer antibody as recombinant protein  
In preferred embodiments, the recombinant protein is an antibody or antibody fragment, given the high commercial demand for such therapeutics and the particular susceptibility of antibody-producing clones to productivity decline.

- describe ideal scenario of same type of recombinant protein  
The ideal scenario involves the use of the same recombinant protein construct across multiple clones, ensuring that differences in expression are attributable to host cell characteristics rather than transgene design, thereby maximizing the predictive power of the method.

- explain method can be carried out with one gene  
The method may be carried out using the expression level of a single gene, although the predictive accuracy is enhanced when multiple genes are analyzed in combination.

- describe preferred embodiments with multiple genes  
Preferred embodiments involve the simultaneous determination of expression levels for at least five genes: Fgfr2, BX842664.2/Hist1h3c, Ptpre, Cspg4, and E130203B14. The combined expression profile of these genes provides a robust signature for distinguishing stable from unstable clones.

- explain analysis of more than one gene  
The analysis of more than one gene permits the detection of multivariate patterns that are not apparent when individual genes are considered in isolation. The synergistic interaction of gene expression changes provides a more accurate and reliable predictor of long-term stability than any single marker.

- describe determining expression level in parallel  
The expression levels of the selected genes may be determined in parallel using multiplexed RT-qPCR assays, enabling the simultaneous quantification of multiple transcripts from a single RNA sample, thereby reducing reagent consumption and increasing throughput.

- explain non-parallel determination of expression level  
In alternative embodiments, the expression levels may be determined sequentially, with each gene assayed in separate reactions. While this approach is less efficient, it remains viable and may be preferred in laboratories without access to multiplexed detection systems.

- describe identical conditions for determining expression level  
The expression levels of all genes are determined under identical experimental conditions, including the same RNA extraction protocol, reverse transcription efficiency, PCR cycling parameters, and data normalization strategy, to ensure comparability and reproducibility across samples.

- prefer early log phase of growth  
The cells are harvested during the early log phase of growth, when metabolic activity is high and transcriptional profiles are most representative of the cell’s physiological state. Harvesting at this stage minimizes variability due to cell cycle phase or nutrient depletion.

- explain cell background of clones  
The candidate cell clones are derived from a single parental host cell line, ensuring genetic homogeneity and minimizing confounding effects arising from clonal variation unrelated to recombinant protein production.

- describe preferred host cells  
The preferred host cells are Chinese hamster ovary (CHO) cells, and more specifically, CHO-K1 or CHO-Dhfr-deficient derivatives adapted for serum-free suspension culture. These cell lines are widely used in industry and exhibit the genomic instability that the invention is designed to mitigate.

- prefer CHO-K1 cell clones  
In particularly preferred embodiments, the host cells are CHO-K1 clones, which have been shown to provide a consistent and well-characterized background for the detection of stability-associated transcriptional signatures.

- describe quantitative RT-PCR  
Quantitative reverse transcription polymerase chain reaction (RT-qPCR) is the preferred method for determining gene expression levels, utilizing TaqMan probes labeled with FAM and MGB to enable highly specific detection of target transcripts with minimal background noise.

- explain specific detection of gene expression levels  
The use of gene-specific primers and probes ensures that only the intended transcript is amplified and detected, avoiding cross-reactivity with homologous sequences or pseudogenes, and providing accurate quantification of mRNA abundance.

- describe log2 transformation of expression values  
The raw quantification cycle (Cq) values are converted to relative expression values using the ΔΔCq method, followed by log2 transformation to normalize the distribution and facilitate statistical analysis.

- explain normalization of expression values  
Expression values are normalized to the geometric mean of two or more stably expressed reference genes, such as Actb and Gapdh, to account for variations in RNA input, reverse transcription efficiency, and sample integrity.

- describe statistical analysis  
Statistical analysis is performed using multivariate techniques, including principal component analysis (PCA), to reduce dimensionality and visualize clustering patterns among clones. Classification is further refined using k-nearest neighbor algorithms to assign each clone to a stability category based on its gene expression profile.

- explain selecting cell clone for further expansion  
A cell clone is selected for further expansion if its gene expression profile falls within the cluster of clones previously identified as stable. Clones whose profiles cluster with unstable phenotypes are excluded from downstream development.

- describe upregulation or downregulation of genes  
The method accounts for both upregulation and downregulation of the selected genes relative to a reference population. For example, elevated expression of Fgfr2 and BX842664.2/Hist1h3c is associated with stability, while elevated expression of E130203B14 and Cspg4 is associated with instability.

- explain selection from multiple clones  
Selection is performed from a pool of multiple clones, with the top-performing candidates identified based on their combined expression signature rather than individual gene expression levels.

- describe preferred selection of best candidate clone  
The preferred selection involves identifying the clone with the most favorable combination of expression levels across all five genes, as determined by PCA and clustering analysis, and advancing that clone for amplification and scale-up.

- explain particularly preferred embodiments of method  
Particularly preferred embodiments involve the use of a five-gene panel, analysis during early log phase, normalization to Actb and Gapdh, and classification via PCA and k-nearest neighbor clustering to achieve greater than 90% accuracy in predicting long-term stability.

- define best expression level  
The best expression level is defined as the combination of transcript abundances that maximizes the distance between the clone’s profile and the centroid of the unstable cluster, as determined by multivariate analysis.

- explain upregulation and downregulation  
Upregulation refers to an increase in transcript abundance relative to the average expression level across the population, while downregulation refers to a decrease. Both phenomena are incorporated into the classification model to capture the full spectrum of transcriptional responses associated with stability.

- describe selection of clones  
Clones are selected for further development based on their position in a multidimensional expression space defined by the five genes, with those located in the stable cluster being advanced and those in the unstable cluster being discarded.

- motivate multiple gene determination  
The determination of multiple genes is motivated by the observation that no single gene provides sufficient predictive power; the combination of genes captures the complex, polygenic nature of cellular stability.

- explain combination of expression values  
The combination of expression values is achieved through multivariate statistical modeling, where each gene contributes a weighted component to a composite score that reflects the likelihood of long-term stability.

- describe preferred embodiments  
Preferred embodiments include the use of automated liquid handling systems for RT-qPCR setup, the inclusion of a reference gene panel, and the application of PCA to generate a three-dimensional representation of clone stability.

- explain indirect comparison  
The method relies on indirect comparison, wherein the expression profile of a new clone is compared against a pre-established reference model derived from previously characterized stable and unstable clones, rather than direct measurement of productivity.

- motivate multiple clone selection  
Multiple clone selection is motivated by the need to balance risk and reward in cell line development, ensuring that several high-potential candidates are advanced for further characterization and process optimization.

- describe three-dimensional representation  
A three-dimensional representation of the first three principal components of gene expression data is generated to visually distinguish stable from unstable clones, with each axis representing a major source of transcriptional variation.

- explain principal component analysis  
Principal component analysis is used to reduce the dimensionality of the gene expression dataset while preserving the maximum amount of variance, enabling the identification of underlying patterns that correlate with stability.

- define further expansion  
Further expansion refers to the process of culturing a selected clone under conditions that promote proliferation and amplification of the recombinant gene, typically involving methotrexate selection and scale-up in bioreactors.

- describe expansion step  
The expansion step involves propagating the selected clone in increasing volumes of culture medium under controlled conditions until sufficient cell biomass is generated for downstream manufacturing processes.

- explain MTX selection  
Methotrexate selection may be applied during or after the expansion step to amplify the recombinant gene copy number, but the selection of the clone itself is performed prior to this step, based solely on endogenous gene expression.

- clarify ex vivo or in-vitro method  
The method is an ex vivo or in-vitro procedure, performed entirely in cell culture systems without the use of animal models or in vivo components.

- introduce host cell for recombinant protein expression  
The host cell used in the method is a mammalian cell line genetically engineered to express a recombinant protein, with CHO cells being the most suitable due to their compatibility with industrial bioprocessing.

- describe artificially modified gene expression  
The method detects naturally occurring differences in gene expression that are not artificially induced but are instead inherent to the genomic and epigenetic state of the cell clone.

- explain overexpression  
Overexpression of the recombinant gene is not required for the method to function, as the predictive power derives from the host cell’s transcriptional state, not the level of transgene expression.

- explain downregulation  
Downregulation of endogenous genes may occur as a consequence of cellular stress or genomic instability, and such changes are captured by the method as indicators of poor long-term performance.

- describe preferred host cells  
Preferred host cells are derived from CHO-K1 or CHO-Dhfr-deficient lines, grown in serum-free suspension culture, and adapted for high-density bioreactor production.

- explain recombinant protein production  
Recombinant protein production is achieved through the stable integration of a gene encoding the target protein into the host cell genome, followed by selection, amplification, and cultivation under controlled conditions.

- describe recombinant gene encoding  
The recombinant gene encodes a protein of therapeutic interest, such as a monoclonal antibody, and is operably linked to regulatory elements that drive its expression in mammalian cells.

- explain preferred recombinant proteins  
Preferred recombinant proteins include full-length immunoglobulins, Fc-fusion proteins, and other glycosylated biologics requiring complex post-translational modifications.

- apply method embodiments to host cell  
The method embodiments are applied directly to the host cell prior to or shortly after transfection, enabling early-stage decision-making in the cell line development pipeline.

- explain term comprising  
The term “comprising” is used in its open-ended sense, meaning that the method may include additional steps or elements beyond those explicitly described without departing from the scope of the invention.

- explain use of articles  
The use of articles such as “a,” “an,” and “the” is intended to encompass both singular and plural referents unless otherwise indicated by context.

- define suitable candidate cell clone  
A suitable candidate cell clone is defined as a clonal population of mammalian cells whose endogenous gene expression profile, as determined by the method, is statistically indistinguishable from that of clones known to exhibit stable recombinant protein production over extended culture periods.