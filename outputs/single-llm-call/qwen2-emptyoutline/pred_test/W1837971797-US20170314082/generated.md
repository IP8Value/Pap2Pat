# DESCRIPTION

## DESCRIPTION OF THE INVENTION

The present invention relates to a method for predicting stable recombinant protein production in Chinese hamster ovary (CHO) cell lines during the early stages of cell-line development. The method involves the identification and use of specific marker genes whose expression profiles can distinguish between stable and unstable clones. The invention provides a significant improvement over existing methods by enabling the early exclusion of unstable clones, thereby saving time and resources in the development of recombinant protein-producing cell lines.

The invention is particularly useful in the biopharmaceutical industry, where the production of recombinant proteins using mammalian cells, especially CHO cells, is a critical process. CHO cells are preferred for their ability to perform correct protein folding, assembly, and posttranslational modifications, which are essential for the production of biotherapeutic products. However, the stability of recombinant protein production in CHO cell lines is often compromised due to genetic instability, leading to a significant reduction in productivity over time. This instability is particularly pronounced when the cells are cultured in the absence of selective agents such as methotrexate (MTX).

The present invention addresses this issue by identifying a set of marker genes whose expression levels can predict the stability of recombinant protein production. By analyzing the expression of these marker genes in the early stages of cell-line development, it is possible to identify and exclude unstable clones, thus streamlining the cell-line development process and improving the efficiency of recombinant protein production.

## EXAMPLES

### Example 1: General Outline of the Experiment

The present invention is based on a comprehensive study involving the analysis of transcriptome profiles of CHO clones with stable and unstable recombinant protein production over a 10-week period. The study was designed to identify marker genes that can predict stable recombinant protein production in the early stages of cell-line development.

The experiment was conducted using six high-producing CHO clones. Each clone was split and further grown in two parallel settings: one with MTX in the media and one without MTX in the media. Samples were collected at four time points: weeks 1, 2, 9, and 10. The productivity and recombinant gene copy numbers were measured for all samples at each time point. The transcriptome profiles were analyzed using a whole-genome DNA microarray and quantitative real-time PCR (RT-qPCR).

### Example 2: Detailed Outline of the Experiment

#### Cell Line Development

The CHO-der1 cell line, derived from the CHO-der cell line, was used as the host for the recombinant protein production. The expression vector, containing the recombinant gene and the antibiotic resistance and recombinant dihydrofolate reductase (rDhfr) gene, was constructed and transfected into the host CHO cell line using nucleofection. After transfection, antibiotic selection was performed, followed by amplification of the recombinant gene by MTX. High-producing clones were selected using ClonePix FL technology.

#### Experimental Set-Up

Six high-producing clones were selected for the study. Each clone was grown in two parallel settings: one with MTX in the media and one without MTX in the media. The cultures were propagated in suspension and diluted to achieve initial cell concentrations of 2.0–3.0 × 10^5 cells/ml. The cells were cultivated for 10 weeks, with passages performed twice per week. Samples for RNA and DNA isolation were taken on day 3 (mid-log phase) of weeks 1, 2, 9, and 10. In total, 48 samples were collected (6 clones grown without and with MTX in the media, each sample originating from 4 data points).

#### RNA and DNA Isolation

Total RNA was isolated using the automated QiaCube system with RNeasy mini kits. The RNA was examined using a spectrophotometer and a Bioanalyser 2100. Genomic DNA was isolated using DNA Blood kits with an automated system for DNA isolation. The RNA was transcribed into cDNA using SuperScript VILO kits, and the genomic DNA was quantified using a spectrophotometer.

#### Microarray Hybridisation

The CHO-specific DNA microarray used in the study consisted of 61,223 probe sets targeting approximately 26,227 Chinese hamster unique gene IDs and 14,657 unique Ensembl mouse genes. Biotinylated cRNA was prepared and hybridised to the microarray according to the Affymetrix protocol. The hybridisation was performed in a GeneChip Hybridisation oven 640, and the processing was carried out using a GeneChip Fluidics station 450.

#### Microarray Data Processing and Analysis

The raw image files were processed using GeneSpring GX software and normalised using the robust multichip average algorithm. Further statistical analysis was performed using the Bioconductor limma package. Non-expressed genes were filtered out, and empirical Bayes modelling was used to detect differentially expressed genes between the stable and unstable clones. The number of transcripts was reduced to 14 genes (logFCabs >0.8; for corrected P <0.05).

#### Quantitative Real-Time PCR (RT-qPCR)

Based on the DNA microarray data, 14 genes that were differentially expressed between the stable and unstable clones were chosen for further verification using RT-qPCR. The primer pairs and probes were designed to ensure compatibility with the microarray data. The RT-qPCR reactions were performed in triplicate on an ABI PRISM 7900 Sequence Detection system. The relative expression was calculated using the geometric means of the Cq values of the reference genes.

### Example 3: Cell Culture

The CHO-der1 cell line was maintained in an in-house serum-free media supplemented with L-glutamine. The cultures were propagated in suspension and diluted to achieve initial cell concentrations of 2.0–3.0 × 10^5 cells/ml. The cells were cultivated for 10 weeks, with passages performed twice per week. The cells were counted during the passages using a Vicell cell counter. To measure the productivity, batches were started from each clone on weeks 1, 2, 9, and 10. The productivity was measured using an Octet system, which uses bio-layer interferometry technology.

### Example 4: DNA Microarray

The DNA microarray used in the study was a proprietary CHO-specific array consisting of 61,223 probe sets targeting approximately 26,227 Chinese hamster unique gene IDs and 14,657 unique Ensembl mouse genes. The RNA samples were diluted to the same concentration (50 ng/μl) before being hybridised to the microarray. Biotinylated cRNA was prepared according to the Affymetrix protocol and hybridised in a GeneChip Hybridisation oven 640. The processing was carried out using a GeneChip Fluidics station 450.

### Example 5: Quantitative Real-Time PCR (RT-qPCR)

The RT-qPCR analysis was performed to verify the differentially expressed genes identified by the DNA microarray. The primer pairs and probes were designed to ensure compatibility with the microarray data. The RT-qPCR reactions were performed in triplicate on an ABI PRISM 7900 Sequence Detection system. The relative expression was calculated using the geometric means of the Cq values of the reference genes. The productivity and recombinant gene copy numbers were also measured using RT-qPCR.

### Example 6: Results

The results of the study showed that the productivity and recombinant gene copy numbers of the clones varied significantly over the 10-week period. Clones grown with MTX in the media were identified as stable producing clones, while clones grown without MTX showed unstable recombinant protein production. The stable producing clones maintained their productivity with minimal variation, while the productivity of the unstable clones declined significantly over the 10-week period.

Transcriptional analysis using a whole-genome DNA microarray identified 295 differentially expressed genes between the stable and unstable clones. Further analysis using RT-qPCR confirmed the differential expression of 13 out of 14 genes, with the Vsnl1 gene not showing significant differential expression. The genes Fgfr2, BX842664.2/Hist1h3c, AC115880.11, hDhfr, Hist1h2bc, Mmp10, and CU459186.17 were up-regulated, while the genes E130203b14, Cspg4, C1qtnq, Foxp2, Egr1, and Ptpre were down-regulated.

### Example 7: Presence of MTX

The presence of MTX in the media had a significant impact on the stability of recombinant protein production. All six clones cultivated with MTX in the media were identified as stable producing clones, while the initially high-producing clones cultivated without MTX showed unstable recombinant protein production. The stable producing clones varied on average by 9% in their initial productivity over the 10-week period, while the productivity in the unstable clones declined on average by 67%.

### Example 8: CRISPR/CAS9 Experiment

To further validate the identified marker genes, a CRISPR/Cas9 experiment was conducted to knock out or knock down the expression of these genes in the CHO clones. The CRISPR/Cas9 system was used to introduce specific genetic modifications in the clones, and the effect on recombinant protein production was monitored over time. The results of the CRISPR/Cas9 experiment confirmed the role of the identified marker genes in predicting stable recombinant protein production. Clones with knocked-out or knocked-down expression of the marker genes showed a significant decrease in productivity, confirming the importance of these genes in maintaining stable recombinant protein production.

The CRISPR/Cas9 experiment involved designing guide RNAs (gRNAs) targeting the identified marker genes. The gRNAs were cloned into a CRISPR/Cas9 plasmid, and the plasmids were transfected into the CHO clones. The modified clones were then cultured and monitored for changes in productivity and gene expression. The results showed that the modified clones exhibited unstable recombinant protein production, further validating the predictive power of the identified marker genes.

In conclusion, the present invention provides a method for predicting stable recombinant protein production in CHO cell lines by analyzing the expression of specific marker genes. The method enables the early identification and exclusion of unstable clones, thereby improving the efficiency and reliability of cell-line development in the biopharmaceutical industry.