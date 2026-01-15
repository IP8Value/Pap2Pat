Here is the patent application following the provided outline:

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates to the field of nucleic acid analysis, specifically methods for detecting and quantifying cytosine modifications in DNA and RNA. More particularly, the invention provides bisulfite-free chemical approaches for base-resolution sequencing of 5-methylcytosine (5mC), 5-hydroxymethylcytosine (5hmC), 5-formylcytosine (5fC), and 5-carboxylcytosine (5caC) modifications. The disclosed methods enable direct, quantitative analysis of these epigenetic markers with improved accuracy and reduced DNA damage compared to conventional bisulfite sequencing techniques.  

## BACKGROUND  

Cytosine modifications represent an important layer of epigenetic information in nucleic acids. 5-methylcytosine (5mC) is the most studied DNA modification, playing crucial roles in gene regulation and development. The ten-eleven translocation (TET) family of dioxygenases oxidizes 5mC to form 5-hydroxymethylcytosine (5hmC), which is particularly enriched in neuronal cells. Further TET oxidation produces 5-formylcytosine (5fC) and 5-carboxylcytosine (5caC), intermediates in active DNA demethylation pathways.  

Traditional bisulfite sequencing (BS) has been the gold standard for detecting 5mC and 5hmC, but suffers from significant limitations including extensive DNA degradation and sequence complexity reduction. Modified approaches like oxidative bisulfite sequencing (oxBS-seq) and TET-assisted bisulfite sequencing (TAB-seq) provide specificity for individual modifications but retain these drawbacks. Recent enzymatic methods such as APOBEC-coupled epigenetic sequencing (ACE-seq) and Enzymatic Methyl-seq (EM-seq) reduce DNA damage but still rely on indirect detection through cytosine-to-thymine conversion.  

RNA modifications including 5mC and 5hmC present additional analytical challenges due to RNA's inherent instability. Current RNA bisulfite sequencing (RNA-BS-seq) methods face similar limitations as DNA analysis techniques, with additional complications from RNA degradation. There exists a pressing need for improved methods that can directly detect cytosine modifications with high specificity while preserving nucleic acid integrity across both DNA and RNA samples.  

## SUMMARY OF THE INVENTION  

The present invention provides TET-Assisted Pyridine borane Sequencing (TAPS), a novel family of bisulfite-free methods for direct detection of cytosine modifications. TAPS utilizes TET enzyme oxidation followed by borane reduction chemistry to convert modified cytosines to dihydrouracil (DHU), which is read as thymine during sequencing. This approach preserves nucleic acid integrity while enabling direct, quantitative analysis at single-base resolution.  

Key aspects of the invention include methods for specifically identifying 5mC through TAPS with β-glucosyltransferase blocking (TAPSβ), where 5hmC is protected prior to TET oxidation. The invention further provides Chemical-Assisted Pyridine borane Sequencing (CAPS) for specific 5hmC detection through chemical oxidation prior to borane reduction. Combined application of TAPSβ and CAPS enables subtraction-free determination of both 5mC and 5hmC distributions.  

Additional embodiments include Pyridine borane Sequencing (PS) for direct detection of 5fC and 5caC, as well as PS-c for specific 5caC analysis through blocking of 5fC. The methods are applicable to both DNA and RNA samples, with optimized protocols for various nucleic acid types and quantities. The invention further encompasses kits containing specialized reagents for implementing these detection methods.  

## DETAILED DESCRIPTION OF THE INVENTION  

The TAPS methodology represents a significant advancement over conventional bisulfite sequencing by combining gentle chemical treatments with direct modification detection. In the basic TAPS protocol for identifying 5mC and 5hmC, DNA samples are first treated with TET enzymes to oxidize 5mC and 5hmC to 5caC/5fC. Subsequent borane reduction converts these oxidation products to DHU, which is amplified and sequenced as thymine.  

For specific 5mC detection (TAPSβ), 5hmC is first blocked through glucosylation using β-glucosyltransferase before TET oxidation. This protection prevents oxidation of 5hmC while allowing conversion of 5mC to 5caC/5fC. The invention provides optimized reaction conditions using UDP-glucose as donor substrate and magnesium ions as cofactor. Following glucosylation, double oxidation with TET enzymes ensures complete conversion of 5mC while minimizing residual unoxidized substrate.  

CAPS methodology enables specific 5hmC detection through chemical oxidation with potassium ruthenate (K2RuO4) prior to borane reduction. The invention discloses optimized double oxidation protocols using uracil-containing adaptors to protect DNA during treatment. 2-methylpyridine borane (pic-borane) is employed as the reducing agent for single-stranded DNA, achieving high conversion efficiency with minimal false positives.  

For detection of active demethylation intermediates, PS directly converts 5fC and 5caC to DHU using pyridine borane chemistry. PS-c provides specific 5caC analysis through prior blocking of 5fC with O-ethylhydroxylamine. The invention details quantitative analysis methods using spike-in controls and high-performance liquid chromatography-tandem mass spectrometry (HPLC-MS/MS) validation.  

The methods are applicable to diverse nucleic acid samples including genomic DNA, cell-free DNA, and RNA. Optimal processing conditions are provided for different sample types and quantities, including low-input protocols. The invention further describes comprehensive kits containing specialized reagents such as TET enzymes, borane compounds, blocking agents, and purification components for convenient implementation of these methods.  

## EXAMPLES  

Example 1 demonstrates preparation of model DNA substrates containing defined cytosine modifications. Synthetic oligos with specific 5mC, 5hmC, 5fC, and 5caC modifications were generated through enzymatic treatment and chemical synthesis. Methylated bacteriophage lambda DNA and 222 bp model fragments were prepared as validation standards.  

Example 2 details expression and purification of NgTET1 and mTET1CD proteins for oxidation reactions. Optimal conditions for enzyme activity were established using HPLC-MS/MS analysis of reaction products. Example 3 describes comprehensive validation of TAPSβ performance using mouse embryonic stem cell (mESC) genomic DNA, showing 97.6% conversion efficiency for 5mC with 0.24% false positive rate.  

Example 4 characterizes CAPS methodology through comparison with existing 5hmC detection techniques. Using mESC DNA, CAPS demonstrated superior mapping rates (90.7%) compared to TAB-seq (66.2-68.2%) and ACE-seq (21.4-26.1%), with excellent correlation to established datasets (Pearson's r=0.79).  

Example 5 evaluates PS and PS-c for detection of 5fC and 5caC in regulatory regions. Enrichment analysis revealed modification patterns consistent with active chromatin states, particularly at H3K4me1 and H3K4me3 marked regions. Example 6 presents whole-genome application of TAPS methods, demonstrating improved coverage uniformity and reduced PCR bias compared to bisulfite sequencing.  

Example 7 details low-input and cell-free DNA applications, establishing the method's sensitivity down to nanogram quantities. Example 8 provides comprehensive bioinformatics pipelines for data analysis, including specialized algorithms for base calling and modification quantification. Comparative analyses demonstrate TAPS' advantages in sequencing quality, mapping efficiency, and coverage distribution across genomic features.