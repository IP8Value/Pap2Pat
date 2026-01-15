Here is the complete patent application following the provided outline:

# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates to the technical field of detection and analysis, specifically to methods for identifying and authenticating meat products based on their volatile organic compound (VOC) profiles. More particularly, the invention pertains to the use of gas chromatography-ion mobility spectroscopy (GC-IMS) combined with multivariate statistical analysis for distinguishing donkey meat varieties by their characteristic VOC fingerprints. This analytical approach enables rapid, accurate, and non-destructive identification of meat origins without requiring complex sample preparation procedures. The technology finds particular application in food quality control, product authentication, and regulatory compliance within the meat processing industry.  

## BACKGROUND  

Meat flavor analysis has become increasingly important in modern food science and industry due to growing consumer demand for high-quality, authentic meat products. The distinctive flavor profiles of different meat varieties serve as key indicators of quality and authenticity, directly influencing consumer preferences and market value. Traditional methods for meat flavor analysis rely primarily on gas chromatography-mass spectrometry (GC-MS) or gas chromatography-olfactometry-mass spectrometry (GC-O-MS) techniques. These conventional approaches require extensive sample preparation including heating, distillation, and extraction steps, which can alter the native volatile compound profiles and lead to inconsistent results.  

Current analytical methods suffer from several limitations that hinder their widespread application in meat authentication. The complex sample pretreatment procedures are time-consuming and labor-intensive, often requiring several hours to complete. The thermal degradation and chemical modification of volatile compounds during sample preparation can distort the true flavor profiles. Moreover, the lengthy analysis times and high equipment costs associated with conventional GC-MS systems make them impractical for routine quality control applications in meat processing facilities.  

There exists a pressing need in the art for improved methods of meat authentication that can overcome these limitations. The food industry requires rapid, reliable, and cost-effective analytical techniques capable of distinguishing between different meat varieties without extensive sample preparation. Such methods would enable real-time quality control, prevent food fraud, and ensure product authenticity throughout the supply chain. The present invention addresses these needs by providing a novel analytical approach based on GC-IMS technology combined with advanced multivariate statistical analysis.  

## SUMMARY  

The present invention provides a novel method for identifying and authenticating meat products, particularly donkey meat varieties, through analysis of their characteristic volatile organic compound (VOC) profiles. The method comprises three principal steps: sample treatment, sample analysis using gas chromatography-ion mobility spectroscopy (GC-IMS), and data analysis through multivariate statistical techniques.  

In the sample treatment step, meat specimens are prepared by placing a standardized quantity of sample into a headspace vial and incubating under controlled temperature and agitation conditions. This gentle preparation method preserves the native VOC profile without requiring chemical extraction or thermal degradation steps. The headspace gas containing the volatile compounds is then automatically injected into the GC-IMS system for analysis.  

The sample analysis step employs a GC-IMS apparatus configured with specific operational parameters optimized for meat VOC profiling. The system separates volatile compounds through gas chromatography using a programmed carrier gas flow, followed by detection via ion mobility spectroscopy under precisely controlled drift tube conditions. This combination provides two-dimensional separation of compounds based on both chromatographic retention time and ion mobility drift time, creating distinctive VOC fingerprints for each meat sample.  

The data analysis step involves processing the GC-IMS spectra through specialized software to generate comparative fingerprints and perform multivariate statistical analysis. Principal component analysis (PCA), partial least squares discriminant analysis (PLS-DA), and orthogonal PLS-DA (OPLS-DA) are employed to differentiate meat varieties based on their VOC profiles. The method identifies specific marker compounds that serve as characteristic indicators for particular meat types, enabling accurate authentication.  

Key advantages of the present method include its rapid analysis time, typically requiring less than 30 minutes per sample, compared to several hours for conventional GC-MS methods. The technique provides stable, reproducible results without complex sample pretreatment steps. The GC-IMS system offers convenient operation and lower maintenance requirements compared to traditional mass spectrometry-based approaches.  

The invention finds application in multiple fields including food quality control, product authentication, regulatory compliance, and supply chain monitoring. Specific applications include distinguishing between different varieties of donkey meat, detecting adulteration in meat products, verifying geographical origin claims, and monitoring product consistency in meat processing operations.  

## DETAILED DESCRIPTION OF THE EMBODIMENTS  

The following detailed description provides specific embodiments of the invention, though it will be understood that various modifications may be made without departing from the scope of the invention.  

**Sample Treatment Embodiment**  
In a preferred embodiment, the sample treatment process begins with obtaining approximately 1.5 grams of meat tissue, preferably from the longissimus dorsi muscle between the 17th and 18th ribs. The sample is placed into a 20 mL headspace glass vial and immediately sealed to prevent VOC loss. The vial is then incubated at 60°C for 15 minutes with constant agitation at 500 rpm to facilitate the release of volatile compounds into the headspace. After equilibration, 500 μL of the headspace gas is automatically injected into the GC-IMS system using a robotic sampling arm. This gentle preparation method preserves the native VOC profile without requiring chemical extraction or derivatization steps.  

**GC Conditions**  
The gas chromatography conditions are optimized for separation of meat-derived volatile compounds. The system employs a capillary column (MXT-5, 15 m × 0.53 mm × 1.0 μm) maintained at a constant temperature of 40°C. High-purity nitrogen (≥99.999%) serves as the carrier gas with a programmed flow rate: initial flow of 2 mL/min for 2 minutes, followed by a linear increase to 20 mL/min over 8 minutes, and finally a ramp to 100 mL/min over 10 minutes. The injector temperature is set to 85°C to ensure complete volatilization of analytes without thermal degradation.  

**IMS Conditions**  
The ion mobility spectroscopy component operates with a 9.8 cm drift tube maintained at 60°C. The drift gas consists of high-purity nitrogen flowing at 150 mL/min. A voltage of 5 kV is applied across the drift tube, and ionization is performed in positive ion mode using a 3H radioactive source. The drift tube temperature is maintained at 45°C to ensure consistent ion mobility measurements. These conditions provide optimal resolution for detecting and differentiating meat-derived volatile compounds.  

**Analysis Procedure**  
The GC-IMS analysis produces three-dimensional data comprising retention time, drift time, and signal intensity for each detected compound. The raw data are processed using specialized software to generate topographic plots and comparative fingerprints. Each sample is analyzed in triplicate to ensure reproducibility, and results are expressed as mean values with standard error measurements. Quality control samples are run periodically to monitor system performance and ensure data reliability.  

**Data Analysis Embodiment**  
The data analysis employs multivariate statistical methods to differentiate meat varieties based on their VOC profiles. Initial processing includes normalization and scaling of the GC-IMS data to account for variations in absolute signal intensities. The processed data are then subjected to principal component analysis (PCA) to visualize sample clustering patterns and identify major sources of variation. Supervised methods including partial least squares discriminant analysis (PLS-DA) and orthogonal PLS-DA (OPLS-DA) are applied to maximize separation between predefined sample groups and identify the most discriminatory volatile compounds.  

**Fingerprint Comparison**  
The VOC fingerprints of different meat samples are compared using specialized visualization tools. The Gallery Plot plug-in generates comparative fingerprints that highlight differences in specific volatile compounds between samples. Differential components are identified based on variations in signal intensity and spatial distribution within the two-dimensional GC-IMS spectra. Characteristic marker compounds are selected based on their consistent presence and abundance patterns within specific meat varieties.  

**Dynamic Principal Component Analysis**  
The dynamic PCA approach incorporates time-dependent variations in VOC profiles to enhance discrimination between meat varieties. This method tracks changes in principal component scores over the course of the chromatographic run, identifying specific retention time windows that contain the most discriminatory information. The dynamic approach improves classification accuracy by focusing analysis on the most relevant portions of the chromatogram.  

**Identification of Donkey Meat Lines**  
Application of the method to donkey meat authentication has identified 17 differential volatile compounds that distinguish between SanFen (SF) and WuTou (WT) donkey varieties. The SF donkey meat shows significantly higher levels of 2-butanol, 2-methyl-1-propanol, benzaldehyde, ethanol, isopropyl alcohol, acetone, and 2-pentanone-m. In contrast, WT donkey meat contains elevated levels of tetrahydrofurane, hexanal-m, pentan-1-ol-d, pentan-1-ol-m, 3-octenal, oct-1-en-3-ol, pentanal-d, (e)-hept-2-enal, pentanal-m, and hexanal-d. These marker compounds serve as reliable indicators for authenticating donkey meat varieties.  

**Example Results**  
In experimental validation, the method successfully differentiated SF and WT donkey meat samples with high accuracy. PCA analysis showed clear separation between the two varieties in the score plots, with the first two principal components explaining over 85% of total variance. OPLS-DA models demonstrated excellent predictive capability (Q2 > 0.7) and passed permutation testing, confirming model validity. Heatmap visualization of the 17 marker compounds revealed distinct clustering patterns that correlated perfectly with the known sample origins.  

**Summary of Advantages**  
The disclosed embodiment offers several advantages over conventional meat authentication methods. The GC-IMS analysis requires no sample pretreatment beyond simple headspace equilibration, preserving the native VOC profile. The total analysis time of approximately 20 minutes represents a significant improvement over traditional GC-MS methods. The two-dimensional separation provided by GC-IMS enhances compound discrimination compared to single-dimension techniques. The combination with multivariate statistical analysis enables robust classification based on comprehensive VOC patterns rather than single markers.  

This concludes the detailed description of the embodiments. The examples provided illustrate the practice of the invention but should not be construed as limiting its scope. Various modifications and equivalent processes may be employed without departing from the spirit of the invention.