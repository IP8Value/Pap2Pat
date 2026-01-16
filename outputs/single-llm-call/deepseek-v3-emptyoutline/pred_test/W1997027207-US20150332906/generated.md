Here is the complete patent application following your outline:

## BACKGROUND

Matrix-assisted laser desorption/ionization time-of-flight (MALDI-TOF) mass spectrometry has emerged as a powerful analytical technique for biomolecular analysis since its development in the late 1980s. The technique combines the soft ionization capabilities of matrix-assisted laser desorption/ionization with the high mass range and rapid analysis time of time-of-flight mass analyzers. Modern MALDI-TOF instruments routinely achieve mass resolutions exceeding 15,000 (FWHM) and mass accuracies better than 5 parts per million (ppm) when using internal calibration standards. These performance characteristics make MALDI-TOF mass spectrometry particularly suitable for applications requiring high mass accuracy, such as peptide mass fingerprinting for protein identification.

Despite these impressive specifications, significant variability in mass measurement accuracy has been observed in practice. Replicate measurements of the same analyte often show mass errors ranging from less than 1 ppm to greater than 20 ppm, even when using internal calibration standards. This variability persists across instruments from different manufacturers, suggesting fundamental limitations in current MALDI-TOF instrument designs and data processing methods. The observed mass errors appear random and uncorrelated between different analytes within the same spectrum, making them particularly difficult to predict or correct.

Detailed investigation of this phenomenon has revealed that the analog-to-digital (AD) conversion process in MALDI-TOF detectors introduces discrete sampling bins that can shift position between acquisitions. Each laser shot initiates a new acquisition cycle, during which ion flight times are measured and digitized according to the instrument's internal clock. The resulting mass spectrum is constructed from these discrete time bins, which typically have widths corresponding to 13-20 ppm of the measured mass. While the bin positions are reset with sub-bin precision for each acquisition, small variations in their absolute positioning can significantly impact the apparent mass of detected ions when considering the high accuracy requirements of modern applications.

Current MALDI-TOF data processing methods typically treat each acquisition independently, applying calibration algorithms to single spectra without considering the statistical properties of replicate measurements. This approach fails to account for the random variations in bin positioning and consequently limits the achievable mass accuracy. There exists a need for improved data processing methods that can overcome these limitations and provide more consistent mass measurement accuracy in MALDI-TOF mass spectrometry.

## SUMMARY

The present invention provides methods and systems for improving mass measurement accuracy in MALDI-TOF mass spectrometry through statistical processing of replicate spectra. The disclosed approach recognizes that variations in mass measurements between replicate spectra follow a normal distribution pattern resulting from random variations in the positioning of discrete sampling bins during analog-to-digital conversion. By acquiring multiple replicate spectra and applying statistical analysis to the population of mass measurements, the invention achieves more accurate and consistent results than possible from single spectra.

Key aspects of the invention include: acquisition of multiple replicate mass spectra for the same sample; identification of corresponding analyte peaks across the replicate spectra; calculation of statistical parameters (mean, standard deviation) for the mass measurements of each analyte; and use of the statistical parameters to determine final mass values and assess measurement quality. The method further includes identification and removal of outlier measurements using statistical tests, enabling improved confidence in the final results.

The statistical processing approach provides several advantages over conventional single-spectrum analysis. First, the mean of multiple measurements provides a more accurate mass estimate than individual measurements, as random variations tend to cancel out. Second, the standard deviation of the measurements provides a quantitative indicator of measurement quality, allowing identification of potentially problematic data. Third, statistical tests can identify and remove outlier measurements that would otherwise degrade accuracy. Finally, the approach can be implemented in automated data acquisition and processing software without requiring hardware modifications to existing instruments.

Experimental results demonstrate that the method significantly improves mass measurement consistency. For example, analysis of tryptic peptides from standard proteins showed that while individual measurements varied by up to 20 ppm, the mean of 10-23 replicate measurements typically achieved errors below 5 ppm. Similar improvements were observed in the analysis of immunoprecipitated proteins, where the method enabled successful protein identification by peptide mass fingerprinting that would have been ambiguous using conventional single-spectrum analysis.

The invention is applicable to all MALDI-TOF mass spectrometers employing analog-to-digital conversion systems, including both reflector and linear mode instruments. The method can be implemented as a software modification to existing instruments, providing immediate performance improvements without hardware changes. Furthermore, the principles may be extended to other mass spectrometry techniques employing similar detection systems, such as LC-TOF and LC-Q-TOF instruments.

## DETAILED DESCRIPTION

The present invention provides methods and systems for improving mass measurement accuracy in MALDI-TOF mass spectrometry through statistical processing of replicate spectra. The following detailed description explains the principles, implementation, and applications of the invention.

**System Overview**

The invention may be implemented on any conventional MALDI-TOF mass spectrometer system comprising: a MALDI ion source for generating ions from a sample mixed with matrix material; a time-of-flight mass analyzer for separating ions according to their mass-to-charge ratios; a detector system including analog-to-digital conversion electronics for measuring ion arrival times; and a data system for controlling instrument operation and processing acquired data. The data system is configured to perform the statistical processing methods described herein, either through specialized software or modifications to existing instrument control software.

**Data Acquisition Method**

The improved mass accuracy method begins with acquisition of multiple replicate mass spectra for the same sample. In preferred embodiments, between 10 and 30 individual spectra are acquired, with each spectrum representing the sum of 100-1000 laser shots. The exact number of spectra and shots per spectrum may be optimized for specific applications, balancing the need for statistical power with practical considerations of analysis time and sample consumption.

Each individual spectrum is processed independently through the instrument's standard calibration procedure, typically using internal calibration standards mixed with the sample or spotted in close proximity. This ensures that each spectrum has its own calibration function derived from its particular set of calibration peaks, accounting for variations in calibration between acquisitions.

**Statistical Processing Method**

Following acquisition of replicate spectra, the invention performs statistical analysis of the resulting mass measurements through the following steps:

1. Peak Detection and Alignment: Corresponding analyte peaks are identified across all replicate spectra. This may be accomplished by matching peaks within a specified mass tolerance window (typically ±0.5 Da) or through more sophisticated peak alignment algorithms that account for small mass shifts between spectra.

2. Mass Measurement Extraction: For each identified analyte, the apparent mass is extracted from each spectrum where the analyte is detected. Detection thresholds may be applied to exclude measurements with insufficient signal-to-noise ratio (e.g., S/N < 5).

3. Statistical Parameter Calculation: For each analyte, the population of mass measurements is analyzed to calculate statistical parameters including:
   - Mean mass value
   - Standard deviation
   - Confidence intervals
   - Normality test results (e.g., Shapiro-Wilk test)

4. Outlier Identification and Removal: Statistical tests (e.g., Grubbs' test) are applied to identify and remove outlier measurements that fall outside expected variation ranges. This step improves the robustness of the final mass determination.

5. Final Mass Determination: The mean of the remaining measurements (after outlier removal) is taken as the final mass value for each analyte. The standard deviation provides a measure of confidence in the result.

**Implementation Considerations**

The statistical processing method may be implemented with several variations and optimizations:

- Bin Size Optimization: The method recognizes that the analog-to-digital conversion process creates discrete sampling bins with widths typically corresponding to 13-20 ppm of the measured mass. Optimal results are obtained when using bin sizes of 0.5-1.0 nanoseconds, as larger bins (e.g., 2.0 ns) begin to compromise the interpolation algorithms used in calibration.

- Calibration Strategy: While each spectrum is individually calibrated, the method benefits from using multiple internal calibration standards (typically 3-5) distributed across the mass range of interest. This provides robust calibration across the entire spectrum.

- Data Quality Assessment: The standard deviation of replicate measurements serves as a key quality metric. Measurements with standard deviations significantly higher than typical values (e.g., >0.015 amu for peptides) may be flagged as potentially unreliable.

- Automated Implementation: The entire process can be automated within instrument control software, including:
  - Automated acquisition of replicate spectra
  - Real-time assessment of data quality
  - Adaptive adjustment of acquisition parameters (e.g., number of spectra) based on observed variability
  - Automated reporting of final mass values with associated confidence metrics

**Experimental Validation**

The effectiveness of the method has been validated through extensive experimentation. Analysis of tryptic digests from standard proteins (BSA, myoglobin, β-casein) demonstrated that while individual mass measurements varied by up to 20 ppm, the mean of 10-23 replicate measurements consistently achieved errors below 5 ppm. Similar improvements were observed in the analysis of immunoprecipitated β-tubulin, where the method enabled unambiguous protein identification that would have been problematic using conventional single-spectrum analysis.

**Applications**

The invention has broad applicability across all fields utilizing MALDI-TOF mass spectrometry, including:

- Protein identification by peptide mass fingerprinting
- Microbial identification in clinical microbiology
- Biomarker discovery and validation
- Pharmaceutical quality control
- Synthetic polymer characterization
- Imaging mass spectrometry

The method is particularly valuable in applications requiring high mass accuracy, such as proteomics research and clinical diagnostics. By improving measurement consistency without requiring hardware modifications, the invention can provide immediate performance benefits to existing MALDI-TOF instruments across these diverse application areas.

**Extensions and Variations**

While described primarily in the context of MALDI-TOF mass spectrometry, the statistical processing approach may be adapted to other mass spectrometry techniques employing analog-to-digital conversion systems, including:

- LC-TOF mass spectrometry
- LC-Q-TOF mass spectrometry
- Other hybrid mass spectrometer configurations

The method may also be extended to include intensity measurements, potentially improving the consistency of quantitative analyses. However, as intensity measurements typically show greater variability than mass measurements, the benefits may be more limited in this application.

**Conclusion**

The disclosed statistical processing method represents a significant advance in MALDI-TOF mass spectrometry, providing improved mass accuracy and measurement consistency through software-based enhancements. By properly accounting for the discrete nature of analog-to-digital conversion in mass spectrometer detectors, the method overcomes fundamental limitations in current data processing approaches. The invention can be readily implemented on existing instruments through software modifications, providing immediate performance benefits across a wide range of applications.