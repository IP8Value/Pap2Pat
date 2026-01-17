# DESCRIPTION

## BACKGROUND OF THE INVENTION

The present invention relates to a method for the analysis of N-linked glycans from human serum for the purpose of diagnosing ovarian cancer. Specifically, the invention provides a method for isolating and analyzing N-linked glycans from human serum using mass spectrometry, and a system for classifying ovarian cancer patients from healthy controls based on the analysis of these glycans.

Ovarian cancer is a significant health issue, with a high mortality rate due to late-stage diagnosis. Current biomarkers, such as CA-125, have limitations in detecting early-stage ovarian cancer. Therefore, there is a critical need for more effective biomarkers and diagnostic methods. Glycans, which are carbohydrate chains attached to proteins, have been shown to be associated with various biological processes, including cancer. N-linked glycans, in particular, have been identified as potential biomarkers for ovarian cancer due to their altered patterns in cancer patients.

Recent advances in mass spectrometry and solid phase extraction techniques have enabled the isolation and detailed analysis of N-linked glycans from human serum. These techniques provide a robust platform for the development of multibiomarker panels that can improve the accuracy of ovarian cancer diagnosis.

## SUMMARY OF THE INVENTION

The present invention provides a method for diagnosing ovarian cancer by analyzing N-linked glycans from human serum. The method includes the following steps:

1. **Isolation of N-Linked Glycans**: N-linked glycans are isolated from human serum using a combination of denaturation, enzymatic release, and solid phase extraction.
2. **Mass Spectrometric Analysis**: The isolated glycans are analyzed using matrix-assisted laser desorption/ionization time-of-flight mass spectrometry (MALDI-TOF MS).
3. **Data Processing and Biomarker Selection**: The mass spectrometric data is processed to identify potential biomarkers based on statistical analysis, such as ANOVA and receiver operating characteristic (ROC) curves.
4. **Classification System**: A scoring system is developed to classify samples as either ovarian cancer patients or healthy controls based on the identified biomarkers.

The invention also provides a system for implementing the method, including software and hardware components for data acquisition, processing, and classification.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS

### Glycan Analysis from Human Serum

The method for analyzing N-linked glycans from human serum involves several key steps to ensure the accurate and reliable identification of biomarkers for ovarian cancer.

#### Step 1: Sample Collection and Preparation
Human serum samples are collected from patients and healthy controls. Blood is drawn into tubes and allowed to clot at room temperature. The clotted blood is then centrifuged at 1000 × g for 10 minutes to separate the serum. The serum is transferred to clean polypropylene tubes and stored at -80°C until further processing.

#### Step 2: Denaturation and Enzymatic Release of N-Linked Glycans
Fifty microliters of serum is mixed with 50 μL of 200 mM NH4HCO3 containing 10 mM dithiothreitol (DTT). The mixture is subjected to moderate protein denaturation by cycling between boiling water and room temperature for 2 minutes. Peptide-N-glycosidase F (PNGase F) is added to release N-linked glycans from the denatured proteins. The reaction is carried out in a microwave-mediated enzyme reaction enhancing system for 10 minutes at 37°C with 400 W of microwave power output.

#### Step 3: Solid Phase Extraction (SPE)
After the enzymatic release, proteins are precipitated by adding ice-cold ethanol and incubating at -80°C for 1 hour. The supernatant containing the released glycans is collected and dried. The glycans are then subjected to SPE using graphitized carbon cartridges. The cartridges are washed and equilibrated with 0.1% trifluoroacetic acid (TFA) in 80% acetonitrile/H2O. The glycan solutions are loaded onto the cartridges and washed with Nanopure water. The glycans are eluted sequentially with 10%, 20%, and 40% acetonitrile/H2O containing 0.05% TFA. The eluted fractions are collected, dried, and dissolved in purified water.

#### Step 4: Mass Spectrometric Analysis
The glycan solutions are analyzed using MALDI-TOF MS. One microliter of the glycan solution is mixed with 1 μL of 70% acetonitrile/12 mM NaCl solution and spotted onto a prespotted MALDI target plate. The samples are dried in a vacuum, and mass spectra are acquired in the positive-ion reflectron mode with an m/z range from 1000 to 3000. Each sample is analyzed in quadruplicate, and the data are calibrated using internal calibrants.

#### Step 5: Data Processing and Biomarker Selection
The mass spectrometric data are processed using software to extract ion peak information. The centroid m/z values and absolute ion peak intensities are tabulated. The data are normalized by dividing each absolute peak intensity by the sum of total peak intensities in the spectrum. Potential biomarkers are selected based on the P values from ANOVA and the area under the ROC curve (AUC). Biomarkers with P values below 10^-9 and AUC values above 0.72 (for 10% ACN/H2O fraction) and 0.75 (for 20% ACN/H2O fraction) are chosen.

#### Step 6: Classification System
A scoring system is developed to classify samples based on the identified biomarkers. Each biomarker has a cut-off value determined from its ROC curve. The ion peak intensity information from each biomarker is converted to a score of +1 (positive for ovarian cancer) or -1 (negative for ovarian cancer). The scores are weighted by the AUC of each biomarker and summed to obtain a total score for each sample. The diagnostic cut-off values for the total score are determined in the training set.

### Example 2
In a study involving 40 healthy controls and 40 ovarian cancer patients, the method was used to develop a multibiomarker panel. The training set was used to select 15 biomarkers, 5 from the 10% ACN/H2O fraction and 10 from the 20% ACN/H2O fraction. The classification efficiency of the biomarkers was evaluated using the area under the ROC curve (AUC). The combined multibiomarker panel achieved an AUC of 0.89 in the training set, with 80-90% sensitivity and 70-83% specificity.

### Example 3
The predictive power of the biomarker panel was evaluated in a blind test set of 60 unknown samples, including 23 healthy controls and 37 ovarian cancer patients. The classification system correctly identified 81-84% of the ovarian cancer patients with 83% specificity. The sensitivity of the biomarker panel was higher than that of CA-125, which achieved 78% sensitivity in the same blind test set.

### Example 4
The method was also tested on a set of 8 ovarian cancer patients who experienced recurrence after initial surgery. All 8 patients were successfully classified as patients, demonstrating the potential of the method for monitoring ovarian cancer recurrence.

### Example 5
The method was further evaluated for its ability to classify different histological types of ovarian cancer. Clear-cell type ovarian cancer showed the highest sensitivity, with the biomarker panel achieving 20% better sensitivity than CA-125. Serous type ovarian cancer also demonstrated slightly better or similar sensitivity to CA-125. The method showed promise for improving the diagnosis of various types of ovarian cancer.

The present invention provides a robust and accurate method for diagnosing ovarian cancer using N-linked glycan analysis. The method offers significant improvements over existing biomarkers and has the potential to enhance early detection and monitoring of ovarian cancer.