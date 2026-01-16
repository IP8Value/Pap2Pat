# DESCRIPTION

## BACKGROUND OF THE INVENTION

The present invention relates generally to the field of molecular biology and, more specifically, to methods and systems for determining copy number variations (CNVs) in DNA samples using digital PCR and digital arrays. CNVs, which are the gains or losses of genomic regions ranging from 500 base pairs to several megabases, have been extensively studied due to their significant roles in human genetic disorders and genetic diversity. Traditional methods for studying CNVs, such as array-based comparative genomic hybridization (array-CGH) and high-density single nucleotide polymorphism (SNP) microarrays, suffer from limitations in resolution and sensitivity. Real-time PCR, while easier to perform, lacks the discriminating power necessary for precise CNV determination beyond a 2-fold difference.

The invention addresses these limitations by utilizing a digital array, a novel nanofluidic biochip that partitions DNA molecules into hundreds or thousands of nanoliter-volume reaction chambers. This allows for the accurate quantification of single DNA target molecules, enabling the determination of CNVs with high precision and reliability. The digital array, combined with multiplex PCR, allows for the simultaneous and independent quantification of multiple genes, effectively eliminating pipetting errors and improving the accuracy of CNV measurements.

## SUMMARY OF THE INVENTION

The present invention provides a robust and easy-to-use platform for studying copy number variations (CNVs) using digital PCR and digital arrays. The invention includes a method for estimating the true concentration of target DNA molecules in a sample and calculating the ratio of true concentrations of multiple sequences, along with the associated confidence intervals. The method involves partitioning a PCR reaction mixture into a large number of nanoliter-volume reaction chambers, thermocycling the chambers, and counting the number of positive chambers to estimate the true concentration of the target molecules.

The key aspects of the invention are as follows:

1. **Partitioning and Amplification**: The DNA sample is partitioned into a plurality of nanoliter-volume reaction chambers using a digital array. Each chamber contains a mixture of sample and reagents, and the DNA molecules are randomly distributed among the chambers. The digital array is then thermocycled to amplify the DNA molecules in each chamber.

2. **Counting Positive Chambers**: After amplification, the number of positive chambers, which contain one or more target DNA molecules, is counted. The ratio of positive chambers to the total number of chambers provides an estimate of the true concentration of the target molecules.

3. **Mathematical Framework**: A mathematical framework is provided to calculate the true concentration of the target molecules from the observed number of positive chambers. The framework uses the relationship between the probability of a chamber being positive and the true concentration of the target molecules, which is modeled as a Poisson process.

4. **Confidence Intervals**: The invention also provides a method for calculating the 95% confidence intervals for the true concentration of the target molecules and the ratio of true concentrations of multiple sequences. This is achieved using statistical sampling and estimation theories, including the application of Fieller's Theorem and numerical algorithms for arbitrary sampling distributions.

5. **Multiplex PCR**: The invention further includes the use of multiplex PCR to simultaneously quantify multiple genes in a single reaction. This allows for the determination of the ratio of true concentrations of the target gene to a reference gene, which reflects the copy number per haploid genome of the target gene.

6. **Applications**: The invention is particularly useful for studying CNVs in human and other organisms, identifying genetic disorders, and understanding genetic diversity. The digital array provides a high-resolution, high-sensitivity platform that is easy to use and robust.

## DETAILED DESCRIPTION OF SPECIFIC EMBODIMENTS

### Partitioning and Amplification

The digital array used in the present invention is a nanofluidic biochip that partitions a PCR reaction mixture into a large number of nanoliter-volume reaction chambers. Each chamber contains a mixture of sample and reagents, and the DNA molecules are randomly distributed among the chambers. The digital array utilized in this invention typically consists of 765 chambers per panel, and up to 12 panels can be used simultaneously. The total volume of the PCR mix in each panel is 4.59 µl (6 nl × 765).

The DNA sample is mixed with PCR reagents, including primers and probes specific to the target and reference genes. The mixture is then loaded into the digital array, and the chambers are sealed. The digital array is thermocycled using a real-time PCR system, such as Fluidigm's BioMark system. The thermocycling conditions typically include a 95°C, 10-minute hot start followed by 40 cycles of two-step PCR: 15 seconds at 95°C for denaturation and 1 minute at 60°C for annealing and extension.

### Counting Positive Chambers

After thermocycling, the chambers are imaged to identify the positive chambers. A positive chamber is one that contains one or more target DNA molecules and has produced a detectable signal, typically through the fluorescence of a reporter dye. The number of positive chambers is counted using digital PCR analysis software, which processes the data and identifies the chambers that have amplified the target DNA.

### Mathematical Framework

The relationship between the probability of a chamber being positive and the true concentration of the target molecules is modeled as a Poisson process. Let \( \lambda \) be the true concentration of the target molecules per chamber. The probability \( p \) of a chamber being positive is given by:

\[ p = 1 - e^{-\lambda} \]

Given the number of positive chambers \( H \) and the total number of chambers \( C \), the estimated probability \( \hat{p} \) is:

\[ \hat{p} = \frac{H}{C} \]

The estimated true concentration \( \hat{\lambda} \) is then:

\[ \hat{\lambda} = -\ln(1 - \hat{p}) \]

### Confidence Intervals

To calculate the 95% confidence interval for the true concentration \( \lambda \), the sampling distribution of \( \hat{p} \) is approximated using the normal distribution. The confidence limits for \( \hat{p} \) are given by:

\[ \hat{p} \pm z_c \sqrt{\frac{\hat{p}(1 - \hat{p})}{C}} \]

where \( z_c \) is the critical value for the desired confidence level (1.96 for 95% confidence). The confidence interval for \( \lambda \) is then:

\[ \left[ -\ln(1 - (\hat{p} - z_c \sqrt{\frac{\hat{p}(1 - \hat{p})}{C}})), -\ln(1 - (\hat{p} + z_c \sqrt{\frac{\hat{p}(1 - \hat{p})}{C}})) \right] \]

### Ratio of Concentrations

To determine the ratio of true concentrations of two genes, the invention uses the estimated concentrations \( \hat{\lambda}_1 \) and \( \hat{\lambda}_2 \) of the target and reference genes, respectively. The ratio \( r \) is given by:

\[ r = \frac{\hat{\lambda}_1}{\hat{\lambda}_2} \]

The 95% confidence interval for the ratio \( r \) is calculated using Fieller's Theorem. The confidence region is constructed by finding the tangents to the confidence ellipse in the two-dimensional plane defined by the sampling distributions of \( \hat{\lambda}_1 \) and \( \hat{\lambda}_2 \). The confidence interval for \( r \) is then:

\[ \left[ \frac{\hat{\lambda}_1 - z_c \sigma_1}{\hat{\lambda}_2 + z_c \sigma_2}, \frac{\hat{\lambda}_1 + z_c \sigma_1}{\hat{\lambda}_2 - z_c \sigma_2} \right] \]

where \( \sigma_1 \) and \( \sigma_2 \) are the standard deviations of the sampling distributions of \( \hat{\lambda}_1 \) and \( \hat{\lambda}_2 \), respectively.

### Multiplex PCR

The invention further includes the use of multiplex PCR to simultaneously quantify multiple genes in a single reaction. This is achieved by using separate PCR primers and probes for each gene. The multiplex PCR reaction mixture is loaded into the digital array, and the chambers are thermocycled as described above. The positive chambers for each gene are counted separately, and the ratio of true concentrations of the target gene to the reference gene is calculated using the methods described above.

### Applications

The invention is particularly useful for studying CNVs in human and other organisms. It can be used to identify genetic disorders, understand genetic diversity, and provide high-resolution, high-sensitivity data for various applications in molecular biology and genetics. The digital array provides a robust and easy-to-use platform that is suitable for both research and clinical settings.

### Example Experiment

To demonstrate the effectiveness of the invention, a spike-in experiment was performed using a synthetic construct to simulate CNVs. A 65-base oligonucleotide identical to a fragment of the human RPP30 gene was synthesized and added to human genomic DNA at different concentrations to create mixtures with known ratios of RPP30 to RNase P. The mixtures were analyzed using the digital array, and the ratios of RPP30 to RNase P were calculated using the methods described above. The results showed that the known ratios lay within the computed 95% confidence intervals, confirming the accuracy and reliability of the invention.

In summary, the present invention provides a robust and accurate platform for studying CNVs using digital PCR and digital arrays. The invention combines advanced partitioning and amplification techniques with sophisticated mathematical and statistical methods to provide high-resolution, high-sensitivity data for a wide range of applications in molecular biology and genetics.