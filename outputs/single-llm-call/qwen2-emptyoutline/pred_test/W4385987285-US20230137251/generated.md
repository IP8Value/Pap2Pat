# DESCRIPTION

## BACKGROUND OF THE INVENTION

### 1. Field of the Invention

The present invention relates to a rapid and high-throughput method for antibiotic susceptibility testing (AST) using inductively coupled LC sensors. More specifically, the invention pertains to a system and method for detecting bacterial growth and assessing antibiotic susceptibility within 30 minutes, thereby providing a critical tool for rapid diagnosis and treatment of drug-resistant infections.

### 2. Description of the Related Art

Drug-resistant infections pose significant challenges, particularly in hospital settings. Approximately 30% of intensive care unit (ICU) patients are affected by these infections, with the incidence being even higher in low- and middle-income countries. According to the World Health Organization (WHO), approximately 2.8 million antibiotic-resistant infections occur annually in the United States alone, and at least 1.27 million people died from antibiotic-resistant infections in 2019. Projections suggest that by 2050, there could be 10 million deaths per year globally if effective treatments for antibiotic resistance are not developed.

Rapid detection of antibiotic resistance is crucial for reducing sepsis mortality and improving antibiotic stewardship programs. Traditional methods of pathogen detection through sample cultures typically take around 13 hours to yield a microbial culture with a sufficient concentration for further analysis. Without rapid AST, patients often receive broad-spectrum antibiotics, which can lead to ineffective treatments and missed opportunities to prevent mortality.

Phenotypic methods such as dilution methods, agar disk diffusion testing, and gradient diffusion methods typically take 1-2 days to produce reliable results. Genotypic ASTs, which detect biomarkers associated with resistance using molecular tools like quantitative PCR (qPCR), whole-genome sequencing, and matrix-assisted laser desorption/ionization time-of-flight mass spectrometry (MALDI-TOF), can produce results in hours. However, these methods require detailed knowledge of antibiotic resistance gene sequences and cannot detect newly developed resistance mechanisms.

Other novel approaches, such as optical imaging, pH sensors, bioluminescence assays, and magnetic or electrochemical biosensors, offer high sensitivity and specificity but often require extensive image processing, complex sample preparation, and sophisticated equipment, making them challenging to implement in clinical settings. Additionally, ASTs based on single-cell detection face challenges in polymicrobial infections with mixed microbial populations.

To address these unmet needs, the present invention provides a rapid, high-throughput phenotypic AST method using inductively coupled LC sensors. This method can detect bacterial growth and assess antibiotic susceptibility within 30 minutes, requiring minimal sample preparation and no species-specific surface modifications. The system is designed to be low-cost, easily integrable into clinical settings, and capable of high-throughput screening.

## BRIEF SUMMARY OF THE INVENTION

The present invention discloses a novel method and system for rapid antibiotic susceptibility testing (AST) using inductively coupled LC sensors. The system comprises a 96-well plate with LC sensors inserted into each well, a receiver coil connected to an impedance analyzer, and a signal processing unit. The LC sensors are designed to exploit the capacitive nature of bacteria, enabling the detection of bacterial growth and antibiotic susceptibility without the need for sample enrichment or species-specific surface modifications.

The method involves diluting a bacterial culture and aliquoting it into the 96-well plate with the LC sensors. The receiver coil, connected to an impedance analyzer, wirelessly communicates with the LC sensors to monitor the resonant frequency shifts caused by bacterial growth. The permittivity of the bacterial culture is calculated from the resonant frequency data, and the growth curve is plotted to determine the antibiotic susceptibility of the bacterial strain within 30 minutes.

The invention offers several advantages over existing AST methods, including rapid detection, high throughput, low cost, and ease of integration into clinical settings. The system can be used to screen multiple bacterial strains and antibiotics simultaneously, providing valuable information for antibiotic stewardship and patient care.

## DETAILED DESCRIPTION OF THE INVENTION

### Example

The present invention is directed to a rapid and high-throughput method for antibiotic susceptibility testing (AST) using inductively coupled LC sensors. The system and method are designed to detect bacterial growth and assess antibiotic susceptibility within 30 minutes, making it a valuable tool for rapid diagnosis and treatment of drug-resistant infections.

#### Sensor Design

The LC sensors used in the present invention are designed to exploit the capacitive nature of bacteria. Each sensor consists of a coil and an interdigitated capacitor (IDC) fabricated on a flexible polyimide flex circuit board. The coil and IDC are arranged to form a resonant circuit, and the resonant frequency of the circuit changes in response to the surrounding environment, allowing the detection of bacterial growth.

The initial design of the sensor comprised a fiberglass printed circuit board with 5 turns of coil on both sides of the PCB and 11 0.5 mm digits spaced 0.5 mm apart. However, this design was found to have a slower detection speed, prompting a redesign to achieve faster detection. Sensitivity analysis using the method of Morris revealed that the number of turns in the coil had the largest influence on the permittivity value, followed by the outer diameter of the coil and the distance between digits. Based on this analysis, the final sensor design features a high coil turn count and a high quantity of IDC digits, enabling rapid detection with high sensitivity.

The wire dimensions were intentionally kept large enough to allow for conventional circuit printing techniques, ensuring easy sensor fabrication with existing manufacturing processes. The sensor is mounted on a double-sided adhesive sheet and coated with a polyurethane protective layer to prevent interference from protein and bacterial settling effects in a static culture environment.

#### Bacterial Growth Monitoring and AST

The system uses low salt LB medium to minimize the conductive nature of the media, which could interfere with the sensor. The high resistance introduced by low salt LB provides a lower background level, making the resonant frequency shift in the system more detectable. Baseline readouts of sterile medium are established and subtracted as background from the experimental data.

Initial growth monitoring tests were conducted with Escherichia coli (E. coli) MG1655, Staphylococcus aureus (S. aureus) ALC2085, and Pseudomonas aeruginosa (P. aeruginosa) PAO1. The permittivity values were normalized to a 0-1 scale, and the growth curves were plotted at 5-minute intervals. The results showed that the permittivity method was more sensitive in detecting bacterial growth compared to optical density (OD600) measurements, with initial signs of growth detected within 30 minutes.

To determine the antibiotic susceptibility of the bacterial strains, ampicillin-sensitive and ampicillin-resistant strains of E. coli were tested. The ampicillin-resistant strain (E. coli MG1655 ASV) grew in the presence of ampicillin, while the sensitive strain (E. coli MG1655) was inhibited, as expected. The growth curves and permittivity data were consistent with the results obtained from OD600 measurements and colony-forming unit (CFU) counts, validating the accuracy of the sensor system.

Subsequent experiments were conducted with additional antibiotics and bacterial species, including vancomycin, ofloxacin, ciprofloxacin, and tobramycin. The test time was set to 30 minutes with 5-minute intervals of sampling, as all tested growth curves showed initial signs of resistance or growth within this time frame. Linear regression was used to analyze the relative permittivity data, and the slope of the best-fit line was used to determine the sensitivity of the tested strain toward an antibiotic.

#### Performance in the Presence of Host Proteins

Clinical samples often contain complex components, such as proteins and cells, which can interfere with the sensor system. To evaluate the feasibility of implementing the system in a clinical setting, the sensor was tested with the addition of fetal bovine serum (FBS) to mimic host serum. The results showed that the sensor functioned well at lower concentrations of FBS, capturing the growth of E. coli wild-type strain MG1655 with little interference. At higher concentrations of FBS, the response was lost, likely due to protein adsorption saturating the free space near the sensor surface.

The coating material and thickness of the chips can be optimized to improve accuracy and resistance to fouling from the growth media or cellular products. A better-optimized coating could further decrease the detection limit and broaden the detection range. The saturation thickness can be described by the following equations, where \(a_1\) and \(a_2\) are functions of digit width \(W\) and gap width \(D\):

\[
d_{\text{sat}} = - \frac{D}{a_1} \ln\left( \frac{0.005}{a_2} \right)
\]
\[
a_1 = 114.97(W + D)^3 + 28.75(W + D)^2 - 9.183(W + D) + 1.631
\]
\[
a_2 = 1293.21(W + D)^3 + 164.87(W + D)^2 - 6.521(W + D) + 6.105
\]

With a thicker free space for sensing, the system can be further optimized to handle more complex samples and longer-term growth patterns. Additional testing with clinical samples is needed to further validate the system and set the thresholds for AMR detection.

#### Conclusion

The present invention provides a rapid and high-throughput method for antibiotic susceptibility testing using inductively coupled LC sensors. The system can detect bacterial growth and assess antibiotic susceptibility within 30 minutes, offering a significant advantage over existing methods. The method is low-cost, requires minimal sample preparation, and can be easily integrated into clinical settings. The system can be further optimized for high-throughput screening and improved performance in the presence of complex clinical samples. This invention has the potential to revolutionize the field of AST, providing a critical tool for rapid diagnosis and treatment of drug-resistant infections.