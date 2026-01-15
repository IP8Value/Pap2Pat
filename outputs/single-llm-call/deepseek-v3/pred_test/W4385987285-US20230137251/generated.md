Here is the complete patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

### 1. Field of the Invention  

The present invention relates to biological sensors and wireless sensor systems for detecting antibiotic susceptibility in bacterial cultures. More specifically, the invention pertains to a wirelessly magnetically coupled LC (inductor-capacitor) sensor system capable of rapidly monitoring bacterial growth and determining antibiotic susceptibility through changes in the complex permittivity of the culture medium. The system is designed for high-throughput phenotypic antibiotic susceptibility testing (AST) and can be integrated into standard 96-well plate formats, enabling rapid detection of antimicrobial resistance (AMR) within 30 minutes.  

### 2. Description of the Related Art  

Drug-resistant infections present significant challenges in clinical settings, particularly in intensive care units (ICUs), where approximately 30% of patients in high-income countries are affected. The prevalence is two to three times higher in low- and middle-income countries. Recent reports from the World Health Organization (WHO) indicate that 2.8 million AMR infections occur annually in the United States alone, with at least 1.27 million global deaths attributed to AMR in 2019. Without effective interventions, AMR-related deaths could rise to 10 million per year by 2050. Rapid detection of AMR is critical for improving patient outcomes, particularly in sepsis cases, where delayed antibiotic treatment drastically reduces survival rates.  

Current methods for pathogen detection rely on culture-based techniques, which typically require a median growth time of 13 hours to achieve microbial concentrations of 10^7–10^8 CFU/mL for further analysis. Traditional phenotypic AST methods, such as broth microdilution, agar disk diffusion, and gradient diffusion tests, often take 1–2 days to yield reliable results. Genotypic ASTs, including qPCR, whole-genome sequencing, and MALDI-TOF, offer faster turnaround times (hours) but are limited by their reliance on known resistance biomarkers, rendering them ineffective against novel resistance mechanisms.  

Emerging approaches, such as optical imaging in microfluidic devices, pH sensors for metabolic byproduct detection, bioluminescent ATP assays, and antibody-coated magnetic or electrochemical biosensors, provide high sensitivity and specificity. However, these methods often require complex sample preparation, extensive data processing, or sophisticated instrumentation, limiting their clinical utility. Additionally, single-cell monitoring techniques struggle with polymicrobial infections, further motivating the need for a rapid, high-throughput phenotypic AST system that is scalable and compatible with clinical workflows.  

## BRIEF SUMMARY OF THE INVENTION  

The present invention introduces a wirelessly magnetically coupled LC sensor system for rapid phenotypic AST. The system exploits the capacitive nature of bacterial cultures to monitor growth and antibiotic susceptibility without requiring sample enrichment or species-specific surface modifications. The sensor operates by detecting changes in the complex permittivity of the culture medium, which correlate with bacterial proliferation or inhibition in the presence of antibiotics.  

Key advantages of the invention include its wireless operation, elimination of integrated power requirements, and compatibility with 96-well plate formats for high-throughput testing. The system achieves detection within 30 minutes, significantly faster than traditional phenotypic methods, and functions effectively in the presence of host proteins, making it suitable for clinical applications. The sensor design incorporates an interdigitated capacitor (IDC) and a coil for inductive coupling, enabling passive monitoring of bacterial growth through resonant frequency shifts in the LC circuit.  

## DETAILED DESCRIPTION OF THE INVENTION  

The invention comprises a system for monitoring bacterial treatment response using a wirelessly coupled LC sensor. The system includes a sensor embedded in a 96-well plate, a detection coil connected to an impedance analyzer, and signal processing algorithms to derive the complex permittivity of the bacterial culture. The permittivity changes are calculated using the following equation derived from sensor parameters and resonant frequency data:  

\[
\varepsilon = \frac{C_{1}}{k\varepsilon_{0}} - \varepsilon_{\mathit{sub}} + \frac{1}{kR_{1}\omega_{zero - inductance}\varepsilon_{0}}
\]  

Here, \(\varepsilon_{0}\) represents the permittivity of free space, \(\varepsilon_{\mathit{sub}}\) is the substrate permittivity, \(\omega_{zero - inductance}\) is the zero-inductance frequency, and \(C_{1}\), \(R_{1}\) are components of the sensor circuit. The cell constant \(k\) of the IDC is defined by:  

\[
k = \frac{\text{l}\left( {N, - ,1} \right)K\left\lbrack {1, - ,\left( \frac{D}{D + W} \right)^{2}} \right\rbrack^{\frac{1}{2}}}{\left( {2,K,\lbrack,\frac{D}{D + W}} \right\rbrack}
\]  

Signal processing involves analyzing the resonant circuit's impedance, where the total impedance \(Z_{\mathit{total}}\) is derived from the mutual inductance \(M\) and sensor impedance \(Z_{\mathit{sensor}}\):  

\[
Z_{\mathit{total}} = Z_{\mathit{int}} + \frac{\omega^{2}M^{2}}{Z_{\mathit{sensor}}}
\]  

The sensor impedance \(Z_{\mathit{sensor}}\) is expressed in the frequency domain as:  

\[
Z_{\mathit{sensor}} = j\omega L_{2} + \frac{R_{1}}{1 + j\omega R_{1}C_{1}}
\]  

By solving these equations, the system determines the resonant and zero-reactance frequencies, enabling real-time monitoring of bacterial growth through permittivity changes.  

### Example  

#### Bacterial Media, Reagents, and Materials  
The system utilizes low-salt LB medium (0.5 g/L NaCl, 10 g/L tryptone, 5 g/L yeast extract) supplemented with 1–5% fetal bovine serum (FBS) to mimic clinical samples. Antibiotics tested include ampicillin, ofloxacin, ciprofloxacin, vancomycin, and tobramycin. Sensors are fabricated on polyimide flex circuit boards and coated with oil-based polyurethane for protection.  

#### Bacterial Sample Preparation  
Overnight cultures are diluted to an OD600 of 0.001 in low-salt LB and aliquoted into 96-well plates containing LC sensors. The plates are incubated at 30°C without agitation, and permittivity measurements are taken every 5 minutes.  

#### Sensor Design and Optimization  
The sensor features a 50-turn coil (0.035 mm wire thickness, 0.06 mm wire width/gap) and an IDC with parameters optimized for sensitivity. COMSOL simulations confirm effective magnetic coupling through the polystyrene well plate.  

#### Bacterial Growth Monitoring and AST  
Growth curves for E. coli, S. aureus, and P. aeruginosa demonstrate rapid detection (30 minutes) of antibiotic susceptibility. Linear regression of permittivity slopes distinguishes resistant and sensitive strains, with results validated by microdilution MIC tests.  

#### Performance with Host Proteins  
The system functions with 2% FBS, demonstrating feasibility for clinical samples. Higher FBS concentrations (5%) saturate sensor surfaces, but design adjustments (e.g., increased digit spacing) can mitigate this.  

#### Future Optimizations  
Further refinements include lithography for miniaturization, microfluidic integration, and coating enhancements to improve fouling resistance. The system is scalable for high-throughput clinical use, with potential for automated plate scanning and reference library integration for MIC determination.  

The invention offers significant advantages over existing phenotypic and genotypic AST methods, including rapid turnaround, simplicity, and compatibility with complex samples. Its wireless, label-free operation makes it a promising tool for combating antimicrobial resistance in clinical and laboratory settings.