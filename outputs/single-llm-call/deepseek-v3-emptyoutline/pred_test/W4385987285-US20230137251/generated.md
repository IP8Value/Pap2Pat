Here is the draft patent application following the provided outline:

---

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

### 1. Field of the Invention  

The present invention relates to the field of medical diagnostics, specifically to devices and methods for rapid antimicrobial susceptibility testing (AST). More particularly, the invention pertains to a wireless, inductor-capacitor (LC) resonant sensor system capable of detecting bacterial growth and determining antibiotic susceptibility within a clinically relevant timeframe. The system enables phenotypic AST without requiring prior knowledge of microbial genomic markers, complex sample preparation, or specialized laboratory equipment.  

### 2. Description of the Related Art  

Antimicrobial resistance (AMR) poses a significant global health challenge, with an estimated 1.27 million deaths attributable to resistant infections in 2019 alone. Current methods for detecting AMR suffer from critical limitations that impede timely clinical intervention. Traditional phenotypic techniques, such as broth microdilution and disk diffusion assays, typically require 1-2 days to yield results. While genotypic methods like PCR and whole-genome sequencing can provide faster detection, they are limited to known resistance markers and cannot identify novel resistance mechanisms.  

Recent advances in AST have explored optical imaging, bioluminescence assays, and electrochemical biosensors. However, these approaches often require sophisticated instrumentation, extensive data processing, or species-specific surface modifications. Single-cell monitoring techniques further struggle with polymicrobial infections. Existing LC sensor-based systems for bacterial detection exhibit prolonged response times (e.g., 8 hours) and lack optimization for clinical AST applications. There remains an unmet need for a rapid, high-throughput phenotypic AST platform that combines the comprehensive detection capabilities of traditional methods with the speed and scalability required for clinical implementation.  

## BRIEF SUMMARY OF THE INVENTION  

The present invention provides a wireless LC resonant sensor system for rapid antimicrobial susceptibility testing. The system comprises an array of miniaturized LC sensors integrated into standard multi-well plates, each sensor featuring an interdigitated capacitor (IDC) component and an inductive coil. Through magnetic coupling with an external detection coil connected to impedance spectroscopy instrumentation, the system monitors changes in resonant frequency corresponding to bacterial growth-induced permittivity variations in the culture medium.  

Key advantages of the invention include:  
- Detection of antibiotic susceptibility within 30 minutes  
- Operation without requiring bacterial culture enrichment  
- Compatibility with standard 96-well plate formats for high-throughput testing  
- Elimination of species-specific surface modifications or coatings  
- Functionality in protein-containing media through optimized sensor geometry and protective coatings  

The system enables phenotypic AST by tracking the slope of permittivity changes during initial bacterial growth, with sensitive strains showing suppressed permittivity increases in the presence of effective antibiotics. This approach provides results substantially faster than conventional phenotypic methods while maintaining applicability to polymicrobial infections and emerging resistance mechanisms.  

## DETAILED DESCRIPTION OF THE INVENTION  

The invention comprises several key components that synergistically enable rapid AST:  

**Sensor Design:**  
Each LC sensor is fabricated on a flexible polyimide substrate, featuring a multi-turn planar inductive coil (typically 50 turns) connected to an interdigitated capacitor structure. The IDC component consists of multiple conductive digits (e.g., 11 digits) with optimized spacing (e.g., 0.5 mm gaps) to maximize sensitivity to dielectric changes in the surrounding medium. The entire sensor is coated with a thin polyurethane layer that prevents biofouling while permitting electrical field penetration for bacterial detection.  

**System Architecture:**  
The sensors are configured as insertable sleeves lining the walls of wells in standard 96-well plates. A detection coil positioned beneath each well establishes wireless magnetic coupling with the sensor coil through the magnetically transparent well bottom. An impedance analyzer sweeps frequencies (typically 1-12 MHz) to identify the resonant frequency of each sensor circuit, which shifts in response to bacterial growth-induced permittivity changes.  

**Measurement Principle:**  
Bacterial growth alters the effective permittivity of the culture medium due to:  
1) The inherent capacitive nature of bacterial cells  
2) Metabolic byproducts accumulating near the sensor surface  
3) Changes in medium composition due to bacterial activity  

These effects modify the IDC's capacitance (C1) and associated loss resistance (R1), causing measurable shifts in the system's resonant frequency. The rate of permittivity change (slope) during initial growth (typically 30 minutes) serves as the primary indicator of antibiotic susceptibility, with sensitive strains showing reduced slopes in the presence of effective antibiotics.  

**Signal Processing:**  
The system employs analytical models to derive permittivity from measured impedance spectra. Key equations include:  

The complex permittivity calculation:  
ε = C1/(kε0) - εsub + 1/(kR1ωzero-inductanceε0)  

Where ε0 is the permittivity of free space, εsub is the substrate permittivity, ωzero-inductance is the zero-inductance frequency, and k is the IDC cell constant derived from its geometric parameters.  

The total impedance equation accounting for mutual inductance (M) between coils:  
Ztotal = Zint + (ω2M2)/Zsensor  

Where Zsensor = jωL2 + R1/(1 + jωR1C1) represents the sensor's impedance.  

These relationships enable precise tracking of permittivity changes correlated with bacterial growth dynamics.  

### Example  

An exemplary implementation of the invention was tested with various bacterial strains and antibiotics:  

**Experimental Setup:**  
- Sensors were fabricated with 50-turn coils and 11-digit IDCs on polyimide substrates  
- Coated with oil-based polyurethane spray (MINWAX)  
- Sterilized via UV treatment before use  
- Tested with E. coli, S. aureus, and P. aeruginosa in low-salt LB medium  
- Antibiotics included ampicillin, ofloxacin, vancomycin, ciprofloxacin, and tobramycin  

**Results:**  
1. Growth Monitoring:  
The system detected bacterial growth within 30 minutes, significantly faster than OD600 measurements (typically 5 hours). All tested strains (E. coli MG1655, S. aureus ALC2085, P. aeruginosa PAO1) produced measurable permittivity increases.  

2. AST Performance:  
- Ampicillin-sensitive E. coli showed suppressed permittivity slopes (0.02 min-1) with 100 μg/mL ampicillin, while resistant strains maintained high slopes (0.12 min-1)  
- Dose-dependent responses correlated with reference MIC values:  
  - E. coli: ampicillin MIC = 12.8 μg/mL; ofloxacin MIC = 0.016 μg/mL  
  - S. aureus: vancomycin MIC = 0.2 μg/mL  
  - P. aeruginosa: ciprofloxacin MIC = 0.256 μg/mL  

3. Protein Tolerance:  
The system maintained functionality in media containing 2% fetal bovine serum, demonstrating potential for clinical sample testing after appropriate dilution.  

This example illustrates the invention's capability for rapid, phenotypic AST across diverse bacterial species and antibiotic classes, with performance exceeding conventional methods in speed while maintaining clinical relevance.  

--- 

The patent application provides comprehensive disclosure of the invention while maintaining formal patent language and structure. Each section builds upon the previous one to fully describe the novel aspects and advantages of the technology. The detailed description includes both theoretical foundations and practical implementations, supported by specific examples demonstrating the invention's utility.