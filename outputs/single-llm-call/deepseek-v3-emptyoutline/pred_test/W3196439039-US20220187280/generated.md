Below is the drafted patent application following the provided outline. The language is formal and technical, adhering to patent drafting conventions.  

---  

# **DESCRIPTION**  

## **FIELD**  
The present invention relates to the field of biotechnology, specifically to genetically encoded biosensors for detecting neuronal death. More particularly, the invention pertains to genetically encoded death indicators (GEDIs) that detect irreversible loss of calcium (Ca²⁺) homeostasis in neurons, providing an early and acute marker of cell death. The invention includes novel variants of GEDIs, methods for their use in neurodegenerative disease models, and applications in high-throughput screening for neuroprotective therapeutics.  

## **BACKGROUND**  
Neurodegenerative diseases such as Parkinson’s disease (PD), Huntington’s disease (HD), Alzheimer’s disease (AD), amyotrophic lateral sclerosis (ALS), and frontotemporal dementia (FTD) are characterized by progressive neuronal dysfunction and death. Current methods for detecting neuronal death rely on dyes, stains, or markers of specific cell death pathways (e.g., apoptosis or necrosis), which suffer from limitations such as delayed signal onset, toxicity, and pathway specificity.  

Existing genetically encoded calcium indicators (GECIs) detect physiological Ca²⁺ transients but are not optimized to distinguish irreversible Ca²⁺ influx associated with cell death. Additionally, conventional methods for tracking neuronal death in vivo, such as fluorescent protein loss or morphological changes, lack precision and temporal resolution. There remains a need for a sensitive, genetically encoded biosensor capable of detecting neuronal death across diverse neurodegenerative conditions without interference from physiological Ca²⁺ fluctuations.  

## **SUMMARY**  
The invention provides genetically encoded death indicators (GEDIs) that detect neuronal death by sensing irreversible Ca²⁺ dysregulation. GEDIs are derived from modified GECIs, including but not limited to CEPIA and GCaMP variants, engineered to exhibit minimal response to physiological Ca²⁺ transients while fluorescing upon pathological Ca²⁺ influx indicative of cell death.  

Key aspects of the invention include:  
1. **GEDI variants** (e.g., RGEDI-P2a-EGFP, GC150-P2a-mApple) with optimized Ca²⁺ affinity for detecting death in vitro and in vivo.  
2. **Pseudo-ratiometric constructs** incorporating self-cleaving peptides (e.g., P2a) for normalization and improved quantification.  
3. **Nuclear-localized GEDIs** (e.g., RGEDI-NLS-P2a-EGFP-NLS) for enhanced signal resolution in whole-organism imaging.  
4. **Methods for longitudinal tracking** of neuronal death in cultured neurons, organotypic slices, and live animals (e.g., zebrafish).  
5. **High-throughput applications** for screening neuroprotective compounds using automated microscopy and survival analysis.  

The GEDI biosensors provide a robust, unbiased, and quantitative tool for studying neurodegeneration, enabling precise determination of neuronal death kinetics and facilitating therapeutic development.  

## **DETAILED DESCRIPTION**  

### **Definitions**  
- **GEDI (Genetically Encoded Death Indicator):** A biosensor derived from GECIs, modified to detect pathological Ca²⁺ influx associated with irreversible neuronal death.  
- **Pseudo-ratiometric:** A dual-fluorescence construct where the GEDI signal is normalized to a co-expressed fluorescent protein (e.g., EGFP, mApple) via a self-cleaving peptide (e.g., P2a).  
- **GEDI threshold:** A empirically determined fluorescence ratio (e.g., ΔF/F) that distinguishes live from dead neurons.  
- **Cumulative Risk of Death (CRD):** A statistical measure derived from longitudinal GEDI data to quantify neuronal survival in disease models.  

### **Vectors**  
The invention encompasses expression vectors for delivering GEDI constructs to neurons, including:  
- **Plasmid vectors** (e.g., phSyn1-driven constructs for neuronal expression).  
- **Viral vectors** (e.g., AAV, lentivirus) for stable transduction in vivo.  
- **Transgenic constructs** (e.g., Tol2kit-based vectors for zebrafish).  

Exemplary constructs include:  
- **phSyn1:RGEDI-P2a-EGFP**  
- **phSyn1:GC150-P2a-mApple**  
- **neuroD:GC150-P2a-mApple** (for zebrafish MN labeling)  

### **Cells**  
GEDIs are expressed in:  
- **Primary neurons** (rat, mouse cortical/hippocampal cultures).  
- **Induced pluripotent stem cell (iPSC)-derived neurons** (e.g., motor neurons from ALS patients).  
- **Cell lines** (e.g., HEK293 for validation).  

### **Animals**  
The invention includes applications in:  
- **Zebrafish larvae** (e.g., neuroD:GC150-P2a-mApple for in vivo death detection).  
- **Rodent models** (e.g., transgenic mice expressing GEDIs in vulnerable neuronal populations).  

### **Methods of Use**  
1. **Longitudinal Imaging:** Automated 4D microscopy tracks GEDI signal in single neurons over days to weeks.  
2. **High-Throughput Screening:** 96-well plate assays quantify neuroprotection using GEDI thresholds.  
3. **In Vivo Death Detection:** GC150 variants enable death monitoring in unanesthetized zebrafish.  
4. **Pathway-Agnostic Death Assay:** GEDIs detect apoptosis, necrosis, and excitotoxicity without bias.  

### **Kits**  
The invention provides kits comprising:  
- **GEDI expression plasmids/viruses.**  
- **Protocols for transfection and imaging.**  
- **Software for automated survival analysis (e.g., R scripts for CRD calculation).**  

## **EXAMPLES**  

### **Example 1: Development and Validation of RGEDI-P2a-EGFP**  
Primary rat cortical neurons were transfected with phSyn1:RGEDI-P2a-EGFP. Electrical stimulation (30 Hz) induced GCaMP6f fluorescence but not RGEDI signal, confirming specificity for pathological Ca²⁺. NaN3 treatment triggered a sustained RGEDI signal increase, establishing a GEDI threshold (ΔF/F > 0.05) for death classification.  

### **Example 2: Glutamate Toxicity and Subpopulation Analysis**  
Neurons expressing RGEDI-P2a-EGFP were exposed to glutamate (0.1–1 mM). Kaplan-Meier analysis revealed resistant subpopulations surviving >108 hours, demonstrating GEDI’s utility in heterogeneity studies.  

### **Example 3: Neurodegenerative Disease Models**  
Co-expression of RGEDI-P2a-EGFP with HttEx1Q97, α-synuclein, or TDP43 in rat neurons showed increased CRD vs. controls, validating GEDI for PD, HD, and ALS/FTD models.  

### **Example 4: iPSC-Derived Motor Neuron Death**  
SOD1 D90A ALS motor neurons exhibited higher death rates than controls (CRD = 1.26), confirming GEDI’s applicability to human neuronal models.  

### **Example 6: In Vivo Zebrafish Imaging**  
neuroD:GC150-P2a-mApple zebrafish larvae showed acute GC150 signal increases after metronidazole-induced MN ablation, enabling death detection in intact organisms.  

---  

This application provides a comprehensive disclosure of the GEDI technology, including its design, utility, and experimental validation. The claims (not included here) would cover the biosensor variants, methods of use, and applications in research and drug discovery.  

Would you like any modifications or additional details in specific sections?