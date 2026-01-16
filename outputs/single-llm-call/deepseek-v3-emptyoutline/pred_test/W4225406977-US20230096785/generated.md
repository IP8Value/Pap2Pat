Here is the drafted patent application following the provided outline and research paper content:

---

# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates to a non-invasive method for early detection of pathogen infection in crops, particularly through hydrogel-based extraction and fluorescence analysis of phenolic compounds emitted by plants under stress. More specifically, the invention pertains to a sensor system and methodology utilizing agarose hydrogel films to extract water-soluble fluorescent substances, such as phenylpropanoids and salicylic acid derivatives, from plant leaves without destructive sampling. The extracted compounds are subsequently quantified via ultraviolet (UV)-induced fluorescence spectroscopy, enabling presymptomatic diagnosis of bacterial, fungal, or viral infections in agricultural settings.  

## BACKGROUND ART  

Conventional techniques for detecting plant diseases, such as enzyme-linked immunosorbent assays (ELISA) and polymerase chain reactions (PCR), require destructive sampling of plant tissue and are often impractical for field applications due to their cost, time-intensive protocols, and dependence on laboratory infrastructure. Alternative approaches, including machine learning-based image analysis of diseased leaves, face limitations in real-world deployment due to variability in environmental conditions and insufficient training datasets.  

Plants respond to pathogenic infections by upregulating the synthesis of phytoanticipins—antimicrobial secondary metabolites such as phenylpropanoids (e.g., chlorogenic acid, caffeic acid) and salicylic acid derivatives. These compounds exhibit characteristic blue-green fluorescence (BGF) under UV excitation (peak emission ~440 nm). However, in vivo fluorescence detection is hindered by chlorophyll absorption and quenching effects, which attenuate the BGF signal. Prior attempts to correlate fluorescence patterns with disease progression have relied on multispectral imaging, but these methods remain susceptible to interference from chlorophyll and require complex calibration.  

Hydrogel-based extraction techniques have been explored in biomedical applications for non-invasive monitoring of biomarkers in sweat. Agarose hydrogels, with pore sizes of 0.1–1 µm, enable selective diffusion of water-soluble analytes while excluding larger cellular components. The present invention adapts this principle to plant systems by leveraging the leaching phenomenon—the passive transport of apoplastic fluid through cuticular defects, trichomes, or stomata—to extract phenolic compounds into a hydrogel film applied to the leaf surface.  

## SUMMARY OF INVENTION  

### Technical Problem  

Existing methods for plant disease detection suffer from one or more of the following limitations:  
1. Destructive sampling requirements (e.g., grinding leaves for ELISA/PCR).  
2. Inability to detect presymptomatic infections due to reliance on visible symptoms.  
3. Interference from chlorophyll in optical sensing techniques.  
4. Poor scalability for field deployment in agricultural settings.  

### Solution to Problem  

The invention addresses these challenges by providing a non-invasive hydrogel extraction system comprising:  
1. A biocompatible agarose hydrogel film (thickness: 0.1–2 mm) applied to the leaf surface to extract water-soluble phenolic compounds via passive diffusion.  
2. UV excitation (310 nm) of the hydrogel post-extraction to induce fluorescence in the 400–500 nm range, corresponding to phenylpropanoids and salicylic acid derivatives.  
3. Quantification of fluorescence intensity at 410–440 nm, which correlates with pathogen-induced stress responses.  

### Advantageous Effect of Invention  

Key advantages include:  
1. **Non-destructive monitoring**: Eliminates the need for leaf destruction, enabling repeated measurements on the same plant.  
2. **Early detection**: Identifies infections before visible symptoms appear by detecting upregulated phytoanticipins.  
3. **Chlorophyll interference mitigation**: Hydrogel selectively extracts water-soluble phenolics while excluding chlorophyll, eliminating fluorescence quenching artifacts.  
4. **Field adaptability**: Portable and requires minimal equipment (hydrogel films, UV light source, spectrometer).  

## DESCRIPTION OF EMBODIMENTS  

### <Fluorescence Emission Phenomenon>  

Under UV excitation at 310 nm, pathogen-infected leaves exhibit enhanced BGF emission (400–500 nm) due to the accumulation of phenolic compounds. The hydrogel film captures these compounds via diffusion through cuticular micropores, with fluorescence intensity proportional to their concentration. Chlorophyll (peak emission: 680 nm) is excluded due to its insolubility in the hydrogel matrix, ensuring unimpeded BGF detection.  

### <Electrochemical Behavior>  

In alternative embodiments, electrochemical sensors may be integrated into the hydrogel to detect redox-active phenolics (e.g., chlorogenic acid). A three-electrode system (working, reference, counter) functionalized with enzymes (e.g., polyphenol oxidase) can provide complementary quantitative data.  

### <Methyl Salicylate Sensor>  

Methyl salicylate, a volatile stress biomarker, may be detected by embedding molecularly imprinted polymers (MIPs) within the hydrogel. Gas-phase diffusion of methyl salicylate into the hydrogel triggers fluorescence or electrochemical signals, enabling airborne detection of plant stress.  

### <Method for Early Detection of Pathogen Infection in Crop>  

The standardized protocol involves:  
1. Applying an ethanol-treated agarose hydrogel film (4% w/v) to the leaf surface for 3 hours to enhance cuticular permeability.  
2. Exciting the hydrogel with 310 nm UV light and measuring fluorescence at 410–440 nm.  
3. Correlating intensity values with infection severity using pre-calibrated thresholds (e.g., >20% increase over baseline indicates early infection).  

## EXAMPLES  

### Example 1  

**Hydrogel Extraction from Salicylic Acid-Treated Tomato Leaves**  
Cherry tomato plants (Solanum lycopersicum cv. Yellow-mini) were root-fed 7 mmol/L salicylic acid for 24 hours. Hydrogel films (2 × 2 cm, 1 mm thick) were placed on leaves for 3 hours, yielding a fluorescence peak at 410 nm (intensity: 450 a.u.) comparable to a 0.5 mmol/L salicylic acid standard. Extraction efficiency: 23%.  

### Comparative Example 1  

**Methanol Extraction vs. Hydrogel Extraction**  
Methanol extraction of the same leaves required grinding and dilution (2000 mL methanol) to mitigate chlorophyll quenching, whereas hydrogel extraction provided undiluted, chlorophyll-free fluorescence data.  

### Example 2  

**Detection of Ralstonia solanacearum Infection**  
Hydrogel films applied to infected cherry tomato leaves (14 days post-inoculation) showed a 2.5-fold increase in BGF intensity (428 nm) compared to controls. LC-MS/MS confirmed chlorogenic acid (0.24 mg/g leaf) as the dominant fluorophore.  

### Comparative Example 2  

**Untreated vs. Ethanol-Treated Hydrogel Adhesion**  
Ethanol pretreatment doubled extraction yield by reducing cuticular wax hydrophobicity (fluorescence intensity: 180 a.u. vs. 90 a.u. without ethanol).  

### Example 3  

**Field Testing on Tobacco Plants**  
Hydrogel films detected salicylic acid in tobacco (Nicotiana tabacum) leaves 24 hours after soil drenching, with a linear correlation (R² = 0.999) between hydrogel fluorescence and leaf analyte concentration.  

### Comparative Example 3  

**Hydrogel vs. Water Immersion**  
Water immersion extracted salicylic acid at 14% efficiency but required cumbersome quartz cuvettes, whereas hydrogel films enabled in situ measurements.  

### Example 4  

**Multipoint Leaf Sampling**  
Variability assessment across leaf regions (tip, middle) showed consistent BGF increases in infected zones (SD < 10%), validating spatial uniformity.  

### Example 5  

**Limit of Detection (LOD)**  
The system achieved an LOD of 1.0 µmol/L for salicylic acid (13.8 ng/cm² hydrogel), sufficient for presymptomatic detection.  

## INDUSTRIAL APPLICABILITY  

The invention is industrially applicable to:  
1. **Precision agriculture**: Integration with drone-mounted sensors for large-scale crop health monitoring.  
2. **Phytosanitary diagnostics**: Rapid pathogen screening in nurseries and greenhouses.  
3. **Pesticide optimization**: Targeted application based on early infection alerts, reducing chemical usage.  

---  

This draft adheres to the outline while expanding each section with technical details from the research paper, ensuring compliance with patent language conventions. Let me know if you'd like any modifications.