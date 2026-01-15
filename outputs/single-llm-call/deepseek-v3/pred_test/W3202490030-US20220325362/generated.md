Here is the complete patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND  

Infectious diseases caused by pathogenic microorganisms such as bacteria and viruses represent a significant threat to global public health. Throughout history, numerous outbreaks including the Spanish flu pandemic of 1918, the HIV/AIDS crisis, and more recently the Ebola and Zika virus epidemics have demonstrated the devastating impact of infectious diseases on human populations. The ongoing COVID-19 pandemic caused by the SARS-CoV-2 coronavirus has further highlighted the critical need for rapid and accurate diagnostic testing to control disease transmission.  

Traditional methods for detecting respiratory viruses include antigen-based tests, serological assays, and nucleic acid amplification techniques. While antigen tests provide rapid results, they often lack sufficient sensitivity for early infection detection. Serological tests measure antibody responses but cannot distinguish between current and past infections. Polymerase Chain Reaction (PCR)-based nucleic acid tests offer high sensitivity and specificity but require specialized equipment and trained personnel, limiting their use in point-of-care settings.  

Recent advances in CRISPR-Cas systems have revolutionized nucleic acid detection technologies. CRISPR-based diagnostic platforms utilizing Cas12 and Cas13 effector proteins have demonstrated remarkable sensitivity and specificity for viral DNA and RNA detection, respectively. These systems leverage the collateral nuclease activities of Cas12 and Cas13 that are activated upon target recognition, enabling amplification-free detection through cleavage of reporter oligonucleotides. However, current CRISPR diagnostics face limitations including target sequence constraints, single-channel signal output, and requirements for pre-amplification at low target concentrations.  

The Type III-A CRISPR-Cas system represents a multifunctional antiviral defense mechanism found in bacteria and archaea. Unlike single-protein Cas effectors, Type III-A systems comprise multi-subunit ribonucleoprotein complexes with four distinct enzymatic activities: target RNA cleavage, collateral DNase activity, cyclic oligoadenylate (cOA) synthesis, and cOA-activated RNase activity. This unique combination of functions provides multiple avenues for nucleic acid detection through parallel signal generation pathways. While the complexity of Type III-A systems has previously hindered their adaptation for diagnostic applications, recent advances in protein expression and purification have enabled their reconstitution for biotechnological purposes.  

## BRIEF SUMMARY  

The present invention provides a novel virus detection system termed MORIARTY (Multipronged, One-pot, target RNA-Induced, Augmentable, Rapid, Test sYstem) based on Type III-A CRISPR-Cas technology. The MORIARTY system harnesses the multifunctional capabilities of the Lactococcus lactis Csm (LlCsm) effector complex and its ancillary protein LlCsm6 to enable sensitive and specific detection of viral nucleic acids through multiple parallel signaling pathways.  

Key advantages of the MORIARTY system include its ability to simultaneously generate DNase and RNase collateral activity signals, its compatibility with both RNA and DNA virus detection, and its capacity for signal amplification through cyclic oligoadenylate synthesis. The system can be configured for either amplification-free direct detection or coupled with isothermal amplification methods such as reverse transcription recombinase polymerase amplification (RT-RPA) for enhanced sensitivity.  

Embodiments of the invention include methods for detecting SARS-CoV-2 viral RNA using reprogrammed LlCsm complexes targeting the Spike gene, optimized reaction conditions for maximal signal output, and multiplexed detection strategies employing multiple guide RNAs. The MORIARTY platform demonstrates attomolar sensitivity for SARS-CoV-2 detection in patient samples and shows strong concordance with quantitative PCR results, while offering advantages in speed, cost, and equipment requirements.  

## DETAILED DESCRIPTION  

The MORIARTY system represents a versatile nucleic acid detection platform based on the Type III-A CRISPR-Cas system from Lactococcus lactis. At the core of the technology is the LlCsm effector complex, a ribonucleoprotein comprising five protein subunits (Csm1-5) and a CRISPR RNA (crRNA) that guides target recognition. Upon binding to complementary viral RNA, the LlCsm complex exhibits four distinct enzymatic activities: specific cleavage of the target RNA by the Csm3 subunit, nonspecific single-stranded DNA degradation by the Csm1 HD domain, synthesis of cyclic oligoadenylate (cOA) second messengers by the Csm1 GGDD motif, and cOA-activated RNA cleavage by the ancillary protein LlCsm6.  

A key innovation of the MORIARTY system is its ability to simultaneously harness multiple enzymatic outputs for detection purposes. The DNase activity of Csm1 provides one signal channel through cleavage of DNA-based fluorescent reporters, while the cOA-activated RNase activity of Csm6 generates a parallel signal through RNA reporter cleavage. This multipronged approach enables cumulative signal generation and flexibility in assay design, as different metal ion conditions can preferentially activate specific enzymatic outputs.  

The system demonstrates exceptional sensitivity for viral RNA detection, achieving femtomolar limits of detection in amplification-free configurations and attomolar sensitivity when coupled with RT-RPA pre-amplification. This performance stems from the inherent signal amplification capacity of the cOA synthesis pathway, where a single target RNA molecule can stimulate production of multiple cOA molecules that each activate numerous Csm6 RNase enzymes.  

The Type III-A CRISPR-Cas system offers several advantages over existing CRISPR diagnostics. Unlike Cas12 and Cas13 systems that require specific protospacer adjacent motifs or activator cleavage for collateral activity, the LlCsm complex recognizes target RNA through base pairing alone and maintains separate active sites for target cleavage and collateral activities. This architecture provides greater flexibility in target site selection and more robust signal generation. Additionally, the system's responsiveness to both viral RNA and its transcription products enables detection of both RNA and DNA viruses.  

Implementation of the MORIARTY system involves several key components:  

1. The LlCsm effector complex reprogrammed with guide sequences complementary to viral targets  
2. The LlCsm6 ancillary protein for cOA-activated RNase activity  
3. Fluorescent DNA and RNA reporter oligonucleotides  
4. Optimized reaction buffers containing appropriate metal ion cofactors  
5. Optional pre-amplification components for enhanced sensitivity  

The system has been successfully applied to SARS-CoV-2 detection through targeting of the viral Spike gene. Multiplexing approaches employing multiple guide RNAs further enhance detection sensitivity and robustness. MORIARTY assays can be completed within 30-50 minutes and show excellent agreement with gold-standard PCR methods in clinical sample testing.  

### EXAMPLES  

**Example 1: MORIARTY System Configuration and Validation**  

The LlCsm effector complex was produced using an all-in-one expression plasmid encoding all Csm subunits and the crRNA. The complex was purified via nickel affinity chromatography followed by size exclusion chromatography, yielding active ribonucleoprotein at concentrations suitable for diagnostic applications. The ancillary protein LlCsm6 was expressed and purified separately.  

To validate system functionality, reactions containing LlCsm, LlCsm6, ATP, and fluorescent DNA and RNA reporters were stimulated with model target RNA. Under magnesium ion conditions, strong RNA-FAM fluorescence signal was observed corresponding to cOA-activated RNase activity, while manganese ions enabled simultaneous detection of both DNA-Alexa (DNase) and RNA-FAM (RNase) signals. Control experiments confirmed signal specificity through use of non-cognate target RNA and catalytic mutants of Csm1 and Csm6.  

The system was further adapted for transcription-coupled detection (T7-MORIARTY) by including T7 RNA polymerase and promoter-containing DNA templates. This configuration enabled detection of DNA targets through their transcription products, demonstrating the system's versatility for both direct RNA detection and amplified DNA detection applications.  

**Example 2: Amplification-Free Detection of SARS-CoV-2 Viral RNA**  

The MORIARTY system was optimized for direct detection of SARS-CoV-2 Spike gene mRNA without pre-amplification. The LlCsm complex was reprogrammed with guide sequences targeting three distinct regions of the Spike gene (nucleotides 22280-22308, 24702-24730, and 25061-25089) to enable multiplexed detection.  

Under optimized magnesium ion and ATP conditions, the system demonstrated femtomolar sensitivity for in vitro transcribed Spike mRNA. Multiplexing with three guide RNAs improved detection limits to approximately 2000 copies/μL when testing quantified SARS-CoV-2 control RNA. Application to human patient nasopharyngeal swab samples showed concordance with qRT-PCR results for high viral load specimens, with statistically significant signal differentiation (p < 0.0070) between positive and negative samples.  

**Example 3: Attomolar Detection of SARS-CoV-2 with Amplification-Coupled MORIARTY**  

To achieve higher sensitivity, the MORIARTY system was coupled with RT-RPA pre-amplification. This approach combined reverse transcription of viral RNA with isothermal DNA amplification using recombinase-polymerase enzymes, followed by T7-MORIARTY detection of amplification products.  

The optimized RT-RPA-T7-MORIARTY protocol demonstrated attomolar sensitivity for SARS-CoV-2 detection, reliably identifying samples with as few as 62 viral copies/μL. Testing of clinical specimens showed strong agreement with qRT-PCR results, correctly identifying 7 of 8 PCR-positive and 3 of 4 PCR-negative patient samples. The entire assay, including amplification and detection steps, could be completed within 50 minutes using standard laboratory equipment.  

### MATERIAL AND METHODS  

**Protein Production:**  
The LlCsm effector complexes were expressed in E. coli using pACYC-derived plasmids encoding all Csm subunits and CRISPR arrays. Mutations were introduced via Q5 site-directed mutagenesis. Proteins were purified by nickel affinity and size exclusion chromatography, with final concentrations determined by spectrophotometry.  

**Guide RNA Reprogramming:**  
CRISPR RNA spacers were redesigned to target SARS-CoV-2 Spike gene sequences using BbsI cloning sites. Three target sites were selected to enable multiplex detection, with careful consideration of protospacer flanking sequence requirements for optimal Csm activation.  

**Nucleic Acid Preparation:**  
SARS-CoV-2 Spike gene mRNA was produced by in vitro transcription from linearized plasmid templates using T7 RNA polymerase. RNA was purified, quantified, and stored at -80°C. Patient samples were processed using viral RNA extraction kits according to manufacturer protocols.  

**Amplification-Free MORIARTY Assay:**  
Reactions contained LlCsm effector complexes, LlCsm6, fluorescent DNA and RNA reporters, metal ion cofactors, and ATP in optimized buffer conditions. Target RNA was added to initiate reactions, with fluorescence monitored in real-time using plate reader instrumentation.  

**T7-MORIARTY Assay:**  
Transcription-coupled detection was performed by including T7 RNA polymerase and promoter-containing DNA templates in reaction mixtures. Fluorescence signals were measured at regular intervals to track reaction progress.  

**RT-RPA Amplification:**  
Viral RNA was reverse transcribed and amplified using recombinase-polymerase amplification kits according to manufacturer protocols, with modifications to include T7 promoter sequences in primer designs.  

**qRT-PCR Validation:**  
Patient samples were tested in parallel using FDA-approved qRT-PCR protocols targeting SARS-CoV-2 nucleocapsid and envelope genes, with cycle threshold values used as reference standards for MORIARTY performance evaluation.  

The complete system represents a significant advance in CRISPR-based diagnostics, combining the sensitivity of nucleic acid testing with the simplicity and versatility needed for widespread deployment in clinical and point-of-care settings.