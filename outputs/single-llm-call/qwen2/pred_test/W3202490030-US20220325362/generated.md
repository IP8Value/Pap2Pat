# DESCRIPTION

## BACKGROUND

The field of infectious disease diagnostics has seen significant advancements in recent years, driven by the need for rapid, accurate, and cost-effective methods to detect pathogens. Traditional methods such as antigen-based tests, serology, and polymerase chain reaction (PCR) have been widely used but come with limitations. Antigen-based tests, while rapid, often lack the necessary sensitivity and require pre-manufactured antibodies. Serological tests, though mature, cannot distinguish between past and active infections. PCR, while highly sensitive and quantitative, requires specialized equipment and trained personnel, making it less suitable for point-of-care settings.

The advent of CRISPR-Cas technology has revolutionized nucleic acid detection, offering a promising alternative to traditional methods. CRISPR-Cas systems, particularly those involving Cas12 and Cas13 enzymes, have been utilized for their high sensitivity and specificity in detecting viral RNA and DNA. These systems leverage the collateral cleavage activity of Cas enzymes, which can be used to cleave reporter molecules, generating a detectable signal.

However, the Type III-A CRISPR-Cas system, or Csm, presents a unique opportunity for viral detection due to its multifaceted enzymatic activities. The Csm complex can cleave viral RNA, synthesize cyclic oligoadenylates (cOA), and activate a collateral RNase activity in the ancillary protein Csm6. These activities can be harnessed to create a versatile and sensitive detection system. Despite these advantages, the realization of a practical Csm-based detection tool has been challenging due to the complexity of enzyme production and the need for multiple components.

This invention, termed MORIARTY (Multipronged, One-pot, target RNA-Induced, Augmentable, Rapid, Test sYstem), addresses these challenges by providing a robust and efficient method for detecting viral RNA, particularly SARS-CoV-2, using the Type III-A CRISPR-Cas system. MORIARTY leverages the dual collateral activities of the Csm complex and Csm6, enabling rapid and sensitive detection without the need for complex equipment. The system is designed to be deployable in point-of-care settings, making it a valuable tool in the fight against infectious diseases.

## BRIEF SUMMARY

The present invention relates to a novel method and system for detecting viral RNA using the Type III-A CRISPR-Cas system. Specifically, the invention provides a multipronged, one-pot, target RNA-induced, augmentable, rapid test system (MORIARTY) that utilizes the Csm complex and Csm6 to detect viral RNA with high sensitivity and specificity. The system can be used for both amplification-free and amplification-coupled detection, making it suitable for a wide range of applications, including point-of-care diagnostics.

The key features of the invention include:
1. **Multipronged Detection**: MORIARTY employs multiple enzymatic activities of the Csm complex, including specific cleavage of viral RNA, collateral DNase activity, cOA synthesis, and cOA-activated collateral RNase activity in Csm6. This multipronged approach enhances the sensitivity and reliability of the detection system.
2. **One-Pot Reaction**: The system is designed to perform all detection steps in a single reaction vessel, simplifying the process and reducing the risk of contamination.
3. **Target RNA-Induced Activity**: The detection is triggered by the presence of target viral RNA, ensuring high specificity.
4. **Augmentable Signal**: The system can be optimized to amplify the detection signal through the use of cyclic oligoadenylates and collateral RNase activity, allowing for the detection of low-concentration viral RNA.
5. **Rapid Detection**: MORIARTY can provide results within a short time frame, making it suitable for rapid diagnostic testing.
6. **Versatility**: The system can be adapted to detect various viral RNA targets, including SARS-CoV-2, and can be used in both amplification-free and amplification-coupled settings.

The invention also includes methods for optimizing the detection system, including the use of different metal ions, ATP concentrations, and multiplex targeting strategies to enhance sensitivity and specificity. Additionally, the invention provides a detailed protocol for the preparation and use of the Csm complex and Csm6, as well as the design and synthesis of target-specific guide RNAs.

## DETAILED DESCRIPTION

### Overview of the Invention

The present invention, MORIARTY, is a novel method and system for detecting viral RNA using the Type III-A CRISPR-Cas system. The system is designed to be multipronged, one-pot, target RNA-induced, augmentable, and rapid, making it highly suitable for point-of-care diagnostics. The key components of the system include the Csm complex, Csm6, and target-specific guide RNAs. The Csm complex is capable of cleaving viral RNA, synthesizing cyclic oligoadenylates (cOA), and activating a collateral RNase activity in Csm6. These activities are harnessed to detect viral RNA with high sensitivity and specificity.

### Components of the System

#### Csm Complex
The Csm complex is a ribonucleoprotein enzyme system that comprises four enzymatic activities:
1. **Specific Cleavage of Viral RNA**: The Csm3 subunit cleaves the viral RNA.
2. **Collateral DNase Activity**: The HD domain of the Csm1 subunit exhibits collateral DNase activity.
3. **cOA Synthesis**: The Csm1 GGDD motif synthesizes cyclic oligoadenylates (cOA).
4. **cOA-Activated Collateral RNase Activity**: The ancillary enzyme Csm6 is activated by cOA to cleave RNA probes.

#### Csm6
Csm6 is an ancillary enzyme that is activated by cOA to exhibit collateral RNase activity. This activity is crucial for amplifying the detection signal.

#### Guide RNAs
Guide RNAs are designed to be complementary to the target viral RNA. The guide RNAs are incorporated into the Csm complex, enabling specific recognition and cleavage of the target RNA.

### Mechanism of Action

The detection process involves the following steps:
1. **Binding and Cleavage of Target RNA**: The Csm complex binds to the target viral RNA and cleaves it. This cleavage triggers the synthesis of cOA by the Csm1 subunit.
2. **cOA Synthesis and Activation of Csm6**: The synthesized cOA activates the collateral RNase activity of Csm6.
3. **Collateral RNase Activity**: The activated Csm6 cleaves RNA probes, generating a detectable signal.
4. **Collateral DNase Activity**: The HD domain of the Csm1 subunit exhibits collateral DNase activity, which can also be used to generate a detectable signal.

### Amplification-Free Detection

MORIARTY can be used for amplification-free detection of viral RNA. The system is optimized to detect low concentrations of viral RNA without the need for pre-amplification. Key factors that influence the sensitivity of amplification-free detection include:
- **Metal Ion Concentrations**: The presence of Mg2+ and Mn2+ ions can significantly impact the sensitivity of the system. Mg2+ ions facilitate cOA synthesis, while Mn2+ ions are required for DNase activity.
- **ATP Concentrations**: The concentration of ATP is critical for the synthesis of cOA and the activation of Csm6.
- **Multiplex Targeting**: Using multiple Csm complexes targeting different regions of the viral RNA can enhance the sensitivity of the system.

### Amplification-Coupled Detection

MORIARTY can also be used in conjunction with amplification techniques, such as reverse transcription and recombinase-polymerase amplification (RT-RPA), to detect viral RNA at even lower concentrations. The RT-RPA step amplifies the viral RNA, which is then used to stimulate the Csm complex and Csm6, leading to the generation of a detectable signal. Key factors that influence the sensitivity of amplification-coupled detection include:
- **Optimization of RT-RPA Conditions**: The conditions for the RT-RPA step, including primer design and reaction conditions, are optimized to maximize the amplification of the viral RNA.
- **Optimization of T7-MORIARTY Conditions**: The conditions for the T7-MORIARTY step, including the concentrations of Csm6, ATP, and metal ions, are optimized to maximize the detection signal.

### Examples

#### Example 1: Amplification-Free Detection of SARS-CoV-2

**Materials and Methods**
- **Csm Complex and Csm6**: The Csm complex and Csm6 were expressed and purified as described in the methods section.
- **Guide RNAs**: Guide RNAs were designed to target the S gene of SARS-CoV-2.
- **Fluorescent Probes**: RNA-FAM and DNA-Alexa probes were used to monitor the collateral RNase and DNase activities, respectively.
- **Reaction Conditions**: The reaction was performed in 33 mM Tris acetate pH 7.6 at 32°C, 66 mM potassium acetate, 0.5 μM DNA-Alexa, 0.5 μM RNA-FAM, 250 nM LlCsm effector complex, 1.0 nM LlCsm6, and a combination of divalent ions and ATP.

**Results**
- **Detection Sensitivity**: The system was able to detect in vitro transcribed SARS-CoV-2 S mRNA at concentrations as low as 5 fM.
- **Multiplex Detection**: Using multiple Csm complexes targeting different regions of the S gene, the detection sensitivity was improved to 2000 copies/μL.

#### Example 2: Amplification-Coupled Detection of SARS-CoV-2

**Materials and Methods**
- **RT-RPA**: The RT-RPA step was performed using TwistAmp Basic kit following the manufacturer's instructions.
- **T7-MORIARTY**: The T7-MORIARTY step was performed in 30 mM K-HEPES pH 7.6, 2 mM spermidine, 0.01% Triton X-100, 17 mM MgCl2, 0.5 μM rNTPs, 10 mM TCEP, and 60 μg/mL T7 RNA polymerase.
- **Fluorescent Probes**: RNA-FAM and DNA-FAM probes were used to monitor the collateral RNase and DNase activities, respectively.

**Results**
- **Detection Sensitivity**: The system was able to detect SARS-CoV-2 control RNA at concentrations as low as 32 cp/μL.
- **Patient Sample Testing**: The system was applied to human patient samples, and the results showed a high degree of agreement with q-RT-PCR results.

### Material and Methods

#### Cloning
The pACYC Lactococcus lactis Csm (LlCsm) effector module plasmid encoding Cas6, Csm1-6, and CRISPR locus was constructed as described previously. Mutations were introduced using Q5 mutagenesis, and guide RNA was incorporated using BbsI sites. Plasmids were verified by sequencing.

#### Protein Expression and Purification
The LlCsm effector complexes were expressed and purified from Escherichia coli NiCo21(DE3) strain. The all-in-one pACYC plasmid was transformed into the cells, and the cells were grown to log phase before induction with IPTG. The N-terminal His6-tag on LlCsm2 enabled isolation of LlCsm RNP using Ni-NTA affinity chromatography. The Ni-NTA elution pools were loaded onto a size-exclusion column equilibrated with storage buffer. LlCsm6 was produced separately and stored in a buffer containing HEPES, NaCl, and 2-mercaptoethanol. T7 RNA polymerase was produced as described previously.

#### In Vitro Transcription of S Gene mRNA
The plasmid encoding the SARS-CoV-2 surface glycoprotein (Spike protein) was amplified and linearized using BamH1. In vitro transcription was performed using T7 RNA polymerase, and the RNA transcript was purified and stored at −80°C.

#### Amplification-Free MORIARTY
For two-channel fluorescence experiments, DNA-Alexa and RNA-FAM probes were designed and synthesized. The reactions were performed in TAPA buffer containing MgCl2, MnCl2, ATP, and target RNA. The fluorescence was measured using a multi-mode microplate reader.

#### T7-MORIARTY
The T7-MORIARTY methodology was designed to track LlCsm co-transcriptional activation of template viral DNA obtained from synthetic DNA or RT-RPA step. The reactions were performed in a buffer cocktail containing TAPA buffer, transcription buffer, DNA-FAM, RNA-FAM, LlCsm effector complex, LlCsm6, MgCl2, MnCl2, rNTPs, TCEP, and T7 RNA polymerase. The fluorescence was measured using a multi-mode microplate reader.

#### RT-RPA
The RT-RPA step was performed using the TwistAmp Basic kit. The reactions were incubated at 42°C for 30 minutes, and the products were transferred to ice until T7-MORIARTY detection.

#### Q-RT-PCR
To quantify copy numbers of in vitro transcribed S gene mRNA, a 2-step laboratory-based assay was performed using SuperScript III First-Strand cDNA kit and PerfeCTa SYBR Green FastMix. The heat-inactivated SARS-CoV-2 control RNA was used as the copy standard. For patient sample testing, the assay was performed using the Luna Universal Probe One-Step RT-qPCR Kit and the CDC primer-probe sets for human RNase P and SARS-CoV-2 Nucleocapsid targets.

### Conclusion

MORIARTY represents a significant advancement in the field of viral RNA detection. By leveraging the multifaceted activities of the Type III-A CRISPR-Cas system, MORIARTY provides a robust, sensitive, and versatile method for detecting viral RNA, particularly SARS-CoV-2. The system is designed to be user-friendly and deployable in point-of-care settings, making it a valuable tool in the fight against infectious diseases. The invention also includes methods for optimizing the detection system, ensuring its applicability to a wide range of viral targets and detection scenarios.