# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to a method and system for isolating and analyzing protein complexes without disrupting cellular integrity. More specifically, the invention involves the use of virus-like particles (VLPs) to trap and preserve protein complexes under native conditions, thereby facilitating the identification and characterization of protein-protein interactions (PPIs) and protein-small molecule interactions.

## BACKGROUND

Protein-protein interactions (PPIs) play a crucial role in various biological processes, including signal transduction, cell cycle regulation, and disease pathogenesis. Traditional methods for studying PPIs, such as affinity purification-mass spectrometry (AP-MS), often require cell lysis, which can disrupt the native environment and lead to the loss of complex integrity. The lysis conditions used in AP-MS strategies are critical for preserving protein complexes, and a multitude of lysis conditions have been described, each with its own advantages and limitations. This variability in lysis conditions can introduce significant experimental challenges and affect the reliability of the results.

To address these limitations, alternative methods have been developed, such as BioID and APEX, which label proximal proteins with biotin using an enzymatic activity fused to a bait protein. However, these methods also have their own set of limitations and may not fully capture the complexity of protein interactions under native conditions.

The present invention introduces a novel approach called Virotrap, which utilizes virus-like particles (VLPs) to trap and preserve protein complexes under native conditions. By expressing a bait protein fused to the GAG protein of a retrovirus, the bait protein and its interaction partners are incorporated into VLPs, which are then purified and analyzed. This method eliminates the need for cell lysis and provides a powerful tool for the unbiased discovery of novel protein interactions and the detection of weak PPIs.

## BRIEF SUMMARY

The present invention provides a method and system for isolating and analyzing protein complexes using virus-like particles (VLPs). The method involves expressing a bait protein fused to the GAG protein of a retrovirus, which results in the formation of VLPs containing the bait protein and its interaction partners. The VLPs are then purified and analyzed to identify and characterize the protein complexes.

In one aspect, the invention provides a method for detecting protein-protein interactions (PPIs) comprising:
1. Expressing a bait protein fused to the GAG protein of a retrovirus in a host cell.
2. Forming virus-like particles (VLPs) containing the bait protein and its interaction partners.
3. Purifying the VLPs.
4. Analyzing the VLPs to identify the interaction partners of the bait protein.

In another aspect, the invention provides a method for detecting protein-small molecule interactions comprising:
1. Expressing a bait protein fused to the GAG protein of a retrovirus in a host cell.
2. Treating the host cell with a small molecule of interest.
3. Forming virus-like particles (VLPs) containing the bait protein, its interaction partners, and the small molecule.
4. Purifying the VLPs.
5. Analyzing the VLPs to identify the interaction partners of the bait protein and the small molecule.

The invention also provides a system for implementing the above methods, including vectors for expressing the bait protein-GAG fusion, host cells for producing the VLPs, and reagents and equipment for purifying and analyzing the VLPs.

## DETAILED DESCRIPTION

### Definitions

- **Virus-like particles (VLPs):** Non-infectious, self-assembling structures that mimic the organization and conformation of authentic viruses but lack the viral genome. VLPs can be produced by expressing viral structural proteins, such as the GAG protein of a retrovirus, in host cells.
- **Bait protein:** A protein of interest that is fused to the GAG protein of a retrovirus to form VLPs containing the bait protein and its interaction partners.
- **Interaction partners:** Proteins that interact with the bait protein and are incorporated into the VLPs.
- **Host cell:** A cell type, such as HEK293T cells, used to express the bait protein-GAG fusion and produce VLPs.
- **Purification:** The process of isolating VLPs from the host cell culture supernatant using techniques such as ultracentrifugation or antibody-based capture.
- **Analysis:** The process of identifying and characterizing the proteins and small molecules present in the VLPs using techniques such as western blotting, mass spectrometry, or other biochemical assays.

### EXAMPLES

#### Example 1: Detection of Binary Interactions Using Virotrap

To demonstrate the ability of Virotrap to detect binary interactions, a set of known protein-protein interaction (PPI) pairs was selected based on published evidence and cytosolic localization. The bait proteins were fused to the GAG protein of a retrovirus, and the prey proteins were expressed with an E-tag. Both the bait and prey constructs were co-transfected into HEK293T cells. After 24 hours, the VLPs were purified using a single-step protocol involving the co-expression of the vesicular stomatitis virus glycoprotein (VSV-G) and a tagged version of this glycoprotein. The VLPs were then analyzed by western blotting to detect the presence of the prey proteins.

Results showed that Virotrap could readily detect reciprocal interactions between several PPI pairs, including CDK2 and CKS1B, LCP2 and GRAP2, and S100A1 and S100B. The method also detected 28 (30%) interactions in the human positive reference set (hsPRS-v1) and 5 (5%) interactions in the random reference set (hsRRS-v1), demonstrating the sensitivity and specificity of Virotrap for detecting binary interactions.

#### Example 2: Detection of Weak PPIs Using Virotrap

To evaluate the sensitivity of Virotrap for detecting weak protein-protein interactions, a panel of MYC-tagged MAL mutant prey proteins with reduced binding affinities was tested against the MYD88 TIR domain as bait. The VLPs were purified and analyzed by western blotting to detect the presence of the prey proteins.

Results showed that Virotrap generally exhibited the same trend as data obtained with the mammalian PPI trap (MAPPIT) assay, a mammalian two-hybrid method. This indicates that Virotrap is capable of detecting weak PPIs, making it a valuable tool for studying protein interactions with low binding affinities.

#### Example 3: Unbiased Discovery of Novel Interactions Using Virotrap

To demonstrate the utility of Virotrap for the unbiased discovery of novel protein interactions, VLP production and purification protocols were scaled up, and the method was applied to several bait proteins, including FADD, A20, IKBKG, TANK, MYD88, and RNF41. The VLPs were purified and analyzed by mass spectrometry to identify the interaction partners of the bait proteins.

Results revealed both known and novel candidate interaction partners. For example, the interaction between A20 and FADD, and the associations with transmembrane proteins (insulin receptor and insulin-like growth factor receptor 1) captured using RNF41 as a bait, were confirmed using alternative methods. Additionally, Virotrap was used to detect dynamic interactions by activating the NF-κB pathway via the tumor necrosis factor (TNF) receptor (TNFRSF1A) using TNFα and performing Virotrap analysis using A20 as bait. This resulted in the additional enrichment of receptor-interacting kinase (RIPK1), TNFR1-associated via death domain (TRADD), TNFRSF1A, and TNF itself, confirming the expected activated complex.

#### Example 4: Detection of Protein-Small Molecule Interactions Using Virotrap

To explore the use of Virotrap for detecting protein interactions with small molecules, the Escherichia coli dihydrofolate reductase protein (eDHFR) was fused to the GAG protein as a bait, and the particle-producing cells were treated with bivalent molecules consisting of methotrexate (MTX) linked via a polyethylene glycol linker to small molecules of interest, including simvastatin, tamoxifen, and reversine.

The VLPs were purified and analyzed by mass spectrometry to identify the interaction partners of the small molecules. Results showed the consistent enrichment of the known targets for simvastatin (HMG-CoA reductase enzyme (HMGCR)) and reversine (Aurora kinase A (AURKA)). The novel interactions of tamoxifen with HSDB17B4 and reversine with NQO2 were confirmed using an orthogonal MASPIT assay, demonstrating the capability of Virotrap to detect protein-small molecule interactions.

### Conclusion

The Virotrap method provides a powerful and complementary approach to traditional AP-MS and other PPI detection methods. By trapping protein complexes in the protective environment of VLPs, Virotrap preserves complex integrity and enables the detection of both known and novel protein interactions, as well as protein-small molecule interactions. Future improvements to the system, including strategies to reduce bait expression and advanced data analysis options, can further enhance the utility of Virotrap in the field of protein complex analysis.