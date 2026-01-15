Here is the patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

The present invention relates generally to the field of genome editing and DNA repair mechanisms. In eukaryotic cells, DNA double-strand breaks (DSBs) are primarily repaired through two major pathways: non-homologous end joining (NHEJ) and homology-directed repair (HDR). NHEJ represents an error-prone repair mechanism that often results in small insertions or deletions at the break site, while HDR enables precise repair using a homologous template.  

The NHEJ pathway involves key proteins including KU70, KU80, and DNA ligase IV (LIG4) that mediate direct ligation of broken DNA ends. In contrast, HDR requires proteins such as CDK1, CtIP, BRCA1, and BRCA2 to facilitate accurate template-based repair. While both pathways serve important biological functions, NHEJ dominates in mammalian cells, particularly during the G1 phase of the cell cycle. This presents a significant limitation for precision genome editing applications that rely on HDR-mediated repair.  

The CRISPR-Cas9 system has emerged as a powerful genome editing tool that creates targeted DSBs. However, the efficiency of CRISPR-mediated HDR remains substantially lower than NHEJ in most cell types. Current approaches to enhance HDR efficiency involve chemical inhibitors or cell cycle synchronization, but these methods suffer from toxicity, limited efficacy, and practical challenges for therapeutic applications. There exists an unmet need for genetic tools that can selectively modulate DNA repair pathway activity to improve HDR outcomes.  

## SUMMARY OF THE INVENTION  

The present invention provides compositions and methods for enhancing homology-directed repair (HDR) during genome editing. The invention utilizes catalytically dead guide RNAs (dgRNAs) in combination with CRISPR activation (CRISPRa) and interference (CRISPRi) systems to simultaneously upregulate HDR factors and downregulate NHEJ components.  

Key aspects of the invention include:  
- A vector comprising a dgRNA targeting HDR genes such as CDK1 or CtIP, operably linked to an MS2 binding loop for recruitment of transcriptional activators  
- A vector comprising a dgRNA targeting NHEJ genes such as KU70, KU80 or LIG4, operably linked to a Com binding loop for recruitment of transcriptional repressors  
- A nonfunctional green fluorescent reporter system (Traffic Light Reporter) for quantifying HDR and NHEJ events  
- A reverse tetracycline-controlled transactivator (rtTA) sequence enabling inducible expression of CRISPRa/i components  
- A composition combining the dgRNA vectors with an active Cas9 nuclease for simultaneous genome editing and DNA repair pathway modulation  
- Methods for enhancing HDR efficiency in cells by introducing the activation and repression vectors  
- A kit comprising the vectors and instructional materials for implementing the HDR enhancement system  

The invention provides a genetic approach to shift the balance of DNA repair toward HDR, enabling more efficient precision genome editing across diverse cell types and applications.  

## DETAILED DESCRIPTION  

### Definitions  

The following terms are defined for clarity in describing the invention:  

"a" and "an" refer to one or more unless the context clearly indicates otherwise.  

"about" indicates an approximate range that would be understood by those skilled in the art, typically within ±10% of the stated value.  

"autologous" refers to cells or tissues derived from the same individual.  

"allogeneic" refers to cells or tissues derived from a different individual of the same species.  

"bp" refers to base pairs of nucleic acids.  

"complementary" describes nucleic acid sequences capable of forming Watson-Crick base pairs.  

"CRISPR/Cas" refers to the clustered regularly interspaced short palindromic repeats and CRISPR-associated protein systems.  

"CRISPR/Cas9" specifically denotes the type II CRISPR system utilizing the Cas9 nuclease.  

"CRISPRa" refers to CRISPR activation systems that increase gene expression.  

"dCas9" indicates a catalytically dead mutant of Cas9 that retains DNA binding capacity.  

"dgRNA" refers to dead guide RNAs that lack catalytic activity but maintain target binding.  

Additional definitions are provided throughout the specification as needed for clarity.  

### Description  

The invention leverages CRISPR systems to overcome limitations in current genome editing technologies. While CRISPR-Cas9 enables targeted DNA breaks, the natural preference for NHEJ over HDR restricts precise editing applications. The present approach combines active Cas9-mediated editing with dgRNA-based transcriptional modulation to enhance HDR outcomes.  

Key advantages include:  
- Simultaneous genome editing and DNA repair pathway modulation using a single Cas9 protein  
- Genetic rather than chemical modulation avoids toxicity concerns  
- Modular design allows targeting of multiple HDR and NHEJ factors  
- Compatibility with inducible expression systems for temporal control  
- Adaptability to viral delivery methods for diverse applications  

This represents a significant advance over existing methods by providing precise control over DNA repair pathway balance through programmable genetic components.  

### Compositions  

The invention provides several vector compositions for HDR enhancement:  

The activation vector comprises a dgRNA targeting HDR genes (e.g., CDK1, CtIP) with MS2 binding loops, enabling recruitment of the MS2-P65-HSF1 (MPH) transcriptional activation complex. The vector includes a U6 promoter driving dgRNA expression and a CMV promoter for MPH component expression.  

The repression vector comprises a dgRNA targeting NHEJ genes (e.g., KU70, KU80, LIG4) with Com binding loops for recruiting the COM-KRAB (CK) repressor domain. Similar regulatory elements control expression of the dgRNA and CK components.  

An inducible version incorporates the rtTA system with TRE3G promoters for doxycycline-controlled expression of MPH or CK effectors. The traffic light reporter vector contains a nonfunctional EGFP variant (bf-Venus) and frameshifted mCherry to quantify HDR and NHEJ events.  

Vectors may include nuclear localization signals, linker sequences, and selection markers. Suitable promoters include constitutive (CMV, EF1α), tissue-specific, or inducible (tetracycline, alcohol-regulated) variants. The components can be packaged into lentiviral or other viral vectors for efficient delivery.  

### Methods  

The invention provides methods for enhancing HDR efficiency in cells:  

1. Introducing an activation plasmid comprising dgRNAs targeting HDR genes (CDK1, CtIP, BRCA1/2) with MS2 binding loops and MPH activator  
2. Introducing a repression plasmid comprising dgRNAs targeting NHEJ genes (KU70, KU80, LIG4) with Com binding loops and CK repressor  
3. Delivering an active Cas9 nuclease and sgRNA to create targeted DSBs  
4. Providing an HDR donor template for precise repair  
5. Optionally using inducible systems (rtTA/TRE3G) for temporal control  

The methods can be implemented via transfection, electroporation, or viral transduction. Cells may be autologous or allogeneic, including stem cells and primary cells. The approach is compatible with both in vitro and in vivo applications.  

Additional method details include:  
- Guide RNA design principles for optimal targeting  
- Viral packaging protocols for lentiviral delivery  
- Administration routes for in vivo applications  
- Verification assays for HDR efficiency measurement  

### Pharmaceutical Compositions  

The invention encompasses pharmaceutical compositions containing the genetic components formulated for therapeutic delivery. Such compositions may include:  

- Viral vectors (lentivirus, AAV) encoding the HDR enhancement system  
- Lipid nanoparticles for nucleic acid delivery  
- Pharmaceutically acceptable carriers and excipients  
- Buffers and stabilizers for nucleic acid preservation  

Administration may occur via injection, infusion, or other parenteral routes. Dosages are adjusted based on target tissue and therapeutic objectives. The compositions enable ex vivo cell modification or direct in vivo genome editing applications.  

## EXPERIMENTAL EXAMPLES  

### Example 1  

The HDR enhancement system was validated using the Traffic Light Reporter (TLR) in HEK293-Cas9 cells. Cells were transfected with:  

1. dgRNA-MS2:MPH targeting CDK1 or CtIP  
2. dgRNA-Com:CK targeting KU70 or KU80  
3. Cas9/sgRNA targeting the TLR locus  
4. EGFP HDR donor template  

Flow cytometry analysis showed HDR efficiency increased from 2.42% (control) to 15.4% (dgCDK1-2 + dgKU80-1). Genomic sequencing confirmed precise HDR-mediated repair in EGFP+ cells.  

Similar enhancements were observed at endogenous loci (AAVS1, ACTB) across multiple cell lines (HEK293, HEK293T, HeLa), with 4-8 fold HDR improvement. Inducible systems (TRE-MPH, TRE-CK) enabled doxycycline-controlled enhancement. Lentiviral delivery maintained efficacy, demonstrating adaptability to viral vectors.  

### Other Embodiments  

The invention encompasses additional embodiments including:  
- Alternative HDR and NHEJ gene targets  
- Different CRISPR effectors (Cpf1, Cas12a)  
- Variant transcriptional regulators  
- Tissue-specific delivery approaches  
- Combination with other genome editing technologies  

The scope of the invention is defined by the claims rather than these specific examples.