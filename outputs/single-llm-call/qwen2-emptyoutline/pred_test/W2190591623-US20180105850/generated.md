# DESCRIPTION

## TECHNICAL FIELD OF THE INVENTION

The present invention relates to the field of biotechnology and metabolic engineering, specifically to the production of L-ornithine using genetically engineered yeast strains. More particularly, the invention involves the optimization of the L-ornithine biosynthetic pathway in *Saccharomyces cerevisiae* through modular pathway engineering and subcellular trafficking engineering to achieve high-level production of L-ornithine.

## BACKGROUND OF THE INVENTION

L-ornithine is an important non-proteinogenic amino acid with various industrial and medical applications. It is a key intermediate in the urea cycle and is also used in the synthesis of L-arginine and polyamines. Despite its significance, the industrial production of L-ornithine has been challenging due to the complex regulatory mechanisms and compartmentalization of the biosynthetic pathway in microorganisms.

*Saccharomyces cerevisiae* (baker's yeast) is a widely used host organism in biotechnological processes due to its well-characterized genetics and robust physiology. However, the production of L-ornithine in *S. cerevisiae* is hindered by several factors, including the Crabtree effect, which diverts carbon flux towards ethanol production, and the compartmentalization of the L-ornithine biosynthetic pathway across different cellular compartments.

To address these challenges, the present invention employs a modular pathway reprogramming (MPR) strategy to systematically optimize the L-ornithine biosynthetic pathway in *S. cerevisiae*. This approach involves the re-casting of the pathway into three distinct modules: (1) L-ornithine degradation or consumption, (2) L-ornithine synthesis, and (3) α-ketoglutarate synthesis. Additionally, the invention includes strategies to alleviate the Crabtree effect and to optimize subcellular trafficking of key intermediates to enhance L-ornithine production.

## SUMMARY OF THE INVENTION

The present invention provides a method for producing L-ornithine using genetically engineered *Saccharomyces cerevisiae* strains. The method involves the following steps:

1. **Re-casting the L-ornithine biosynthetic pathway into three modules**:
   - **Module 1**: L-ornithine degradation or consumption, which includes the reactions involving ornithine carbamoyltransferase (OTC; ARG3) and ornithine aminotransferase (CAR2), as well as the arginase reaction (CAR1).
   - **Module 2**: L-ornithine synthesis, which encompasses the conversion of α-ketoglutarate to L-ornithine through a series of enzymatic reactions.
   - **Module 3**: α-ketoglutarate synthesis, which involves the glucose uptake reactions, glycolysis, and the upstream part of the TCA cycle, including the respiratory chain.

2. **Optimizing L-ornithine consumption**:
   - Fine-tuning the expression of ARG3 to create a leaky arginine auxotrophy, allowing for low-level L-arginine synthesis to support growth while minimizing negative regulation on L-ornithine biosynthesis.
   - Deleting the CAR2 gene to block the futile cycle of L-ornithine transamination.

3. **Subcellular trafficking engineering**:
   - Enhancing mitochondrial L-ornithine biosynthesis by overexpressing key genes (ARG5,6, ARG7, ARG8, and ARG2).
   - Overexpressing transporters (ORT1 and AGC1) to improve the transport of L-ornithine and L-glutamate between the mitochondria and cytoplasm.
   - Re-localizing the entire L-ornithine biosynthetic pathway to the cytoplasm by introducing a synthetic cytosolic pathway from *Escherichia coli* and *Corynebacterium glutamicum*.

4. **Alleviating the Crabtree effect**:
   - Overexpressing TCA cycle genes to enhance α-ketoglutarate supply.
   - Overexpressing alternative NADH oxidases (AOX and NDI1) to improve NADH consumption.
   - Reducing glucose uptake rate by overexpressing a truncated form of MTH1 (MTH1-ΔT) to minimize overflow metabolism to ethanol.

5. **Urea cycle engineering**:
   - Overexpressing CAR1 to increase L-ornithine synthesis and reduce the L-arginine pool, thereby preventing feedback inhibition.

The invention also provides genetically engineered *S. cerevisiae* strains capable of producing high levels of L-ornithine, as well as methods for their cultivation and use in industrial processes.

## DETAILED DESCRIPTION

The present invention is directed to the production of L-ornithine using genetically engineered *Saccharomyces cerevisiae* strains. The detailed description below outlines the specific strategies and methods employed to achieve high-level production of L-ornithine.

### Re-casting the L-Ornithine Biosynthetic Pathway into Three Modules

The L-ornithine biosynthetic pathway in *S. cerevisiae* is complex and involves multiple enzymatic reactions distributed across different cellular compartments. To optimize this pathway, the invention employs a modular pathway reprogramming (MPR) strategy, dividing the pathway into three distinct modules:

1. **Module 1: L-Ornithine Degradation or Consumption**
   - This module includes the reactions involving ornithine carbamoyltransferase (OTC; ARG3) and ornithine aminotransferase (CAR2), as well as the arginase reaction (CAR1). These enzymes are responsible for the degradation or consumption of L-ornithine, which can divert the flux away from L-ornithine production.
   - **Optimization Strategy**: Fine-tune the expression of ARG3 to create a leaky arginine auxotrophy, allowing for low-level L-arginine synthesis to support growth while minimizing negative regulation on L-ornithine biosynthesis. Additionally, delete the CAR2 gene to block the futile cycle of L-ornithine transamination.

2. **Module 2: L-Ornithine Synthesis**
   - This module encompasses the conversion of α-ketoglutarate to L-ornithine through a series of enzymatic reactions. The key enzymes involved in this module include those encoded by ARG5,6, ARG7, ARG8, and ARG2.
   - **Optimization Strategy**: Enhance mitochondrial L-ornithine biosynthesis by overexpressing these key genes. Additionally, overexpress transporters (ORT1 and AGC1) to improve the transport of L-ornithine and L-glutamate between the mitochondria and cytoplasm. Re-localize the entire L-ornithine biosynthetic pathway to the cytoplasm by introducing a synthetic cytosolic pathway from *E. coli* and *C. glutamicum*.

3. **Module 3: α-Ketoglutarate Synthesis**
   - This module involves the glucose uptake reactions, glycolysis, and the upstream part of the TCA cycle, including the respiratory chain. The key enzymes involved in this module include those encoded by PDA1, PYC2, CIT1, ACO2, and IDP1.
   - **Optimization Strategy**: Overexpress TCA cycle genes to enhance α-ketoglutarate supply. Overexpress alternative NADH oxidases (AOX and NDI1) to improve NADH consumption. Reduce glucose uptake rate by overexpressing a truncated form of MTH1 (MTH1-ΔT) to minimize overflow metabolism to ethanol.

### Leaky Arginine Auxotrophy Enables L-Ornithine Production

One of the key challenges in producing L-ornithine is the feedback inhibition and repression of the L-ornithine biosynthetic pathway by L-arginine. To overcome this, the invention employs a strategy to create a leaky arginine auxotrophy by fine-tuning the expression of the ARG3 gene, which encodes ornithine carbamoyltransferase (OTC).

- **Expression Tuning**: Replace the native promoter of ARG3 with either the glucose-regulated HXT1 promoter or the low-activity KEX2 promoter. This allows for low-level expression of ARG3, enabling the production of L-ornithine while maintaining sufficient L-arginine synthesis to support growth.
- **Gene Deletion**: Delete the CAR2 gene to block the futile cycle of L-ornithine transamination, which can further enhance L-ornithine production.

### Subcellular Trafficking Engineering and Pathway Translocation

The L-ornithine biosynthetic pathway in *S. cerevisiae* is compartmentalized, with key intermediates such as L-glutamate, α-ketoglutarate, and L-ornithine synthesized in different cellular compartments. To optimize the pathway, the invention employs subcellular trafficking engineering and pathway translocation strategies.

- **Mitochondrial L-Ornithine Biosynthesis**: Overexpress key genes (ARG5,6, ARG7, ARG8, and ARG2) involved in mitochondrial L-ornithine biosynthesis. Additionally, overexpress transporters (ORT1 and AGC1) to improve the transport of L-ornithine and L-glutamate between the mitochondria and cytoplasm.
- **Cytosolic Pathway Re-localization**: Introduce a synthetic cytosolic pathway from *E. coli* and *C. glutamicum* to re-localize the entire L-ornithine biosynthetic pathway to the cytoplasm. This strategy can enhance L-ornithine production by concentrating the enzymes and intermediates in a single compartment.

### Crabtree Effect Attenuation Improves Carbon Flux to L-Ornithine

The Crabtree effect in *S. cerevisiae* diverts a significant portion of the carbon flux towards ethanol production, limiting the availability of carbon for L-ornithine synthesis. To alleviate this effect, the invention employs several strategies:

- **Overexpression of TCA Cycle Genes**: Overexpress genes involved in the conversion of pyruvate to α-ketoglutarate, including PDA1, PYC2, CIT1, ACO2, and IDP1, to enhance α-ketoglutarate supply.
- **Overexpression of Alternative NADH Oxidases**: Overexpress alternative NADH oxidases (AOX and NDI1) to improve NADH consumption and reduce overflow metabolism to ethanol.
- **Reduction of Glucose Uptake Rate**: Overexpress a truncated form of MTH1 (MTH1-ΔT) to reduce glucose uptake rate and minimize overflow metabolism to ethanol.

### Urea Cycle Engineering Enables L-Ornithine Titre Improvement

The urea cycle in *S. cerevisiae* can be exploited to enhance L-ornithine production. The arginase reaction, encoded by CAR1, degrades L-arginine into urea and L-ornithine. By overexpressing CAR1, the invention aims to increase L-ornithine synthesis and reduce the L-arginine pool, thereby preventing feedback inhibition.

- **Overexpression of CAR1**: Overexpress the CAR1 gene to enhance the arginase reaction and increase L-ornithine production.

## EXAMPLES

### Strain Construction of Ornithine-Overproducing Strains

The construction of ornithine-overproducing strains involved a series of genetic modifications to optimize the L-ornithine biosynthetic pathway. The following steps were taken:

1. **Fine-Tuning ARG3 Expression**: Replace the native promoter of ARG3 with the HXT1 or KEX2 promoter to create a leaky arginine auxotrophy.
2. **Deletion of CAR2**: Delete the CAR2 gene to block the futile cycle of L-ornithine transamination.
3. **Overexpression of Mitochondrial L-Ornithine Biosynthesis Genes**: Overexpress ARG5,6, ARG7, ARG8, and ARG2 to enhance mitochondrial L-ornithine biosynthesis.
4. **Overexpression of Transporters**: Overexpress ORT1 and AGC1 to improve the transport of L-ornithine and L-glutamate between the mitochondria and cytoplasm.
5. **Re-localization of the Cytosolic Pathway**: Introduce a synthetic cytosolic pathway from *E. coli* and *C. glutamicum* to re-localize the entire L-ornithine biosynthetic pathway to the cytoplasm.
6. **Overexpression of TCA Cycle Genes**: Overexpress PDA1, PYC2, CIT1, ACO2, and IDP1 to enhance α-ketoglutarate supply.
7. **Overexpression of Alternative NADH Oxidases**: Overexpress AOX and NDI1 to improve NADH consumption and reduce overflow metabolism to ethanol.
8. **Reduction of Glucose Uptake Rate**: Overexpress MTH1-ΔT to reduce glucose uptake rate and minimize overflow metabolism to ethanol.
9. **Overexpression of CAR1**: Overexpress the CAR1 gene to enhance the arginase reaction and increase L-ornithine production.

### L-Arginine Leaky Auxotroph Enables L-Ornithine Overproduction

To create a leaky arginine auxotrophy, the native promoter of the ARG3 gene was replaced with the HXT1 or KEX2 promoter. This allowed for low-level expression of ARG3, enabling the production of L-ornithine while maintaining sufficient L-arginine synthesis to support growth. The deletion of the CAR2 gene further enhanced L-ornithine production by blocking the futile cycle of L-ornithine transamination.

### Pathway Re-Localization and Subcellular Trafficking Engineering Elevates L-Ornithine Synthesis

To optimize the L-ornithine biosynthetic pathway, the invention employed subcellular trafficking engineering and pathway translocation strategies. Key genes involved in mitochondrial L-ornithine biosynthesis (ARG5,6, ARG7, ARG8, and ARG2) were overexpressed, and transporters (ORT1 and AGC1) were overexpressed to improve the transport of L-ornithine and L-glutamate between the mitochondria and cytoplasm. Additionally, a synthetic cytosolic pathway from *E. coli* and *C. glutamicum* was introduced to re-localize the entire L-ornithine biosynthetic pathway to the cytoplasm, enhancing L-ornithine production.

### ‘Crabtree Negative’ S. cerevisiae Construction Enables Efficient Carbon Channeling to L-Ornithine

To alleviate the Crabtree effect, the invention overexpressed TCA cycle genes (PDA1, PYC2, CIT1, ACO2, and IDP1) to enhance α-ketoglutarate supply. Alternative NADH oxidases (AOX and NDI1) were overexpressed to improve NADH consumption and reduce overflow metabolism to ethanol. The glucose uptake rate was reduced by overexpressing MTH1-ΔT, further minimizing overflow metabolism to ethanol and enhancing L-ornithine production.

### Example 5

The final engineered strain, M1dM2qM3e, overexpressed CAR1 to enhance the arginase reaction and increase L-ornithine production. This strain achieved a final L-ornithine titre of 1,041 mg/l, representing a 23-fold improvement in L-ornithine production compared to the parent strain. The combination of modular pathway reprogramming, subcellular trafficking engineering, and Crabtree effect attenuation strategies demonstrated the potential of *S. cerevisiae* for high-level production of L-ornithine.

This detailed description and examples illustrate the effectiveness of the invention in optimizing the L-ornithine biosynthetic pathway in *S. cerevisiae* and achieving high-level production of L-ornithine. The genetically engineered strains and methods described herein provide a robust platform for industrial applications in the production of L-ornithine and other amino acid-derived chemicals.