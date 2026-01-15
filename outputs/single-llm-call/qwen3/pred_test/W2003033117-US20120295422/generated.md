# DESCRIPTION

## BACKGROUND

- introduce InGaN semiconductor material  
Indium gallium nitride (InGaN) is a ternary compound semiconductor belonging to the group III-nitride family, composed of indium, gallium, and nitrogen in varying atomic ratios. This material system exhibits a tunable bandgap that spans from approximately 0.7 eV to 3.4 eV, enabling emission across the visible spectrum from infrared to ultraviolet. The ability to precisely control the indium content within the InGaN lattice allows for the engineering of optoelectronic properties critical for light-emitting diodes, laser diodes, and photodetectors operating in the green, blue, and near-ultraviolet regions. Due to its direct bandgap nature and high radiative recombination efficiency, InGaN has become the foundational active material in modern solid-state lighting and display technologies. Its compatibility with gallium nitride (GaN) heterostructures further facilitates the formation of quantum wells and barriers essential for efficient carrier confinement and photon generation.

- describe applications of group III-V nitride materials  
Group III-V nitride materials, including GaN, AlN, InN, and their alloys, have revolutionized the field of optoelectronics by enabling high-brightness, energy-efficient light sources that replace traditional incandescent and fluorescent technologies. These materials are extensively deployed in full-color displays, automotive lighting, general illumination, UV sterilization systems, and high-speed optical communication devices. Their robustness under high current densities, thermal stability, and resistance to radiation make them uniquely suited for harsh environments such as aerospace, military, and industrial applications. Moreover, the development of GaN-based LEDs has significantly reduced global electricity consumption for lighting, contributing to sustainability goals. The commercial success of blue and white LEDs, which earned the 2014 Nobel Prize in Physics, underscores the transformative impact of these materials on modern technology and energy infrastructure.

- explain multi-quantum well structure  
A multi-quantum well (MQW) structure consists of alternating thin layers of semiconductor materials with differing bandgaps, typically formed by embedding multiple quantum wells of a lower-bandgap material—such as InGaN—within a higher-bandgap barrier material—such as GaN. Each quantum well acts as a nanoscale potential trap for electrons and holes, confining them in one dimension and enhancing the probability of radiative recombination. The periodic repetition of these wells and barriers creates a superlattice that amplifies the overall light emission efficiency while allowing precise control over the emitted wavelength through variations in well width and indium composition. The quality of the MQW structure is critically dependent on the abruptness of the interfaces between wells and barriers, the uniformity of indium distribution within the wells, and the absence of defects such as dislocations or phase segregation.

- describe limitations of conventional method  
Conventional methods for fabricating InGaN/GaN MQWs involve sequential growth of quantum wells and barriers at distinct temperatures, typically ramping from the lower growth temperature of the InGaN well to the higher temperature required for GaN barrier formation. This thermal transition often leads to thermal decomposition of the indium-rich InGaN layer, resulting in indium segregation, surface roughening, and non-uniform compositional distribution. The resulting inhomogeneities degrade the optical quality of the quantum wells, broadening the emission linewidth and reducing internal quantum efficiency. Additionally, the rapid temperature changes induce strain gradients and interfacial defects that compromise the abruptness of the well-barrier junctions, diminishing carrier confinement and increasing non-radiative recombination pathways. These limitations are particularly acute in green-emitting LEDs, where higher indium content is required but also more thermally unstable.

- motivate need for new method  
The growing demand for high-efficiency green LEDs—essential for full-color displays and energy-efficient white lighting—has exposed the inadequacies of conventional growth protocols. Current methods fail to consistently produce InGaN quantum wells with sufficient indium incorporation and structural integrity, leading to poor device performance, low yield, and elevated manufacturing costs. Without a reliable technique to stabilize the InGaN well during the transition to barrier growth, the industry remains constrained in its ability to achieve the desired wavelength range with narrow emission linewidths and high output power. A method that preserves the integrity of the quantum well during thermal cycling is therefore urgently needed to enable next-generation LEDs with improved efficiency, color purity, and manufacturability.

- summarize challenges of producing light with longer wavelengths  
Producing light at longer wavelengths, particularly in the green region of the spectrum, requires a higher indium content in the InGaN quantum well, which inherently increases lattice mismatch with the GaN barrier and exacerbates strain-induced defects. Higher indium concentrations also lower the thermal stability of the InGaN layer, making it highly susceptible to decomposition during subsequent high-temperature growth steps. Furthermore, the increased strain promotes phase separation and indium clustering, leading to localized emission variations and broadened photoluminescence spectra. These effects collectively reduce the internal quantum efficiency and prevent the realization of high-performance green LEDs. Overcoming these challenges necessitates a novel fabrication approach that decouples the thermal stability of the quantum well from the growth conditions of the barrier, thereby enabling precise control over composition, interface abruptness, and defect density.

## SUMMARY

- introduce method for fabricating active region  
A novel method for fabricating the active region of a gallium nitride-based light-emitting diode is disclosed, wherein the integrity of the indium gallium nitride quantum well is preserved during the transition from well growth to barrier growth through the introduction of a thin, in-situ annealed gallium nitride cap layer. This method enables the formation of high-quality, indium-rich quantum wells with homogeneous composition and abrupt interfaces, significantly enhancing the optical performance of green-emitting devices.

- describe fabricating potential well  
The potential well is fabricated by depositing a thin layer of indium gallium nitride at a growth temperature between 750°C and 850°C using metalorganic chemical vapor deposition, with the indium mole fraction maintained between 0.20 and 0.35 to achieve emission in the green spectral range. The deposition rate is carefully controlled to ensure uniform thickness and stoichiometric composition across the substrate surface, minimizing the formation of indium-rich clusters or compositional fluctuations that would otherwise degrade radiative efficiency.

- describe annealing and stabilizing potential well  
Following the deposition of the InGaN quantum well, a thin gallium nitride layer, ranging in thickness from 5 Å to 40 Å, is deposited at the same temperature without interruption. This cap layer acts as a protective barrier that suppresses the thermal decomposition of the underlying InGaN during subsequent temperature ramping. The cap layer is then subjected to a controlled thermal annealing step, during which surface atoms rearrange to form a smooth, defect-free interface while preserving the indium content and structural integrity of the quantum well. This stabilization process prevents indium segregation and ensures a homogeneous distribution of indium atoms within the well.

- describe fabricating potential barrier  
After stabilization of the quantum well and its cap layer, the growth temperature is incrementally increased by at least 100°C to a range between 850°C and 950°C, at which point a gallium nitride barrier layer is deposited. The abrupt transition from the stabilized cap layer to the barrier material results in a sharp, atomically defined interface with minimal interdiffusion or lattice distortion. The barrier layer is grown to a thickness sufficient to confine carriers within the quantum well, typically between 10 nm and 20 nm, while maintaining low defect density and high crystalline quality.

## DETAILED DESCRIPTION

- provide overview of invention  

### Overview

- describe process for growing single InGaN/GaN quantum well structure  
The process for growing a single InGaN/GaN quantum well structure begins with the preparation of a silicon substrate, which is cleaned and pre-treated to form a nucleation layer of aluminum nitride and aluminum gallium nitride to mitigate lattice mismatch and thermal expansion differences. Following this, a thick n-type gallium nitride layer is deposited as a buffer. The quantum well is then formed by depositing an indium gallium nitride layer at a temperature optimized for indium incorporation, followed immediately by a thin gallium nitride cap layer grown at the same temperature. The structure is then thermally stabilized before transitioning to the higher temperature required for barrier growth.

- illustrate temperature vs. time diagram  
A temperature versus time diagram illustrates the sequence of thermal events during growth, showing a constant temperature plateau during InGaN well deposition, an uninterrupted transition to cap layer deposition at identical temperature, a brief thermal stabilization period with minimal temperature fluctuation, and a controlled ramp to the higher barrier growth temperature. The diagram demonstrates that the cap layer is grown and annealed without thermal interruption, preserving the quantum well’s structural and compositional integrity.

- describe fabricating InGaN/GaN MQW structure  
The multi-quantum well structure is formed by repeating the sequence of InGaN well, GaN cap, and GaN barrier layers in a periodic fashion, typically six to ten times. Each period is grown under identical conditions to ensure uniformity across the entire active region. The cap layer thickness is optimized to balance protection against decomposition with minimal optical absorption, resulting in a structure with high internal quantum efficiency and narrow emission linewidth.

- illustrate cross-section view of exemplary LED  
A cross-sectional view of an exemplary light-emitting diode shows the silicon substrate, followed by a stress-relieving AlN/AlGaN buffer, an n-type GaN layer, a stack of alternating InGaN quantum wells and GaN barriers with intervening GaN cap layers, a p-type GaN:Mg layer, and a transparent conductive oxide contact. The cap layers are depicted as thin, continuous interfaces between wells and barriers, emphasizing their role in preserving well integrity.

- describe conventional process for fabricating active region  
In conventional fabrication, the InGaN quantum well is deposited at a low temperature, followed by an abrupt temperature increase to grow the GaN barrier. This rapid ramp causes thermal decomposition of the InGaN layer, leading to indium out-diffusion, surface roughening, and poor interface abruptness. The resulting inhomogeneities manifest as broad photoluminescence peaks and reduced emission intensity.

- illustrate flow chart of conventional process  
A flow chart of the conventional process depicts sequential steps: substrate preparation, buffer growth, InGaN well deposition, rapid temperature ramp, GaN barrier deposition, and p-type layer growth. Arrows indicate the temperature jump between well and barrier steps, annotated with a warning symbol indicating decomposition risk.

- describe limitations of conventional method  
The conventional method suffers from irreversible degradation of the quantum well during temperature transitions, resulting in non-uniform indium distribution, defective interfaces, and increased non-radiative recombination. These limitations are exacerbated for green-emitting devices, where higher indium content renders the well more susceptible to thermal instability.

- motivate need for new method  
The need for a method that decouples thermal stress from quantum well formation is paramount to achieving high-efficiency green LEDs. Without stabilization of the InGaN layer during barrier growth, the optical performance of the device remains suboptimal, limiting commercial viability.

- describe process for fabricating active region of GaN-based LED  
The new process introduces a cap layer grown at the well temperature immediately after InGaN deposition, followed by a controlled annealing step that stabilizes the interface before the temperature is increased for barrier growth. This eliminates decomposition and enables abrupt, high-quality interfaces.

- illustrate flow chart of new process  
A flow chart of the new process shows substrate preparation, buffer growth, InGaN well deposition, cap layer deposition at same temperature, thermal stabilization, temperature ramp, barrier deposition, and p-type layer growth. The cap layer and stabilization steps are highlighted as critical innovations.

- describe fabricating potential well  
The potential well is formed by depositing InGaN with indium content between 20% and 35% at a temperature of 800°C, ensuring optimal indium incorporation without decomposition. The layer thickness is maintained between 2 nm and 4 nm to balance carrier confinement and strain management.

- describe annealing and stabilizing potential well  
The cap layer, deposited immediately after the well, is held at the same temperature for 30 to 120 seconds, allowing surface atoms to reorganize and form a stable, defect-free interface. This annealing step prevents indium segregation and locks in the desired composition.

- describe fabricating potential barrier  
The barrier is grown at a temperature 100°C to 150°C higher than the well, with the cap layer acting as a buffer that prevents thermal shock to the quantum well. The resulting interface is atomically sharp, with minimal interdiffusion.

- summarize advantages of new method  
The new method significantly improves the uniformity of indium distribution, enhances interface abruptness, reduces photoluminescence linewidth, increases emission intensity, and enables high-efficiency green LED production with improved yield and reproducibility.

## EXAMPLE

- describe exemplary embodiment of fabricating active region  
An exemplary embodiment involves the growth of a six-period InGaN/GaN MQW structure on a silicon substrate using metalorganic chemical vapor deposition. The InGaN quantum wells are grown at 800°C with an indium composition of 28%, each with a thickness of 3 nm. Immediately following each well, a 15 Å thick GaN cap layer is deposited at the same temperature. The structure is then annealed for 60 seconds at 800°C before the temperature is increased to 920°C for the growth of a 15 nm GaN barrier. This sequence is repeated six times, followed by deposition of a 130 nm p-type GaN:Mg layer.

- describe fabricating InGaN potential well  
The InGaN potential well is formed by introducing trimethylindium and ammonia at a flow rate ratio of 1:500, with trimethylgallium supplied at a rate sufficient to maintain a Ga:N ratio of 1:1. The substrate temperature is stabilized at 800°C for five minutes prior to growth, ensuring uniform nucleation and minimizing surface roughness.

- describe annealing and stabilizing potential well  
After deposition of the 15 Å GaN cap layer, the growth chamber is held at 800°C for 60 seconds under a continuous flow of ammonia and hydrogen. During this period, surface diffusion of indium atoms is enhanced, allowing redistribution into energetically favorable sites and suppression of phase segregation. The cap layer effectively seals the quantum well, preventing indium loss.

- describe fabricating GaN barrier  
The GaN barrier is grown by increasing the reactor temperature to 920°C and adjusting the trimethylgallium and ammonia flow rates to achieve a growth rate of 0.8 μm/h. The transition from cap to barrier is seamless, resulting in an interface with a root-mean-square roughness of less than 0.5 nm, as confirmed by atomic force microscopy. The resulting structure exhibits a photoluminescence peak at 530 nm with a full width at half maximum of 28 nm, demonstrating superior optical quality compared to conventional structures.