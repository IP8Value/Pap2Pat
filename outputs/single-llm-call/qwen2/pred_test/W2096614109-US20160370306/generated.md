# DESCRIPTION

## BACKGROUND

Serial femtosecond crystallography (SFX) is a cutting-edge structural biology technique designed to solve challenging protein structures from submicrometer to micrometer-sized crystals at room temperature. In SFX, nanocrystals or microcrystals are delivered into the beam path of a hard X-ray free-electron laser (XFEL) using a liquid or viscous stream. The XFEL radiation consists of femtosecond pulses typically delivered at a rate of 1 to 120 Hz, and diffraction patterns are captured before the crystals are destroyed. This method has revolutionized the field of structural biology, particularly for proteins that are difficult to crystallize in large quantities.

However, traditional methods for delivering protein crystals to the XFEL beam, such as the gas dynamic virtual nozzle (GDVN) liquid jet, have significant drawbacks. The GDVN liquid jet moves at a velocity of 10 to 20 m/s, delivering crystals much faster than required to replenish the protein crystals between X-ray pulses at a pulse repetition rate of 120 Hz. Consequently, only about one out of every 10,000 crystals is probed by the X-ray pulses, leading to high protein consumption and making it particularly problematic for membrane proteins and other proteins that can only be produced in small amounts.

Membrane proteins, which constitute 60% of current drug targets, are especially challenging due to their insolubility in water and the necessity to extract them from the membrane in the form of protein-detergent micelles. While crystallization in lipidic cubic phase (LCP) has been successful for many membrane proteins, it is often difficult to crystallize large multi-domain membrane complexes in LCP due to the curvature of the lipid bilayer and the low diffusion constants of large membrane protein complexes.

To address these challenges, a new delivery medium based on agarose has been developed. Agarose, a versatile polysaccharide polymer extracted from seaweed, dissolves in water at high temperatures and forms a gel upon cooling. This property makes it an ideal candidate for embedding and delivering protein crystals to the XFEL beam. The agarose medium can be used over a wide range of temperatures and is compatible with various crystallization conditions, making it a promising alternative to existing delivery methods.

## SUMMARY OF THE INVENTION

The present invention relates to a method for delivering protein crystals to an X-ray free-electron laser (XFEL) beam using an agarose-based medium. The method involves embedding pre-grown protein crystals into a solution of agarose and glycerol, which is then extruded as a continuous stream to the XFEL beam. The agarose medium provides a stable and continuous stream, maintains crystal integrity, and produces minimal background scattering, making it suitable for both soluble and membrane protein crystals.

The key features of the invention include:
1. **Agarose-Glycerol Solution**: A solution of 5.6% agarose and 30% glycerol is prepared to form a stable, continuous stream.
2. **Crystal Embedding**: Pre-grown protein crystals are embedded into the agarose-glycerol solution using a syringe setup, ensuring a homogeneous distribution of crystals.
3. **Extrusion**: The agarose stream containing the embedded crystals is extruded at a flow rate of 160 nl/min, significantly reducing protein consumption compared to traditional methods.
4. **Compatibility**: The agarose medium is compatible with a wide range of crystallization conditions, including high salt concentrations and polyethylene glycols (PEGs).
5. **Low Background Scattering**: The agarose medium produces minimal background scattering, especially at low resolution, making it ideal for large unit cells and medium-to-low resolution limits.

This invention extends the capabilities of serial femtosecond crystallography to a broader range of protein complexes, including those that are difficult to express and isolate in large amounts.

## DETAILED DESCRIPTION OF THE INVENTION

### Preparation of the Agarose-Glycerol Solution

The agarose-glycerol solution is prepared by dissolving 7% ultralow-gelling-temperature agarose (Sigma–Aldrich, catalog No. A5030) in a solution of 30% glycerol and the crystallization buffer. The solution is mixed in a 15 ml centrifuge tube and submerged in a water bath filled with boiling water for 30 minutes. The agarose solution is then drawn up into a 100 µl syringe (Hamilton, Model 1710) that has been warmed by drawing up and quickly ejecting boiling water 10 to 15 times. The agarose is allowed to equilibrate to room temperature for approximately 20 minutes before the protein crystals are mixed into the agarose medium.

### Embedding of Protein Crystals

Pre-grown protein crystals are embedded into the agarose-glycerol solution using a syringe setup. A second syringe is filled with 5 µl of the highly concentrated protein crystal suspension in the crystallization mother liquor. The syringes containing agarose and the protein crystals are connected using a syringe coupler, and the solutions are mixed back and forth at least 40 times to ensure a homogeneous distribution of crystals. The final concentration of agarose in the mixture is 5.6%.

### Extrusion and Data Collection

The agarose stream containing the embedded protein crystals is extruded from a 50 µm capillary into the X-ray interaction region using the LCP injector at a flow rate of 160 nl/min. Data are collected using the CXI instrument at the Linac Coherent Light Source (LCLS) at SLAC. The diffraction patterns are processed using software such as Cheetah and CrystFEL, and the structure is solved using molecular replacement and refined using phenix.refine.

### Example 1: Phycocyanin (PC) Crystals

#### Protein Purification and Crystallization

Phycocyanin (PC) was isolated from Thermosynechococcus elongatus. The protein was obtained by disrupting a concentrated suspension of cells using a microfluidizer at 124 MPa. The resulting suspension was further purified by ultracentrifugation at 50,000 g for 1 hour, followed by concentration using Amicon Ultra-15 spin filters (Millipore, 100 kDa cutoff). PC was crystallized by free interface diffusion using a precipitant solution consisting of 1.0 M ammonium sulfate and 40 mM MES pH 6.4. Crystals of 1 to 5 µm in size formed after 1 day and were confirmed via second-order nonlinear imaging of chiral crystals.

#### Preparation of the Agarose and Embedding of Crystals

A solution of 5.6% agarose and 30% glycerol was prepared in a solution of 15% PEG 2000, 30 mM MgCl2, and 75 mM HEPES pH 7.0. The agarose was dissolved in 600 µl glycerol and 1.4 ml of the precipitant solution. The agarose solution was drawn up into a 100 µl syringe and allowed to equilibrate to room temperature. 5 µl of the PC crystal suspension was mixed into the agarose medium using a syringe coupler, and the mixture was extruded at a flow rate of 160 nl/min.

#### Data Collection and Processing

Data were collected using the CXI instrument at the LCLS. A complete data set was collected from PC crystals delivered in the agarose medium in approximately 72 minutes. The diffraction patterns were processed using Cheetah, yielding 41,100 diffraction patterns that contained 25 or more Bragg spots. 14,143 patterns were indexed and integrated using CrystFEL. The structure was solved using molecular replacement with PDB entry 4gy3 as the search model and refined using phenix.refine. The refined structure resulted in an R work of 18.7% and an R free of 25.5%.

### Conclusion

The agarose-based delivery method for serial femtosecond crystallography offers several advantages over traditional methods. It significantly reduces protein consumption, maintains crystal integrity, and produces minimal background scattering, making it suitable for both soluble and membrane protein crystals. The method is compatible with a wide range of crystallization conditions and can be used both in vacuum and at ambient pressure. This invention extends the capabilities of SFX to a broader range of protein complexes, including those that are difficult to express and isolate in large amounts, thereby advancing the field of structural biology.