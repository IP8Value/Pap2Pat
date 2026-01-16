# DESCRIPTION

## STATEMENT AS TO FEDERALLY SPONSORED RESEARCH

The invention described herein was not made in the performance of work under a federally sponsored research or development program.

## BACKGROUND

Protonic ceramics (PCs) have gained significant attention in recent years due to their potential applications in intermediate-temperature (300–600 °C) protonic ceramic energy devices (PCEDs) such as fuel cells, electrolysis cells, reversible electrochemical cells, membrane reactors, hydrogen-permeable membranes, water-permeable membranes, and solid-state ammonia synthesis cells. Traditional manufacturing techniques for PCs, such as dry pressing, screen printing, tape casting, and paste extrusion, have limitations in terms of scalability, geometric complexity, and microstructural control. These limitations hinder the practical application of PCEDs, as they cannot cost-effectively and rapidly produce sizable devices with large active areas, high surface-area-to-volume ratios, and desired performance.

Additive manufacturing (3D printing) offers a promising solution to these challenges. By integrating digital microextrusion-based 3D printing with rapid and precise laser processing (drying, sintering, cutting, and polishing), a new advanced manufacturing process called Laser 3D Printing (L3DP) has been developed. This process enables the fabrication of PCs with complex geometries, controlled microstructures, and desired crystal structures, thereby overcoming the limitations of traditional manufacturing techniques.

## SUMMARY

The present invention relates to a method and system for manufacturing protonic ceramic parts using Laser 3D Printing (L3DP). The L3DP method integrates microextrusion-based 3D printing with rapid and precise laser processing, including drying, sintering, cutting, and polishing. This integration allows for the fabrication of protonic ceramic parts with complex geometries, controlled microstructures, and desired crystal structures. The method is particularly useful for producing parts for intermediate-temperature protonic ceramic energy devices (PCEDs) such as fuel cells, electrolysis cells, and membrane reactors.

The L3DP method includes the following steps:
1. **Preparation of Printable Pastes**: Preparing pastes from ceramic precursor powders, binders, and solvents.
2. **Microextrusion-Based 3D Printing**: Depositing the pastes layer-by-layer to form the desired geometry.
3. **Rapid Laser Drying**: Using a CO2 laser to rapidly dry each printed layer to prevent shape deformation.
4. **Precise Laser Machining**: Using a picosecond YAG laser for precise cutting and polishing of the printed layers.
5. **Rapid Laser Reactive Sintering**: Using a CO2 laser to sinter the printed layers into the desired crystal structures and microstructures.
6. **Post-Treatment**: Performing additional treatments such as coating and conventional sintering to achieve the final desired properties and functions.

The L3DP method is capable of manufacturing a variety of protonic ceramic parts, including pellets, cylinders, cones, rings, straight tubes, lobed tubes, microchannel membranes, and half cells. The method ensures high accuracy, rapid processing, and the ability to produce parts with complex geometries and controlled microstructures.

## DETAILED DESCRIPTION

### Preparation of Printable Pastes

The first step in the L3DP method involves preparing printable pastes from ceramic precursor powders, binders, and solvents. For example, to prepare a paste for a 40 wt% BaCe0.7Zr0.1Y0.1Yb0.1O3-δ (BCZYYb) + 60 wt% NiO hydrogen electrode material, the following procedure is used:

1. **Ball Milling**: Stoichiometric amounts of carbonate and oxide precursors (BaCO3, CeO2, Y2O3, ZrO2, Yb2O3, and NiO) are ball-milled for 48 hours with isopropanol as the grinding solvent and 3 mm YSZ as the grinding media.
2. **Mixing**: The dry ball-milled powder is mixed with 15 wt% deionized water, 0.7 wt% dispersant (Darven 821A), and 1–3 wt% (based on water amount) binder (HPMC) using a vacuum mixer for 30 minutes.
3. **Paste Adjustment**: The amounts of water, dispersant, and binder can be adjusted according to the specific material composition to achieve the desired viscosity for 3D printing.

### Microextrusion-Based 3D Printing

The prepared pastes are fed into designated plastic syringe reservoirs and driven through a microextruder with a needle-type nozzle (0.5 mm diameter) using compressed air. The distance between the extruder nozzle and the platform substrate is typically equal to the thickness of the wet layer (around 450 μm). The paste extrusion flow rate is usually 0.3 mL/min, and the stage moving speed is 15 mm/s, resulting in a filament width of approximately 740 μm. The tool paths for printing each layer are adjusted to match the desired geometry of the part. For example, a tubular part is printed using a spiral line path, while a simple square thin film is printed using a line-by-line bi-directional path.

### Rapid Laser Drying

To prevent shape deformation and speed up the 3D printing process, a CO2 laser is used to rapidly dry each printed layer immediately after deposition. The laser beam is defocused by 15 mm to increase its spot size to approximately 1 mm and lower the laser energy density. The optimized laser operation parameters of 10 W power and a scan rate of 15 mm/s efficiently dry the green layers without noticeable shrinkage or reactions. The use of a Galvano scanner allows for higher laser power and faster scan rates, further reducing the drying time.

### Precise Laser Machining

The picosecond YAG laser (ps-laser) is used for precise cutting and polishing of the green layers. For laser cutting, the ps-laser is focused on a spot with a size of 18 μm using a 5× lens (NA = 0.13). The repetition rate, laser energy, and laser scan rate are set to 10 kHz, 150 μJ per pulse, and 5 mm/s, respectively, allowing for a cutting depth of 150 μm. This results in very accurate cutting, enabling the creation of microchannels and complex geometries. For laser polishing, the laser operation parameters are set to a repetition rate of 1 kHz, laser energy of 114.4 μJ per pulse, and a laser scan rate of 50 mm/s, achieving a smooth finishing surface for subsequent processing.

### Rapid Laser Reactive Sintering

Rapid laser reactive sintering (RLRS) is used to sinter the thoroughly dried green layers into the desired crystal structures and microstructures. The CO2 laser, fixed on the Z-axis, is applied for the RLRS of the PC parts. The laser beam, defocused by 20 mm, forms a line-shaped spot with a size of around 8 mm, providing homogeneous and moderate laser energy density. The laser power and moving speed are set to 20 W and 0.1 mm/s, respectively. The microstructures of the sintered layers are controlled by optimizing the laser parameters such as laser power, moving speed, defocus distance, and hatching pattern spacing.

### Post-Treatment

The L3DP method can produce both green and sintered protonic ceramic parts. For green parts, additional post-treatments such as coating and conventional sintering are required to achieve the desired microstructures, properties, and functions. For example, the L3DP-derived green anode tubes can be prefired at 1050 °C for 12 hours to vaporize the paste solvent and partially burn the binders or dispersants. After prefiring, an electrolyte precursor slurry is dip-coated on the outside surface of the tubes, and the green half-cells are dried in air for two days. Finally, the co-firing of the green half-cells is carried out at 1450 °C for 12 hours with a ramp rate of 1 °C/min. For single-component PC green parts, conventional firing/sintering at high temperatures is performed in a box furnace to obtain the sintered PC parts.

## EXAMPLES

### Example 1—Integrated Multi-Laser 3D Manufacturing System

An integrated multi-laser 3D manufacturing system was developed for the fabrication of protonic ceramic parts. The system consists of X-Y and Z stages, microextruders, a CO2 laser, a picosecond YAG laser, and a Galvano scanner. The CO2 laser is used for rapid drying and sintering, while the picosecond YAG laser is used for precise cutting and polishing. The system allows for the advanced manufacturing of green or sintered ceramic parts by combining 3D printing, laser processing, and in-situ consolidation.

### Example 2—CO2 Laser Sintering of Sol-Gel Deposition

A CO2 laser was used to sinter a sol-gel deposited layer of BCZYYb on a fused silica substrate. The green layer was thoroughly dried using the CO2 laser, and then the RLRS process was applied to sinter the layer into the desired crystal structure and microstructure. The resulting sintered layer was fully densified and exhibited the correct crystal phase, as confirmed by XRD analysis.

### Example 3—CO2 Laser Melting of Ceramic Paste

A CO2 laser was used to melt a ceramic paste containing 40 wt% BCZYYb + 60 wt% NiO. The paste was 3D printed onto a fused silica substrate, and the CO2 laser was applied to sinter the green layer into a fully densified electrolyte film. The resulting film was well-bonded to the substrate and exhibited the desired crystal structure and microstructure, as confirmed by SEM and XRD analysis.

### Example 4—Thick Er-Doped Silica Films Sintered by CO2 Laser for Scintillation Applications

Er-doped silica films were prepared by 3D printing and sintered using a CO2 laser. The films were 3D printed onto a fused silica substrate and then subjected to the RLRS process. The resulting sintered films were fully densified and exhibited the desired crystal structure and microstructure, making them suitable for scintillation applications.

### Example 5—High-Resolution Laser 3D Printing of Transparent Fused Silica Glass

High-resolution laser 3D printing was used to fabricate transparent fused silica glass parts. The process involved 3D printing a paste of fused silica particles and then using a CO2 laser to sinter the green layers into fully densified, transparent parts. The resulting parts exhibited high transparency and the desired crystal structure and microstructure, as confirmed by optical and SEM analysis.

### Example 6—Laser-Assisted Embedding of all-Glass Optical Fiber Sensors into Bulk Ceramics for High-Temperature Applications

All-glass optical fiber sensors were embedded into bulk ceramics using a laser-assisted embedding process. The process involved 3D printing a ceramic paste and then using a picosecond YAG laser to precisely cut channels for the optical fibers. The fibers were then embedded into the channels, and the parts were sintered using a CO2 laser. The resulting parts exhibited excellent bonding between the ceramic matrix and the optical fibers, making them suitable for high-temperature applications.

### Example 7—SiC Applications

Silicon carbide (SiC) parts were fabricated using the L3DP method. The process involved 3D printing a paste of SiC particles and then using a CO2 laser to sinter the green layers into fully densified parts. The resulting parts exhibited the desired crystal structure and microstructure, as confirmed by XRD and SEM analysis, making them suitable for high-temperature and wear-resistant applications.

### Example 8—3D Printing of all-Glass Fiber-Optic Pressure Sensor for High-Temperature Applications

An all-glass fiber-optic pressure sensor was 3D printed using the L3DP method. The process involved 3D printing a paste of glass particles and then using a picosecond YAG laser to precisely cut channels for the optical fibers. The fibers were embedded into the channels, and the parts were sintered using a CO2 laser. The resulting sensor exhibited excellent sensitivity and stability, making it suitable for high-temperature pressure sensing applications.

### Example 9—Laser-Assisted Embedding of all-Glass Optical Fiber Sensors into Bulk Ceramics for High-Temperature Applications

All-glass optical fiber sensors were embedded into bulk ceramics using a laser-assisted embedding process. The process involved 3D printing a ceramic paste and then using a picosecond YAG laser to precisely cut channels for the optical fibers. The fibers were embedded into the channels, and the parts were sintered using a CO2 laser. The resulting parts exhibited excellent bonding between the ceramic matrix and the optical fibers, making them suitable for high-temperature applications.

This detailed description and the provided examples illustrate the versatility and capabilities of the L3DP method for manufacturing protonic ceramic parts with complex geometries, controlled microstructures, and desired crystal structures. The method offers significant advantages over traditional manufacturing techniques, enabling the rapid and cost-effective production of high-performance PCEDs.