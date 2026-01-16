# DESCRIPTION

## ACKNOWLEDGEMENT OF GOVERNMENT SUPPORT

This invention was made with government support under NIH fellowship grant F32MH118724 and NIH grants R01NS104925 and R01NS118424 awarded by the National Institutes of Health. The government has certain rights in the invention.

## FIELD

The present invention relates to the field of neural recording and stimulation devices, specifically to the fabrication of high-resolution, 3D-printed electrode arrays using two-photon lithography. These arrays are designed for precise and minimally invasive neural recordings and stimulation in various animal models, including small animals like zebra finches and mice.

## BACKGROUND

Neural recording and stimulation devices are essential tools in neuroscience and neuroengineering. Traditional microfabricated electrode arrays, while effective, often suffer from limitations in resolution, flexibility, and the ability to conform to complex brain geometries. These limitations can lead to suboptimal signal-to-noise ratios (SNR) and tissue damage, particularly in small animal models where the brain's size and density pose additional challenges.

Two-photon lithography (TPL) is a 3D printing method that uses femtosecond pulses of infrared light to polymerize an ultraviolet photoresist at the focal point of a high-numerical-aperture lens. This technique allows for the creation of complex polymer shapes at micron resolution, enabling the fabrication of highly customized and precise 3D structures. By integrating TPL with thin-film fabrication processes, it is possible to create 3D-printed electrode arrays with enhanced performance and minimal invasiveness.

Prior art in the field of neural electrodes includes various approaches to improve recording and stimulation capabilities. However, these methods often lack the resolution and flexibility required to address the specific needs of small animal models and complex brain regions. There is a need for a method that combines high-resolution 3D printing with robust thin-film fabrication techniques to produce customizable, high-performance neural electrodes.

## SUMMARY

The present invention provides a method for fabricating high-resolution, 3D-printed electrode arrays using two-photon lithography (TPL) and thin-film fabrication processes. The method involves the following steps:

1. **Design and Printing**: Designing 3D electrode structures using standard 3D CAD software and printing them using a TPL system. The photoresist used is a hybrid resist based on OrmoComp®, a glass-like, biocompatible member of the ORMOCER® family, with added photoinitiator, stabilizing agent, and fluorescein for in situ imaging during printing.

2. **Thin-Film Fabrication**: Depositing a sacrificial layer of chromium (Cr) on a silicon wafer, patterning the Cr layer to form the outline of the traces and electrode sites, and transferring the wafer to the 3D printer for the printing process. The wafer is then metalized via a non-directional sputter deposition process, and the Cr layer is lifted off to define the traces and electrodes.

3. **Insulation and Tip Opening**: Coating the wafer with parylene C for insulation, removing a small region of the parylene C insulation layer at the tip of each electrode to expose the underlying platinum, and using femtosecond laser milling to achieve this with high precision.

4. **Finalization and Release**: Cutting the devices from the wafer using a high pulse energy laser, bonding the flex cables to an Omnetics connector, and releasing the individual devices.

The resulting 3D-printed electrode arrays offer several advantages over traditional microfabricated electrodes, including:

- **High Resolution**: The ability to print electrodes with micron-scale features, enabling precise and minimally invasive neural recordings and stimulation.
- **Customizable Geometry**: The flexibility to design and print a wide range of electrode shapes, including biomimetic structures and porous electrodes, to optimize performance for specific applications.
- **Enhanced Performance**: Improved signal-to-noise ratios (SNR) and reduced tissue damage, particularly in small animal models.
- **Robust Fabrication**: A wafer-scale, fully compatible process with standard Si and flexible polyimide device fabrication techniques, ensuring reproducibility and scalability.

## DETAILED DESCRIPTION

### Design and Printing

The first step in the fabrication process involves designing the 3D electrode structures using standard 3D CAD software. The designs are then uploaded into the TPL system software as STL files for print voxelization. The photoresist used is a hybrid resist based on OrmoComp®, a glass-like, biocompatible member of the ORMOCER® family. The resist is modified with a photoinitiator (2,4,6-trimethyl benzoyl phosphine oxide, TPO), a stabilizing agent (3,5-Di-tertbutyl-4-hydroxytoluene, BHT), and fluorescein for in situ imaging during printing.

The TPL system uses a 780 nm Chameleon Discovery laser with a 100 fs pulse width and 80 MHz repetition rate, set to approximately 40 mW power. The laser is focused through a 20x Nikon immersion lens (NA 0.7) to initiate polymerization in the photoresist. After printing, the substrates are submerged in Ormodev developer for 12 hours to remove un-polymerized photoresist, followed by a rinse in isopropanol. The development process is followed by a 10-minute UV cure at 395 nm to increase the overall degree of crosslinking in the polymerized resist and enhance the mechanical stability of the structures.

### Thin-Film Fabrication

The thin-film fabrication process begins with the preparation of a silicon wafer. Prime grade 75 mm Si wafers with 300 nm of thermal oxide are used. A base layer of polyimide (HD MicroSystems PI2611) is spun onto the surface and cured at 350°C for 30 minutes in a nitrogen environment to a final thickness of 6 µm. An adhesion promoter (HD Microsystems VM652) is added to the edge of the wafer before polyimide spin coating.

A sacrificial layer of chromium (Cr) is then sputtered onto the polyimide surface at 3 mTorr DC. To define the metal traces of the electrode array, AZ-1512 photoresist is spun onto the wafer surface and patterned using a Süss Microtec MJB4 mask aligner with an exposure dose of 70 mJ/cm². The photolithography masks are written in-house on a direct write laser lithography system. After development, the Cr layer is etched using Transene Chromium Etchant 1020 to form the mask for the final traces and define the print locations. The photoresist is removed via sonication in acetone at 37 kHz for 5 minutes.

Before 3D printing, the patterned wafer is cleaned with oxygen plasma for 90 seconds in a March plasma etcher at a pressure of 300 mTorr and RF power of 100 W to increase print adhesion. The wafer is then transferred to the TPL system, and the surface finding and printing for each electrode in the array are automated. After printing and developing the 3D electrode structures, the devices are plasma cleaned again to increase metal adhesion to the prints. The print structures are then sputtered with titanium (15 nm) and platinum (200 nm) in an Angstrom Engineering sputter system at 3 mTorr (Ti-DC) and 10 mTorr (Pt-RF). The Cr sacrificial layer is lifted off using Transene Chromium Etchant 1020 at 60°C for 15 minutes. To ensure complete removal of all metal flakes, the wafers are transferred to multiple fresh etchant baths with agitation, followed by a thorough rinse in deionized water.

Finally, the Omnetics connector contact pads are masked with Kapton tape, and a 3 µm thick layer of parylene C is deposited over the wafer using a Labcoater 4200.

### Insulation and Tip Opening

The wafer is coated with parylene C for insulation. A small region of the parylene C insulation layer is removed at the tip of each electrode to expose the underlying platinum. This is achieved using femtosecond laser milling with a 1035 nm wavelength pulsed laser (Coherent, Monaco). The laser is coaligned with the 3D printing laser in the TPL system. The initial alignment points on the wafer are found using the 3D printing laser to avoid damage. The parylene C is then removed from the tips of all 16 electrodes in a single cut process at a 1 MHz pulsing setting. The cut raster scans a 2-3 micron thick volume, including all electrode tips. During this process, emitted ultraviolet light from the ablation process is imaged with photomultiplier tubes in a standard two-photon imaging fashion. This emitted light can be used to calibrate the ablation power. The laser ablation process takes less than one minute per array.

### Finalization and Release

To release the entire device from the wafer, the 1035 nm pulsed laser is used to cut through the polyimide and parylene layers using the programmed motion of a precision translation stage. The wafer is then placed in warm water to release the individual devices. Finally, Omnetics connectors are attached to the device pads via anisotropic conductive film (3M, ACF 7371).

### Porous Stimulating Electrodes

In addition to the standard 3D-printed electrodes, the invention also includes a method for creating porous stimulating electrodes. These devices are fabricated on silicon wafers for easy electrochemical testing. The process involves pre-established traces connected to raised 3D metal surfaces in a second metal sputtering step. Structures with pore cross-sections ranging from 40 µm² to 400 µm², solid prints, and planar electrodes lacking 3D prints for controls are printed. Sputtering at elevated pressure is used to reduce directionality, resulting in interior metalization of the porous structures.

### Electrochemical Measurements

Cyclic voltammetry (CV) and electrochemical impedance spectroscopy (EIS) data are collected using a high surface area Pt counter electrode and an Ag/AgCl reference electrode on a Gamry Reference 600 potentiostat. Measurements are conducted in phosphate-buffered saline (pH ~7.2) consisting of 0.126 M NaCl, 0.081 M Na₂HPO₄, and 0.022 M NaH₂PO₄. Before measurement, the electrolyte solution is sparged with He gas for approximately 30 minutes to remove dissolved O₂. CV curves are cycled at 50 mV/s until differences between subsequent scans are no longer observed. Additional complete device checks are performed via Open Ephys using Intan chips.

### Neural Implantation and Recordings

The 3D-printed electrode arrays are tested in zebra finches and mice to evaluate their performance in capturing relevant physiological signals from the brain. The devices are positioned precisely with a micromanipulator, holding and applying load directly behind the array with a suction pipette. In zebra finches, the devices pick up high-SNR spikes on multiple channels, demonstrating their ability to capture neuronal populations at the single-unit level. In mice, local field potentials (LFPs) are recorded in the olfactory bulb to test the devices' ability to capture neural correlates of ethologically relevant behavior. The LFP rhythms recorded from the devices faithfully recapitulate the aperiodic breathing rhythm of awake mice, demonstrating their capability to capture relevant physiological signals at high temporal resolution.

### Biomimetic Geometries and Insertion Tests

To reduce tissue insertion forces, the invention includes the development of biomimetic electrode geometries. The mosquito proboscis, with its unique tip geometry, serves as inspiration for reducing insertion force while resisting buckling. 3D-printed test structures with sharp spikes resembling the point of the mosquito needle are created. These devices can be readily inserted in songbird brains with the dura removed, and surprisingly, they can also be inserted without removing the dura, providing a faster and safer implantation process.

### Porous Stimulating Electrodes

Porous stimulating electrodes are fabricated to improve stimulation performance. The protruding surfaces of 3D stimulating electrodes provide better electrical contact between the electrode and neural tissue, with potential applications in cortical micro-ECoG recording and stimulation, as well as peripheral nerve interfacing. 3D-printed macro-pores can increase the surface area of the electrodes while maintaining the same overall displaced tissue volume, potentially leading to improved charge injection capacity and reduced fibrotic encapsulation.

### Small Animal Model Electrodes

Initial applications of the 3D-printed electrode arrays are anticipated in small animal models, where the devices can be fabricated to conform to specific spatial profiles within target brain regions. Integrated flex cables allow these devices to be mounted on micro-drives to sample multiple depths serially, extending the operational period of animal experiments and providing the unique ability to sample 3D volumes in behaving animals.

### Conclusion

The present invention provides a robust, wafer-scale method for fabricating high-resolution, 3D-printed electrode arrays using two-photon lithography and thin-film fabrication processes. These arrays offer enhanced performance, customizable geometry, and minimal invasiveness, making them ideal for a wide range of neuroscience and neuroengineering applications. The fabrication methods described herein can be adopted at many research universities and have the potential to provide new tools for researchers and clinicians in the field of neural interfacing.