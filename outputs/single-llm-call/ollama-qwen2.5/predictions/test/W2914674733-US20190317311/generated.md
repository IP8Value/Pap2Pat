- **Lens Variations**: To address lens-to-lens variations and deviations from the ideal sample-to-lens distance, we characterized each of the 96 custom-designed objectives. This involved measuring their focal lengths and NA values to ensure consistency across all units. The data was used to calibrate the system for optimal image quality.

- **Culture Plate Warping**: We compared two plate types (COC UV-Star and polystyrene Cell Star) to measure well flatness. The COC plates showed less warping, which is crucial for maintaining consistent imaging conditions. This reduced warping minimizes out-of-focus aberrations and improves overall image quality.

- **Meniscus Effects**: The liquid meniscus in the wells introduced significant aberrations due to its unpredictable shape. We developed a ray-tracing method to compensate for these effects by estimating the meniscus profile and adjusting the illumination angles accordingly. This improved the accuracy of FPM phase restoration.

- **GPU Acceleration**: GPU acceleration was critical for processing the large volume of ptychographic data. The system utilized four Nvidia Tesla K80 GPUs, which processed image segments in parallel. By pipelining the read-render-write process and using feather blending for stitching, we achieved a significant speedup in phase image reconstruction.

- **Siemens Star Targets**: Siemens star resolution phase targets were fabricated on gold-coated glass using focused ion beam (FIB) etching. The pattern was etched into the gold surface, followed by further etching to expose the glass substrate. These targets provided a standardized way to evaluate the resolution and performance of the imaging system.

- **Cell Culture**: U2OS cells expressing eGFP were cultured in DMEM with 10% FBS and antibiotics. Cells were seeded at 8,000 per well, incubated overnight, and then fixed with formaldehyde. The wells were hydrated with DPBS before imaging to ensure consistent conditions for all samples.

- **Microsphere Sample**: Polystyrene microspheres (2 μm diameter) were used to assess the flatness of the 96-well plate wells. A dilute suspension was deposited into a well, and a cover glass was added. Ptychographic imaging with local aberration recovery was used to measure the defocus distances within each well.

- **Microwell Plate Flatness**: The flatness of COC UV-Star and polystyrene Cell Star plates was measured using an absorbance microplate reader and a research-grade FPM microscope. The COC plates showed better overall flatness, which is essential for maintaining consistent imaging conditions across all wells.