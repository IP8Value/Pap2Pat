Here is the complete patent application following the provided outline:

---

# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates to the field of photonic integrated circuits (PICs), particularly to electro-optic modulators fabricated on heterogeneous silicon-lithium niobate (Si-LN) platforms. More specifically, the invention pertains to an ultra-compact, high-efficiency, and large-bandwidth electro-optic modulator comprising distinct light-routing and light-modulating waveguides optimized for transverse-electric (TE) polarized single-mode operation. The modulator is designed to achieve superior performance metrics, including a modulation efficiency of 1.76 V·cm, a modulation bandwidth exceeding 350 GHz, a bend radius as small as 10 μm, and an edge-to-edge waveguide separation of 0.7 μm. The invention is particularly suited for applications in large-scale PICs, such as microwave photonics and optical neural networks, where dense integration of high-performance modulators is critical.  

## BACKGROUND ART  

Electro-optic modulators are fundamental components in photonic integrated circuits, enabling the conversion of electrical signals into optical signals. Conventional modulators based on silicon-on-insulator (SOI) platforms suffer from limitations such as low modulation bandwidth, nonlinearity, and high optical loss. Lithium niobate (LN) has emerged as a promising alternative due to its strong Pockels effect, but bulk LN modulators exhibit large footprints and low modulation efficiency due to poor optical mode confinement.  

Recent advancements in thin-film LN technology have enabled the development of modulators with improved performance. For instance, ridge LN waveguides and hybrid LN-SOI or SOI-LN configurations have been demonstrated. However, these approaches often involve complex fabrication processes, such as piecewise LN bonding or etching, and suffer from trade-offs between modulation efficiency, bandwidth, and integration density. For example, SiNx-LN modulators exhibit a large bend radius (~300 μm), limiting their suitability for compact PICs.  

There remains an unmet need for an electro-optic modulator that simultaneously achieves high modulation efficiency, large bandwidth, and ultra-compact dimensions while leveraging robust fabrication processes compatible with large-scale integration.  

## SUMMARY OF THE INVENTION  

The present invention addresses the aforementioned limitations by providing a novel electro-optic modulator architecture on a heterogeneous Si-LN platform. The modulator comprises:  

1. **Light-Routing Waveguides**: Optimized for ultra-compact routing, these waveguides feature silicon dimensions of 600 nm width and 220 nm height, enabling a bend radius of 10 μm and an edge-to-edge separation of 0.7 μm with minimal crosstalk. The optical mode is predominantly confined in the silicon layer (76% confinement), yielding a high effective refractive index (neff = 2.6) and a small mode area (0.16 μm²).  

2. **Light-Modulating Waveguides**: Designed for high-efficiency modulation, these waveguides employ silicon dimensions of 480 nm width and 90 nm height, ensuring strong optical confinement in the LN layer (83% confinement). The waveguides are coupled to coplanar waveguide (CPW) electrodes optimized for velocity and impedance matching, achieving a modulation efficiency of 1.76 V·cm and a bandwidth exceeding 350 GHz.  

3. **Bilevel Mode Transition Tapers**: These tapers facilitate low-loss (98.6% efficiency) mode conversion between the light-routing and light-modulating waveguides, enabling seamless integration of routing and modulating functionalities.  

4. **Monolithic Si-LN Wafer Fabrication**: The modulator is fabricated on a monocrystalline Si-LN wafer via wafer bonding, eliminating the need for LN etching or piecewise bonding. The design is compatible with standard lithography and deposition processes, ensuring scalability for large-scale PICs.  

The invention achieves unprecedented performance metrics, including a 30-fold reduction in bend radius compared to prior art, while maintaining compatibility with existing fabrication technologies.  

## DETAILED DESCRIPTION OF THE INVENTION  

### Structural Configuration  

The electro-optic modulator of the present invention is implemented as a Mach-Zehnder interferometer (MZI) comprising the following key components:  

1. **Light-Routing Waveguides**:  
   - Fabricated with silicon dimensions of 600 nm width and 220 nm height on an x-cut LN wafer (700 nm thickness).  
   - Designed to support TE-polarized single-mode operation with a mode area of 0.16 μm² and an effective refractive index of 2.6.  
   - Enable ultra-compact routing with a bend radius of 10 μm (0.01 dB/90° loss) and edge-to-edge separation of 0.7 μm (beat length ~1 cm).  

2. **Light-Modulating Waveguides**:  
   - Feature silicon dimensions of 480 nm width and 90 nm height to maximize LN mode confinement (83%) while minimizing metal-induced loss (~0.6 dB/cm).  
   - Integrated with CPW electrodes in a ground-signal-ground (GSG) configuration, with a signal electrode width of 7 μm and a gap of 4.5 μm.  
   - Optimized for high modulation efficiency (Vπ·L = 1.76 V·cm) and large bandwidth (>350 GHz for 5 mm interaction length).  

3. **Mode Transition Tapers**:  
   - Implemented as bilevel tapers with lower (90 nm height) and upper (130 nm height) sections, enabling 98.6% coupling efficiency over a total length of 146.4 μm.  
   - The taper design minimizes mode mismatch losses and is compatible with standard etching processes.  

4. **Multimode Interference (MMI) Couplers**:  
   - Configured with a width of 2.7 μm, length of 6.6 μm, and output separation of 0.7 μm to avoid evanescent coupling.  
   - Exhibit an insertion loss of 0.06 dB, ensuring high splitting efficiency.  

### Electro-Optic Modulation Mechanism  

The modulator leverages the Pockels effect in LN, where an applied electric field induces a refractive index change via the electro-optic tensor. Key aspects include:  

- **DC Field Analysis**: A 1 V applied voltage generates an in-plane electric field (Ex, Ez) aligned with the extraordinary axis of LN, maximizing the r33 coefficient (30.8 pm/V).  
- **Effective Index Perturbation**: The refractive index change (Δneff) is calculated using mode solver simulations, yielding a linear response with applied voltage (Figure 6(d)).  
- **Push-Pull Operation**: The MZI arms are driven in opposite phases, halving the half-wave voltage (Vπ = 1.6 V for 11 mm interaction length).  

### Microwave-Optical Co-Design  

The CPW electrodes are co-optimized with the optical waveguides to achieve:  

- **Velocity Matching**: The microwave effective index (nm = 2.4–2.75) closely matches the optical group index (ng = 2.41), minimizing phase mismatch.  
- **Impedance Matching**: The characteristic impedance (Z0 = 50–57 Ω) aligns with generator/load impedances, reducing RF reflections.  
- **Low Microwave Loss**: The electrode dimensions (400 nm height, 7 μm width) limit the microwave loss to 5–25 dB/cm at 100 GHz.  

### Fabrication Tolerance Analysis  

The invention accounts for process variations:  

- **Etch Depth Sensitivity**: The bilevel taper maintains >98% efficiency across etch depths of 70–110 nm (Figure 9).  
- **Bandwidth Robustness**: The modulation bandwidth remains >350 GHz for waveguide widths of 450–550 nm (Figure 10).  

### Performance Benchmarks  

Compared to prior art (ridge LN, LN-Si, SiNx-LN), the invention demonstrates:  

- **30× Smaller Bend Radius**: 10 μm vs. 300 μm for SiNx-LN.  
- **Highest Bandwidth**: >350 GHz vs. 106 GHz for LN-Si.  
- **Lowest Vπ·L**: 1.76 V·cm vs. 6.7 V·cm for LN-Si.  

The modulator is fabricated via wafer bonding, silicon patterning/etching, and metal deposition, ensuring compatibility with large-scale PIC production.  

---  

This patent application provides a comprehensive description of the invention, adhering to formal patent language and the specified outline. Each section is elaborated with technical details to ensure clarity and enforceability.