# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of photonic integrated circuits (PICs) and, more specifically, to a novel electro-optic modulator that combines the high refractive index of silicon (Si) with the superior electro-optic properties of lithium niobate (LN). The invention aims to address the limitations of existing modulators by providing a compact, high-efficiency, and high-bandwidth solution suitable for complex PICs, particularly in applications such as microwave photonics and optical neural networks.

## BACKGROUND ART

Photonic integrated circuits (PICs) have revolutionized various fields, including telecommunications, computing, and sensing, by integrating multiple optical components onto a single chip. Key components in these circuits include electro-optic modulators, which convert electrical signals into optical signals. Traditional materials like silicon-on-insulator (SOI) have been widely used for fabricating modulators, but they suffer from limitations such as low modulation bandwidth, nonlinearity, and high loss. Lithium niobate (LN), known for its strong Pockels effect, has emerged as a promising material for high-performance modulators. However, bulk LN modulators have large footprints and low modulation efficiency, making them unsuitable for complex PICs.

Recent advancements in thin-film LN technology have enabled the realization of compact modulators. Various configurations, such as LN ridge and hybrid waveguides, have been explored. For instance, the LN-SOI configuration, where LN is deposited on an SOI wafer, has achieved modulation efficiencies of 6.7 V·cm and bandwidths up to 106 GHz. Similarly, the SOI-LN configuration, where silicon or silicon nitride waveguides are fabricated on a thin-film LN wafer, has demonstrated modulation efficiencies of 2.1 V·cm. Despite these improvements, the bend radius of the SiN_x-LN waveguide can reach 300 μm, limiting the integration density of PICs.

To overcome these challenges, there is a need for an ultra-compact and high-performance electro-optic modulator that can support tight bend radii and minimal waveguide separation while maintaining high modulation efficiency and bandwidth. The present invention addresses this need by proposing a novel modulator design that leverages the high refractive index of silicon and the superior electro-optic properties of lithium niobate.

## SUMMARY OF THE INVENTION

The present invention provides a novel electro-optic modulator designed for use in photonic integrated circuits (PICs). The modulator is configured on a monocrystalline silicon-lithium niobate (Si-LN) wafer and includes light-routing waveguides and light-modulating waveguides. The light is primarily confined to the silicon layer for routing and to the lithium niobate layer for modulation, enabling ultra-compact and high-performance operation.

The modulator comprises:
1. **Light-Routing Waveguides**: These waveguides are designed to route light with minimal loss and tight bends. They consist of silicon with dimensions optimized to achieve a small mode area and high effective refractive index, allowing for a bend radius as small as 10 μm.
2. **Light-Modulating Waveguides**: These waveguides are designed to modulate light efficiently. They consist of silicon and lithium niobate with dimensions optimized to maximize optical confinement in the LN layer, achieving a modulation efficiency of 1.76 V·cm and a modulation bandwidth exceeding 350 GHz.
3. **Mode Transition Tapers**: Bilevel tapers are used to transition the optical mode between the light-routing and light-modulating waveguides, ensuring efficient mode conversion with minimal loss.
4. **Multimode Interference (MMI) Couplers**: These couplers split and combine light in the Mach-Zehnder interferometer configuration, facilitating the operation of the modulator.
5. **Coplanar Waveguide (CPW) Electrodes**: These electrodes are designed to achieve velocity matching and impedance matching, enabling a large modulation bandwidth.

The invention further includes methods for fabricating the modulator, which involve wafer bonding, silicon patterning, silicon etching, and metal deposition. The proposed modulator supports a bend radius of 10 μm and edge-to-edge waveguide separation of 0.7 μm, making it highly suitable for ultra-compact and large-scale PICs.

## DETAILED DESCRIPTION OF THE INVENTION

### Structure and Design

#### Light-Routing Waveguides

The light-routing waveguides are designed to route light with minimal loss and tight bends. They consist of silicon with a width of 600 nm and a height of 220 nm, which are optimized to achieve a small mode area and high effective refractive index. The effective refractive index of the light-routing waveguide is 2.6, and the mode area is 0.16 μm². This design allows for a bend radius of 10 μm with a loss of around 0.01 dB per 90° bend, comparable to that of a standard Si waveguide.

#### Light-Modulating Waveguides

The light-modulating waveguides are designed to modulate light efficiently. They consist of silicon with a width of 480 nm and a height of 90 nm, and a thin-film LN layer with a thickness of 700 nm. The effective refractive index of the light-modulating waveguide is 2.02, and the mode area is 0.76 μm². The optical mode is primarily confined to the LN layer, achieving a modulation efficiency of 1.76 V·cm and a modulation bandwidth exceeding 350 GHz.

#### Mode Transition Tapers

Bilevel tapers are used to transition the optical mode between the light-routing and light-modulating waveguides. The lower layer of the taper has a height of 90 nm and a width that transitions from 600 nm to 480 nm over a length of 44.2 μm. The upper layer of the taper has a height of 130 nm and a width that transitions from 600 nm to 120 nm over a length of 102.2 μm. The coupling efficiency of the bilevel taper is approximately 98.6%, with minimal loss attributed to mode mismatch at the tip of the upper taper.

#### Multimode Interference (MMI) Couplers

MMI couplers are used to split and combine light in the Mach-Zehnder interferometer configuration. The MMI section has a width of 2.7 μm, a length of 6.6 μm, and an edge-to-edge separation of 0.7 μm to avoid evanescent coupling between the two output modes. The insertion loss of the MMI is approximately 0.06 dB.

#### Coplanar Waveguide (CPW) Electrodes

CPW electrodes are designed to achieve velocity matching and impedance matching, enabling a large modulation bandwidth. The CPW structure consists of a signal electrode and two ground electrodes in a ground-signal-ground (GSG) form. The gap between the signal electrode and the ground electrode is 4.5 μm, and the height and width of the electrodes are optimized to achieve a microwave effective index of 2.4 to 2.75 and an optical group index of 2.41. The microwave loss ranges from 5 dB/cm to 25 dB/cm, and the characterized impedance ranges from 50 to 57 ohms.

### Performance Evaluation

#### Bend Radius and Waveguide Separation

The light-routing waveguide supports a bend radius of 10 μm with minimal loss, much smaller than those of LN-ridge waveguides, SiN_x-LN waveguides, and light-modulating waveguides. The edge-to-edge waveguide separation is 0.7 μm, with a beat length of approximately 1 cm, indicating weak coupling and low crosstalk between adjacent waveguides.

#### Modulation Efficiency and Bandwidth

The modulation efficiency of the proposed modulator is 1.76 V·cm, and the modulation bandwidth exceeds 350 GHz. The modulation efficiency and bandwidth are influenced by the thickness of the LN layer and the dimensions of the electrodes. For an interaction length of 5 mm, the modulation bandwidth is maximized with a LN thickness of 700 nm, achieving a bandwidth of 350 GHz. The modulation efficiency can be further improved by reducing the interaction length, with a V_π value of approximately 1.6 V for an interaction length of 11 mm.

### Fabrication Process

The fabrication process for the proposed modulator involves several steps:
1. **Wafer Bonding**: Monocrystalline silicon is bonded to a thin-film LN wafer.
2. **Silicon Patterning**: Silicon is patterned to form the light-routing and light-modulating waveguides.
3. **Silicon Etching**: Silicon is etched to create the bilevel tapers and other necessary structures.
4. **Metal Deposition**: Metal electrodes are deposited to form the CPW structure.

### Conclusion

The present invention provides a novel electro-optic modulator that combines the high refractive index of silicon with the superior electro-optic properties of lithium niobate. The modulator is designed to be ultra-compact, with a bend radius of 10 μm and edge-to-edge waveguide separation of 0.7 μm, and high-performance, with a modulation efficiency of 1.76 V·cm and a modulation bandwidth exceeding 350 GHz. The robust and compatible fabrication process ensures that the modulator can be integrated into complex PICs, making it highly suitable for applications in microwave photonics and optical neural networks.