# DESCRIPTION

## CROSS REFERENCE

This application claims the benefit of U.S. Provisional Application No. 63/XXXXXX, filed on [Date], which is hereby incorporated by reference in its entirety.

## FIELD OF THE INVENTION

The present invention relates to the field of radiotherapy, specifically to the creation of intensity-modulated electron fields for therapeutic purposes. More particularly, the invention pertains to a passive radiotherapy intensity modulator for electrons (PRIME) capable of delivering highly conformal, intensity-modulated electron therapy.

## BACKGROUND AND DESCRIPTION OF PRIOR ART

Intensity modulation (IM) is a critical component in delivering highly conformal and homogeneous dose distributions tailored to individual patients. While IM technologies are well-established for x-ray therapy using multileaf collimators (MLCs) or metal compensators, and for proton therapy using scanned spot beams, intensity modulation for electron therapy remains underdeveloped. This is primarily due to the challenges associated with the air gap in x-ray MLCs and the need for helium in the treatment head to reduce multiple Coulomb scattering (MCS) in scanned electron beams.

Several approaches have been explored to achieve electron intensity modulation, including the use of multileaf collimators (MLCs) and scanned electron beams. However, these methods have not been widely adopted due to high costs, integration difficulties, and other practical limitations. For instance, the air gap in x-ray MLCs prevents adequate conformity, and the use of helium in the treatment head for scanned electron beams is complex and expensive.

The present invention addresses these limitations by introducing a passive radiotherapy intensity modulator for electrons (PRIME). PRIME is a low-cost, readily available technology that can be integrated into existing electron therapy systems. It consists of a collection of small area island blocks and island apertures strategically positioned to deliver a desired intensity-modulated electron fluence distribution. This approach leverages the principles of multiple Coulomb scattering (MCS) to achieve the desired intensity patterns, making it a practical solution for various clinical applications.

## SUMMARY OF THE INVENTION

The present invention provides a passive radiotherapy intensity modulator for electrons (PRIME) capable of creating intensity-modulated electron fields. PRIME comprises a collection of small area island blocks and island apertures arranged in a plane perpendicular to the central beam axis. The locations and areas of these island blocks and apertures are selected to deliver a desired intensity-modulated electron fluence distribution.

The invention is particularly useful for three types of electron conformal therapy (ECT): segmented-field ECT, bolus ECT, and modulated electron radiation therapy (MERT). PRIME can be used to improve dose homogeneity in bolus ECT, reduce dose heterogeneity in segmented-field ECT, and enable full-range intensity modulation in MERT.

Key features of the invention include:
1. **Island Blocks**: Small area blocks of varying diameters located on a hexagonal grid inside the aperture of a custom electron collimating insert. These blocks remove a fraction of the electrons incident on their entry surfaces, thereby modulating the electron fluence.
2. **Island Apertures**: Small area apertures of varying diameters located on a hexagonal grid inside the collimating insert. These apertures allow a fraction of the electrons to pass through, further modulating the electron fluence.
3. **Combination of Island Blocks and Apertures**: A hybrid approach that combines both island blocks and apertures to achieve a wide range of intensity modulation, from 0% to 100%.

The invention also includes methods for constructing and optimizing the intensity modulators, as well as techniques for integrating them into existing treatment planning systems. The use of PRIME can significantly enhance the precision and effectiveness of electron therapy, making it a valuable tool in the treatment of various cancers and other medical conditions.

## DETAILED DESCRIPTION OF THE INVENTION

### Concept of Passive Intensity Modulators

The passive radiotherapy intensity modulator for electrons (PRIME) is a device designed to deliver an electron fluence distribution that varies with position in the plane perpendicular to the central beam axis. PRIME consists of a collection of small area island blocks and island apertures arranged in a plane inside or just upstream of the aperture of a collimating insert. The locations and areas of these island blocks and apertures are selected to deliver a desired intensity-modulated electron fluence distribution.

#### Island Blocks

Island blocks are small, high-density blocks of varying diameters located on a hexagonal grid inside the aperture of a custom electron collimating insert. These blocks remove a fraction of the electrons incident on their entry surfaces, thereby modulating the electron fluence. The thickness of the island blocks should be sufficient to stop primary electrons.

#### Island Apertures

Island apertures are small, high-density apertures of varying diameters located on a hexagonal grid inside the collimating insert. These apertures allow a fraction of the electrons to pass through, further modulating the electron fluence. The thickness of the collimating insert containing the island apertures should also be sufficient to stop primary electrons.

#### Combination of Island Blocks and Apertures

A hybrid approach that combines both island blocks and apertures can achieve a wide range of intensity modulation, from 0% to 100%. This combination is particularly useful for modulated electron radiation therapy (MERT), where full-range intensity modulation is required.

### Intensity Modulation 50%–100%

The island blocks remove most, ideally all, electrons incident on their entry surfaces from the beam. The relative electron fluence (intensity) for the central region of the beam can be calculated using the following formula:

\[ I_{\text{island blocks}}(d, r) = I_0 \left[ 1 - \left( \frac{\pi}{2\sqrt{3}} \right) \left( \frac{d}{r} \right)^2 \right] \]

where \( I_0 \) is the intensity with no island blocks, \( d \) is the diameter of the island block, and \( r \) is the separation between the island blocks. This formula allows an estimate of the block diameter at each point on a hexagonal grid to be calculated based on the desired underlying intensity.

### Intensity Modulation 0%–50%

For intensity modulation in the range of 0% to 50%, small island apertures in the collimating insert are used. The relative electron fluence (intensity) for the central region of the beam can be calculated using the following formula:

\[ I_{\text{island apertures}}(d, r) = I_0 \left( \frac{\pi}{2\sqrt{3}} \right) \left( \frac{d}{r} \right)^2 \]

where \( I_0 \) is the intensity with no island apertures, \( d \) is the diameter of the island aperture, and \( r \) is the separation between the island apertures. This formula allows an estimate of the aperture diameter at each point on a hexagonal grid to be calculated based on the desired underlying intensity.

### Range of Island Block Parameters (D, R) for Range of Intensity Reduction Factors

The useful range of sizes (cross-sectional area) for island blocks and island apertures is determined by the desired intensity reduction factor (IRF). For example, for hexagonally packed circular island blocks of diameter \( d \) and separation \( r \), the reduced local intensity is given by:

\[ I_{\text{island blocks}}(d, r) = I_0 \left[ 1 - \left( \frac{\pi}{2\sqrt{3}} \right) \left( \frac{d}{r} \right)^2 \right] \]

Similarly, for hexagonally packed circular island apertures of diameter \( d \) and separation \( r \), the local intensity is given by:

\[ I_{\text{island apertures}}(d, r) = I_0 \left( \frac{\pi}{2\sqrt{3}} \right) \left( \frac{d}{r} \right)^2 \]

These formulas allow an estimate of the block or aperture diameter at each point on a hexagonal grid to be calculated based on the desired underlying intensity.

### Construction of Intensity Modulators

Methods for the construction of intensity modulators are currently under development. The intensity modulator will consist of small cross-section island blocks and island apertures strategically located in the collimating space occupied by the custom electron insert. The central axes of both the island blocks and island apertures should follow the diverging rays emanating from the virtual source of the electron beam. The shape of the cross-section of the island blocks and island apertures should be circular, forming right oblique cylinders whose axes coincide with diverging ray lines.

#### Construction Constraints

Optimally, the intensity modulator will be small cross-section island blocks and island apertures strategically located in the collimating space occupied also by the custom electron insert. Island blocks could be cylinders fixed in space in the aperture of the insert collimator, achieved by embedding them in a low-density foam that could be accounted for in the fluence calculation, but have little impact. Island apertures could be circular holes in the electron insert's collimating portion. Ideally, the central axes of both the island blocks and island apertures should follow the diverging rays emanating from the virtual source of the electron beam. Also, the shape of the cross-section of the island blocks and island apertures should be circular, forming right oblique cylinders whose axes coincide with diverging ray lines.

The island apertures will be the same thickness (g cm−2) as the custom electron insert, usually sufficient to stop electrons from the highest beam energy (20 MeV). It is recommended that the island blocks be the same thickness so as to be able to be used at all electron energies. The collimating material should be a high-density metal, possibly the same or similar material as the custom electron inserts. Presently, most inserts are fabricated using low melting point lead alloy (Cerrrobend) or copper. Tungsten alloy is another potential material for the island blocks, its being denser, harder, and less toxic than lead, all advantages for a block material.

### Island Block Intensity Modulator Proof of Principle

Initial proof of principle compared measurement with calculation for a prototype IM. The prototype was constructed by inserting lead wire (0.2 cm diameter × 2.0 cm thick) into a 2.0 cm thick piece of Styrofoam on a hexagonal grid with r = 0.5 cm, corresponding to an IRF of 0.85. The block matrix, which consisted of five rows of 6–7 pins with the central pin located at the central axis, was abutted to the upstream side of the final trimmer. Relative dose measurements were made along x = 0 (in-plane) with a 16 MeV electron beam and 10 × 10 cm² field at 2.0 cm depth (100 cm SSD) on an Elekta Infinity accelerator. These measurements were made using a p-type electron dosimetry diode detector (EFD3G, #300–605) with an active volume diameter of 0.2 cm and thickness of 0.006 cm (IBA Dosimetry, Bartlett, TN, USA). The diode was connected to the scanning main control unit of the RFA-200 Water Phantom 2D scanning tank using OmniPro scanning software (IBA Dosimetry, Bartlett, TN, USA). For comparison, the off-axis dose profile was also calculated using the PBA for identical conditions.

### Patient Example

Kudchadker et al. showed how IM improved planning target volume (PTV) dose homogeneity for bolus ECT of a head and neck patient (right buccal mucosa). We used the reported intensity distribution for that patient to design an intensity modulator, which closely provided the desired IM dose distribution at a depth of 2 cm in water. Details of the design process for the intensity modulator remain under investigation and will be reported later, but preliminary results for this patient are shown.

### Example 1

**Application in Bolus Electron Conformal Therapy (ECT)**

Bolus electron conformal therapy (ECT) is a technique used to deliver conformal electron therapy to superficial tumors. One of the challenges in bolus ECT is the creation of hot and cold spots due to the irregular surface of the bolus. PRIME can be used to modulate the electron fluence, thereby improving dose homogeneity in the treatment target volume (PTV).

**Design Process**

1. **Determine Desired Intensity Pattern**: Using treatment planning software, the desired intensity pattern is determined based on the patient's anatomy and tumor location.
2. **Optimize Island Block and Aperture Sizes**: The sizes and positions of the island blocks and apertures are optimized using an inverse planning algorithm to achieve the desired intensity pattern.
3. **Construct Intensity Modulator**: The optimized design is used to construct the intensity modulator, which is then placed in the collimating insert of the electron beam.
4. **Verify Dose Distribution**: The dose distribution is verified using measurements and dose calculations to ensure that the desired intensity pattern is achieved.

### Example 2

**Application in Segmented-Field Electron Conformal Therapy**

Segmented-field electron conformal therapy involves the use of multiple electron fields of differing energies to treat a target volume. One of the challenges in this approach is the creation of dose heterogeneity in the abutting regions of the fields. PRIME can be used to modulate the electron fluence at the edges of the fields, thereby reducing dose heterogeneity.

**Design Process**

1. **Determine Desired Intensity Pattern**: Using treatment planning software, the desired intensity pattern is determined based on the patient's anatomy and the abutting regions of the fields.
2. **Optimize Island Block and Aperture Sizes**: The sizes and positions of the island blocks and apertures are optimized using an inverse planning algorithm to achieve the desired intensity pattern.
3. **Construct Intensity Modulator**: The optimized design is used to construct the intensity modulator, which is then placed in the collimating insert of the electron beam.
4. **Verify Dose Distribution**: The dose distribution is verified using measurements and dose calculations to ensure that the desired intensity pattern is achieved.

### Example 3

**Application in Modulated Electron Radiation Therapy (MERT)**

Modulated electron radiation therapy (MERT) involves the use of multiple intensity-modulated electron fields of varying angles and energies to treat a target volume. PRIME can be used to achieve full-range intensity modulation, from 0% to 100%, enabling precise dose delivery to the target volume while minimizing dose to surrounding healthy tissues.

**Design Process**

1. **Determine Desired Intensity Pattern**: Using treatment planning software, the desired intensity pattern is determined based on the patient's anatomy and the angles and energies of the fields.
2. **Optimize Island Block and Aperture Sizes**: The sizes and positions of the island blocks and apertures are optimized using an inverse planning algorithm to achieve the desired intensity pattern.
3. **Construct Intensity Modulator**: The optimized design is used to construct the intensity modulator, which is then placed in the collimating insert of the electron beam.
4. **Verify Dose Distribution**: The dose distribution is verified using measurements and dose calculations to ensure that the desired intensity pattern is achieved.

## CONCLUSION

The passive radiotherapy intensity modulator for electrons (PRIME) represents a significant advancement in the field of electron therapy. By providing a practical and low-cost solution for intensity modulation, PRIME can enhance the precision and effectiveness of electron therapy, making it a valuable tool in the treatment of various cancers and other medical conditions. The invention is particularly useful for bolus ECT, segmented-field ECT, and MERT, and offers numerous opportunities for future research and development.