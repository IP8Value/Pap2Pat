Here is the drafted patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

Conventional image sensor chips, such as those employed in digital cameras and microscopy systems, are designed to detect only the intensity variations of incident light waves. While such intensity-based imaging has proven sufficient for many applications, it fails to capture the phase front variations of light waves, which contain valuable additional information about imaged objects. This limitation is particularly problematic in biomedical imaging, where many biological specimens are effectively transparent and only modulate the phase front of transmitted light.  

Existing phase microscopy techniques, including differential interference contrast (DIC) microscopy, phase contrast microscopy, and Hoffman phase microscopy, have been used for decades to visualize such phase-modulating specimens. However, these techniques suffer from several significant drawbacks. First, they mix phase information with intensity information, creating ambiguities in rendered images and preventing straightforward quantitative phase analysis. Second, they require specialized optical components that must be switched in and out during operation, increasing system complexity. Third, DIC microscopy in particular produces artifacts when imaging birefringent samples due to its reliance on polarized light. Finally, these systems are relatively expensive, limiting their broader adoption in biomedical applications where cost considerations are important.  

More recent phase microscopy techniques have attempted to address some of these limitations, but they typically require laser sources and sophisticated optical arrangements, making them impractical for routine use. Other approaches that calculate optical phase by collecting multiple images at different focal planes suffer from intrinsic speed limitations due to the need for physical actuation of the camera. There remains a significant unmet need for a simple, cost-effective solution that can directly measure both intensity and phase front variations of light waves without requiring complex optical configurations or expensive components.  

## BRIEF SUMMARY OF THE INVENTION  

The present invention discloses a novel wavefront image sensor (WIS) chip capable of simultaneously measuring both the intensity and phase front variations of an incident light field. The WIS chip comprises a two-dimensional array of circular apertures defined on top of a metal-coated image sensor chip (such as a CCD or CMOS sensor), with a transparent spacer separating the apertures from the sensor pixels.  

When a light wave impinges upon the aperture array, each aperture forms a projection spot on the sensor pixels underneath. The center position of each projection spot shifts according to the phase gradient of the incident light wave over its corresponding aperture, while the total signal from the projection spot provides a measure of the local light intensity. By analyzing both the position and intensity of these projection spots, the WIS can separately retrieve the intensity and phase information of the incident light wave.  

A key innovation of the WIS is its operation in a high Fresnel number regime, where light transmitted through an aperture self-focuses before spreading due to diffraction. By positioning the sensor at this self-focusing plane, the WIS achieves tightly confined projection spots while maintaining sensitivity to phase front gradients. This enables the creation of a compact, high-density phase-sensitive image sensor that can be fabricated using standard semiconductor manufacturing processes.  

The WIS can be incorporated into standard optical systems, such as microscopes, to provide simultaneous intensity and quantitative phase imaging without requiring specialized optical components. Compared to existing phase microscopy techniques, the WIS offers several advantages: (1) it provides truly quantitative phase measurements, (2) it is immune to birefringence artifacts, (3) it clearly separates intensity and phase information, (4) it is simple and inexpensive to implement, and (5) it does not require complex bulk optical arrangements.  

## DETAILED DESCRIPTION OF THE INVENTION  

The wavefront image sensor (WIS) of the present invention represents a significant advancement in optical sensing technology by enabling simultaneous measurement of both intensity and phase front variations of an incident light field. The fundamental architecture of the WIS comprises three primary components: a two-dimensional array of precisely defined apertures, a metal light-blocking layer, and an underlying image sensor chip separated from the apertures by a transparent spacer layer.  

The aperture array is fabricated on top of a conventional image sensor chip (such as a CCD or CMOS sensor) that has been coated with a metal layer (typically aluminum) to block light transmission except through the defined apertures. Each aperture in the array is circular with a diameter optimized for the intended application, typically in the range of 5-10 micrometers. The spacing between adjacent apertures is carefully selected to provide sufficient sampling of the incident wavefront while maintaining adequate separation between projection spots.  

A critical aspect of the WIS design is the spacer layer that separates the aperture plane from the photosensitive elements of the image sensor chip. The thickness and refractive index of this spacer are chosen to position the sensor at the plane where light transmitted through each aperture undergoes self-focusing in the high Fresnel number regime. This self-focusing effect produces tightly confined projection spots that are essential for accurate determination of both intensity and phase gradient information.  

In operation, when a light wave impinges upon the WIS, each aperture transmits a portion of the incident light which forms a projection spot on the underlying sensor pixels. The lateral position of each projection spot shifts in proportion to the local phase gradient of the incident light wave over the corresponding aperture. This relationship can be expressed mathematically as:  

Δx = H · θx  

where Δx is the spot displacement in the x-direction, H is the effective distance from the aperture to the sensor plane, and θx is the normalized phase gradient in the x-direction. A similar relationship holds for displacement in the y-direction. The total signal from the pixels underlying each aperture provides a simultaneous measurement of the local light intensity.  

To achieve high measurement precision, the WIS employs a specialized algorithm for determining projection spot positions with sub-pixel accuracy. This algorithm uses cyclic complex weights applied to the pixel values underlying each aperture to calculate precise spot centroids. The algorithm is robust against noise and provides excellent linearity in phase gradient measurements over a wide dynamic range.  

The WIS can be fabricated using standard semiconductor manufacturing techniques. In one embodiment, the process begins with a commercial CMOS image sensor chip from which the protective glass window has been removed. The sensor surface is planarized with a layer of SU8 resin or similar material, followed by deposition of a metal light-blocking layer (typically 150 nm of aluminum). Photolithography is then used to pattern the aperture array in the metal layer. The SU8 layer serves both as a planarization layer to nullify any microlenses present on the commercial sensor and as part of the spacer between apertures and pixels.  

The performance characteristics of the WIS have been extensively characterized. Under typical operating conditions, the WIS can measure local phase gradients with a sensitivity better than 0.1 milliradians. The linear response range extends to at least ±15 milliradians, sufficient for most microscopy applications. The intensity measurement capability matches that of conventional image sensors, with the added benefit of simultaneous phase information.  

When incorporated into a standard microscope in place of a conventional camera, the WIS transforms the microscope into a wavefront microscope (WM) capable of simultaneous bright-field and quantitative phase gradient imaging. The WM provides several advantages over traditional phase microscopy techniques:  

1. Quantitative phase measurements: Unlike DIC microscopy which provides only qualitative phase contrast, the WM delivers truly quantitative phase gradient data that can be used for precise optical path length measurements and other quantitative analyses.  

2. Immunity to birefringence artifacts: Because the WM does not rely on polarized light for phase imaging, it can image strongly birefringent samples without the artifacts that plague DIC microscopy.  

3. Orthogonal phase information: The WM captures phase gradient information in two orthogonal directions simultaneously, providing a more complete characterization of the sample's phase properties than single-axis techniques like DIC.  

4. Simplified operation: The WM requires no moving parts or special optical components beyond those found in a standard microscope, making it easier to use and more reliable than traditional phase microscopy systems.  

5. Cost-effectiveness: Since the WIS can be fabricated using standard semiconductor processes, it can be produced at costs comparable to conventional image sensors, making phase microscopy accessible to a much broader range of users.  

The WIS technology has been demonstrated in several practical applications including imaging of polystyrene microspheres, unstained starfish embryos, and birefringent potato starch granules. In all cases, the WM provided superior performance compared to DIC microscopy, particularly in situations involving stained or birefringent samples where DIC images become ambiguous or artifact-laden.  

Beyond microscopy, the WIS has numerous potential applications in fields such as adaptive optics, machine vision, texture assessment, and object ranging. In the medical field, it could significantly impact procedures like LASIK surgery and high-resolution retinal imaging. The compact size and low cost of the WIS make it practical for applications where traditional Shack-Hartmann sensors are too bulky or expensive to consider.  

Future developments of the WIS technology may include integration with color imaging capabilities by using stratified color sensor chips or Bayer-pattern sensors with enhanced spot localization algorithms. Additional improvements could involve optimizing the aperture spacing for specific applications or developing versions with enhanced sensitivity for low-light conditions.  

The wavefront image sensor represents a fundamental advance in optical sensing technology by providing a simple, cost-effective solution for simultaneous intensity and phase measurements. Its compatibility with standard semiconductor manufacturing processes ensures that it can be produced in large quantities at reasonable cost, while its straightforward integration into existing optical systems guarantees broad applicability across multiple fields.