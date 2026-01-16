# DESCRIPTION

## BACKGROUND

The field of object recognition has seen significant advancements with the advent of machine learning and computer vision technologies. Traditional approaches to object recognition involve capturing digital images using cameras, converting these images from analog to digital format, and then processing them using digital processors to implement artificial neural networks (ANNs). However, these systems are often bulky, power-hungry, and suffer from latency issues due to the sequential processing of different technology modules. Additionally, they are vulnerable to cyber-attacks, which can compromise data security.

To address these challenges, an emerging physical platform for object recognition is the optical neural network (ONN). ONNs utilize photonic elements and circuits to form a layered architecture that emulates digital ANNs, directly processing optical signals from target objects. In an ideal ONN, optical signals are manipulated by layers of elements that perform linear transformations and nonlinear activations, which are pre-trained to enable the network to perform device-specific computing tasks. ONNs offer several advantages over traditional digital ANNs, including computational speed characterized by the propagation speed of light, the potential for being entirely passive (requiring no additional power after the optical input is generated), and enhanced data security due to the intrinsic and engineered materials of ONNs.

Various optical platforms have been explored for implementing ONNs, including integrated photonic circuits and diffractive ONNs based on metamaterials. Integrated photonic circuits use tunable Mach-Zehnder interferometers (MZIs) to encode signals in an array of pulses and conduct linear transformations. However, the large size of MZIs limits the expressive power and capabilities of these ONNs. Diffractive ONNs, on the other hand, use the physics of wave dynamics to manipulate continuous optical wavefronts, achieving linear transformations and nonlinear activations through light scattering at optically linear and nonlinear inclusions of the metamaterial. While 3D metamaterial ONNs can provide a large expressive power, current manufacturing techniques only allow for reliable fabrication of 2D versions without nonlinear inclusions, which significantly reduces their expressive power.

A more viable approach to realizing diffractive ONNs is to condense the depth of the network into a few discrete layers of diffractive components. These components use the Huygens-Fresnel principle to form diffraction patterns as wavefronts propagate from one layer to the next. The 2D distribution of scattering elements on a diffractive component can be trained to shape a wavefront in phase, amplitude, and polarization, making this architecture ideal for image-based computer vision applications, including object recognition. Previous work has demonstrated such diffractive ONNs, which are composed of multiple layers of "diffractive surfaces" and can classify objects such as hand-written digits and fashion objects. However, these approaches have limitations in terms of pixel size and the ability to modulate all properties of light, which restricts their expressive power and practical utility.

To overcome these limitations, we propose and demonstrate a diffractive ONN based on metasurfaces, referred to as a metasurface "smart glass." Metasurfaces are 2D versions of metamaterials that utilize strong interactions between light and 2D nanostructured thin films to control light in desired ways. They are composed of a 2D array of nano-pillars (meta-units) of various cross-sectional shapes and can offer complete and precise manipulation of optical phase, amplitude, and polarization across the wavefront with sub-wavelength resolution. The collective response of millions of sub-wavelength meta-units enables efficient parallel computing with a high level of expressive power, allowing tasks typically solved using complex, multi-layered networks to be accomplished by a metasurface singlet or doublet. Metasurfaces are manufactured using CMOS-compatible nanofabrication techniques, enabling miniaturization of the discrete-layered diffractive neural networks operating in the optical spectral range, where light sources and detectors are readily available. Our metasurface smart glasses do not require any power supply or digital processor, acting as passive computing devices that operate at the speed of light. The potential of metasurface-based neural networks has been demonstrated in classification tasks involving simple binary objects such as digits and fashion objects.

## SUMMARY

The present invention relates to a diffractive optical neural network (ONN) based on metasurfaces, specifically a metasurface "smart glass," that directly processes light waves scattered by an object using its internal nanostructures. The metasurface smart glass is designed to recognize and classify objects, such as hand-written digits and alphabetical letters, with high accuracy and robustness. The metasurface is composed of a 2D array of nano-pillars (meta-units) that can precisely manipulate the phase, amplitude, and polarization of the optical wavefront with sub-wavelength resolution. The metasurface smart glass operates passively, requiring no power supply or digital processor, and processes light at the speed of light. It is manufactured using CMOS-compatible nanofabrication techniques, enabling miniaturization and integration into compact devices.

The metasurface smart glass can be configured as a single-layered or multi-layered device, depending on the complexity of the recognition task. Single-layered metasurfaces are suitable for recognizing a limited number of classes of objects, such as four or ten hand-written digits, with high accuracy. Multi-layered metasurfaces, such as metasurface doublets, can handle more complex tasks, such as human facial verification, by mapping an image into a low-dimensional representation and evaluating the similarity between two images.

The invention also includes methods for training the metasurface smart glass using a dataset of input objects and optimizing the phase profile of the metasurface to maximize the light intensity within specific zones on the detection plane corresponding to the classification labels of the input objects. The training process involves numerically computing the propagation of light waves through the diffractive network using the Rayleigh-Sommerfeld diffraction theory and minimizing a loss function that evaluates the cross-entropy between the calculated intensity distribution and the target intensity distribution.

Additionally, the invention demonstrates the use of polarization multiplexing to enhance the expressive power of the metasurface smart glass, allowing it to perform multiple tasks simultaneously. For example, a single metasurface can recognize alphabetical letters and distinguish their typographic styles (normal or italic) using light at orthogonal polarization states. The metasurface smart glass can also be extended to recognize a larger number of classes of objects and to handle more complex objects by increasing the width and depth of the ONN, such as using phase-amplitude metasurfaces, implementing polarization and wavelength multiplexing, and cascading metasurface layers.

The metasurface smart glass offers several advantages over traditional digital ANNs, including high computational speed, low power consumption, compact form factor, and enhanced data security. These features make it particularly suitable for edge perception devices in applications where minimal service, resilience to interference, high energy efficiency, and information security are critical.

## DETAILED DESCRIPTION

### Overview of the Metasurface Smart Glass

The metasurface smart glass is a diffractive optical neural network (ONN) that directly processes light waves scattered by an object using its internal nanostructures. The metasurface is composed of a 2D array of nano-pillars (meta-units) that can precisely manipulate the phase, amplitude, and polarization of the optical wavefront with sub-wavelength resolution. The metasurface operates passively, requiring no power supply or digital processor, and processes light at the speed of light. It is manufactured using CMOS-compatible nanofabrication techniques, enabling miniaturization and integration into compact devices.

### Working Principle

The working principle of the metasurface smart glass is illustrated in the context of recognizing hand-written digits. An input object, such as a hand-written digit "4," generates an optical wavefront with characteristic amplitude and phase profiles when excited by an incident coherent light beam. This complex optical wavefront propagates over a certain distance (object distance) and is then processed by the metasurface, which superimposes a phase modulation to the wavefront. The modulated light wave further propagates over a certain distance (imaging distance) in the forward direction and produces an optical diffraction pattern that lights up a few predefined zones on the detection plane. The zone that receives the highest optical intensity identifies the initial object. The input object, the metasurface, and the detection plane represent, respectively, an input layer, a hidden layer, and an output layer of a neural network, and every pixel in either one of the three layers represents an artificial neuron. In this configuration, each neuron in the hidden layer is connected to all the neurons in the input layer via optical interference, and each neuron in the output layer is similarly connected to all the neurons in the hidden layer. The optical interference provides a form of nonlinear activation by generating cross-products of optical wavelets. The phase modulation at each neuron of the hidden layer represents a trainable linear transformation. Object recognition is accomplished by training all the neurons in the hidden layer to maximize the light intensity within a specific zone of the output layer, depending on the classification label of the input object.

### Design and Fabrication

The metasurface smart glass is designed for operation in the near-infrared light at a wavelength of 1,550 nm. The input object and the metasurface both have a dimension of 500 λ × 500 λ and are digitized into 1000 × 1000 pixels. The object and imaging distances are both 2000 λ. The smart glass is composed of a single metasurface modeled as a phase mask with zero thickness on a substrate with a thickness of 322.58 λ (~500 µm) and a refractive index of 1.44 (silicon dioxide). The metasurface is made of amorphous silicon for its low extinction coefficient in the near-infrared and is composed of meta-units 1 µm in height and arranged in a square lattice with a periodicity of 750 nm on a silicon dioxide substrate.

### Training Process

The training process involves feeding optically coherent, binary images (e.g., hand-written digits and alphabetic letters) into the neural network and numerically computing the propagation of light waves through the diffractive network using the Rayleigh-Sommerfeld diffraction theory. A loss function is defined to evaluate the cross-entropy between the calculated intensity distribution over the detection plane and the target intensity distribution, which is 1 for the zone that matches with the label of the input and 0 elsewhere. The phase profile of the metasurface is iteratively adjusted using a large number of input objects during the training process, where the loss function is minimized using the "Adam" optimization algorithm adapted from the stochastic gradient-based optimization method. Several measures are taken to improve the robustness of the ONN against experimental errors, such as non-uniform optical illumination, random mispositioning of the input object, smart glass, and detection zones, and random variations of the object and imaging distances. An auxiliary term, proportional to the ratio between the intensity in the predefined zones of the detection plane and the total intensity in the detection plane, is subtracted from the overall loss function to increase the contrast of the zones of interest over the optical background.

### Experimental Setup

A schematic of the experimental setup is shown in the figures. A telecom laser beam (λ = 1,550 nm) is incident on a photomask to create input optical objects. The photomask is made of a black emulsion photo-plotted on a mylar sheet, containing a 2D array of objects (i.e., numerical digits or alphabetic letters) that are transparent within an object and opaque outside of it. The incident beam has a diameter of approximately 3 mm, which is much larger than the size of individual input objects (0.775 mm × 0.775 mm), to minimize non-uniformity in illumination. A motorized translation stage each time moves one input object to the central axis of the optical setup. The input object is relayed by a telescope with a unity magnification and the relayed object is superimposed onto a square aperture (0.775 mm × 0.775 mm), so that stray light from adjacent input objects on the photomask is blocked. The diffraction pattern of the object is then processed by the metasurface smart glass, and the output image is collected by a microscope with an objective focused on the detection plane and measured by an InGaAs camera. The optical intensities in the predefined detection zones are extracted from the image, and the identity of the input object is predicted according to the zone receiving the highest intensity.

### Recognition of Hand-Written Digits

The first functionality demonstrated experimentally is to recognize 4 classes of numerical digits, {0, 1, 3, 4}, from the MNIST hand-written digit database. The phase modulation is trained to concentrate light scattered from the binary image of a digit into one of the four square zones on the detection plane. The trained phase modulation is implemented by a metasurface based on meta-units that are optically isotropic. The metasurface smart glass successfully classifies digits based on the resulting intensity distributions on the detection plane. The observed diffraction patterns on the detection plane agree well with analytically calculated diffraction patterns, indicating that the metasurface provides a precise phase modulation consistent with the design. The experimental recognition accuracy of 99.14% is achieved based on measurement of 116 input digits (4 classes and N>25 for each class).

Next, the classification of all 10 classes of hand-written digits is explored using a single-layered metasurface smart glass. The trained optical phase profile of the metasurface is implemented similarly using a metasurface based on optically isotropic meta-units. The 10 circular detection zones are arranged in a circular array on the detection plane. The experimental recognition accuracy of 78.37% is achieved based on measurement of 208 input digits (10 classes and N>10 for each class). The experimental accuracy is lower than the theoretical accuracy of 86.50%, suggesting that the ONN has a reduced robustness against experimental errors.

### Polarization Multiplexing and Multitasking Smart Glasses

To reduce the complexity of the 10-digit recognition task, a polarization-multiplexing strategy is devised. The smart glass is constructed using the birefringent meta-unit library to provide distinct phase modulations for light polarized in orthogonal directions. Horizontally polarized light is used for recognizing digits {1, 3, 4, 7, 8}, and vertically polarized light is used for recognizing digits {0, 2, 5, 6, 9}. The training process reports accuracies of 94.80% and 94.00% for the two groups of digits, respectively. The experimental recognition accuracies achieved based on measurement of 111 input digits belonging to the first group and 97 digits belonging to the second group are 90.99% and 81.44%, respectively. The phase coverage provided by the birefringent meta-unit library is more discrete than that of the isotropic meta-unit library, leading to deviations in the phase responses of the fabricated birefringent metasurface from the desired phase profiles.

Polarization multiplexing is also utilized to realize a multi-tasking metasurface smart glass that classifies typed alphabetical letters and simultaneously distinguishes the typographic styles of the letters. When incident illumination is polarized in the horizontal direction, scattered light from a letter with a certain font is modulated by the smart glass to preferentially light up one of the four zones on the detection plane, corresponding to 4 letters: {A, B, C, D}. Scattered light polarized in the vertical direction falls in one of the two zones in the upper row to indicate if the letter is normal or italic. Experiments using 168 inputs (4 letters each with 21 fonts and 2 typographic styles) demonstrate accuracies of 92.81% and 100% for letter classification and typographic style recognition, respectively.

### Facial Verification Using Double-Layered Metasurface Smart Glass

Complex recognition tasks beyond digit or letter classification require metasurfaces with enhanced expressive power. A metasurface doublet is theoretically demonstrated for human facial verification. The metasurface doublet maps an image into a 3×3 intensity array on the detection plane, and the similarity between two images is evaluated by calculating the Euclidean distance between the two resulting intensity arrays. If the Euclidean distance is below a threshold, the two images are considered a match; if the distance is above the threshold, the two images are considered to represent distinct persons. Using a dataset consisting of photos of 100 people, each person with 14 distinct photos, the photos of 90 people are used to train the metasurface doublet, and the photos of the remaining 10 people are used in the test. The result shows that when the threshold Euclidean distance is appropriately chosen, the rate of false acceptance and the rate of false rejection are both approximately 10%, resulting in a verification accuracy of approximately 80%.

### Robustness of Metasurface ONN

In simulation, an ONN consisting of a single metasurface is usually sufficient to provide a high accuracy of >90% for simple tasks such as digit or letter recognition. However, experiments may report a lower accuracy by a few percent to 20%. This discrepancy is related to the robustness of metasurface smart glasses against experimental errors, and the intensity contrast between the detection zones with the highest and second-highest intensities can quantify the robustness of the ONN design. The inter-zone contrast positively correlates with the degree of agreement between theoretical and experimental recognition accuracies. By considering this inter-zone contrast in the loss function or by increasing its weight in the loss function while training the ONN, the impact of experimental errors on the performance of the ONNs can be mitigated.

### Increasing Expressive Power of Metasurface ONN

The expressive power of the metasurface smart glass can be increased by scaling up the width and depth of the ONN. The ONN depth can be increased by using a multi-layered metasurface architecture, as demonstrated by the metasurface doublet for human facial verification. The ONN width can be increased by employing several strategies. First, polarization multiplexing can double the expressive power of a metasurface. Second, metasurfaces providing complete and independent control of optical phase and amplitude can be more powerful building blocks of an ONN compared to phase-only metasurfaces. Third, wavelength-multiplexing can introduce an additional dimension to increase the expressive power of an ONN by engineering the optical dispersion of meta-units. Lastly, including an array of distinct metasurfaces in each layer of the neural network is an effective approach to increase its expressive power. Preliminary investigations indicate that a single layer of 10 distinct metasurfaces can classify 10 classes of incoherent objects with an accuracy higher than 90%.

### Conclusion

The metasurface smart glass is a diffractive optical neural network that can recognize and classify objects with high accuracy and robustness. It operates passively, requiring no power supply or digital processor, and processes light at the speed of light. The metasurface is composed of a 2D array of nano-pillars (meta-units) that can precisely manipulate the phase, amplitude, and polarization of the optical wavefront with sub-wavelength resolution. The metasurface smart glass can be configured as a single-layered or multi-layered device, depending on the complexity of the recognition task. The invention also includes methods for training the metasurface smart glass and enhancing its expressive power through various strategies. The metasurface smart glass offers several advantages over traditional digital ANNs, including high computational speed, low power consumption, compact form factor, and enhanced data security, making it particularly suitable for edge perception devices in applications where minimal service, resilience to interference, high energy efficiency, and information security are critical.

### EXAMPLES

#### Example 1: Recognition of Hand-Written Digits

**Objective:** To demonstrate the recognition of 4 classes of numerical digits, {0, 1, 3, 4}, using a single-layered metasurface smart glass.

**Methodology:**
1. **Phase Modulation Training:** The phase modulation of the metasurface is trained to concentrate light scattered from the binary image of a digit into one of the four square zones on the detection plane.
2. **Experimental Setup:** A telecom laser beam (λ = 1,550 nm) is incident on a photomask containing a 2D array of hand-written digits. The input object is relayed by a telescope with a unity magnification and superimposed onto a square aperture. The diffraction pattern of the object is processed by the metasurface smart glass, and the output image is collected by a microscope and measured by an InGaAs camera.
3. **Data Analysis:** The optical intensities in the predefined detection zones are extracted from the image, and the identity of the input object is predicted according to the zone receiving the highest intensity.

**Results:**
- The metasurface smart glass successfully classifies digits based on the resulting intensity distributions on the detection plane.
- The observed diffraction patterns on the detection plane agree well with analytically calculated diffraction patterns.
- The experimental recognition accuracy of 99.14% is achieved based on measurement of 116 input digits (4 classes and N>25 for each class).

#### Example 2: Recognition of All 10 Classes of Hand-Written Digits

**Objective:** To demonstrate the recognition of all 10 classes of hand-written digits using a single-layered metasurface smart glass.

**Methodology:**
1. **Phase Modulation Training:** The phase modulation of the metasurface is trained to concentrate light scattered from the binary image of a digit into one of the 10 circular zones on the detection plane.
2. **Experimental Setup:** Similar to Example 1, but with a larger number of detection zones.
3. **Data Analysis:** The optical intensities in the predefined detection zones are extracted from the image, and the identity of the input object is predicted according to the zone receiving the highest intensity.

**Results:**
- The experimental recognition accuracy of 78.37% is achieved based on measurement of 208 input digits (10 classes and N>10 for each class).
- The experimental accuracy is lower than the theoretical accuracy of 86.50%, suggesting that the ONN has a reduced robustness against experimental errors.

#### Example 3: Polarization Multiplexing for 10-Digit Recognition

**Objective:** To demonstrate the use of polarization multiplexing to reduce the complexity of the 10-digit recognition task.

**Methodology:**
1. **Phase Modulation Training:** The metasurface is constructed using the birefringent meta-unit library to provide distinct phase modulations for light polarized in orthogonal directions.
2. **Experimental Setup:** Horizontally polarized light is used for recognizing digits {1, 3, 4, 7, 8}, and vertically polarized light is used for recognizing digits {0, 2, 5, 6, 9}.
3. **Data Analysis:** The optical intensities in the predefined detection zones are extracted from the image, and the identity of the input object is predicted according to the zone receiving the highest intensity.

**Results:**
- The training process reports accuracies of 94.80% and 94.00% for the two groups of digits, respectively.
- The experimental recognition accuracies achieved based on measurement of 111 input digits belonging to the first group and 97 digits belonging to the second group are 90.99% and 81.44%, respectively.

#### Example 4: Multi-Tasking Metasurface Smart Glass

**Objective:** To demonstrate a multi-tasking metasurface smart glass that classifies typed alphabetical letters and simultaneously distinguishes the typographic styles of the letters.

**Methodology:**
1. **Phase Modulation Training:** The metasurface is constructed using the birefringent meta-unit library to provide distinct phase modulations for light polarized in orthogonal directions.
2. **Experimental Setup:** Horizontally polarized light is used for classifying letters {A, B, C, D}, and vertically polarized light is used for distinguishing the typographic styles (normal or italic).
3. **Data Analysis:** The optical intensities in the predefined detection zones are extracted from the image, and the identity of the input object is predicted according to the zone receiving the highest intensity.

**Results:**
- Experiments using 168 inputs (4 letters each with 21 fonts and 2 typographic styles) demonstrate accuracies of 92.81% and 100% for letter classification and typographic style recognition, respectively.

#### Example 5: Facial Verification Using Double-Layered Metasurface Smart Glass

**Objective:** To demonstrate the use of a double-layered metasurface smart glass for human facial verification.

**Methodology:**
1. **Phase Modulation Training:** The metasurface doublet is trained to map an image into a 3×3 intensity array on the detection plane.
2. **Experimental Setup:** A dataset consisting of photos of 100 people, each person with 14 distinct photos, is used for training and testing the ONN.
3. **Data Analysis:** The similarity between two images is evaluated by calculating the Euclidean distance between the two resulting intensity arrays. If the Euclidean distance is below a threshold, the two images are considered a match; if the distance is above the threshold, the two images are considered to represent distinct persons.

**Results:**
- The result shows that when the threshold Euclidean distance is appropriately chosen, the rate of false acceptance and the rate of false rejection are both approximately 10%, resulting in a verification accuracy of approximately 80%.