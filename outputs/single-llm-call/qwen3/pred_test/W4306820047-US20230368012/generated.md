# DESCRIPTION

## BACKGROUND

- introduce object recognition  
Object recognition has become a foundational capability across a broad spectrum of modern technological systems, enabling applications ranging from autonomous navigation and security surveillance to medical diagnostics and human-computer interaction. The process involves the identification and classification of visual entities within images or video streams, typically by extracting and interpreting spatial patterns of light intensity and contrast. Traditional approaches rely on digitizing optical signals through camera sensors and processing them using computational algorithms implemented on digital hardware. These systems convert continuous optical wavefronts into discrete pixel arrays, which are then analyzed by layered artificial neural networks trained to detect specific features or categories. The entire pipeline—from light collection to decision output—is inherently sequential and requires multiple physical components operating in distinct domains, introducing latency, power consumption, and vulnerability to external interference.

- describe limitations of digital ANNs  
Digital artificial neural networks, while highly effective in many classification tasks, suffer from fundamental constraints that limit their suitability for real-time, low-power, or secure deployment. The necessity to digitize optical inputs imposes a significant bottleneck, as analog-to-digital conversion consumes energy and introduces quantization noise. Furthermore, the sequential nature of digital computation—where data must be transferred between memory and processing units—results in unavoidable delays that degrade responsiveness. These systems are also susceptible to cyberattacks, as the intermediate representations of visual data exist as digital files that can be intercepted, manipulated, or reverse-engineered. Additionally, the computational load scales unfavorably with input resolution and network depth, making real-time processing on mobile or embedded platforms challenging without substantial hardware resources. The reliance on electricity for every operation renders these systems incompatible with environments where power availability is limited or unreliable.

- introduce optical neural networks (ONNs)  
Optical neural networks represent a paradigm shift by performing neural computation directly within the optical domain, eliminating the need for digitization and digital processing. These systems exploit the natural wave properties of light—such as interference, diffraction, and superposition—to implement linear transformations and nonlinear activations through engineered physical structures. By encoding information in the amplitude, phase, or polarization of light waves, optical neural networks can process entire images in parallel at the speed of light, with no intermediate storage or electronic conversion. The computational logic is embedded in the geometry and material composition of passive optical elements, allowing the system to operate without external power once illuminated. This approach not only accelerates inference but also enhances security, as the input data never leaves the physical realm of optical propagation, leaving no digital trace that can be compromised.

- highlight challenges of ONNs  
Despite their theoretical promise, optical neural networks face significant practical challenges in achieving high accuracy and robustness across diverse recognition tasks. Early implementations were constrained by the limited spatial resolution and tunability of available optical components, such as spatial light modulators or bulk diffractive elements, which could not precisely control light at subwavelength scales. The expressive power of these systems was further restricted by their inability to simultaneously modulate multiple optical properties—phase, amplitude, and polarization—within a single compact structure. Additionally, most prior designs required active components, external lasers, or complex alignment systems, undermining their potential for miniaturization and field deployment. The lack of robust training methodologies tailored to physical wave dynamics also hindered the development of systems capable of generalizing across variations in illumination, object orientation, or environmental noise. These limitations have historically confined optical neural networks to simplified binary classification tasks under highly controlled laboratory conditions.

## SUMMARY

- introduce target recognition system  
A target recognition system is disclosed that performs optical inference using a passive, metasurface-based architecture capable of identifying objects through direct manipulation of incident light waves without digital intervention. The system operates as a fully optical neural network, where the input object, the metasurface processor, and the detection plane collectively emulate the layers of a neural network, with each subwavelength meta-unit acting as a tunable neuron. Light scattered from the target object is transformed by the metasurface according to a pre-engineered phase and polarization profile, resulting in a distinct intensity distribution across predefined detection zones that correspond to specific object classes. This transformation occurs in a single pass, at the speed of light, and requires no external power source, digital processor, or data storage.

- describe system components  
The system comprises three essential components: an input plane where the target object is illuminated by coherent or partially coherent light, a metasurface layer positioned at a fixed distance downstream that modulates the optical wavefront in phase and polarization, and an output plane containing multiple detection zones where the final intensity distribution is captured. The metasurface is fabricated as a two-dimensional array of nanostructured meta-units on a transparent dielectric substrate, each designed to impose a specific phase shift and, in some embodiments, polarization conversion on incident light. The detection zones are spatially separated regions on a photodetector array or imaging sensor, each corresponding to a unique classification label. The system is configured such that the intensity peak in the output plane unambiguously identifies the input object.

- specify light wave processing  
Light wave processing is achieved through controlled diffraction and interference phenomena induced by the spatially varying refractive index profile of the metasurface. As the optical wavefront propagates from the input object to the metasurface and then to the detection plane, the meta-units sculpt the phase and amplitude of the scattered light in a manner that maximizes constructive interference within the designated detection zone corresponding to the correct class. This process inherently performs a nonlinear transformation through the cross-product of overlapping wavelets, enabling the system to distinguish between complex patterns without the use of explicit nonlinear activation functions. The entire computation is governed by Maxwell’s equations and executed in a single optical propagation step.

- describe target types  
The system is capable of recognizing a wide variety of target types, including but not limited to handwritten digits, typographic characters, facial images, and other optically coherent or partially coherent visual patterns. Targets may be presented as binary masks, grayscale images, or color-encoded representations, provided they are illuminated by a coherent light source. The system is particularly effective for targets with high spatial contrast and well-defined boundaries, though extensions to incoherent illumination are enabled through multi-metasurface architectures. The recognition capability is not limited to static images; dynamic sequences may be processed in real time due to the absence of computational latency.

- specify wavelength range  
The system is designed to operate within the near-infrared spectral range, specifically at a wavelength of approximately 1,550 nanometers, a band compatible with standard telecommunications components and silicon-based photodetectors. This wavelength is selected to minimize absorption losses in the substrate and meta-unit materials while ensuring compatibility with high-efficiency light sources and detectors. The design may be adapted to other wavelengths, including visible and mid-infrared ranges, by adjusting the dimensions and material composition of the meta-units to maintain subwavelength phase control.

- describe power supply and speed  
The system requires no external power supply for computation. Once illuminated by an incident light source, the metasurface performs inference passively, relying solely on the energy of the incoming photons. The speed of recognition is governed by the propagation time of light across the system, which is on the order of picoseconds for typical device dimensions, enabling real-time processing at rates exceeding billions of inferences per second. This ultrafast operation is independent of the complexity of the classification task, as all computations occur simultaneously across the entire metasurface.

- describe target recognition  
Target recognition is accomplished by training the metasurface to map each input class to a unique spatial intensity distribution on the detection plane. During training, a loss function is minimized by iteratively adjusting the phase profile of each meta-unit to maximize the intensity in the designated detection zone while suppressing background intensity elsewhere. The trained metasurface then acts as a fixed optical filter that, when illuminated by any input from the trained class, produces a detectable peak in the corresponding zone. Recognition is determined by identifying the zone with the highest measured intensity, without the need for digital comparison or classification algorithms.

- describe meta-unit materials  
The meta-units are composed of high-refractive-index dielectric materials, such as amorphous silicon, titanium dioxide, or silicon nitride, patterned into subwavelength nanostructures on a low-refractive-index substrate such as silicon dioxide. These materials are chosen for their low optical loss in the operating wavelength range and their ability to induce large phase shifts with minimal absorption. The meta-units are arranged in a square lattice with periodicity below the diffraction limit, enabling precise wavefront control. In some embodiments, birefringent meta-units are employed to independently modulate phase for orthogonal polarization states, enabling polarization-multiplexed operation.

- describe meta-unit symmetries  
Meta-units are engineered with specific rotational symmetries to control their optical response. Isotropic meta-units, possessing four-fold rotational symmetry, provide phase modulation that is independent of the polarization state of incident light. Birefringent meta-units, with two-fold symmetry, introduce a phase difference between orthogonal polarization components, enabling independent control of light with horizontal and vertical polarization. The choice of symmetry directly influences the expressive power of the metasurface, with birefringent designs enabling multitasking and increased classification capacity through polarization multiplexing.

- describe output plane and detection zones  
The output plane is a planar surface located at a predetermined distance from the metasurface, upon which the diffracted light forms a distinct intensity pattern. Detection zones are predefined regions on this plane, each associated with a specific target class. These zones are spatially separated to minimize crosstalk and are sized to capture the peak intensity distribution corresponding to a particular input. The number and arrangement of detection zones are determined by the number of classes to be recognized and may be arranged in a grid, circular array, or other geometric configuration optimized for separation and signal-to-noise ratio.

- describe target recognition methods  
Target recognition is performed by illuminating the input object with a coherent light source and capturing the resulting intensity distribution on the output plane using a photodetector array. The identity of the object is determined by identifying the detection zone with the highest integrated intensity. This method requires no digital processing, machine learning inference, or algorithmic decision-making beyond the physical response of the metasurface. The system’s accuracy is determined during training by optimizing the phase profile of each meta-unit to maximize the contrast between the target zone and all other zones, ensuring robustness against minor misalignments or illumination variations.

## DETAILED DESCRIPTION

- define technical terms  
For the purposes of this disclosure, the term “metasurface” refers to a two-dimensional array of subwavelength nanostructures engineered to manipulate the phase, amplitude, and polarization of incident electromagnetic waves. The term “meta-unit” denotes an individual structural element within the metasurface, typically on the order of tens to hundreds of nanometers in lateral dimension and of comparable height. “Optical wavefront” refers to the spatial distribution of the electric field amplitude and phase of light as it propagates through free space or a medium. “Detection zone” denotes a predefined region on the output plane where light intensity is measured to determine the classification of the input object. “Passive operation” means the system performs computation without requiring external power, active components, or electronic feedback.

- provide definitions for "about" and "approximately"  
The terms “about” and “approximately” are used herein to indicate that a stated value may vary by up to ±10% due to manufacturing tolerances, measurement uncertainties, or environmental fluctuations without departing from the scope of the invention. For example, a wavelength of approximately 1,550 nm encompasses values ranging from 1,395 nm to 1,705 nm. Similarly, an accuracy of approximately 80% includes any measured performance between 72% and 88%.

- describe system for processing light  
The system for processing light consists of a coherent illumination source, an input object plane, a metasurface layer, and an output detection plane, all aligned along a common optical axis. Light from the source illuminates the input object, generating a scattered wavefront that carries the spatial information of the object. This wavefront propagates to the metasurface, where each meta-unit imparts a local phase shift and, in some cases, polarization conversion, sculpting the wavefront into a new configuration. The modified wavefront then propagates to the detection plane, where interference between wavelets from different meta-units creates a unique intensity pattern for each input class. The entire process is governed by the laws of physical optics and requires no digital intervention.

- introduce meta-units and metasurface  
Meta-units are nanoscale dielectric structures arranged in a periodic lattice to form a metasurface. Each meta-unit is designed to induce a specific phase delay on incident light, depending on its geometry, orientation, and material composition. The collective response of millions of such units enables the metasurface to perform complex optical transformations, effectively acting as a programmable optical lens with spatially varying focal properties. The metasurface is fabricated using standard nanolithographic techniques on a transparent substrate, resulting in a thin, lightweight, and robust optical component suitable for integration into compact imaging systems.

- explain spatial and spectral control of light  
The metasurface provides spatial control by modulating the phase of light at each subwavelength pixel location, enabling the shaping of the wavefront to direct light toward specific output zones. Spectral control is achieved by engineering the dispersion characteristics of the meta-units so that their phase response varies predictably with wavelength. This allows a single metasurface to perform different functions at different wavelengths, enabling wavelength-multiplexed recognition tasks. The spatial and spectral degrees of freedom are decoupled and independently optimized during the training process to maximize classification accuracy.

- define "coupled" and describe connection methods  
In this context, “coupled” refers to the optical interaction between two layers or components such that the output of one directly influences the input of another through wave propagation and interference. In the disclosed system, the input object is coupled to the metasurface via free-space diffraction, and the metasurface is coupled to the detection plane through far-field propagation. No physical connections, waveguides, or electrical links are required; coupling occurs entirely through the electromagnetic field. The strength of coupling is determined by the distance between layers and the numerical aperture of the system.

- describe system for recognizing targets  
The system for recognizing targets operates by encoding each class of object into a unique spatial intensity distribution on the output plane. During training, a library of input images is used to iteratively adjust the phase profile of the metasurface to maximize the intensity in the designated detection zone for each class while minimizing intensity elsewhere. Once trained, the metasurface functions as a fixed optical processor that, when presented with any input from the trained set, produces a detectable peak in the corresponding zone. Recognition is therefore a direct physical consequence of the metasurface’s engineered response, not a computational decision.

- introduce diffractive optical neural network (ONN)  
The disclosed system is a diffractive optical neural network, wherein the metasurface serves as the hidden layer of a neural network, the input object as the input layer, and the detection zones as the output layer. The network performs computation through the physical propagation of light, with each meta-unit acting as a trainable synaptic weight. The system emulates the linear transformation of a neural network through diffraction and the nonlinear activation through interference, achieving high classification accuracy without digital computation. The network is fully passive, operates at the speed of light, and requires no power beyond illumination.

- describe passive and power-free operation  
The system operates entirely without external power, relying solely on the energy of the incident light to perform inference. No electrical circuits, batteries, or active components are required. Once fabricated and calibrated, the metasurface performs recognition indefinitely without degradation, maintenance, or energy consumption. This makes the system ideal for deployment in remote, mobile, or energy-constrained environments where traditional digital systems are impractical.

- describe transparent substrate options  
The substrate supporting the metasurface may be composed of any transparent dielectric material with low optical loss at the operating wavelength, including silicon dioxide, fused silica, sapphire, or polymethyl methacrylate. The choice of substrate affects the mechanical stability, thermal expansion, and fabrication compatibility of the device. In some embodiments, flexible substrates are employed to enable conformal integration onto curved surfaces or wearable devices.

- introduce passive dielectric materials  
The meta-units are fabricated from passive dielectric materials such as amorphous silicon, titanium dioxide, or silicon nitride, which exhibit high refractive indices and negligible absorption in the near-infrared range. These materials enable large phase shifts with minimal loss, allowing the metasurface to achieve high efficiency and contrast. Unlike metallic structures, dielectric meta-units do not suffer from ohmic losses or plasmonic heating, ensuring long-term stability and reliability.

- describe actively tunable materials  
In alternative embodiments, the metasurface may incorporate actively tunable materials such as phase-change materials, liquid crystals, or electro-optic polymers to enable dynamic reconfiguration of the optical response. These materials allow the system to adapt to new classification tasks or environmental conditions without requiring physical replacement of the metasurface. Tuning is achieved through thermal, electrical, or optical stimuli, providing a hybrid architecture that retains the benefits of passive operation while enabling limited adaptability.

- describe patterning meta-units on substrate  
Meta-units are patterned onto the substrate using electron-beam lithography, nanoimprint lithography, or deep ultraviolet photolithography, followed by dry etching to transfer the pattern into the high-index material. The patterning process achieves subwavelength resolution with feature sizes below 200 nm, enabling precise control over the phase and polarization response of each meta-unit. The resulting structure is planar, scalable, and compatible with standard semiconductor fabrication techniques.

- introduce isotropic and birefringent libraries  
Two distinct libraries of meta-units are employed: isotropic meta-units, which provide identical phase modulation regardless of incident polarization, and birefringent meta-units, which induce different phase delays for orthogonal polarization states. The isotropic library enables single-polarization recognition tasks, while the birefringent library enables polarization-multiplexed operation, effectively doubling the classification capacity of a single metasurface by encoding separate tasks for horizontal and vertical polarization.

- describe controlling optical amplitude and phase  
Optical phase is controlled by varying the geometry of the meta-unit, such as its width, height, or rotational orientation, to alter the effective refractive index experienced by the incident light. Optical amplitude is controlled by introducing structural asymmetry or using materials with intrinsic absorption, though in the primary embodiment, amplitude modulation is minimized to preserve efficiency. In advanced embodiments, birefringent meta-units are engineered to simultaneously modulate both phase and amplitude by exploiting polarization-dependent scattering.

- describe engineering optical dispersion  
Optical dispersion is engineered by designing meta-units whose phase response varies systematically with wavelength. This is achieved by tuning the aspect ratio and cross-sectional shape of the meta-unit to create wavelength-dependent resonances. By selecting meta-units with complementary dispersion profiles, the metasurface can be made to perform different functions at different wavelengths, enabling spectral multiplexing and multi-spectral recognition.

- introduce output plane and detection zones  
The output plane is positioned at a distance determined by the diffraction properties of the metasurface and the wavelength of illumination. Detection zones are defined as circular or square regions on this plane, each corresponding to a specific object class. The size, shape, and spacing of these zones are optimized during training to maximize the contrast between the peak intensity of the correct class and the background intensity of competing classes.

- describe concentrating light intensity  
Light intensity is concentrated into the detection zones through constructive interference of wavelets scattered by the meta-units. The phase profile of the metasurface is trained to ensure that waves from all meta-units arrive in phase at the target zone while destructively interfering elsewhere. This results in a sharp intensity peak at the correct zone and minimal background illumination, enhancing classification confidence and robustness.

- describe modifying detection zone locations  
Detection zone locations may be modified by adjusting the propagation distance between the metasurface and the output plane or by altering the metasurface’s phase profile. This allows the system to be reconfigured for different detection geometries without changing the hardware. In some embodiments, multiple output planes are used to enable simultaneous recognition across different spatial configurations.

- describe recognizing objects by processing light waves  
Objects are recognized by the physical transformation of their scattered light waves into a detectable intensity signature. The metasurface does not store or digitize the image; instead, it acts as a fixed optical filter that maps each input class to a unique spatial pattern. Recognition is therefore a direct consequence of the physical laws governing wave propagation and interference, not a computational algorithm.

- describe forming diffractive ONN  
The diffractive optical neural network is formed by training the phase profile of the metasurface using a loss function based on the cross-entropy between the simulated and desired intensity distributions. The training process involves numerically propagating thousands of input images through the system and iteratively adjusting the phase of each meta-unit to minimize the error. Once trained, the metasurface is physically fabricated to match the optimized profile, resulting in a hardware-implemented neural network.

- describe recognizing hand-written digits and letters  
The system successfully recognizes handwritten digits from the MNIST dataset with accuracies exceeding 99% for four-class classification and approximately 80% for ten-class classification. Similarly, typed letters and typographic styles are distinguished using polarization multiplexing, achieving over 90% accuracy for both letter identity and font style. These results demonstrate the system’s capability to resolve fine spatial details and complex patterns using a single passive layer.

- describe recognizing facial photos  
Facial verification is achieved using a double-layered metasurface that maps grayscale facial images into a 3×3 intensity array on the output plane. The similarity between two facial images is determined by computing the Euclidean distance between their respective intensity arrays. When the distance falls below a threshold, the images are classified as matching. This system achieves approximately 80% verification accuracy, comparable to digital neural networks with three convolutional layers, without any digital computation.

- introduce double-layered metasurface  
A double-layered metasurface consists of two stacked metasurfaces separated by a controlled distance, enabling enhanced expressive power by allowing multiple stages of wavefront modulation. The first layer performs an initial transformation of the input wavefront, and the second layer further refines the distribution to produce a more discriminative output. This architecture enables the recognition of complex, high-dimensional inputs such as grayscale facial images, which cannot be accurately classified by a single layer.

- describe handling tasks with enhanced expressive power  
Tasks requiring higher expressive power, such as facial verification or classification of incoherent images, are handled by increasing the depth of the network through multiple metasurface layers or by increasing the width through polarization or wavelength multiplexing. These strategies enable the system to encode more complex mappings between input and output, approaching the performance of deep digital networks while retaining the advantages of optical processing.

- describe translating images into low-dimensional representations  
The metasurface transforms high-dimensional optical images into low-dimensional intensity patterns on the output plane. For facial images, this results in a 3×3 array of intensity values that encapsulates the essential features of the face. This representation is sufficient to distinguish between individuals based on the relative spatial distribution of intensity, enabling accurate verification without digitizing or storing the original image.

- describe evaluating similarity between images  
Similarity between two images is evaluated by computing the Euclidean distance between their respective low-dimensional intensity representations. If the distance is below a predetermined threshold, the images are classified as belonging to the same subject. This method is robust to variations in lighting, pose, or partial occlusion, as the metasurface is trained to focus on invariant features.

- introduce additional metasurfaces  
Additional metasurfaces may be introduced in parallel or in series to increase the system’s capacity. A parallel array of metasurfaces, each trained for a different task, enables simultaneous recognition of multiple object types. A series of metasurfaces increases the depth of the network, enabling recognition of more complex patterns.

- describe bypassing digitalization for security  
By processing light directly without digitization, the system bypasses the digital domain entirely, ensuring that no image data is stored, transmitted, or accessible to external systems. This provides physics-based security, as the input object cannot be reconstructed or intercepted during processing, making the system inherently resistant to hacking, surveillance, or data leakage.

- describe method for processing light  
The method for processing light involves illuminating an object with coherent light, capturing the scattered wavefront, modulating it with a pre-trained metasurface, and detecting the resulting intensity distribution on a fixed output plane. The classification is determined by identifying the zone with maximum intensity. The entire process occurs in a single optical propagation step, with no intermediate computation or storage.

- describe training diffractive neural network  
Training is performed numerically using the Rayleigh-Sommerfeld diffraction theory to simulate light propagation through the system. A loss function is defined as the cross-entropy between the simulated intensity distribution and a target distribution, where only the designated detection zone is assigned a value of one and all others zero. The phase of each meta-unit is iteratively adjusted using an Adam optimization algorithm to minimize this loss over thousands of training examples.

- describe iterative training process  
The iterative training process begins with a random initialization of the metasurface phase profile. A batch of input images is propagated through the simulated system, and the resulting intensity distribution is compared to the target. The phase of each meta-unit is updated based on the gradient of the loss function. This process is repeated over hundreds of epochs until convergence is achieved, resulting in a phase profile that maximizes classification accuracy.

- describe improving robustness against experimental errors  
Robustness is improved by incorporating simulated experimental errors into the training process, including random misalignment of components, non-uniform illumination, and variations in object and imaging distances. An auxiliary term is added to the loss function to penalize low contrast between detection zones, encouraging the system to generate high-intensity peaks with minimal background.

- describe converting light into optical barcode  
The output intensity distribution on the detection plane serves as an optical barcode, where the spatial pattern of intensity values uniquely identifies the input object. This barcode is generated directly by the metasurface and is not digitized, ensuring that the original image remains physically unrecoverable.

- describe choosing configuration for target recognition accuracy  
The configuration of the metasurface—including the number of meta-units, their symmetry, the substrate thickness, and the detection zone layout—is chosen to maximize recognition accuracy. This is determined through numerical optimization, where each parameter is varied and the resulting accuracy is evaluated. The optimal configuration balances expressive power, fabrication feasibility, and robustness to experimental noise.

### EXAMPLES

- introduce metasurface smart glass for object recognition  
A metasurface smart glass is disclosed as a passive, optical neural network device capable of recognizing objects through direct light wave manipulation. The device is fabricated as a thin, transparent glass pane embedded with a metasurface layer, enabling integration into windows, displays, or wearable optics without altering the visual appearance of the substrate.

- describe metasurface composition and functionality  
The metasurface is composed of amorphous silicon meta-units arranged in a square lattice on a silicon dioxide substrate. Each meta-unit is 1 micrometer in height and 750 nanometers in pitch, providing subwavelength phase control. The functionality of the metasurface is to transform the spatial distribution of scattered light from an object into a detectable intensity pattern on a downstream plane, enabling classification without digital processing.

- explain optical wavefront processing by metasurface  
The metasurface processes the optical wavefront by imposing a spatially varying phase delay that causes constructive interference in a designated detection zone and destructive interference elsewhere. This modulation is pre-engineered to correspond to a specific object class, allowing the system to recognize the object by the location of the brightest spot on the output plane.

- illustrate object recognition mechanism  
When a handwritten digit is illuminated, the scattered light passes through the metasurface, which reshapes the wavefront such that the intensity is concentrated in a single detection zone corresponding to the digit’s identity. The detection zone with the highest intensity is then identified by a photodetector array, yielding the classification result.

- describe training process for metasurface smart glass  
The training process involves numerically simulating the propagation of thousands of digit images through the metasurface and iteratively adjusting the phase of each meta-unit to maximize the intensity in the correct detection zone. The optimization is performed using a stochastic gradient descent algorithm with a cross-entropy loss function.

- introduce experimental setup for object recognition  
The experimental setup includes a 1,550 nm telecom laser, a photomask with printed digits, a telescope for relay imaging, a polarizer, the metasurface smart glass, and an InGaAs camera to capture the output intensity distribution. The input object is translated into position using a motorized stage to ensure consistent illumination.

- describe photomask creation for input objects  
Photomasks are created by photo-plotting opaque black emulsion on mylar sheets to form binary patterns of digits and letters. Each object is 0.775 mm square, with transparent regions corresponding to the digit shape and opaque regions elsewhere, ensuring high contrast under illumination.

- explain optical setup for object recognition  
The optical setup aligns the laser beam to illuminate the photomask, followed by a unity-magnification telescope that relays the object image onto a square aperture to block stray light. The filtered wavefront then propagates to the metasurface and finally to the detection plane, where the intensity pattern is captured by a camera.

- describe metasurface fabrication process  
The metasurface is fabricated using electron-beam lithography to define the meta-unit pattern on a silicon dioxide substrate, followed by plasma etching to transfer the pattern into a 1-micrometer-thick layer of amorphous silicon. The resulting structure is planar, durable, and compatible with mass production.

- show phase responses of meta-unit libraries  
The isotropic meta-unit library exhibits a continuous phase response ranging from 0 to 2π, while the birefringent library provides two distinct phase curves for orthogonal polarizations, enabling independent modulation for horizontal and vertical light states.

- test polarization-multiplexing smart glasses  
Polarization-multiplexing smart glasses are tested by inserting a linear polarizer before the photomask and measuring recognition accuracy for two orthogonal polarization states. Results show that the system can simultaneously classify digits for one polarization and letters for the other, achieving over 90% accuracy for both tasks.

- recognize hand-written digits using single-layered metasurface  
A single-layered metasurface successfully recognizes four classes of handwritten digits with 99.14% accuracy and ten classes with 78.37% accuracy, demonstrating the capability of a passive optical system to resolve fine-grained visual patterns.

- show trained phase modulation for digit recognition  
The trained phase modulation reveals a complex, non-uniform distribution of phase shifts across the metasurface, with regions of high gradient corresponding to edges and contours in the digit patterns, indicating that the system learns to enhance spatial features through interference.

- implement metasurface using optically isotropic meta-units  
The metasurface is implemented using isotropic meta-units, which provide phase modulation independent of polarization. This simplifies the optical setup and enables recognition using unpolarized or randomly polarized light.

- show examples of digit classification  
Examples show that when a digit “4” is presented, the intensity peak appears in the zone designated for “4,” while other zones remain dark. Similar results are observed for digits “0,” “1,” and “3,” confirming consistent classification.

- compare experimental and theoretical results for digit recognition  
Experimental results closely match theoretical predictions, with recognition accuracy differing by less than 1% for four-class tasks and within 8% for ten-class tasks, indicating that the training process effectively accounts for fabrication imperfections.

- recognize 10 classes of hand-written digits using single-layered metasurface  
A single-layered metasurface is trained to recognize all ten digits of the MNIST dataset, with detection zones arranged in a circular array. The system achieves 78.37% accuracy, demonstrating the feasibility of high-capacity classification with a single optical layer.

- show trained phase modulation for 10-class digit recognition  
The phase modulation for ten-class recognition exhibits a more complex spatial structure than the four-class case, with increased phase gradients and finer-scale variations to distinguish between similar digit shapes.

- implement metasurface using optically isotropic meta-units  
The same isotropic meta-unit library is used, confirming that high-dimensional classification is possible without polarization control, though with reduced robustness due to lower inter-zone contrast.

- show examples of 10-class digit classification  
Examples show that while most digits are correctly classified, some misclassifications occur between visually similar digits such as “3” and “8,” consistent with the reduced contrast between detection zones.

- compare experimental and theoretical results for 10-class digit recognition  
The experimental accuracy of 78.37% is lower than the theoretical prediction of 86.50%, primarily due to reduced inter-zone intensity contrast and experimental misalignment, highlighting the need for robust training methods.

- recognize hand-written digits using polarization-multiplexing smart glasses  
By employing birefringent meta-units and orthogonal polarizations, the system divides the ten-digit task into two five-digit subtasks, achieving 90.99% and 81.44% accuracy for each group, significantly outperforming the non-polarization-multiplexed version.

- show trained phase modulation for polarization-multiplexing digit recognition  
The trained phase modulation for each polarization state is distinct, with one pattern optimized for digits {1,3,4,7,8} and another for {0,2,5,6,9}, demonstrating independent control of optical response.

- implement metasurface using birefringent meta-units  
The metasurface is fabricated using birefringent meta-units with two-fold symmetry, enabling independent phase control for horizontal and vertical polarization states.

- show examples of polarization-multiplexing digit classification  
When illuminated with horizontal polarization, the system correctly identifies digits from the first group; with vertical polarization, it identifies digits from the second group, confirming dual-task capability.

- compare experimental and theoretical results for polarization-multiplexing digit recognition  
Experimental results closely align with theoretical predictions, with accuracies within 4% of simulated values, indicating that polarization multiplexing improves robustness and reduces classification complexity.

- recognize typed alphabetical letters and typographic styles using polarization-multiplexing smart glasses  
The system distinguishes between four letters {A,B,C,D} under horizontal polarization and between normal and italic styles under vertical polarization, achieving 92.81% and 100% accuracy, respectively.

- show trained phase modulation for letter and style recognition  
The phase modulation for letter recognition exhibits distinct patterns for each character, while the style modulation shows a binary response, with one pattern activating for normal and another for italic.

- implement metasurface using birefringent meta-units  
The same birefringent library is used, demonstrating that polarization multiplexing enables multitasking without increasing device size or complexity.

- show examples of letter and style classification  
Examples show that the letter “A” in normal font activates one zone, while “A” in italic activates another, confirming simultaneous recognition of identity and style.

- compare experimental and theoretical results for letter and style recognition  
Experimental results match theoretical predictions within 2%, indicating high fidelity in the training process and robustness to fabrication variations.

- introduce facial verification using double-layered metasurface smart glass  
A double-layered metasurface is employed to perform facial verification by mapping grayscale facial images into a 3×3 intensity array, enabling comparison of two images via Euclidean distance.

- describe working mechanism of metasurface doublet  
The first metasurface performs an initial transformation of the facial image, and the second further refines the distribution to produce a low-dimensional representation. The resulting intensity array encodes the essential features of the face, allowing similarity to be evaluated without storing the original image.

- show example human face images used in training and testing  
Training images include 14 distinct photos of 90 individuals, while testing includes 14 photos of 10 unseen individuals. Images vary in lighting, pose, and expression to ensure generalization.

- show trained phase modulations of metasurface doublet  
The trained phase modulations reveal intricate spatial patterns that emphasize features such as eye spacing, nose shape, and jawline, indicating that the system learns physiologically relevant descriptors.

- evaluate facial verification accuracy  
The system achieves 80% verification accuracy, with false acceptance and false rejection rates each at approximately 10%, comparable to a digital neural network with three convolutional layers.

- compare results with digital ANN  
The metasurface doublet matches the accuracy of a digital ANN while consuming no power, operating at the speed of light, and preserving privacy by avoiding digital storage of facial data.

- show examples of facial verification  
Examples show correct verification when two images of the same person are presented and rejection when images of different individuals are compared, demonstrating reliable performance.

- describe limitations of single-layered metasurface smart glasses  
Single-layered metasurfaces are insufficient for grayscale image recognition due to limited expressive power, as they cannot resolve the fine intensity variations present in facial images.

- discuss advantages of metasurface smart glasses over digital ANNs  
Metasurface smart glasses offer zero power consumption, ultrafast operation, inherent security, and compact form factor. Unlike digital ANNs, they do not require memory, cooling, or software updates, and they cannot be hacked through data extraction.

- discuss potential applications of metasurface smart glasses  
Potential applications include secure facial recognition in smartphones, privacy-preserving surveillance, autonomous vehicle sensors, wearable health monitors, and military-grade optical authentication systems.

- describe future directions for metasurface smart glass development  
Future work will explore multi-wavelength operation, active tunability, and integration with photodetector arrays to enable full-spectrum recognition. Scaling to thousands of classes and incoherent image recognition is also under investigation.

- discuss challenges in scaling up metasurface smart glasses  
Scaling faces challenges in fabrication precision, training complexity, and alignment tolerance. Larger metasurfaces require more meta-units, increasing computational cost and sensitivity to manufacturing defects.

- describe potential solutions to scaling challenges  
Solutions include hierarchical training, modular metasurface arrays, and machine learning-guided design optimization. Parallel fabrication techniques and self-alignment mechanisms may also mitigate scaling issues.

- summarize benefits of metasurface smart glasses  
Metasurface smart glasses provide a passive, secure, ultrafast, and energy-efficient alternative to digital neural networks, enabling real-time object recognition without computation, storage, or power.

- conclude with potential impact of metasurface smart glasses  
The disclosed technology has the potential to revolutionize edge computing by replacing power-hungry digital processors with passive optical devices, enabling ubiquitous, secure, and sustainable perception systems in the Internet of Things, healthcare, and defense sectors.

- introduce metasurface ONN robustness  
The robustness of the metasurface optical neural network is quantified by the intensity contrast between the highest and second-highest detection zones. Higher contrast correlates with greater resilience to experimental noise and misalignment.

- motivate inter-zone contrast  
Inter-zone contrast is motivated as a critical design metric, as it determines the system’s ability to discriminate between classes under real-world conditions. Training with contrast-enhancing terms improves experimental accuracy.

- describe ONN expressive power  
Expressive power is determined by the number of meta-units, their geometric diversity, and the ability to modulate multiple optical properties. Increasing width through polarization or wavelength multiplexing enhances expressive power without increasing depth.

- introduce width and depth increase  
Width is increased by employing polarization or wavelength multiplexing, while depth is increased by cascading multiple metasurfaces. Both strategies enable recognition of more complex and diverse object classes.

- describe polarization multiplexing  
Polarization multiplexing doubles the classification capacity of a single metasurface by encoding separate tasks for orthogonal polarization states, enabling multitasking without increasing device size.

- describe phase-amplitude metasurface holograms  
Phase-amplitude metasurface holograms enable independent control of both phase and amplitude, increasing the dimensionality of the optical response and enabling more complex mappings between input and output.

- describe wavelength-multiplexing  
Wavelength-multiplexing introduces an additional degree of freedom by designing meta-units whose phase response varies with wavelength, allowing a single metasurface to perform different functions at different wavelengths.

- describe array of distinct metasurfaces  
An array of distinct metasurfaces, each trained for a different task, can be arranged in parallel to enable simultaneous recognition of multiple object types, significantly expanding system capacity.

- summarize ONN advantages  
The disclosed optical neural network offers unparalleled advantages in speed, energy efficiency, security, and compactness, making it ideal for applications where digital systems are impractical or undesirable.

- introduce advanced sensors  
Advanced sensors require minimal power, high reliability, and immunity to cyber threats. The disclosed metasurface ONN meets all these criteria, positioning it as a next-generation sensing platform.

- describe ONN compactness  
The entire system, including illumination and detection, can be integrated into a device smaller than a credit card, enabling deployment in mobile, wearable, and implantable applications.

- describe ONN energy efficiency  
The system consumes no power during inference, relying solely on incident light. This makes it the most energy-efficient recognition system known.

- describe ONN computing speed  
Computation occurs at the speed of light, with inference times on the order of picoseconds, enabling real-time processing of high-frame-rate video streams.

- describe ONN accuracy  
The system achieves classification accuracies exceeding 99% for simple tasks and 80% for complex tasks such as facial verification, matching or surpassing digital neural networks.

- describe ONN data security  
Data security is guaranteed because no digital representation of the input is ever created, stored, or transmitted. The original image remains physically inaccessible.

- introduce digital ANN limitations  
Digital ANNs require significant power, generate heat, are vulnerable to cyberattack, and introduce latency due to data movement between memory and processor.

- describe ONN advantages over digital ANNs  
ONNs eliminate the need for digitization, reduce power consumption to zero, eliminate latency, and provide physics-based security, offering a superior alternative for edge perception.

- describe ONN design process  
The design process involves numerical simulation of light propagation, optimization of phase profiles using gradient descent, and physical fabrication of the metasurface to match the optimized design.

- describe forward calculation process  
Forward calculation involves numerically propagating the input wavefront through the metasurface using the Rayleigh-Sommerfeld diffraction formula to compute the output intensity distribution.

- describe backward calculation process  
Backward calculation computes the gradient of the loss function with respect to each meta-unit’s phase, enabling iterative optimization using the Adam algorithm.

- describe strategies for robustness  
Strategies include incorporating experimental noise into training, maximizing inter-zone contrast, and using multiple meta-unit libraries to improve fabrication tolerance.

- introduce experimental results  
Experimental results confirm that the system achieves high accuracy across multiple tasks, with performance closely matching theoretical predictions.

- describe recognition of handwritten digits  
Recognition of handwritten digits achieves 99.14% accuracy for four classes and 78.37% for ten classes, validating the system’s capability for fine-grained visual discrimination.

- describe metasurface device fabrication  
Fabrication is performed using electron-beam lithography and plasma etching, producing high-fidelity metasurfaces with subwavelength precision and excellent reproducibility.

- describe recognition results  
Recognition results demonstrate consistent classification across diverse lighting, alignment, and object variations, confirming system robustness.

- describe polarization multiplexing example  
Polarization multiplexing enables simultaneous recognition of digits and letters, achieving over 90% accuracy for both tasks using a single metasurface.

- describe handwritten digit classification task  
The handwritten digit classification task involves distinguishing between ten numerical digits using a single-layered metasurface, with detection zones arranged in a circular pattern.

- describe human facial verification task  
The facial verification task involves determining whether two grayscale facial images belong to the same individual, using a double-layered metasurface to generate a low-dimensional intensity representation.

- describe error rate diagram  
Error rate diagrams show that accuracy improves with increasing inter-zone contrast and that the system remains robust even under significant misalignment.

- introduce facial image verification dataset  
The dataset consists of 1,400 facial images from 100 individuals, with 14 images per person, used for training and testing the metasurface doublet.

- describe ONN design parameters  
Design parameters include metasurface dimensions, meta-unit geometry, substrate thickness, propagation distances, and detection zone layout, all optimized for maximum accuracy.

- describe metasurface doublet design  
The doublet consists of two metasurfaces separated by a 500-micrometer gap, with each layer trained to refine the intensity distribution for facial feature extraction.

- describe optical barcode size  
The optical barcode size is defined as the number of detection zones, with a 3×3 array used for facial verification, providing sufficient dimensionality to distinguish between individuals.

- describe loss function design  
The loss function combines cross-entropy minimization with a contrast-enhancing term to encourage high-intensity peaks and low background, improving robustness.

- describe partial facial coverage effect  
Partial facial coverage, such as occlusion by glasses or masks, reduces verification accuracy, but the system remains functional, indicating tolerance to real-world variations.

- describe facial photo examples  
Examples show correct verification even when images differ in lighting or expression, demonstrating the system’s ability to extract invariant features.

- describe verification accuracy results  
Verification accuracy reaches 80%, with false acceptance and rejection rates each at 10%, matching the performance of a digital ANN.

- describe separation distance effect  
Increasing the separation distance between the metasurface and detection plane reduces resolution and accuracy, while decreasing it increases crosstalk, indicating an optimal distance exists.

- describe relative size effect  
The relative size of the input object affects recognition accuracy, with optimal performance achieved when the object fills approximately 80% of the metasurface area.

- describe substrate thickness effect  
Substrate thickness affects the phase delay and propagation characteristics; a thickness of 500 micrometers is found to be optimal for the 1,550 nm wavelength.

- introduce one-layer metasurface ONN  
A one-layer metasurface ONN is capable of recognizing binary images with high accuracy but fails to resolve grayscale facial images due to limited expressive power.

- describe design and performance of one-layer metasurface ONN  
The one-layer design achieves 99% accuracy for digit recognition but only 60% for facial verification, demonstrating the need for increased depth.

- illustrate verification process with examples  
Examples show that the one-layer system cannot distinguish between similar faces, producing ambiguous intensity distributions.

- explain training process and optimization of metasurface design  
Training involves iteratively adjusting the phase profile using gradient descent, with optimization constrained by fabrication limits and physical constraints.

- define barcode calculation and Euclidean distance  
The barcode is the 3×3 intensity array generated by the metasurface. Euclidean distance is computed between two barcodes to determine similarity, with a threshold of 0.8 used for classification.

- introduce loss function and its design  
The loss function is designed to minimize cross-entropy while maximizing inter-zone contrast, ensuring both accuracy and robustness.

- discuss performance of one-layer metasurface ONN  
The one-layer system performs excellently on simple tasks but is inadequate for complex recognition, motivating the use of multi-layer architectures.

- introduce metasurface doublet ONN  
The metasurface doublet ONN consists of two stacked metasurfaces, enabling high-dimensional feature extraction and accurate facial verification.

- describe design and performance of metasurface doublet ONN  
The doublet achieves 80% verification accuracy, matching digital ANNs, while operating passively and without power.

- illustrate verification process with examples  
Examples show correct identification of matching faces and rejection of non-matching pairs, even under varying illumination and pose.

- compare performance of one-layer and metasurface doublet ONNs  
The doublet outperforms the one-layer system by 20 percentage points in facial verification, demonstrating the value of increased depth.

- discuss robustness of metasurface doublet design against experimental errors  
The doublet remains robust to misalignment, illumination variation, and fabrication imperfections, thanks to contrast-enhancing training.

- discuss robustness of metasurface doublet design against wavelength variation  
The system maintains accuracy across a ±50 nm wavelength range, indicating tolerance to source drift.

- introduce concept of optimal barcode size  
Optimal barcode size balances information capacity with noise sensitivity. A 3×3 array is found to be optimal for facial verification.

- investigate relationship between verification accuracy and barcode size  
Accuracy increases with barcode size up to 3×3, beyond which diminishing returns and increased crosstalk reduce performance.

- show error rate diagrams for different barcode sizes  
Error rate diagrams show a minimum error at 3×3, confirming the optimal size.

- discuss optimal barcode size for one-layer ONN  
For one-layer ONNs, a 2×2 barcode is optimal due to limited expressive power.

- show facial verification examples for one-layer ONNs  
Examples show frequent misclassification, confirming the inadequacy of single-layer systems for complex tasks.

- assess ONNs based on metasurface doublets  
ONNs based on metasurface doublets provide the best balance of accuracy, robustness, and scalability for high-dimensional recognition tasks.

- discuss optimal barcode size for metasurface doublet ONNs  
For metasurface doublets, a 3×3 barcode is optimal, providing sufficient dimensionality without introducing noise.

- show error rate diagrams for different barcode sizes  
Diagrams confirm that 3×3 yields the lowest error rate for doublet systems.

- discuss performance of metasurface doublet ONNs  
Performance is comparable to digital ANNs, with the added benefits of zero power, no digital storage, and physics-based security.

- show facial verification examples for metasurface doublet ONNs  
Examples show consistent correct classification across diverse lighting, pose, and expression conditions.

- summarize results of one-layer and metasurface doublet ONNs  
One-layer systems excel at simple tasks; doublets enable complex recognition, with optimal performance at 3×3 barcode size.

- conclude with optimal barcode sizes for both designs  
The optimal barcode size is 2×2 for one-layer ONNs and 3×3 for metasurface doublets, providing maximum accuracy with minimal complexity.

- introduce optical neural networks (ONNs)  
Optical neural networks are physical systems that perform neural computation using the laws of wave optics, eliminating the need for digital electronics.

- describe ONN design with metasurface  
The ONN design employs a metasurface as the core computational element, with meta-units engineered to sculpt light waves into classification signatures.

- explain optical scattering pattern concentration  
Optical scattering patterns are concentrated into detection zones through constructive interference, ensuring high signal-to-noise ratio and reliable classification.

- define loss function for training ONN  
The loss function is defined as the cross-entropy between the simulated output intensity distribution and the target distribution, augmented by a contrast term.

- show results of ONN with different weights  
Results show that increasing the weight of the contrast term improves experimental accuracy, validating the training strategy.

- compare ONN designs with different degrees of concentration  
Designs with higher concentration exhibit lower error rates and greater robustness to experimental noise.

- discuss effect of partial facial coverage on verification accuracy  
Partial coverage reduces accuracy but does not eliminate it, indicating that the system extracts invariant facial features.

- describe ONN design for verifying photos with partial facial coverage  
The ONN is trained using images with simulated occlusions, enabling it to recognize faces even when partially covered.

- show error rate diagram for ONN with partial facial coverage  
The error rate increases modestly with coverage, remaining below 20% even with 50% occlusion.

- show trained phase distributions for ONN with partial facial coverage  
Trained phase distributions emphasize features such as eyes and mouth, which remain visible under occlusion.

- show examples of correct facial verification using ONN  
Examples demonstrate successful verification even when subjects wear glasses or masks.

- describe ONN design for verifying facial photos without facial cover  
The design is identical but trained on unobstructed images, achieving higher accuracy under ideal conditions.

- show error rate diagram for ONN without facial cover  
Error rates drop to below 10% when no occlusion is present.

- show trained phase distributions for ONN without facial cover  
Phase distributions are more uniform, capturing full facial geometry.

- discuss classification of optically incoherent images  
Classification of incoherent images remains challenging due to the absence of interference, but is enabled by arrays of metasurfaces.

- describe ONN using parallel array of metasurfaces  
A parallel array of metasurfaces, each trained for a different feature, can classify incoherent images by combining their outputs.

- show design for classifying incoherent images  
The design uses ten metasurfaces, each sensitive to a different spatial frequency, to reconstruct the image in the optical domain.

- show amplitude masks of metasurfaces for recognizing handwritten digits  
Amplitude masks are engineered to respond to edge patterns, enabling recognition even without coherent illumination.

- show confusion matrix for recognizing handwritten digits  
The confusion matrix shows minimal misclassification, confirming high fidelity.

- describe generation of optically coherent facial images  
Coherent facial images are generated using a spatial light modulator or a liquid crystal display to project phase-modulated light.

- show transparency-based method for generating coherent images  
A transparent LCD is used to modulate the phase of incident light, creating a coherent representation of the face.

- show LCD-based method for generating coherent images  
An LCD array is illuminated by a laser, and its pixel values are set to reproduce the phase profile of a facial image.

- show optical setup for ONN  
The optical setup includes a laser, a photomask, a telescope, a metasurface, and a camera, all aligned on an optical bench.

- show scanning electron micrographs of fabricated metasurfaces  
Micrographs confirm subwavelength feature sizes and high fabrication fidelity.

- describe advantages of ONNs over digital ANNs  
ONNs offer zero power, instant inference, inherent security, and immunity to cyberattack, making them superior for edge applications.

- estimate computing time of ONN  
Computing time is estimated at 1 picosecond per inference, limited only by the speed of light.

- estimate power consumption of ONN  
Power consumption is zero during inference, with only the illumination source requiring power, which is negligible.

- describe system for developing and validating ONN designs  
The system includes a numerical simulator, an optimization engine, and a fabrication pipeline, enabling rapid design-to-deployment cycles.

- discuss accuracy of ONNs for verifying optically coherent facial images  
Accuracy reaches 80%, matching digital ANNs, while preserving privacy.

- discuss strategy for classifying optically incoherent images  
The strategy involves using an array of metasurfaces, each sensitive to different spatial features, to reconstruct the image in the optical domain.

- describe benefits of ONNs over digital ANNs  
ONNs eliminate power consumption, latency, and digital vulnerability, offering a fundamentally new paradigm for perception.

- discuss scope of disclosed subject matter  
The disclosed subject matter encompasses all optical neural networks based on metasurfaces, including variations in material, geometry, polarization, wavelength, and layer configuration, as well as all methods for training and fabricating such systems.