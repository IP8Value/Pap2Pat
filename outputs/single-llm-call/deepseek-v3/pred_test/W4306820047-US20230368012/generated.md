Here is the patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND  

The field of object recognition has become increasingly important across numerous applications including image annotation, vehicle tracking, pedestrian detection, and facial recognition. Conventional approaches utilize digital images captured by cameras and videos, processing these through machine learning models to translate high-dimensional visual signals into lower-dimensional representations. This conventional technology stack requires a compound optical system to form images, an optoelectronic sensor for analog-to-digital conversion, and digital processors to implement artificial neural networks (ANNs).  

However, these conventional systems suffer from significant limitations. They tend to be bulky and power-hungry, exhibit slow reaction times due to latency between technology modules, and are vulnerable to cyber-attacks. These problems are exacerbated as the demand for high power efficiency, computational speed, and data security increases with the explosion of data volume and the widespread availability of mobile devices with computer vision capabilities.  

Optical neural networks (ONNs) have emerged as a promising physical platform for object recognition. ONNs utilize photonic elements and circuits to form a layered architecture that emulates digital ANNs, directly processing optical signals from target objects. In an ideal ONN, optical signals are manipulated by layers of elements that perform linear transformations and nonlinear activations, pre-trained to enable the network to perform specific computing tasks. ONNs offer computational speeds characterized by the propagation speed of light and can operate entirely passively, requiring no additional power after the optical input is generated. Furthermore, computational algorithms are hard-coded into the intrinsic and engineered materials of ONNs, ensuring data security.  

Despite their potential, existing ONNs face challenges. Traditional layered architectures based on integrated photonic circuits, such as those using tunable Mach-Zehnder interferometers (MZIs), are limited by the large size of MZIs relative to optical wavelengths, which restricts their expressive power and ability to handle complex tasks. Diffractive ONNs based on metamaterials have been proposed, utilizing wave dynamics to perform artificial neural computing. However, current manufacturing techniques limit these to two-dimensional versions without nonlinear inclusions, significantly reducing their expressive power.  

## SUMMARY  

The present invention provides a target recognition system comprising a diffractive optical neural network (ONN) based on metasurfaces, referred to herein as a "metasurface smart glass." This system directly processes light waves scattered by an object using internal nanostructures, enabling high-speed, power-free, and secure object recognition.  

The system includes a metasurface composed of an array of subwavelength meta-units that collectively manipulate the phase, amplitude, and polarization of incident light. The metasurface is fabricated using CMOS-compatible nanofabrication techniques, allowing for miniaturization and integration into compact devices. The system operates in the optical spectral range, leveraging readily available light sources and detectors.  

The metasurface smart glass processes light waves through spatial and spectral control, enabling the recognition of various target types, including hand-written digits, alphabetical letters, and facial images. The system operates within a wavelength range of approximately 1,550 nm, though other wavelengths may be utilized depending on the application. The system is entirely passive, requiring no power supply or digital processor, and operates at the speed of light.  

The meta-units are constructed from passive dielectric materials, such as amorphous silicon, and may also incorporate actively tunable materials for enhanced functionality. These meta-units are patterned on a transparent substrate, such as silicon dioxide, and are arranged in isotropic or birefringent configurations to control optical amplitude and phase. The system includes an output plane with predefined detection zones, where light intensity is concentrated to identify target objects.  

Target recognition is achieved by training the diffractive ONN to maximize light intensity within specific detection zones corresponding to the classification labels of input objects. The training process involves iterative optimization of the phase profile of the metasurface using a loss function that evaluates the cross-entropy between calculated and target intensity distributions. The system is robust against experimental errors, such as non-uniform illumination and mispositioning of components, through measures incorporated during training.  

## DETAILED DESCRIPTION  

### Definitions  

For the purposes of this patent application, the following terms are defined as follows:  

- "About" or "approximately" refers to a range of values within ±10% of the stated value, unless otherwise specified.  
- "Coupled" refers to a physical or functional connection between components, which may be direct or indirect and may include intermediate elements.  

### System for Processing Light  

The invention comprises a system for processing light waves to recognize target objects. The system includes a metasurface composed of an array of meta-units, each designed to manipulate the phase, amplitude, and polarization of incident light. The metasurface is fabricated on a transparent substrate, such as silicon dioxide, and operates in the near-infrared spectrum (e.g., λ = 1,550 nm).  

The meta-units are arranged in a square lattice with a periodicity of 750 nm and have a height of 1 µm. They are constructed from materials with low extinction coefficients in the near-infrared, such as amorphous silicon, to minimize optical losses. The meta-units may be optically isotropic or birefringent, depending on the desired functionality.  

### Spatial and Spectral Control of Light  

The metasurface provides precise spatial and spectral control of light by modulating the phase, amplitude, and polarization of the optical wavefront. This modulation is achieved through the collective response of millions of subwavelength meta-units, enabling efficient parallel computing with high expressive power. The metasurface can be trained to perform specific tasks, such as object recognition, by optimizing the phase profile of the meta-units.  

The system includes an output plane with predefined detection zones, where light intensity is concentrated to identify target objects. The detection zones are arranged in specific patterns, such as square or circular arrays, depending on the classification task. The system maximizes light intensity within the detection zone corresponding to the correct identity of the input object, ensuring robust recognition.  

### System for Recognizing Targets  

The invention further comprises a system for recognizing targets using a diffractive ONN. The ONN is formed by the metasurface, which acts as a hidden layer in the neural network architecture. The input layer consists of the target object, and the output layer is the detection plane. Each pixel in these layers represents an artificial neuron, connected via optical interference.  

The ONN operates passively and power-free, processing optical signals at the speed of light. It is trained using a dataset of optically coherent images, such as hand-written digits or facial photos, and optimizes the phase modulation of the metasurface to maximize recognition accuracy. The training process includes measures to improve robustness against experimental errors, such as non-uniform illumination and component mispositioning.  

### Passive and Power-Free Operation  

The metasurface smart glass operates entirely passively, requiring no additional power after the optical input is generated. This is achieved through the use of passive dielectric materials and the absence of electronic components. The system is thus highly energy-efficient and suitable for applications where power consumption is a critical concern.  

### Transparent Substrate Options  

The metasurface is fabricated on a transparent substrate, such as silicon dioxide, with a refractive index of 1.44. The substrate thickness is approximately 500 µm (322.58 λ), ensuring mechanical stability while minimizing optical aberrations. Other substrate materials, such as glass or polymers, may also be used depending on the application.  

### Passive Dielectric Materials  

The meta-units are constructed from passive dielectric materials, such as amorphous silicon, which exhibit low optical losses in the near-infrared spectrum. These materials enable precise control of the optical wavefront without the need for active tuning.  

### Actively Tunable Materials  

In some embodiments, the meta-units may incorporate actively tunable materials, such as liquid crystals or phase-change materials, to enable dynamic control of the optical response. These materials can be tuned using external stimuli, such as electric fields or temperature changes, to adapt the metasurface for different tasks or conditions.  

### Patterning Meta-Units on Substrate  

The meta-units are patterned on the substrate using CMOS-compatible nanofabrication techniques, such as electron-beam lithography or nanoimprinting. The patterning process ensures high precision and uniformity, enabling the metasurface to perform complex optical transformations.  

### Isotropic and Birefringent Libraries  

The meta-units may be designed from isotropic or birefringent libraries, depending on the desired functionality. Isotropic meta-units have a cross-section with four-fold symmetry, providing uniform phase modulation for all polarization states. Birefringent meta-units have a cross-section with two-fold symmetry, enabling distinct phase modulation for orthogonal polarization states.  

### Controlling Optical Amplitude and Phase  

The metasurface provides complete control over the optical amplitude and phase of the incident light. This is achieved through the design of the meta-units, which can independently modulate the amplitude and phase of the optical wavefront. The system can thus perform complex optical transformations, such as focusing, beam steering, and holography.  

### Engineering Optical Dispersion  

The optical dispersion of the meta-units can be engineered by controlling their size and shape. This enables the metasurface to operate over a broad wavelength range or to perform wavelength-multiplexed tasks, such as color imaging or spectroscopy.  

### Output Plane and Detection Zones  

The output plane includes predefined detection zones where light intensity is concentrated to identify target objects. The detection zones are arranged in specific patterns, such as square or circular arrays, depending on the classification task. The system maximizes light intensity within the detection zone corresponding to the correct identity of the input object, ensuring robust recognition.  

### Concentrating Light Intensity  

The metasurface is trained to concentrate light intensity within specific detection zones, corresponding to the classification labels of input objects. This is achieved by optimizing the phase profile of the metasurface to maximize the contrast between the target zone and other zones.  

### Modifying Detection Zone Locations  

The locations of the detection zones can be modified to adapt the system for different tasks or conditions. This flexibility enables the metasurface smart glass to perform a wide range of object recognition tasks, from simple digit classification to complex facial verification.  

### Recognizing Objects by Processing Light Waves  

The system recognizes objects by processing light waves scattered by the target object. The metasurface modulates the optical wavefront, and the resulting diffraction pattern is analyzed to identify the object. The system can recognize various types of objects, including hand-written digits, alphabetical letters, and facial images.  

### Forming Diffractive ONN  

The diffractive ONN is formed by the metasurface, which acts as a hidden layer in the neural network architecture. The input layer consists of the target object, and the output layer is the detection plane. The ONN operates passively and power-free, processing optical signals at the speed of light.  

### Recognizing Hand-Written Digits and Letters  

The system can recognize hand-written digits and letters with high accuracy. For example, it can classify four classes of digits {0, 1, 3, 4} with an accuracy exceeding 99% and ten classes of digits with an accuracy of approximately 80%. The system can also recognize alphabetical letters and their typographic styles (e.g., normal or italic) with accuracies exceeding 90%.  

### Recognizing Facial Photos  

The system can perform facial verification by translating grayscale images of human faces into low-dimensional representations. The similarity between two images is evaluated by calculating the Euclidean distance between their resulting intensity arrays. The system achieves a verification accuracy of approximately 80%, comparable to that of a conventional digital ANN with three convolutional layers.  

### Double-Layered Metasurface  

For more complex tasks, such as facial verification, the system may incorporate a double-layered metasurface. This architecture enhances the expressive power of the ONN, enabling it to handle tasks beyond simple digit or letter classification. The double-layered metasurface maps an image into a 3×3 intensity array on the detection plane, allowing for the evaluation of image similarity.  

### Handling Tasks with Enhanced Expressive Power  

The double-layered metasurface enables the system to handle tasks with enhanced expressive power, such as facial verification. This is achieved by increasing the depth of the ONN, analogous to adding layers in a digital ANN. The system can thus perform more complex recognition tasks with high accuracy.  

### Translating Images into Low-Dimensional Representations  

The system translates images into low-dimensional representations by mapping them onto predefined detection zones. This enables the evaluation of image similarity using simple metrics, such as the Euclidean distance between intensity arrays.  

### Evaluating Similarity Between Images  

The similarity between two images is evaluated by calculating the Euclidean distance between their resulting intensity arrays. If the distance is below a threshold, the images are considered a match; otherwise, they are considered distinct. This approach enables robust facial verification with high accuracy.  

### Additional Metasurfaces  

The system may incorporate additional metasurfaces to further enhance its expressive power. For example, an array of distinct metasurfaces can be used to classify incoherent objects with high accuracy. This approach increases the width of the ONN, enabling it to handle more complex tasks.  

### Bypassing Digitalization for Security  

The system bypasses digitalization by processing optical signals directly in the physical domain. This ensures data security, as there is no digital representation of the subject that could be vulnerable to cyber-attacks. The system thus provides physics-guaranteed security for sensitive applications.  

### Method for Processing Light  

The invention includes a method for processing light waves to recognize target objects. The method comprises the steps of:  

1. Illuminating the target object with coherent light to generate an optical wavefront.  
2. Modulating the wavefront using a metasurface composed of an array of meta-units.  
3. Concentrating the modulated light onto predefined detection zones on an output plane.  
4. Identifying the target object based on the distribution of light intensity within the detection zones.  

### Training Diffractive Neural Network  

The method further includes training the diffractive neural network by:  

1. Providing a dataset of optically coherent images, such as hand-written digits or facial photos.  
2. Calculating the propagation of light waves through the diffractive network using numerical methods, such as the Rayleigh-Sommerfeld diffraction theory.  
3. Defining a loss function to evaluate the cross-entropy between calculated and target intensity distributions.  
4. Iteratively adjusting the phase profile of the metasurface to minimize the loss function using optimization algorithms, such as the Adam optimizer.  

### Iterative Training Process  

The training process is iterative, involving repeated adjustments to the phase profile of the metasurface to improve recognition accuracy. The process includes measures to enhance robustness against experimental errors, such as non-uniform illumination and component mispositioning.  

### Improving Robustness Against Experimental Errors  

The system is designed to be robust against experimental errors through measures incorporated during training. These include simulating non-uniform illumination, random mispositioning of components, and variations in object and imaging distances. An auxiliary term is added to the loss function to increase the contrast between detection zones, further enhancing robustness.  

### Converting Light into Optical Barcode  

The system converts light into an optical barcode by mapping the input image onto a low-dimensional intensity array. This barcode can be used to evaluate image similarity and perform tasks such as facial verification.  

### Choosing Configuration for Target Recognition Accuracy  

The configuration of the metasurface is chosen to maximize target recognition accuracy. This includes selecting the number and arrangement of detection zones, the design of the meta-units, and the training parameters. The system can be adapted for various tasks by modifying these configurations.  

## EXAMPLES  

### Metasurface Smart Glass for Object Recognition  

An exemplary embodiment of the invention is a metasurface smart glass for object recognition. The smart glass comprises a single-layered metasurface fabricated on a silicon dioxide substrate, with meta-units arranged in a square lattice. The metasurface is trained to recognize hand-written digits by concentrating light intensity into predefined detection zones.  

### Metasurface Composition and Functionality  

The metasurface is composed of amorphous silicon meta-units with a height of 1 µm and a periodicity of 750 nm. The meta-units are designed to modulate the phase of incident light at a wavelength of 1,550 nm. The metasurface is trained using a dataset of hand-written digits from the MNIST database.  

### Optical Wavefront Processing by Metasurface  

The metasurface processes the optical wavefront by superimposing a phase modulation that directs light into specific detection zones. The modulation is optimized to maximize the contrast between the target zone and other zones, ensuring robust recognition.  

### Object Recognition Mechanism  

The object recognition mechanism involves illuminating the target object with coherent light, modulating the resulting wavefront with the metasurface, and analyzing the diffraction pattern on the detection plane. The object is identified based on the detection zone with the highest light intensity.  

### Training Process for Metasurface Smart Glass  

The training process involves numerically simulating the propagation of light through the diffractive network and iteratively adjusting the phase profile of the metasurface to minimize the loss function. The process includes measures to enhance robustness against experimental errors.  

### Experimental Setup for Object Recognition  

The experimental setup includes a telecom laser (λ = 1,550 nm), a photomask containing input objects, a metasurface smart glass, and an InGaAs camera for detecting the output pattern. The setup is designed to minimize non-uniform illumination and stray light.  

### Photomask Creation for Input Objects  

The photomask is created by plotting black emulsion on a mylar sheet, with transparent apertures defining the input objects. The mask contains a 2D array of objects, such as hand-written digits or alphabetical letters, each aligned to the central axis of the optical setup.  

### Optical Setup for Object Recognition  

The optical setup includes a telescope with unity magnification to relay the input object onto a square aperture, blocking stray light from adjacent objects. The diffraction pattern is processed by the metasurface and imaged onto the detection plane using a microscope objective.  

### Metasurface Fabrication Process  

The metasurface is fabricated using electron-beam lithography to pattern amorphous silicon meta-units on a silicon dioxide substrate. The meta-units are arranged in a square lattice with a periodicity of 750 nm and a height of 1 µm.  

### Phase Responses of Meta-Unit Libraries  

The phase responses of the meta-unit libraries are characterized to ensure precise modulation of the optical wavefront. Isotropic meta-units provide uniform phase modulation for all polarization states, while birefringent meta-units enable distinct modulation for orthogonal polarizations.  

### Testing Polarization-Multiplexing Smart Glasses  

Polarization-multiplexing smart glasses are tested by illuminating the input object with light at orthogonal polarization states. The metasurface provides distinct phase modulations for each polarization, enabling multi-tasking recognition.  

### Recognizing Hand-Written Digits Using Single-Layered Metasurface  

The system recognizes four classes of hand-written digits {0, 1, 3, 4} with an accuracy exceeding 99%. The phase modulation is trained to concentrate light into four square detection zones, and the metasurface is implemented using isotropic meta-units.  

### Trained Phase Modulation for Digit Recognition  

The trained phase modulation is optimized to maximize light intensity within the detection zone corresponding to the correct digit. The modulation profile is implemented by the metasurface, ensuring precise wavefront shaping.  

### Implementing Metasurface Using Optically Isotropic Meta-Units  

The metasurface is implemented using optically isotropic meta-units, which provide uniform phase modulation for all polarization states. The meta-units are arranged in a square lattice with a periodicity of 750 nm.  

### Examples of Digit Classification  

Experimental results demonstrate successful classification of hand-written digits, with the diffraction patterns on the detection plane matching theoretical predictions. The system achieves an accuracy of 99.14% for four-class digit recognition.  

### Comparing Experimental and Theoretical Results for Digit Recognition  

The experimental results show good agreement with theoretical predictions, with the target detection zone receiving approximately 22% higher intensity than other zones. This large inter-zone contrast ensures robust recognition.  

### Recognizing 10 Classes of Hand-Written Digits Using Single-Layered Metasurface  

The system recognizes ten classes of hand-written digits with an accuracy of approximately 80%. The phase modulation is trained to concentrate light into ten circular detection zones, and the metasurface is implemented using isotropic meta-units.  

### Trained Phase Modulation for 10-Class Digit Recognition  

The trained phase modulation is optimized to maximize light intensity within the detection zone corresponding to the correct digit. The modulation profile is implemented by the metasurface, though the inter-zone contrast is reduced compared to the four-class case.  

### Implementing Metasurface Using Optically Isotropic Meta-Units  

The metasurface is implemented using optically isotropic meta-units, arranged in a square lattice with a periodicity of 750 nm. The meta-units provide uniform phase modulation for all polarization states.  

### Examples of 10-Class Digit Classification  

Experimental results demonstrate classification of ten classes of hand-written digits, with the target detection zone receiving the highest intensity. The system achieves an accuracy of 78.37%, lower than the theoretical accuracy of 86.50% due to reduced robustness.  

### Comparing Experimental and Theoretical Results for 10-Class Digit Recognition  

The experimental results show reduced inter-zone contrast compared to theoretical predictions, making the system more susceptible to experimental errors. The accuracy is lower than in the four-class case, highlighting the trade-off between complexity and robustness.  

### Recognizing Hand-Written Digits Using Polarization-Multiplexing Smart Glasses  

The system recognizes ten classes of hand-written digits using polarization-multiplexing smart glasses. The digits are divided into two groups, each recognized using light at orthogonal polarization states. The metasurface is implemented using birefringent meta-units.  

### Trained Phase Modulation for Polarization-Multiplexing Digit Recognition  

The trained phase modulation provides distinct phase profiles for orthogonal polarization states, enabling simultaneous recognition of two groups of digits. The modulation is optimized to maximize inter-zone contrast for each polarization.  

### Implementing Metasurface Using Birefringent Meta-Units  

The metasurface is implemented using birefringent meta-units, which provide distinct phase modulation for orthogonal polarization states. The meta-units are arranged in a square lattice with a periodicity of 750 nm.  

### Examples of Polarization-Multiplexing Digit Classification  

Experimental results demonstrate classification of ten classes of hand-written digits using polarization-multiplexing, with accuracies of 90.99% and 81.44% for the two groups. The system outperforms the non-birefringent device, though the phase coverage is more discrete.  

### Comparing Experimental and Theoretical Results for Polarization-Multiplexing Digit Recognition  

The experimental results show higher accuracy than the non-birefringent device, though lower than theoretical predictions due to discrete phase coverage. The system demonstrates the potential of polarization-multiplexing for complex recognition tasks.  

### Recognizing Typed Alphabetical Letters and Typographic Styles Using Polarization-Multiplexing Smart Glasses  

The system recognizes typed alphabetical letters and their typographic styles (normal or italic) using polarization-multiplexing smart glasses. The metasurface provides distinct phase modulations for each polarization state, enabling simultaneous letter and style recognition.  

### Trained Phase Modulation for Letter and Style Recognition  

The trained phase modulation is optimized to concentrate light into four detection zones for letter recognition and two zones for style recognition. The modulation is implemented using birefringent meta-units.  

### Implementing Metasurface Using Birefringent Meta-Units  

The metasurface is implemented using birefringent meta-units, which provide distinct phase modulation for orthogonal polarization states. The meta-units are arranged in a square lattice with a periodicity of 750 nm.  

### Examples of Letter and Style Classification  

Experimental results demonstrate classification of alphabetical letters and typographic styles with accuracies of 92.81% and 100%, respectively. The system showcases the multi-tasking capability of polarization-multiplexing smart glasses.  

### Comparing Experimental and Theoretical Results for Letter and Style Recognition  

The experimental results show high accuracy for both letter and style recognition, matching theoretical predictions. The system demonstrates the potential of polarization-multiplexing for complex multi-tasking recognition.  

### Introducing Facial Verification Using Double-Layered Metasurface Smart Glass  

The system performs facial verification using a double-layered metasurface smart glass. The metasurface doublet translates grayscale facial images into low-dimensional intensity arrays, enabling similarity evaluation via Euclidean distance.  

### Working Mechanism of Metasurface Doublet  

The metasurface doublet processes the optical wavefront through two sequential layers, enhancing the expressive power of the ONN. The doublet maps the input image into a 3×3 intensity array on the detection plane.  

### Example Human Face Images Used in Training and Testing  

The system is trained and tested using a dataset of photos of 100 people, each with 14 distinct photos. The photos of 90 people are used for training, and the remaining 10 for testing.  

### Trained Phase Modulations of Metasurface Doublet  

The phase modulations of the metasurface doublet are optimized to maximize the contrast between intensity arrays for different facial images. The modulations are implemented using isotropic meta-units.  

### Evaluating Facial Verification Accuracy  

The system achieves a facial verification accuracy of approximately 80%, comparable to a digital ANN with three convolutional layers. The rate of false acceptance and false rejection are both approximately 10% at an optimal threshold.  

### Comparing Results with Digital ANN  

The system's performance is comparable to a digital ANN with three convolutional layers, though the ONN operates passively and at the speed of light. The comparison highlights the potential of metasurface-based ONNs for secure and efficient facial verification.  

### Examples of Facial Verification  

Experimental results demonstrate successful facial verification, with the system correctly matching images of the same person and distinguishing images of different people. The system showcases the potential of ONNs for secure biometric applications.  

### Limitations of Single-Layered Metasurface Smart Glasses  

Single-layered metasurface smart glasses are limited in expressive power, making them suitable for simple tasks like digit recognition but less effective for complex tasks like facial verification. The double-layered architecture addresses this limitation.  

### Advantages of Metasurface Smart Glasses Over Digital ANNs  

Metasurface smart glasses offer several advantages over digital ANNs, including passive operation, high speed, energy efficiency, and data security. They bypass digitalization, ensuring physics-guaranteed security for sensitive applications.  

### Potential Applications of Metasurface Smart Glasses  

Potential applications include edge perception devices, biometric security systems, and compact sensors for infrastructure-limited environments. The glasses can fundamentally reshape data collection and analysis by condensing measurement and computing into a single passive device.  

### Future Directions for Metasurface Smart Glass Development  

Future work may explore scaling up the width and depth of ONNs, incorporating nonlinear materials for enhanced functionality, and extending the system to recognize incoherent objects. These advancements could enable broader applications in computer vision.  

### Challenges in Scaling Up Metasurface Smart Glasses  

Scaling up metasurface smart glasses involves challenges such as manufacturing complexity, maintaining uniformity across large areas, and integrating multiple metasurface layers. Potential solutions include advanced nanofabrication techniques and modular designs.  

### Potential Solutions to Scaling Challenges  

Potential solutions include using phase-amplitude metasurfaces, polarization and wavelength multiplexing, and arrays of distinct metasurfaces. These approaches can enhance the expressive power of ONNs while addressing manufacturing challenges.  

### Summarizing Benefits of Metasurface Smart Glasses  

Metasurface smart glasses offer compact, energy-efficient, and secure object recognition, operating at the speed of light. They bypass digitalization, ensuring privacy and resilience against cyber-attacks, and are suitable for edge perception applications.  

### Concluding with Potential Impact of Metasurface Smart Glasses  

Metasurface smart glasses have the potential to revolutionize object recognition by providing passive, high-speed, and secure computing. They can enable new applications in biometrics, autonomous systems, and IoT devices, fundamentally reshaping the future of data collection and analysis.  

### Introducing Metasurface ONN Robustness  

The robustness of metasurface ONNs is quantified by the inter-zone contrast, which measures the intensity difference between the target detection zone and other zones. Higher contrast correlates with better agreement between theoretical and experimental results.  

### Motivating Inter-Zone Contrast  

Inter-zone contrast is a key metric for evaluating ONN robustness. By maximizing this contrast during training, the system can mitigate the impact of experimental errors, such as non-uniform illumination and component mispositioning.  

### Describing ONN Expressive Power  

The expressive power of an ONN is determined by its ability to perform complex recognition tasks. This power can be enhanced by increasing the width and depth of the network, such as through polarization multiplexing or multi-layered architectures.  

### Introducing Width and Depth Increase  

Increasing the width and depth of an ONN enhances its expressive power, enabling it to handle more complex tasks. Width can be increased through polarization or wavelength multiplexing, while depth can be increased by adding metasurface layers.  

### Describing Polarization Multiplexing  

Polarization multiplexing doubles the expressive power of a metasurface by enabling distinct phase modulations for orthogonal polarization states. This approach is demonstrated in multi-tasking smart glasses for letter and style recognition.  

### Describing Phase-Amplitude Metasurface Holograms  

Phase-amplitude metasurface holograms provide complete control over the optical wavefront, enabling more complex transformations than phase-only metasurfaces. These holograms can enhance the expressive power of ONNs for advanced recognition tasks.  

### Describing Wavelength-Multiplexing  

Wavelength-multiplexing introduces an additional dimension to increase ONN expressive power. By engineering the optical dispersion of meta-units, a single metasurface can encode distinct amplitude-phase profiles at different wavelengths.  

### Describing Array of Distinct Metasurfaces  

An array of distinct metasurfaces in each layer of the ONN can further enhance expressive power. Preliminary results show that a single layer of 10 distinct metasurfaces can classify 10 classes of incoherent objects with high accuracy.  

### Summarizing ONN Advantages  

ONNs offer several advantages over digital ANNs, including passive operation, high speed, energy efficiency, and data security. They can perform complex recognition tasks directly in the physical domain, bypassing digitalization.  

### Introducing Advanced Sensors  

Advanced sensors based on ONNs can be deployed in infrastructure-limited environments, requiring minimal service and offering resilience to interference. These sensors can fundamentally reshape data collection and analysis by condensing measurement and computing into a single passive device.  

### Describing ONN Compactness  

ONNs are highly compact, with metasurfaces fabricated using CMOS-compatible techniques. Their small form factor enables integration into portable devices and edge perception systems.  

### Describing ONN Energy Efficiency  

ONNs operate entirely passively, requiring no additional power after the optical input is generated. This energy efficiency makes them suitable for battery-limited or remote applications.  

### Describing ONN Computing Speed  

ONNs compute at the speed of light, offering ultra-fast recognition without the latency inherent in digital systems. This speed is critical for real-time applications such as autonomous vehicles and security systems.  

### Describing ONN Accuracy  

ONNs achieve high recognition accuracy, comparable to digital ANNs for certain tasks. Their accuracy can be further improved through advanced training techniques and enhanced expressive power.  

### Describing ONN Data Security  

ONNs provide physics-guaranteed data security by bypassing digitalization. Computational algorithms are hard-coded into the metasurface materials, ensuring resilience against cyber-attacks.  

### Introducing Digital ANN Limitations  

Digital ANNs suffer from limitations such as high power consumption, latency, and vulnerability to cyber-attacks. These limitations are exacerbated in edge perception applications, where energy efficiency and security are critical.  

### Describing ONN Advantages Over Digital ANNs  

ONNs overcome the limitations of digital ANNs by offering passive operation, high speed, energy efficiency, and data security. They can perform complex recognition tasks directly in the physical domain, without digital processing.  

### Describing ONN Design Process  

The ONN design process involves defining the network architecture, selecting meta-unit libraries, and training the metasurface phase profile. The process includes measures to enhance robustness against experimental errors.  

### Describing Forward Calculation Process  

The forward calculation process simulates light propagation through the diffractive network using numerical methods, such as the Rayleigh-Sommerfeld diffraction theory. This process is used to evaluate the performance of the ONN during training.  

### Describing Backward Calculation Process  

The backward calculation process adjusts the phase profile of the metasurface to minimize the loss function. This process uses optimization algorithms, such as the Adam optimizer, to iteratively improve recognition accuracy.  

### Describing Strategies for Robustness  

Strategies for enhancing robustness include simulating experimental errors during training, adding auxiliary terms to the loss function, and optimizing inter-zone contrast. These measures ensure reliable performance under real-world conditions.  

### Introducing Experimental Results  

Experimental results demonstrate the performance of metasurface smart glasses in various recognition tasks, including digit classification, letter and style recognition, and facial verification. The results validate the theoretical predictions and highlight the potential of ONNs.  

### Describing Recognition of Handwritten Digits  

The system recognizes hand-written digits with high accuracy, achieving 99.14% for four-class classification and approximately 80% for ten-class classification. The results showcase the capability of ONNs for simple recognition tasks.  

### Describing Metasurface Device Fabrication  

Metasurface devices are fabricated using electron-beam lithography to pattern amorphous silicon meta-units on a silicon dioxide substrate. The fabrication process ensures high precision and uniformity, enabling complex optical transformations.  

### Describing Recognition Results  

Recognition results demonstrate the system's ability to classify input objects based on the distribution of light intensity within detection zones. The results show good agreement between theoretical and experimental performance.  

### Describing Polarization Multiplexing Example  

Polarization multiplexing enables multi-tasking recognition, such as classifying alphabetical letters and their typographic styles. The system achieves high accuracy for both tasks, demonstrating the potential of polarization-multiplexing ONNs.  

### Describing Handwritten Digit Classification Task  

The handwritten digit classification task involves recognizing four or ten classes of digits from the MNIST database. The system achieves high accuracy through optimized phase modulation and robust training techniques.  

### Describing Human Facial Verification Task  

The human facial verification task involves translating grayscale facial images into low-dimensional representations and evaluating similarity via Euclidean distance. The system achieves an accuracy of approximately 80%, comparable to digital ANNs.  

### Describing Error Rate Diagram  

Error rate diagrams illustrate the performance of the ONN in facial verification, showing the rates of false acceptance and false rejection at different thresholds. The diagrams highlight the trade-offs in system design.  

### Introducing Facial Image Verification Dataset  

The facial image verification dataset includes photos of 100 people, each with 14 distinct images. The dataset is used to train and test the ONN, ensuring robust performance across diverse facial images.  

### Describing ONN Design Parameters  

ONN design parameters include the number and arrangement of detection zones, the meta-unit library, and the training algorithm. These parameters are optimized to maximize recognition accuracy and robustness.  

### Describing Metasurface Doublet Design  

The metasurface doublet design enhances the expressive power of the ONN, enabling complex tasks like facial verification. The doublet consists of two sequential metasurface layers, each optimized for specific wavefront transformations.  

### Describing Optical Barcode Size  

The optical barcode size refers to the dimensionality of the intensity array used to represent the input image. The size is optimized to balance recognition accuracy and computational efficiency.  

### Describing Loss Function Design  

The loss function evaluates the cross-entropy between calculated and target intensity distributions. It includes auxiliary terms to enhance inter-zone contrast and robustness against experimental errors.  

### Describing Partial Facial Coverage Effect  

Partial facial coverage reduces the accuracy of facial verification by obscuring key features. The ONN is designed to mitigate this effect through robust training techniques and optimized detection zones.  

### Describing Facial Photo Examples  

Facial photo examples illustrate the system's ability to verify images of the same person and distinguish images of different people. The examples showcase the potential of ONNs for secure biometric applications.  

### Describing Verification Accuracy Results  

Verification accuracy results demonstrate the system's performance in matching and distinguishing facial images. The results highlight the trade-offs between false acceptance and false rejection rates.  

### Describing Separation Distance Effect  

The separation distance between the input object and the metasurface affects the diffraction pattern and recognition accuracy. The system is designed to maintain robust performance across a range of separation distances.  

### Describing Relative Size Effect  

The relative size of the input object and the metasurface influences the optical transformation and recognition accuracy. The system is optimized to handle objects of varying sizes within a predefined range.  

### Describing Substrate Thickness Effect  

The substrate thickness impacts the mechanical stability and optical performance of the metasurface. The system is designed with a substrate thickness of approximately 500 µm to balance these factors.  

### Introducing One-Layer Metasurface ONN  

A one-layer metasurface ONN is suitable for simple recognition tasks, such as digit classification. The design and performance of such ONNs are characterized by high accuracy and robustness against experimental errors.  

### Describing Design and Performance of One-Layer Metasurface ONN  

The one-layer metasurface ONN is designed to maximize inter-zone contrast and recognition accuracy. Experimental results demonstrate high performance in digit classification, with accuracies exceeding 99% for four-class tasks.  

### Illustrating Verification Process with Examples  

The verification process involves illuminating the input object, modulating the wavefront with the metasurface, and analyzing the diffraction pattern. Examples illustrate successful classification of hand-written digits and letters.  

### Explaining Training Process and Optimization of Metasurface Design  

The training process involves iterative optimization of the metasurface phase profile to minimize the loss function. The process includes measures to enhance robustness and inter-zone contrast.  

### Defining Barcode Calculation and Euclidean Distance  

The barcode calculation translates the input image into a low-dimensional intensity array. The Euclidean distance between barcodes evaluates image similarity, enabling tasks like facial verification.  

### Introducing Loss Function and Its Design  

The loss function evaluates the performance of the ONN during training. It includes terms to maximize inter-zone contrast and minimize recognition errors, ensuring robust performance.  

### Discussing Performance of One-Layer Metasurface ONN  

The one-layer metasurface ONN achieves high accuracy for simple recognition tasks, though its expressive power is limited for complex tasks. The performance is characterized by high inter-zone contrast and robustness.  

### Introducing Metasurface Doublet ONN  

The metasurface doublet ONN enhances expressive power by incorporating two sequential metasurface layers. This architecture enables complex tasks like facial verification with accuracy comparable to digital ANNs.  

### Describing Design and Performance of Metasurface Doublet ONN  

The metasurface doublet is designed to maximize recognition accuracy and robustness. Experimental results demonstrate successful facial verification, with an accuracy of approximately 80%.  

### Illustrating Verification Process with Examples  

Examples illustrate the metasurface doublet's ability to verify facial images, correctly matching images of the same person and distinguishing images of different people. The process showcases the potential of ONNs for biometric applications.  

### Comparing Performance of One-Layer and Metasurface Doublet ONNs  

The comparison highlights the trade-offs between simplicity and expressive power. One-layer ONNs are suitable for simple tasks, while doublet ONNs enable complex tasks like facial verification.  

### Discussing Robustness of Metasurface Doublet Design Against Experimental Errors  

The metasurface doublet is designed to be robust against experimental errors, such as non-uniform illumination and component mispositioning. The design includes measures to enhance inter-zone contrast and recognition accuracy.  

### Discussing Robustness of Metasurface Doublet Design Against Wavelength Variation  

The metasurface doublet is robust against wavelength variations, ensuring consistent performance across a range of operating conditions. This robustness is achieved through careful design of the meta-unit dispersion.  

### Introducing Concept of Optimal Barcode Size  

The optimal barcode size balances recognition accuracy and computational efficiency. The size is determined through empirical testing and optimization during the training process.  

### Investigating Relationship Between Verification Accuracy and Barcode Size  

The relationship between verification accuracy and barcode size is characterized by a trade-off between dimensionality and performance. Larger barcodes offer higher accuracy but require more complex metasurface designs.  

### Showing Error Rate Diagrams for Different Barcode Sizes  

Error rate diagrams illustrate the performance of the ONN at different barcode sizes. The diagrams highlight the optimal size for balancing false acceptance and false rejection rates.  

### Discussing Optimal Barcode Size for One-Layer ONN  

The optimal barcode size for a one-layer ONN is determined through empirical testing. The size is chosen to maximize recognition accuracy while maintaining robustness against experimental errors.  

### Showing Facial Verification Examples for One-Layer ONNs  

Facial verification examples demonstrate the performance of one-layer ONNs at the optimal barcode size. The examples showcase the system's ability to match and distinguish facial images.  

### Assessing ONNs Based on Metasurface Doublets  

ONNs based on metasurface doublets are assessed for their performance in complex recognition tasks. The assessment highlights the advantages of doublet architectures over single-layer designs.  

### Discussing Optimal Barcode Size for Metasurface Doublet ONNs  

The optimal barcode size for metasurface doublet ONNs is determined through empirical testing. The size is chosen to maximize facial verification accuracy while maintaining robustness.  

### Showing Error Rate Diagrams for Different Barcode Sizes  

Error rate diagrams illustrate the performance of metasurface doublet ONNs at different barcode sizes. The diagrams highlight the optimal size for balancing false acceptance and false rejection rates.  

### Discussing Performance of Metasurface Doublet ONNs  

The performance of metasurface doublet ONNs is characterized by high accuracy and robustness in complex recognition tasks. The results demonstrate the potential of doublet architectures for advanced applications.  

### Showing Facial Verification Examples for Metasurface Doublet ONNs  

Facial verification examples demonstrate the performance of metasurface doublet ONNs at the optimal barcode size. The examples showcase the system's ability to match and distinguish facial images with high accuracy.  

### Summarizing Results of One-Layer and Metasurface Doublet ONNs  

The results highlight the trade-offs between simplicity and expressive power. One-layer ONNs are suitable for simple tasks, while doublet ONNs enable complex tasks like facial verification.  

### Concluding with Optimal Barcode Sizes for Both Designs  

The optimal barcode sizes for one-layer and doublet ONNs are determined through empirical testing. The sizes are chosen to maximize recognition accuracy while maintaining robustness against experimental errors.  

### Introducing Optical Neural Networks (ONNs)  

ONNs are emerging as a powerful platform for object recognition, offering passive operation, high speed, energy efficiency, and data security. They perform computations directly in the physical domain, bypassing digitalization.  

### Describing ONN Design with Metasurface  

The ONN design incorporates metasurfaces composed of subwavelength meta-units to manipulate the optical wavefront. The design is optimized for specific recognition tasks through iterative training.  

### Explaining Optical Scattering Pattern Concentration  

The ONN concentrates optical scattering patterns into predefined detection zones, enabling object recognition based on light intensity distribution. The concentration is optimized to maximize inter-zone contrast.  

### Defining Loss Function for Training ONN  

The loss function evaluates the cross-entropy between calculated and target intensity distributions. It includes auxiliary terms to enhance robustness and inter-zone contrast during training.  

### Showing Results of ONN with Different Weights  

Results demonstrate the performance of the ONN with different weighting factors in the loss function. The weights are optimized to balance recognition accuracy and robustness against experimental errors.  

### Comparing ONN Designs with Different Degrees of Concentration  

The comparison highlights the trade-offs between concentration and recognition accuracy. Higher concentration improves inter-zone contrast but may reduce the system's ability to handle complex tasks.  

### Discussing Effect of Partial Facial Coverage on Verification Accuracy  

Partial facial coverage reduces verification accuracy by obscuring key features. The ONN is designed to mitigate this effect through robust training techniques and optimized detection zones.  

### Describing ONN Design for Verifying Photos with Partial Facial Coverage  

The ONN design includes measures to enhance verification accuracy for photos with partial facial coverage. The design is optimized to maintain robust performance despite obscured features.  

### Showing Error Rate Diagram for ONN with Partial Facial Coverage  

The error rate diagram illustrates the performance of the ONN in verifying photos with partial facial coverage. The diagram highlights the trade-offs between false acceptance and false rejection rates.  

### Showing Trained Phase Distributions for ONN with Partial Facial Coverage  

The trained phase distributions demonstrate the metasurface's ability to handle partial facial coverage. The distributions are optimized to maximize recognition accuracy despite obscured features.  

### Showing Examples of Correct Facial Verification Using ONN  

Examples showcase the ONN's ability to correctly verify facial images with partial coverage. The examples highlight the system's robustness and potential for real-world applications.  

### Describing ONN Design for Verifying Facial Photos Without Facial Cover  

The ONN design for verifying photos without facial cover is optimized to maximize recognition accuracy. The design includes measures to enhance inter-zone contrast and robustness.  

### Showing Error Rate Diagram for ONN Without Facial Cover  

The error rate diagram illustrates the performance of the ONN in verifying photos without facial cover. The diagram highlights the system's high accuracy and robustness.  

### Showing Trained Phase Distributions for ONN Without Facial Cover  

The trained phase distributions demonstrate the metasurface's ability to verify photos without facial cover. The distributions are optimized to maximize recognition accuracy and inter-zone contrast.  

### Discussing Classification of Optically Incoherent Images  

Optically incoherent images represent a challenge for ONNs due to reduced expressive power in the absence of optical interference. The system is designed to handle incoherent images through advanced training techniques.  

### Describing ONN Using Parallel Array of Metasurfaces  

A parallel array of metasurfaces enhances the ONN's ability to classify incoherent images. The array increases the network's width, enabling more complex recognition tasks.  

### Showing Design for Classifying Incoherent Images  

The design includes measures to optimize the ONN for classifying incoherent images. The design is validated through experimental testing and comparison with theoretical predictions.  

### Showing Amplitude Masks of Metasurfaces for Recognizing Handwritten Digits  

Amplitude masks illustrate the metasurface's ability to recognize handwritten digits. The masks are optimized to maximize recognition accuracy and inter-zone contrast.  

### Showing Confusion Matrix for Recognizing Handwritten Digits  

The confusion matrix demonstrates the ONN's performance in recognizing handwritten digits. The matrix highlights the system's high accuracy and robustness against misclassification.  

### Describing Generation of Optically Coherent Facial Images  

Optically coherent facial images are generated using transparency-based or LCD-based methods. The images are used to train and test the ONN for facial verification tasks.  

### Showing Transparency-Based Method for Generating Coherent Images  

The transparency-based method involves plotting facial images on a photomask. The method ensures high coherence and uniformity for accurate ONN training and testing.  

### Showing LCD-Based Method for Generating Coherent Images  

The LCD-based method uses a spatial light modulator to generate coherent facial images. The method offers flexibility and precision in image generation for ONN validation.  

### Showing Optical Setup for ONN  

The optical setup includes a laser, photomask, metasurface, and detector. The setup is designed to minimize non-uniform illumination and stray light, ensuring accurate ONN performance.  

### Showing Scanning Electron Micrographs of Fabricated Metasurfaces  

Scanning electron micrographs illustrate the precision and uniformity of fabricated metasurfaces. The micrographs validate the quality of the nanofabrication process.  

### Describing Advantages of ONNs Over Digital ANNs  

ONNs offer several advantages over digital ANNs, including passive operation, high speed, energy efficiency, and data security. They perform computations directly in the physical domain, bypassing digitalization.  

### Estimating Computing Time of ONN  

The computing time of an ONN is characterized by the propagation speed of light, offering ultra-fast recognition without the latency inherent in digital systems.  

### Estimating Power Consumption of ONN  

The power consumption of an ONN is minimal, as it operates entirely passively after the optical input is generated. This energy efficiency makes it suitable for battery-limited applications.  

### Describing System for Developing and Validating ONN Designs  

The system for developing and validating ONN designs includes numerical simulation, experimental testing, and optimization. The system ensures robust performance across diverse recognition tasks.  

### Discussing Accuracy of ONNs for Verifying Optically Coherent Facial Images  

ONNs achieve high accuracy in verifying optically coherent facial images, comparable to digital ANNs. The accuracy is validated through experimental testing and comparison with theoretical predictions.  

### Discussing Strategy for Classifying Optically Incoherent Images  

The strategy for classifying incoherent images involves increasing the ONN's width through parallel arrays of metasurfaces. The strategy is validated through experimental testing and optimization.  

### Describing Benefits of ONNs Over Digital ANNs  

ONNs offer several benefits over digital ANNs, including passive operation, high speed, energy efficiency, and data security. They can perform complex recognition tasks directly in the physical domain.  

### Discussing Scope of Disclosed Subject Matter  

The disclosed subject matter encompasses metasurface-based ONNs for object recognition, including design, fabrication, training, and validation. The scope highlights the potential of ONNs for diverse applications in computer vision.