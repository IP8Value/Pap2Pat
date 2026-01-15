# DESCRIPTION

## FIELD OF THE INVENTION

- introduce proximity sensor

The present invention relates to a proximity sensor system capable of distinguishing between metallic and non-metallic objects through frequency-resolved phase response analysis. The sensor operates by exciting a resonant structure embedded within a transmission line, wherein the phase characteristics of the reflected or transmitted signal are analyzed to determine not only the presence of an object in proximity but also its material composition. Unlike conventional capacitive proximity sensors that rely solely on amplitude changes in a fixed-frequency excitation signal, this invention leverages the frequency-dependent phase behavior of resonant circuits to enable material-specific discrimination. The sensor architecture comprises multiple independently tuned resonant elements, each associated with a distinct resonant frequency, arranged in a spatially distributed configuration to permit multi-directional or multi-point detection. The system is particularly suited for industrial automation, robotics, touchless human-machine interfaces, security screening, and medical device applications where precise object classification is critical to operational safety and functionality. The invention enables a single sensor array to simultaneously detect proximity and classify object type without requiring physical contact, additional sensors, or complex imaging systems, thereby reducing system complexity, power consumption, and cost while enhancing reliability and spatial resolution.

## BACKGROUND OF THE INVENTION

- limitations of conventional sensors
- prior art examples

Conventional proximity sensors, including inductive, capacitive, ultrasonic, and infrared variants, are widely employed in industrial and consumer applications for object detection. However, these systems are fundamentally limited in their ability to differentiate between metallic and non-metallic materials. Capacitive sensors, for instance, respond to changes in dielectric properties but cannot reliably distinguish between a plastic object and a human hand, nor between aluminum and steel, because they measure only the magnitude of capacitance change without regard to the phase dynamics of the electromagnetic interaction. Inductive sensors, while sensitive to conductive materials, are blind to non-conductive objects and often require larger physical footprints to achieve adequate sensing ranges. Ultrasonic and infrared sensors, though capable of detecting presence, lack the spectral resolution necessary to infer material composition and are susceptible to environmental interference such as temperature gradients, dust, or ambient light. Prior art systems have attempted to improve discrimination by employing multiple sensor types in parallel, resulting in bulky, power-intensive, and costly multi-sensor arrays. Other approaches involve time-domain signal analysis or machine learning algorithms applied to amplitude-only data, but these methods require extensive calibration, suffer from low reproducibility across materials, and cannot operate in real-time without significant computational overhead. Furthermore, existing capacitive sensor arrays that utilize frequency multiplexing rely on amplitude variations at discrete frequencies to infer object position, but they do not exploit the phase shift inherent in resonant systems, which contains critical information about the nature of the coupling between the object and the sensor. As a result, prior systems are incapable of reliably classifying object type based on electromagnetic interaction dynamics, limiting their utility in applications requiring material-specific responses, such as automated sorting, anti-tamper detection, or gesture recognition involving different hand materials or tools.

## SUMMARY OF THE INVENTION

- motivate resonant structure
- describe phase change detection
- distinguish metallic and non-metallic objects
- introduce method for determining object type
- describe proximity sensor embodiment
- summarize object type determination

The invention is motivated by the discovery that the phase response of a resonant capacitive structure, when perturbed by a nearby object, exhibits a material-dependent signature that is distinct for metallic versus non-metallic targets. Unlike amplitude-based detection, which is influenced by distance and object size, the phase shift at and near the resonant frequency encodes information about the conductivity and permittivity of the interacting medium. Metallic objects induce a dominant inductive coupling that results in a negative phase shift, whereas non-metallic, dielectric objects produce a capacitive coupling that yields a positive phase shift. This fundamental distinction enables the invention to classify object type with high fidelity using a single sensor array. A method for determining object type is introduced, wherein the phase of the transmitted or reflected signal is measured across a range of frequencies surrounding each resonant mode, and the direction and magnitude of the phase deviation are analyzed to determine whether the proximate object is metallic or non-metallic. The proximity sensor embodiment comprises a microstrip transmission line integrated with multiple resonant elements, each comprising a capacitive patch coupled to an inductive component—such as a surface-mounted inductor, a meandered on-chip inductor, or a split-ring metamaterial structure—tuned to a unique resonant frequency. Each resonant element operates independently within its designated frequency band, allowing simultaneous detection of multiple objects or directional proximity events. The sensor is excited by a swept-frequency signal, and the phase response at each resonant peak is compared against a baseline reference to determine both the presence of an object and its material classification. The invention thus summarizes a novel object type determination method that combines spatial multiplexing with spectral phase analysis, enabling a compact, low-power, high-resolution proximity sensor capable of distinguishing metallic from non-metallic objects without requiring direct contact, additional sensing modalities, or external computational processing beyond real-time phase comparison.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS

- motivate resonant structure
- describe phase change detection
- distinguish metallic and non-metallic objects
- introduce method for determining object type
- describe proximity sensor embodiment
- summarize object type determination
- describe FIG. 1
- motivate phase change detection
- describe metallic object phase change
- describe non-metallic object phase change
- summarize object type determination
- describe FIG. 2
- describe phase change on resonant frequency
- describe phase change on off-resonant frequencies
- summarize object type determination
- describe FIGS. 3A and 3B
- describe proximity sensor embodiment
- describe power source and sensor unit
- describe detecting unit and processor
- describe object type determination
- describe FIG. 4
- describe processor operation
- describe object type determination
- describe multiple resonant frequencies
- describe resonant structure design
- describe FIGS. 5 and 6
- describe detecting and processing unit

The resonant structure is motivated by the need to isolate material-specific electromagnetic interactions within a shared sensing platform. Each resonant element is designed to exhibit a sharp, well-defined resonance with a high quality factor, ensuring minimal spectral crosstalk between adjacent elements. Phase change detection is performed by sweeping the excitation frequency across a range encompassing each resonant frequency and recording the phase angle of the transmission or reflection coefficient. When a metallic object approaches the capacitive patch, the induced eddy currents generate a secondary magnetic field that opposes the incident field, resulting in a net inductive loading of the resonator and a measurable negative phase shift. In contrast, a non-metallic object, such as wood, plastic, or human tissue, increases the local permittivity without inducing significant currents, leading to a capacitive loading effect and a corresponding positive phase shift. This dichotomy in phase response forms the basis for distinguishing object type. A method for determining object type is introduced wherein the phase deviation at each resonant frequency is compared to a pre-calibrated threshold: a phase shift below zero indicates a metallic object, while a phase shift above zero indicates a non-metallic object. The proximity sensor embodiment comprises a printed circuit board with a microstrip transmission line, upon which multiple capacitive sensing patches are arranged, each connected to a distinct inductive element. A power source, such as a voltage-controlled oscillator or a synthesized RF signal generator, provides a swept-frequency excitation signal to the transmission line. A detecting unit, comprising a directional coupler and a phase-sensitive receiver, captures the reflected or transmitted signal, which is then digitized and processed by a microprocessor. The processor compares the measured phase at each resonant frequency against stored baseline phase profiles and applies a classification algorithm to assign a material type to each detected event. FIG. 1 illustrates the overall architecture of the sensor system, showing the excitation source, transmission line, resonant elements, and signal processing chain. FIG. 2 presents the phase response curves for metallic and non-metallic objects at resonance, clearly demonstrating the opposing phase trends. At off-resonant frequencies, the phase remains largely unchanged, confirming that the material discrimination is localized to the resonant modes. FIGS. 3A and 3B depict the circuit model and simulated phase responses, validating the theoretical basis of the invention. The proximity sensor embodiment, as shown in FIG. 4, integrates the power source, sensor unit, and processor on a single substrate, enabling compact deployment. The processor operates by sampling the phase at discrete frequency points surrounding each resonance, computing the deviation from baseline, and applying a decision rule based on sign and magnitude. Multiple resonant frequencies are achieved by varying the inductance of each element, with values selected to ensure sufficient spectral separation. The resonant structure design employs either surface-mounted inductors for prototyping or meandered on-chip inductors and split-ring metamaterial structures for high-quality-factor implementations, as shown in FIGS. 5 and 6. The detecting and processing unit includes a phase detector, analog-to-digital converter, memory for baseline storage, and a classifier algorithm that outputs a material classification signal for each resonant channel, enabling real-time, multi-object, material-aware proximity sensing.