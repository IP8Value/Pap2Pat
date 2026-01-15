# DESCRIPTION

## BACKGROUND

- motivate privacy concerns

The increasing integration of sensing technologies into domestic and institutional environments has raised significant concerns regarding personal privacy and data security. Traditional methods of human activity monitoring, such as video surveillance and wearable sensors, inherently intrude upon the private sphere by capturing visually identifiable behaviors, bodily movements, or physiological signals that can be traced back to specific individuals. These systems often require continuous recording, storage, and transmission of sensitive data, creating vulnerabilities to unauthorized access, misuse, or long-term surveillance. Even when anonymization techniques are employed, the contextual richness of the collected data—such as the timing, duration, and spatial pattern of movements—can enable re-identification through behavioral profiling. In environments where autonomy and dignity are paramount, such as elderly care facilities, private residences, or medical recovery units, the psychological burden of being perpetually observed can diminish the very quality of life these technologies aim to enhance. Furthermore, regulatory frameworks governing data collection in healthcare and residential settings impose stringent requirements on consent, data minimization, and purpose limitation, which many existing monitoring systems fail to satisfy. There is therefore a critical need for a non-invasive, passive sensing paradigm that detects human presence and activity without capturing visual, auditory, or biometric identifiers, thereby preserving individual privacy while enabling meaningful ambient intelligence. This need is particularly acute in scenarios where continuous, unobtrusive monitoring is required over extended periods, such as during sleep, rest, or routine daily activities, where the presence of cameras or wearable devices is impractical, undesirable, or ethically problematic.

## SUMMARY

- introduce system for remote sensing

A novel system for remote, non-contact sensing of human activity is disclosed, leveraging the principles of radio frequency backscattering and polarization modulation to detect subtle movements within an environment without requiring any wearable or attached devices. The system operates by emitting circularly polarized radio frequency waves from a fixed reader antenna and analyzing the reflected signals that have interacted with a human body, distinguishing these reflections from direct line-of-sight signals through polarization-state transformation. This approach enables the detection of human presence and motion across a wide spatial volume using a minimal number of passive components, eliminating the need for user cooperation, device maintenance, or visual monitoring.

- describe RFID emitter

The RFID emitter comprises a radio frequency transmitter configured to generate a continuous wave signal at a frequency band allocated for industrial, scientific, and medical use, typically within the 902 to 928 MHz range. The emitter is coupled to a circularly polarized antenna designed to radiate electromagnetic energy with a defined handedness—either right-hand or left-hand circular polarization—ensuring consistent polarization state across the operational volume. The emitter is capable of sustained transmission at controlled power levels, sufficient to energize passive RFID tags within its range while maintaining compliance with regulatory limits on radiated emissions.

- describe circular-polarized passive RFID tag

The circularly-polarized passive RFID tag is a self-contained, battery-free device comprising a spiral antenna, a matching network, and an integrated circuit. The spiral antenna is configured to receive incident circularly polarized waves and re-radiate modulated signals with reversed handedness upon interaction with a human body. The tag is fabricated on a low-cost dielectric substrate and is designed to be mounted on fixed surfaces such as walls, ceilings, or furniture, rendering it unobtrusive and maintenance-free. Its passive nature ensures no power source is required, and its polarization characteristics are engineered to maximize signal discrimination between direct and reflected paths.

- describe RFID reader

The RFID reader is a dual-function transceiver unit comprising a transmitter for emitting circularly polarized radio frequency waves and a receiver for capturing backscattered signals. The reader is equipped with a high-sensitivity receiver capable of detecting minute variations in signal amplitude and phase, with sampling rates sufficient to resolve transient human movements occurring at sub-second intervals. The reader is configured to operate in full-duplex mode, enabling simultaneous transmission and reception without interference, and is coupled to a digital signal processing unit that performs real-time analysis of received signal characteristics.

- describe system features

The system exhibits several distinguishing features that collectively enable high-accuracy, privacy-preserving remote sensing. First, it requires no user-worn devices, eliminating behavioral compliance burdens. Second, it does not capture visual, auditory, or biometric data, ensuring compliance with privacy regulations. Third, it operates effectively in cluttered environments with multiple reflective surfaces, as the polarization-based discrimination suppresses multipath interference from non-human objects. Fourth, it achieves extended detection range and spatial coverage using a single tag-reader pair, reducing deployment complexity and cost. Fifth, it is scalable to multiple tag-reader configurations for larger or multi-room environments without introducing cross-talk or signal interference.

- describe spiral antenna

The spiral antenna employed in the circularly polarized passive RFID tag is an Archimedean spiral structure fabricated using conductive traces on a rigid dielectric substrate. The spiral geometry provides frequency-independent circular polarization characteristics over a broad bandwidth, ensuring consistent performance across the entire RFID operating band. The antenna’s physical dimensions are optimized to achieve a realized gain sufficient to support reliable backscatter communication while maintaining a compact form factor suitable for surface mounting. The spiral’s symmetry ensures uniform radiation patterns in both azimuth and elevation planes, enabling omnidirectional sensitivity to human motion from any direction within the detection volume.

- describe matching network and IC

The matching network is a T-shaped impedance transformer designed to conjugately match the complex input impedance of the integrated circuit to the output impedance of the spiral antenna. This network comprises discrete inductive and capacitive elements arranged in a symmetrical topology to cancel reactive components and maximize power transfer efficiency under passive operation. The integrated circuit is a commercial RFID chip configured to modulate the incident carrier wave in response to received energy, encoding data into the backscattered signal via load modulation. The chip is selected for its low power consumption, high sensitivity, and compatibility with standard RFID communication protocols.

- describe RFID transceiver device

The RFID transceiver device integrates the emitter and receiver functions into a single housing, enabling synchronized transmission and reception operations. The transceiver includes a frequency synthesizer for stable carrier generation, a power amplifier for signal transmission, a low-noise amplifier for signal reception, and a mixer for down-conversion of the backscattered signal to baseband. Digital control circuitry manages timing, modulation, and data acquisition, ensuring precise synchronization between transmitted and received signals. The transceiver is interfaced with an external computing platform via wired or wireless communication links for data processing and system control.

- describe RFID reflection analysis system

The RFID reflection analysis system is a software and hardware module that processes the amplitude and phase data of received backscattered signals to detect and classify human activity. The system first establishes a baseline model of the environment during a calibration phase, capturing the steady-state reflection signature in the absence of movement. During operation, the system continuously compares real-time measurements against this baseline, identifying deviations attributable to human motion. The analysis employs statistical filtering, frequency-domain decomposition, and pattern recognition algorithms to distinguish between different types of movement, such as respiration, limb motion, or torso rotation, based on their unique spectral and temporal signatures.

- introduce method for remote sensing

A method for remote sensing of human activity is disclosed, comprising the steps of installing a circularly polarized RFID reader antenna and a circularly polarized passive RFID tag in a fixed spatial configuration, emitting circularly polarized radio frequency waves from the reader antenna, receiving the waves by the tag, modulating the waves via backscatter in response to environmental changes, re-emitting the modulated waves with reversed polarization handedness, receiving the re-emitted waves at the reader antenna, analyzing the amplitude and phase variations of the received signal, and performing a contextual action based on the analysis, such as triggering an alert, logging an event, or adjusting environmental controls.

- describe method features

The method features include the suppression of direct line-of-sight signals through polarization mismatch, the enhancement of human-body-reflected signals through polarization reversal, the use of a single tag-reader pair to cover a large detection volume, the elimination of the need for user-worn devices, and the ability to detect subtle movements such as respiration and finger motion without visual or biometric data capture. The method further includes a calibration procedure that models the static environment’s reflection signature, enabling the isolation of dynamic changes caused by human activity. The method is robust to environmental noise, temperature fluctuations, and minor changes in ambient reflectivity, and operates continuously without requiring recalibration under normal conditions.

- introduce apparatus for remote sensing

An apparatus for remote sensing is disclosed, comprising a circularly polarized RFID reader antenna, a circularly polarized passive RFID tag, an RFID transceiver unit, and a reflection analysis system. The reader antenna and tag are mounted in fixed positions relative to each other, with their polarization states configured to be cross-circularly polarized. The transceiver unit emits and receives radio frequency signals, while the reflection analysis system processes the received signal to detect human activity. The apparatus is configured for installation in residential, clinical, or commercial environments without requiring structural modification, and is designed to operate autonomously with minimal user interaction.

## DETAILED DESCRIPTION

- introduce RFID system for object detection

An RFID system for object detection is disclosed wherein the detection of human presence and motion is achieved through the analysis of polarization-modulated backscattered radio frequency signals. Unlike conventional RFID systems that rely on line-of-sight communication between reader and tag to identify tagged objects, this system exploits the physical interaction between electromagnetic waves and the human body to generate detectable signal variations. The system does not require objects to be tagged or tracked; instead, it detects the presence of any human body within the interrogation volume by analyzing how the body alters the polarization state of reflected radio waves. This enables the detection of untagged, uncooperative, or unknown subjects without compromising privacy or requiring active participation.

- describe privacy benefits of RFID system

The privacy benefits of this RFID system stem from its complete absence of visual, auditory, or biometric data collection. No images, voice recordings, heart rate measurements, or identifiable physiological signals are captured or stored. The system responds only to the gross dielectric and conductive properties of the human body as they perturb the polarization of radio waves, yielding no information about identity, gender, age, or specific physical characteristics. The raw output consists solely of amplitude and phase variations in a radio signal, which, when processed, reveal only the timing, duration, and type of movement—not the individual performing it. This design inherently satisfies data minimization principles under global privacy regulations and eliminates the risk of surveillance misuse, making it suitable for deployment in sensitive environments such as bedrooms, bathrooms, and healthcare facilities.

- detail RFID emitter and tag installation

The RFID emitter and tag are installed in fixed, non-movable positions within the environment to be monitored. The emitter is mounted on one wall or surface, while the tag is mounted on an opposing wall, ceiling, or floor, such that their relative orientation forms a cross-polarized configuration. Installation is non-invasive, requiring only adhesive mounting or mechanical fastening to existing structures. The emitter and tag are positioned to avoid direct line-of-sight alignment, ensuring that the primary signal path between them is obstructed by ambient structures, thereby forcing any detectable signal to arise from reflections off moving human bodies. The distance between emitter and tag may range from one to five meters, depending on the size of the room and desired coverage area.

- explain circular polarization of RF waves

Circular polarization refers to the rotational orientation of the electric field vector of a radio wave as it propagates through space. In right-hand circular polarization (RHCP), the electric field rotates clockwise when viewed in the direction of propagation, while in left-hand circular polarization (LHCP), it rotates counterclockwise. When a circularly polarized wave reflects off a conductive or high-permittivity surface such as the human body, the handedness of the polarization reverses: RHCP becomes LHCP and vice versa. This reversal is a fundamental property of electromagnetic wave interaction with dielectric media and is exploited in this system to distinguish human-reflected signals from direct-path signals, which retain their original polarization and are therefore suppressed by polarization mismatch.

- describe RFID tag configuration

The RFID tag is configured as a passive, battery-free device with a spiral antenna, a conjugate-matching network, and an integrated circuit. The spiral antenna is etched onto a rigid FR4 substrate and is designed to resonate across the entire UHF RFID band, ensuring consistent performance regardless of frequency hopping within the reader’s transmission protocol. The matching network is composed of surface-mount inductors and capacitors arranged in a T-configuration to cancel the reactive component of the chip’s impedance, maximizing power transfer from the antenna to the circuit. The integrated circuit is a commercially available RFID chip capable of load modulation, enabling it to encode data into the reflected signal without requiring an internal power source.

- detail passive RFID tag operation

The passive RFID tag operates by harvesting energy from the incident circularly polarized radio wave emitted by the reader. This harvested energy powers the integrated circuit, which then modulates the impedance of the antenna in a predefined pattern, causing the tag to re-radiate a portion of the incident energy with altered amplitude and phase. The re-radiated signal retains the original frequency but carries the modulation signature of the chip. When the tag is exposed to a human body, the reflection of the incident wave off the body alters the electromagnetic environment around the tag, inducing changes in the amplitude and phase of the re-radiated signal. These changes are captured by the reader and analyzed to infer human activity.

- explain modulated RF wave emission

The modulated RF wave emission from the tag is a low-power, backscattered signal whose amplitude and phase are perturbed by the proximity and motion of the human body. The modulation is not intentional data transmission but rather an unintentional consequence of the body’s interaction with the electromagnetic field surrounding the tag. As the human body moves, it alters the local dielectric constant and conductivity, which in turn affects the antenna’s loading, resonant frequency, and radiation efficiency. These minute changes are encoded into the reflected signal as variations in amplitude and phase, which are measurable by the reader’s high-sensitivity receiver.

- describe RFID reflection analysis system

The RFID reflection analysis system is a computational module that receives raw amplitude and phase data from the reader and transforms it into actionable insights about human activity. The system first performs a calibration phase, during which it records the steady-state reflection signature of the environment in the absence of movement. This signature serves as a baseline against which all subsequent measurements are compared. During operation, the system subtracts the baseline from real-time measurements to isolate dynamic changes caused by human motion. It then applies Fourier transforms, statistical anomaly detection, and machine learning classifiers to identify patterns corresponding to specific activities such as respiration, head movement, or arm waving.

- detail calibration process

The calibration process involves placing the system in the target environment and allowing it to collect a minimum of ten seconds of continuous signal data under static conditions, during which no human movement occurs. The system samples the amplitude and phase of the backscattered signal across all frequency channels used by the reader, creating a comprehensive model of the environment’s electromagnetic signature. This model includes the effects of fixed reflectors such as walls, furniture, and appliances, and is stored in memory as a reference. Once calibrated, the system enters monitoring mode, where it continuously compares incoming data to the calibrated model to detect deviations indicative of human activity.

- explain steady-state reflection environment

The steady-state reflection environment refers to the electromagnetic condition of the space when no human movement is present. In this state, all reflections originate from fixed objects such as walls, floors, and furniture, and the resulting signal amplitude and phase remain temporally stable over time. The steady-state model captures the cumulative effect of all multipath reflections, including those caused by temperature variations, humidity changes, or minor structural shifts. By modeling this baseline, the system can distinguish between environmental noise and true human-induced perturbations, ensuring high detection accuracy and low false alarm rates.

- analyze variations in RF waves for object properties

Variations in the amplitude and phase of the reflected RF waves are analyzed to infer the physical properties and motion characteristics of the object causing the perturbation. The magnitude of the variation correlates with the size and dielectric contrast of the object, while the temporal pattern of the variation encodes the type and speed of motion. For example, respiration produces low-frequency, periodic oscillations due to chest expansion, whereas arm waving generates higher-frequency, transient spikes. The system uses these signatures to classify activities without requiring prior knowledge of the subject’s identity or behavior.

- describe benefits of circular polarization

The use of circular polarization provides several key benefits over linear polarization. First, it enables robust discrimination between direct and reflected signals through polarization reversal, significantly improving signal-to-noise ratio. Second, it eliminates orientation sensitivity, allowing the tag to be mounted in any physical orientation without loss of performance. Third, it reduces interference from multipath reflections off non-human objects, which typically preserve their polarization state. Fourth, it enhances detection range by amplifying the human-reflected signal relative to background noise, enabling coverage of large areas with a single tag-reader pair.

- highlight no device attachment requirement

The system requires no device to be attached to, worn by, or carried by the individual being monitored. This eliminates the need for user compliance, reduces the risk of device loss or malfunction, and removes psychological barriers associated with continuous surveillance. The passive nature of the tag ensures it requires no maintenance, no battery replacement, and no software updates, making it ideal for long-term deployment in homes, hospitals, and assisted living facilities.

- reference external publication for further details

For additional technical details regarding the design of spiral antennas, impedance matching networks, and polarization-based signal discrimination, reference is made to publicly available literature on microwave antenna theory and passive RFID system design, including works on circular polarization in backscatter communication and dielectric interaction modeling.

- introduce system 100 for RFID-based remote sensing

System 100 comprises a circularly polarized RFID reader antenna, a circularly polarized passive RFID tag, an RFID transceiver unit, and a reflection analysis system, all interconnected to enable remote, non-contact human activity detection. The system is designed for installation in fixed environments and operates autonomously without user intervention. The reader antenna and tag are mounted on opposing surfaces, with their polarization states configured to be cross-circularly polarized, ensuring that direct signals are suppressed while human-reflected signals are enhanced.

- detail RFID reader device and antenna

The RFID reader device includes a high-frequency transmitter, a low-noise receiver, a digital signal processor, and a circularly polarized antenna. The antenna is designed with a helical or spiral geometry to produce uniform circular polarization across the operational bandwidth. The reader is capable of frequency hopping within the 902–928 MHz band to avoid interference and is synchronized with the tag’s modulation timing to ensure accurate signal capture.

- describe RFID transmitter and receiver

The RFID transmitter generates a continuous wave signal at a fixed power level, which is radiated by the circularly polarized antenna. The receiver is a superheterodyne architecture with a low-noise amplifier, mixer, and analog-to-digital converter, capable of resolving phase variations as small as 0.1 degrees and amplitude variations as small as 0.1 dB. The receiver operates continuously, sampling the backscattered signal at a rate of up to 250 samples per second.

- explain circular-polarized RF wave emission

The circularly polarized RF wave emission is generated by the reader antenna in a controlled, continuous manner, with a fixed handedness—either right-hand or left-hand. The wave propagates through the environment and interacts with the passive tag, which re-radiates a portion of the energy. When a human body enters the field, it reflects a portion of the incident wave with reversed handedness, which is then received by the reader antenna and analyzed for activity signatures.

- detail RFID reader antenna positioning

The RFID reader antenna is positioned on a vertical surface such as a wall or door frame, oriented such that its polarization axis is perpendicular to the polarization axis of the tag. The antenna is mounted at a height of approximately 1.2 meters above the floor to align with the center of mass of a standing human. The positioning ensures that the direct line-of-sight path between reader and tag is obstructed by ambient structures, forcing any detectable signal to arise from reflections off the human body.

- describe reflected RF wave polarization reversal

When a circularly polarized radio wave reflects off a human body, its handedness is reversed due to the dielectric and conductive properties of skin and tissue. A right-hand circularly polarized wave incident on the body becomes left-hand circularly polarized upon reflection, and vice versa. This reversal enables the reader antenna, which is configured to receive the opposite handedness, to efficiently capture the reflected signal while rejecting the direct-path signal, which retains its original polarization.

- explain RFID tag operation

The RFID tag operates passively, harvesting energy from the incident circularly polarized wave to power its integrated circuit. The circuit modulates the antenna’s impedance, causing the tag to re-radiate a portion of the incident energy with a modulated amplitude and phase. When a human body enters the vicinity of the tag, it alters the electromagnetic environment, inducing changes in the modulation pattern that are detectable by the reader.

- detail modulated RF wave retransmission

The modulated RF wave retransmission occurs without active transmission; instead, the tag acts as a passive reflector whose reflection coefficient is modulated by the integrated circuit. The modulation is induced by the body’s proximity and motion, which perturb the antenna’s resonance and coupling characteristics. The resulting retransmitted signal carries the signature of the human movement, encoded in variations of amplitude and phase.

- describe reflected modulated RF wave analysis

The reflected modulated RF wave is analyzed by comparing its amplitude and phase to a pre-established baseline model of the environment. Deviations from this baseline are identified as indicators of human activity. The analysis employs statistical methods to distinguish between transient events and sustained motion, and applies classification algorithms to identify specific activities such as respiration, head nodding, or hand waving.

- detail RFID reflection analysis system components

The RFID reflection analysis system comprises a data acquisition module, a calibration engine, a signal processing unit, and an activity classifier. The data acquisition module collects raw amplitude and phase samples from the reader. The calibration engine generates the steady-state model. The signal processing unit applies filtering, frequency transformation, and noise reduction. The activity classifier uses machine learning models trained on known activity signatures to categorize detected events.

- explain calibration system operation

The calibration system operates by collecting a continuous stream of signal data during a period of environmental stability, typically lasting ten seconds or longer. It computes the mean and variance of amplitude and phase across all frequency channels and stores these values as a reference model. This model is updated periodically to account for slow environmental changes, such as temperature drift or furniture movement, ensuring long-term operational accuracy.

- describe steady-state model creation

The steady-state model is created by averaging the amplitude and phase measurements collected over multiple cycles of the reader’s frequency-hopping sequence. Each frequency channel is treated independently, and the resulting model is a multi-dimensional vector representing the electromagnetic signature of the environment in the absence of human motion. This model serves as the baseline against which all subsequent measurements are compared.

- detail reflection analysis system operation

The reflection analysis system operates by subtracting the steady-state model from real-time measurements to isolate dynamic changes. It then applies a sliding window algorithm to detect temporal patterns, performs Fourier analysis to extract frequency components, and classifies the resulting features using pre-trained models. Output is generated as event logs, alerts, or control signals, depending on system configuration.

- explain external interface operation

The external interface allows the system to communicate with external devices such as servers, cloud platforms, or home automation systems. Data is transmitted via Wi-Fi, Bluetooth, or wired Ethernet, and may include activity timestamps, event types, and statistical summaries. The interface also accepts configuration commands, such as sensitivity thresholds or activity classification rules.

- describe network and cloud-based server system

The network and cloud-based server system receives data from multiple RFID reflection analysis systems and aggregates it into a centralized monitoring platform. The server performs long-term trend analysis, generates alerts for abnormal behavior, and provides remote access to caregivers or family members. Data is encrypted in transit and at rest, and access is controlled via authentication protocols to ensure privacy compliance.

- introduce local computerized device

A local computerized device is provided for on-site processing and control. It may be a dedicated embedded system or a general-purpose computer running monitoring software. The device receives raw data from the RFID reader, performs real-time analysis, and triggers local responses such as lighting, alarms, or notifications without requiring internet connectivity.

- detail system 200 for RFID-based remote sensing

System 200 is an alternative embodiment of the system, comprising multiple circularly polarized tag-reader pairs arranged in a grid pattern to cover a larger area. Each pair operates independently, with its own calibration model and analysis engine. Data from all pairs is fused to create a spatial map of human activity, enabling localization and tracking of movement across multiple rooms or zones.

- describe physical arrangement of system components

The physical arrangement of system components is designed for minimal visual impact and maximum detection efficiency. The reader antenna and tag are mounted on opposing surfaces, with their polarization axes oriented orthogonally. The tag is placed at least one meter from the reader to ensure sufficient separation for polarization discrimination. The system is installed at standard human height to optimize detection of torso and limb motion.

- explain system installation in various environments

The system is installed in diverse environments including residential bedrooms, hospital patient rooms, nursing homes, and office workspaces. Installation requires no structural modification and can be completed in minutes. The system adapts to different room geometries and material compositions through automated calibration, ensuring consistent performance regardless of environment.

- detail moving object detection

Moving object detection is achieved by continuously comparing real-time signal measurements to the calibrated steady-state model. Any deviation exceeding a predefined threshold is flagged as a potential movement. The system distinguishes between human motion and non-human disturbances—such as opening a door or moving a chair—by analyzing the temporal profile, frequency content, and spatial consistency of the signal variation.

- describe RFID reader antenna and tag positioning

The RFID reader antenna and tag are positioned such that their direct line-of-sight path is obstructed by fixed structures, ensuring that the primary detectable signal arises from reflections off moving human bodies. The tag is mounted on a wall opposite the reader, with both components aligned at a height of 1.2 meters above the floor. The polarization axes are oriented perpendicularly to each other to maximize polarization mismatch for direct signals.

- explain perpendicular installation of components

Perpendicular installation of the reader antenna and tag ensures that the electric field vectors of the transmitted and received waves are orthogonal, resulting in maximum polarization mismatch for direct-path signals. This configuration suppresses the direct signal by more than 30 dB, while human-reflected signals, which undergo polarization reversal, are received with minimal loss, thereby maximizing signal-to-noise ratio.

- detail system operation in various scenarios

In a bedroom scenario, the system detects respiration and turning during sleep. In a bathroom scenario, it detects entry and exit, as well as prolonged inactivity that may indicate a fall. In a living room, it detects sitting, standing, and walking. In each case, the system operates without visual monitoring, preserving privacy while providing reliable activity logs.

- highlight system flexibility and customization

The system is highly flexible and customizable. The sensitivity threshold, detection range, and activity classification rules can be adjusted remotely via software. Multiple tag-reader pairs can be deployed to cover large or complex environments. The system can be integrated with existing smart home platforms, medical alert systems, or caregiver notification services.

- conclude system description

The described system provides a novel, privacy-preserving method for remote human activity sensing using circularly polarized RFID technology. By exploiting polarization reversal upon reflection from the human body, it achieves high detection accuracy over large areas without requiring wearable devices, visual monitoring, or biometric data collection. The system is low-cost, scalable, and suitable for long-term deployment in sensitive environments.

- introduce RFID system

An RFID system is disclosed for detecting human activity through polarization-based signal discrimination. The system comprises a circularly polarized reader antenna and a circularly polarized passive tag, configured such that their polarization states are cross-oriented to suppress direct-path signals while enhancing human-reflected signals.

- describe perpendicular installation of RFID reader antenna and RFID tag

The RFID reader antenna and RFID tag are installed perpendicularly to each other on opposing surfaces, with their polarization axes oriented at 90 degrees relative to one another. This configuration ensures that the direct-path signal is attenuated by polarization mismatch, while human-reflected signals, which reverse polarization upon reflection, are efficiently received.

- describe alternative physical arrangements of RFID reader antenna and RFID tag

Alternative physical arrangements include mounting the tag on the ceiling and the reader on a wall, or placing both components on the same wall with a reflective surface between them. In all configurations, the relative polarization orientation is maintained to ensure cross-polarization suppression of direct signals.

- introduce RFID tag embodiment

An RFID tag embodiment is disclosed comprising a spiral antenna, a matching network, and an integrated circuit, all integrated onto a single rigid substrate. The tag is passive, battery-free, and designed for permanent surface mounting.

- describe spiral antenna of RFID tag

The spiral antenna of the RFID tag is an Archimedean spiral with a radius of 13 cm, fabricated on a 0.8 mm thick FR4 substrate. The spiral geometry ensures frequency-independent circular polarization over the entire UHF RFID band, with an axial ratio bandwidth exceeding 105 degrees.

- describe matching network of RFID tag

The matching network is a T-shaped impedance transformer composed of surface-mount inductors and capacitors, designed to conjugately match the 20–145j ohm impedance of the integrated circuit to the spiral antenna, maximizing power transfer efficiency under passive operation.

- describe integrated circuit of RFID tag

The integrated circuit is an Impinj Monza R6 RFID chip, configured for load modulation and low-power operation. The chip modulates the antenna’s reflection coefficient in response to harvested energy, encoding environmental perturbations into the backscattered signal.

- describe use of multiple RFID tags

Multiple RFID tags may be deployed in a single environment to extend coverage, improve localization accuracy, or provide redundancy. Each tag operates independently with its own calibration model, and data from all tags is fused to create a comprehensive activity map.

- introduce circular-polarized passive RFID tag embodiment

A circular-polarized passive RFID tag embodiment is disclosed, comprising a spiral antenna, a conjugate-matching network, and an integrated circuit, all integrated into a single, low-profile, surface-mountable unit.

- describe spiral antenna of circular-polarized passive RFID tag

The spiral antenna of the circular-polarized passive RFID tag is an Archimedean spiral with a radius of 13 cm, fabricated on a rigid FR4 substrate. The antenna exhibits a realized gain of 3.3 dBi and an axial ratio bandwidth of 105 degrees, ensuring consistent circular polarization across the entire UHF RFID band.

- describe matching network of circular-polarized passive RFID tag

The matching network of the circular-polarized passive RFID tag is a T-shaped network composed of discrete inductors and capacitors, designed to cancel the reactive component of the integrated circuit’s impedance and maximize power transfer efficiency under passive operation.

- introduce method for RFID-based remote sensing using cross circular polarization

A method for RFID-based remote sensing using cross circular polarization is disclosed, comprising the steps of installing a circularly polarized RFID reader antenna and a circularly polarized passive RFID tag in a fixed physical environment, emitting circularly polarized radio frequency waves from the reader antenna, receiving the waves by the tag, modulating the received waves via backscatter in response to environmental perturbations, re-emitting the modulated waves with reversed polarization handedness, receiving the re-emitted waves at the reader antenna, analyzing the amplitude and phase variations of the received signal, and performing a contextual action based on the analysis.

- install RFID reader antenna and RFID tag in physical environment

The RFID reader antenna and RFID tag are installed in fixed positions within a physical environment, with their polarization states configured to be cross-circularly polarized. The tag is mounted on a surface opposite the reader, ensuring that the direct line-of-sight path is obstructed.

- emit circular-polarized RF waves from RFID reader antenna

The RFID reader antenna emits continuous circularly polarized radio frequency waves at a frequency within the UHF band, with a power level sufficient to energize the passive tag.

- receive circular-polarized RF waves by RFID tag

The circularly polarized RF waves are received by the spiral antenna of the passive RFID tag, which harvests energy from the incident wave to power its integrated circuit.

- modulate received circular-polarized RF waves by RFID tag

The integrated circuit modulates the impedance of the antenna in response to the harvested energy, causing the tag to re-radiate a portion of the incident wave with a modulated amplitude and phase.

- emit modulated circular-polarized RF waves from RFID tag

The modulated circularly polarized RF waves are re-emitted from the tag, with their polarization handedness preserved unless altered by interaction with a human body.

- receive modulated circular-polarized RF waves by RFID reader antenna

The RFID reader antenna receives the modulated circularly polarized RF waves. If the waves have been reflected by a human body, their handedness is reversed, enabling efficient reception by the reader antenna.

- analyze received modulated circular-polarized RF waves

The received modulated circularly polarized RF waves are analyzed for amplitude and phase variations relative to a pre-established steady-state model of the environment.

- perform action based on analysis of received modulated circular-polarized RF waves

An action is performed based on the analysis, such as generating an alert, logging an event, adjusting environmental controls, or notifying a caregiver.

- introduce method for calibrating RFID-based remote sensing using cross-circular polarization

A method for calibrating RFID-based remote sensing using cross-circular polarization is disclosed, comprising the steps of entering a calibration mode, collecting data on the steady-state environment, creating a steady-state model of the physical environment, transitioning to a moving object monitoring mode, subtracting the steady-state model from real-time measurements, determining changes in phase due to the presence of a moving object, determining whether the physical environment is empty of moving objects, and optionally returning to calibration mode if environmental conditions change significantly.

- enter calibration mode

The system enters a calibration mode upon initialization or upon user command, during which no human activity is permitted in the monitored environment.

- collect data on steady-state environment

The system collects a continuous stream of amplitude and phase data from the RFID reader over a period of at least ten seconds, sampling across all frequency channels used by the reader.

- create steady-state model of physical environment

The system computes the mean and variance of amplitude and phase for each frequency channel and stores these values as a steady-state model representing the electromagnetic signature of the environment in the absence of human motion.

- transition to moving object monitoring mode

After the steady-state model is created, the system transitions to moving object monitoring mode, during which real-time measurements are continuously compared to the model.

- subtract steady-state model from data obtained from reflected modulated RF waves

The system subtracts the steady-state model from real-time measurements to isolate dynamic changes attributable to human movement.

- determine change in phase due to presence of moving object

Changes in phase are analyzed to detect subtle movements such as respiration or finger motion, which produce low-frequency, periodic phase shifts.

- determine whether physical environment is empty of moving objects

The system determines whether the environment is empty by verifying that all measured deviations fall below a predefined noise threshold.

- enter calibration mode or remain in moving object monitoring mode

If environmental changes exceed a threshold—such as due to furniture movement or temperature drift—the system automatically re-enters calibration mode. Otherwise, it remains in monitoring mode.

- discuss variations of methods and systems

Variations of the disclosed methods and systems include the use of multiple tag-reader pairs, integration with machine learning models for activity classification, deployment in multi-room environments, and adaptation for use in industrial or security applications.

- discuss omission, substitution, or addition of procedures or components

Omission of the calibration step is possible in static environments with minimal environmental variation. Substitution of the spiral antenna with a patch antenna or meandered structure is feasible with equivalent polarization performance. Addition of a local display or audio alert is possible for user feedback.

- discuss combination of features from different embodiments

Features from different embodiments may be combined, such as using multiple circularly polarized tags with a single reader, or integrating the reflection analysis system with a cloud-based analytics platform.

- discuss well-known processes, structures, and techniques

Well-known processes such as frequency hopping, impedance matching, and load modulation are employed in accordance with standard RFID practices. Structures such as FR4 substrates and surface-mount components are selected for cost-effectiveness and manufacturability.

- discuss changes in function and arrangement of elements

Changes in the function or arrangement of elements are contemplated, such as relocating the reader antenna to the ceiling or arranging multiple tags in a linear array. Such modifications do not depart from the inventive concept.

- discuss modifications, alternative constructions, and equivalents

Modifications, alternative constructions, and equivalents are intended to be encompassed within the scope of the invention. For example, the use of different dielectric substrates, alternative matching network topologies, or different polarization schemes may be employed without departing from the core principle of polarization-based signal discrimination for human activity detection.