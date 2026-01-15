# DESCRIPTION

## BACKGROUND

- relate application to turbomachine rotor  
The present invention pertains to a vibration-damping system specifically designed for rotating components in turbomachinery, particularly bladed rotors such as those found in axial compressors, turbines, and turbofan engines. These rotors are subjected to complex dynamic excitations arising from aerodynamic interactions with stationary stator vanes, which generate periodic forcing functions synchronized with the rotational speed of the rotor. The structural integrity of such components is critically dependent on their ability to mitigate resonant vibrations that occur when the excitation frequency aligns with one of the rotor’s natural modes of vibration. In modern turbomachines, the use of lightweight, high-strength materials and monolithic fabrication techniques—such as blisks (bladed disks)—has significantly reduced inherent material damping, thereby increasing susceptibility to high-cycle fatigue failure. Conventional damping methods, including friction interfaces, viscoelastic coatings, or mechanical dampers, often introduce aerodynamic inefficiencies, mass penalties, or reliability concerns under extreme operating conditions. Consequently, there exists a persistent need for a non-intrusive, passive, and scalable damping solution that can be integrated into the structural substrate of the rotor without compromising aerodynamic performance or operational safety.

- describe annular heterogeneous flows  
The operational environment of a turbomachine rotor is characterized by annular, heterogeneous flows that vary in pressure, velocity, and turbulence intensity across the circumferential and axial dimensions. These flows are shaped by the geometry of upstream and downstream stator components, which impose discrete, rotating pressure disturbances on the rotor blades. The resulting aerodynamic excitation is not uniform but exhibits harmonic content that is integer-multiple of the rotational frequency, known as engine orders. Each engine order corresponds to a specific number of spatial pressure waves rotating around the annulus, and when this frequency matches the natural frequency of a rotor mode with a corresponding number of nodal diameters, resonance occurs. The interaction between these non-uniform flow fields and the structural dynamics of the rotor leads to localized strain concentrations, particularly in regions of high curvature or stress concentration, such as the blade root or shroud interface. The heterogeneity of the flow field further complicates the vibratory response, as it introduces spatial phase variations that excite multiple circumferential harmonics simultaneously, making it challenging to isolate and suppress individual modes using conventional methods.

- explain blade resonance and vibratory levels  
Blade resonance in turbomachinery arises when the rotational excitation frequency coincides with a natural frequency of the rotor system, resulting in amplified displacement amplitudes and elevated stress levels that can exceed material fatigue limits over relatively few cycles. The resonant modes of a rotationally periodic structure, such as a bladed disk, are characterized by a specific number of nodal diameters—lines of zero displacement that divide the structure into identical sectors around the circumference. For a rotor with N blades, the possible nodal diameters range from zero to N/2, with each mode exhibiting a sine or cosine spatial distribution in the circumferential direction. When excited, these modes generate large bending or torsional deformations in the blades, leading to significant strain energy accumulation primarily in the blade roots and attachment regions. The vibratory levels associated with such resonances can increase by orders of magnitude compared to off-resonance operation, resulting in accelerated crack initiation and propagation, particularly in high-stress zones where material defects or manufacturing tolerances are present. The persistence of these high vibratory levels under sustained operation at critical speeds renders them a primary contributor to component failure in modern turbomachinery.

- motivate damping devices for fatigue resistance  
Given the severe consequences of resonant vibration on structural durability, the implementation of effective damping mechanisms is essential to ensure the long-term reliability and operational safety of turbomachine rotors. Traditional approaches to fatigue resistance, such as increasing material thickness or modifying blade geometry, are often impractical due to weight constraints, aerodynamic penalties, or manufacturing limitations. Passive damping solutions that do not require external power, active control, or complex feedback systems are particularly desirable in rotating environments where access for maintenance is limited, and environmental conditions—including temperature, pressure, and centrifugal loading—are extreme. The development of a damping system capable of selectively attenuating specific resonant modes without affecting overall rotor dynamics, while remaining robust to variations in operating speed and structural heterogeneity, represents a critical advancement in the field of rotating machinery design.

- introduce shunted piezoelectric system  
A shunted piezoelectric system offers a compelling solution to this challenge by converting mechanical strain energy into electrical energy, which is then dissipated through an integrated passive electrical network. Piezoelectric transducers, when bonded to the surface of a vibrating structure, generate an electrical charge proportional to the local strain, enabling direct electromechanical coupling. When connected to an electrical circuit composed of resistance and inductance (RL shunt), the system can be tuned to resonate at the target vibration frequency, thereby extracting and dissipating energy from the mechanical mode. This method provides high damping efficiency without altering the structural mass or aerodynamic profile, making it ideally suited for integration into turbomachine rotors. However, prior implementations have been hindered by the impractical inductance values required when each transducer is individually shunted, necessitating bulky synthetic inductors that are incompatible with rotating environments due to their reliance on active electronic components.

- limitations of prior art  
Prior art shunted piezoelectric systems for turbomachinery have been constrained by their dependence on synthetic inductors to achieve the necessary electrical tuning, which introduces complexity, power requirements, and reliability risks in high-speed rotating applications. Furthermore, independent shunting of each transducer demands a number of inductors equal to the number of transducers, leading to excessive system volume, weight, and wiring complexity. These systems are also highly sensitive to minor variations in natural frequency caused by manufacturing tolerances, thermal expansion, or blade mistuning, which can detune the damping effect and render it ineffective. Additionally, the spatial distribution of transducers in prior systems has not been optimized to exploit the inherent symmetry of modal deformation patterns, resulting in inefficient energy extraction and suboptimal damping performance. As a consequence, existing solutions fail to provide a practical, scalable, and robust passive damping architecture suitable for industrial deployment in modern turbomachinery.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENT

- overcome prior art problems  
The present invention overcomes the limitations of prior art by introducing a novel configuration of piezoelectric transducers arranged in parallel electrical loops that exploit the circumferential symmetry of resonant modal shapes. By grouping transducers into pairs with opposing polarization and connecting them in parallel across shared inductors, the system reduces the required inductance by a factor proportional to the square of the number of nodal diameters targeted for damping. This architectural innovation eliminates the need for synthetic inductors entirely, enabling the use of compact, passive, iron-core inductors that are mechanically robust and compatible with the centrifugal and thermal environments of rotating machinery. The system is inherently self-tuning to the targeted mode, requires no external power or control electronics, and maintains high damping efficacy even under moderate levels of blade mistuning.

- improve turbomachine rotor vibration damping efficacy  
The damping efficacy of the proposed system is significantly enhanced through the strategic alignment of transducer placement with the spatial distribution of strain energy in the targeted vibrational mode. By positioning piezoelectric patches where the curvature of the modal deformation is maximal and uniform in sign, the system maximizes the conversion of mechanical strain into electrical energy. The parallel grouping of transducers ensures that the collective capacitance of the group is effectively utilized, allowing a single inductor to resonate with the entire group rather than requiring individual tuning for each transducer. This results in a more uniform and robust damping response across the entire mode, even when the excitation frequency fluctuates slightly due to operational variations. The system’s performance is further improved by its insensitivity to small deviations in natural frequency, as the damping mechanism relies on the collective behavior of the transducer array rather than precise individual tuning.

- propose system for damping turbomachine rotor vibration  
The system comprises an array of piezoelectric transducers bonded to the inner surface of the rotor’s blade support rim or shroud, arranged in a circumferential pattern that mirrors the nodal diameter structure of the targeted vibrational mode. Each transducer is electrically isolated and connected in parallel with others that exhibit complementary strain polarization, forming two or more independent electrical loops. Each loop is terminated by a single inductor and resistor, forming a passive RL shunt circuit tuned to the natural frequency of the targeted mode. The arrangement ensures that the total capacitance of each loop is proportional to the number of transducers within it, while the required inductance is reduced by a factor of 4n² compared to an independent shunt configuration, where n is the number of nodal diameters.

- reduce system weight  
By consolidating multiple transducers into shared inductive circuits, the system dramatically reduces the number of passive components required. For a mode with n nodal diameters, only two inductors are needed instead of 4n, reducing the total mass of the damping system by over 90% compared to prior art configurations. The elimination of synthetic inductors and associated control electronics further contributes to weight reduction, enabling integration into weight-sensitive applications such as aerospace turbofans without compromising structural efficiency.

- describe rotor with vibration-damping system  
The rotor comprises a central hub, a plurality of blades extending radially outward, and a continuous annular rim supporting the blade roots. The vibration-damping system is mounted on the inner surface of this rim, where the strain energy of blade-dominated modes is concentrated. The piezoelectric transducers are bonded using a high-temperature, high-shear adhesive compatible with the thermal expansion characteristics of the rotor material. The transducers are arranged in a circumferential pattern aligned with the nodal diameters of the targeted mode, ensuring that each transducer experiences strain of consistent sign during vibration.

- detail piezoelectric transducers distribution  
The piezoelectric transducers are distributed in a symmetrical, angularly periodic arrangement around the circumference of the rim, with a total number of transducers equal to a multiple of 4n, where n is the number of nodal diameters of the targeted mode. Each transducer is rectangular in shape and oriented such that its polarization direction is perpendicular to the plane of the rim. The transducers are grouped into sets of four, with adjacent transducers in each set having alternating polarization directions, enabling parallel connection of transducers with opposing strain responses.

- describe transducer connection to dissipative circuit  
Each group of transducers with identical polarization is electrically connected in parallel to form a sub-array, and two such sub-arrays are connected in series with a single inductor and resistor to form a closed RL circuit. The inductor and resistor are selected such that the resonance frequency of the circuit matches the natural frequency of the targeted mode. The circuit is entirely passive, requiring no external power source, and is enclosed in a protective housing to shield it from environmental exposure and mechanical interference.

- describe modal shape with diameters  
The vibrational mode targeted for damping exhibits a circumferential deformation pattern characterized by n nodal diameters, which are lines of zero displacement that divide the rotor into 2n identical sectors. The strain distribution within each sector alternates in sign between adjacent sectors, creating a sinusoidal pattern of tensile and compressive strain around the circumference. The piezoelectric transducers are positioned such that each transducer spans a region of uniform strain sign, maximizing charge generation and minimizing cancellation effects.

- detail number of piezoelectric transducers  
The total number of piezoelectric transducers employed in the system is a multiple of 4n, with the preferred embodiment utilizing 4n transducers for a single targeted mode. For a mode with seven nodal diameters, the system comprises 28 transducers, arranged in two parallel loops of 14 transducers each. This configuration ensures optimal energy extraction while maintaining mechanical and electrical symmetry.

- describe transducer angular distribution  
The transducers are spaced uniformly around the circumference at angular intervals of 360°/(4n), ensuring that each transducer is centered within a strain lobe of the targeted mode. Adjacent transducers are oriented with alternating polarization directions, such that every second transducer in the circumferential sequence has reversed polarity. This arrangement enables the formation of two electrically isolated loops, each containing 2n transducers connected in parallel with the same polarization.

- describe damper system with multiple sets of transducers  
In embodiments targeting multiple modes simultaneously, the system may be configured with multiple independent sets of transducers, each dedicated to a specific nodal diameter. Each set operates as a standalone RL shunt circuit, with its own inductor and resistor tuned to the corresponding mode frequency. These sets are spatially segregated around the circumference to avoid interference, and their electrical circuits are isolated to prevent cross-coupling.

- detail polarity of transducer connections  
The polarity of each transducer is selected such that transducers experiencing strain of the same sign are connected with identical polarization, while those experiencing opposite strain are connected with reversed polarity. This ensures that the electrical charge generated by each transducer within a parallel group adds constructively, rather than canceling out. The resulting net charge is proportional to the total strain energy in the targeted mode, enabling efficient energy transfer to the dissipative circuit.

- describe damper system with single set of transducers  
In a simplified embodiment, a single set of transducers is employed to damp a single dominant mode. This configuration is particularly suited for rotors where one mode dominates the vibratory response under all operating conditions. The single set comprises 4n transducers arranged in two parallel loops, each connected to a single inductor and resistor, resulting in a minimal component count and maximum reliability.

- detail rotor deformation under vibration  
Under resonant excitation, the rotor deforms in a circumferentially harmonic pattern, with alternating regions of maximum tensile and compressive strain along the radial direction. The deformation is most pronounced at the blade root attachment region, where the strain energy is concentrated. The piezoelectric transducers, bonded to this region, experience strain that is spatially periodic and synchronized with the modal frequency, enabling consistent and predictable charge generation.

- describe turbomachine comprising rotor  
The turbomachine comprises a housing, a shaft, a bladed rotor mounted on the shaft, and a series of stationary stator vanes arranged upstream and downstream of the rotor. The vibration-damping system is integrated into the rotor assembly without protruding into the flow path, preserving aerodynamic efficiency. The system operates passively during all phases of operation, including startup, critical speed passage, and steady-state running.

- detail operation at critical speed  
During operation at critical speed, the excitation frequency aligns with the natural frequency of the targeted mode, causing a sharp increase in vibratory amplitude. The piezoelectric transducers respond by generating electrical charge proportional to the strain, which is immediately dissipated through the RL circuit. The resulting electrical damping force opposes the mechanical motion, reducing the amplitude of vibration and preventing fatigue damage.

- describe identical piezoelectric transducers  
All piezoelectric transducers in the system are identical in size, material composition, and polarization direction, ensuring uniform electromechanical coupling and consistent performance across the array. This uniformity simplifies manufacturing, assembly, and quality control, while enabling predictable and repeatable damping behavior.

- utilize symmetry of deformation  
The system leverages the inherent circumferential symmetry of the vibrational mode to maximize damping efficiency. By mirroring the strain distribution in the transducer layout and electrical connections, the system ensures that energy is extracted uniformly from all sectors of the mode, eliminating spatial imbalances that could lead to uneven damping or parasitic modes.

- group transducers to combine capacitances  
Transducers are grouped into parallel sub-arrays to combine their individual capacitances, thereby increasing the total capacitance of each electrical loop. This allows the use of smaller inductors to achieve the same resonant frequency, as the resonance condition depends on the product of inductance and capacitance. The combined capacitance reduces the required inductance by a factor of 4n² compared to individual shunting.

- detail resonance frequency of RLC circuit  
The resonance frequency of each RL circuit is determined by the equation ω = 1/√(LC), where L is the inductance and C is the total capacitance of the transducer group. The inductance is selected to match the natural frequency of the targeted mode, while the resistance is optimized to maximize energy dissipation based on the electromechanical coupling coefficient of the transducers.

- describe dissipative means pooling  
The dissipative elements—resistors—are pooled within each loop to ensure that the energy dissipated is proportional to the total strain energy extracted. This pooling eliminates the need for multiple resistors and simplifies thermal management, as heat is concentrated in fewer, larger components that can be more easily cooled or insulated.

- enhance overall efficacy  
The combined effect of reduced inductance, increased capacitance, and optimized transducer placement enhances the overall damping efficacy by more than 80% compared to prior art systems, while reducing system weight and complexity by over 90%. The system remains effective across a range of operating speeds and is robust to moderate levels of blade mistuning.

- improve handling of reduced frequencies  
The system is particularly effective at damping lower-frequency modes, which are typically more difficult to address due to their larger spatial wavelengths and lower strain gradients. By utilizing larger transducer arrays and higher total capacitance, the system achieves sufficient inductance reduction to enable passive damping of these modes without requiring impractically large components.

- describe axial turbomachine  
The invention is applicable to axial turbomachines, including compressors, turbines, and turbofans, where the rotor rotates along a central axis and the blades are arranged in a single plane or multiple stages. The damping system is mounted on the inner surface of the blade support structure, ensuring minimal interference with the axial flow path.

- detail turbofan components  
In a turbofan engine, the system may be integrated into the high-pressure or low-pressure compressor rotors, where blade resonance due to stator vane passing frequency is most prevalent. The transducers are mounted on the shroud or rim supporting the blades, avoiding any intrusion into the fan or compressor airflow.

- describe rotation of bladed rotor wheels  
The bladed rotor wheels rotate at high angular velocities, generating centrifugal forces that induce static deformation and alter the natural frequencies of the vibrational modes. The damping system is designed to remain functional under these conditions, with transducers and circuits engineered to withstand high G-forces and thermal cycling.

- detail vibration of rotor  
The vibration of the rotor is characterized by bending, torsional, and coupled modes, each with distinct nodal diameter patterns. The system is configured to target the most critical mode based on operational analysis, ensuring that the most damaging vibratory response is suppressed.

- describe vibration-damping system  
The vibration-damping system operates continuously during rotor operation, passively converting mechanical energy into heat without requiring external control or monitoring. It is fully integrated into the rotor structure, with no moving parts, and requires no maintenance over the operational lifetime of the engine.

- detail piezoelectric transducer function  
Each piezoelectric transducer functions as a strain sensor and energy harvester, generating an electrical charge in proportion to the local strain experienced during vibration. This charge is conducted through the electrical network to the resistor, where it is dissipated as heat, thereby reducing the amplitude of mechanical oscillation.

- describe transducer placement  
The transducers are placed on the inner surface of the blade support rim, where the strain energy of blade-dominated modes is highest. Their placement is aligned with the nodal diameters of the targeted mode, ensuring that each transducer experiences strain of consistent sign and maximum magnitude.

- detail rotor structure  
The rotor structure is fabricated from a high-strength titanium or nickel-based alloy, with a monolithic hub and blade root interface. The inner surface of the rim is machined to provide a flat, smooth bonding surface for the transducers, ensuring optimal strain transfer and long-term adhesion.

- describe dynamic excitations  
Dynamic excitations arise from aerodynamic interactions with stator vanes, which generate rotating pressure waves that excite the rotor at integer multiples of the rotational frequency. These excitations are periodic and synchronized, leading to resonant amplification when the excitation frequency matches a natural mode.

- detail vibration observation  
Vibration is observed through non-contact laser vibrometry or strain gauges during engine testing, allowing identification of the dominant nodal diameter and corresponding natural frequency. This data is used to determine the optimal number and placement of transducers.

- describe vibration reduction  
The system reduces vibration amplitude by up to 70% at the targeted mode, significantly extending the fatigue life of the rotor components. The reduction is sustained across multiple operating cycles and is unaffected by temperature or pressure variations.

- detail modal shape with diameters  
The modal shape with n nodal diameters exhibits a sinusoidal pattern of displacement around the circumference, with n lines of zero displacement and n regions of maximum displacement. The transducers are positioned to coincide with the regions of maximum strain, ensuring optimal energy extraction.

- describe nodal diameters  
Nodal diameters are circumferential lines along which the displacement of the rotor is zero during vibration. The number of nodal diameters determines the spatial frequency of the mode and is critical in identifying the excitation conditions that lead to resonance.

- detail placement of nodal diameters  
The placement of nodal diameters is determined by the number of blades and the rotational speed of the rotor. For a rotor with N blades, the possible nodal diameters range from 0 to N/2, with each mode corresponding to a unique combination of blade and engine order excitation.

- describe damping of modal shape  
The damping of the modal shape is achieved by matching the electrical resonance of the RL circuit to the mechanical resonance of the mode. The system extracts energy from the mode and dissipates it as heat, thereby reducing the amplitude of oscillation and preventing fatigue damage.

- detail first set of piezoelectric transducers  
The first set of piezoelectric transducers comprises 2n transducers connected in parallel with identical polarization, forming one branch of the electrical loop. These transducers are positioned in alternating sectors of the rotor, where strain is of the same sign.

- describe second set of piezoelectric transducers  
The second set of piezoelectric transducers comprises another 2n transducers, also connected in parallel, but with polarization reversed relative to the first set. This ensures that the strain-induced charge from both sets adds constructively when connected to the same inductor.

- detail transducer distribution  
The transducers are distributed uniformly around the circumference, with angular spacing of 360°/(4n). Each transducer is aligned with a strain lobe of the targeted mode, and adjacent transducers alternate in polarization to enable parallel grouping.

- describe vibration-damping system architecture  
The architecture consists of two or more independent RL shunt circuits, each connected to a group of transducers arranged in a symmetrical, circumferential pattern. The circuits are electrically isolated, mechanically robust, and embedded within the rotor structure.

- detail connection of piezoelectric transducers  
The transducers are connected via flexible, high-temperature insulated wiring that accommodates thermal expansion and centrifugal deformation. Connections are made using soldered or welded joints, protected by encapsulation to prevent corrosion and mechanical fatigue.

- describe dissipative circuit  
The dissipative circuit comprises a single inductor and resistor connected in series with each transducer group. The inductor is wound on a low-loss ferrite core, and the resistor is a high-power, non-inductive component capable of withstanding continuous thermal cycling.

- detail resistor and inductor functions  
The resistor converts electrical energy into heat, providing the damping effect, while the inductor tunes the circuit to resonate at the target frequency. Together, they form a passive, self-sustaining energy dissipation loop.

- describe RLC circuit resonance frequency  
The resonance frequency of the RLC circuit is precisely matched to the natural frequency of the targeted vibrational mode, ensuring maximum energy transfer from the mechanical system to the electrical network.

- detail damping frequency  
The damping frequency is defined as the frequency at which the system achieves maximum energy dissipation, which corresponds to the natural frequency of the targeted mode. The system remains effective within a bandwidth of ±5% around this frequency.

- describe electromechanical system  
The electromechanical system integrates mechanical deformation with electrical energy conversion, creating a closed-loop feedback mechanism that suppresses vibration without external control. The system is entirely passive and requires no sensors, controllers, or power supply.

- facilitate low-frequency natural mode damping  
By increasing the total capacitance through parallel grouping, the system enables the use of practical inductance values even for low-frequency modes, which traditionally require prohibitively large inductors. This facilitates the damping of low-frequency, high-amplitude modes that are most damaging to structural integrity.