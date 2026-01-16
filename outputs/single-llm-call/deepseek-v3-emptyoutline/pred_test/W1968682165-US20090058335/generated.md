Here is the complete patent application following the provided outline and incorporating the invention described in the research paper:

---

# DESCRIPTION  

## BACKGROUND OF INVENTION  

Rotating machinery traditionally relies on mechanical bearings to maintain the connection between the rotor and stator. While mechanical bearings are cost-effective for applications where maintenance access is available and operational demands do not exceed bearing limitations, they are unsuitable for high-performance applications involving extreme conditions such as high speeds, elevated temperatures, or environments where maintenance is impractical. Examples include flywheel energy storage systems in spacecraft or motor-generators integrated into jet engine shafts. In such cases, bearingless motors are preferred due to their ability to operate without mechanical contact, thereby eliminating wear and enabling operation in harsh environments.  

Existing bearingless motor designs typically incorporate separate levitation windings on the stator, distinct from the motor windings. These designs have been implemented in various motor types, including synchronous reluctance, induction, permanent magnet, and switched reluctance motors. A key characteristic of these designs is that the levitation windings have a different number of poles than the motor windings, which decouples the levitation and motoring functions. While this approach allows the motor and magnetic bearings to share the same iron structure—reducing rotor length and increasing bending mode frequencies—it does not fully utilize all available copper for both levitation and torque generation. This limitation restricts the system's flexibility in redistributing power between motoring and levitation functions as operational needs change.  

To address these shortcomings, a novel bearingless motor design has been developed that eliminates the need for separate levitation windings. This innovation enables all motor iron and copper to contribute to either levitation or torque generation, depending on system requirements. The design leverages field-oriented control to independently manage torque and levitation forces, providing greater efficiency and adaptability. Additionally, the motor features a conical air gap, which facilitates force generation in both radial and axial directions, enabling full five-axis levitation.  

## SUMMARY OF INVENTION  

The present invention discloses a fully magnetically levitated bearingless motor that integrates levitation and torque generation functions without requiring separate windings. The motor is wound such that pole pairs remain electrically isolated, allowing independent control of rotor reference frame d-axis currents for each pole pair. By selectively varying these currents, a flux imbalance is induced on the rotor periphery, generating a net force for levitation. This approach ensures that all copper and iron within the motor can be utilized for either levitation or torque production, optimizing resource allocation and system performance.  

A key innovation of this design is the conical air gap, which enables force components to be directed both radially and axially. When two conical motors are combined on a single rotor with opposing cone orientations, the net axial force can be precisely controlled by adjusting the difference in d-axis currents between the top and bottom motors. This configuration achieves full five-axis levitation, eliminating the need for additional axial or radial magnetic bearings.  

The motor is driven by a specialized power electronics system capable of delivering arbitrary rotor reference frame d- and q-axis currents to each pole pair. Although this system requires more switches than a conventional three-phase motor drive, the individual switch ratings can be reduced, allowing for tailored voltage and current specifications based on application needs. Furthermore, the design inherently provides fault tolerance, as the independent control of pole pairs ensures continued operation even if one pole pair fails.  

Control of the motor is accomplished through a magnetic circuit model and finite element analysis, which predict force generation and optimize current distribution. Experimental validation has demonstrated successful five-axis levitation and rotation, confirming the motor's ability to operate without mechanical bearings while maintaining high efficiency and reliability.  

## DETAILED DESCRIPTION  

The bearingless motor of the present invention employs a unique winding configuration in which pole pairs are not internally connected, enabling independent control of each pole pair's rotor reference frame d-axis current. This independence allows for the deliberate creation of flux imbalances on the rotor periphery, which in turn generate controllable net forces for levitation. Traditional motors inherently balance lateral forces to prevent vibrations, but by disconnecting pole pairs, this balance is intentionally disrupted to achieve levitation.  

The motor's power electronics system comprises multiple switches to independently regulate d- and q-axis currents for each pole pair. For example, a prototype implementation utilizes 18 switches to manage three pole pairs, compared to the six switches required for a standard three-phase motor. While this increases complexity, it offers advantages in fault tolerance and flexibility. Switch ratings can be adjusted based on application-specific voltage and current requirements, ensuring optimal performance.  

Force generation is modeled using a magnetic circuit that accounts for the yoke, backiron, air gaps, teeth, and permanent magnets. Simulations demonstrate that applying positive or negative d-axis currents to individual pole pairs produces predictable force vectors. These vectors are phase-dependent on the rotor's electrical angle, and their magnitudes remain constant. By analyzing these vectors, a control scheme is developed to select the optimal pair of force vectors for generating a desired net force. The desired force is then transformed into the appropriate d-axis current commands for the relevant pole pairs.  

The conical air gap design further enhances the motor's capabilities by introducing axial force components. When two conical motors are mounted on a single rotor with opposing cone orientations, the difference in their d-axis currents determines the net axial force. This configuration allows for precise axial positioning without interfering with radial force generation. A control block dedicated to axial force compensation ensures that unintended axial forces from radial control are neutralized, maintaining stability.  

Experimental validation of the prototype motor confirms the feasibility of five-axis levitation. The motor controller, operating with a 0.3 ms sample time, successfully maintains rotor position and rotation. Testing reveals minor discrepancies between simulated and experimental loop gains, attributed to mechanical modes in the motor baseplate. These findings underscore the importance of robust mechanical design in achieving optimal performance.  

In summary, the invention provides a bearingless motor that maximizes the utility of all copper and iron for levitation and torque generation. Its conical air gap enables full five-axis levitation, while its fault-tolerant design ensures reliability in demanding applications. The motor's compact axial length raises bending mode frequencies, simplifying control in high-speed operations. This advancement represents a significant improvement over prior bearingless motor designs, offering enhanced efficiency, flexibility, and performance.  

--- 

This patent application thoroughly describes the invention in formal patent language while adhering to the provided outline. Each section is detailed and self-contained, ensuring the document stands alone without reference to the original research paper.