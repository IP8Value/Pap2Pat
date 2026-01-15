# DESCRIPTION

## FIELD OF THE INVENTION

- relate to resonant devices

The present invention relates to resonant devices, particularly to microelectromechanical systems configured as vibratory gyroscopes that operate in the bulk acoustic wave regime. These devices are designed for high-precision angular rate sensing in environments subject to mechanical shock, vibration, thermal fluctuations, and pressure variations. The invention specifically concerns the structural and operational architecture of a substrate-decoupled bulk-acoustic-wave gyroscope that achieves exceptional environmental stability through a novel mechanical decoupling mechanism, enabling consistent performance across diverse operational conditions without reliance on complex signal processing or external calibration. The resonant elements are fabricated using single-crystal silicon substrates and are engineered to support degenerate in-plane vibrational modes with matched resonance frequencies, high quality factors, and minimal cross-mode coupling. The device is suitable for integration into consumer electronics, automotive inertial navigation systems, industrial robotics, and aerospace instrumentation where long-term drift, bias instability, and vibration-induced errors must be minimized.

## BACKGROUND OF THE INVENTION

- introduce micro-machined vibratory gyroscopes

Micro-machined vibratory gyroscopes have become integral components in modern motion-sensing systems, enabling precise detection of rotational motion in applications ranging from handheld navigation devices to autonomous vehicle control. These sensors rely on the Coriolis effect to convert angular velocity into measurable mechanical displacement, typically by exciting a resonant structure into a known vibrational mode and detecting the induced motion in a perpendicular mode due to rotation. While various architectures have been developed, including tuning-fork and ring-shaped designs, many suffer from sensitivity to environmental perturbations such as temperature drift, mechanical shock, and substrate-borne acoustic noise. These limitations restrict their utility in high-reliability applications where consistent performance under uncontrolled conditions is required.

- motivate popularity of gyroscopes

The popularity of gyroscopes has grown substantially due to the increasing demand for inertial navigation in mobile platforms, including smartphones, drones, and wearable devices. As these systems operate in dynamic, real-world environments, the ability to maintain accurate orientation estimates without external references—such as GPS—has become critical. This has driven the need for gyroscopes with low noise, high bandwidth, and exceptional immunity to external disturbances. Consumer-grade devices, in particular, require compact form factors, low power consumption, and robustness against common environmental stressors, including vibration, acceleration, and thermal cycling.

- describe limitations of TFG technology

Traditional tuning-fork gyroscopes (TFGs) have dominated the market due to their relatively simple fabrication and mature integration with complementary metal-oxide-semiconductor (CMOS) electronics. However, TFGs operate at low frequencies, typically below 10 kHz, rendering them highly susceptible to environmental acceleration and vibration. Their mechanical proof masses respond directly to linear motion, generating spurious outputs that mimic angular rotation. Although acceleration rejection techniques using symmetric proof masses have been employed, these approaches increase device footprint, complicate fabrication, and require intricate calibration to compensate for asymmetries introduced during manufacturing. Moreover, TFGs remain vulnerable to low-frequency pressure fluctuations that can induce acoustic interference, raising concerns regarding unintended audio surveillance in sensitive environments.

- introduce acceleration suppression mechanisms

Acceleration suppression mechanisms have been introduced to mitigate the effects of linear motion on TFGs by employing dual-mass configurations that generate common-mode responses to acceleration while preserving differential responses to Coriolis forces. While effective in theory, these mechanisms demand precise geometric symmetry and are sensitive to fabrication tolerances, often requiring post-fabrication trimming or active feedback compensation. The added complexity increases cost and reduces yield, making such solutions impractical for high-volume consumer applications.

- describe BAW gyros

Bulk-acoustic-wave (BAW) gyroscopes represent a promising alternative, operating at frequencies in the megahertz range and leveraging the inertia of a continuous, monolithic resonant structure rather than discrete proof masses. Their high operating frequency inherently attenuates the influence of low-frequency environmental vibrations, and their axis-symmetric geometry minimizes sensitivity to linear acceleration. Furthermore, BAW devices exhibit high quality factors even at moderate vacuum levels, enabling low noise and high resolution without the need for ultra-high vacuum packaging. These characteristics make them particularly attractive for applications demanding long-term stability and environmental resilience.

- identify need for improved gyroscope

Despite their advantages, conventional BAW gyroscopes remain vulnerable to damping coupling, a phenomenon arising from asymmetries in energy dissipation between degenerate vibrational modes. These asymmetries are often induced by anisotropic substrate properties, non-uniform anchor geometries, or irregular stress distributions at the resonator-substrate interface. The resulting mismatch in damping coefficients generates a zero-rate output (ZRO) signal that is indistinguishable from true rotational motion, leading to bias instability and temperature-dependent drift. There exists a critical need for a BAW gyroscope architecture that actively decouples the resonant structure from its substrate, thereby equalizing the damping characteristics of the degenerate modes and eliminating environmentally induced bias errors without relying on complex electronic compensation.

## SUMMARY OF THE INVENTION

- introduce methods for minimizing environmental dependencies

The present invention introduces novel methods for minimizing environmental dependencies in bulk-acoustic-wave gyroscopes by physically isolating the resonant structure from substrate-induced energy dissipation mechanisms. This is achieved through a substrate-decoupling structure that attenuates strain transmission from the resonator to the underlying substrate, thereby ensuring that energy loss occurs predominantly through intrinsic and symmetric mechanisms such as thermoelastic damping, rather than through variable anchor losses. By eliminating the dependence of damping on external boundary conditions, the device achieves consistent quality factors across temperature, pressure, and mechanical stress variations, resulting in a substantial reduction in bias instability and environmental drift.

- describe decoupling mechanism

The decoupling mechanism comprises a set of flexure members arranged in a radially symmetric pattern between the central resonator and the anchor points. These flexures are designed with abrupt angular transitions and optimized cross-sectional geometries to function as mechanical low-pass filters, selectively attenuating high-frequency strain waves generated by the resonator’s motion while permitting the transmission of static or quasi-static forces necessary for structural integrity. The flexures are fabricated as part of the same monolithic silicon structure as the resonator, ensuring perfect alignment and eliminating the need for heterogeneous bonding or additional materials.

- motivate mode matching

Mode matching is critical to the performance of axis-symmetric gyroscopes, as any frequency split between the drive and sense modes leads to stiffness coupling and quadrature error. The invention ensures precise mode matching through integrated electrostatic tuning electrodes positioned adjacent to the antinodes of each mode, enabling dynamic compensation of residual mass and stiffness asymmetries introduced during fabrication. This active tuning, combined with the passive decoupling structure, ensures that the resonance frequencies of the degenerate modes remain aligned under all operational conditions, thereby minimizing the need for recalibration and enhancing long-term reliability.

- introduce substrate-decoupling structure

The substrate-decoupling structure is configured as a circular annulus of interconnected flexure arms, each terminating in a narrow, high-aspect-ratio tether that connects to the substrate at discrete anchor points. The annulus is designed to distribute strain uniformly around the resonator perimeter, ensuring that the stress field at the resonator-substrate interface remains symmetric for both degenerate vibrational modes. This symmetry prevents differential energy dissipation and eliminates the primary source of damping coupling.

- describe configuration of substrate-decoupling structure

The substrate-decoupling structure is configured such that each flexure arm exhibits a double-folded fish-hook geometry, with two opposing arms arranged in a mirrored configuration around the resonator’s central axis. This mirrored arrangement ensures that the mechanical compliance and energy dissipation characteristics are identical for both orthogonal vibrational modes, regardless of their spatial orientation. The flexures are fabricated with precisely controlled widths and lengths to achieve a target mechanical impedance that maximizes strain attenuation while maintaining structural rigidity under static loads.

- introduce mirrored arrangement of double-folded fish-hook spring

The mirrored arrangement of the double-folded fish-hook springs ensures that the direction of strain propagation and the resulting thermoelastic losses are identical for both degenerate modes. Each spring pair is oriented such that the direction of maximum deformation during in-plane resonance aligns with the principal crystal axes of the silicon substrate, minimizing anisotropic losses. The mirrored symmetry guarantees that any variation in material properties or substrate-induced stress affects both modes equally, thereby canceling out differential damping effects.

- describe resonant apparatus

The resonant apparatus comprises a circular annulus of single-crystal silicon with a central void, surrounded by a series of capacitive electrodes arranged in a concentric pattern. The annulus supports two degenerate second-elliptical vibrational modes, each with antinodes positioned at 90-degree intervals. The electrodes are configured to enable electrostatic excitation of the drive mode, capacitive readout of the sense mode, and fine-tuning of modal frequencies through electrostatic stiffening. The entire structure is suspended above the substrate by the substrate-decoupling structure, which isolates it from mechanical coupling to the underlying silicon wafer.

- describe gyroscope apparatus

The gyroscope apparatus integrates the resonant structure with a hermetically sealed packaging system that maintains a controlled internal pressure between 1 and 10 Torr. The device includes through-silicon vias for electrical connectivity and metal traces routed to external pins, enabling direct interfacing with integrated circuitry. The resonator is biased with a polarization voltage to establish a capacitive sensing field, while tuning and decoupling electrodes are driven by a feedback controller that maintains mode matching and minimizes quadrature error. The entire system operates as a self-contained, high-performance angular rate sensor with no moving parts beyond the resonant structure.

- describe method of manufacturing bulk acoustic wave resonator element

The method of manufacturing the bulk acoustic wave resonator element begins with a silicon-on-insulator wafer comprising a 40-micrometer-thick single-crystal silicon layer and a 2-micrometer-thick buried oxide layer. Deep reactive ion etching is employed to define the resonator geometry, the substrate-decoupling flexures, and the surrounding electrode structures. A sacrificial oxide layer is deposited to form the capacitive gaps, followed by selective polysilicon deposition to form the electrodes. A second sacrificial layer is added to define out-of-plane gaps for auxiliary sensors, and the entire structure is released through a hydrofluoric acid etch. Finally, a capping wafer is bonded to the device to form a hermetic enclosure, and through-silicon vias are completed to enable electrical access. The process is compatible with standard CMOS fabrication lines and enables batch production of high-yield, high-performance gyroscopes.

## DETAILED DESCRIPTION

- define annulus

The annulus refers to the ring-shaped structural element of the resonator that surrounds a central void and supports the degenerate vibrational modes. It is fabricated as a continuous, monolithic structure from single-crystal silicon and is characterized by a uniform thickness and a precisely controlled outer and inner diameter. The annulus is designed to sustain high-order in-plane vibrational modes with minimal energy loss and is surrounded by a symmetric array of capacitive electrodes that facilitate excitation and sensing. The geometry of the annulus is optimized to ensure that the strain distribution during vibration remains radially symmetric, thereby minimizing coupling to the substrate and enabling the decoupling mechanism to function effectively.

- motivate gyroscope discussion

The design of the gyroscope is motivated by the need to eliminate environmentally induced bias errors that plague conventional inertial sensors. Traditional approaches rely on electronic compensation, which introduces latency, power consumption, and calibration complexity. In contrast, this invention achieves superior performance by addressing the root cause of bias instability—differential damping between degenerate modes—through mechanical decoupling. This approach eliminates the need for complex feedback algorithms, reduces power requirements, and enhances long-term reliability, making the device uniquely suited for applications requiring continuous, uncalibrated operation in unpredictable environments.

- derive equations of motion

The motion of the resonant structure is described by a pair of coupled second-order differential equations representing the dynamics of the drive and sense modes. These equations include terms for effective mass, stiffness, damping, and Coriolis coupling, with the damping terms explicitly decoupled from substrate-dependent variables due to the presence of the substrate-decoupling structure. The resulting equations demonstrate that the damping coefficients for both modes are now governed primarily by intrinsic material losses, which are identical for both modes due to the symmetric design, thereby eliminating the damping-coupling term b21.

- explain Coriolis effect

The Coriolis effect arises when a mass in motion experiences a perpendicular force due to rotation about an axis normal to the plane of motion. In this device, the drive mode is excited into sustained oscillation, creating a reference velocity field. When the device rotates, the Coriolis force induces a displacement in the sense mode that is proportional to the angular rate. This displacement is detected capacitively and demodulated to extract the rotation signal, with the amplitude of the output directly corresponding to the magnitude of the applied angular velocity.

- describe rotation-rate gyroscope

The rotation-rate gyroscope is a closed-loop system in which the drive mode is maintained in self-oscillation through a feedback loop that regulates amplitude and frequency. The sense mode is monitored for displacement caused by the Coriolis force, and the resulting signal is demodulated using an I/Q architecture to extract the rotation rate. The system includes electrostatic tuning electrodes that dynamically adjust the stiffness of each mode to maintain exact frequency matching, ensuring optimal sensitivity and minimal quadrature error.

- illustrate drive and sense modes

The drive mode is characterized by a vibrational pattern with displacement maxima along one diameter of the annulus, while the sense mode exhibits maxima along the perpendicular diameter. These modes are orthogonal and degenerate in an ideal system, with identical resonance frequencies and quality factors. The capacitive electrodes are positioned to align with the antinodes of each mode, enabling efficient excitation and detection. The substrate-decoupling structure ensures that the mechanical impedance and energy dissipation characteristics of both modes remain identical under all environmental conditions.

- derive natural frequencies

The natural frequencies of the drive and sense modes are derived from the eigenvalues of the system’s mass and stiffness matrices, which are rendered equal by the symmetric design of the resonator and the decoupling structure. The resulting frequency split is reduced to less than 1 Hz through electrostatic tuning, and the modes remain matched across temperature ranges from −40°C to 85°C, demonstrating the effectiveness of the mechanical decoupling approach.

- quantify energy loss

Energy loss in the system is quantified through the quality factor Q, defined as the ratio of stored energy to energy dissipated per cycle. The substrate-decoupling structure increases the anchor loss quality factor by more than four orders of magnitude compared to conventional designs, making thermoelastic damping the dominant loss mechanism. This ensures that Q remains predictable and stable, following a known 1/T^n temperature dependence that can be compensated with minimal computational overhead.

- solve sense-mode displacement

The displacement of the sense mode under the influence of rotation is solved analytically, yielding an expression proportional to the angular rate, the quality factor, and the drive-mode amplitude. The solution confirms that the output signal is free from damping-induced offsets, as the damping coefficients for both modes are now equalized by the decoupling structure, eliminating the term b21 that previously caused indistinguishable bias errors.

### Mode-to-Mode Coupling in Vibratory Gyroscopes

- introduce mode-to-mode coupling in vibratory gyroscopes

Mode-to-mode coupling in vibratory gyroscopes arises from structural imperfections that introduce unintended stiffness and damping interactions between the drive and sense modes. These interactions generate spurious displacement signals that corrupt the rotation-rate measurement, particularly when the coupling terms are environmentally dependent.

- describe zero-rate output (ZRO) in rotation-rate gyros

Zero-rate output (ZRO) refers to the non-zero output signal exhibited by a gyroscope in the absence of angular rotation. In conventional devices, ZRO is caused by stiffness coupling, damping coupling, or electrostatic asymmetries. The present invention eliminates environmentally induced ZRO by ensuring that damping coupling is rendered negligible through mechanical decoupling.

- model cross-excitation between drive and sense modes

Cross-excitation between the drive and sense modes is modeled using coupled differential equations that include stiffness and damping coupling terms. The substrate-decoupling structure eliminates the damping coupling term b21 by ensuring that energy dissipation is symmetric for both modes, thereby removing the primary source of environmentally dependent ZRO.

- define stiffness-coupling and damping-coupling terms

Stiffness-coupling terms arise from asymmetries in the elastic properties of the resonator, while damping-coupling terms arise from differences in energy dissipation between modes. The invention mitigates stiffness coupling through electrostatic tuning and eliminates damping coupling through mechanical decoupling.

- illustrate gyroscope with stiffness and damping coupling terms

The gyroscope is illustrated with electrodes positioned to apply tunable electrostatic forces that counteract stiffness coupling, while the substrate-decoupling structure ensures that damping coupling is absent due to symmetric energy dissipation.

- describe ZRO signal generated by stiffness coupling

The ZRO signal generated by stiffness coupling is phase-shifted by 90 degrees relative to the Coriolis signal and can be removed through quadrature cancellation using feedback control. The invention reduces this signal to negligible levels through precise electrostatic tuning.

- explain rejection of ZRO signal using I-Q demodulation

I-Q demodulation separates the Coriolis signal from the quadrature error by projecting the sensed displacement onto in-phase and quadrature components. The quadrature component is fed back to the tuning electrodes to nullify stiffness coupling, ensuring long-term stability.

- describe cancellation of stiffness coupling using electrostatic forces

Electrostatic forces are applied via electrodes positioned between the antinodes of the modes to induce controlled stiffening or softening, thereby canceling residual stiffness coupling. This feedback mechanism is continuously active and requires no external calibration.

- show frequency response of BAW disk gyroscope before compensation

Before compensation, the frequency response of a conventional BAW disk gyroscope exhibits a frequency split and a high level of quadrature error due to anisotropic damping and stiffness.

- show frequency response after electrostatic mode decoupling

After electrostatic decoupling, the frequency split is eliminated, and the quadrature error is reduced by more than 30 dB, but damping-induced ZRO remains.

- show frequency response after electrostatic mode tuning

After full mode tuning and decoupling, the system exhibits a perfectly matched resonance peak with no measurable quadrature error and no environmentally dependent drift.

- describe damping-coupling force generated by b21

The damping-coupling force generated by b21 arises when the energy loss mechanisms for the two modes differ, causing one mode to induce velocity-dependent forces on the other. This force generates a ZRO signal that is in phase with the Coriolis signal and cannot be separated by conventional demodulation techniques.

- explain undistinguishable ZRO generated by b21

Because the ZRO signal generated by b21 shares the same phase and frequency as the true Coriolis signal, it cannot be distinguished by electronic means alone. The invention eliminates this problem by ensuring that b21 is effectively zero through symmetric mechanical decoupling.

- describe sources of damping coupling

Sources of damping coupling include anchor loss, surface scattering, and thermoelastic damping when these mechanisms are not symmetrically distributed between the two modes. In conventional devices, anisotropic substrates such as (100) silicon cause differential anchor loss, leading to significant damping coupling.

- define damping ratio of a second-order system

The damping ratio of a second-order system is defined as the ratio of actual damping to critical damping. In this invention, the damping ratios of both modes are rendered equal, ensuring that the system’s response is symmetric and stable.

- explain asymmetries in loss mechanisms

Asymmetries in loss mechanisms arise from non-uniform stress distribution, substrate anisotropy, and irregular anchor geometries. The substrate-decoupling structure eliminates these asymmetries by preventing strain from reaching the substrate.

- express damping coupling term b21 in terms of individual damping terms

The damping coupling term b21 is expressed as the difference between the individual damping coefficients of the two modes. In the present invention, this difference is reduced to less than 0.1% of the mean damping coefficient.

- describe quality factor and its relation to damping coefficient

The quality factor Q is inversely proportional to the damping coefficient. By ensuring that the damping coefficients of both modes are identical, the invention ensures that their quality factors are matched, eliminating environmentally induced drift.

- express total energy lost in a resonator

The total energy lost in the resonator is the sum of contributions from anchor loss, thermoelastic damping, and surface scattering. In the present invention, anchor loss is reduced by a factor exceeding 10^4, making thermoelastic damping the dominant and symmetric loss mechanism.

- describe losses associated with viscous damping

Viscous damping, or squeeze-film damping, is minimized by operating the device at moderate vacuum levels where the film thickness is small compared to the resonator dimensions. Due to the axis-symmetric geometry, viscous damping affects both modes equally and does not contribute to coupling.

- describe thermoelastic damping (TED)

Thermoelastic damping arises from the cyclic conversion of mechanical energy into heat due to localized temperature gradients during vibration. The design of the resonator ensures that TED is symmetrically distributed between both modes, making it a predictable and compensatable loss mechanism.

- describe scattering losses due to surface roughness

Scattering losses due to surface roughness are minimized through high-precision fabrication and are negligible compared to the dominant thermoelastic damping mechanism.

- describe intrinsic losses of the material

Intrinsic losses of the silicon material are consistent and temperature-dependent, following a known 1/T^n relationship. These losses are identical for both modes and do not contribute to coupling.

- describe energy dissipated from the resonator through its anchor point

Energy dissipated through the anchor point is the primary source of environmental sensitivity in conventional devices. The substrate-decoupling structure prevents this energy from reaching the substrate, thereby eliminating anchor loss as a variable.

- describe environment-dependent damping coupling

Environment-dependent damping coupling arises when changes in temperature, pressure, or mechanical stress alter the strain field at the resonator-substrate interface, causing differential energy dissipation. The invention eliminates this phenomenon by decoupling the resonator from the substrate.

- introduce Bulk-Acoustic Wave (BAW) disk gyroscopes

Bulk-Acoustic Wave disk gyroscopes are axis-symmetric resonators that support degenerate vibrational modes at high frequencies. They are inherently immune to linear acceleration but suffer from damping coupling due to substrate interactions.

- describe advantages of BAW gyroscopes

Advantages include high operating frequency, immunity to shock and vibration, high quality factors, and compatibility with wafer-level packaging. The present invention enhances these advantages by eliminating damping coupling.

- illustrate capacitive BAW disk resonator and its cross section

The capacitive BAW disk resonator is illustrated with concentric electrodes surrounding the annulus, with ultra-narrow capacitive gaps of 270 nm enabling high-sensitivity readout. The cross section shows the suspended structure above the substrate, connected via the decoupling flexures.

- describe use of ultra-narrow capacitive gaps

Ultra-narrow capacitive gaps of 270 nm enable high electrostatic coupling efficiency, allowing for low-voltage drive and high-sensitivity readout without compromising mechanical stability.

- describe use of second elliptical modes for rate detection

The second elliptical modes (n=3) are selected for their high symmetry, high frequency, and low sensitivity to crystalline orientation. These modes provide optimal sensitivity and minimal cross-talk.

- describe anchor loss in a BAW resonator

Anchor loss occurs when vibrational energy is transmitted from the resonator to the substrate through the anchor points. In conventional designs, this loss is asymmetric and environmentally sensitive.

- express anchor loss in terms of energy lost and stored

Anchor loss is expressed as the ratio of energy dissipated at the anchor interface to the total energy stored in the resonator. In the present invention, this ratio is reduced by four orders of magnitude.

- describe stress and strain exerted by the anchor onto the substrate

The stress and strain exerted by the anchor onto the substrate are determined by the resonator’s deformation profile. The substrate-decoupling structure attenuates these strains, preventing energy transmission.

- introduce Substrate-Decoupled BAW Gyroscopes

Substrate-Decoupled BAW Gyroscopes are a novel class of inertial sensors that achieve environmental stability by mechanically isolating the resonator from substrate-induced damping.

- illustrate decoupled resonant capacitive BAW gyroscope

The decoupled gyroscope is illustrated with the resonator suspended by a symmetric array of double-folded fish-hook flexures, each connecting to a discrete anchor point on the substrate.

- describe decoupling mechanism using spring-like flexure members

The decoupling mechanism uses spring-like flexure members with abrupt angular transitions to act as mechanical low-pass filters, attenuating high-frequency strain waves while transmitting static forces.

- describe placement and design of decoupling mechanism

The decoupling mechanism is placed circumferentially around the resonator, with each flexure arm oriented to align with the principal crystal axes of the silicon substrate. The design ensures that strain is distributed evenly and symmetrically.

- illustrate top view of resonant element suspended by decoupling mechanism

The top view shows the circular annulus suspended by eight symmetric flexure arms, each terminating in a narrow tether connected to the substrate.

- illustrate close-up view of spring-pair

The close-up view reveals the double-folded fish-hook geometry of each spring pair, with mirrored orientation ensuring identical mechanical behavior for both modes.

- describe abrupt angular transitions in flexure members

Abrupt angular transitions in the flexure members create localized strain concentration zones that dissipate high-frequency vibrational energy before it reaches the anchor points.

- illustrate top view of entire circular annulus of resonant element

The top view of the annulus shows the complete geometry of the resonator, including the electrode patterns, release holes, and flexure arms, all arranged with perfect rotational symmetry.

- describe ability to utilize as both yaw-gyroscope and pitch/roll gyroscope

The axis-symmetric design enables the device to function as a yaw-rate gyroscope when mounted horizontally, and as a pitch/roll gyroscope when mounted vertically, without modification.

- illustrate close-up view of displacement of springs during in-plane resonance mode

During in-plane resonance, the springs deform in a radial direction, with strain concentrated at the angular transitions, preventing transmission to the substrate.

- illustrate close-up view of displacement of springs during out-of-plane resonance mode

During out-of-plane resonance, the springs experience torsional deformation, but the geometry ensures that no net strain is transmitted to the substrate.

- describe features enabled by decoupling mechanism

The decoupling mechanism enables unmatched environmental stability, elimination of damping-induced ZRO, consistent quality factor across temperature, and immunity to shock and vibration.

- describe tailoring of Quality Factor (Q) and resonance frequencies

The quality factor and resonance frequencies are tailored through the geometry of the flexures and the electrostatic tuning electrodes, allowing for precise control of sensitivity, bandwidth, and stability without external calibration.