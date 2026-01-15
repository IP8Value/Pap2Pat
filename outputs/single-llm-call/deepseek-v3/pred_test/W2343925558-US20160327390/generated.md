Here is the patent application following your outline and incorporating the research paper content:

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates to resonant devices, and more particularly to micromachined vibratory gyroscopes utilizing bulk acoustic wave (BAW) resonators with improved environmental robustness. The disclosed apparatus and methods address critical limitations in conventional gyroscope designs by implementing a substrate-decoupling structure that minimizes anchor loss and damping coupling between degenerate vibration modes. This invention finds particular utility in high-performance inertial sensing applications requiring immunity to shock, vibration, and temperature variations, including but not limited to automotive safety systems, personal navigation devices, and industrial motion control applications.  

## BACKGROUND OF THE INVENTION  

Micro-machined vibratory gyroscopes have become essential components in modern motion sensing systems, enabling applications ranging from consumer electronics to automotive safety controls. These devices operate by detecting Coriolis-induced energy transfer between two degenerate vibration modes when subjected to angular rotation. Among various implementations, tuning-fork gyroscopes (TFGs) have dominated commercial markets due to their relatively simple fabrication and moderate performance characteristics. However, TFGs suffer from fundamental limitations including sensitivity to environmental vibrations and linear accelerations, which manifest as output drift and reduced accuracy in practical operating conditions.  

The limitations of TFG technology stem primarily from their low operating frequencies (typically below 50 kHz), which fall within the spectrum of common environmental vibrations. While acceleration suppression mechanisms using redundant proof masses can partially mitigate these issues, such approaches significantly increase device size and require complex calibration procedures. Furthermore, recent concerns about the acoustic sensitivity of consumer-grade gyroscopes have highlighted the need for more robust rotation sensing technologies resistant to environmental interference.  

Bulk-acoustic wave (BAW) gyroscopes present an alternative approach by operating at MHz-range frequencies with high quality factors. These axis-symmetric devices inherently reject low-frequency environmental noise while maintaining compact form factors suitable for high-volume production. However, conventional BAW gyroscopes remain susceptible to damping coupling effects caused by asymmetries in loss mechanisms between degenerate vibration modes. Such asymmetries, often resulting from substrate interactions in anisotropic materials like (100) single-crystal silicon, introduce environmentally dependent offset variations that degrade performance.  

There exists a pressing need in the field for improved gyroscope designs that combine the high-frequency advantages of BAW resonators with effective isolation from substrate-induced losses. The present invention addresses this need through novel structural configurations that decouple the resonant element from its substrate while maintaining precise control over modal characteristics.  

## SUMMARY OF THE INVENTION  

The present invention provides methods and apparatus for minimizing environmental dependencies in vibratory gyroscopes through innovative structural designs. A key aspect involves decoupling the resonant element from its substrate using specialized spring-like flexure members arranged in a symmetric configuration. This substrate-decoupling structure effectively isolates the resonator from anchor losses while maintaining precise control over vibration mode characteristics.  

Mode matching between degenerate vibration modes constitutes another critical feature of the invention. Through electrostatic tuning electrodes and optimized structural geometries, the disclosed apparatus achieves precise frequency matching between drive and sense modes, significantly reducing zero-rate output (ZRO) errors. The substrate-decoupling structure comprises a mirrored arrangement of double-folded fish-hook springs that provide balanced mechanical support while minimizing energy dissipation to the substrate.  

The resonant apparatus of the invention features a circular annulus structure with integrated capacitive transduction elements for excitation and detection of high-frequency bulk acoustic waves. Electrodes with ultra-narrow gaps (approximately 270 nm) enable efficient electrostatic actuation and sensing while maintaining compact device dimensions. The gyroscope apparatus further incorporates compensation electrodes for active tuning of resonance frequencies and cancellation of stiffness coupling between modes.  

Manufacturing methods for the bulk acoustic wave resonator element utilize modified high-aspect ratio poly- and single-crystal silicon (HARPSS) processes. The fabrication sequence includes deep reactive ion etching of device layers, sacrificial oxide growth for gap definition, polysilicon refill for electrode formation, and wafer-level packaging for hermetic encapsulation. Through-silicon vias provide electrical connections while maintaining the integrity of the decoupling structure.  

## DETAILED DESCRIPTION  

The invention employs an annulus structure as the primary resonant element, with specific geometric parameters optimized for high-frequency bulk acoustic wave operation. The circular symmetry of the annulus supports degenerate vibration modes essential for Coriolis-based rotation sensing. When subjected to angular velocity about an axis normal to its plane, the resonator exhibits coupled motion governed by the equations:  

\[ m_{11}\ddot{q}_1(t) + b_{11}\dot{q}_1(t) + b_{12}\dot{q}_2(t) + k_{11}q_1(t) + k_{12}q_2(t) = \sum_{i=1}^k F_{1,i} - 2\lambda m_{22}\Omega(t)\dot{q}_2(t) \]  

\[ m_{22}\ddot{q}_2(t) + b_{22}\dot{q}_2(t) + b_{21}\dot{q}_1(t) + k_{22}q_2(t) + k_{21}q_1(t) = \sum_{i=1}^k F_{2,i} + 2\lambda m_{11}\Omega(t)\dot{q}_1(t) \]  

where Ω(t) represents the applied rotation rate, λ denotes the angular gain determined by device geometry, and q₁(t), q₂(t) are generalized coordinates for the two degenerate modes. The Coriolis effect manifests through the cross-coupled velocity terms, enabling rotation rate detection via measurement of sense mode displacement.  

In a rotation-rate gyroscope configuration, one mode (drive mode) is excited into steady oscillation while the orthogonal mode (sense mode) responds to Coriolis forces proportional to the input rotation. Natural frequencies of the modes are derived from effective mass and stiffness parameters, with ideal operation requiring precise frequency matching. Energy loss mechanisms are quantified through quality factor analysis, where total Q accounts for various damping contributions including anchor loss, thermoelastic damping, and viscous effects.  

The sense-mode displacement solution reveals three primary components: the desired Coriolis response, stiffness-coupling induced ZRO, and damping-coupling induced ZRO. While the Coriolis term provides the useful rotation signal, the ZRO components represent error sources requiring minimization through design and control techniques.  

### Mode-to-Mode Coupling in Vibratory Gyroscopes  

Mode-to-mode coupling in vibratory gyroscopes represents a fundamental challenge addressed by the present invention. Zero-rate output (ZRO) in rotation-rate gyroscopes stems primarily from unwanted energy transfer between drive and sense modes, modeled through stiffness-coupling (k₁₂, k₂₁) and damping-coupling (b₁₂, b₂₁) terms. These coupling mechanisms generate error signals that can mask true rotation measurements, particularly in environmentally variable conditions.  

The gyroscope apparatus incorporates specific features to mitigate coupling effects. Stiffness coupling generates a ZRO signal phase-shifted by 90° relative to the Coriolis response, enabling rejection through I-Q demodulation techniques. Furthermore, electrostatic forces applied via tuning electrodes (VT₁, VT₂) provide active cancellation of stiffness coupling by introducing compensating spring softening effects.  

Damping coupling presents a more challenging issue as it produces ZRO signals in-phase with the Coriolis response. The invention addresses this through symmetric mechanical design of the decoupling structure and optimization of energy loss pathways. Key damping mechanisms include:  

1. Viscous damping through squeeze-film effects in narrow capacitive gaps  
2. Thermoelastic damping (TED) in compressive/expansive regions  
3. Surface scattering losses due to roughness  
4. Intrinsic material losses  
5. Anchor loss through substrate coupling  

The substrate-decoupling structure specifically targets anchor loss reduction by minimizing strain energy transmission to the substrate. This is achieved through spring-like flexure members exhibiting abrupt angular transitions that effectively reflect acoustic energy back into the resonator. The symmetric arrangement ensures balanced damping characteristics between degenerate modes, preventing environmentally dependent ZRO variations.  

Bulk-Acoustic Wave (BAW) disk gyroscopes according to the invention leverage several advantageous characteristics. The high operating frequency (typically 4 MHz) inherently rejects low-frequency environmental vibrations. Ultra-narrow capacitive gaps (270 nm) enable efficient electrostatic transduction while maintaining compact dimensions. Second elliptical modes (n=3) are employed for rate detection due to their optimal balance of sensitivity and fabrication tolerance.  

Anchor loss in conventional BAW resonators is expressed through the integral of strain energy density at the resonator-substrate interface:  

\[ \Delta W = \pi \int_{V_s} \sigma_s \epsilon_s dV \]  

where σₛ and εₛ represent interface stress and strain. The invention's substrate-decoupled (SD) BAW gyroscope dramatically reduces this term through its isolation structure, increasing effective Qanchor by over four orders of magnitude compared to center-supported designs.  

The decoupling mechanism employs spring-pairs arranged symmetrically about the resonator perimeter. Each spring features optimized geometry including folded beam segments and stress-relieving transitions that collectively minimize strain transmission. During operation, the springs accommodate both in-plane and out-of-plane resonance modes while maintaining structural integrity.  

Key features enabled by the decoupling mechanism include:  

1. Tailored Quality Factors through controlled energy loss pathways  
2. Precisely matched resonance frequencies between degenerate modes  
3. Immunity to substrate-induced damping asymmetries  
4. Configurability as both yaw-rate and pitch/roll-rate sensors  
5. Stable performance across temperature variations  

The complete gyroscope apparatus integrates these mechanical innovations with electronic control systems for drive-loop oscillation, sense-mode detection, and active compensation of residual coupling effects. Wafer-level packaging maintains optimal pressure conditions (1-10 Torr) while through-silicon vias provide reliable electrical interconnections.  

[Continued in next response due to length]