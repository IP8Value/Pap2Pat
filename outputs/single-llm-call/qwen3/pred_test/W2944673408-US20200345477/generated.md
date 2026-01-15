# DESCRIPTION

## BACKGROUND OF THE INVENTION

### Field of the Invention

The present invention relates to methods and systems for evaluating the mechanical behavior of elastomeric medical devices, particularly breast implants, under compressive loading conditions. The invention provides a novel analytical framework for quantifying geometric deformation and engineering mechanical properties through the interpretation of load-displacement data obtained during standardized compression testing. This field encompasses the biomechanical characterization of soft, deformable structures used in reconstructive and cosmetic surgery, where the ability to predict shape retention, surface interaction, and durability under physiological loads is critical to clinical performance and long-term patient safety.

### Description of Related Art

Current methods for evaluating breast implants rely heavily on standardized mechanical tests such as tensile strength, tear resistance, and cyclic fatigue assessments, which are primarily focused on material properties of the shell or fill medium. These tests, while valuable for regulatory compliance and quality control, do not capture the full mechanical response of the implant as a complete system under dynamic, multi-dimensional compression. The interaction between the implant and its surrounding tissue environment—particularly the breast capsule—is governed not only by shell integrity but also by how the implant deforms, moves, and redistributes internal volume under load. Existing testing protocols lack a systematic approach to correlate observed load-displacement behavior with geometric changes in implant shape, surface area, and strain distribution. As a result, manufacturers are unable to predict how design modifications, such as internal structure or fill composition, influence the implant’s overall stability during everyday activities like walking, lying down, or physical exertion. This gap limits the ability to optimize implant geometry for reduced capsular contracture, improved durability, and enhanced tactile realism.

### Limitations of Current Methods

Traditional mechanical testing methods fail to account for the three-dimensional shape evolution of breast implants during compression. Measurements of shell thickness, elongation, or ultimate strength provide no insight into how the implant’s external profile changes under load, nor do they quantify the surface area that interfaces with the capsule wall. Without a means to calculate geometric parameters such as diameter, projection, and surface area from readily obtainable load and plate spacing data, clinicians and engineers are left with incomplete information about implant behavior. Furthermore, existing protocols do not distinguish between the effects of internal structure and material properties on overall shape stability, making it difficult to isolate the contribution of design features such as dual-lumen systems or internal baffles. The absence of a validated model to predict implant geometry from mechanical inputs has hindered the development of predictive tools for clinical outcomes.

### Current Mechanical Testing

Current industry-standard mechanical testing involves subjecting implants to uniaxial compression between parallel plates using load frames equipped with force and displacement sensors. These tests are conducted under controlled environmental conditions and follow guidelines from ASTM and ISO standards. While these procedures yield reproducible load-displacement curves, they are interpreted solely in terms of force resistance or failure thresholds, without extracting geometric or strain-based metrics. The data collected are not transformed into engineering properties such as diametric strain, projection strain, or tangent moduli, which would provide deeper insight into the implant’s mechanical behavior. Consequently, the results are limited to binary assessments of compliance or failure, rather than a continuous, quantitative characterization of shape stability.

### Need for Simplified Method

There exists a critical need for a simplified, automated method that transforms raw load and plate spacing measurements into comprehensive mechanical descriptors of implant performance. Such a method would eliminate the need for manual caliper measurements, reduce variability between testing centers, and enable real-time analysis during dynamic compression cycles. A streamlined approach that relies only on data already captured by standard load frames—force and platen displacement—would significantly lower the barrier to adoption in both research and manufacturing environments. This would allow for rapid comparison of implant designs, accelerated development cycles, and more accurate prediction of clinical behavior without requiring additional instrumentation or complex imaging techniques.

### Potential Benefits of New Method

The proposed method offers numerous advantages over existing approaches. It enables the calculation of implant diameter, surface area, and multi-dimensional strains directly from load-displacement data, providing a complete picture of shape change under load. It introduces the concept of engineering tangent moduli as quantitative indicators of shape stability, allowing for direct comparison between implant types regardless of initial geometry. The method validates a quasi-equilibrium assumption that permits the use of dynamic compression data to approximate static behavior, thereby reducing test duration and increasing throughput. Additionally, by accounting for surface friction through lubricated and unlubricated testing conditions, the method provides insight into how implant-tissue interactions may vary in vivo. These capabilities collectively empower manufacturers to design implants with improved durability, reduced capsular contracture risk, and enhanced tactile realism, while enabling regulators to evaluate implants based on clinically relevant mechanical metrics.

## OBJECTS OF THE INVENTION

- Provide a simplified means to determine breast implant geometries by deriving diameter and surface area from load and plate spacing measurements without requiring direct visual or manual dimensional analysis.  
- Provide a means to determine engineering mechanical properties such as projection strain, diametric strain, areal strain, and tangent moduli that quantify shape stability and resistance to deformation under compressive loads.  
- Provide a simplified method for assuming quasi-equilibrium during dynamic compression testing, enabling the use of continuous load-displacement data to accurately predict static geometric behavior and eliminate the need for time-consuming stepwise measurements.  
- Provide a means to assess the impact of design changes, such as internal structure, fill composition, or shell thickness, on overall implant shape stability and mechanical response, thereby enabling predictive optimization of implant performance prior to clinical testing.  
- Provide a comprehensive evaluation of breast implant properties that integrates geometric, strain-based, and modulus-derived metrics into a unified analytical framework, allowing for direct comparison of implants across manufacturers and material types.

## BRIEF SUMMARY OF THE INVENTION

The invention introduces a novel method for characterizing the mechanical behavior of elastomeric medical devices, particularly breast implants, through the analysis of load and plate spacing data acquired during compression testing. The method is based on a validated geometric model that describes the implant as a composite of a central cylindrical region and an outer half-torus, enabling the calculation of implant diameter and surface area as functions of measured plate spacing and known implant volume. A quasi-equilibrium assumption is employed to permit the use of dynamic compression data to derive geometric and mechanical properties equivalent to those obtained from static testing, significantly reducing test time and increasing reproducibility. From the derived diameter and plate spacing, engineering strains—including projection strain, diametric strain, and areal strain—are computed to quantify shape change, and tangent moduli are calculated as local slopes of stress-strain curves to assess resistance to deformation. The method is specifically applied to breast implants but is adaptable to any elastomeric device with a deformable, fluid-filled structure. The invention provides accurate, repeatable, and clinically relevant mechanical descriptors that correlate with implant durability, fold flaw resistance, and capsular interaction. Validation studies demonstrate that calculated diameters deviate from direct measurements by less than 2% under loads exceeding 5 N, and that the quasi-equilibrium assumption introduces negligible additional error. The method enables manufacturers to evaluate design modifications with unprecedented precision and provides regulators with a standardized, quantitative framework for assessing implant performance beyond traditional material tests.

## BRIEF DESCRIPTION OF SYMBOLS & NUMBERS

- Define process steps as a sequence of operations including calibration, implant loading, data acquisition, geometric computation, and property derivation.  
- Define breast implant manufacturing step as the production of a sealed, sterilized, fluid-filled elastomeric device with a defined volume and shell structure, wherein the implant is intended for implantation into a human body.  
- Define recording of breast implant properties as the measurement and documentation of initial volume, shell thickness, and nominal dimensions prior to mechanical testing.  
- Define recording of plate spacing and load as the continuous digital acquisition of vertical displacement between compression platens and corresponding force applied to the implant during compression.  
- Define computation of geometry and properties as the mathematical derivation of implant diameter, surface area, strain, and tangent modulus from recorded plate spacing and load data using a validated geometric model and quasi-equilibrium assumption.  
- Define breast implant or elastomeric device as a medical implant composed of one or more elastomeric shells enclosing a fluid medium, wherein the device is deformable under compressive load and exhibits shape retention characteristics.  
- Define geometric model components as a central cylinder representing the flattened portion of the implant between platens and an outer half-torus representing the curved perimeter region, together forming a composite volume-preserving geometry.  
- Define symbols for properties and variables as follows: F for applied load, H for plate spacing, D for implant diameter, V for implant volume, A for surface area, εH for projection strain, εD for diametric strain, εA for areal strain, S for planform stress, ESH for projection tangent modulus, ESD for diametric tangent modulus, and ESA for areal tangent modulus.

## DETAILED DESCRIPTION OF THE INVENTION

- Relate to breast implants undergoing dynamic compression by describing the continuous application of increasing load between two parallel platens, during which load and plate spacing are recorded at high frequency to capture the full deformation profile.  
- Describe breast implant composition as a sealed, fluid-filled structure composed of one or more nested elastomeric shells, wherein the internal volume remains constant during compression and the shell material exhibits nonlinear elastic behavior.  
- Introduce load frame apparatus as a computer-controlled mechanical testing system equipped with a precision load cell and linear displacement sensor, wherein the upper and lower platens are polished stainless steel surfaces with a surface roughness of less than 0.2 micrometers.  
- Describe plate spacing and displacement as the vertical distance between the two compression platens, measured with sub-millimeter accuracy, and defined as the primary indicator of implant deformation along the axis of loading.  
- Explain dynamic load program as a continuous, linear increase in applied force from a minimal initial load to a maximum target load at a constant cross-head speed, designed to simulate quasi-static physiological loading conditions.  
- Describe load cell and force measurement as the use of a calibrated force transducer capable of resolving forces from 0.1 N to 1000 N with a precision of ±0.5%, ensuring accurate determination of the load-deformation relationship.  
- Summarize process steps of test method as: calibrating the load frame, preparing the implant, establishing initial conditions, applying dynamic compression, recording load and plate spacing, computing geometric parameters, and deriving mechanical properties.  
- Calibrate load frame by performing a zero-load offset adjustment and applying known reference loads to verify linearity and accuracy of the force and displacement sensors.  
- Manufacture and load breast implant by obtaining a sterilized, commercially available implant with a known internal volume, ensuring no prior use or damage, and placing it centrally on the lower platen.  
- Record breast implant volume and shell thickness by referencing manufacturer specifications and verifying shell thickness at multiple perimeter locations using calibrated calipers.  
- Adjust plate spacing to initial state by lowering the upper platen until a minimal load of 0.445 N is achieved, at which point the initial plate spacing and implant diameter are recorded.  
- Optionally lubricate interface by applying a thin, uniform layer of silicone oil to both platen surfaces to simulate a low-friction in vivo environment.  
- Implement dynamic load program by initiating continuous compression at a cross-head speed of 25.4 cm/minute until the target load of 534 N is reached, with data sampled at intervals not exceeding 0.1 seconds.  
- Record plate spacing and load by storing the time-series data from the load cell and displacement sensor in a digital acquisition system for subsequent analysis.  
- Compute breast implant geometry and engineering properties by applying a geometric model to calculate diameter and surface area from plate spacing and volume, then deriving strain and tangent modulus values from the load-displacement curve.  
- Describe composite geometric model as a volume-conserving shape composed of a central cylinder of diameter d and height H, capped by an outer half-torus whose major radius is derived from H and d, such that the total volume equals the known implant volume.  
- Derive equation for breast implant-platen contact diameter by solving the volume conservation equation for d as a function of H and V, yielding a quadratic expression with a single physically meaningful root.  
- Simplify equation for breast implant diameter by substituting the calculated contact diameter into a geometric relationship that defines the total implant diameter as the sum of plate spacing and contact diameter.  
- Calculate surface area of breast implant by summing the areas of the two circular faces of the central cylinder and the outer half-torus, expressed as a function of diameter and plate spacing.  
- Compute engineering stresses by dividing the applied load by the initial planform area of the implant, defined as the area of a circle with the initial diameter.  
- Calculate engineering strains by normalizing dimensional changes relative to initial values, including projection strain as the relative reduction in plate spacing, diametric strain as the relative reduction in diameter, and areal strain as the relative reduction in surface area.  
- Compute engineering moduli by calculating the local slope of the stress-strain curve at each data point, yielding tangent moduli that describe the implant’s increasing resistance to deformation as strain accumulates.  
- Illustrate load-displacement measurements by presenting curves of load versus plate spacing for multiple implants, demonstrating consistent, repeatable behavior across replicate tests.  
- Validate quasi-equilibrium assumption by comparing static and dynamic load-plate spacing curves, showing that the dynamic data fall within ±1% of the static data across the entire load range, confirming that transient effects do not significantly alter geometric response.  
- Calculate breast implant diameter from plate spacing by applying the derived geometric equation to each recorded plate spacing value, producing a continuous diameter profile throughout the compression cycle.  
- Compute breast implant surface area by substituting the calculated diameter and measured plate spacing into the surface area equation, generating a continuous profile of external surface change under load.  
- Calculate diametric strain by subtracting the initial diameter from each calculated diameter, dividing by the initial diameter, and expressing the result as a percentage.  
- Compute planform stress by dividing each recorded load by the initial planform area, producing a stress profile that correlates directly with the applied load.  
- Determine engineering moduli by computing the difference in stress and strain between consecutive data points, dividing the stress difference by the strain difference, and assigning the result to the midpoint of the interval.  
- Describe shape stability of breast implant as the inverse relationship between strain magnitude and tangent modulus, wherein lower strain and higher modulus indicate greater resistance to deformation under load.  
- Conclude invention embodiments by affirming that the method provides a complete, automated, and clinically predictive framework for evaluating breast implant performance based on fundamental mechanical principles.

### First Invention Embodiment

- Illustrate load-displacement measurements by presenting data for two distinct breast implants, one with a dual-lumen structure and one with a single-lumen gel fill, showing that the dual-lumen implant exhibits less plate spacing reduction at equivalent loads.  
- Validate quasi-equilibrium assumption by overlaying dynamic and static load-plate spacing curves for both implants, demonstrating that the dynamic data deviate from static data by less than 0.6% across the entire load range, confirming the validity of the assumption.  
- Describe breast implant #1 as a dual-lumen saline-filled implant with a nested shell and internal baffle structure, having a volume of 335 cc and an initial diameter of 12.3 cm.  
- Describe breast implant #2 as a cohesive silicone gel-filled implant with a single-shell structure, having a volume of 335 cc and an initial diameter of 12.4 cm, with a shell thickness 20% greater than that of implant #1.

### Second Invention Embodiment

- Calculate breast implant diameter from plate spacing by applying the geometric equation to the dynamic load-displacement data of implant #1 and implant #2, producing continuous diameter profiles that reveal the dual-lumen implant maintains a smaller diameter under load.  
- Validate quasi-equilibrium assumption by comparing the calculated diameters from dynamic data to direct manual measurements from static tests, finding agreement within ±2.5% for implant #1 and ±1.1% for implant #2 over the 22.2 N to 534 N load range.  
- Describe breast implant #1 and #2 as having nearly identical initial volumes and shell thicknesses, yet exhibiting significantly different geometric responses under compression, with implant #1 showing 2.8% to 7.8% smaller diameter than implant #2 at loads exceeding 130 N.

### Third Invention Embodiment

- Calculate diametric strain by applying the strain formula to the calculated diameters of implant #1, implant #2, and a third implant with a standard gel fill, revealing that implant #1 exhibits 6% to 19% lower diametric strain than the gel implants across the 100 N to 500 N load range.  
- Describe breast implant #1, #2, and #3 as having identical initial volumes but differing internal structures, with implant #1 demonstrating the least diametric deformation, implant #2 exhibiting moderate deformation, and implant #3 showing the greatest deformation under equivalent loads.  
- Demonstrate that the reduction in diametric strain for implant #1 correlates with its superior clinical outcomes, including lower rupture and capsular contracture rates.

### Fourth Invention Embodiment

- Compute planform stress by dividing the recorded load by the initial planform area common to all three implants, producing stress profiles that confirm the load is normalized to a consistent reference area, enabling direct comparison of mechanical response.  
- Show that the planform stress for implant #1 is identical to that of the gel implants at equivalent loads, indicating that the difference in shape stability is not due to material stiffness but to structural geometry.  
- Confirm that the stress-strain behavior of implant #1 diverges from the gel implants at strains above 10%, indicating increased resistance to deformation beyond a threshold level of compression.

### Fifth Invention Embodiment

- Determine engineering moduli by computing the tangent moduli for projection, diametric, and areal strain for implant #1, implant #2, and implant #3, revealing that implant #1 exhibits tangent moduli 60% to 130% higher than the gel implants at strains above 20%.  
- Describe breast implant #1, #2, and #3 as having similar initial material properties but vastly different shape stability, with implant #1 demonstrating superior resistance to deformation due to its internal structure, not its shell thickness or fill viscosity.  
- Establish that the higher tangent moduli of implant #1 correspond directly to its clinical performance, including absence of fold flaw failures and lower capsular contracture incidence.

### Additional Embodiments

- Contemplate application to other elastomeric devices such as vascular grafts, urethral implants, or soft tissue expanders, wherein shape stability under compressive or tensile loads is critical to function and safety.  
- Describe multi-lumen structure as a configuration comprising two or more interconnected fluid compartments separated by internal membranes or baffles, wherein the structure resists deformation by redistributing internal pressure and limiting shell folding.  
- Motivate use in cyclic fatigue testing by demonstrating that shape stability directly influences stress concentration at the implant perimeter, thereby predicting fatigue life without requiring prolonged testing cycles.  
- Specify cyclic fatigue load frequencies as ranging from 0.5 Hz to 5 Hz, simulating physiological activities such as walking, running, and breathing.  
- Specify crosshead speeds as 25.4 cm/minute for dynamic testing, a rate chosen to balance test duration with physiological relevance.  
- Describe data acquisition system as a computer-controlled system sampling load and displacement at 10 Hz or higher, with software capable of real-time geometric and mechanical property computation.  
- Specify force range as 0.1 N to 1000 N, encompassing both low-level tactile interactions and high-level physiological or trauma loads.  
- Contemplate contoured plate surfaces as an alternative to flat platens, enabling simulation of non-uniform tissue contact, such as that occurring against ribcage or chest wall.  
- Integrate software program as a standalone application that receives raw load and displacement data, applies the geometric model, and outputs diameter, strain, and modulus profiles with graphical and statistical summaries.  
- Test with lubricated interface to simulate low-friction in vivo conditions and with unlubricated interface to simulate high-friction or encapsulated states, revealing that friction significantly alters strain magnitude and modulus values.  
- Characterize anatomically shaped breast implants by adapting the geometric model to account for non-circular cross-sections, enabling application to teardrop or asymmetrical implants.  
- Specify platen materials as polished stainless steel, polycarbonate, or coated aluminum, each selected for low surface roughness and chemical compatibility with implant materials.  
- Specify platen surface finish as a surface roughness of 0.1 to 0.2 micrometers Ra, ensuring minimal frictional interference and consistent contact mechanics.  
- Contemplate elastomeric devices with various geometries including spherical, elliptical, or irregular forms, and adapt the geometric model by introducing additional parameters to describe non-cylindrical deformation profiles.  
- List materials for elastomeric devices as silicone elastomers, polyurethanes, thermoplastic elastomers, or hybrid composites, each compatible with the method’s assumptions of volume conservation and nonlinear elasticity.