Here is the complete patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND  

Metamaterials are artificially structured materials engineered to exhibit electromagnetic properties not found in naturally occurring substances. These materials derive their unique characteristics from precisely designed subwavelength structural elements rather than their chemical composition. Metamaterials enable unprecedented control over electromagnetic waves, including manipulation of polarization, phase, and amplitude across various frequency ranges from microwave to optical regimes. Conventional optical elements such as lenses, polarizers, and beam splitters rely on macroscopic geometric configurations that are limited by diffraction effects and bulk material properties. In contrast, metamaterials and metasurfaces—their two-dimensional counterparts—achieve electromagnetic control through subwavelength structuring, enabling ultrathin optical components with enhanced functionalities.  

## SUMMARY  

The present invention discloses a method for creating metadevices through an inverse electromagnetic design approach combined with additive manufacturing techniques. This method involves defining desired electromagnetic functionalities through boundary conditions, employing an objective-first optimization algorithm to solve the inverse electromagnetic problem, and fabricating the resulting designs using three-dimensional printing technologies. The system for generating metadevices comprises a computing platform that executes the inverse design algorithm and interfaces with additive manufacturing equipment to produce physical devices. Principal features of this invention include the ability to create broadband, high-efficiency dielectric metadevices with complex geometries unattainable through conventional design methods, the use of low-loss polymer materials compatible with consumer-grade 3D printers, and the scalability of the approach across various electromagnetic frequency ranges.  

## DETAILED DESCRIPTION  

### Introduction to Dielectric Metadevices  

Dielectric metadevices represent a class of electromagnetic components that manipulate wave propagation through subwavelength dielectric structures rather than metallic resonators. These devices overcome limitations associated with plasmonic metasurfaces, particularly their high optical losses and narrow operational bandwidths. The present invention focuses on dielectric metadevices fabricated from high-impact polystyrene (HIPS), a low-cost polymer with excellent millimeter-wave transparency and mechanical stability.  

### Limitations of Conventional Optical Elements  

Traditional optical elements suffer from several fundamental constraints. Bulk lenses and prisms require significant material thickness to achieve desired phase profiles, resulting in heavy and bulky components. Diffractive optical elements, while thinner, typically exhibit strong wavelength dependence and efficiency limitations due to their periodic nature. Conventional design approaches for these elements rely on analytical solutions or iterative parameter sweeps of simple geometric shapes, severely restricting the available design space and performance optimization possibilities.  

### Motivation for Metamaterials and Metasurfaces  

Metamaterials and metasurfaces offer solutions to these limitations by enabling local control of electromagnetic properties through subwavelength structuring. Metasurfaces in particular provide a platform for ultra-thin optical components that can impart spatially varying phase, amplitude, and polarization responses. However, existing metasurface designs based on resonant scattering from predefined geometric elements (such as nanopillars or V-antennas) remain constrained by the limited degrees of freedom in their parameter space, resulting in devices with narrow bandwidths and restricted functionality.  

### Drawbacks of Existing Metasurface Designs  

Current metasurface implementations face several significant challenges. Plasmonic designs suffer from high ohmic losses, while dielectric resonator-based approaches are limited by their resonant nature, leading to narrow operational bandwidths. Traditional design methodologies that rely on library-based approaches—where a limited set of geometric shapes with known phase responses are arranged to approximate desired wavefronts—cannot simultaneously optimize for multiple performance metrics such as efficiency, bandwidth, and polarization control.  

### Inverse Electromagnetic Design Method  

The invention employs an inverse electromagnetic design methodology that directly solves for material distributions satisfying desired electromagnetic functionality. This approach formulates device design as an optimization problem where the electromagnetic wave equation serves as a constraint, and the design objective is to minimize the difference between desired and achieved field transformations. Unlike conventional forward design methods that start with predefined geometries, inverse design explores the full parameter space of possible material distributions, enabling discovery of non-intuitive structures with superior performance characteristics.  

### Objective-First Inverse Design Algorithm  

The core of the invention utilizes an objective-first algorithm that decomposes the nonlinear electromagnetic optimization problem into tractable subproblems. The method alternates between solving for the electric field distribution while fixing the permittivity distribution, and vice versa, iteratively converging toward solutions that satisfy both the wave equation and desired boundary conditions. This approach effectively navigates the high-dimensional design space to discover dielectric structures that implement complex electromagnetic transformations with high efficiency.  

### Optimization Algorithm with Electromagnetic Simulation  

The design process integrates full-wave electromagnetic simulations with the optimization algorithm. At each iteration, the system computes electromagnetic field distributions using numerical solvers that accurately model wave propagation through complex dielectric structures. The algorithm adjusts the material distribution based on the difference between simulated and target field patterns, gradually refining the design toward optimal performance. This computationally intensive process benefits from parallel computing architectures to handle the large parameter spaces involved in free-space metadevice design.  

### Benefits of Additive Manufacturing for Metadevice Fabrication  

Additive manufacturing enables realization of the complex geometries generated by the inverse design process. Three-dimensional printing provides several advantages for metadevice fabrication: it allows creation of structures with high aspect ratios and intricate internal features; supports rapid prototyping and design iteration; and offers scalability from laboratory-scale production to mass manufacturing. The invention specifically utilizes fused deposition modeling (FDM) with high-impact polystyrene due to its excellent millimeter-wave properties and compatibility with consumer-grade 3D printers.  

### Properties of High Impact Polystyrene (HIPS) Material  

High-impact polystyrene serves as the preferred dielectric material for millimeter-wave metadevices in this invention due to its favorable electromagnetic and mechanical properties. HIPS exhibits a low loss tangent (tanδ < 0.003) in the 26-38 GHz frequency range, with a real dielectric constant of approximately 2.3 (refractive index n ≈ 1.52). These properties enable efficient wave propagation while maintaining sufficient contrast with air (ε = 1) for effective phase control. The material's mechanical robustness and thermal stability during printing make it ideal for creating free-standing dielectric structures with subwavelength features.  

### Equation for Device Thickness  

The invention establishes a relationship between device thickness and achievable phase modulation. For a binary device composed of air and HIPS, the phase difference Δφ between regions follows:  

Δφ = 2π(n - 1)t/λ  

where n is the refractive index, t is the thickness, and λ is the operating wavelength. To enable full 2π phase control, the minimum device thickness must satisfy:  

t ≥ λ/(n - 1)  

For HIPS (n ≈ 1.52), this results in a minimum thickness of approximately 2λ to achieve the required phase modulation range.  

### Method for Testing Electromagnetic Properties  

The invention includes a characterization system for evaluating metadevice performance. A vector network analyzer generates input signals transmitted through a horn antenna to create plane wave illumination. The device under test is surrounded by radar-absorbing material to minimize reflections. Far-field measurements employ a scanning horn antenna to map angular transmission patterns, while near-field characterization uses a probe antenna on an XY stage to measure focused beam profiles. This setup enables comprehensive evaluation of device performance across frequency bands of interest.  

### Design, Fabrication, and Characterization Process  

The complete workflow encompasses three stages: computational design using the inverse algorithm, additive manufacturing of the physical device, and electromagnetic characterization. The design phase converts desired functionality into boundary conditions for the optimization algorithm. Fabrication involves converting the numerical design into 3D printer instructions and executing the print with HIPS material. Characterization verifies device performance against design specifications and provides feedback for design refinement.  

### Potential Applications  

Metadevices created through this invention find applications in numerous fields including: millimeter-wave communication systems requiring compact beam steering components; radar systems benefiting from flat, lightweight lenses; polarization control devices for sensing applications; and experimental setups needing customized electromagnetic wave manipulation. The scalability of the approach suggests potential applications across the electromagnetic spectrum from microwave to optical frequencies.  

### Inverse Electromagnetic Design Approach  

The inverse design methodology represents a paradigm shift from conventional electromagnetic component design. Rather than starting with predefined structures, the method begins with desired functionality expressed as input-output field transformations. The algorithm then searches the full permittivity distribution space to discover structures that implement these transformations. This approach enables discovery of non-intuitive geometries that outperform conventional designs in efficiency, bandwidth, and functionality.  

### Design and Fabrication Processes  

The design process initiates with specification of device functionality through electromagnetic boundary conditions. These conditions define the relationship between input and output fields that the metadevice must implement. The inverse algorithm then generates a permittivity distribution satisfying these conditions while accounting for material constraints. The resulting design undergoes conversion to a 3D printable format, typically as an STL file, which guides the additive manufacturing system in building the physical device layer by layer.  

### Objective-First Algorithm for Optimization  

The objective-first algorithm addresses the nonlinear inverse problem by reformulating it as a constrained optimization:  

min_{ε,E} ||∇ × ∇ × E - ω²εE||²  

subject to boundary conditions representing desired functionality. The algorithm alternates between solving for fields (E) and permittivity (ε), gradually converging to solutions that satisfy both the wave equation and performance objectives. This approach proves particularly effective for designing devices requiring complex wavefront transformations.  

### Non-Convex Optimization Problem  

The inverse design problem is inherently non-convex, with many local minima in the solution space. The invention employs specialized optimization techniques to navigate this challenging landscape, including regularization methods to avoid unphysical solutions and multi-start approaches to explore diverse regions of the parameter space. Computational efficiency is maintained through adjoint methods that enable gradient calculations with only two electromagnetic simulations per iteration.  

### Boundary Conditions for Meta-Gratings and Metalenses  

The invention implements different boundary condition strategies for various device types. Meta-gratings employ periodic boundary conditions along the diffraction direction while using perfectly matched layers (PMLs) in other directions. Metalenses utilize PMLs on all sides except the output boundary, where the target field pattern represents focusing to a specified focal point. These boundary treatments enable accurate modeling of free-space devices while maintaining computational efficiency.  

### Two-Dimensional Inverse Design Approach  

While the physical devices are three-dimensional, the invention initially employs two-dimensional design to reduce computational complexity. This approximation remains valid for devices with substantial height-to-wavelength ratios where edge effects become negligible. The 2D approach enables rapid design exploration and optimization before final verification through full 3D simulations.  

### Free-Space Polarization Splitter Meta-Grating  

A key embodiment is a polarization-splitting meta-grating that deflects orthogonal polarizations to opposite diffraction orders. The device operates at 33 GHz with a period of 1.8 cm, producing ±30° deflection angles. The inverse-designed structure achieves higher efficiency and broader bandwidth than conventional blazed gratings by optimally distributing dielectric material to create polarization-dependent phase profiles.  

### Flat Metalens Device  

The invention includes flat metalenses that focus incident plane waves to specified focal points. These devices demonstrate focusing efficiency exceeding 60% with bandwidths greater than 25% of the center frequency. The lenses maintain subwavelength thickness while achieving numerical apertures up to 0.8, rivaling conventional bulk lenses in performance while offering substantial advantages in weight and thickness.  

### Design Parameters for Metadevice  

The polarization-splitting meta-grating incorporates several key design parameters: a 2 cm device thickness to enable full 2π phase control, a 1.8 cm period set by the desired 30° deflection angle at 33 GHz, and a binary permittivity distribution constrained to air (ε = 1) and HIPS (ε = 2.3). These parameters guide the inverse design algorithm toward physically realizable solutions while meeting performance targets.  

### Operation Frequency and Deflection Angle  

The primary meta-grating embodiment operates at 33 GHz (λ = 9.1 mm in free space) with deflection angles of ±30° for orthogonal polarizations. The device maintains functionality across a 26-38 GHz bandwidth (Δλ/λ ≈ 33%) due to its non-resonant operation principle. This broadband performance represents a significant advantage over resonant metasurface designs that typically operate over much narrower bandwidths.  

### Measurement of Far-Field Angular Transmission  

Characterization of the meta-grating involves far-field angular transmission measurements. A vector network analyzer feeds a transmitting horn antenna positioned to create normally incident plane waves. A receiving horn antenna mounted on a rotation stage scans from -40° to +40° in 2° increments, measuring transmitted power at each angle. This data reveals the device's polarization-splitting efficiency and angular selectivity.  

### Simulated and Measured Power Distributions  

Comparison between simulated and measured power distributions validates the design approach. For the polarization splitter at 33 GHz, simulations predict 90% transmission efficiency to desired diffraction orders, while measurements show 76% and 54% for parallel and perpendicular polarizations respectively. The discrepancy arises from fabrication imperfections and finite device size effects, yet the measurements confirm the predicted functionality and angular selectivity.  

### Full-Field Electromagnetic Simulations  

Full-wave simulations provide insight into wave propagation through the metadevices. For the polarization splitter, these reveal how the complex dielectric distribution creates polarization-dependent phase profiles. The simulations show propagation paths through dielectric regions that accumulate different phase shifts for orthogonal polarizations, ultimately producing the desired angular separation in the far field.  

### Simulated Hz Field Amplitudes  

Field simulations visualize device operation. The Hz component (parallel polarization) shows wavefronts bending toward the +30° diffraction order as they interact with the dielectric structure. The field patterns demonstrate how the inverse-designed geometry creates gradual phase accumulation rather than abrupt phase jumps characteristic of resonant metasurfaces, explaining the broadband operation.  

### Simulated E Field Amplitudes  

Perpendicular polarization (Ez component) exhibits distinct propagation behavior, with wavefronts bending toward the -30° diffraction order. The field patterns reveal how the same physical structure interacts differently with each polarization due to the orientation of electric field vectors relative to dielectric boundaries, enabling polarization-dependent functionality from a single device.  

### Broadband Operation Bandwidth  

The invention achieves exceptional bandwidth compared to conventional designs. The polarization splitter maintains functionality from 27-38 GHz (Δλ/λ ≈ 33%), while the polarization-independent bending device operates from 26-44 GHz (Δλ/λ ≈ 55%). This broadband performance stems from the non-resonant operation principle enabled by inverse design, where functionality arises from propagation effects rather than narrowband resonances.  

### Simulated and Measured Far-Field Intensity  

Frequency-dependent far-field measurements confirm broadband operation. Both simulated and experimental results show consistent angular deflection across the operational band, with minimal variation in deflection angle or efficiency. The measurements demonstrate less than 3 dB variation in peak intensity across the 26-38 GHz range for all device embodiments.  

### Additional Metadevices  

Beyond the primary polarization splitter, the invention includes several additional metadevice embodiments. These include a 15° polarization splitter with larger periodicity (3.5 cm) for reduced deflection angle, and a polarization-independent beam bender that deflects all polarizations to the same diffraction order. Each device demonstrates the flexibility of the inverse design approach to meet diverse application requirements.  

### Design and Fabrication of 15° Polarization Splitter  

The 15° polarization splitter illustrates design scalability. With a 3.5 cm period set by the grating equation for 15° deflection at 33 GHz, this device shows how the inverse algorithm adapts to different performance specifications. The larger period accommodates more gradual dielectric transitions, enabling lower deflection angles while maintaining high efficiency and broadband operation.  

### Simulated and Experimental Far-Field Intensity  

Comparison of simulated and measured performance for the 15° splitter shows excellent agreement. The device achieves rejection ratios (peak intensity to maximum sidelobe) of 8.2 dB and 10.6 dB for parallel and perpendicular polarizations respectively, close to simulated predictions. The angular deflection remains stable across the operational bandwidth, confirming robust performance.  

### Simulated Fields and Broadband Data  

Field simulations for the 15° splitter reveal its operational principles. The dielectric distribution creates gentler phase gradients compared to the 30° device, resulting in smaller deflection angles. Broadband data shows consistent performance from 31-37 GHz (Δλ/λ ≈ 18%) for perpendicular polarization, while parallel polarization maintains functionality across an even wider band.  

### Wave Propagation in 15° Polarization Splitter  

Analysis of wave propagation explains polarization-dependent behavior. Perpendicular polarization couples to resonant modes propagating along the device's length, which reverses the bending direction compared to parallel polarization. This physical mechanism enables polarization splitting while maintaining broadband operation for parallel polarization.  

### Simulated Hz and Ez Fields  

Field components reveal distinct interactions for each polarization. Hz fields (parallel polarization) show smooth bending toward the +15° direction, while Ez fields (perpendicular polarization) exhibit more complex patterns with energy channeling toward the -15° direction. These differences arise from the orientation-dependent interaction of each polarization with the dielectric structure.  

### Simulated and Experimental Far-Field Intensity Maps  

Frequency-angle intensity maps provide comprehensive performance visualization. Both simulations and measurements show consistent angular deflection across the operational band, with minimal variation in deflection angle or efficiency. The maps confirm that device functionality remains stable despite frequency changes, a key advantage over resonant designs.  

### Polarization-Independent Millimeter-Wave Bending  

The polarization-independent bender demonstrates another operational mode. This device deflects both polarizations to the same diffraction order (+30°), achieving rejection ratios of 10.1 dB and 12.4 dB for perpendicular and parallel polarizations respectively. The design outperforms conventional blazed gratings by reducing power in undesired diffraction orders by factors of 2.8 and 2.0 for parallel and perpendicular polarizations.  

### Simulated and Experimental Far-Field Intensity  

Performance comparison shows the inverse-designed bender's superiority. While a triangular blazed grating sends significant power to higher diffraction orders (47-51% to unwanted orders), the inverse-designed device reduces this to 18-23%. The improvement stems from the algorithm's ability to optimize the dielectric distribution for specific performance metrics rather than relying on simple geometric shapes.  

### Metadevice Design  

The polarization-independent bender's design emerges from the inverse algorithm without predefined geometric constraints. The resulting structure features complex dielectric distributions that simultaneously satisfy performance objectives for both polarizations. This demonstrates the algorithm's ability to discover non-intuitive solutions that outperform conventional designs.  

### Simulated Fields and Broadband Data  

Field simulations show how the bender achieves polarization-independent operation. The dielectric distribution creates phase profiles that bend both polarizations similarly, despite their different interactions with the structure. Broadband data confirms consistent performance from 26-44 GHz, with deflection angles stable across this wide range.  

### Performance Comparison with Blazed Grating  

Direct comparison highlights the invention's advantages. The inverse-designed bender maintains higher efficiency to the desired diffraction order while suppressing unwanted orders more effectively than a blazed grating of similar thickness. This improvement comes from the algorithm's ability to explore the full design space rather than being limited to simple geometric shapes.  

### Metalens Design  

The invention includes flat metalenses designed using the same inverse approach. These devices convert incident plane waves into focused beams with subwavelength thickness. Two embodiments demonstrate focal lengths of 2λ and 15λ, with numerical apertures of 0.8 and 0.36 respectively, showcasing the method's flexibility across different focusing requirements.  

### Simulated and Measured Power Distribution  

Metalens characterization confirms focusing performance. Both simulated and measured intensity profiles show clear focal spots at designed positions, with full-width half-maximum beam diameters of 0.5 cm (2λ lens) and 1.1 cm (15λ lens). The devices maintain focusing capability across a 28-40 GHz bandwidth, demonstrating broadband operation.  

### Scalability of Methods  

The invention's approach scales across the electromagnetic spectrum. While demonstrated at millimeter wavelengths, the same design and fabrication principles apply to other frequencies provided suitable dielectric materials and fabrication resolution are available. An electron microscope image of a scaled-down infrared device confirms feasibility at shorter wavelengths.  

### Process for Creating Metadevice  

The complete metadevice creation process involves several steps: First, the desired electromagnetic functionality is defined through input-output field relationships. These specifications translate into boundary conditions for the inverse algorithm. The algorithm then generates a permittivity distribution satisfying these conditions while accounting for material constraints. The resulting design undergoes conversion to a 3D printable format and fabrication using additive manufacturing. Finally, the physical device undergoes electromagnetic characterization to verify performance.  

### Platform for Design and Fabrication  

The invention provides an integrated platform combining computational design tools with additive manufacturing capabilities. This closed-loop system enables rapid prototyping of metadevices with customized electromagnetic functionalities. The platform's flexibility supports iterative design refinement based on characterization results, accelerating development of optimized devices.  

### Computing System Architecture  

The design system employs a computing platform with specialized processors for electromagnetic simulation and optimization. The architecture includes high-performance CPUs or GPUs for numerical computations, sufficient memory for handling large design spaces, and storage for maintaining design libraries and simulation results. The system interfaces with additive manufacturing equipment through standard or proprietary communication protocols.  

### Design Algorithm Implementation  

The inverse design algorithm is implemented as software running on the computing platform. The implementation includes modules for electromagnetic simulation, optimization routines, and design visualization. The software accepts user-defined performance specifications and constraints, then autonomously generates metadevice designs meeting these requirements through the objective-first optimization process.  

### Additive Manufacturing Integration  

The invention tightly couples design and fabrication through direct integration with additive manufacturing systems. The computing platform converts optimized designs into machine instructions for 3D printers, typically in G-code or similar formats. This integration enables seamless transition from digital design to physical realization, supporting rapid iteration and customization.  

### Network and Device Communication  

The system supports multiple communication modes between design and fabrication components. These include network-based communication for distributed systems, direct device-to-device connections for integrated setups, and various file transfer protocols. The flexibility in communication methods accommodates different laboratory and production environments.  

### Use of Illustrative Embodiments  

The described embodiments serve as examples demonstrating the invention's principles and capabilities. The polarization splitters, beam benders, and metalenses illustrate applications of the inverse design methodology to different electromagnetic functionalities. These examples do not limit the scope of possible metadevices that can be created using the disclosed approach.  

### Modifications and Variations  

The invention encompasses various modifications and variations. These include alternative dielectric materials beyond HIPS, different additive manufacturing techniques such as stereolithography or selective laser sintering, and adaptations of the inverse algorithm for specialized applications. The core principles remain applicable across these variations.  

### Scope of Invention  

The invention's scope covers the combined use of inverse electromagnetic design with additive manufacturing to create functional metadevices. This includes the methods, systems, and resulting devices across the electromagnetic spectrum. The scope extends to any application where the disclosed approach enables superior electromagnetic components compared to conventional design and fabrication methods.  

### Claims and Equivalents  

The claims define the legal boundaries of the invention, covering the novel aspects of the combined inverse design and additive manufacturing approach. Equivalents include alternative implementations that achieve substantially the same results through similar means, even if differing in specific details from the described embodiments.  

### Purpose of Invention  

The invention aims to overcome limitations in conventional electromagnetic component design and fabrication. By enabling creation of high-performance, broadband metadevices through an integrated computational and manufacturing platform, the invention addresses needs for compact, efficient, and customizable electromagnetic components across various applications.  

### Practical Applications  

Practical applications span communications, sensing, imaging, and experimental physics. Specific implementations include compact beam steering devices for millimeter-wave radar systems, flat lenses for lightweight optical systems, polarization control components for quantum information systems, and customized wavefront shaping devices for research applications.  

### Advantages of Invention  

Key advantages include: broadband operation surpassing resonant designs; high efficiency through full design space exploration; rapid prototyping enabled by additive manufacturing; scalability across the electromagnetic spectrum; and the ability to create multifunctional devices that simultaneously optimize multiple performance metrics. These advantages position the invention as a transformative approach to electromagnetic component development.  

### Conclusion  

The invention represents a significant advancement in electromagnetic device technology by combining inverse design methodologies with additive manufacturing. This synergistic approach enables creation of metadevices with unprecedented performance characteristics, including broadband operation, high efficiency, and complex functionality. The platform's flexibility supports rapid development of customized components across diverse applications, promising to transform how electromagnetic systems are designed and implemented.