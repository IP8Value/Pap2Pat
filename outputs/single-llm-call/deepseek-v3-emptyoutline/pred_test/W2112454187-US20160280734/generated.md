Here is the drafted patent application following the provided outline:

# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates to an automated radiosynthesizer system for the production of positron emission tomography (PET) tracers. More specifically, the invention provides a modular, computer-controlled platform capable of performing high-pressure, high-temperature radiochemical reactions while minimizing exposure of system components to corrosive reagents and volatile solvents. The system incorporates disposable cassettes containing all fluid paths and a movable gas handling robot to enable rapid transition between different tracer syntheses without hardware reconfiguration.  

## BACKGROUND  

Positron emission tomography has become an essential tool for non-invasive disease detection, cancer staging, and drug development. While automated synthesis of common PET tracers like [18F]FDG is well-established, many promising 18F-labeled tracers remain difficult to produce automatically due to their demanding synthesis conditions. These tracers often require high-temperature reactions in volatile solvents or involve corrosive reagents that exceed the pressure and chemical resistance limits of conventional radiosynthesizers.  

Previous attempts to automate such syntheses have required modifications to the chemistry itself to accommodate equipment limitations, potentially compromising yield or specific activity. Traditional synthesizers face particular challenges with permanent tubing and valve connections that become exposure points for high pressures and corrosive reagents. There exists a need for a radiosynthesizer that can handle these demanding conditions while maintaining reliability and enabling straightforward transition from tracer development to routine production.  

## SUMMARY  

The present invention provides an automated radiosynthesizer system that overcomes the limitations of conventional equipment through several key innovations. The system comprises three main components working in concert: multiple reactor modules, a reagent and gas handling robot, and disposable cassettes containing all fluid paths.  

Each reactor module features a movable reaction vessel that seals against different positions on a disposable cassette, enabling dynamic reconfiguration of fluid paths for different chemical operations. This design eliminates permanent tubing connections to the reaction vessel, allowing sealed high-pressure reactions while protecting sensitive system components. The reactors incorporate active liquid cooling and precise temperature control up to 185°C.  

The reagent and gas handling robot provides a single movable interface for delivering inert gas, vacuum, and sensitive reagents from sealed vials. This reduces the number of valves and seals compared to systems with multiple fixed connections. Disposable cassettes contain all wetted components including reagent vials, fluid paths, and purification cartridges, eliminating cleaning requirements between syntheses.  

The system has been validated through successful automated production of challenging PET tracers including d-[18F]FAC and l-[18F]FMAU, demonstrating comparable or superior yields to manual methods while handling pressures and temperatures that exceed conventional synthesizer capabilities. The modular design supports rapid switching between different tracer syntheses by simply changing cassettes and software protocols.  

## DETAILED DESCRIPTION OF ILLUSTRATED EMBODIMENTS  

The radiosynthesizer system comprises three primary subsystems: reactor assemblies, a reagent and gas handling robot, and disposable cassettes, all controlled by an integrated computer system. Each component will be described in detail below.  

**Reactor Assemblies**  
The system incorporates three independent reactor modules, each containing a 5 mL glass V-vial held within a three-segment spring-loaded chuck. The chuck ensures excellent thermal contact between the vial and heating elements, with each segment containing a 100W cartridge heater and K-type thermocouple for precise temperature control. Active liquid cooling circulates coolant through channels in all reactors in series, enabling rapid temperature reduction after heating.  

Each reactor module moves horizontally among several predefined positions beneath its cassette and vertically via pneumatic cylinders to seal against different portions of the cassette gasket. This dynamic sealing capability enables distinct operations including reagent addition, sealed reactions, evaporations, and product transfers. Hall effect sensors provide position feedback, while a rear-mounted camera allows visual monitoring of liquid levels and reaction progress.  

**Reagent and Gas Handling Robot**  
A three-axis robotic system incorporates both a vial gripper and gas supply interface. The gripper transports reagent vials between storage positions and addition locations on the cassettes, while the gas supply provides both inert gas and vacuum through a single movable interface. This design eliminates multiple fixed gas connections that could leak or fail.  

The gas supply features a spring-loaded vacuum port mounted above an inert gas port to ensure proper sealing when engaged. An in-line check valve prevents backflow of vapors, while a cold trap protects the vacuum pump from solvent condensation. The robot includes safety mechanisms such as Hall effect sensors to detect missing vials and prevent movement when components are not properly positioned.  

**Disposable Cassettes**  
The molded polyurethane cassettes contain all disposable fluid path components including stainless steel needles, PTFE tubing, three-way stopcock valves, and a PTFE-coated silicone gasket. Each cassette includes:  

- Eleven inverted reagent vial storage positions with crimped septum caps  
- Three addition positions with dual upward-pointing needles for fluid delivery and vial pressurization  
- Multiple stopcock valves for configuring fluid paths during cartridge trapping and purification  
- Built-in collection vials for waste and intermediate products  

Cassettes slide into alignment rails on the synthesizer and lock into place, with stopcock valves engaging rotary pneumatic actuators. The design allows complete replacement of all wetted components between syntheses, eliminating cleaning requirements and cross-contamination risks.  

**Control System**  
A Linux server communicates with a programmable logic controller (PLC) that coordinates all subsystem operations including:  

- Precise temperature control of reactors via solid-state relays  
- Movement of linear actuators for reactor positioning and robot motion  
- Pneumatic operations for vial gripping, gas supply engagement, and valve actuation  
- Monitoring of radioactivity detectors and system sensors  

Analog pressure regulators maintain separate gas supplies for pneumatic actuators (60 psig) and liquid transfers (3-15 psig). The software organizes synthesis protocols into modular "unit operations" that can be combined to construct diverse tracer production sequences.  

**System Operation**  
The synthesizer performs automated tracer production through coordinated execution of fundamental chemistry operations:  

1. **Radioisotope Handling**: [18F]fluoride is trapped on preconditioned QMA cartridges with [18O]H2O recovery, then eluted into reaction vessels using the robot-delivered eluent.  

2. **Reagent Addition**: The robot transports sealed vials to addition positions where inert gas pressure transfers reagents through cassette fluid paths into reaction vessels.  

3. **Sealed Reactions**: Reactors press vessels against cassette gaskets to maintain high internal pressures during heating, with <1.5% solvent loss measured at 165°C.  

4. **Evaporations**: Simultaneous heating, vacuum application, and inert gas flow remove solvents through dedicated cassette ports.  

5. **Transfer and Purification**: Products move through purification cartridges via gas pressure, with stopcock valves directing flow to waste or subsequent reactors.  

**Validation and Performance**  
The system has demonstrated successful production of multiple PET tracers including:  

- d-[18F]FAC: 15.5% ± 2.3% decay-corrected yield (n=12) in 110 min  
- l-[18F]FMAU: 13.2% ± 3.1% decay-corrected yield (n=10) in 105 min  

These results compare favorably with manual methods while offering superior reproducibility and reduced radiation exposure. The disposable cassette approach has enabled synthesis of additional tracers including [18F]FDG, [18F]FLT, and [18F]SFB without hardware modifications.  

The system's unique combination of movable reactors, disposable cassettes, and centralized reagent handling provides unprecedented flexibility for both tracer development and routine production while overcoming the pressure and corrosion limitations of conventional synthesizers.