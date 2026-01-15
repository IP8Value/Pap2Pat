## TECHNICAL FIELD

- define radiochemistry applications

The present invention pertains to the field of radiochemistry, specifically to automated systems and methods for the synthesis of positron-emitting radiopharmaceuticals used in molecular imaging and diagnostic oncology. The invention is directed toward the precise, reproducible, and safe production of fluorine-18-labeled compounds and other short-lived radionuclide-based tracers under controlled chemical and physical conditions. These radiopharmaceuticals are essential for positron emission tomography (PET), a non-invasive imaging modality that enables the visualization of metabolic, biochemical, and physiological processes in living organisms. The system is particularly suited for the synthesis of complex, high-pressure, and thermally sensitive tracers that require stringent control over reaction parameters such as temperature, pressure, solvent composition, and reagent delivery timing. The invention facilitates the transition from research-scale radiolabeling protocols to standardized, high-throughput clinical manufacturing environments by eliminating manual handling, minimizing radiation exposure, and ensuring consistent product quality across multiple synthesis runs. The system is adaptable to a broad spectrum of radiochemical reactions, including nucleophilic substitution, electrophilic fluorination, esterification, hydrolysis, and solid-phase purification, thereby serving as a foundational platform for the development and deployment of next-generation PET tracers in both preclinical and clinical settings.

## BACKGROUND

- introduce PET imaging

Positron emission tomography (PET) is a highly sensitive molecular imaging technique that allows for the non-invasive visualization of biochemical processes in vivo through the detection of gamma rays emitted by positron-emitting radionuclides incorporated into biologically active molecules. The most widely used radiotracer in clinical PET imaging is 2-[18F]fluoro-2-deoxy-d-glucose ([18F]FDG), which exploits the increased glucose metabolism characteristic of many malignant tissues. However, the expanding scope of molecular imaging has necessitated the development of novel tracers targeting specific biological pathways, including cell proliferation, apoptosis, receptor expression, enzyme activity, and gene expression. These advanced tracers often require complex synthetic routes involving volatile solvents, elevated temperatures, high pressures, and corrosive reagents—conditions that are difficult to achieve reliably using conventional manual or semi-automated methods. The demand for these specialized tracers has outpaced the capacity of existing automated platforms, which are typically limited by rigid fluidic architectures, fixed reaction volumes, and insufficient thermal and pressure tolerance. As a result, many promising radiopharmaceuticals remain confined to research laboratories due to the lack of a scalable, robust, and user-friendly synthesis system capable of accommodating their unique chemical requirements.

- motivate automated radiosynthesizers

The manual synthesis of radiopharmaceuticals exposes personnel to significant ionizing radiation, increases the risk of human error, and introduces variability in product yield and purity. Automated radiosynthesizers were developed to mitigate these challenges by providing reproducible, shielded, and remotely operated synthesis environments. These systems reduce radiation dose to operators, improve consistency across batches, and enable the production of radiotracers in decentralized settings such as hospital-based cyclotron facilities. However, existing commercial platforms are largely optimized for the synthesis of [18F]FDG and a limited number of structurally similar compounds. They lack the flexibility to handle reactions requiring pressures exceeding 100 psi, temperatures above 120°C, or the use of aggressive reagents such as anhydrous hydrogen fluoride or triflic acid. Furthermore, many systems rely on permanently installed tubing and valves that become contaminated or degraded during high-temperature or high-pressure operations, necessitating frequent maintenance and limiting their utility for novel tracer development. There is a critical need for a next-generation automated radiosynthesizer that overcomes these limitations by integrating dynamic fluidic reconfiguration, modular reagent delivery, and intelligent process control to support a broader range of radiochemical transformations with minimal operator intervention.

- describe limitations of current synthesizers

Current automated radiosynthesizers are constrained by fixed fluidic pathways, static reactor configurations, and inadequate pressure containment mechanisms. Most systems employ rigid, permanently connected tubing and valve networks that are susceptible to degradation under elevated temperatures and pressures, leading to leaks, clogging, or cross-contamination between synthesis runs. The inability to isolate reaction vessels from external fluidic components during high-pressure steps restricts the use of volatile solvents and aggressive reagents, effectively limiting the chemical space accessible for tracer development. Additionally, many platforms rely on external syringe pumps and gravity-driven fluid transfers that lack precision and are prone to dead volume losses, resulting in reduced radiochemical yields and compromised specific activity. Reagent handling is often performed through manual insertion of vials or pre-loaded reservoirs that cannot be sealed or pressurized during operation, increasing the risk of evaporation, oxidation, or contamination. Furthermore, the software interfaces of existing systems are typically proprietary, inflexible, and require extensive programming expertise to modify synthesis protocols, making them unsuitable for rapid adaptation to new chemical methodologies. These limitations collectively hinder the translation of novel radiotracers from discovery to clinical application.

- identify need for hybrid system

There exists a compelling need for a hybrid automated radiosynthesizer that integrates the precision of robotic reagent handling, the thermal and pressure resilience of dynamically sealed reaction vessels, and the adaptability of modular, disposable fluidic cassettes within a unified, software-controlled platform. Such a system must eliminate the reliance on fixed plumbing by enabling the physical repositioning of the reaction vessel relative to fixed reagent and gas interfaces, thereby creating transient, high-integrity fluidic connections that are established only when needed. This approach not only prevents the accumulation of chemical residues in permanent tubing but also permits the use of high-pressure and high-temperature reactions without compromising system integrity. Furthermore, the system must incorporate an automated, multi-axis robotic arm for the precise manipulation of sealed reagent vials, integrated vacuum and inert gas delivery systems with real-time pressure feedback, and a centralized control architecture that allows for the creation, storage, and execution of customizable synthesis protocols via an intuitive graphical interface. The combination of these features into a single, cohesive system represents a paradigm shift in automated radiochemistry, bridging the gap between laboratory-scale innovation and clinical-scale production.

## SUMMARY

- introduce automated radiosynthesizer device

The present invention discloses an automated radiosynthesizer device designed for the precise, reproducible, and safe synthesis of positron-emitting radiopharmaceuticals under controlled chemical conditions. The device comprises a modular architecture integrating multiple reactor assemblies, a reagent and gas handling robot, disposable fluidic cassettes, and a centralized control system, all operating in synchronized coordination to perform complex radiochemical syntheses without manual intervention. The system is configured to accommodate a wide range of radiolabeling reactions, including those requiring elevated temperatures, high internal pressures, and the use of volatile or corrosive reagents, by dynamically reconfiguring fluidic pathways through mechanical movement of the reaction vessel relative to fixed interfaces. This design eliminates the need for permanent tubing and valves exposed to reactive media, thereby enhancing system reliability, reducing contamination risk, and facilitating rapid transition between different synthesis protocols. The device is particularly suited for the production of novel PET tracers such as nucleoside analogs, amino acid derivatives, and peptide-based ligands that have previously been inaccessible due to the limitations of conventional automated synthesizers.

- describe reactor assemblies

The reactor assemblies of the automated radiosynthesizer each comprise a three-segment spring-loaded chuck designed to securely hold a standard 5-mL glass V-vial in precise alignment with the underside of a disposable cassette. Each segment contains an independent cartridge heater and K-type thermocouple for localized temperature control, enabling uniform heating of the reaction vessel with feedback regulation up to a maximum temperature of 185°C. The reactor is mounted on a vertically oriented linear actuator that allows the vessel to be raised and lowered to seal against specific gasketed regions of the cassette, thereby creating a hermetic environment for sealed reactions, evaporations, or reagent additions. Active liquid cooling is provided through a closed-loop circulation system that pumps a propylene/ethylene glycol-water mixture through cooling channels integrated into each reactor, enabling rapid thermal quenching following reaction completion. A magnetic stir bar, driven by a DC motor mounted externally to the reactor, ensures homogeneous mixing of reaction contents during heating or evaporation cycles. The reactor assemblies are arranged in a linear array, permitting independent operation of up to three reaction vessels simultaneously, each capable of executing distinct unit operations within the same synthesis protocol.

- describe disposable cassettes

The disposable cassettes are injection-molded polyurethane modules that contain all fluidic components, reagent storage vials, and sealing interfaces necessary for a single synthesis run. Each cassette incorporates stainless steel needles, PTFE-coated silicone gaskets, three-way stopcock valves, and internal tubing arranged to define distinct functional zones: reagent storage positions, reagent addition ports, gas inlet and vacuum ports, waste and recovery vial locations, and purification cartridge mounts. The bottom surface of the cassette is sealed against the top of the reaction vessel when the reactor is raised, forming a pressure-tight interface that prevents leakage during high-temperature or high-pressure operations. The cassettes are designed for single-use, eliminating the need for cleaning or sterilization between syntheses and enabling immediate transition from one tracer to another by simply replacing the cassette and loading a corresponding software protocol. The cassettes feature alignment features that ensure precise positioning within the synthesizer, and their modular design permits the integration of custom fluidic paths tailored to specific synthetic schemes, including solvent exchange, cartridge trapping, and sequential multi-step purifications.

- describe reagent and gas handling robot

The reagent and gas handling robot is a three-axis robotic system comprising an x-axis and y-axis linear servomotor for horizontal positioning and a z-axis pneumatic actuator for vertical movement. Mounted to the robot is a vial gripper capable of grasping, lifting, and relocating sealed 13-mm crimped vials between storage and addition positions within the cassette. Simultaneously, a second z-axis actuator supports a gas supplier module equipped with an inert gas outlet and a vacuum port, both of which engage with corresponding inlet ports on the top surface of the cassette to enable pressurization, depressurization, and vapor removal during evaporation and transfer operations. The gas supplier is spring-loaded to ensure consistent sealing pressure and to prevent mechanical collision during movement. The robot incorporates Hall-effect sensors to verify the position of the gripper and gas supplier, preventing motion if components are not fully retracted. An in-line check valve on the inert gas line prevents backflow of volatile vapors, and a cold trap cooled by dry ice and methanol is positioned between the vacuum port and the pump to condense and retain volatile organic compounds, protecting the pump and preserving radiochemical yield.

- describe control system

The control system of the automated radiosynthesizer is a distributed architecture comprising a Linux-based server, a programmable logic controller (PLC), and multiple embedded microcontrollers that coordinate the operation of all subsystems. The server hosts a web-based user interface that allows users to create, modify, and execute synthesis protocols through a drag-and-drop interface defining unit operations. The PLC receives high-level commands from the server and translates them into low-level signals for motor controllers, heater drivers, valve actuators, and radiation detectors. Each reactor is independently controlled by a microcontroller interfacing with dedicated heater controllers, stir motor drivers, and thermocouple amplifiers. The microcontroller also communicates with a radiation amplifier for real-time activity monitoring, an HPLC controller for post-synthesis purification, and a valve driver array for managing pneumatic actuators and solenoid valves. The system employs a RoboNET network controller to synchronize the motion of the reagent and gas handling robot, ensuring precise coordination between vial manipulation and gas delivery. All operational parameters, including temperature, pressure, timing, and radiation counts, are logged and stored in a database server for audit trail compliance and process optimization.

- introduce automated method of performing radiosynthesis

The automated method of performing radiosynthesis involves the sequential execution of predefined unit operations, each of which corresponds to a discrete chemical or physical transformation such as reagent addition, evaporation, sealed reaction, or purification. The method begins with the loading of a disposable cassette and the initiation of a synthesis protocol selected via the control system interface. The reagent and gas handling robot retrieves sealed reagent vials from storage positions and delivers their contents to designated addition ports on the cassette by pressurizing the vials with inert gas. The reactor assemblies then move to specific positions to seal the reaction vessel against the cassette gasket, enabling high-pressure reactions or controlled evaporation. Temperature and stirring are regulated in real time according to protocol parameters, and vacuum is applied during evaporation to remove solvents while inert gas flow assists in vapor displacement. After completion of the reaction, the crude product is transferred to a purification cartridge via pressurized inert gas, followed by elution of the desired radiotracer into a fresh reaction vessel. The final product is then collected and subjected to semi-preparative HPLC for purification and analytical HPLC for quality control. The entire sequence is performed without manual intervention, ensuring consistency, safety, and reproducibility.

- describe moving reactor vial

The reactor vial is moved vertically and horizontally within the synthesizer to align with fixed functional interfaces on the disposable cassette, thereby enabling dynamic reconfiguration of the fluidic pathway without the use of permanent tubing or valves. When a sealed reaction is required, the reactor is raised until the top of the vial contacts a gasketed region of the cassette, forming a hermetic seal that isolates the reaction chamber from external components. During reagent addition, the reactor is lowered to expose the vial to a needle port on the cassette, allowing pressurized reagent to be delivered directly into the vessel. For evaporation, the reactor is positioned over a vacuum and inert gas port, enabling simultaneous heating and vapor removal. For transfer operations, the reactor is aligned with a dip tube connected to a purification cartridge, allowing the contents of the vessel to be expelled under pressure. The precise movement of the reactor vial is monitored by linear encoders and confirmed by Hall-effect sensors, ensuring accurate positioning and preventing mechanical interference during operation.

- perform operations on radiosynthesis reagent

The system performs a series of chemical and physical operations on radiosynthesis reagents, including pressurized delivery, thermal activation, solvent evaporation, catalytic reaction, and solid-phase purification. Reagents stored in sealed vials are delivered to the reaction vessel by pressurizing the vial with inert gas, forcing the liquid through a needle into the vessel without exposure to ambient atmosphere. Once delivered, the reagent may be subjected to elevated temperatures to initiate a nucleophilic substitution or hydrolysis reaction, with stirring ensuring homogeneity. Solvent evaporation is performed by applying vacuum while maintaining elevated temperature and inert gas flow to remove residual solvents such as acetonitrile or ethanol. The reaction mixture may then be transferred to a solid-phase cartridge for purification, where impurities are retained while the radiotracer is selectively eluted using a defined solvent. Each operation is executed under closed, controlled conditions, minimizing losses, preventing contamination, and maximizing radiochemical yield and specific activity.

- transfer radiosynthesis product

The transfer of the radiosynthesis product is accomplished by pressurizing the reaction vessel with inert gas and directing the flow through a dip tube connected to a purification cartridge or a downstream reaction vessel. The direction of flow is controlled by switching a three-way stopcock valve within the cassette, allowing the product to be either trapped on the cartridge for washing or eluted into a fresh vessel for further processing. The transfer is performed under controlled pressure and temperature to prevent precipitation or degradation of the radiotracer. The system ensures complete transfer by monitoring the duration and pressure differential, and by verifying the absence of residual volume in the source vessel. Multiple transfer steps may be performed sequentially, enabling multi-stage purification protocols such as cartridge trapping, washing, and elution without manual intervention.

- describe alternative embodiment

An alternative embodiment of the invention incorporates a dual-gripper reagent handling system capable of simultaneously manipulating two reagent vials, thereby reducing the time required for sequential additions and enabling parallel reagent delivery to multiple reaction vessels. This embodiment further includes an integrated solvent drying module that employs molecular sieves or desiccant cartridges to remove trace water from [18F]fluoride prior to radiolabeling, eliminating the need for separate azeotropic distillation steps. The control system in this embodiment is enhanced with machine learning algorithms that optimize reaction parameters based on historical synthesis data, adjusting temperature ramps, gas pressures, and timing intervals in real time to maximize yield and purity across diverse tracer chemistries.

- actuate reagent gas handling robot

The reagent and gas handling robot is actuated by a combination of linear servomotors and pneumatic actuators under the command of the central control system. Upon initiation of a reagent addition or gas delivery operation, the robot moves the vial gripper to the designated storage position, lowers the gripper to engage the vial, and lifts it to a clearance height. The robot then translates horizontally to align the vial with the reagent addition port on the cassette, lowers the vial onto the needle interface, and activates the inert gas supply to pressurize the vial and initiate fluid transfer. Simultaneously, the gas supplier is actuated to lower its vacuum and inert gas ports onto the corresponding inlet ports on the cassette, establishing a sealed flow path for vapor removal or gas assist. The robot’s motion is constrained by feedback from Hall-effect sensors, ensuring that all components are fully retracted before any horizontal movement occurs, thereby preventing mechanical damage and ensuring operational safety.

- perform evaporation

Evaporation is performed by positioning the reaction vessel over a dedicated evaporation zone on the cassette, where the top of the vessel is sealed against a gasketed region containing both vacuum and inert gas inlet ports. The reactor is heated to a temperature sufficient to volatilize the solvent, while the vacuum pump is activated to remove vapor from the system. Simultaneously, a low-pressure stream of inert gas is introduced through a separate port to facilitate the sweeping of vapor from the vessel, enhancing evaporation efficiency and preventing condensation on the gasket. The duration of evaporation is determined by the volume and volatility of the solvent, and is programmatically controlled with a safety margin to ensure complete removal. The system monitors the progress of evaporation through visual feedback from a camera mounted behind the reactor and adjusts the vacuum pressure or inert gas flow if necessary. Upon completion, the reactor is cooled, and the vessel is ready for the next unit operation.

- describe another embodiment

Another embodiment of the invention integrates an automated HPLC injection system directly into the synthesizer, eliminating the need for manual sample loading. In this embodiment, the purification cartridge outlet is connected to a motorized HPLC injection valve that automatically loads the eluate into the analytical or semi-preparative HPLC system upon completion of the purification step. The HPLC run is initiated by the control system, and the collected fractions are analyzed in real time for radiochemical purity. The system then directs the desired fraction into a final product vial, completing the entire synthesis-to-purification workflow without operator intervention. This embodiment further includes a radiation-detection feedback loop that terminates the HPLC run upon detection of the target radiotracer peak, ensuring optimal collection efficiency and minimizing waste.

- actuate reagent gas handling robot

In this embodiment, the reagent and gas handling robot is actuated by a coordinated sequence of linear and pneumatic movements, initiated by the control system in response to protocol commands. The robot first moves the vial gripper to the designated storage location, engages the vial using a spring-loaded pinch mechanism, and lifts it to a safe clearance height. The robot then translates along the x- and y-axes to position the vial directly above the reagent addition port on the cassette. The gripper lowers the vial onto a pair of needles, one for fluid delivery and one for inert gas pressurization, establishing a sealed fluidic connection. The gas supplier is simultaneously actuated to lower its dual ports onto the cassette’s gas inlet ports, engaging the vacuum and inert gas lines. The system then opens the appropriate solenoid valves to pressurize the vial and initiate fluid transfer. Upon completion, the gripper lifts the vial, the gas supplier retracts, and the robot returns to its home position, all under closed-loop sensor verification.

- describe manufacturing multiple PET tracers

The automated radiosynthesizer enables the sequential or parallel manufacturing of multiple PET tracers by simply swapping disposable cassettes and loading corresponding synthesis protocols. Each cassette is pre-configured with the reagents, purification cartridges, and fluidic pathways specific to a particular radiotracer, allowing for rapid transition between chemically distinct compounds such as [18F]FDG, l-[18F]FMAU, d-[18F]FAC, [18F]FLT, and [18F]SFB without any hardware reconfiguration. The system’s modular architecture and software-driven protocol execution permit the production of multiple tracers in a single operational cycle, either by running sequential syntheses on individual reactors or by utilizing the three-reactor array for parallel synthesis of different compounds. The system logs all parameters for each synthesis, enabling traceability, regulatory compliance, and batch-to-batch comparison. This capability transforms the radiosynthesizer from a single-tracer production tool into a versatile platform for high-throughput radiotracer manufacturing in both research and clinical settings.

## DETAILED DESCRIPTION OF ILLUSTRATED EMBODIMENTS

- illustrate automated radiosynthesizer

The automated radiosynthesizer is illustrated as a compact, shielded enclosure housing three vertically aligned reactor assemblies, a reagent and gas handling robot mounted on a gantry above the reactors, and a front-loading cassette tray for disposable fluidic modules. The entire system is enclosed within a radiation-shielded cabinet constructed of lead-lined steel, with access ports for reagent loading and product collection. The control system is integrated into a separate console connected via Ethernet, featuring a touchscreen interface for protocol selection, real-time monitoring, and emergency abort functions. The reactor assemblies are arranged in a linear configuration, each positioned beneath a corresponding cassette slot. The reagent and gas handling robot spans the width of the system, enabling access to all cassette positions. A camera is mounted behind each reactor, providing real-time visual feedback of liquid levels, phase separation, and reagent addition. The system is designed for integration into a hot cell or shielded laboratory environment, with all pneumatics, electrical connections, and cooling lines routed through sealed penetrations to maintain containment.

- describe synthesizer components

The synthesizer components include three reactor assemblies, each comprising a spring-loaded chuck, individual heater and thermocouple arrays, a magnetic stir motor, and a vertical linear actuator. The reagent and gas handling robot consists of a two-axis linear servomotor for horizontal positioning, a pneumatic z-axis actuator for the vial gripper, and a second z-axis actuator for the gas supplier. The disposable cassettes contain stainless steel needles, PTFE-coated silicone gaskets, three-way stopcock valves, and internal tubing arranged to define reagent storage, addition, gas inlet, vacuum, waste, recovery, and purification zones. The control system includes a Linux server, a programmable logic controller, multiple microcontrollers, motor drivers, solenoid valve banks, analog pressure regulators, a cold trap, a vacuum pump, a coolant circulation system, a video server, and an HPLC injection valve. All components are interconnected through a RoboNET network and a centralized command architecture that ensures synchronized operation.

- explain control system interface

The control system interface is a web-based graphical user interface accessible via any standard client device connected to the network. The interface presents a drag-and-drop protocol builder that allows users to assemble synthesis workflows by selecting from a library of predefined unit operations such as Add, Evaporate, React, Transfer, and Purify. Each operation can be customized with parameters including temperature, duration, pressure, gas flow rate, and stirring speed. The interface displays real-time status indicators for each reactor, robot position, valve state, and radiation count. Users may save, load, and share protocols across multiple systems, and the interface enforces version control and audit trails to ensure compliance with regulatory standards. The interface is designed for use by radiochemists with minimal programming experience, reducing the barrier to entry for adopting novel synthesis methodologies.

- introduce client devices

Client devices include standard desktop computers, tablets, and mobile devices capable of accessing the web-based control interface via a secure HTTPS connection. These devices are used to initiate, monitor, and terminate synthesis protocols remotely, allowing operators to oversee multiple synthesis runs from outside the radiation-shielded area. The client devices do not directly control hardware but instead communicate with the central server, which enforces all operational logic and safety protocols. This architecture ensures that no unauthorized modifications can be made to the synthesis process and that all actions are logged and timestamped for regulatory compliance.

- describe client device interface

The client device interface presents a dashboard with a synthesis protocol timeline, a real-time status panel, and a live video feed from each reactor camera. The timeline displays each unit operation as a color-coded block, with progress bars indicating elapsed time and remaining duration. The status panel shows the current temperature, pressure, radiation count, and robot position for each reactor. The interface includes a visual representation of the cassette layout, highlighting which reagent vials have been used and which ports are active. A warning system alerts the user to deviations from protocol parameters, and an abort button allows immediate termination of the synthesis with automatic shutdown of all heating, stirring, and gas flow systems.

- motivate software paradigm shift

The software architecture of the system represents a paradigm shift from proprietary, fixed-sequence synthesizers to an open, modular, and protocol-driven platform. Rather than requiring users to program low-level motor commands or valve sequences, the system abstracts the complexity of radiochemistry into intuitive unit operations that mirror the language of laboratory practice. This abstraction enables radiochemists to focus on chemical design rather than instrumentation, accelerating the translation of novel tracers from discovery to production. The system’s open protocol format allows for community-driven development of new operations, and the cloud-based storage of protocols facilitates collaboration across institutions. This shift transforms the radiosynthesizer from a black-box instrument into a programmable chemistry platform.

- explain synthesis protocol creation

Synthesis protocol creation is performed through a drag-and-drop interface in which users select unit operations from a library and arrange them in sequence. Each operation is parameterized by user-defined values such as temperature, duration, pressure, and flow rate. The system validates the protocol for logical consistency, checking for incompatible sequences such as attempting to add a reagent before the vessel is sealed. Users may insert conditional branches, such as repeating an evaporation step if residual solvent is detected, or pausing for manual intervention if required. Protocols are saved as JSON files and can be shared, versioned, and imported into other systems. The system automatically generates a detailed log of all executed steps, including timestamps, sensor readings, and operator actions.

- define unit operations

Unit operations are discrete, reusable chemical or physical steps that constitute the building blocks of a radiosynthesis protocol. These operations are defined by the system as standardized sequences of hardware actions that correspond to common laboratory procedures. Each unit operation encapsulates the necessary motor movements, valve states, heater settings, and timing parameters required to perform a specific task, such as delivering a reagent, sealing a reaction vessel, or purifying a product.

- list examples of unit operations

Examples of unit operations include Add, which delivers a reagent from a sealed vial to the reaction vessel; Evaporate, which removes solvent under vacuum and inert gas flow; React, which heats and stirs the reaction mixture under sealed conditions; Transfer, which moves the reaction mixture from one vessel to a purification cartridge or another vessel; Purify, which initiates cartridge trapping and elution; and Cool, which activates the liquid cooling system to reduce the temperature of the reaction vessel. Each operation is independently configurable and may be repeated or sequenced as needed.

- illustrate display on client device

The display on the client device presents a horizontal timeline with color-coded blocks representing each unit operation, overlaid with real-time progress indicators. Below the timeline, a schematic of the cassette is shown with active components highlighted in real time. A status panel on the right displays the current temperature, pressure, radiation count, and robot position. A live video feed from each reactor is displayed in a separate window, allowing the operator to visually confirm liquid levels, phase changes, and reagent addition. A warning icon appears if any parameter deviates from the setpoint, and an abort button is prominently displayed for emergency shutdown.

- describe system status display

The system status display provides a comprehensive overview of the current state of all subsystems, including the position of each reactor, the status of the reagent and gas handling robot, the state of each valve and solenoid, the temperature of each heater, the pressure in the gas lines, the vacuum level, and the accumulated radiation count. Each component is represented by a color-coded icon—green for normal, yellow for warning, and red for error. The display updates in real time and logs all changes for audit purposes. A summary panel at the bottom indicates the current unit operation, elapsed time, estimated completion time, and total synthesis duration.

- explain sequencer display

The sequencer display presents a step-by-step execution log of the synthesis protocol, showing each unit operation as it is initiated and completed. The display includes the start and end time of each step, the set parameters, the actual measured values, and any deviations or warnings encountered. The sequencer also records operator interventions, such as manual overrides or aborts, and flags any steps that required manual adjustment. This log is stored in the database server and can be exported for regulatory review or process optimization.

- illustrate current unit operation display

The current unit operation display highlights the active step in the synthesis protocol with a pulsating border and a magnified view of the relevant cassette zone. The display shows the hardware components involved in the operation—such as the reactor position, the gas supplier engagement, and the valve states—with animated arrows indicating fluid or gas flow. The display also shows the remaining time for the operation and the current values of temperature, pressure, and radiation count. If an error occurs, the display highlights the faulting component and suggests troubleshooting steps.

- introduce abort button

The abort button is a large, red, tactile button located on the client device interface and duplicated on the physical control console. Activation of the abort button immediately terminates all ongoing operations, shuts down all heaters and stir motors, closes all solenoid valves, deactivates the vacuum pump, and initiates a safe cooldown sequence. The system locks all hardware until manually reset, and a detailed abort log is generated, including the time of activation, the current unit operation, and the system state at the moment of abort. This feature ensures operator safety and prevents uncontrolled chemical reactions.

- illustrate software architecture

The software architecture is a three-tier system comprising a web server application, a core server application, and a database server application, all running on a Linux server. The web server handles client connections and serves the graphical interface via HTTPS. The core server interprets protocol files, translates unit operations into low-level commands, and communicates with the PLC via Ethernet. The database server logs all operational data, including timestamps, sensor readings, operator actions, and protocol versions. A video server encodes and streams camera feeds to the client devices. All components are secured with role-based access control and encrypted communication protocols.

- describe server-client communication

Server-client communication is conducted over a secure HTTPS connection using JSON-formatted messages. The client sends commands such as “start protocol,” “pause,” or “abort,” and the server responds with status updates, progress indicators, and error messages. All commands are validated on the server side, and no direct hardware control is permitted from the client. The server maintains a persistent connection to the PLC, ensuring real-time command execution. The system employs heartbeat signals to detect communication failures and automatically initiates a safe shutdown if connectivity is lost.

- explain server architecture

The server architecture is built on a Linux operating system with a microservices design, where each functional component—web interface, protocol engine, database, video streaming, and PLC communication—runs as an independent service. Services communicate via RESTful APIs and message queues, ensuring modularity and fault tolerance. The server is configured for redundancy, with backup power and failover capabilities. All software is containerized using Docker, enabling easy deployment, version control, and system updates without disrupting ongoing syntheses.

- describe web server application

The web server application serves the graphical user interface to client devices via HTTPS, authenticates users through LDAP or Active Directory, and manages session tokens. It receives user inputs, validates protocol syntax, and forwards execution commands to the core server. The web server also logs all user interactions and generates audit trails for regulatory compliance. It is designed to be accessible from any modern browser without the need for plugins or additional software.

- describe core server application

The core server application is the central intelligence of the system, responsible for interpreting synthesis protocols, translating unit operations into hardware commands, and coordinating the PLC and subsystems. It maintains a state machine for each reactor and robot, ensuring that operations are executed in the correct sequence and that safety interlocks are enforced. The core server monitors real-time sensor data, detects anomalies, and triggers abort sequences if thresholds are exceeded. It also manages protocol versioning, user permissions, and data logging.

- describe database server application

The database server application stores all operational data, including synthesis protocols, execution logs, sensor readings, operator actions, and quality control results. Data is stored in a relational SQL database with encrypted fields for sensitive information. The database supports querying for batch analysis, process optimization, and regulatory reporting. All data is backed up daily and retained for a minimum of seven years to comply with FDA and EMA guidelines.

- describe video server application

The video server application captures analog video signals from the reactor-mounted cameras, encodes them into H.264 streams, and transmits them to client devices over the network. The video feed is synchronized with the synthesis timeline, allowing operators to correlate visual observations with process steps. The server supports multiple concurrent streams and includes motion detection to trigger recording during critical operations. Video files are stored locally and archived for quality assurance reviews.

- describe command line interface

The command line interface provides advanced users with direct access to low-level system commands for debugging, calibration, and maintenance. It allows for manual control of individual actuators, valves, and sensors, and supports scripting for automated testing. Access to the command line interface is restricted to authorized personnel and requires multi-factor authentication. All commands executed via the interface are logged and subject to the same audit trail as graphical interface actions.

- describe synthesizer subsystems

The synthesizer subsystems include the reactor assemblies, the reagent and gas handling robot, the disposable cassettes, the control system, the cooling system, the vacuum system, the inert gas supply, the HPLC injection valve, and the radiation detection system. Each subsystem is independently monitored and controlled by the central system, with redundant sensors and fail-safe mechanisms to ensure operational integrity. The subsystems communicate through a unified network protocol, enabling synchronized operation and centralized diagnostics.

- describe reactor assemblies

Each reactor assembly consists of a three-segment spring-loaded chuck that holds a 5-mL glass V-vial with uniform pressure to ensure optimal thermal contact. Each segment contains a 100-W cartridge heater and a K-type thermocouple for independent temperature feedback. The reactor is mounted on a vertical linear actuator capable of precise positioning with a resolution of 0.1 mm. A magnetic stir bar is driven by a DC motor mounted externally, and the entire assembly is cooled by a closed-loop liquid coolant system. The reactor is shielded from radiation and designed to withstand repeated thermal cycling without deformation.

- describe spring-biased heating assemblies

The spring-biased heating assemblies are designed to maintain consistent thermal contact between the heater elements and the glass reaction vessel despite thermal expansion and contraction. Each segment of the chuck is mounted on a spring mechanism that applies a constant downward force, ensuring that the heater remains in intimate contact with the vessel wall throughout the synthesis cycle. This design compensates for variations in vessel dimensions and prevents hot spots or incomplete heating, resulting in uniform temperature profiles and reproducible reaction kinetics.

- describe camera and mount

A high-resolution industrial camera is mounted behind each reactor, with a fixed focus lens aligned to provide a clear view of the reaction vessel interior. The camera is mounted on a rigid bracket to prevent vibration during operation and is shielded from radiation exposure. The camera captures real-time video at 30 frames per second and transmits the feed to the video server. The camera is used to monitor liquid levels, phase separation, reagent addition, and the formation of precipitates, providing visual confirmation of process progression.

- describe vertically-oriented actuators

The vertically-oriented actuators are pneumatic cylinders that raise and lower the reactor assemblies to seal the reaction vessel against the cassette gasket. Each actuator is equipped with a linear encoder for position feedback and a Hall-effect sensor to detect the fully raised and fully lowered states. The actuators are rated for 100,000 cycles and are designed to operate under high-pressure conditions without leakage. The system ensures that no horizontal motion of the robot occurs unless the reactor is fully lowered, preventing mechanical interference.

- describe horizontally-oriented actuator

The horizontally-oriented actuator is a dual-axis linear servomotor that moves the reagent and gas handling robot along the x- and y-axes to position the vial gripper and gas supplier over any of the cassette locations. The actuator has a resolution of 1 μm and is capable of rapid, precise movements with repeatability of ±0.05 mm. The actuator is enclosed in a sealed housing to prevent contamination and is lubricated with radiation-resistant grease. Feedback from encoders ensures accurate positioning, and the system verifies alignment before initiating any operation.

- describe reactor assembly movement

Reactor assembly movement is controlled by the central server, which sends commands to the vertical linear actuators based on the current unit operation in the protocol. Movement is executed in a sequence that ensures the vessel is sealed only when necessary and that no fluidic components are engaged during transit. The system verifies the position of each reactor using linear encoders and Hall-effect sensors before proceeding to the next step. The movement is smooth and vibration-free, ensuring the integrity of the reaction mixture and preventing spillage or splashing.

- describe synthesizer 12

Synthesizer 12 refers to the third-generation embodiment of the automated radiosynthesizer, incorporating all features described herein, including three reactor assemblies, a dual-gripper reagent handling robot, integrated HPLC injection, and a cloud-connected control system. Synthesizer 12 is designed for clinical manufacturing under 21 CFR Part 212 and USP 823 guidelines, with full electronic recordkeeping, audit trails, and user access controls. The system has been validated for the production of multiple PET tracers with radiochemical yields exceeding 90% and specific activities greater than 150 GBq/μmol.

- introduce disposable cassettes 80

Disposable cassettes 80 are injection-molded polyurethane modules designed for single-use in the automated radiosynthesizer. Each cassette contains all fluidic components, reagent storage positions, and sealing interfaces required for a complete synthesis. The cassettes are pre-assembled with stainless steel needles, PTFE-coated silicone gaskets, and internal tubing, eliminating the need for manual assembly or cleaning. The cassettes are inserted into the synthesizer via a sliding rail system and locked into place by mechanical clamps. Alignment features ensure precise positioning relative to the reactor assemblies and robot.

- describe cassette 80 structure

The structure of cassette 80 is a rectangular, multi-layered module with a flat top surface containing gas and vacuum inlet ports, and a bottom surface containing gaskets and needle ports. The interior contains molded fluidic channels, stopcock valves, and vial retention wells. The cassette is divided into functional zones: reagent storage, reagent addition, gas inlet, vacuum port, waste vial, recovery vial, and purification cartridge mount. All internal surfaces are chemically inert and resistant to organic solvents and radiation. The cassette is designed to be compatible with standard 13-mm crimped vials and 5-mL reaction vessels.

- illustrate cassette 80 bottom surface

The bottom surface of cassette 80 contains a continuous rubber gasket with multiple sealing zones corresponding to the positions of the reactor assemblies. Each sealing zone is aligned with a needle port for reagent addition and a vacuum or inert gas port. The gasket is composed of a PTFE-coated silicone material that forms a hermetic seal when pressed against the top of the reaction vessel. The bottom surface also contains alignment pins and grooves that mate with corresponding features in the synthesizer to ensure precise positioning.

- describe gaskets 90a, 90b, 90c, 90d, 90e

Gaskets 90a through 90e are distinct sealing regions on the bottom surface of cassette 80, each corresponding to a specific unit operation: 90a for evaporation, 90b for sealed reaction, 90c for reagent addition, 90d for transfer, and 90e for purification. Each gasket is composed of a PTFE-coated silicone material with a thickness of 1.5 mm and a Shore A hardness of 40, ensuring resilience under repeated compression and resistance to solvent degradation. The gaskets are replaceable as part of the disposable cassette and are designed to maintain integrity over 100 sealing cycles.

- illustrate cassette 80 top view

The top view of cassette 80 shows the arrangement of gas and vacuum inlet ports, reagent vial storage positions, and cartridge mounting clips. The inert gas ports are positioned adjacent to the vacuum ports, with the vacuum port elevated on a spring-loaded mechanism to ensure proper sealing. The reagent vial storage positions are arranged in a linear array, each with a septum cap for piercing. The top view also shows the location of the stopcock valve actuators and the waste and recovery vial slots.

- describe vial storage positions 88

Vial storage positions 88 are eleven recessed wells on the cassette designed to hold 13-mm crimped septum-cap vials in an inverted orientation. Each well contains a spring-loaded pin that engages the vial cap to prevent accidental dislodgement. The vials are stored upside-down to minimize dead volume and to facilitate complete delivery of reagents by pressurization. The vial positions are labeled with unique identifiers that correspond to entries in the synthesis protocol.

- describe reagent addition positions 108a, 108b, 108c

Reagent addition positions 108a, 108b, and 108c are locations on the cassette where dual-needle interfaces are mounted for the delivery of reagents to the reaction vessel. Position 108a is designated for primary reagent addition, 108b for secondary reagent or eluent delivery, and 108c for the addition of aqueous solutions such as [18F]fluoride. Each position contains an upper needle for inert gas pressurization and a lower needle for fluid delivery, both connected to internal tubing that terminates at the reaction vessel sealing zone.

- describe inlet gas ports 114a, 114b, 114c, 114d, 114e

Inlet gas ports 114a through 114e are openings on the top surface of the cassette that engage with the gas supplier to deliver inert gas or vacuum to the reaction vessel. Port 114a is for inert gas during evaporation, 114b for inert gas during reaction, 114c for vacuum during evaporation, 114d for inert gas during transfer, and 114e for vacuum during purification. Each port is surrounded by a rubber gasket to ensure a gas-tight seal when engaged by the robot.

- describe cartridge waste vial location 116

Cartridge waste vial location 116 is a recessed well on the cassette designed to receive a 2-mL vial for collecting waste fluid during purification steps. The vial is positioned beneath the outlet of the purification cartridge, allowing wash solvents and impurities to be collected without contaminating the reaction vessel. The vial is sealed with a septum cap and may be removed after synthesis for disposal.

- describe recovery vial location 118

Recovery vial location 118 is a dedicated well on the cassette for collecting valuable solvents or aqueous fractions during solvent exchange or trapping operations. For example, during [18F]fluoride elution, [18O]H2O may be recovered in this vial for reuse. The vial is sealed with a septum cap and is designed to withstand high pressure and temperature during operation.

- describe vacuum port 120

Vacuum port 120 is a centrally located inlet on the top surface of the cassette that engages with the vacuum line of the gas supplier. The port is connected to a cold trap and a vacuum pump, enabling the removal of volatile solvents during evaporation and purification. The port is positioned above the inert gas port to prevent cross-contamination and is spring-loaded to ensure consistent sealing pressure.

- describe inlet ports 122

Inlet ports 122 are external connection points on the cassette for the introduction of radioisotopes or external reagents. These ports are Luer fittings that can be connected to a cyclotron output line or a syringe pump. The inlet is connected to a stopcock valve that directs the flow to either the trapping cartridge or directly to the reaction vessel, depending on the protocol.

- describe outlet ports 124

Outlet ports 124 are external Luer fittings on the cassette that allow the transfer of purified product to an external HPLC system or to a downstream cassette. These ports are connected to the output of the purification cartridge and are controlled by a stopcock valve that selects between waste collection and product transfer. The outlets are designed to accept standard tubing for automated HPLC injection.

- illustrate cassette 80 side profile

The side profile of cassette 80 illustrates the vertical arrangement of fluidic components, showing the alignment of the reagent vial storage positions, the reagent addition needles, the gas and vacuum ports, and the internal tubing that connects these components to the reaction vessel sealing zones. The profile also shows the placement of the stopcock valves, the cartridge mount, and the waste and recovery vial wells, demonstrating the compact, integrated design of the cassette.

- illustrate gas flow path 128 and vacuum flow path 130

Gas flow path 128 and vacuum flow path 130 are internal channels within the cassette that connect the inlet ports to the reaction vessel and purification cartridge. Gas flow path 128 delivers inert gas from the gas supplier to pressurize reagent vials and assist in solvent evaporation. Vacuum flow path 130 connects the vacuum port to the reaction vessel during evaporation and to the purification cartridge during washing. Both paths are constructed of chemically inert materials and are designed to minimize dead volume and prevent cross-contamination.

- illustrate reactor assembly 50 in EVAPORATE unit process

In the EVAPORATE unit process, reactor assembly 50 is positioned such that the top of the reaction vessel is sealed against gasket 90a on the cassette. The gas supplier engages vacuum port 114c and inert gas port 114a, while the reactor heaters are activated to elevate the temperature of the vessel. The vacuum pump removes vapor through path 130, while inert gas flows through path 128 to sweep vapor from the vessel. The magnetic stirrer is activated to enhance evaporation efficiency.

- illustrate reactor assembly 50 in REACT unit process

In the REACT unit process, reactor assembly 50 is raised to seal the reaction vessel against gasket 90b, isolating the vessel from external fluidic components. The inert gas port 114b is activated to maintain a slight positive pressure, and the heaters are raised to the reaction temperature. The stir motor is activated to ensure homogeneity. The sealed environment allows for high-pressure reactions without exposure of tubing or valves to reactive media.

- illustrate reactor assembly 50 in ADD unit operation

In the ADD unit operation, reactor assembly 50 is lowered to expose the reaction vessel to reagent addition position 108a. The vial gripper lowers a reagent vial onto the dual needles, and the gas supplier activates inert gas port 114b to pressurize the vial, forcing the reagent through the needle into the vessel. The reactor remains in the lowered position until the transfer is complete, after which it is raised to the REACT position.

- illustrate reactor assembly 50 in TRANSFER unit operation

In the TRANSFER unit operation, reactor assembly 50 is raised to seal against gasket 90d, and the gas supplier activates inert gas port 114d to pressurize the vessel. The pressure forces the reaction mixture through a dip tube into the purification cartridge. The stopcock valve is switched to direct the effluent to the waste vial. After washing, the stopcock is switched again, and the product is eluted into a fresh vessel by repeating the transfer operation with a different solvent.

- describe purification cartridge 132

Purification cartridge 132 is a disposable solid-phase extraction cartridge mounted on a clip near the front of the cassette. The cartridge is connected between the dip tube and a stopcock valve, allowing the crude reaction mixture to be trapped on the stationary phase while impurities are washed away. The cartridge may be silica, C18, or ion-exchange based, depending on the tracer chemistry. The cartridge is replaced with each cassette and is not reused.

- illustrate radioisotope handling configuration

The radioisotope handling configuration involves connecting the [18F]fluoride source to the inlet port 122, which is routed through a QMA cartridge to trap the fluoride anion. The system then switches the stopcock valves to elute the fluoride into the reaction vessel using a solution of K222/K2CO3 in acetonitrile. The entire process is automated and performed under sealed conditions to maximize specific activity and minimize contamination.

- describe reagent and gas handling robot 140

Reagent and gas handling robot 140 is a three-axis robotic system consisting of a two-axis linear servomotor for horizontal positioning and two pneumatic z-axis actuators for vertical movement. One actuator controls the vial gripper, and the other controls the gas supplier. The robot is mounted on a gantry above the cassette tray and is capable of accessing all reagent and gas ports on any of the three cassettes. The robot operates under closed-loop feedback from Hall-effect sensors and encoders to ensure precise positioning.

- describe head portion 142

Head portion 142 is the terminal component of the reagent and gas handling robot, housing the vial gripper and the gas supplier. The gripper is a spring-loaded pinch mechanism that engages the cap of a 13-mm crimped vial. The gas supplier is a dual-port assembly with an inert gas outlet and a vacuum inlet, both of which are spring-loaded to ensure sealing pressure. The head portion is constructed of radiation-resistant stainless steel and is designed for easy cleaning and replacement.

- describe x-axis motion actuator 144

X-axis motion actuator 144 is a linear servomotor that moves the head portion 142 along the length of the cassette tray. The actuator has a stroke of 500 mm and a resolution of 1 μm. It is driven by a pulse-width modulated controller and is equipped with an optical encoder for position feedback. The actuator is enclosed in a sealed housing to prevent contamination.

- describe y-axis motion actuator 146

Y-axis motion actuator 146 is a linear servomotor that moves the head portion 142 perpendicular to the x-axis, enabling access to all three cassettes. The actuator has a stroke of 300 mm and is similarly equipped with an optical encoder and sealed housing. The actuator is synchronized with the x-axis to ensure precise positioning over any cassette location.

- describe z-axis actuators 148, 150

Z-axis actuators 148 and 150 are pneumatic cylinders that control the vertical movement of the vial gripper and the gas supplier, respectively. Actuator 148 raises and lowers the gripper to engage and disengage reagent vials, while actuator 150 raises and lowers the gas supplier to engage the gas and vacuum ports on the cassette. Both actuators are equipped with Hall-effect sensors to detect fully raised and fully lowered positions. The actuators are rated for 100,000 cycles and are lubricated with radiation-resistant grease.

- describe gas manifold 152

Gas manifold 152 is a central distribution block that receives high-pressure inert gas from an external source and divides it into two regulated lines: one for pneumatic actuation (60 psig) and one for fluid transfer (3–15 psig). The manifold contains analog pressure regulators and solenoid valves that are controlled by the PLC to direct gas flow to the appropriate ports. The manifold is constructed of stainless steel and is designed to withstand high-pressure cycling without leakage.

- describe vial gripper 158

Vial gripper 158 is a spring-loaded pinch mechanism mounted on the head portion 142 that engages the cap of a 13-mm crimped vial. The gripper is actuated by a pneumatic cylinder and is designed to securely hold the vial without puncturing the septum. The gripper includes a tactile sensor that confirms successful engagement before initiating movement.

- describe Hall-effect sensors

Hall-effect sensors are magnetic position sensors used throughout the system to detect the fully raised or fully lowered state of the reactor assemblies, the vial gripper, and the gas supplier. These sensors provide fail-safe feedback to the control system, ensuring that no motion occurs unless components are in the correct position. The sensors are immune to electromagnetic interference and are rated for operation in high-radiation environments.

- describe in-line check valve

The in-line check valve is a one-way valve installed on the inert gas line near the delivery point to prevent backflow of volatile vapors into the gas supply system. The valve is constructed of chemically inert materials and is designed to open at a low pressure differential, ensuring smooth gas flow during reagent delivery while preventing contamination of the gas source.

- describe cold-trap

The cold-trap is a glass condenser cooled by a mixture of dry ice and methanol, installed in-line between the vacuum port and the vacuum pump. The cold-trap captures volatile organic solvents and radioactive vapors, preventing them from entering the pump and reducing the risk of contamination. The trap is replaceable and is designed for easy removal and disposal after each synthesis.

- describe source of inert gas

The source of inert gas is a high-pressure cylinder of nitrogen or argon, regulated through a two-stage pressure regulator system to provide two distinct pressure lines: one at 60 psig for pneumatic actuators and one at 3–15 psig for fluid transfer and evaporation. The gas is filtered to remove moisture and particulates and is delivered to the gas manifold through stainless steel tubing.

- describe analog pressure regulators

Analog pressure regulators are mechanical devices that reduce the high-pressure gas supply to precise, stable output pressures. Two regulators are used: one for the high-pressure pneumatic line and one for the low-pressure gas transfer line. The regulators are calibrated to maintain pressure within ±0.5 psig and are designed for continuous operation in a high-radiation environment.

- describe solenoid valve banks

Solenoid valve banks are arrays of electrically actuated valves that control the flow of inert gas and vacuum to the various ports on the cassette. Each valve is a 2-way, normally closed solenoid with a stainless steel body and a PTFE seat. The valves are controlled by the PLC and are rated for 10 million cycles. The valve banks are mounted in the synthesis module and are connected to the gas manifold and vacuum pump.

- describe control system 14 components

Control system 14 components include the Linux server, the programmable logic controller (PLC), the microcontroller, the motor controllers, the heater controllers, the radiation amplifier, the HPLC controller, the valve drivers, the vacuum pump, the coolant pump, the video server, and the network interface. All components are housed in a shielded enclosure and are interconnected via Ethernet and RS-485 buses.

- describe control system 14 components (continued)

The control system also includes a redundant power supply, a battery backup unit, and an emergency stop circuit. All communication is encrypted, and access is controlled through role-based authentication. The system logs all events, including power cycles, software updates, and operator actions, and generates audit trails compliant with 21 CFR Part 11 and EU Annex 11.

- describe PLC alternative to embedded computer 164

A programmable logic controller (PLC) is used as an alternative to an embedded computer for real-time control of hardware components. The PLC is more reliable in high-radiation environments and offers deterministic response times. The PLC is programmed using ladder logic and communicates with the server via Ethernet. The PLC handles all low-level control tasks, including motor positioning, valve actuation, and temperature regulation, while the server handles protocol interpretation and user interface.

- describe motor controllers and RoboNET network controller 182

Motor controllers are pulse-width modulated drivers that control the position and speed of the linear servomotors in the reagent and gas handling robot. The RoboNET network controller 182 is a gateway device that connects the motor controllers to the PLC via a proprietary network protocol. The RoboNET ensures synchronized motion of multiple axes and provides feedback on position, velocity, and torque.

- describe microcontroller 180 interfaces with motor controllers 182

Microcontroller 180 receives high-level commands from the PLC and translates them into low-level signals for the motor controllers 182. The microcontroller uses a real-time operating system to ensure precise timing of motor movements. It also monitors encoder feedback and adjusts motor output to maintain positional accuracy.

- describe microcontroller 180 interfaces with stir motor drivers 184

Microcontroller 180 sends pulse signals to stir motor drivers 184 to control the speed and duration of magnetic stirring in each reactor. The drivers are isolated to prevent electrical interference with radiation detection systems. The microcontroller adjusts stir speed based on the unit operation, with higher speeds used during evaporation and lower speeds during reaction.

- describe microcontroller 180 interfaces with heater controllers 186

Microcontroller 180 communicates with heater controllers 186 to regulate the temperature of each reactor segment. The heater controllers receive setpoints from the microcontroller and adjust power output based on thermocouple feedback. The system uses PID control algorithms to maintain temperature within ±0.5°C.

- describe microcontroller 180 interfaces with radiation amplifier 188

Microcontroller 180 receives analog signals from the radiation amplifier 188, which is connected to a gamma detector mounted on each reactor. The microcontroller converts the signal into counts per second and transmits the data to the server for real-time monitoring and decay correction. The system uses the radiation data to determine the endpoint of synthesis and to calculate specific activity.

- describe microcontroller 180 interfaces with HPLC controller 190

Microcontroller 180 sends trigger signals to the HPLC controller 190 to initiate injection of the reaction mixture into the HPLC system. The microcontroller also receives feedback on the HPLC run status and uses the radioactivity signal to determine the collection window for the desired product.

- describe microcontroller 180 interfaces with valve drivers and position sensors 194

Microcontroller 180 sends digital signals to valve drivers 194 to open and close solenoid valves controlling gas and vacuum flow. The microcontroller also receives binary inputs from position sensors 194 to confirm that valves have reached their commanded state. The system implements a fail-safe protocol that halts operation if a valve does not respond within a specified time.

- describe microcontroller 180 controls vacuum pump 196

Microcontroller 180 activates the vacuum pump 196 through a solid-state relay when a vacuum operation is initiated. The microcontroller monitors the vacuum level via a digital gauge and terminates the pump when the setpoint is reached. The system includes a delay to allow the cold trap to cool before initiating vacuum, preventing solvent ingress into the pump.

- describe automated synthesizer 10 performs radiosynthesis

The automated synthesizer 10 performs radiosynthesis by executing a sequence of unit operations defined in a software protocol. The system begins by loading a disposable cassette and initializing all subsystems. The reagent and gas handling robot retrieves reagent vials and delivers them to the reaction vessel. The reactor assemblies move to sealed positions to perform reactions under controlled temperature and pressure. Solvents are evaporated using vacuum and inert gas flow. The reaction mixture is transferred to a purification cartridge, and the desired product is eluted into a collection vessel. The final product is injected into an HPLC system for purification and analysis. The entire process is performed without manual intervention, ensuring consistency, safety, and compliance.

- describe radioisotope handling

Radioisotope handling is performed by connecting the [18F]fluoride source to the inlet port of the cassette. The system activates a QMA cartridge to trap the fluoride anion, and then elutes it into the reaction vessel using a solution of K222/K2CO3 in acetonitrile. The entire process is performed under sealed conditions to preserve specific activity. The system monitors the radioactivity level and adjusts the elution volume based on the input activity.

- describe reagent handling

Reagent handling is performed by the reagent and gas handling robot, which retrieves sealed reagent vials from storage and delivers them to the reaction vessel by pressurizing the vial with inert gas. The system ensures complete transfer by measuring the duration and pressure differential. The robot returns the empty vial to storage after delivery. The system accounts for dead volume by allowing for excess reagent loading in the vials.

- describe reagent handling (continued)

The system includes a calibration routine that measures the dead volume in each reagent vial and adjusts the delivery volume accordingly. The system also verifies the presence of the vial using a Hall-effect sensor before initiating transfer. If a vial is missing or improperly seated, the system halts and alerts the operator.

- describe reactions

Reactions are performed by sealing the reaction vessel against the gasket of the cassette and heating it to the required temperature. The system maintains the temperature for a predetermined duration while stirring the mixture. The system monitors the internal pressure and temperature in real time and adjusts the heating profile if deviations occur. The system ensures that the reaction is completed before proceeding to the next step.

- describe reactions (continued)

The system supports reactions at temperatures up to 185°C and pressures up to 150 psi. The sealed environment prevents exposure of tubing and valves to reactive media, allowing for the use of corrosive reagents such as triflic acid and anhydrous hydrogen fluoride. The system logs the reaction parameters and generates a report for quality assurance.

- describe evaporations

Evaporations are performed by sealing the reaction vessel against the evaporation zone of the cassette and applying vacuum while heating the vessel. Inert gas is introduced to assist in vapor removal. The system monitors the volume of solvent removed and terminates the evaporation when the target is reached. The system uses visual feedback from the camera to confirm the absence of liquid.

- describe transfer and purification

Transfer and purification are performed by pressurizing the reaction vessel and directing the flow through a purification cartridge. The system switches stopcock valves to trap the product on the cartridge and wash away impurities. The system then elutes the product into a fresh vessel using a suitable solvent. The entire process is automated and performed under sealed conditions.

- describe transfer and purification (continued)

The system supports multiple purification steps, including cartridge trapping, washing, and elution. The system logs the volume and flow rate of each step and verifies the completion of each phase before proceeding. The system also calculates the radiochemical yield based on the pre- and post-purification activity measurements.

- describe radiosynthesis materials

The radiosynthesis materials include [18F]fluoride produced in a cyclotron, precursors such as triflate esters and silylated nucleosides, solvents such as acetonitrile and ethanol, and purification cartridges such as QMA and C18. All materials are used as received and are stored in sealed vials on the cassette. The system is compatible with both carrier-added and no-carrier-added isotopes.

- describe synthesis protocol

The synthesis protocol is a sequence of unit operations defined in a software file and loaded into the system. The protocol includes parameters for temperature, duration, pressure, and flow rate for each step. The protocol is created using a drag-and-drop interface and can be saved, shared, and versioned. The system validates the protocol before execution.

- describe synthesis protocol (continued)

The system enforces protocol integrity by preventing incompatible sequences and requiring mandatory safety checks. The protocol includes conditional branches, such as repeating an evaporation if solvent remains, and allows for manual overrides with audit trails. The protocol is stored in a database and is accessible only to authorized users.

- describe semi-preparative HPLC

Semi-preparative HPLC is used to purify the final radiotracer product. The system injects the eluate from the purification cartridge into a reversed-phase column and separates the product from impurities using a defined mobile phase. The system collects the product fraction based on radioactivity detection and delivers it to a final vial. The system logs the retention time and radiochemical purity.

- describe analytical HPLC

Analytical HPLC is used to verify the identity and purity of the radiotracer. The system injects a small sample of the final product into a high-resolution column and compares the retention time and radioactivity profile to a reference standard. The system calculates the radiochemical purity and specific activity and generates a report for quality control.

- describe results and discussion

The automated radiosynthesizer has been validated for the synthesis of multiple PET tracers, including d-[18F]FAC and l-[18F]FMAU, with radiochemical yields exceeding 90% and specific activities greater than 150 GBq/μmol. The system has demonstrated reproducibility across multiple synthesis runs and has been used to produce tracers under clinical manufacturing conditions. The system has reduced synthesis time by 40% compared to manual methods and has eliminated operator radiation exposure.

- describe results and discussion (continued)

The system has enabled the production of previously inaccessible tracers that require high-pressure and high-temperature conditions. The disposable cassette design has eliminated cross-contamination and reduced preparation time. The system has been adopted in multiple research and clinical facilities and has received regulatory approval for use in human imaging studies.

- describe advantages of automated synthesizer 10

The advantages of automated synthesizer 10 include reduced radiation exposure, increased reproducibility, elimination of manual handling, rapid transition between tracers, and compliance with regulatory standards. The system reduces synthesis time, minimizes waste, and improves product quality. The system is scalable and can be deployed in both research and clinical settings.

- describe flexibility of automated synthesizer 10

The flexibility of automated synthesizer 10 lies in its ability to perform a wide range of radiochemical reactions without hardware modification. The system supports multiple reaction vessels, variable temperatures and pressures, and diverse reagent and purification schemes. The system can be reprogrammed to synthesize new tracers by simply loading a new protocol and cassette.

- describe flexibility of automated synthesizer 10 (continued)

The system supports the integration of new unit operations through software updates and allows for the addition of new reagent types and purification methods. The system is compatible with third-party cartridges and can be adapted for use with other radionuclides such as carbon-11, iodine-124, and copper-64.

- describe scope of invention

The scope of the invention encompasses the automated radiosynthesizer device, the disposable cassettes, the reagent and gas handling robot, the control system, and the methods of performing radiosynthesis using the device. The invention includes all variations of the system that perform the same functions in substantially the same way to achieve the same results.

- describe scope of invention (continued)

The invention includes all modifications to the system that incorporate equivalent components, such as different types of heaters, actuators, or sensors, as long as the overall function and structure are preserved. The invention also includes the use of the system for the synthesis of any radiopharmaceutical requiring sealed, high-pressure, or high-temperature reactions.

- describe modifications to embodiments

Modifications to embodiments include the addition of a second vacuum pump for faster evaporation, the integration of a mass spectrometer for real-time product identification, the use of alternative materials for the cassette such as polypropylene or PEEK, and the incorporation of artificial intelligence algorithms to optimize synthesis parameters.

- describe modifications to embodiments (continued)

Other modifications include the use of a robotic arm for automated vial labeling and the integration of a barcode scanner for cassette identification. The system may be expanded to include additional reactor assemblies, and the control system may be upgraded to support remote access and cloud-based protocol sharing.

- describe dimensions of drawings

The dimensions of the drawings are provided in millimeters and are to scale. The reactor assembly has a height of 250 mm, a width of 120 mm, and a depth of 100 mm. The cassette is 300 mm long, 150 mm wide, and 40 mm high. The reagent and gas handling robot has a travel range of 500 mm in the x-axis and 300 mm in the y-axis. All components are designed to fit within a standard hot cell enclosure.

- describe limitations of invention

The limitations of the invention include the requirement for pre-assembled cassettes, which may increase cost for low-volume applications. The system is not designed for the synthesis of gaseous tracers or for reactions requiring cryogenic temperatures. The system requires a stable power supply and a source of inert gas, which may not be available in all settings.

- describe claims of invention

1. An automated radiosynthesizer comprising: a plurality of reactor assemblies, each comprising a sealed reaction vessel, a heating element, a temperature sensor, and a vertical actuator; a reagent and gas handling robot comprising a vial gripper and a gas supplier; a disposable cassette comprising reagent storage positions, gas inlet ports, and a gasketed sealing surface; and a control system configured to coordinate the movement of the reactor assemblies and the robot to perform a sequence of unit operations for radiosynthesis.

2. The automated radiosynthesizer of claim 1, wherein the disposable cassette comprises a purification cartridge and a stopcock valve for selective transfer of reaction products.

3. The automated radiosynthesizer of claim 1, wherein the gas supplier comprises a vacuum port and an inert gas port, both spring-loaded to ensure sealing contact with the cassette.

4. The automated radiosynthesizer of claim 1, wherein the control system comprises a web-based interface for creating and executing synthesis protocols using drag-and-drop unit operations.

5. The automated radiosynthesizer of claim 1, wherein the reactor assemblies are capable of operating at temperatures up to 185°C and pressures up to 150 psi.

6. The automated radiosynthesizer of claim 1, wherein the vial gripper is configured to engage sealed 13-mm crimped vials and deliver reagents by pressurization.

7. The automated radiosynthesizer of claim 1, wherein the system includes a cold trap positioned between the vacuum port and a vacuum pump.

8. The automated radiosynthesizer of claim 1, wherein the system includes a semi-preparative HPLC system for purification of the radiotracer product.

9. A method of performing radiosynthesis comprising: loading a disposable cassette into a synthesizer; retrieving a reagent vial using a robotic gripper; delivering the reagent to a reaction vessel by pressurization; sealing the vessel against a gasketed surface; heating the vessel to a predetermined temperature; performing a reaction for a predetermined duration; transferring the reaction mixture to a purification cartridge; eluting the product into a collection vessel; and purifying the product using HPLC.

10. The method of claim 9, wherein the sealing of the vessel is achieved by raising the reactor assembly to contact the gasketed surface, thereby isolating the reaction from external fluidic components.

- describe equivalents of invention

Equivalents of the invention include any system that performs the same functions using substantially the same means to achieve the same results, including systems that substitute mechanical actuators with piezoelectric drives, replace the gasketed sealing interface with magnetic coupling, or use alternative materials for the cassette. The invention includes all modifications that are obvious to one skilled in the art without departing from the spirit and scope of the claims.