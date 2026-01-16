# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to a modular, computer-controlled radiosynthesizer for the automated synthesis of positron-emission tomography (PET) tracers. More specifically, the invention pertains to a three-reactor synthesizer that integrates a reagent and gas handling robot, disposable cassettes, and a control system to facilitate the production of complex PET tracers, such as 2-deoxy-2-[18F]fluoro-β-d-arabinofuranosylcytosine (d-[18F]FAC) and 2-deoxy-2-[18F]fluoro-5-methyl-β-l-arabinofuranosyluracil (l-[18F]FMAU), under high-pressure and high-temperature conditions.

## BACKGROUND

Positron-emission tomography (PET) has revolutionized the field of medical imaging, enabling non-invasive disease detection, cancer staging, and drug efficacy screening. Among the various PET tracers, 2-[18F]fluoro-2-deoxy-d-glucose ([18F]FDG) stands out due to its ease of production, manageable half-life, and widespread application. The increasing demand for [18F]FDG has driven the development of automated radiosynthesizers, which have significantly reduced costs, enabled production at multiple sites, and minimized radiation exposure to radiochemists.

However, many 18F-labeled PET tracers, particularly those used for imaging cell proliferation and reporter gene expression, require high-pressure, high-temperature reactions and the use of volatile or corrosive reagents. These requirements pose significant challenges for existing automated synthesizers, often necessitating modifications to the chemistry to fit within the operational constraints of the equipment. For instance, nucleoside analogs like d-[18F]FAC and l-[18F]FMAU, which are crucial for imaging cell proliferation and assessing chemotherapy drug efficacy, often require high-temperature reactions in volatile solvents.

To address these limitations, a modular and computer-controlled platform was developed, featuring movable components that seal the reaction vessel against an inert stopper during reactions. This platform was further refined into a fully automated synthesizer, the ELIXYS, which integrates three reactors, a reagent and gas handling robot, and disposable cassettes. The ELIXYS synthesizer is designed to handle a wide range of synthesis protocols, including those involving high-pressure and high-temperature reactions, while ensuring user-friendliness and reliability.

## SUMMARY

The present invention provides a modular, computer-controlled radiosynthesizer for the automated synthesis of PET tracers. The synthesizer, referred to as the ELIXYS, comprises three key components: a set of three reactors, a reagent and gas handling robot, and disposable cassettes. The reactors are designed to move to various positions beneath the cassettes, sealing the reaction vessel against a gasket to enable high-pressure and high-temperature reactions. The reagent and gas handling robot automates the delivery of reagents and gases, while the disposable cassettes store reagents and provide the primary fluid path, facilitating rapid setup and transition from tracer development to routine production.

The ELIXYS synthesizer is capable of performing a wide range of chemistry unit operations, including radioisotope handling, reagent addition, reactions, evaporations, and transfers. The system is designed to be user-friendly, with a drag-and-drop software interface that allows users to customize synthesis protocols. The synthesizer has been validated through the successful synthesis of d-[18F]FAC and l-[18F]FMAU, demonstrating comparable yields and synthesis times to other reported methods. Additionally, the ELIXYS has been used to synthesize several other PET tracers, including 2-deoxy-2-[18F]fluoro-5-ethyl-β-d-arabinofuranosyluracil (d-[18F]FEAU), [18F]FDG, [18F]FLT, [18F]FHBG, and [18F]SFB, without the need for hardware or plumbing changes.

## DETAILED DESCRIPTION OF ILLUSTRATED EMBODIMENTS

The ELIXYS radiosynthesizer is a modular, computer-controlled platform designed for the automated synthesis of PET tracers. The system consists of three main components: a set of three reactors, a reagent and gas handling robot, and disposable cassettes. Each component plays a critical role in the synthesis process, enabling the production of complex tracers under high-pressure and high-temperature conditions.

### Reactors

The ELIXYS synthesizer includes three reactors, each capable of holding a 5-mL glass V-vial. The reactors are designed to move to various positions beneath the cassettes, sealing the reaction vessel against a gasket to enable high-pressure and high-temperature reactions. Each reactor is equipped with three 100-W cartridge heaters and K-type thermocouples for precise temperature control, with a maximum operating temperature of 185°C. The reactors are actively cooled using a propylene/ethylene glycol and water mixture pumped through cooling channels. The reaction vessel is sealed against a gasket on the cassette, allowing for dynamic configuration of the fluid path for different unit operations.

### Reagent and Gas Handling Robot

The reagent and gas handling robot is a pneumatically actuated system that includes a vial gripper and a gas supplier. The vial gripper moves reagent vials between storage and addition positions, while the gas supplier provides inert gas and vacuum to the cassettes. The robot is equipped with Hall effect sensors to detect the position of the vial gripper and gas supplier, ensuring reliable operation. The gas supplier is mounted on a z-axis actuator and seals to the gas inlet gaskets on the cassettes, forming a gas-tight seal. The system is designed to minimize the number of connections and seals, enhancing reliability.

### Disposable Cassettes

Disposable cassettes are designed to contain all disposable components and fluid paths, eliminating the need for cleaning or customization between syntheses. Each cassette includes 11 reagent vial storage positions, stainless steel needles, tubing, and a PTFE-coated silicone gasket. The cassettes are preassembled and slide into the ELIXYS system, ensuring accurate positioning. The fluid paths are configured to perform various unit operations, including radioisotope handling, reagent addition, reactions, evaporations, and transfers. The cassettes also include stopcock valves for cartridge trap and release, purification, and waste collection.

### Control System

The ELIXYS synthesizer is controlled by a Linux server, which communicates with a programmable logic controller (PLC) over Ethernet. The PLC drives various subsystems, including linear actuators, pneumatics, cooling, heating, stirring, and HPLC injection. The system includes motor controllers for the reagent and gas handling robot and the reactors, as well as analog pressure regulators for inert gas and vacuum. The control system also houses solid-state relays for reactor temperature control, a cooling system, a video server for reactor cameras, and an electronically controlled HPLC injection valve.

### System Operations

The ELIXYS performs automated syntheses by completing a sequence of chemistry unit operations. These operations include radioisotope handling, reagent addition, reactions, evaporations, and transfers. Each operation involves the interaction of the ELIXYS subsystems and disposable cassettes to carry out the desired chemical steps.

#### Radioisotope Handling

For radiochemistry with [18F]fluoride, a preconditioned quaternary methylammonium (QMA) cartridge is installed between two tubes coming from the cassette. The [18F]fluoride source solution flows through the cartridge, and [18O]H2O is collected in a recovery vial. The reagent and gas handling robot drives the eluent through the cartridge and into the reaction vessel. Multiple elutions or rinses can be performed to increase the efficiency of [18F]fluoride collection.

#### Reagent Handling

To add a reagent, the vial gripper moves to the reagent storage position, grasps the vial, and moves it to the designated reagent addition location. The gas supplier lowers, the inert gas valve opens, and the vial is placed on a pair of needles, pressurizing the vial and transferring its contents. After addition, the vial gripper lifts the empty reagent vial, the gas supplier disengages, and the vial is returned to its original storage position.

#### Reactions

To maintain high internal pressure during superheated reactions, the reaction vessel is sealed against the gasket on the cassette. The reactor is heated to the desired temperature, with optional stirring. Once the desired temperature is reached, heating and stirring continue for the desired reaction time. After the reaction, the heaters are turned off, and the cooling pump is activated to lower the temperature.

#### Evaporations

Evaporation of solvents occurs by sealing the reaction vessel against the gasket at the evaporate position. The vessel is heated with the option of stirring, and the gas supplier provides both vacuum and inert gas to remove vapor. The required time for evaporation is determined by measuring the maximum time needed for complete evaporation and multiplying by a safety factor.

#### Transfer and Purification

Sep-Pak purification cartridges are connected to designated Luer fittings on the cassette. A dip tube acts as the fluid path for the transfer of crude products. The transfer unit operation begins with the reaction vessel sealing against the transfer position. The gas supplier provides inert gas to pressurize the reaction vessel, moving the fluid through the dip tube and to the Sep-Pak. The stopcock position is switched to direct the fluid to a waste collection vial or to the input of the next cassette. The desired product is eluted from the Sep-Pak by adding the elution solvent to the first reaction vessel and repeating the transfer unit operation.

### Radiosynthesis

The ELIXYS synthesizer has been validated through the successful synthesis of d-[18F]FAC and l-[18F]FMAU. The synthesis protocols for these tracers involve a series of unit operations, including radioisotope handling, reagent addition, reactions, evaporations, and transfers. The crude products are purified by semi-preparative HPLC, and the desired products are collected and verified by analytical HPLC. The yields and synthesis times for d-[18F]FAC and l-[18F]FMAU are comparable to other reported methods, demonstrating the effectiveness of the ELIXYS synthesizer.

### In Vivo Imaging

The ELIXYS-produced d-[18F]FAC was used for in vivo imaging in conscious C57BL/6 mice. The images obtained were comparable to those produced using a manually operated apparatus, with similar uptake in the gastrointestinal tract and hematopoietic organs. The ELIXYS synthesizer has also been used to synthesize other PET tracers, including d-[18F]FEAU, [18F]FDG, [18F]FLT, [18F]FHBG, and [18F]SFB, without the need for hardware or plumbing changes.

### Conclusion

The ELIXYS synthesizer is a versatile and reliable platform for the automated synthesis of PET tracers. Its modular design, computer-controlled operations, and use of disposable cassettes enable the production of complex tracers under high-pressure and high-temperature conditions. The system has been validated through the successful synthesis of d-[18F]FAC and l-[18F]FMAU, and it has the potential to significantly enhance the production of a wide range of PET tracers for both preclinical and clinical applications.