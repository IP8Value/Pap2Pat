# DESCRIPTION

## FIELD

- relate to monitoring ionizing radiation systems

The present invention relates to systems and methods for the real-time, non-invasive monitoring of nuclear reactor operations through the detection of escaping neutrons at stand-off distances from the reactor core. Specifically, the invention provides a novel approach to nuclear safeguards by leveraging the correlation between neutron flux measured exterior to reactor shielding and the isotopic composition of fissile material within the reactor core. This system enables the detection of anomalies in nuclear fuel cycle activities—including unauthorized fuel diversion, undeclared refueling, or alterations in irradiation schedules—without requiring physical access to the reactor core or direct sampling of nuclear materials. The method is applicable to a broad range of nuclear reactor types, including research reactors, small modular reactors, pressurized heavy water reactors, and other thermal neutron systems where fissile isotopes undergo transmutation over time. The invention is particularly suited for deployment by international nuclear regulatory bodies seeking to enhance verification capabilities under existing safeguards frameworks, especially in facilities where traditional inspection protocols are limited by design constraints, operational secrecy, or logistical inaccessibility.

## BACKGROUND

- introduce nuclear reactor safeguards

Nuclear reactor safeguards are international measures designed to verify that nuclear materials and facilities are used exclusively for peaceful purposes and are not diverted for weapons development or other illicit activities. These safeguards are implemented by organizations such as the International Atomic Energy Agency (IAEA) and rely on a combination of physical inspections, tamper-indicating seals, video surveillance, and accounting of nuclear material inventories. The goal is to detect any unauthorized removal, alteration, or misuse of fissile materials such as uranium-235, plutonium-239, or other isotopes capable of sustaining a fission chain reaction. While these methods have proven effective in many contexts, their timeliness and sensitivity are often constrained by the infrequency of on-site inspections and the inability to monitor continuous changes in core composition during reactor operation.

- describe existing safeguards systems

Existing safeguards systems at nuclear facilities typically employ passive monitoring tools such as surveillance cameras, sealed containers with tamper-proof indicators, and periodic physical inventories of fresh and spent fuel assemblies. These systems are supplemented by operator-reported operational logs, reactor power histories, and fuel burn-up calculations derived from reactor physics simulations. However, such approaches are inherently reactive and cannot detect real-time deviations in core isotopic composition. For example, if fissile material is removed during a scheduled refueling cycle or if fuel is irradiated for a shorter duration than declared to reduce plutonium production, these changes may remain undetected until the next scheduled inspection, which may occur weeks or months later.

- discuss limitations of existing systems

The limitations of current safeguards systems are particularly acute in facilities with sealed-core designs, such as small modular reactors, or in research reactors that undergo frequent online refueling. In these cases, direct access to the core is restricted, and physical sampling of fuel is either impossible or impractical. Furthermore, operator declarations of fuel composition and operational parameters may be incomplete, inaccurate, or intentionally misleading. The reliance on periodic inspections also creates temporal gaps in monitoring, during which illicit activities could occur without detection. These shortcomings undermine the ability of safeguards regimes to meet internationally recognized timeliness goals for the detection of significant quantities of fissile material, particularly in environments where the risk of clandestine diversion is elevated.

- introduce antineutrino detection

In recent years, antineutrino detection has been proposed as a complementary tool for reactor monitoring, based on the principle that antineutrinos are emitted in proportion to the fission rate within the reactor core. While antineutrino detectors can provide continuous, non-intrusive measurements of reactor power and, to some extent, isotopic composition, they require large, complex, and expensive instrumentation, often necessitating underground placement to mitigate cosmic ray backgrounds. Their spatial resolution is limited, and they are highly sensitive to detector calibration, distance from the core, and shielding geometry, making them impractical for widespread deployment in diverse reactor environments.

- discuss prior art

Prior art in the field of reactor monitoring has focused primarily on internal neutron detectors used for reactor control and safety systems, such as fission chambers and ionization chambers placed within the reactor vessel or reflector. These instruments provide high-resolution data but are inaccessible for external verification and cannot be utilized for safeguards purposes without physical access to the reactor. Other approaches have explored the use of gamma-ray spectroscopy or neutron activation analysis for fuel verification, but these methods require proximity to fuel assemblies or the introduction of external radiation sources, which are incompatible with non-invasive safeguards objectives. No prior system has demonstrated the capability to detect and quantify changes in fissile isotope inventory through the measurement of escaping neutrons at stand-off distances exceeding 50 meters, nor has any prior system established a quantitative, time-resolved correlation between neutron flux per unit reactor power and the weighted isotopic composition of the core.

## SUMMARY

- introduce system for monitoring nuclear reactor

The present invention introduces a novel system for monitoring nuclear reactor operations through the detection of neutrons escaping from the reactor core at stand-off distances of up to 100 meters. The system enables the continuous, non-intrusive verification of reactor fuel cycle activities by correlating measured neutron flux with reactor power output and historical isotopic inventory data to detect deviations indicative of unauthorized fuel manipulation.

- describe neutron detector configuration

The system comprises an array of large-area neutron detectors positioned at multiple locations exterior to the reactor’s primary radiation shielding. Each detector is configured to detect thermalized neutrons originating from fission reactions within the reactor core, with sufficient sensitivity to resolve variations in neutron emission rates over time. The detectors are arranged in a spatially distributed configuration to enable cross-validation of signals and to distinguish between genuine core-related phenomena and localized environmental interferences.

- describe controller functionality

A central system controller receives real-time input signals from each neutron detector and from the reactor’s operational monitoring systems, including reactor power output, coolant flow rates, and temperature profiles. The controller processes these inputs to normalize neutron detection rates with respect to reactor power, thereby isolating variations attributable to changes in fissile isotope composition rather than fluctuations in reactor output.

- describe aberrant change detection

The controller is programmed to detect aberrant changes in the neutron flux per unit reactor power by comparing real-time measurements against a baseline established during normal reactor operation. Deviations exceeding statistically defined thresholds trigger alerts, prompting further investigation by safeguards authorities. The system is capable of identifying anomalies as small as a 5% change in the weighted isotopic composition of the core, corresponding to the movement or addition of approximately one kilogram of fissile material.

- introduce second neutron detector

The system incorporates multiple neutron detectors, each located at a distinct stand-off position relative to the reactor core. The use of a second, and optionally additional, neutron detector enables the discrimination of localized interferences—such as the temporary shielding of neutrons by refueling equipment—from genuine changes in core composition. Correlation between signals from multiple detectors allows for the identification of spatially consistent anomalies that reflect true core behavior.

- describe isotopic concentration determination

The system determines isotopic concentration changes by applying a mathematical model that relates the neutron flux per unit reactor power to the mass inventory of key fissile isotopes, including uranium-235, plutonium-239, and plutonium-241. The model accounts for the distinct fission cross-sections and energy released per fission for each isotope, enabling the system to infer relative changes in isotopic composition without requiring prior knowledge of absolute fuel masses.

- describe alert generation

When a statistically significant deviation is detected, the system generates an automated alert that includes the magnitude, timing, and location of the anomaly, along with a confidence assessment based on statistical analysis of the detector data. Alerts are transmitted to authorized safeguards personnel via secure communication channels and may be integrated into existing nuclear facility monitoring networks.

- introduce expected neutron flux

The system calculates an expected neutron flux per unit reactor power based on a simulated model of the reactor’s fuel cycle, incorporating known operational parameters, fuel loading history, and predicted burn-up patterns. This expected flux serves as a reference against which measured neutron flux values are compared.

- describe simulated and baseline neutron flux

The baseline neutron flux per unit reactor power is established during a period of normal, declared reactor operation and is updated iteratively as new operational data become available. Simulated neutron flux values are derived from neutron transport codes that model the reactor’s geometry, fuel composition, and shielding characteristics, and are used to validate the accuracy of the baseline and to predict expected variations over time.

- describe neutron detector types

The neutron detectors employed in the system are thermal neutron detectors, including boron-coated straw detectors and boron-10 lined stainless steel tubes filled with helium-3 and noble gas mixtures. These detectors are optimized for high signal-to-background ratios and are capable of operating reliably in outdoor or industrial environments with minimal environmental shielding.

- describe moderator configuration

Each neutron detector is surrounded by a hydrogenous moderator, such as high-density polyethylene, to thermalize fast neutrons emitted from the reactor core. The moderator thickness is selected to maximize detection efficiency for neutrons in the epithermal energy range, which predominate in the leakage spectrum from shielded reactor cores.

- introduce method of monitoring reactor

The method of monitoring the reactor involves continuously measuring neutron flux at multiple stand-off locations, normalizing the flux with respect to reactor power, comparing the normalized flux to a baseline or simulated expectation, and generating alerts upon detection of statistically significant deviations. The method does not require physical access to the reactor core and is compatible with existing reactor instrumentation and safeguards protocols.

## DETAILED DESCRIPTION

- introduce reactor safeguards regimes

Reactor safeguards regimes are established under international agreements to ensure that nuclear materials are not diverted from peaceful applications to weapons programs. These regimes depend on the timely detection of anomalies in nuclear material flows, reactor operations, and fuel cycle activities. The present invention enhances the effectiveness of these regimes by introducing a continuous, real-time monitoring capability that complements traditional inspection methods and reduces the temporal window during which illicit activities can occur undetected.

- describe illicit use of reactor facilities

Illicit use of reactor facilities may include the intentional reduction of fuel irradiation time to minimize plutonium-240 content, the unauthorized removal of fissile material during refueling, or the substitution of low-enriched uranium with high-enriched material to increase weapons-grade plutonium production. Such activities are difficult to detect using conventional safeguards because they do not necessarily alter reactor power output or trigger alarms in internal safety systems.

- summarize current safeguard monitoring systems

Current safeguard monitoring systems rely on periodic inspections, sealed containers, video surveillance, and operator declarations. These systems are vulnerable to temporal gaps, human error, and deliberate concealment. They provide no continuous measurement of core isotopic composition and are unable to detect subtle changes in fuel burn-up patterns or fuel composition that may indicate diversion.

- motivate need for real-time monitoring

The need for real-time monitoring arises from the increasing deployment of advanced reactor designs, such as small modular reactors and molten salt reactors, which feature sealed cores, online refueling, and limited access for physical inspection. Without continuous, non-intrusive monitoring, these facilities present significant challenges to international safeguards, as deviations in fuel composition may occur without leaving a trace detectable by traditional means.

- introduce anti-neutrino detectors

Anti-neutrino detectors have been proposed as a means of monitoring reactor operations from a distance, as they are emitted in proportion to the fission rate and can be detected outside containment. However, these detectors require large volumes of target material, deep underground placement to reduce cosmic background, and sophisticated data analysis to resolve isotopic changes. Their size, cost, and sensitivity to environmental conditions limit their practicality for widespread deployment.

- describe limitations of anti-neutrino detectors

Anti-neutrino detectors are highly sensitive to distance from the core, shielding geometry, and detector calibration. They cannot resolve spatial variations in core composition, are affected by reactor shutdowns and restarts, and require months of data collection to achieve sufficient statistical precision for isotopic analysis. Moreover, they are incapable of detecting localized events such as refueling maneuvers or temporary shielding of neutron leakage.

- introduce neutron detectors as alternative

Neutron detectors offer a practical alternative to anti-neutrino detectors by providing high signal-to-noise ratios at stand-off distances, compatibility with existing industrial environments, and the ability to detect both power fluctuations and isotopic changes. Unlike anti-neutrinos, neutrons are emitted in much greater numbers and can be moderated and detected using compact, robust instrumentation.

- describe positioning of neutron detectors

Neutron detectors are positioned at multiple locations outside the reactor’s primary shielding, typically at distances ranging from 10 to 100 meters from the core. Detectors are placed at varying azimuthal angles and elevations to capture the spatial distribution of neutron leakage and to enable triangulation of signal anomalies.

- define standoff distance

Standoff distance is defined as the minimum linear distance between the outer surface of the reactor’s primary radiation shield and the nearest point of the neutron detector. In this invention, standoff distances of 17 meters or greater are employed to ensure that detectors are located outside the reactor building or containment structure, thereby enabling non-intrusive, unobtrusive monitoring.

- describe potential configurations of neutron detectors

Neutron detectors may be mounted on fixed structures, mobile trailers, or embedded in permanent monitoring stations. They may be arranged in linear arrays, circular perimeters, or three-dimensional grids around the reactor. Each configuration is selected based on facility layout, shielding characteristics, and the desired spatial resolution of anomaly detection.

- motivate use of neutron detectors

The use of neutron detectors is motivated by their ability to detect changes in fissile isotope inventory through the neutron flux per unit reactor power, a relationship that is both physically robust and quantitatively predictable. Unlike other methods, neutron detection provides direct, real-time insight into the core’s isotopic composition without requiring assumptions about fuel burn-up models or operator declarations.

- describe unknowns prior to developing new monitoring system

Prior to this invention, it was unknown whether neutron leakage at stand-off distances could be reliably correlated with isotopic changes in the reactor core, whether environmental and operational variables could be sufficiently controlled to enable meaningful detection, or whether compact, field-deployable neutron detectors could achieve the necessary sensitivity and stability for safeguards applications.

- summarize discovery of neutron detection at standoff distances

The invention demonstrates, for the first time, that the neutron flux per unit reactor power measured at stand-off distances is linearly proportional to the weighted sum of fissile isotope masses in the core. This discovery enables the detection of kilogram-scale changes in fissile material inventory within weeks of occurrence, meeting and exceeding international safeguards timeliness goals.

- describe advantages of neutron detectors over anti-neutrino detectors

Neutron detectors are significantly smaller, less expensive, and more robust than anti-neutrino detectors. They require no underground placement, operate effectively in ambient environments, and provide higher temporal resolution. They are also capable of detecting localized events such as refueling maneuvers, which are invisible to anti-neutrino detectors.

- summarize isotopic properties of fuel

The fissile isotopes uranium-235, plutonium-239, and plutonium-241 exhibit distinct fission cross-sections and energy yields per fission, resulting in different contributions to the neutron flux per unit reactor power. These differences form the physical basis for detecting changes in isotopic composition, as the neutron emission rate per unit power varies predictably with the relative abundance of each isotope.

- describe how neutron detection can facilitate detection of changes in isotopic composition

By measuring the neutron flux per unit reactor power over time and comparing it to a baseline established during normal operation, the system detects deviations that correspond to changes in the relative mass of fissile isotopes. For example, a decrease in uranium-235 mass or an increase in plutonium-239 mass alters the neutron emission rate per unit power in a quantifiable manner, enabling the system to infer the nature and magnitude of the change.

- describe how readings from multiple detectors can be coordinated

Readings from multiple detectors are analyzed in concert to distinguish between global core changes and localized interferences. A consistent deviation across multiple detectors indicates a genuine change in core composition, while a deviation observed at only one location suggests environmental interference, such as the temporary blocking of neutron leakage by refueling equipment.

- describe advantages of smaller neutron detectors

Smaller neutron detectors, each weighing approximately 10 kilograms, are advantageous because they are easily transportable, scalable, and can be deployed in large numbers without prohibitive cost or logistical burden. Their compact size allows for flexible placement around reactor facilities and facilitates rapid reconfiguration in response to changing operational conditions.

- compare event rates of neutron and anti-neutrino detectors

Neutron detectors achieve event rates on the order of tens to hundreds of counts per minute under normal reactor operation, whereas anti-neutrino detectors typically record fewer than one event per hour. This higher event rate enables neutron detectors to achieve statistical significance in days rather than months, greatly improving the timeliness of anomaly detection.

- introduce equation for neutron detection

The number of neutrons detected is proportional to the population of neutrons in the reactor core, which in turn is proportional to the average neutron flux, the average neutron velocity, and the core volume. This relationship is expressed as \( n_{\det} \propto \frac{\langle \phi \rangle}{\langle v \rangle} V \), where \( \langle \phi \rangle \) is the average neutron flux, \( \langle v \rangle \) is the average neutron velocity, and \( V \) is the core volume.

- introduce equation for reactor power

The reactor thermal power is determined by the sum of fission rates across all fissile isotopes, each weighted by their respective energy yield and fission cross-section. This is expressed as \( P_{tot} = V \langle \phi \rangle \sum_i N_i \langle \sigma_{f,i} \rangle E_{f,i} \), where \( N_i \) is the number density of the ith fissile isotope, \( \langle \sigma_{f,i} \rangle \) is its average fission cross-section, and \( E_{f,i} \) is the energy released per fission.

- describe monitoring system 100

Monitoring system 100 comprises a nuclear reactor, an array of neutron detectors positioned at stand-off distances, a data acquisition system, and a central controller. The system operates continuously, collecting and analyzing neutron flux data in real time to detect deviations from expected behavior.

- describe nuclear reactor 102

Nuclear reactor 102 is a thermal neutron reactor with a core containing fissile material, including uranium-235 and plutonium isotopes, and is surrounded by a radiation shield designed to attenuate neutron and gamma emissions.

- describe reactor core 104 and fuel bundles 106

Reactor core 104 contains fuel bundles 106 composed of fissile material arranged in a lattice structure. The fuel bundles are periodically replaced or repositioned during online refueling, altering the isotopic composition of the core over time.

- describe radiation shield 110

Radiation shield 110 surrounds the reactor core and is composed of layers of water, steel, and high-density concrete to attenuate neutron and gamma radiation. Neutrons that escape through gaps or imperfections in the shield are detected by the external neutron detectors.

- describe escaping neutrons 108a

Escaping neutrons 108a are fast neutrons emitted from the reactor core that are partially moderated by the surrounding materials and detected as thermal neutrons by the external neutron detectors. The rate of escaping neutrons is proportional to the neutron population in the core and is sensitive to changes in fissile isotope inventory.

- describe correlation between escaping neutrons and reactor core traits

There exists a direct, quantifiable correlation between the rate of escaping neutrons and the weighted sum of fissile isotope masses in the reactor core. Changes in the relative abundance of uranium-235, plutonium-239, or plutonium-241 result in predictable changes in the neutron flux per unit reactor power, enabling the detection of fuel manipulation.

- describe neutron detectors 120

Neutron detectors 120 are large-area thermal neutron detectors, including boron-coated straw detectors and boron-10 lined stainless steel tubes, each configured to detect neutrons with high efficiency and low background noise.

- describe stand-off perimeter 122

Stand-off perimeter 122 is the region surrounding the reactor where neutron detectors are positioned at distances of 17 meters or greater from the primary radiation shield. This perimeter is established to ensure that detectors are located outside the reactor building and are not subject to direct physical access or tampering.

- describe advantages of multiple radiation detectors

Multiple radiation detectors enable the system to distinguish between true core anomalies and localized interferences. By comparing signals across multiple locations, the system identifies spatially consistent deviations that reflect genuine changes in core composition.

- describe detector array

The detector array consists of two or more neutron detectors arranged in a spatially distributed pattern around the reactor. Each detector is independently calibrated and time-stamped, and their outputs are synchronized to enable cross-correlation analysis.

- describe monitoring neutron flux

Neutron flux is monitored continuously by each detector, with data recorded at intervals of one minute or less. The raw count rate is corrected for background radiation and normalized by the reactor’s thermal power output.

- describe monitoring changes in neutron flux per unit reactor power

Changes in neutron flux per unit reactor power are tracked over time to identify deviations from the established baseline. These deviations are analyzed using statistical methods to determine whether they are consistent with known operational events or indicative of unauthorized activity.

- describe system 100

System 100 integrates the reactor, neutron detectors, data acquisition hardware, and central controller into a unified monitoring platform. The system operates autonomously, generating alerts and maintaining audit logs for safeguards verification.

- introduce neutron flux per unit reactor power baseline

The neutron flux per unit reactor power baseline is established during a period of normal, declared reactor operation and is updated iteratively as new data are collected. The baseline incorporates historical reactor power profiles, fuel loading records, and simulated neutron transport models.

- detect changes in neutron flux per unit reactor power

Changes in neutron flux per unit reactor power are detected by comparing real-time measurements to the baseline using statistical hypothesis testing. Deviations exceeding a predefined confidence threshold trigger an alert.

- alert system users to changes

When a significant change is detected, the system generates an alert that includes the magnitude of the deviation, the time of occurrence, the affected detector locations, and a confidence level. Alerts are transmitted to authorized safeguards personnel via encrypted communication channels.

- investigate changes in neutron flux per unit reactor power

Authorized personnel may initiate an investigation by reviewing historical data, comparing detector signals across locations, and correlating anomalies with known operational events such as refueling or maintenance.

- describe radiation detectors

Radiation detectors in the system are specifically configured to detect thermal neutrons and are shielded from gamma radiation and other background sources. Each detector is calibrated using known neutron sources to ensure accuracy and stability.

- locate radiation detectors outside stand-off perimeter

Radiation detectors are located outside the stand-off perimeter to ensure non-intrusive monitoring and to prevent physical tampering. Their placement is determined by shielding geometry, neutron leakage patterns, and environmental conditions.

- describe neutron detectors

Neutron detectors are designed to detect neutrons originating from nuclear fission reactions within the reactor core. They generate output signals proportional to the neutron flux incident upon their sensitive volume and transmit these signals to the system controller.

- generate output signals based on neutron flux detected

Each neutron detector generates an electrical pulse for each detected neutron, which is amplified, shaped, and digitized to produce a time-stamped count rate. These output signals are transmitted to the system controller for analysis.

- transmit output signals to system controller

Output signals from each neutron detector are transmitted via wired or wireless communication links to the system controller, which aggregates and processes the data in real time.

- describe system controller

The system controller receives input signals from the neutron detectors and from the reactor’s operational sensors. It performs normalization, baseline comparison, statistical analysis, and alert generation. The controller also stores historical data and maintains an audit trail for safeguards verification.

- receive input signals from reactor and sensors

The system controller receives real-time input signals from the reactor’s power monitoring system, coolant flow sensors, temperature probes, and other operational parameters. These inputs are used to normalize neutron flux measurements.

- monitor power output of reactor

The system controller continuously monitors the reactor’s thermal power output to ensure that neutron flux measurements are properly normalized and that deviations are not attributable to changes in reactor power.

- store information about reactor

The system controller stores information regarding reactor design, fuel loading history, operational logs, and simulated neutron flux profiles. This information is used to refine the baseline and improve the accuracy of anomaly detection.

- determine if reactor is operating within reported parameters

The system controller compares measured neutron flux per unit power with expected values derived from reactor design and declared operational parameters. Discrepancies beyond statistical tolerance trigger an investigation.

- calibrate system for given reactor

The system is calibrated for each reactor by establishing a baseline neutron flux per unit power during a period of known, declared operation. Calibration accounts for reactor geometry, shielding, and detector placement.

- model expected changes to fuel composition over time

The system employs neutron transport simulations to model the expected evolution of fissile isotope inventory over time, based on declared fuel loading, burn-up rates, and refueling schedules.

- measure baseline of neutron flux per unit reactor power

The baseline is measured during a period of stable, declared operation and is updated periodically as new operational data are collected. The baseline incorporates both empirical measurements and simulated predictions.

- detect deviations from baseline

Deviations from the baseline are detected using statistical methods such as control charts, moving averages, or Bayesian inference. Deviations exceeding predefined thresholds are flagged for review.

- operate system as compliance and audit tool

The system serves as an independent compliance and audit tool, providing verifiable data that can be used to confirm or challenge operator declarations. Its operation is transparent, secure, and tamper-resistant.

- determine expected neutron flux per unit reactor power

Expected neutron flux per unit reactor power is determined using a mathematical model based on the known isotopic composition of the fuel and the physical properties of fission cross-sections and energy yields.

- compare measured neutron flux to expected neutron flux

Measured neutron flux is compared to the expected flux using statistical tests to determine whether the difference is significant. The comparison is performed continuously and in real time.

- generate output based on comparison

The system generates output in the form of alerts, trend plots, and confidence metrics based on the comparison between measured and expected neutron flux. Output is formatted for human review and automated integration into safeguards databases.

- alert system users to differences in neutron flux

When a statistically significant difference is detected, the system alerts authorized users via secure communication channels, providing details on the nature, timing, and location of the anomaly.

- describe neutron detectors used in system

The neutron detectors used in the system are boron-coated straw detectors and boron-10 lined stainless steel tubes filled with helium-3 and noble gas mixtures. These detectors are chosen for their high thermal neutron sensitivity, low background response, and durability in industrial environments.

- achieve detector signal-to-background ratio

The system achieves a signal-to-background ratio greater than 10:1 under normal reactor operation, ensuring that neutron signals from the reactor are clearly distinguishable from cosmic and environmental background radiation.

- describe moderator used to slow neutrons

A hydrogenous moderator, such as high-density polyethylene, is placed around each neutron detector to thermalize fast neutrons emitted from the reactor core. The moderator thickness is optimized to maximize detection efficiency for neutrons in the epithermal energy range.

- convert fast neutrons to thermal neutrons

Fast neutrons emitted from the reactor core are slowed to thermal energies through elastic scattering with hydrogen atoms in the moderator. Thermal neutrons are then captured by boron-10 nuclei in the detector, producing detectable ionization events.

- describe alternative radiation detectors

Alternative radiation detectors, such as lithium-6 coated scintillators or helium-3 proportional counters, may be used in place of boron-based detectors, provided they meet the required sensitivity, stability, and environmental tolerance.

- describe Boron Coated Straw detector

The boron-coated straw detector consists of multiple sealed aluminum tubes, each lined with a thin layer of boron-10 enriched boron carbide and filled with argon-carbon dioxide gas. Neutron capture by boron-10 produces alpha particles and lithium ions, which ionize the gas and generate detectable electrical pulses.

- describe B10+ stainless steel tubes detector

The B10+ stainless steel tube detector consists of sealed stainless steel tubes lined with boron-10 enriched coating and filled with helium-3 and noble gases. Neutron capture produces charged particles that generate ionization signals, which are amplified and time-stamped.

- describe detection of thermal neutrons

Thermal neutrons are detected through neutron capture reactions with boron-10 or helium-3, which produce charged particles that ionize the detector gas. The resulting electrical pulses are counted and time-stamped to produce a neutron flux rate.

- describe NRU reactor components

The National Research Universal (NRU) reactor is a heavy water moderated and cooled research reactor with online refueling capability. Its core contains low-enriched uranium fuel and is surrounded by layers of water, steel, and concrete shielding.

- detail BCS detector setup

The boron-coated straw (BCS) detector consists of 49 sealed straws arranged in a 7×7 matrix, each 1 meter in length and 7.5 mm in diameter. The straws are biased at +1000 volts and connected to a summing amplifier and digital acquisition system.

- explain signal processing components

Signal processing components include charge-sensitive amplifiers, pulse-shaping amplifiers, single-channel analyzers, and time-stamping digital controllers. These components convert raw detector pulses into calibrated, time-resolved count rates.

- describe data acquisition system

The data acquisition system collects, timestamps, and stores neutron count data from each detector at intervals of one minute or less. Data are transmitted to the system controller via secure digital links.

- detail experimental setup

The experimental setup included two neutron detectors placed at 17 meters and 69 meters from the NRU reactor core, with high-density polyethylene moderators surrounding each detector. Reactor power data were collected simultaneously from the facility’s operational sensors.

- describe detector placement locations

Detectors were placed at two locations: one within the reactor building, two levels below the core, and one outside the building in a trailer at ground level. The locations were chosen to represent different shielding environments and neutron leakage paths.

- explain data collection procedure

Data were collected continuously over a period of months, with neutron count rates and reactor power recorded at one-minute intervals. Background counts were measured during reactor shutdown periods.

- compare detector count rate with reactor power

Detector count rates were normalized by reactor power to eliminate variations due to changes in reactor output. The resulting neutron flux per unit power was analyzed for temporal trends.

- show correlation between detector count rate and reactor power

A strong linear correlation was observed between detector count rate and reactor power, with correlation coefficients exceeding 0.98. This confirmed that neutron leakage is directly proportional to reactor power.

- describe signal to background ratio

The signal-to-background ratio was greater than 10:1 at both detector locations, demonstrating that reactor-generated neutrons were clearly distinguishable from cosmic and environmental background.

- explain difference in signal between locations A and B

The signal at Location B was 7.5 times greater than at Location A due to differences in shielding and overburden. Location A had more concrete and steel shielding, which attenuated neutron leakage more effectively.

- describe simulated neutron emission

Simulated neutron emission was modeled using Monte Carlo transport codes, which predicted the spatial and energy distribution of neutrons escaping from the reactor core. Simulation results were consistent with experimental measurements.

- derive ratio of neutron flux to reactor power

The ratio of neutron flux to reactor power was derived from experimental data and found to vary linearly with the weighted isotopic composition of the core, as predicted by theoretical models.

- show linear dependence of detector count rate on reactor power

The detector count rate exhibited a linear dependence on reactor power, with a slope that remained stable over time unless altered by changes in core composition.

- compare experimental and simulated data

Experimental data closely matched simulated neutron emission profiles, validating the accuracy of the underlying physics models and confirming the feasibility of using neutron detection for isotopic monitoring.

- describe application of system for reactor monitoring

The system can be applied to any thermal neutron reactor, including research reactors, small modular reactors, and pressurized heavy water reactors. It provides continuous, non-intrusive verification of fuel cycle activities.

- explain use of isotopic-specific characteristics

Isotopic-specific characteristics, such as fission cross-sections and energy yields, are used to interpret changes in neutron flux per unit power. These characteristics enable the system to distinguish between changes in uranium-235 and plutonium-239 inventories.

- describe measurement of variations in reactor power

Variations in reactor power are measured using the reactor’s own operational sensors and are used to normalize neutron detection rates, ensuring that observed deviations are attributable to isotopic changes rather than power fluctuations.

- explain analysis of variation in detection rate

Analysis of variation in detection rate involves statistical comparison of real-time data to a baseline, using control charts and moving averages to identify trends that deviate from expected behavior.

- describe application for independent verification

The system provides an independent verification mechanism that can be used by international safeguards agencies to confirm or challenge operator declarations without requiring physical access to the reactor.

- simplify model for neutron flux per unit reactor power

The model for neutron flux per unit reactor power is simplified to a linear relationship between the weighted sum of fissile isotope masses and the measured neutron flux, based on the known fission properties of uranium-235, plutonium-239, and plutonium-241.

- describe changes in fissile isotope inventory

Changes in fissile isotope inventory occur due to burn-up, transmutation, and refueling. These changes alter the neutron flux per unit power in a predictable manner, enabling detection of fuel manipulation.

- show predicted reduction in neutron flux

Simulations predict that the removal of 1 kg of uranium-235 reduces the neutron flux per unit power by approximately 4%, while the removal of 100 g of plutonium-239 reduces it by approximately 0.5%.

- describe changes in average fissile isotope masses

Changes in the average mass of fissile isotopes over time are tracked using reactor physics simulations and are correlated with measured neutron flux per unit power to detect anomalies.

- show changes in measured neutron count rate and predicted neutron flux

Measured neutron count rates and predicted neutron flux values show close agreement, with deviations within 5%, confirming the accuracy of the model and the sensitivity of the detection method.

- describe demonstration of stand-off reactor monitoring

The invention demonstrates, for the first time, that stand-off neutron monitoring can detect changes in fissile isotope inventory at distances exceeding 60 meters from the reactor core, with a sensitivity sufficient to detect kilogram-scale fuel movements within weeks.

- explain factors affecting detection efficiency

Factors affecting detection efficiency include detector size, moderator thickness, shielding geometry, distance from the core, and environmental background. These factors are accounted for during system calibration and baseline establishment.

- compare BCS and B10+ detector data

BCS and B10+ detector data show similar trends in neutron flux per unit power, confirming that different detector types can be used interchangeably in the system, provided they are properly calibrated.

- describe environmental shielding and overburden

Environmental shielding and overburden, such as building walls, soil, and concrete structures, attenuate neutron leakage and must be characterized during system deployment to ensure accurate baseline establishment.

- show correlation between detector signals at locations A and B

Detector signals at Locations A and B show strong temporal correlation, confirming that both detectors are responding to the same core phenomena, despite differences in shielding and distance.

- describe regular deviations in signal at Location B

Regular 50% reductions in signal at Location B were correlated with the timing of online refueling activities, during which a fuel rod flask temporarily blocked neutron leakage from the top of the reactor.

- explain timing of online refueling activities

Online refueling activities involve the insertion and removal of fuel rods while the reactor remains operational. These activities cause temporary changes in neutron leakage patterns, which are detectable by neutron detectors positioned to view the reactor top.

- describe blocking of neutrons by fuel rod flask

The fuel rod flask, when positioned over the reactor top during refueling, acts as a neutron shield, reducing the number of neutrons escaping from the core and causing a temporary drop in detector count rate.

- explain spikes in count rate during refueling

Spikes in count rate during refueling occur when fuel rods are exchanged between the core and the flask, causing a transient increase in neutron leakage due to the reconfiguration of fuel geometry.

- motivate multiple detectors

Multiple detectors are motivated by the need to distinguish between genuine core anomalies and localized interferences. A deviation observed at only one location is likely environmental, while a deviation observed at multiple locations is likely core-related.

- describe detector placement

Detectors are placed at multiple locations around the reactor to capture different neutron leakage paths and to enable spatial correlation of signals. Placement is optimized to maximize signal-to-noise ratio and to minimize vulnerability to interference.

- introduce method 600

Method 600 is a process for monitoring nuclear reactor operations using stand-off neutron detection. The method includes steps for data acquisition, normalization, baseline comparison, anomaly detection, and alert generation.

- describe step 602

Step 602 involves positioning neutron detectors at stand-off distances from the reactor core, outside the primary radiation shielding, and configuring them to detect thermal neutrons.

- describe step 604

Step 604 involves continuously measuring neutron flux from each detector and transmitting the data to a central controller.

- describe step 606

Step 606 involves receiving real-time reactor power data from the facility’s operational sensors and normalizing neutron flux measurements by reactor power.

- describe step 608

Step 608 involves comparing the normalized neutron flux to a baseline established during normal reactor operation and determining whether deviations exceed statistical thresholds.

- describe step 610

Step 610 involves generating an alert if a statistically significant deviation is detected, including details on the magnitude, timing, and location of the anomaly.

- describe step 612

Step 612 involves storing all data, alerts, and operational logs in a secure, tamper-resistant database for audit and verification purposes.

- describe step 614

Step 614 involves periodically updating the baseline using new operational data and simulated neutron flux predictions to maintain accuracy over time.

- describe relative analysis

Relative analysis involves comparing neutron flux per unit power at a given time to a reference baseline established during a prior period of known operation, without requiring absolute knowledge of fuel masses.

- describe absolute analysis

Absolute analysis involves using reactor physics simulations to predict the expected neutron flux per unit power based on declared fuel inventories and comparing it to measured values.

- describe iterative analysis

Iterative analysis involves continuously updating the baseline and simulation model as new data are collected, improving the accuracy of anomaly detection over time.

- describe independent analysis

Independent analysis involves verifying system outputs against external data sources, such as operator declarations or physical inspection records, to ensure consistency and integrity.

- describe controller functionality

The controller performs real-time data normalization, statistical analysis, baseline comparison, alert generation, and secure data logging. It is designed to operate autonomously and to resist tampering or unauthorized access.

- describe stand-off neutron monitoring

Stand-off neutron monitoring is the practice of detecting neutrons escaping from a reactor core at distances greater than 10 meters from the primary shielding, without requiring physical access to the reactor or its fuel.

- introduce Monte Carlo model

A Monte Carlo neutron transport model is used to simulate the emission, moderation, and leakage of neutrons from the reactor core, providing a theoretical basis for the expected neutron flux per unit power.

- describe detector model

The detector model incorporates the geometry, material composition, and efficiency of each neutron detector, allowing for accurate prediction of count rates based on simulated neutron flux distributions.

- describe simulation approach

The simulation approach involves modeling the reactor core, shielding, and detector array using Monte Carlo N-Particle (MCNP) software, with neutron sources derived from fission spectra and material compositions.

- describe stage 1

Stage 1 involves simulating the neutron emission spectrum from the reactor core based on declared fuel composition and burn-up history.

- describe stage 2

Stage 2 involves modeling the transport of neutrons through the reactor shielding and surrounding environment to predict the neutron flux at each detector location.

- describe stage 3

Stage 3 involves comparing simulated neutron fluxes with measured detector counts to validate the model and refine the baseline for anomaly detection.

- describe simulation results

Simulation results show excellent agreement with experimental measurements, confirming that the neutron flux per unit power is a reliable indicator of fissile isotope inventory.

- describe neutron energy spectra

Neutron energy spectra measured at stand-off distances are dominated by epithermal neutrons, with a tail extending into the fast neutron region, consistent with partial moderation by reactor shielding.

- describe area sums

Area sums refer to the total neutron detection area of the detector array, which is optimized to maximize signal-to-noise ratio and detection sensitivity.

- describe reactor-power normalized neutron flux

Reactor-power normalized neutron flux is the ratio of neutron count rate to reactor power, which is the key parameter used to detect changes in fissile isotope inventory.

- describe BCS detector

The BCS detector is a boron-coated straw detector consisting of 49 sealed aluminum tubes, each coated with boron-10 enriched boron carbide and filled with argon-carbon dioxide gas. It is sensitive to thermal neutrons and produces a cumulative signal from all straws.

- describe B10+ detector

The B10+ detector is a stainless steel tube detector lined with boron-10 enriched coating and filled with helium-3 and noble gases. It provides high signal-to-noise performance and is suitable for long-term, unattended operation.

- describe data acquisition

Data acquisition involves the continuous recording of neutron count rates and reactor power at one-minute intervals, with time-stamping and secure transmission to the central controller.

- describe data analysis

Data analysis involves normalization of neutron flux by reactor power, statistical comparison to baseline, detection of deviations, and generation of alerts. Analysis is performed using automated algorithms and validated by human review.

- describe time-series plots

Time-series plots display neutron count rate, reactor power, and neutron flux per unit power as functions of time, enabling visual identification of trends and anomalies.

- describe detector count rate

Detector count rate is the number of neutron detection events recorded per unit time, corrected for background and normalized by reactor power.

- describe comparison to reactor power

Comparison to reactor power involves plotting neutron count rate against reactor power to confirm linearity and to identify deviations that are not attributable to power fluctuations.

- describe scope of invention

The scope of the invention includes all systems and methods for monitoring nuclear reactor operations through stand-off neutron detection, regardless of reactor type, detector configuration, or deployment environment.

- describe non-limiting nature

The invention is not limited to the specific embodiments described herein. Variations in detector type, number, placement, signal processing, and analytical methods are encompassed within the scope of the invention, provided that the core principle of detecting isotopic changes through neutron flux per unit reactor power at stand-off distances is maintained.