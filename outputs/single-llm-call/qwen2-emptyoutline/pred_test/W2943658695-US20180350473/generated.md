# DESCRIPTION

## FIELD

The field of the present invention pertains to the monitoring of nuclear reactor fuel cycle activities, specifically through the detection of neutrons emitted from the reactor core. The invention involves the use of large-area neutron detectors placed at stand-off distances from the reactor to monitor the reactor's power and isotopic inventory. This technique is particularly useful for nuclear safeguards, enabling the detection of any unauthorized changes in the reactor's fuel composition or operational status.

## BACKGROUND

Nuclear reactor safeguards are essential for ensuring that nuclear materials are not diverted from peaceful uses. The International Atomic Energy Agency (IAEA) employs various measures to verify the integrity of nuclear facilities, including the use of tamper-indicating seals, unattended video surveillance cameras, and regular inspections. However, these traditional methods are often insufficient for monitoring small modular reactors (SMRs) and research reactors due to their diverse designs and operational characteristics.

Research reactors, in particular, pose significant safeguards challenges due to their wide variation in design, the type of fuel used (often highly-enriched uranium), and the potential for target irradiation. The IAEA has identified the development of new instruments and techniques to detect the establishment of nuclear fuel cycle activities as a high priority. One promising approach is the use of neutron detection to monitor reactor fuel cycle activities.

Neutrons are a prominent emanation from fission nuclear reactors, and their detection can provide real-time information about the reactor's power and isotopic inventory. Previous work has focused on using neutron detectors outside of reactor shielding to determine power density distribution within reactors. However, the present invention explicitly demonstrates how escaping fast neutrons, detected as thermalized neutrons, can be related to monitoring the fissile isotope inventory of a reactor core for safeguards purposes.

## SUMMARY

The present invention provides a method and system for monitoring nuclear reactor fuel cycle activities using large-area neutron detectors placed at stand-off distances from the reactor. The invention is based on the principle that the number of neutrons detected is proportional to the population of neutrons in the reactor core, which in turn is related to the reactor's power and the isotopic composition of the fuel.

Key aspects of the invention include:
1. **Detector Placement**: Large-area neutron detectors are strategically placed at various locations around the reactor, both inside and outside the reactor building, to monitor neutron emissions.
2. **Signal Coordination**: Signals from multiple detectors are coordinated to discriminate against interferences and provide a comprehensive picture of the reactor's operational status.
3. **Data Analysis**: The neutron detection count rate is correlated with the reactor's thermal power and isotopic inventory to detect any unauthorized changes in the reactor's fuel composition or operational status.
4. **Environmental Considerations**: The method accounts for environmental factors such as reactor shielding, overburden, and operational variations to ensure accurate monitoring.

The invention offers several advantages:
- **Non-Invasive Monitoring**: The use of stand-off detectors allows for continuous, non-invasive monitoring of the reactor without disrupting its operation.
- **Economical and Practical**: The use of large-area neutron detectors provides an economical and practical means of supporting the achievement of IAEA safeguards goals.
- **High Sensitivity**: The method can detect changes in the reactor's isotopic inventory with high sensitivity, enabling the timely detection of any unauthorized activities.

## DETAILED DESCRIPTION

### Theory

The technique of stand-off reactor monitoring using neutron detection is based on the principle that the number of neutrons detected (\(n_{\det}\)) is proportional to the population of neutrons in the reactor core (\(n_{pop}\)). This relationship can be expressed as:
\[ n_{\det} \propto n_{pop} \propto \frac{\left\langle \phi \right\rangle}{\left\langle v \right\rangle}V, \]
where \(\left\langle \phi \right\rangle\) is the average neutron flux in the reactor core, \(\left\langle v \right\rangle\) is the average speed of the neutrons in the reactor core, and \(V\) is the volume of the reactor core.

For monoenergetic incident neutrons, the volumetric rate of fission is given by \(\Sigma_{f}\phi\), where \(\Sigma_{f} = N\sigma_{f}\) is the macroscopic fission cross-section, \(\sigma_{f}\) is the microscopic cross-section, \(\phi\) is the flux of monoenergetic neutrons, and \(N\) is the number density of the fissile nuclei. In a reactor, the neutron energy spectrum is not monoenergetic, the neutron flux varies with spatial location and time, and the spatial distribution of fissile material is not uniform. Therefore, the fission rate is determined by integrating \(\Sigma_{f}\left( {\mathbf{r},E_{n},t} \right)\phi\left( {\mathbf{r},E_{n},t} \right)\) over all locations (\(\mathbf{r}\)), and neutron energies (\(E_{n}\)) in the reactor.

For thermal neutron reactors, where most fissions occur in the thermal neutron energy range, one can assume that \(\phi\) is an appropriate space and energy average flux of thermal neutrons, and \(\Sigma_{f}\) is a corresponding average macroscopic cross-section. By multiplying \(\Sigma_{f}\phi\) by the volume \(V\) of the reactor, as well as the energy released per fission \(E_{f}\), the reactor thermal power \(P_{tot}\) can be estimated by:
\[ P_{tot} = V\left\langle \phi \right\rangle\sum\limits_{i}\left\langle \Sigma_{f,i} \right\rangle E_{f,i} = V\left\langle \phi \right\rangle\sum\limits_{i}N_{i}\left\langle \sigma_{f,i} \right\rangle E_{f,i}, \]
where the summation index \(i\) runs over the fissile isotope species in the reactor core.

The average factors in Equation (2) should technically be integrals over energy and space, and therefore vary as a function of fuel burn-up distribution or refueling over time. However, these factors can be approximated as appropriately averaged factors that do not vary with energy or as a function of location. Remarkably, as demonstrated in this work, these average factors work very well under the assumptions stated and can be used to verify the change in isotopic composition, \(N_i\), over time.

### Detector Locations

In the present invention, two large-area neutron detectors were placed at two locations in proximity to a nuclear research reactor, the National Research Universal (NRU) reactor in Chalk River, Ontario, Canada. One location (Location A) was within the NRU reactor building, approximately 17 meters from the NRU reactor core, two levels below the main reactor floor. The other location (Location B) was outside of the NRU reactor building in a portable trailer building, approximately 69 meters from the NRU reactor core. These locations were chosen for their difference in proximity to the reactor and the varying shielding burden to neutrons incident on the detectors.

### Neutron Moderation

The large-area neutron detectors employed in this invention are thermal neutron detectors, most sensitive in detecting neutrons in thermal equilibrium with their environment. Enhancement of their detection rate when exposed to neutrons with energy greater than that of thermal equilibrium can be achieved through the use of hydrogenous moderating material surrounding the detector. To determine the average energy of the neutrons incident upon the neutron detectors, the thickness of high-density polyethylene (HDPE) surrounding a single detector tube at Location B was varied. The optimal thickness of HDPE for the detector count rate was found to be 2.5 cm, suggesting that the average energy of neutrons incident on the detector was substantially less than 2 MeV but greater than thermal energy, i.e., in an epithermal regime.

### Neutron Count Rate Variation With Reactor Power

The neutron detection count rate followed the reactor thermal power through its temporal fluctuations well while the NRU was at power. This suggests that the ratio of reactor power to detector count rate is a meaningful quantity to follow. A linear regression fit applied to the B10+ detector raw count rate at Location B as a function of average reactor power during reactor start-up and shut-down periods demonstrated a clear linear dependence. This confirms that neutron detection outside of reactor shielding can be used to monitor changes in the in-core neutron flux, which is useful for monitoring and verifying nuclear reactor fuel cycle activities from a nuclear safeguards point of view.

### Neutron Count Rate Variation With Isotopic Inventory

The attribute that makes neutron monitoring outside of reactor shielding a useful tool for verification purposes in safeguards applications is that the technique is sensitive to changes in fissile isotope inventory in the reactor core. The neutron detector count rate was recorded as a function of time, over the course of weeks and even months. The quantities that varied over this time scale were the fissile isotope masses, particularly as the U-235 in fresh fuel was burned up and other isotopes of U and Pu were produced via transmutation or introduced via on-line refueling.

The neutron detector count rate per unit reactor power is proportional to the average in-core neutron flux per unit thermal reactor power, which is a function of the weighted isotopic composition of the reactor core. The weighted isotopic composition is based on the masses of fissile isotopes present at the time of measurement, estimated from neutron diffusion code simulations. The data showed a clear linear relationship between the neutron count rate per unit reactor power and the weighted isotopic composition, confirming the sensitivity of the technique to changes in the reactor's isotopic inventory.

### Implementation and Practical Considerations

The invention can be implemented using large-area neutron detectors such as boron-lined detectors and B10+ detectors. These detectors are placed at strategic locations around the reactor to monitor neutron emissions. The signals from multiple detectors are coordinated to discriminate against interferences and provide a comprehensive picture of the reactor's operational status.

Environmental factors such as reactor shielding, overburden, and operational variations are accounted for to ensure accurate monitoring. The method can detect changes in the reactor's isotopic inventory with high sensitivity, enabling the timely detection of any unauthorized activities. The invention is particularly useful for monitoring small modular reactors (SMRs) and research reactors, which pose significant safeguards challenges due to their diverse designs and operational characteristics.

### Conclusion

The present invention provides a method and system for monitoring nuclear reactor fuel cycle activities using large-area neutron detectors placed at stand-off distances from the reactor. The invention is based on the principle that the number of neutrons detected is proportional to the population of neutrons in the reactor core, which in turn is related to the reactor's power and isotopic inventory. The method offers non-invasive, economical, and practical monitoring of reactor operations, enabling the timely detection of any unauthorized activities. This invention is particularly valuable for enhancing nuclear safeguards and ensuring the peaceful use of nuclear materials.