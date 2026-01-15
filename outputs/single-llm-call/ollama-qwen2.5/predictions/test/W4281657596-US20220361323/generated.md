- **Small Craniotomy for Zebra Finches**: Adult zebra finches were anesthetized using isoflurane gas anesthesia (1.5% in oxygen). A small craniotomy was performed above the songbird analog of the motor cortex (HVC) to allow for acute spontaneous recordings under anesthesia.

- **Mice Anesthesia and Surgery**: Mice were anesthetized with isoflurane gas, and a head plate was added. A thermistor was implanted between the nasal bone and inner nasal epithelium to measure respiration. A craniotomy was performed above one olfactory bulb, contralateral to the thermistor implantation.

- **Post-Surgical Care for Mice**: After surgery, the craniotomy site was covered with a biocompatible silicone elastomer sealant (Kwik-cast, WPI). The mice were given 3 days to recover before further procedures.

- **Recording Setup and Data Acquisition**: Probes were inserted using a custom fixture. Recordings were taken via OpenEphys software using a 128-channel data acquisition system (RHD2000; Intan Technologies) at a 30 kHz sampling frequency. Sniffing was recorded with intranasally implanted thermistors.

- **Data Analysis Methods**: Electrophysiological and sniffing data were analyzed in MATLAB. Kilosort and Phy2 were used for spike analysis. Inhalation and exhalation times were determined by identifying peaks and troughs in the temperature signal after smoothing with a 25 ms moving window, excluding sniffs outside the 5th to 95th percentile duration.

- **Electrochemical Measurements**: Electrochemical impedance spectroscopy (EIS) and cyclic voltammetry (CV) data were collected using a high surface area Pt counter electrode and an Ag/AgCl reference electrode. Measurements were conducted in phosphate-buffered saline, with the solution sparged to remove dissolved oxygen before testing.

- **Device Finalization and Release**: The 1035 nm pulsed laser was used to cut through polyimide and parylene layers for device release. Devices were then detached from the wafer by placing it in warm water. Omnetics connectors were attached to the device pads using anisotropic conductive film (ACF).

- **Laser Tip Ablation Process**: A 1035 nm pulsed laser was used to remove parylene C from the tips of the prints. The laser was coaligned with the 3D printing system and scanned a 2-3 micron thick volume at a 1 MHz pulsing setting, taking less than one minute per array.

- **FIB Milling for Internal Assessment**: Focused ion beam (FIB) milling was used to slice open printed structures and assess internal metalization. The process was performed using a ThermoFisher Helios Hydra PFIB with an Ar/O2 beam.

- **Porous Electrode Fabrication**: Porous electrodes were fabricated on Si wafers by patterning an initial Ti/Pt layer, isolating it with SU-8 photoresist, and creating openings for electrical connections. A Cr sacrificial mask was used to define print locations, followed by 3D printing and metalization.

- **Parylene C Deposition**: After the final device assembly, a 3 µm thick layer of parylene C was deposited over the wafer using a Labcoater 4200. The Omnetics connector contact pads were masked with Kapton tape to prevent parylene deposition.

- **Device Release from Wafer**: The entire device was released from the wafer by cutting through polyimide and parylene layers with a 1035 nm pulsed laser, followed by placing the wafer in warm water. Omnetics connectors were then attached to the device pads using anisotropic conductive film (ACF).

- **Electrophysiological Data Checks**: Additional complete device checks were performed using OpenEphys and Intan chips. Cyclic voltammetry (CV) curves were cycled at 50 mV/s until differences between subsequent scans were no longer observed, ensuring stable device performance.

- **Ethical Compliance**: All experimental procedures involving animals were approved by the Institutional Animal Care and Use Committee (IACUC) at the University of Oregon and complied with the National Institutes of Health Guide to the Care and Use of Laboratory Animals.