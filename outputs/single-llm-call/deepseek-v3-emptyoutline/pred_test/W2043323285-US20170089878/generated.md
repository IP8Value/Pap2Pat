Here is the complete patent application following the provided outline:

# DESCRIPTION  

## STATEMENT AS TO RIGHTS TO INVENTIONS MADE UNDER FEDERALLY-SPONSORED RESEARCH AND DEVELOPMENT  

The invention described herein was made with government support under contracts awarded by the U.S. Department of Energy Wind and Water Power Technologies Office and The Electric Power Research Institute. The government has certain rights in the invention.  

## BACKGROUND OF THE INVENTION  

Hydropower facilities present significant hazards to fish populations during downstream passage through turbines and other hydraulic structures. Fish may experience injury or mortality due to exposure to severe hydraulic conditions including rapid pressure changes, shear stress, turbulence, mechanical strikes from turbine blades, and cavitation effects. While field studies using live fish provide some biological performance data, they fail to identify the specific hydraulic conditions, physical stresses, or precise locations where injurious conditions occur.  

Previous attempts to measure these conditions resulted in the development of an autonomous sensor device known as the Gen 1 Sensor Fish. While this device represented an advancement in measuring hydraulic conditions during fish passage, it possessed several limitations including size constraints, functional limitations, deployment challenges, and high costs. Furthermore, the original design could not withstand the extreme conditions found in high-head dams with Francis turbines or pump storage facilities, nor could it measure the full range of pressures encountered in such environments.  

There exists an unmet need for an improved sensor device capable of withstanding extreme hydraulic conditions while providing comprehensive, high-resolution data about the physical stresses experienced by fish during turbine passage. Such a device would enable more accurate evaluation of turbine designs and operations to minimize ecological impacts while maintaining power generation efficiency.  

## SUMMARY  

The present invention provides a next-generation autonomous sensor device, referred to as the Gen 2 Sensor Fish, designed to overcome the limitations of previous sensor technologies. The device comprises a compact, neutrally buoyant housing containing multiple integrated sensors capable of measuring three-dimensional linear acceleration, three-dimensional rotational velocity, orientation, pressure, and temperature at high sampling frequencies.  

Key improvements over prior devices include enhanced robustness to withstand extreme conditions encountered in high-head dams and Francis turbines, expanded pressure measurement capabilities, increased data acquisition and storage capacity, improved communication capabilities, and reduced manufacturing costs. The device features a recovery mechanism enabling surface retrieval after data collection, along with active locating systems using radiofrequency transmission and high-intensity visual indicators.  

The invention further includes calibration methods ensuring measurement accuracy within ±2% for pressure, ±5% for acceleration and rotational velocity, ±4° for orientation, and ±2°C for temperature. Laboratory and field evaluations demonstrate the device's ability to reliably collect data under actual turbine operating conditions while surviving impacts up to 600g.  

## DESCRIPTION  

The Gen 2 Sensor Fish comprises a cylindrical housing measuring approximately 24.5 mm in diameter and 89.9 mm in length, with a mass of about 42.1 grams. The device is designed to be neutrally buoyant in fresh water, with size and density characteristics similar to yearling salmon smolts. The internal components are arranged such that the center of gravity coincides with the geometric center of the device.  

The sensor suite includes a three-axis accelerometer (ADXL377) with a ±200g measurement range and 10,000g shock survival rating, a three-axis gyroscope (ITG-3200) with ±2000°/s range, and an eCompass module (LSM303DLHC) combining a three-axis accelerometer and magnetometer. Pressure measurements are obtained through a micromachined silicon sensor (MS5412-BM) with 12 bar range and 30 bar overpressure rating, while temperature is monitored using a linear-response sensor (TC1046) with -40°C to +125°C range.  

A low-power microcontroller coordinates data acquisition from all sensors at 2048 Hz, storing up to five minutes of data in non-volatile flash memory. Power is supplied by a rechargeable battery system. The recovery mechanism employs nichrome wire cutters to release spring-loaded weights at predetermined times, converting the device to positive buoyancy for surface recovery.  

The device includes multiple locating features including a 146 MHz RF transmitter and four high-intensity orange LEDs (with alternative color options available). Activation occurs via magnetic switch, followed by system self-checks and configurable delay periods before data collection. Post-retrieval, data is downloaded through a docking station interface supporting simultaneous processing of multiple devices.  

Calibration procedures have been developed for each sensor subsystem. Pressure calibration uses reference standards in hyperbaric chambers, while acceleration calibration employs precision linear test tracks with reference accelerometers. Rotational velocity calibration combines mechanical rotation fixtures with high-speed videography analysis. Orientation calibration uses controlled magnetic field environments, and temperature calibration employs ice-point reference methods.  

Field deployment protocols have been established for various hydraulic structures, with successful testing completed at operational dam spillways. The device has demonstrated survival through impacts up to 600g and reliable operation in actual turbine passage conditions.  

The complete system includes supporting software for device configuration, data download, unit conversion, and visualization. The software package enables conversion of raw binary data to physical units using calibration coefficients and provides tools for analysis of complex multi-sensor datasets.  

The invention represents a significant advancement in hydraulic condition monitoring, particularly for fish passage evaluation in hydropower systems. By providing comprehensive, high-resolution data about the physical stresses experienced during turbine passage, the Gen 2 Sensor Fish enables improved turbine design and operation to minimize ecological impacts while maintaining power generation efficiency.