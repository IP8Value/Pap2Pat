Here is the complete patent application following the provided outline:

# DESCRIPTION  

## STATEMENT AS TO RIGHTS TO INVENTIONS MADE UNDER FEDERALLY-SPONSORED RESEARCH AND DEVELOPMENT  

The United States Government has rights in this invention pursuant to Contract No. DE-AC05-76RL01830 between the U.S. Department of Energy and Battelle Memorial Institute. The development of this invention was supported in part by funding from the U.S. Department of Energy Wind and Water Power Technologies Office and The Electric Power Research Institute. The U.S. Army Corps of Engineers provided additional support for field deployment activities.  

## BACKGROUND OF THE INVENTION  

The present invention relates generally to the field of hydroelectric power generation and specifically to devices for monitoring hydraulic conditions experienced by fish passing through hydroelectric turbines. Fish passage through hydroturbines and other hydraulic structures presents significant ecological challenges, as fish may be injured or killed when exposed to severe hydraulic conditions including rapid pressure changes, shear stress, turbulence, blade strikes, and cavitation.  

With many hydroelectric turbines in the Columbia and Snake River systems nearing the end of their operational lifespans, there exists a critical need to evaluate new turbine designs that can minimize ecological impacts while maintaining power generation efficiency. Current evaluation methods using live fish, while necessary for biological performance assessment, cannot determine the specific hydraulic conditions or physical stresses experienced during passage.  

Prior art sensor devices, such as the first generation Sensor Fish (Gen 1), provided some measurement capabilities but were limited by size, functionality, deployment requirements, and cost constraints. The Gen 1 device could not withstand the extreme conditions found in high-head dams with Francis turbines or pump storage facilities, and its pressure measurement range was insufficient for many applications.  

There exists an unmet need for an improved sensor device that can accurately measure the full range of hydraulic conditions while withstanding extreme environments. The ideal device would feature enhanced robustness, increased measurement capabilities, improved data acquisition and storage capacity, and reduced production costs compared to previous solutions.  

## SUMMARY  

The present invention provides a second generation Sensor Fish (Gen 2) device that overcomes the limitations of prior art solutions. The Gen 2 Sensor Fish comprises a compact, autonomous sensor package capable of measuring three-dimensional linear accelerations, rotational velocities, orientation, pressure, and temperature during passage through hydraulic structures.  

Key features of the invention include a durable housing measuring approximately 24.5 mm in diameter and 89.9 mm long, with neutral buoyancy in fresh water. The device incorporates multiple precision sensors including a three-axis accelerometer with ±200 g range, a three-axis gyroscope with ±2000°/s range, a pressure sensor with 12 bar range, and temperature sensors with -40 to +125°C range.  

The device operates through an automated sequence beginning with magnetic activation, followed by sensor data collection at 2048 Hz sampling rate, storage of up to 5 minutes of data in non-volatile flash memory, and activation of a recovery mechanism after a programmed delay. The recovery system utilizes a spring-loaded weight release mechanism triggered by nichrome wire heating, causing the device to become positively buoyant and surface for retrieval.  

Data download occurs through a docking station interface that also recharges the internal battery. The complete system includes analysis software capable of converting raw sensor data into physical units and generating visualizations of the measured parameters. Compared to previous solutions, the present invention offers significantly improved measurement capabilities, durability, and cost-effectiveness for evaluating fish passage conditions in hydroelectric facilities.  

## DESCRIPTION  

The present invention provides a comprehensive solution for monitoring hydraulic conditions experienced during passage through hydroelectric turbines and other water control structures. The device, termed Gen 2 Sensor Fish, represents a significant advancement over prior art through its combination of precision measurement capabilities, robust construction, and autonomous operation.  

The invention encompasses multiple embodiments optimized for different hydraulic environments, including Kaplan turbines, Francis turbines, and pump storage facilities. The primary embodiment features a cylindrical housing constructed from durable materials capable of withstanding extreme pressure and impact conditions. The housing dimensions of 24.5 mm diameter and 89.9 mm length, combined with a mass of approximately 42.1 g, provide neutral buoyancy characteristics similar to yearling salmon smolt.  

The sensor fish housing contains a carefully arranged complement of measurement components positioned to maintain balance and measurement accuracy. A low-power 16-bit PIC microcontroller serves as the central processing unit, coordinating data acquisition from multiple sensors and managing power consumption. The microcontroller interfaces with a 3-axis accelerometer (ADXL377) providing ±200 g measurement range per axis with 10,000 g shock survival rating. This primary accelerometer features user-selectable bandwidth from 0.5-1300 Hz for optimal measurement of turbine passage dynamics.  

A 3-axis gyroscope (ITG-3200) provides rotational velocity measurements with ±2000°/s range per axis, incorporating 16-bit analog-to-digital conversion and I²C digital interface. The gyroscope includes an embedded temperature sensor and features configurable low-pass filtering to optimize signal quality. An eCompass module (LSM303DLHC) combines a 3-axis digital accelerometer (±16 g range) and 3-axis magnetometer (±8.1 gauss range) for orientation measurement, with additional temperature sensing capability.  

Pressure measurement is accomplished through a Wheatstone bridge type sensor (MS5412-BM) with 12 bar (174 psi) full-scale range and 30 bar overpressure rating. The pressure sensor connects to an instrumentation amplifier on the circuit board with precisely calibrated gain and offset settings. Temperature monitoring utilizes a primary sensor (TC1046) with -40 to +125°C range and linear 6.25 mV/°C output, mounted in thermal contact with the pressure sensor housing for accurate water temperature measurement.  

The recovery mechanism represents a critical innovation enabling device retrieval after passage. The system employs two spring-loaded weights secured by fishing line looped over nichrome wire elements. Upon command from the microcontroller, controlled current pulses heat the nichrome wires, severing the fishing line and releasing the weights to make the device positively buoyant. The mechanism includes redundant activation paths and failsafe timing to ensure reliable operation in various hydraulic conditions.  

Device operation begins with magnetic activation of a reed switch, initiating a self-check routine that verifies battery voltage and memory status. Following a configurable delay period, the system begins acquiring sensor data at 2048 Hz, storing measurements in non-volatile flash memory. After data collection completes, the device waits a programmed interval before activating the recovery mechanism and surfacing. Once at the surface, high-intensity orange LEDs and a 146 MHz RF beacon assist in location and retrieval.  

Data download occurs through a docking station interface that connects to the device via a two-pin serial interface. The docking station incorporates battery charging circuitry and a TTL-to-USB converter for data transfer to analysis computers. Custom communication software enables configuration, data retrieval, and conversion of raw binary data into calibrated physical units. The complete system supports simultaneous management of multiple Sensor Fish devices for large-scale evaluation projects.  

The invention includes several specialized components to ensure reliable operation. A low quiescent current LDO regulator provides stable power to sensitive analog circuits, while a 12-bit successive approximation ADC enables precise measurement of analog sensor outputs. Flash memory organization optimizes data storage efficiency, and SPI communication protocols ensure reliable data transfer. The system incorporates status indication through a bi-color LED and utilizes an internal low-frequency oscillator for timing-critical operations.  

The sensor fish design places particular emphasis on component arrangement to maintain measurement accuracy. The primary accelerometer is positioned at the geometric center of the device coinciding with the center of mass, while other sensors are distributed to minimize interference effects. Flexible circuit boards accommodate the cylindrical form factor, and thermal management techniques ensure temperature measurement accuracy. The complete device represents a significant advancement in hydraulic condition monitoring, with particular utility for evaluating fish passage conditions in hydroelectric facilities.