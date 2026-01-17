# DESCRIPTION

## BACKGROUND

The field of turbomachinery, particularly in the design and operation of bladed disks or drums, faces significant challenges related to structural vibrations and high-cycle fatigue. These issues arise primarily due to the low damping characteristics of modern materials and fabrication techniques, which can lead to severe vibrations and potential failure. Various methods have been explored to increase damping, including blade friction damping, friction ring dampers, viscoelastic damping treatments, and piezoelectric shunts. Among these, piezoelectric shunts, which convert mechanical energy into electrical energy that is then dissipated in an electrical network, have shown promise. However, traditional piezoelectric shunt methods, such as R-shunts and RL shunts, have limitations in terms of performance and practicality, especially in rotating machinery.

R-shunts, which involve only resistors, offer limited performance but are simple and robust. RL shunts, which include both resistors and inductors, provide superior damping but require precise tuning to the targeted vibration modes. This tuning is challenging and sensitive to variations in natural frequencies, making it impractical for many applications, particularly in rotating machines where large inductors are required. Active components, such as synthetic inductors, are often used to address this issue but introduce complexity and reliability concerns.

To overcome these limitations, a novel method for damping a specific mode with \( n \) nodal diameters has been developed. This method involves organizing piezoelectric transducers in parallel loops, which significantly reduces the demand on inductors while maintaining effective damping. Specifically, a set of \( 4n \) piezoelectric transducers (PZT patches) can be arranged in two parallel loops of \( 2n \) patches each. This configuration reduces the inductance requirement by \( 4n^2 \) compared to independent inductive loops, making it feasible to use passive components in practical applications. This approach is particularly useful for damping critical modes in turbomachinery, where the excitation frequencies are multiples of the rotation speed and the mode shapes are harmonic in the circumferential direction.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENT

The invention pertains to a method for enhancing the damping of a specific mode with \( n \) nodal diameters in rotationally periodic structures, such as bladed disks or drums. The method utilizes piezoelectric transducers (PZT patches) organized in parallel loops to reduce the demand on inductors and improve the practicality of passive RL shunt damping. The following detailed description outlines the preferred embodiment of the invention, including the theoretical basis, experimental validation, and practical implementation.

### Theoretical Basis

#### Rotationally Periodic Structures

A bladed disk or drum equipped with \( N \) blades exhibits cyclic symmetry, with an interblade phase angle of \( 2\pi/N \). The free vibration modes of such structures are harmonic in the circumferential direction, leading to nodal lines across the mode shapes called nodal diameters. The natural frequencies of these modes are typically plotted as a function of the number of nodal diameters, with the maximum number being \( N/2 \) for even \( N \) and \( (N-1)/2 \) for odd \( N \).

The modes can be categorized into drum-dominated and blade-dominated modes. Drum-dominated modes tend to have increasing natural frequencies with the number of nodal diameters, while blade-dominated modes have relatively constant natural frequencies. The forced response of periodic structures to a rotating point force can be analyzed to identify the critical mode with \( n \) nodal diameters that is likely to be excited and must be targeted for damping.

#### RL Shunt Damping

##### Independent Loops

When multiple PZT patches are available, several architectures can be employed for RL shunt damping. For a mode with natural frequency \( \omega_i \), the inductance \( L \) of the RL shunt should be selected according to:

\[ L = \frac{1}{\omega_i^2 C} \]

where \( C \) is the electrical capacitance of one PZT patch. The performance of the RL shunt depends critically on the tuning of the inductor, while the value of the resistance \( R \) is less sensitive. The optimal resistance \( R \) is given by:

\[ R = \sqrt{\frac{L}{C}} K_i \]

where \( K_i \) is the total effective electromechanical coupling factor of mode \( i \), which is the sum of the effective electromechanical coupling factors of the \( p \) transducers:

\[ K_i^2 = \sum_{j=1}^{p} K_{ij}^2 \]

If the natural frequencies of a set of \( N \) modes are very close to each other, the tuning of the RL shunt can be based on the average frequency \( \omega = \sum \omega_i / N \). The tuning of the resistor requires the knowledge of the electromechanical coupling factors, but the effectiveness of the RL shunt is less sensitive to the tuning of the resistor. Therefore, the resistance can be selected according to:

\[ R = \sqrt{\frac{L}{C}} \left( \frac{1}{N} \sum_{i=1}^{N} K_i^2 \right) \]

All the \( p \) RL shunt circuits have the same tuning.

##### Parallel Loops

When targeting a specific mode with \( n \) nodal diameters, the mode shape can be exploited to optimize the arrangement of PZT patches. The sine and cosine mode shapes in the circumferential direction have antinodes and nodes that can be aligned with the electrodes on the PZT patches. If the electrodes are designed such that there are 4 independent electrodes per nodal diameter, the curvature of the patches can maintain the same sign over the electrodes, allowing for effective charge production.

By mounting the patches in pairs with inverted polarization and connecting them in two independent loops of \( 2n \) patches each, the overall inductance requirement is reduced by \( 4n^2 \) compared to independent loops. The inductance \( L \) of one loop becomes:

\[ L = \frac{1}{4n^2 \omega_i^2 C} \]

The optimal resistance \( R \) for the parallel loops is given by:

\[ R = \sqrt{\frac{L}{2nC}} K_i \]

where \( K_i \) is the effective electromechanical coupling factor of a single patch.

### Experimental Validation

#### Experiment on a Circular Plate

An experimental setup was used to compare the performance of independent RL shunt and parallel shunt configurations on a circular plate. The plate was equipped with 12 PZT patches, and synthetic inductors were used to provide the required inductance. The frequency response functions (FRFs) between a point force and the plate velocity were measured using a laser vibrometer.

The results showed that both the independent RL shunt and the parallel shunt configurations exhibited similar damping performance for the first mode with \( n = 3 \) nodal diameters. This demonstrated the feasibility of the parallel shunt configuration in reducing the inductance requirement while maintaining effective damping.

#### RL Shunt Damping of a Bladed Drum

A monobloc bladed drum with 76 blades was used to further validate the method. The drum was fabricated in a single piece, resulting in extremely low natural damping (\( \xi \approx 10^{-4} \)). To avoid interference with the aerodynamics, 28 PZT patches operating in \( d_{31} \) mode were glued to the inner side of the blade support rim.

A finite element model was developed to simulate the behavior of the drum, and the PZT patches were modeled using specific piezoelectric elements. The drum was excited using 38 noncontact voice coil actuators, and the velocity of the blade tips was measured using a 2D Polytec laser scanner.

The independent RL shunt configuration, using 28 synthetic inductors, was able to damp most of the 1F modes, but the inductance requirement of 0.7 H for each inductor was impractical for passive components. The parallel shunt configuration, using 28 PZT patches connected in two independent loops of 14 patches each, reduced the inductance requirement to 50 mH. The results showed that the parallel shunt configuration was nearly as effective as the independent shunt for the targeted mode 1F7, despite the presence of blade mistuning.

### Practical Implementation

The method for damping a specific mode with \( n \) nodal diameters using parallel loops of PZT patches can be implemented in various turbomachinery applications. The key steps include:

1. **Identifying the Critical Mode**: Analyze the natural frequency versus nodal diameters plot to identify the critical mode with \( n \) nodal diameters that is likely to be excited.
2. **Designing the PZT Patch Layout**: Arrange \( 4n \) PZT patches in two parallel loops of \( 2n \) patches each, ensuring that the electrodes are designed to align with the mode shape.
3. **Selecting the Inductance and Resistance**: Calculate the required inductance and resistance based on the natural frequency and effective electromechanical coupling factor of the targeted mode.
4. **Implementing the Electrical Circuit**: Connect the PZT patches to the inductors and resistors to form the parallel shunt configuration.
5. **Testing and Validation**: Perform experimental testing to validate the performance of the parallel shunt configuration and ensure effective damping of the targeted mode.

This method offers a practical and effective solution for enhancing the damping of critical modes in turbomachinery, particularly in applications where passive components are preferred over active components. The reduced inductance requirement and simplified electrical circuitry make it suitable for use in rotating machines, improving their reliability and performance.