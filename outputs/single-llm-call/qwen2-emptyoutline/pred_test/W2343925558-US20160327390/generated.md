# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to micromachined gyroscopes, specifically to a high-frequency bulk-acoustic wave (BAW) gyroscope with a stress isolation system that significantly reduces anchor loss and improves environmental performance. This gyroscope is particularly useful for applications requiring high precision and robustness against environmental disturbances, such as consumer electronics, automotive safety systems, and industrial navigation.

## BACKGROUND OF THE INVENTION

Micromachined gyroscopes have revolutionized various fields, from consumer electronics to automotive safety systems, by enabling precise motion detection and control. Traditional low-frequency flexural tuning-fork gyroscopes (TFGs) are widely used but suffer from significant limitations. These devices are often sensitive to random vibrations and linear accelerations, leading to long-term drift and reduced accuracy. Moreover, recent concerns about the high sensitivity of consumer-grade gyroscopes to low-frequency pressure signals have raised security issues related to potential eavesdropping.

To address these challenges, high-frequency gyroscopes based on bulk-acoustic wave (BAW) resonators have been developed. These devices operate in the MHz range and exhibit high quality factors at moderate vacuum levels, making them inherently resistant to environmental vibrations and shocks. However, the performance of BAW gyroscopes can be compromised by damping mismatches between the degenerate modes, which can lead to environmentally dependent offset variations.

The present invention introduces a substrate-decoupled (SD) BAW gyroscope that incorporates a stress isolation system to significantly reduce anchor loss and improve environmental performance. This innovation enhances the robustness and reliability of the gyroscope, making it suitable for high-volume production and a wide range of applications.

## SUMMARY OF THE INVENTION

The present invention provides a high-frequency bulk-acoustic wave (BAW) gyroscope with a stress isolation system that reduces anchor loss and improves environmental performance. The gyroscope comprises an active resonator region anchored through a stress isolation system, which effectively decouples the resonator from its substrate at the resonance frequency of the modes of vibration. Electrodes with ultra-narrow capacitive gaps surround the structure to enable electrostatic excitation, readout, and frequency tuning.

Key features of the invention include:
1. **Stress Isolation System**: The stress isolation system minimizes anchor loss by attenuating the strain induced on the center anchor, thereby reducing the energy dissipated into the substrate.
2. **High-Frequency Operation**: The gyroscope operates in the MHz range, inherently rejecting the effects of random vibrations and shocks.
3. **Mode Matching**: The degenerate modes of the gyroscope can be precisely matched using electrostatic tuning electrodes, ensuring optimal performance.
4. **Environmental Robustness**: The design significantly reduces the impact of environmental disturbances, such as temperature variations, vibrations, and shocks, on the gyroscope's performance.

The invention is particularly useful for applications requiring high precision and robustness, such as consumer electronics, automotive safety systems, and industrial navigation.

## DETAILED DESCRIPTION

### Mode-to-Mode Coupling in Vibratory Gyroscopes

Vibratory gyroscopes, including the substrate-decoupled (SD) BAW gyroscope of the present invention, rely on the principles of mode-to-mode coupling to detect rotation. These devices use the degenerate modes of a vibrating structure to sense angular velocity. The behavior of an SD-BAW gyroscope is described by two orthogonal, second-order systems coupled to each other by a Coriolis force proportional to the applied angular velocity.

The equations of motion for the two degenerate modes can be expressed as:
\[
m_{11}\ddot{q}_1(t) + b_{11}\dot{q}_1(t) + b_{12}\dot{q}_2(t) + k_{11}q_1(t) + k_{12}q_2(t) = \sum_{i=1}^{k} F_{1,i} - 2\lambda m_{22}\Omega(t)\dot{q}_2(t)
\]
\[
m_{22}\ddot{q}_2(t) + b_{22}\dot{q}_2(t) + b_{21}\dot{q}_1(t) + k_{22}q_2(t) + k_{21}q_1(t) = \sum_{i=1}^{k} F_{2,i} + 2\lambda m_{11}\Omega(t)\dot{q}_1(t)
\]

In these equations:
- \( \Omega(t) \) is the rate of rotation applied around an axis normal to the plane of modal vibration.
- \( \lambda \) is the angular gain of the gyroscope, determined by the device's geometry and mode shape.
- \( m_{11}, k_{11}, b_{11} \) are the effective mass, stiffness, and damping associated with mode 1, respectively.
- \( m_{22}, k_{22}, b_{22} \) are the corresponding parameters for mode 2.
- \( k_{12}, k_{21} \) are the stiffness-coupling terms.
- \( b_{12}, b_{21} \) are the damping-coupling coefficients.
- \( F_{1,i}, F_{2,j} \) are the electrostatic forces acting on modes 1 and 2, respectively.
- \( q_1(t), q_2(t) \) are the generalized coordinates of modes 1 and 2, respectively.

In an ideal axis-symmetric gyroscope, the properties of both modes are identical, leading to equal resonance frequencies. However, material and fabrication imperfections can cause differences in the elastic and inertial characteristics of the degenerate modes, resulting in a frequency split. Additionally, structural imperfections can produce undesired mode-to-mode coupling, which can be corrected using electrostatic spring softening and mode-decoupling electrodes.

The SD-BAW gyroscope of the present invention incorporates a stress isolation system to minimize anchor loss, ensuring that the damping mechanisms affecting the two modes are as symmetric as possible. This design significantly reduces the impact of environmental disturbances on the gyroscope's performance, making it highly robust and reliable for a wide range of applications.