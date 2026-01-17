# DESCRIPTION

## BACKGROUND OF THE INVENTION

### Technical Field

The present invention relates to a Spin Current Magnetization Rotational Element (SC-MRE) and, more specifically, to a three-terminal Spin Orbit Torque (SOT)-Magnetic Random Access Memory (MRAM) device designed for high write endurance and low power consumption. The invention is particularly useful in the context of edge devices and Internet of Things (IoT) applications where low power consumption and high reliability are critical.

### Description of Related Art

The rapid development and widespread adoption of the Internet of Things (IoT) have necessitated the creation of edge devices with significantly reduced power consumption. Traditional Complementary Metal-Oxide-Semiconductor (CMOS) technology, while widely used, faces significant challenges in terms of power consumption, especially when deployed in edge-AI devices that require extensive machine learning and product-sum operations. Spin-Transfer-Torque (STT)-Magnetic Random Access Memory (MRAM) has emerged as a promising alternative due to its non-volatility and potential for low power consumption. However, achieving high write endurance and low write current remains a significant challenge.

Embedded perpendicular STT-MRAM has shown promise in replacing embedded flash memory, but to fully integrate it into advanced systems such as Last Level Cache (LLC) and Static Random Access Memory (SRAM), it must exhibit high performance with write endurance exceeding \(10^{12}\) cycles and low write current. Unfortunately, the write current in perpendicular STT-MRAM increases with decreasing write pulse width, leading to stress and damage on the tunnel barrier layer of Magnetic Tunnel Junctions (MTJs).

To address these limitations, three-terminal SOT-MRAM has been proposed. SOT effects, including the spin Hall effect and the Rashba-Edelstein effect, enable rapid magnetization switching of the free layer using in-plane current through a SOT effective layer, such as heavy metals. This approach aims to reduce stress on the MTJ tunnel barrier and achieve high write endurance. Despite theoretical advantages, experimental verification of high endurance and low damage to the tunnel barrier has been lacking.

## SUMMARY OF THE INVENTION

### SUMMARY OF THE INVENTION

The present invention provides a Spin Current Magnetization Rotational Element (SC-MRE) that demonstrates high write endurance of \(10^{12}\) cycles. The SC-MRE utilizes the Spin Orbit Torque (SOT) effect to switch the magnetization of the free layer in a Magnetic Tunnel Junction (MTJ) without causing significant damage to the tunnel barrier. The invention includes a three-terminal device configuration with a tungsten (W) SOT line, which allows for efficient magnetization switching with low write current and high endurance.

The key features of the invention include:
1. **High Write Endurance**: The SC-MRE achieves write endurance of \(10^{12}\) cycles, making it suitable for high-performance memory applications.
2. **Low Write Current**: The use of the SOT effect enables magnetization switching with low write current, reducing power consumption and stress on the MTJ.
3. **Three-Terminal Configuration**: The device design includes a tungsten SOT line, which minimizes damage to the tunnel barrier and enhances reliability.
4. **Non-Volatility**: The SC-MRE retains its data even when power is removed, making it ideal for edge devices and IoT applications.

## DETAILED DESCRIPTION OF THE INVENTION

### Spin Current Magnetization Rotational Element

The Spin Current Magnetization Rotational Element (SC-MRE) is a three-terminal device that leverages the Spin Orbit Torque (SOT) effect to achieve high write endurance and low power consumption. The device consists of a Magnetic Tunnel Junction (MTJ) with a tungsten (W) SOT line, which facilitates rapid and efficient magnetization switching of the free layer.

#### Device Structure

The SC-MRE is fabricated using a top-pinned type MTJ with a tungsten layer as the SOT line. The stacking layer includes:
- **Substrate**: Thermal oxide Si
- **SOT Line**: Tungsten (W)
- **Free Layer**: CoFeB-based
- **Tunnel Barrier**: Magnesium Oxide (MgO)
- **Pinned Layer**: CoFeB-based Synthetic Antiferromagnetic (SAF) layer
- **Antiferromagnetic Layer**: IrMn
- **Capping Layer**: Protective material

The dimensions of the W-SOT line are as follows:
- **Thickness**: 3 nm
- **Length**: 700 nm
- **Width**: 360 nm

The shape of the MTJ is elliptical, measuring 120 nm by 360 nm. The direction of magnetization of the free layer is in-plane and orthogonal to the direction of the write current through the SOT line. This configuration is achieved through the use of a thick free layer and shape anisotropy, allowing for magnetization switching without an external magnetic field.

#### Fabrication Process

The SC-MRE is fabricated using a sputter method without breaking vacuum. The W-SOT line is deposited with high resistivity (approximately 400 µΩ⋅cm) to optimize SOT efficiency. The MTJ stack is then patterned using standard lithographic techniques to form the elliptical shape.

#### Operation and Performance

The SC-MRE operates by injecting a pulse current through the W-SOT line to switch the magnetization of the free layer. The pulse current is generated using a pulse generator, and the MTJ resistance is measured using DC voltage. The write pulse width can vary from 1 µs to 1 ns, and the threshold current density for magnetization switching is determined by averaging multiple switching events.

Experimental results demonstrate that the SC-MRE can achieve high write endurance of \(10^{12}\) cycles. The MTJ resistance shows minimal drift over the write cycles, maintaining a Tunneling Magnetoresistance (TMR) ratio of approximately 90%. However, some write errors occur at lower current densities, which can be mitigated by increasing the write current density to 8.3 × 10^7 A/cm². Despite this, higher write current densities can lead to electrical failures, such as electrical-open and electrical-short conditions, primarily due to Joule heating in the W-SOT line.

To further improve reliability, the resistivity of the W-SOT line and its length can be optimized. Additionally, the use of alternative materials with lower resistivity may help reduce Joule heating and enhance write endurance.

#### Array Implementation

A small SC-MRE array of 8 × 8 elements was fabricated to evaluate the performance and reliability of the device in a more practical setting. The array includes a write/read analog circuit controlled by a Field-Programmable Gate Array (FPGA) to measure the speed and duration of 64 MTJ elements. The write/read circuit consists of selectors, a write-pulse generating circuit, and a read circuit.

Accelerated tests were conducted to assess the failure modes of the SC-MRE array. The results show that the number of failure elements increases with higher write current densities, consistent with the behavior observed in single MTJ elements. Cross-sectional Transmission Electron Microscopy (TEM) images reveal that electrical-open failures are associated with the bursting of the W-SOT line, while electrical-short failures are caused by metal bridges from the W-SOT line.

#### Conclusion

The Spin Current Magnetization Rotational Element (SC-MRE) represents a significant advancement in the field of low-power, high-endurance memory devices. By utilizing the SOT effect, the SC-MRE achieves write endurance of \(10^{12}\) cycles with minimal damage to the MTJ tunnel barrier. The three-terminal configuration with a tungsten SOT line ensures efficient magnetization switching and high reliability, making the SC-MRE an ideal candidate for edge devices and IoT applications. Further optimization of the device parameters, such as the resistivity and length of the W-SOT line, can enhance the performance and reliability of the SC-MRE, paving the way for its widespread adoption in advanced memory systems.