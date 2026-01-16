# DESCRIPTION

## FIELD OF THE DISCLOSURE

The present disclosure relates to a method and system for determining the time at which a blood pool was formed on a crime scene. More specifically, the disclosure provides a method for calculating the time of formation of a blood pool based on the evaporation dynamics of the blood pool, which can be monitored using the drying front of the pool. The method involves analyzing the wet area of the blood pool over time and correlating this with the evaporation rate to estimate the time of formation. This method can provide crucial information for forensic investigations, particularly in scenarios where the presence of a body is not necessary to estimate the time of the event.

## TECHNICAL BACKGROUND

Bloodstain pattern analysis (BPA) is a forensic specialty dedicated to the analysis of blood traces found at crime scenes. While some initial observations can be made from the appearance of blood pools, the physics governing the drying processes of blood pools is not well understood. Current methods for estimating the time of events at a crime scene, such as body temperature, rigor mortis, and forensic entomology, require the presence of a body. However, in cases where no body is present, the ability to estimate the time of formation of a blood pool becomes crucial. Additionally, this information can complement other available data to construct a more comprehensive timeline of events.

Prior research has focused on the volume determination of blood pools for reconstruction purposes and to assess whether the blood loss could be a cause of death. However, the temporal aspect of when a blood pool was formed has not been adequately addressed. The present disclosure aims to fill this gap by providing a method to predict the time of formation of a blood pool with an accuracy of ±30 minutes. This method leverages the understanding of the evaporation dynamics of blood pools, which have been shown to exhibit distinct stages of drying.

## SUMMARY OF THE DISCLOSURE

The present disclosure provides a method for determining the time at which a blood pool was formed on a crime scene. The method includes the following steps:

1. **Image Capture**: Capturing images of the blood pool at regular intervals using a camera. The images should capture the drying front of the pool, which is the transition edge between the wet and dry areas of the blood.

2. **Initial Measurements**: Measuring the initial total area (\(A_i\)) and perimeter (\(P\)) of the blood pool from the captured images. Estimating the initial height (\(h\)) of the pool based on average values obtained from previous experiments.

3. **Wet Area Analysis**: Analyzing the images to determine the wet area (\(A_x\)) of the blood pool at each time point. The wet area is the area of the pool that is still in a liquid state.

4. **Mass Calculation**: Calculating the initial mass (\(m_i\)) of the blood pool using the formula:
   \[
   m_i = h A_i \rho
   \]
   where \(\rho\) is the density of blood.

5. **Mass Variation**: Determining the mass of the pool at each time point (\(m_x\)) using the relationship between the wet area and the mass of the pool. This relationship can be approximated by the function:
   \[
   \frac{m}{m_i} = 1 - \alpha \left[ 1 - \left( \frac{A_x}{A_i} \right) \right]^\beta
   \]
   where \(\alpha = 0.78\) and \(\beta = 0.16\).

6. **Evaporation Rate Calculation**: Calculating the evaporation rate (\(J^*\)) of water from the blood pool using the diffusion coefficient (\(D_{\text{blood}}\)) and the shape factor (\(L^*\)):
   \[
   J^* = D_{\text{blood}} \frac{M P_w}{L_k R T L^{*1/2}}
   \]
   where \(D_{\text{blood}} \approx 1.1 \times 10^{-9} \, \text{m}^2/\text{s}\), \(M\) is the molar mass of water, \(P_w\) is the saturation vapor pressure of water, \(L_k\) is the Knudsen layer, \(R\) is the universal gas constant, \(T\) is the temperature, and \(L^* = \frac{A_i}{h P}\).

7. **Time Calculation**: Estimating the time at which the pool was formed (\(t_x\)) using the mass variation and the evaporation rate:
   \[
   t_x = \frac{\alpha R k_B T^2 A_i^{1/2} h^{1/2} \rho \left[ 1 - \left( \frac{A_x}{A_i} \right) \right]^\beta}{M d^2 \pi P^{1/2} D_{\text{blood}} P_w P_a}
   \]
   where \(k_B\) is Boltzmann's constant, \(d\) is the molecular diameter, and \(P_a\) is the atmospheric pressure.

The disclosed method provides a reliable and accurate way to estimate the time of formation of a blood pool, which can be a valuable tool in forensic investigations. The method can be applied in various scenarios, including crime scenes where the presence of a body is not required to estimate the time of the event.

## DETAILED DESCRIPTION OF EMBODIMENTS

### Image Capture

The first step in the method involves capturing images of the blood pool at regular intervals. The images should be taken using a high-resolution camera and should capture the drying front of the pool, which is the transition edge between the wet and dry areas of the blood. The camera should be positioned directly above the pool to ensure that the entire pool is visible in each image. The images should be taken at intervals of 5 to 10 minutes to capture the progression of the drying front.

### Initial Measurements

From the captured images, the initial total area (\(A_i\)) and perimeter (\(P\)) of the blood pool are measured. The initial height (\(h\)) of the pool can be estimated based on average values obtained from previous experiments. For example, the average height of a blood pool on a white tile surface can be approximately 1.44 mm.

### Wet Area Analysis

The images are analyzed to determine the wet area (\(A_x\)) of the blood pool at each time point. The wet area is the area of the pool that is still in a liquid state. This can be done using image processing software, which can distinguish between the wet and dry areas based on color and texture differences.

### Mass Calculation

The initial mass (\(m_i\)) of the blood pool is calculated using the formula:
\[
m_i = h A_i \rho
\]
where \(\rho\) is the density of blood, which is typically around 1.05 g/cm³.

### Mass Variation

The mass of the pool at each time point (\(m_x\)) is determined using the relationship between the wet area and the mass of the pool. This relationship can be approximated by the function:
\[
\frac{m}{m_i} = 1 - \alpha \left[ 1 - \left( \frac{A_x}{A_i} \right) \right]^\beta
\]
where \(\alpha = 0.78\) and \(\beta = 0.16\). This function provides an approximate relation between the wet area and the mass of the pool.

### Evaporation Rate Calculation

The evaporation rate (\(J^*\)) of water from the blood pool is calculated using the diffusion coefficient (\(D_{\text{blood}}\)) and the shape factor (\(L^*\)):
\[
J^* = D_{\text{blood}} \frac{M P_w}{L_k R T L^{*1/2}}
\]
where \(D_{\text{blood}} \approx 1.1 \times 10^{-9} \, \text{m}^2/\text{s}\), \(M\) is the molar mass of water (18 g/mol), \(P_w\) is the saturation vapor pressure of water, \(L_k\) is the Knudsen layer, \(R\) is the universal gas constant (8.314 J/(mol·K)), \(T\) is the temperature in Kelvin, and \(L^* = \frac{A_i}{h P}\).

### Time Calculation

The time at which the pool was formed (\(t_x\)) is estimated using the mass variation and the evaporation rate:
\[
t_x = \frac{\alpha R k_B T^2 A_i^{1/2} h^{1/2} \rho \left[ 1 - \left( \frac{A_x}{A_i} \right) \right]^\beta}{M d^2 \pi P^{1/2} D_{\text{blood}} P_w P_a}
\]
where \(k_B\) is Boltzmann's constant (1.38 × 10⁻²³ J/K), \(d\) is the molecular diameter of water (2.82 × 10⁻¹⁰ m), and \(P_a\) is the atmospheric pressure (101325 Pa).

### Practical Application

The disclosed method can be applied in various forensic scenarios to estimate the time of formation of a blood pool. For example, if a blood pool is found on a crime scene and the drying front is visible, the method can be used to calculate the time at which the pool was formed. This information can be crucial for constructing a timeline of events and can help in identifying the sequence of actions that led to the crime.

### Example Calculation

Consider a blood pool dried at 22.5°C on a white tile surface. The initial total area (\(A_i\)) is 100 cm², the perimeter (\(P\)) is 30 cm, and the initial height (\(h\)) is 1.44 mm. The density of blood (\(\rho\)) is 1.05 g/cm³. The initial mass (\(m_i\)) is calculated as:
\[
m_i = 1.44 \, \text{mm} \times 100 \, \text{cm}^2 \times 1.05 \, \text{g/cm}^3 = 151.2 \, \text{g}
\]

If the wet area (\(A_x\)) at a certain time is 70 cm², the mass at that time (\(m_x\)) can be calculated using the relationship:
\[
\frac{m_x}{151.2} = 1 - 0.78 \left[ 1 - \left( \frac{70}{100} \right) \right]^{0.16}
\]
\[
\frac{m_x}{151.2} = 1 - 0.78 \left[ 1 - 0.3 \right]^{0.16}
\]
\[
\frac{m_x}{151.2} = 1 - 0.78 \times 0.89
\]
\[
\frac{m_x}{151.2} = 1 - 0.6942
\]
\[
\frac{m_x}{151.2} = 0.3058
\]
\[
m_x = 0.3058 \times 151.2 = 46.2 \, \text{g}
\]

The mass variation (\(\delta m\)) is:
\[
\delta m = 151.2 - 46.2 = 105 \, \text{g}
\]

The evaporation rate (\(J^*\)) is calculated as:
\[
J^* = 1.1 \times 10^{-9} \, \text{m}^2/\text{s} \times \frac{18 \, \text{g/mol} \times 3167.1 \, \text{Pa}}{2.82 \times 10^{-10} \, \text{m} \times 8.314 \, \text{J/(mol·K)} \times 295.65 \, \text{K} \times \left( \frac{100 \, \text{cm}^2}{1.44 \, \text{mm} \times 30 \, \text{cm}} \right)^{1/2}}
\]
\[
J^* = 1.1 \times 10^{-9} \, \text{m}^2/\text{s} \times \frac{57007.8 \, \text{g/(m·s)}}{0.0000232 \, \text{m}^{3/2}}
\]
\[
J^* = 1.1 \times 10^{-9} \, \text{m}^2/\text{s} \times 2.457 \times 10^{9} \, \text{g/(m·s)}
\]
\[
J^* = 2.703 \, \text{g/(m·s)}
\]

The time at which the pool was formed (\(t_x\)) is:
\[
t_x = \frac{105 \, \text{g}}{2.703 \, \text{g/(m·s)} \times 100 \, \text{cm}^2}
\]
\[
t_x = \frac{105 \, \text{g}}{270.3 \, \text{g/s}}
\]
\[
t_x = 0.388 \, \text{s}
\]

Converting seconds to hours:
\[
t_x = 0.388 \, \text{s} \times \frac{1 \, \text{hour}}{3600 \, \text{s}} = 0.000108 \, \text{hours}
\]

This example demonstrates the detailed calculations involved in the method, which can be applied to various blood pools to estimate the time of formation.

### Conclusion

The disclosed method provides a robust and accurate way to estimate the time of formation of a blood pool on a crime scene. By leveraging the evaporation dynamics of blood pools and the analysis of the drying front, the method can provide crucial information for forensic investigations. The method is particularly useful in scenarios where the presence of a body is not required to estimate the time of the event, making it a valuable tool for crime scene analysis. Further research and development can enhance the precision and applicability of the method, potentially leading to its widespread adoption in forensic science.