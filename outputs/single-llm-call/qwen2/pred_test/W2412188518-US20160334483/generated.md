# DESCRIPTION

## BACKGROUND OF THE INVENTION

Magnetic Resonance Imaging (MRI) is a non-invasive diagnostic tool widely used in medical imaging. Radial ultrashort echo time (UTE) MRI sequences, such as Zero Echo Time (ZTE) and Sweep Imaging with Fourier Transformation (SWIFT), are particularly useful for imaging tissues with extremely short T2* relaxation times. However, these sequences often suffer from image blurring due to off-resonance effects and rapid signal decay. Higher static magnetic field strengths exacerbate these issues due to increased chemical shifts and accelerated T2* signal decay.

One approach to mitigate image blurring is to use a higher readout bandwidth. However, this comes with significant challenges, particularly in clinical settings. For instance, in ZTE, a higher readout bandwidth necessitates a higher excitation bandwidth, leading to increased Specific Absorption Rate (SAR) and Radio Frequency (RF) pulse peak power, which are major limitations for human applications. Additionally, the slow transmit/receive (T/R) switching of most standard clinical MRI scanners results in missing critical data points around the center of k-space, further degrading image quality.

Pointwise-encoding time reduction with radial acquisition (PETRA) is a recent technique that combines ZTE and Single-Point Imaging (SPI) to overcome the T/R switching limitation. While PETRA improves image quality, it still faces challenges when using higher bandwidths, such as a larger missing region around the k-space center and increased acquisition time.

## SUMMARY OF THE INVENTION

The present invention introduces a novel PETRA technique with gradient modulation (GM-PETRA) that enables high readout bandwidths while maintaining a relatively low excitation bandwidth. GM-PETRA significantly reduces SAR and RF peak power and minimizes the missing center k-space region. This invention addresses the limitations of conventional PETRA and ZTE techniques, providing improved image quality and reduced image blurring due to off-resonance and fast T2* signal decay.

The key features of GM-PETRA include:
1. **Gradient Modulation**: After the receiver gate is turned on, the gradient amplitude ramps up from the excitation gradient to a higher readout gradient, followed by a short delay time for acquiring missing k-space points.
2. **Reduced SAR and RF Peak Power**: By using a lower excitation bandwidth, GM-PETRA reduces the SAR and RF peak power, making it more suitable for clinical applications.
3. **Improved Image Quality**: GM-PETRA effectively reduces image blurring caused by off-resonance and fast T2* signal decay, leading to clearer and more detailed images.

## DETAILED DESCRIPTION OF THE INVENTION

### Overview of GM-PETRA

In traditional PETRA, signal acquisition begins immediately after a hard pulse excitation and a T/R switching delay using gradients with constant amplitude. This results in a missing region around the k-space center, which is subsequently filled using SPI. However, using a higher readout bandwidth increases the number of missing k-space center points, leading to longer acquisition times and reduced image quality.

GM-PETRA overcomes these limitations by introducing gradient modulation. After the receiver gate is turned on, the gradient amplitude starts ramping up from the excitation gradient to a higher readout gradient, followed by a short delay time for acquiring missing k-space points. This approach allows for faster k-space sampling while maintaining a lower excitation bandwidth, thereby reducing SAR and RF peak power.

### Gradient Modulation in GM-PETRA

In GM-PETRA, the gradient amplitude is modulated to achieve a higher readout bandwidth while keeping the excitation bandwidth relatively low. The process involves the following steps:
1. **Excitation Pulse**: A hard pulse is applied to excite the spins.
2. **T/R Switching Delay**: A short delay is introduced to allow for T/R switching.
3. **Gradient Ramp-Up**: The gradient amplitude is gradually increased from the excitation gradient (G_ex) to the readout gradient (G_max).
4. **Short Delay Time**: A short delay time (t_d) is inserted to acquire missing k-space points using SPI.
5. **Data Acquisition**: The k-space is sampled more quickly due to the higher readout bandwidth, reducing the overall acquisition time.

By increasing the gradient amplitude (readout bandwidth), k-space is sampled more rapidly, which helps to reduce image blurring caused by off-resonance and fast T2* signal decay. Additionally, the lower excitation bandwidth reduces the amount of RF peak power and SAR, making GM-PETRA more suitable for clinical applications.

### Simulation and Experimental Validation

To evaluate the effectiveness of GM-PETRA, numerical simulations and experimental tests were conducted.

#### Simulation

Numerical simulations were performed using a 3D Shepp-Logan phantom containing four compartments: off-resonance, on-resonance, and two short T2* values. The k-space signal was calculated to assess the impact of off-resonance and T2* signal decay on image quality. The results showed that off-resonance blurring improved as the bandwidth increased. Specifically, GM-PETRA with a 60-125 kHz gradient modulation showed comparable off-resonance artifacts to PETRA with a 120 kHz excitation. Further improvements were observed with a 60-200 kHz gradient modulation. For short T2* spins, increasing the bandwidth improved edge sharpness and preserved short T2* signals.

#### Experiments

Experiments were conducted using a 7 T MRI scanner with a head coil. The following tests were performed:
1. **Breast Phantom**: A breast phantom with fat and water compartments was imaged to evaluate off-resonance blurring. The results showed that GM-PETRA significantly reduced off-resonance artifacts compared to PETRA.
2. **Apple Imaging**: An apple was imaged to assess the sensitivity to short T2* signals. GM-PETRA demonstrated higher sensitivity to signals from the mesocarp (flesh) and improved visualization of fine structures compared to PETRA and 3D GRE.
3. **Inner Ear Imaging**: Inner ear imaging of a healthy subject was conducted to evaluate the performance of GM-PETRA in a clinical setting. The results showed that GM-PETRA provided better visualization of complex structures in the inner ear, such as the cochlea and semicircular canals, compared to PETRA and 3D GRE.

### Image Reconstruction

Image reconstruction was performed offline using a home-built program written in C++/CUDA. The k-space sampling density, including the effects from the ramp sampling in GM-PETRA, was compensated using iterative density correction. The density-corrected radial k-space data were then reconstructed to a 3D image using non-uniform fast Fourier transform (NUFFT).

### Advantages and Trade-offs

#### Advantages
1. **Reduced Image Blurring**: GM-PETRA effectively reduces image blurring caused by off-resonance and fast T2* signal decay, leading to clearer and more detailed images.
2. **Lower SAR and RF Peak Power**: By using a lower excitation bandwidth, GM-PETRA reduces SAR and RF peak power, making it more suitable for clinical applications.
3. **Improved Image Quality**: GM-PETRA provides better visualization of complex structures, such as those found in the inner ear, compared to conventional PETRA and 3D GRE.

#### Trade-offs
1. **Increased Acoustic Noise**: The use of gradient modulation in GM-PETRA can increase acoustic noise compared to the original PETRA sequence.
2. **Sensitivity to Gradient Performance**: As the bandwidth increases, the sequence becomes more sensitive to gradient performance, including group delays and eddy currents. Proper calibration and optimization are necessary to maintain image quality.

### Conclusion

GM-PETRA represents a significant advancement in radial UTE MRI techniques. By enabling high readout bandwidths while maintaining a relatively low excitation bandwidth, GM-PETRA reduces image blurring, lowers SAR and RF peak power, and improves image quality. This invention has the potential to enhance the diagnostic capabilities of MRI, particularly in applications involving tissues with extremely short T2* relaxation times.