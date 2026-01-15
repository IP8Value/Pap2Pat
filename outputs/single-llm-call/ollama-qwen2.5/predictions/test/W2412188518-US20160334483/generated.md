# Patent Application for Gradient Modulation in PETRA Technique

## Background of the Invention

The present invention relates to a method and system for improving image quality in Magnetic Resonance Imaging (MRI) using the Proton Electron Transition Radiation Acquisition (PETRA) technique. Specifically, it addresses the issue of off-resonance blurring and fast T2* signal decay by introducing gradient modulation.

## Summary of the Invention

The invention introduces a method to enhance PETRA imaging by modulating the readout gradients. This modulation allows for higher readout bandwidths while maintaining lower excitation bandwidths, thereby reducing image blurring artifacts caused by off-resonance and T2* signal decay. The technique also mitigates limitations related to Specific Absorption Rate (SAR) and Radio Frequency (RF) peak power.

## Detailed Description of the Invention

### Introduction to PETRA Technique

PETRA is an advanced MRI technique that provides high-resolution images with ultrashort echo times. It is particularly useful for imaging tissues with short T2* relaxation times, such as bone and cartilage. However, PETRA often suffers from image blurring due to off-resonance effects and fast signal decay.

### Problem Statement

Off-resonance blurring in PETRA images can be caused by chemical shifts and magnetic susceptibility differences. Fast T2* signal decay further exacerbates this issue, leading to poor image quality. Additionally, increasing the excitation bandwidth to reduce these artifacts is limited by SAR and RF peak power constraints.

### Solution: Gradient Modulation

Gradient modulation in PETRA involves dynamically adjusting the readout gradients during data acquisition. This allows for higher readout bandwidths while keeping the excitation bandwidth lower. The method reduces off-resonance blurring and preserves short T2* signals, improving overall image quality.

### Methodology

1. **Data Acquisition**: During PETRA imaging, gradient modulation is applied to the readout gradients.
2. **Bandwidth Adjustment**: The readout bandwidth is increased while maintaining a lower excitation bandwidth.
3. **Image Reconstruction**: Density-corrected radial k-space data are reconstructed using non-uniform fast Fourier transform (NUFFT).

### Advantages of Gradient Modulation

1. **Reduced Off-Resonance Blurring**: Higher readout bandwidths minimize artifacts caused by chemical shifts and susceptibility differences.
2. **Preservation of Short T2* Signals**: Fast sampling around the k-space center helps retain signals from tissues with short relaxation times.
3. **SAR and RF Peak Power Mitigation**: Lower excitation bandwidths reduce SAR and RF peak power, making the technique more feasible for clinical use.

### Experimental Validation

1. **Simulation Studies**: Simulations demonstrated improved edge sharpness and reduced off-resonance artifacts as the readout bandwidth increased.
2. **Phantom Experiments**: Phantom images showed enhanced visualization of small structures and reduced blurring with gradient modulation.
3. **In Vivo Imaging**: Inner ear imaging in a healthy subject confirmed better visualization of complex structures compared to conventional PETRA and 3D GRE techniques.

### Example Application: Breast Phantom

Breast phantom images with PETRA and GM-PETRA showed significant improvements in visualizing small gaps between fat compartments and reducing blurring around water compartments as the bandwidth increased.

### Example Application: Apple Imaging

Apple imaging using GM-PETRA revealed finer structures in the mesocarp (flesh) that were almost invisible with 3D GRE due to their extremely fast T2* signal decay. The PETRA image was visually blurry compared to GM-PETRA because of off-resonance and susceptibility differences.

### Example Application: Inner Ear Imaging

Inner ear imaging using GM-PETRA showed improved visualization of small structures, such as the cochlea and semicircular canals, which were difficult to see with conventional PETRA due to strong susceptibility differences at air-tissue interfaces.

### Discussion

Gradient modulation in PETRA provides a flexible approach to setting excitation and readout bandwidths. This flexibility helps mitigate SAR and RF peak power limitations while improving image quality. The technique is particularly useful for imaging tissues with short T2* relaxation times and high susceptibility differences.

### Tradeoffs

1. **Acoustic Noise**: Gradient modulation can increase acoustic noise compared to the original PETRA sequence.
2. **Gradient Performance Sensitivity**: Higher bandwidths make the sequence more sensitive to gradient performance issues, such as group delays and eddy currents.

### Future Directions

Future research could explore combining gradient modulation with frequency-modulated pulse excitation to further alleviate limitations and improve image quality. Additionally, advanced reconstruction methods, such as compressed sensing, can be used to enhance image quality and accelerate data acquisition.

## Conclusion

GM-PETRA significantly reduces off-resonance blurring and fast T2* signal decay compared to conventional PETRA techniques. By modulating the readout gradients, it improves image quality and mitigates SAR and RF peak power limitations without requiring specific hardware modifications on clinical scanners.

### Acknowledgments

This study was supported by National Institutes of Health grants P41EB015894 and S10RR026783 and the W.M. Keck Foundation.