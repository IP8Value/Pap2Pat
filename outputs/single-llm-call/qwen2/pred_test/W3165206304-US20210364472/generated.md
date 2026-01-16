# DESCRIPTION

## BACKGROUND OF THE INVENTION

### 1. Field of the Invention

The present invention relates generally to non-destructive testing and inspection of composite materials, and more specifically to a method and system for accurately identifying and quantifying foreign object defects within carbon fiber reinforced polymer (CFRP) laminates using ultrasonic immersion c-scan techniques. The invention provides a robust and high-resolution method for detecting and sizing defects, which is particularly useful in the aerospace and automotive industries where the integrity of composite materials is critical.

### 2. Description of Related Art

Composite materials, particularly carbon fiber reinforced polymers (CFRPs), are widely used in various industries due to their high strength-to-weight ratio and anisotropic properties. However, the manufacturing process for these materials is often manual and prone to defects such as delaminations, broken fibers, inclusions, and incorrect layup. These defects can significantly reduce the service life and structural integrity of composite components, necessitating the development of reliable non-destructive testing (NDT) methods.

Ultrasound is one of the most popular NDT techniques due to its safety, portability, relatively low cost, and ease of use. Several studies have explored the use of ultrasound for detecting and characterizing defects in CFRPs. Benammar et al. used signal processing techniques to detect delaminations, while Poudel et al. employed fuzzy logic and artificial neural networks (ANNs) to identify foreign objects. Hasiotis et al. aimed to trace the shape of foreign objects using ultrasound c-scan data, but the size estimation was often inaccurate. Li et al. demonstrated an edge detection method based on the standard deviations of ultrasound data to outline delamination defects, but the accuracy was not quantitatively assessed.

Despite these advancements, there remains a need for a method that can accurately size small foreign object defects in CFRP laminates. The present invention addresses this need by providing a method that combines advanced ultrasonic data enhancement techniques with edge detection algorithms to achieve high-resolution sizing of defects.

## SUMMARY OF THE INVENTION

The present invention provides a method and system for accurately identifying and quantifying foreign object defects within carbon fiber reinforced polymer (CFRP) laminates using ultrasonic immersion c-scan techniques. The method involves the following steps:

1. **Preprocessing of A-Scans**: Normalizing the ultrasonic wave energy data to enhance the clarity of the defect signals.
2. **Enhancement of Ultrasonic Data**: Applying a Gaussian filter to smooth the ultrasonic data and reduce noise, followed by interpolation to improve the resolution of the c-scan images.
3. **Edge Detection**: Using the magnitude of the gradient to identify the edges of the foreign object defects.
4. **Sizing of Defects**: Capturing the peak of the gradient along the boundary of the defect to determine its size accurately.

The invention also includes a system for implementing the method, comprising an ultrasonic immersion c-scan setup, a data acquisition unit, and a data processing unit. The system is designed to provide high-resolution images of the defects, enabling precise sizing and characterization.

The key advantages of the present invention include:
- **High Accuracy**: The method achieves an average error of 0.11 mm in determining the diameter of foreign object defects, which is a significant improvement over existing techniques.
- **Robustness**: The method is effective for defects of various sizes and depths within the laminate.
- **Automation Potential**: The method can be automated, making it suitable for large-scale industrial applications.

## DETAILED DESCRIPTION

### 1. Foreign Object Fabrication

To facilitate a robust study, twelve Polytetrafluoroethylene (PTFE) foreign objects were fabricated using a Silhouette Cameo. A strip of PTFE measuring 0.05 mm thick was cut into twelve circles of different diameters: three with a 12.7 mm diameter, three with a 6.35 mm diameter, three with a 3.18 mm diameter, and three with a 1.59 mm diameter. The true size of the foreign objects was measured using 3D microscopy (VR-3000, Keyence, Osaka, Japan) to ensure accuracy. The measurements revealed that the fabricated samples were slightly smaller than their nominal sizes, highlighting the importance of determining the true sizes of the Teflon foreign objects before placing them within the carbon fiber laminate.

### 2. Laminate Fabrication

Twelve carbon fiber composites were fabricated with a layup of (0/30/60/0/45/0)s. During the layup process, intentional PTFE foreign objects were placed between layers 3 and 4, 6 and 7, and 9 and 10. The laminates were labeled A–D according to the size of the foreign object placed in the sample, with A corresponding to the largest, 12.7 mm, inclusion and D referring to the smallest, 1.59 mm, inclusion. Each sample was also assigned a number code, 1–3, indicating the position of the foreign object within the laminate. The laminates were fabricated using 3K, 6 oz. plain weave carbon fiber and cured using the Vacuum Assisted Resin Transfer Method (VARTM) with a Proset INF 114 resin and Proset 211 hardener mixture. After curing, the samples were cut to a nominal size of 50.8 mm × 50.8 mm.

### 3. Scanning Setup

All twelve samples were scanned from the tool-side using a custom ultrasonic immersion c-scan system. Each scan covered an area of 38.1 mm × 38.1 mm in a raster pattern with a spacing of 0.2 mm between scans. The samples were leveled to ensure the transducer was normal to the surface of the sample. A 9.53 mm spherically focused 7.5 MHz Videoscan transducer (V320-SU-F1.50IN-PTF, Olympus, Tokyo, Japan) was used, driven by an ultrasonic pulser/receiver (EUT 3160, US Ultratek, Walnut Creek, CA, USA) with a 65 ns negative square wave pulse width at 200 V. The transducer was moved by two Velmex BiSlide translation stages, and a Linear Voltage Displacement Transducer (LVDT) from RDP Electronics monitored the position of the transducer. Custom LabVIEW programs controlled the c-scan system, and custom MATLAB codes analyzed the ultrasonic data.

### 4. A-Scan Preprocessing Methods

A-scans represent the normalized wave energy as a function of time and are the most basic representation of ultrasonic scan data. The a-scans were preprocessed to enhance the clarity of the defect signals. The front wall echo and back-wall echo were identified, and the signal was gated to cover three-quarters of the peak-to-peak distance between laminae. The signal was saturated at the front wall to provide better resolution of the internal features of the samples.

### 5. Enhancement of the Ultrasonic Data

To improve the ultrasonic data, a Gaussian filter was applied to smooth the data and reduce noise. The ultrasonic scan intensity was shifted in time to account for variations in the surface of the sample. The data was then smoothed using a two-dimensional Gaussian filter, and the resolution was enhanced through interpolation. The Gaussian filter muted the effects of spurious intensity, and the interpolation improved the resolution at the edges of the foreign objects.

### 6. Use of Gradient for Edge Detection

Edge detection is fundamental to image analysis. The magnitude of the gradient of the c-scan data was used to perform two-dimensional edge detection. The partial derivatives were calculated using a central difference method, and the magnitude of the gradient was evaluated at every point. The peak of the gradient along the boundary of the defect was captured to determine the size of the foreign object accurately.

### 7. Sizing of Foreign Objects

The 6 dB drop technique, a traditional sizing method, was applied to the c-scan images to demonstrate the improvement in the data. The results showed that the 6 dB drop technique had significant errors in sizing small foreign objects. The maximum gradient transition (MGT) method was then applied, which involved capturing the peak of the gradient along the boundary of the defect. The MGT method achieved an average error of 0.11 mm in determining the diameter of foreign object defects, representing a significant improvement over existing techniques.

### 8. Conclusion

The present invention provides a robust and high-resolution method for accurately identifying and quantifying foreign object defects within carbon fiber reinforced polymer (CFRP) laminates. The method combines advanced ultrasonic data enhancement techniques with edge detection algorithms to achieve high-resolution sizing of defects. The invention is particularly useful in the aerospace and automotive industries, where the integrity of composite materials is critical. Future work will focus on automating the analysis technique and studying new types of manufactured defects, including irregularly shaped defects.