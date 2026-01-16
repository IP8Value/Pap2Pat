# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to a high-accuracy image-based Sun position sensor and tracking system for use in solar energy applications, particularly in high concentration photovoltaic (HCPV) systems. The invention specifically addresses the need for improved accuracy and stability in solar tracking systems, which are crucial for maximizing the efficiency and power output of solar energy systems.

## DESCRIPTION OF THE RELATED ART

Solar power has gained significant attention in recent decades due to environmental concerns and the depletion of fossil fuels. High concentration photovoltaic (HCPV) systems, which require high accuracy dual-axis solar trackers, are particularly sensitive to the precision of the tracking mechanism. Traditional sun-tracking methods, including open-loop and closed-loop control systems, have been extensively studied and implemented. Closed-loop systems typically use Sun position sensors, such as bar-shadow photosensors and four-quadrant light sensors, to provide feedback for tracking the Sun. However, these sensors often suffer from issues such as low sensitivity under cloudy conditions and characteristic mismatches among photodiodes, leading to reduced power output.

To address these limitations, image-based Sun position sensors have been developed. These sensors use charge-coupled devices (CCDs) or complementary metal-oxide-semiconductor (CMOS) photodetectors to capture the Sun's image and determine its position. While these sensors offer improved accuracy and stability, they still face challenges in maintaining high performance under varying weather conditions and low irradiation levels.

## SUMMARY OF THE INVENTION

The present invention provides an image-based Sun position sensor and tracking system that significantly enhances the accuracy and stability of solar tracking. The system includes a self-designed reflecting Cassegrain telescope with adjustable magnification, a high-resolution webcam, and a tracking controller with an embedded image processing algorithm. The reflecting Cassegrain telescope is designed to obtain clear and enlarged Sun images, which are essential for accurate position detection. The high-resolution webcam captures these images, and the tracking controller processes the images to determine the Sun's position and control the solar tracker.

Key features of the invention include:
1. **Reflecting Cassegrain Telescope**: The telescope is designed to reflect and enlarge Sun images while maintaining a compact and lightweight structure. It includes a right-angle prism and an eyepiece to adjust the image size and clarity.
2. **High-Resolution Webcam**: The webcam captures high-quality images of the Sun, which are processed to determine the Sun's position.
3. **Image Processing Algorithm**: The algorithm uses advanced techniques such as image binarization, edge detection, and the three-point circle method to accurately determine the Sun's position and control the solar tracker.

The invention offers several advantages over existing systems:
- **High Accuracy**: The system achieves a tracking accuracy of within 0.04°, even under varying weather conditions.
- **Stability**: The system maintains high performance and stability, ensuring consistent power output from the solar energy system.
- **Versatility**: The system can be easily integrated into various solar tracking applications, including HCPV systems and conventional PV systems.

## DESCRIPTION OF THE PREFERRED EMBODIMENTS

### Reflecting Cassegrain Telescope

The reflecting Cassegrain telescope is a critical component of the invention, designed to obtain clear and enlarged Sun images. The telescope consists of two concave mirrors, a right-angle prism, and an eyepiece. The primary concave mirror reflects incoming sunlight to the secondary concave mirror, which then reflects the light to the right-angle prism. The prism changes the direction of the light and directs it to the eyepiece, where the image is focused and enlarged.

The design of the telescope is optimized using the OSLO® tool to ensure high modulation transfer function (MTF) values, indicating excellent resolving power and sharpness. The MTF analysis shows that the telescope can achieve a modulation value of over 0.8 at 20 cycles/mm, ensuring that the Sun images are clear and well-defined.

### High-Resolution Webcam

The high-resolution webcam is used to capture the Sun images produced by the reflecting Cassegrain telescope. The webcam has a resolution of 2,304 × 1,536 pixels, providing detailed and high-quality images. The combination of the 15× magnification telescope and the high-resolution webcam offers the best tracking accuracy, with a resolution of 0.0017°/pixel.

### Image Processing Algorithm

The image processing algorithm is a key component of the tracking system, responsible for accurately determining the Sun's position from the captured images. The algorithm involves several steps:

1. **Image Binarization**: The algorithm converts the captured HSL color image into a binary image by setting an appropriate threshold for lightness. This step helps to distinguish the Sun image from other objects and noise in the image.
2. **Edge Detection**: The Sobel method is used to detect the edges of the Sun image in the binary image. This step helps to identify the boundary of the Sun image.
3. **Center Calculation**: The three-point circle method is used to estimate the central coordinates of the Sun image. Three non-collinear points on the edge of the Sun image are selected, and the intersection of the perpendicular bisectors of the line segments formed by these points is calculated to determine the center of the Sun image.

The tracking controller uses the calculated central coordinates to determine whether the solar tracker is precisely aimed at the Sun. If the central coordinates deviate from the center of the photo frame, the controller sends commands to the solar tracker to adjust its position accordingly.

### Experimental Results

To validate the performance of the image-based Sun position sensor and tracking system, a series of experiments were conducted in a laboratory setting. A Sun image simulator was used to generate simulated solar images, and the tracking system was tested under various conditions.

1. **Telescope and Webcam Combinations**: Seven experimental cases were conducted to compare the performance of different combinations of telescope magnification and webcam resolution. The combination of a 15× magnification telescope and a high-resolution webcam provided the best tracking accuracy and resolution.
2. **Tracking Accuracy**: The tracking accuracy of the system was measured by stopping the simulated Sun image at a fixed position and recording the central coordinates of the Sun image. The change in central coordinates was less than ±2 pixels, indicating high stability and accuracy.
3. **Threshold Value Testing**: The threshold value used to determine whether the solar tracker moves was tested at 5 and 10 pixels. The results showed that the tracking error was less than 0.04° in the X coordinate and 0.03° in the Y coordinate for a threshold value of 5 pixels, and less than 0.07° in the X coordinate and 0.05° in the Y coordinate for a threshold value of 10 pixels.
4. **Sun Image Shadowing**: The system was tested under conditions where the Sun image was partially shadowed. The tracking accuracy remained within 0.04°, demonstrating the system's high immunity to different Sun images.

### Conclusion

The present invention provides a robust and accurate image-based Sun position sensor and tracking system for solar energy applications. The system combines a reflecting Cassegrain telescope, a high-resolution webcam, and an advanced image processing algorithm to achieve high tracking accuracy and stability. The system has been validated through extensive laboratory testing and has demonstrated excellent performance under various conditions. Future work will focus on field testing and further optimization of the tracking algorithms to develop an optimal tracking system for commercial applications.