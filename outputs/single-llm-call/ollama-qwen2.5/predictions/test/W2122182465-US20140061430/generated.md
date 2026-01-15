## Patent Application for Image-Based Solar Tracking System

### Background

Solar energy systems benefit significantly from accurate solar tracking. Traditional methods like bar-shadow photosensors and four-quadrant light sensors have limitations under low irradiation conditions. This application describes an image-based solar tracking system that uses a high-resolution webcam and a reflecting Cassegrain telescope to achieve precise sun tracking, even on cloudy days.

### Summary

The invention provides a solar tracking system comprising a self-designed reflecting Cassegrain telescope with adjustable magnification, a high-resolution webcam, and a tracking controller with an embedded image processing algorithm. The system captures clear Sun images, processes them to determine the Sun's position, and sends commands to the tracker to follow the Sun accurately.

### Detailed Description

#### Output of the Image-Based Solar Tracking System

The output of the solar tracking system is precise data for the central coordinates of the Sun image. This data is used by the tracking controller to send commands to the solar tracker, ensuring it follows the Sun's trajectory with high accuracy, even under varying weather conditions.

#### Output of the Tracking Controller

The tracking controller processes the Sun image data and sends motor control signals to adjust the position of the solar tracker. The controller uses an embedded algorithm to calculate the necessary movements based on the deviation between the current and desired positions of the Sun image.

#### Image Processing Algorithm

The image processing algorithm converts the captured images into a binary format using lightness thresholds, applies edge detection methods like Sobel, and calculates the central coordinates of the Sun image using the three-point circle method. This ensures accurate tracking even when parts of the Sun are obscured by clouds or other obstructions.

#### Image-Based Sun Position Sensor

The image-based Sun position sensor consists of a reflecting Cassegrain telescope with adjustable magnification (5–15×) and a high-resolution webcam (2,304 × 1,536 pixels). The combination of high resolution and magnification provides the best tracking accuracy, with a resolution of 0.0017° per pixel.

#### Sun Image Simulator

The sun image simulator generates simulated solar images to test the performance of the tracking system. It uses a high-resolution camera with neutral density and infrared filters to capture clean and high-contrast Sun images, which are then used to simulate different sky conditions.

#### Solar Tracking System

The solar tracking system includes the image-based Sun position sensor, a motorized tracker, and a control unit. The tracker adjusts its position based on commands from the controller to follow the Sun's trajectory accurately, ensuring optimal energy capture.

#### Adjustable Enlargement Telescope

The adjustable enlargement telescope (5–15× magnification) is designed to provide clear and detailed Sun images. It can be easily connected to a webcam and mounted on the tracking system due to its lightweight and compact design.

#### High-Resolution Webcam

A high-resolution webcam (2,304 × 1,536 pixels) captures the Sun images with high detail. The combination of this webcam and the adjustable telescope provides the best resolution for accurate Sun position detection.

#### Tracking Controller

The tracking controller processes the image data from the webcam, calculates the necessary adjustments, and sends commands to the motorized tracker. It uses an embedded algorithm to ensure stable and precise tracking under various conditions.

#### Embedded Image Processing Algorithm

The embedded image processing algorithm includes steps for converting images to binary format, edge detection, and calculating central coordinates. This ensures that the system can accurately track the Sun even when parts of the image are obscured or under low irradiation conditions.

#### Experimental Setup

The experimental setup includes a sun image simulator, a solar tracking system with an image-based Sun position sensor, and a control unit. The setup is used to test the performance of the tracking system under different conditions and to optimize its parameters.

#### Performance Testing

Performance testing involves measuring the uncertainty of the Sun position sensor, the impact of different magnification-webcam combinations, and the system's accuracy in various weather conditions. The results show that the system maintains high tracking accuracy, even on cloudy days.

### Conclusion

The image-based solar tracking system offers a robust solution for accurate sun tracking under various conditions. By combining a reflecting Cassegrain telescope with a high-resolution webcam and an embedded image processing algorithm, the system ensures optimal energy capture and stable operation. Future work will focus on field testing and iterative parameter adjustments to further optimize performance.