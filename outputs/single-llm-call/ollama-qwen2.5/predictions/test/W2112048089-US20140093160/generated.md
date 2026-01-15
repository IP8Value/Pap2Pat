### Introduction

In this paper, we present an advanced noninvasive method for tracking tumors using digitally reconstructed radiograph (DRR) sequences derived from real-patient 4DCT data. The primary goal is to achieve high accuracy and real-time performance in tumor tracking, which is crucial for Image-Guided Radiation Therapy (IGRT). Our approach leverages two methods: 3D Regression (3DR) and 2D Regression (2DR), both of which utilize HOG features and regression models.

### Methodology

The methodology involves training a regression model on the initial pair of DRR images where the tumor position is known. This trained model then tracks the tumor in subsequent frames automatically. The 3DR method learns a joint 3D regression model, while the 2DR method estimates tumor positions separately in two orthogonal views and combines them. Both methods are designed to handle the challenges posed by low-resolution, low-contrast DRR sequences.

### Data and Challenges

We use orthogonal DRR sequences from 10 patients' 4DCT data, which include different tumor locations, shapes, and internal volume characteristics. The DRR sequences present significant challenges such as limited resolution, low contrast, noise, and imaging artifacts. These issues make tumor tracking more difficult and push the algorithm to its limits.

### Performance Evaluation

The performance of our 3DR and 2DR methods is compared with state-of-the-art techniques, including optical flow (OF) and particle filter-based methods (Pℓ1 and PHOG). Our results show that 2DR achieves an average error of 1.05 pixels, while 3DR has an average error of 1.16 pixels. These errors are significantly lower than those of other methods, demonstrating the robustness and accuracy of our approach.

### Results

Our experiments on 10 different patient datasets show that 2DR and 3DR consistently achieve the most accurate tracking results. The average tumor displacement in the ground truth is 13.86 pixels, and 2DR estimates are 92.5% accurate. The performance of other methods varies significantly, with some losing track of the tumor under certain scenarios. Our methods also process frames much faster, making them suitable for real-time applications.

### Conclusion

We have developed a noninvasive tumor tracking method that outperforms existing techniques in both accuracy and speed. This tracker can reduce the prescribed treatment volume margins from several centimeters to millimeters, thereby minimizing radiation exposure to healthy tissue. Future work will focus on extending the regression model to incorporate biomechanical tissue constraints for more complex tumor shapes.

### Future Work

As a future study, we plan to enhance our regression model by incorporating biomechanical tissue constraints. This will enable better tracking of very complex tumor shapes and further improve the accuracy and robustness of our method. Additionally, we aim to validate our approach on a larger dataset and explore its application in other medical imaging modalities.
