# DESCRIPTION

## BACKGROUND

The field of visual motion detection is critical for various applications, including surveillance, robotics, and autonomous driving. Traditional methods for motion detection often rely on optic flow techniques, which estimate the spatial changes in consecutive image frames. While these methods can produce accurate results, they are computationally intensive and not suitable for real-time implementation. Biological visual systems, on the other hand, have evolved highly efficient neural circuits to detect visual motion. These circuits operate in parallel and in continuous time, enabling fast and effective motion detection.

Inspired by the principles of biological motion detection, this invention introduces a novel motion detection algorithm based on local phase information of visual scenes. The algorithm leverages the properties of phase information to detect motion in a computationally efficient manner, making it suitable for real-time implementation on parallel hardware.

## SUMMARY

The present invention provides a method and system for detecting visual motion using local phase information. The method includes the following steps:

1. **Global Phase of Images**: The global phase of an image is defined as the phase of the Fourier transform of the image. The global phase plays a crucial role in representing the structure of the image.

2. **Local Phase of Images**: The local phase of an image is defined as the phase of the Short-Time Fourier Transform (STFT) of the image. The local phase captures the phase information within localized regions of the image.

3. **The Global Phase Equation for Translational Motion**: The change in global phase over time is related to the translational motion of the image. This relationship is described by a simple equation involving the frequency components and the velocity of the motion.

4. **The Local Phase Equation for Translational Motion**: The change in local phase over time is related to the translational motion within localized regions of the image. This relationship is described by an equation that includes an additional term to account for the non-uniformity of the motion.

5. **The Block Structure for Computing the Local Phase**: The image is divided into overlapping blocks, and the local phase is computed for each block using Gaussian windows. The 2D Fourier transform is applied to each block to extract the local phase information.

6. **The Phase-Based Detector**: The change in local phase is used to detect motion. The Radon transform is applied to the change in local phase to identify the direction and strength of the motion. A Phase Motion Indicator (PMI) is computed for each block to determine whether motion is present.

7. **Radon Transform on the Change of Phases**: The Radon transform is used to analyze the change in local phase and to identify the direction of motion. The PMI is computed based on the Radon transform to robustly detect motion.

8. **Examples of Phase-Based Motion Detection**: The method is applied to various video sequences to demonstrate its effectiveness in detecting local motion and its use in motion segmentation tasks. The results are compared to those obtained using traditional optic flow techniques.

## DETAILED DESCRIPTION

### Global Phase of Images

The global phase of an image \( u = u(x, y) \) is defined as the phase of the Fourier transform of the image. Mathematically, the Fourier transform of \( u \) is given by:
\[ \hat{U}(\omega_x, \omega_y) = \int_{\mathbb{R}^2} u(x, y) e^{-j(\omega_x x + \omega_y y)} dx dy \]
where \( \hat{U}(\omega_x, \omega_y) \in \mathbb{C} \) and \( (\omega_x, \omega_y) \in \mathbb{R}^2 \).

In polar coordinates, the Fourier transform can be expressed as:
\[ \hat{U}(\omega_x, \omega_y) = \hat{A}(\omega_x, \omega_y) e^{j\hat{\phi}(\omega_x, \omega_y)} \]
where \( \hat{A}(\omega_x, \omega_y) \in \mathbb{R} \) is the amplitude and \( \hat{\phi}(\omega_x, \omega_y) \) is the global phase of the Fourier transform of \( u \).

### Local Phase of Images

The local phase of an image \( u = u(x, y) \) is defined as the phase of the Short-Time Fourier Transform (STFT) of the image. The STFT is given by:
\[ U(\omega_x, \omega_y, x_0, y_0) = \int_{\mathbb{R}^2} u(x, y) w(x - x_0, y - y_0) e^{-j(\omega_x (x - x_0) + \omega_y (y - y_0))} dx dy \]
where \( w(x, y) \) is a real-valued window function centered at \( (x_0, y_0) \).

In polar coordinates, the STFT can be expressed as:
\[ U(\omega_x, \omega_y, x_0, y_0) = A(\omega_x, \omega_y, x_0, y_0) e^{j\phi(\omega_x, \omega_y, x_0, y_0)} \]
where \( A(\omega_x, \omega_y, x_0, y_0) \in \mathbb{R} \) is the local amplitude and \( \phi(\omega_x, \omega_y, x_0, y_0) \) is the local phase of the STFT.

### The Global Phase Equation for Translational Motion

For a visual stimulus \( u = u(x, y, t) \) undergoing pure translational motion, the change in global phase over time is related to the velocity of the motion. Mathematically, the change in global phase is given by:
\[ \frac{d\hat{\phi}(\omega_x, \omega_y, t)}{dt} = - \omega_x v_x(t) - \omega_y v_y(t) \]
where \( v_x(t) \) and \( v_y(t) \) are the velocity components in the \( x \) and \( y \) directions, respectively.

### The Local Phase Equation for Translational Motion

For a visual stimulus \( u = u(x, y, t) \) undergoing translational motion within a localized region, the change in local phase over time is related to the velocity of the motion. Mathematically, the change in local phase is given by:
\[ \frac{d\phi_{00}(\omega_x, \omega_y, t)}{dt} = - \frac{ds_x(t)}{dt} \omega_x - \frac{ds_y(t)}{dt} \omega_y + \mathfrak{v}_{00}(\omega_x, \omega_y, t) \]
where \( s_x(t) \) and \( s_y(t) \) are the total lengths of translation in the \( x \) and \( y \) directions, respectively, and \( \mathfrak{v}_{00}(\omega_x, \omega_y, t) \) is an additional term to account for the non-uniformity of the motion.

### The Block Structure for Computing the Local Phase

The image is divided into overlapping blocks, and the local phase is computed for each block using Gaussian windows. The 2D Fourier transform is applied to each block to extract the local phase information. The Gaussian windows are defined as:
\[ (\mathcal{T}_{kl} w)(x, y) = e^{-\frac{(x - x_k)^2 + (y - y_l)^2}{2\sigma^2}} \]
where \( x_k = k b_0 \) and \( y_l = l b_0 \), and \( b_0 \) is the distance between two neighboring windows.

The 2D Fourier transform of the windowed video signal is given by:
\[ \int_{\mathbb{R}^2} u(x, y, t) (\mathcal{T}_{kl} w)(x, y) e^{-j(\omega_x (x - x_k) + \omega_y (y - y_l))} dx dy = A_{kl}(\omega_x, \omega_y, t) e^{j\phi_{kl}(\omega_x, \omega_y, t)} \]

### The Phase-Based Detector

The change in local phase is used to detect motion. The Radon transform is applied to the change in local phase to identify the direction and strength of the motion. The Phase Motion Indicator (PMI) is computed for each block to determine whether motion is present. The PMI is given by:
\[ PMI_{kl} = \max_{\theta \in [0, \pi)} \sum_{\rho} \left| \frac{(\mathcal{R} \frac{d\phi_{kl}}{dt})(\rho, \theta, t_0)}{\mathfrak{c}(\rho, \theta)} \right| \]
where \( \mathcal{R} \) is the Radon transform, and \( \mathfrak{c}(\rho, \theta) \) is a correction term due to different lengths of line integrals in the bounded domain.

### Radon Transform on the Change of Phases

The Radon transform of the change in local phase is given by:
\[ (\mathcal{R} \frac{d\phi_{kl}}{dt})(\rho, \theta, t) = \int_{\mathbb{R}} \frac{d\phi_{kl}}{dt}(\rho \cos \theta - s \sin \theta, \rho \sin \theta + s \cos \theta, t) 1_C(\rho \cos \theta - s \sin \theta, \rho \sin \theta + s \cos \theta) ds \]
where \( 1_C(\omega_x, \omega_y) \) is the indicator function of the circular bounded domain \( C \).

### Examples of Phase-Based Motion Detection

The method is applied to various video sequences to demonstrate its effectiveness in detecting local motion and its use in motion segmentation tasks. The results are compared to those obtained using traditional optic flow techniques. The method is shown to be computationally efficient and robust under different contrast and illumination conditions. The detected motion is used to segment moving objects from the background, and the results are visually compared to those obtained using optic flow-based methods.

The phase-based motion detection algorithm is highly parallelizable and can be efficiently implemented on parallel hardware, such as GPUs and FPGAs. The algorithm is particularly useful for real-time applications where fast and accurate motion detection is required.