Here is the outline of the desired patent application.
Per bullet point, write roughly 800 words.

Example outline (bullet points are the lines starting with '- '):
## DESCRIPTION OF THE INVENTION
- describe discovery of ODAM protein in human epithelial cancers
- describe method for aiding in diagnosis and management of cancer
- describe specific embodiments of the invention
- describe methods for determining presence of ODAM or anti-ODAM antibodies

In the example above, each line beginning with '- ' is a bullet point.

```md
# DESCRIPTION

## BACKGROUND

- introduce parallel imaging
- describe linear approaches
- limitations of linear approaches

## BRIEF SUMMARY

- introduce preferred embodiments
- motivate temporal sensitivity
- describe first aspect
- describe second aspect
- describe third aspect
- disclaim limitations

## DETAILED DESCRIPTION OF THE DRAWINGS AND PRESENTLY PREFERRED EMBODIMENTS

- introduce linear parallel imaging algorithms
- limitations of linear parallel imaging algorithms
- introduce self-consistent parallel imaging with temporal sensitivity estimation (TSPIRIT)
- describe TSPIRIT reconstruction
- describe application of TSPIRIT to cardiac real-time cine imaging
- describe fully automated reconstruction
- describe various clinical uses of TSPIRIT
- show flow chart of method for parallel imaging with temporal sensitivity in magnetic resonance reconstruction
- describe acts in flow chart
- describe flexibility of acts in flow chart
- show example implementation of method of parallel imaging
- describe data flow for temporal sensitivity framework
- acquire magnetic resonance data
- describe k-space data acquisition
- describe coil usage
- describe sequence of transmissions
- describe frames representing different times
- describe reduction factor
- perform initial reconstruction
- describe linear reconstruction
- filter frames
- describe temporal filtering
- output filtered frames
- estimate coil sensitivity information
- describe iterative self-constraint parallel imaging reconstruction calibration
- perform non-linear reconstruction
- describe non-linear solver
- describe least square matrix inversion solver
- describe non-linear conjugate gradient solver
- output reconstructed data
- combine reconstructed data
- describe sum-of-square coil combination
- generate output data
- describe further processing of output data
- output to memory or transmission
- describe display of output
- describe system for parallel imaging with temporal sensitivity in magnetic resonance reconstruction
- describe MR system
- describe memory
- describe processor
- describe display
- store data in memory
- describe data stored in memory
- implement flow as program
- describe implementation languages
- describe system components
- describe alternative system configurations
- calculate signal-to-noise ratio (SNR)
- describe region of interest
- show images reconstructed using TSPIRIT and linear methods
- describe advantages of TSPIRIT
```

You need to draft a complete patent application that strictly follows the outline's section order and headings. Do not skip any bullet points. Use formal patent language. The generated patent must not be shorter than the research paper in word count.

Here is the research paper that describes the invention:

```md
# Summary

To improve the image quality and increase the signal-noise-ratio of real-time exercise stress cardiac cine imaging, we extended the recently proposed SPIRIT image reconstruction by incorporating temporal coil sensitivity estimation and spatial regularization. The proposed method was tested on 10 volunteers and the average SNR gain was 38.2% without increasing ghosting artifacts.

# Background

Myocardial wall motion can now be assessed using CMR immediately following treadmill exercise [1,2]. Given that stress-induced wall motion abnormalities rapidly fade after cessation of exercise, imaging must be completed as quickly as possible. Additionally, shortness of breath following exercise precludes the use of segmented k-space acquisitions making real-time imaging the only practical choice for post-exercise cine. While this eliminates the needs for ECG triggering and breath-holding, signal-noise-ratio (SNR) and temporal and spatial resolution are typically sacrificed. Image degradation can be more severe for post-exercise cine due to rapid heart rate and exaggerated breathing. To improve the quality of exercise stress cine, we propose to extend the SPIRIT reconstruction technique [3] by incorporating temporal sensitivity estimation (TSPIRIT) and spatial regularization.

# Methods

10 healthy volunteers (6 males; age 23.1−41.1 yrs) with normal left ventricular thickness underwent free-breathing real-time exercise stress cine examinations after having given written consent. An MR compatible treadmill system was utilized [1] together with a 1.5T scanner (Avanto, Siemens) and a 32-channel coil (Rapid MRI). Three slices (one short-axis and two long-axis views) were acquired in each subject using the following sequence parameters: bSSFP, TR1.09/TE0.9ms, image matrix 160×80, flip angle 58°, resolution 2.44×2.44 mm2, bandwidth 1420Hz, acceleration rate 4 with time-interleaved sampling of k-space.

All temporally interleaved k-space frames were averaged to generate ACS lines and GRAPPA was first used to estimate full k-space for every frame. Karhunen-Loeve transform filtering was applied to improve sensitivity estimation, and SPIRIT calibration was performed for every frame. The under-sampled k-space and estimated kernels served as inputs to a non-linear solver. An LSQR matrix inversion solver was performed, and then a non-linear conjugate gradient solver was called with spatial regularization. Final images were generated after convergence and compared to images of conventional TGRAPPA reconstruction [4] of the same raw data. SNR was estimated [5] and ghosting artifacts caused by chest wall motion were quantified by computing the peak spatial cross-correlation ratio along the phase-encoding direction [6].

# Results

Figure 1 shows an example of the improved image quality of the proposed TSPIRIT technique compared to TGRAPPA. Figure 2 plots measured SNR and artifact scores, showing a significant increase of SNR (mean relative gain 38.2±17.8%, P<1e-5) without raising ghosting artifacts (P=0.960). This is consistent with visual reading.

# Conclusions

A new TSPRIT reconstruction scheme has been proposed by extending the SPIRIT method with temporal sensitivity estimation and spatial regularization. This technique was shown to significantly improve the quality of exercise stress real-time cine imaging. In vivo testing shows 38.2% SNR gain can be achieved without raising ghosting artifacts.

# Funding

NIH grant R01 HL102450
```
