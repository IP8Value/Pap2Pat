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

## FIELD OF THE INVENTION

- introduce field of invention

## SUMMARY

- introduce arterial pressure measurement
- describe current practice
- describe advantages of current practice
- describe limitations of current practice
- introduce hydraulic connection distortions
- describe transfer function of hydraulic connection
- describe distortions caused by hydraulic connection
- describe impact on derived indices
- describe clinical staff detection of defects
- describe proposed solutions
- describe limitations of proposed solutions
- introduce need for improvement
- describe aim of invention
- describe method for improving quality of measured signal
- describe device for improving quality of measured signal
- describe steps of method
- determine dynamic parameters of hydraulic connection
- analyse frequential content of incident signal
- detect situation of hydraulic connection
- describe situation (a)
- describe situation (b)
- describe situation (c)
- describe mechanical action on tubing
- describe synchronisation of mechanical action
- describe alternative mechanical actions
- describe detection of decoupling
- describe applanation tonometry
- describe pulse wave velocity propagation
- describe comparison with reference value
- describe alarm in case of decoupling
- describe device for continuous monitoring and correction
- describe monitoring and correcting module
- describe actuator module
- describe integration into conventional measuring chain
- describe advantages of device
- describe ease of installation
- describe transparency for hospital staff
- describe neutralisation of device
- describe first processor
- describe second processor
- describe redundant treatments
- describe actuator module
- describe wired connection
- describe wireless connection
- describe rechargeable battery
- describe casing for monitoring module
- describe actuator module connection
- describe processing system
- describe communication with monitoring module
- describe applanation tonometry casing

## DETAILED DESCRIPTION

- introduce monitoring device for measuring arterial pressure
- describe device components: monitoring and correcting module, actuator module, remote processing system
- describe monitoring and correcting module functions
- characterise dynamic parameters of hydraulic connection
- perform frequential analysis of incident arterial pressure signal
- estimate aptitude of hydraulic connection for reproducing incident signal
- detect measuring artefacts and incidents
- provide information on hydraulic connection state and corrective action
- describe three operating modes: active, passive, neutral
- describe architecture of monitoring and correcting module
- describe two processors: first processor for signal processing, second processor for correction
- describe communication between processors and with actuator module and remote processing system
- describe power management module
- describe energy efficiency of monitoring and correcting module
- describe actuator module functions
- apply mechanical action to tubing to excite hydraulic connection
- describe pulse and sinusoidal loads for characterising hydraulic connection
- synchronise mechanical action with diastole
- describe actuator module components: actuator, power source
- describe remote processing system functions
- process measured signal and deduce derived indices
- display arterial pressure plottings and record measurements
- perform quality follow-up and calibration
- describe methods for processing signal
- determine dynamic parameters of hydraulic connection
- describe percussion method using fast flushes
- extract response to fast flush from haemodynamic signal
- calculate damping factor and natural frequency
- describe limitations of percussion method
- describe harmonic method
- apply brief series of sinusoidal loads
- describe advantages of harmonic method
- describe limitations of harmonic method
- describe combination of percussion and harmonic methods
- describe advantages of combination
- describe limitations of combination
- describe other methods for processing signal
- describe implementation of monitoring device
- describe casing for monitoring and correcting module
- describe connectors for interconnecting module to existing measuring device
- describe simplified man-machine interface
- describe sterilisation of monitoring and correcting module
- describe actuator module implementation
- describe advantages of actuator module implementation
- describe remote processing system implementation
- introduce percussion method
- describe actuator module
- motivate harmonic method
- describe harmonic analysis
- derive formulas for z and f0
- describe need for segmenting mechanical action
- motivate analysis of signal spectrum
- describe principle of signal spectrum analysis
- motivate determination of frequential content
- describe FFT analysis
- estimate aptitude of hydraulic connection
- describe first method for estimation
- describe second method for estimation
- illustrate second method
- motivate monitoring of hydraulic connection
- describe detection of artefacts
- describe detection of abnormal attenuation
- describe correction and alert
- describe logging of incidents
- motivate detection of decoupling
- describe detection module
- illustrate variation in aortic and radial pressures
- describe frequential analysis method
- illustrate spectral content of aortic and radial pressures
- describe measurement of pulse wave velocity
- describe Moens-Korteweg equation
- illustrate decrease in pulse wave velocity
- describe distribution of arterial pressure
- illustrate mean pulse wave velocity
- describe principle of second detection method
- describe measurement of time delay
- describe estimation of pulse wave velocity
- describe variant using brachial and carotid measurements
- describe transfer function for aortic pressure
- describe operation of tonometric casing
- describe monitoring of radial arterial pressure
- describe warning generation
- describe input of distances
- describe data fusion device
- conclude invention
```

You need to draft a complete patent application that strictly follows the outline's section order and headings. Do not skip any bullet points. Use formal patent language. The generated patent must not be shorter than the research paper in word count.

Here is the research paper that describes the invention:

```md
# Introduction

Invasive monitoring in critically ill patients allows a continuous measurement of arterial pressure, cardiac output, and the derivation of dynamic predictors of fluid responsiveness. However, the pressure signal may be altered by the dynamic characteristics of the fluid-filled tubing. The aim of the present study was to evaluate the reliability of radial artery blood pressure measurement and derived indexes during the early period after cardiac surgery.

# Methods

After IRB approval, 30 patients admitted to the ICU after elective cardiac surgery (CABG: 16, valve surgery: 11; combined: 3) with a radial artery catheter were included. In the ICU, an independent continuous recording of arterial pressure during at least 18 hours was started via a double-head pressure transducer (Flotrac; Edwards Lifesciences, Irvine, CA, USA) for a retrospective analysis and three fast flushes were performed. First, the whole record was examined for episodes of overdamping (Ov) or attenuation (At). Ov was defined as a decrease in systolic (sAP), an increase diastolic (dAP), and an unchanged mean pressure (mAP). At was defined as a decrease in sAP, dAP and mAP. Second, three periods of 10 minutes during the first hour were analysed assuming that the dynamic characteristics remained constant. This allowed the correction of the distorted raw signals and the study of the consequences of an underdamped signal on sAP, pulse pressure variation (PPV) and dP/dt as an estimate of left ventricular contractility. A paired t test was used for statistical comparison, P < 0.05 was considered statistically significant.

# Results

Mean age was 69 ± 13 years, 14 patients received noradrenaline, eight patients dobutamine, and nine patients volume expansion. During the whole record, the number of episodes of Ov or At ranged from 0 to 15 with a duration of 0 to 6 hours: 17 patients had at least one episode of Ov and/or At tracing, 10 patients had at least two episodes, eight patients had at least five episodes. Seven episodes lasted more than 20 minutes and three more than 1 hour. During the first hour, sAP was overestimated by 5.0 ± 1.4 mmHg (P < 0.0001) (range: 0.3 to 5.9) or by 4.3 ± 0.9% (range: 0.4 to 15.9%), raw PPV was 9.5 ± 7.3 versus 10.0 ± 7.8 for the corrected PPV (range from -2.6 to 4.3); raw dP/dt was overestimated by 134 ± 47 mmHg/second (P < 0.0001) (range: -13 to 353) or by 24 ± 6%.

# Conclusion

These results showed that frequent artefacts and distortions induced by the fluid-filled tubing could modify the arterial waveform and could lead to inaccurate therapy [1]. More attention should be paid to the quality of the pressure signal.
```
