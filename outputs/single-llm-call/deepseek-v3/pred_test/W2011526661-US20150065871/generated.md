Here is the patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND  

The blood-brain barrier (BBB) presents a significant challenge for delivering therapeutic agents to targeted brain regions. Current techniques for BBB modulation suffer from limitations including invasiveness, poor spatial specificity, and drug-specific restrictions. Intracranial injections, chemical modification of drugs, or attaching agents to BBB-modifying chemicals often lack precision and may inadvertently affect non-targeted brain regions. These methods carry risks of damaging overlying brain structures when targeting deep subcortical regions such as the basal ganglia.  

Existing ultrasound-based approaches for BBB opening face challenges related to ultrasound beam aberrations caused by the skull. The trabecular layer of the skull induces heterogeneities in both speed of sound and density, leading to strong phase aberrations of the acoustic beam. Higher ultrasound frequencies exacerbate these aberrations as the wavelength approaches the size of skull bone heterogeneities (typically around 1 mm). While reducing ultrasound frequency decreases phase aberrations, it also increases focal region size and the likelihood of inertial cavitation that may cause permanent tissue damage.  

## SUMMARY  

The present invention introduces a transcranial monitoring system for real-time monitoring of safe blood-brain barrier opening. The system comprises an ultrasound transducer operating at an intermediate frequency of approximately 500 kHz, which provides an optimal balance between targeting accuracy and safety without requiring phase aberration correction. The system includes components for precise stereotactic targeting of specific brain regions and real-time monitoring of cavitation behavior through acoustic emission analysis.  

The ultrasound transducer is configured to deliver focused ultrasound pulses in combination with systemically injected microbubbles to cause reversible increases in BBB permeability at localized target regions. A passive cavitation detector (PCD) monitors acoustic emissions from microbubbles in real-time, enabling immediate assessment of treatment safety and efficacy. The system includes computer program products that convert stereotactic manipulator settings into precise targeting coordinates and predict the region of BBB opening.  

## DETAILED DESCRIPTION  

The invention provides a system for real-time transcranial monitoring of safe blood-brain barrier opening. The system comprises a 500 kHz center frequency focused ultrasound transducer (H-107, Sonic Concepts, WA, USA) attached to a stereotaxic manipulator for precise targeting. The transducer is configured to deliver ultrasound pulses at focal maximum pressures between 0.20 and 0.30 MPa for durations sufficient to achieve localized BBB opening (typically 2 minutes).  

System 100 for real-time monitoring includes the ultrasound transducer and targeting components. Subjects are positioned in a stereotaxic frame under general anesthesia induced by ketamine (5-15 mg/kg IM) and maintained with isoflurane (1-4% inhaled). The ultrasound transducer is attached to a Kopf stereotaxic manipulator to enable targeting in stereotaxic coordinates. Negative control sonications are performed in the absence of microbubbles to establish baseline acoustic emission levels.  

Following intravenous injection of size-isolated 4-5 μm microbubbles prepared through differential centrifugation, sonication proceeds for the prescribed duration. Post-sonication controls are performed immediately after treatment while microbubbles remain in circulation. The system compensates for tissue attenuation by increasing emission amplitude by approximately 7.15 dB (factor of 2.28) compared to calibration measurements in water, accounting for skull attenuation (-5.7 dB at 500 kHz), scalp attenuation (-0.9 dB/cm over 0.5 cm), and brain tissue attenuation (-0.5 dB/cm over 2 cm).  

A spherically focused hydrophone (Y-107, Sonic Concepts) positioned through the center hole of the FUS transducer monitors acoustic emissions in real-time. The hydrophone connects to a digitizer through a 20-dB amplifier (5800, Olympus NDT) for passive cavitation detection. The transducer-hydrophone assembly is mounted with their focal regions fully overlapping within the confocal volume.  

Individualized targeting utilizes T1-weighted stereotactically aligned structural images. Software (stereotax.R) converts stereotaxic manipulator settings into stereotaxic coordinates through nine free parameters: medio-lateral drive (ml), anterior-posterior position (ap), dorsoventral drive (dv), azimuth rotation, elevation angle, left/right arm position, ml/dv drive alignment (stereo), and transducer attachment (finger). The software predicts the focal point and approach angle, projecting the expected BBB opening region onto structural images.  

For targeting in stereotactic coordinate frames, the software calculates up to eight different manipulator settings that target a specified neural structure from a desired approach angle, typically chosen for perpendicular skull incidence. BBB opening verification uses contrast-enhanced MRI with gadodiamide (Omniscan™). High-resolution structural T1 images (3D Spoiled Gradient-Echo, TR/TE = 20/1.4 ms) acquired pre- and 30 minutes post-injection (0.15 ml/kg IV) highlight regions of increased BBB permeability. Additional T2-weighted (TR/TE = 3000/80 ms) and Susceptibility-Weighted Images (TR/TE = 19/27 ms) assess potential tissue damage.  

Image analysis registers pre- and post-T1 images to stereotaxically aligned references using FSL's FLIRT routine. Gadodiamide concentration is estimated by dividing post- by pre-T1 images, then removing symmetric vascular signals through hemisphere flipping. Targeting accuracy assessment rotates and shifts resulting images into a coordinate frame with origin at predicted focus and z-axis along approach angle.  

Real-time monitoring analyzes the frequency content of backscattered acoustic signals to infer microbubble cavitation behavior. Stable cavitation produces harmonic modes (nf, n=1,2,...6), sub-harmonic (f/2) and ultra-harmonic (nf/2, n=3,5,7,9) frequencies, while inertial cavitation generates broadband noise. The system filters 300-kHz bandwidths around harmonics and 100-kHz bandwidths around sub/ultra-harmonics to isolate broadband signals between 0.6-5.2 MHz.  

Two metrics quantify cavitation behavior: Broadband Energy Increase (BEI) indicates inertial cavitation, while Harmonic Energy Increase (HEI) indicates stable cavitation. These are calculated relative to negative control sonications without microbubbles. Positive control sonications (0.05-0.35 MPa, 2-sec duration) performed post-treatment characterize the pressure-response relationship.  

### Example 1  

Experimental results demonstrate the system's performance in targeting basal ganglia structures. Seventeen sonications targeted the caudate nucleus (6) and putamen (11) in two macaque monkeys. Targeting accuracy analysis revealed mean focal points 0.2±1.0 mm posterior, 1.9±1.7 mm ventral, and 1.4±1.4 mm shifted towards the transducer from intended targets. The mean targeting error measured 2.5±1.2 mm laterally and 1.5±1.3 mm axially (3.1±1.3 mm combined). Systematic errors averaged 2.7 mm across targets, while random errors measured 1.5±0.7 mm.  

BBB opening volumes averaged 115±44 mm³, with larger openings at higher pressures (0.30 MPa). Real-time PCD monitoring showed HEI increases ≥15 dB during sonication, indicating stable cavitation, while BEI remained below the 6 dB safety threshold. Pressure-response measurements revealed HEI increases starting at 0.15 MPa, reaching approximately 10 dB by 0.25 MPa, with no BEI increase up to 0.35 MPa.  

Safety assessments using T2-weighted and SWI MRI sequences detected no edema or hemorrhage. A preliminary BBB closing timeline experiment showed nearly complete restoration within two days for a 126 mm³ opening at 0.30 MPa.  

Targeting error analysis identified three primary sources: geometric errors from stereotaxic manipulator settings (1-2 mm), analysis errors from image registration and enhancement quantification (~1 mm), and ultrasound aberration errors causing focal depth shifts. The observed 6.5 mm axial shift matched predictions from skull-induced aberrations.  

### Example 2  

Additional experiments targeted the caudate nucleus and putamen with analysis focusing on PCD response correlation with BBB opening volume. The system demonstrated reliable detection of stable cavitation through HEI monitoring, with successful BBB openings (94%) when HEI exceeded 5 dB. However, HEI magnitude did not strongly correlate with opening volume within the 0.20-0.30 MPa pressure range.  

The system's alternative embodiment for real-time monitoring includes the FUS transducer, hydrophone, PC workstation, and customized program for PCD signal acquisition. In vitro techniques using desiccated macaque and human skulls enabled pressure calibration and skull attenuation measurements (-5.7 dB at 500 kHz).  

Signal processing quantified cavitation levels through stable cavitation dose (SCD) and inertial cavitation dose (ICD), with cavitation signal-to-noise ratio (SNR) analysis demonstrating reliable transcranial detection. The system achieved safe BBB opening with volume variations corresponding to pressure adjustments, while maintaining ICD below safety thresholds.  

B-mode imaging provided cavitation visualization, while PCD served as an indirect monitoring tool. Investigations of pulse length effects (100- vs 5000-cycle pulses) established optimal parameters for transcranial detection. The system demonstrated reliable PCD monitoring through skull, with methods to increase cavitation SNR and estimate in situ cavitation strength despite nonlinear ultrasound scattering effects.  

In vivo techniques confirmed the system's ability to monitor cavitation doses during BBB opening and achieve safe, targeted openings in specific brain regions. The noninvasive nature of the focused ultrasound setup, combined with stereotactic targeting and real-time passive cavitation analysis, enables precise brain-drug delivery with opening durations of approximately two days at moderate pressure parameters.