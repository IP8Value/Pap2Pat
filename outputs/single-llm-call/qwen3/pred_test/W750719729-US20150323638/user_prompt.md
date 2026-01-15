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

- relate to imaging methods

## BACKGROUND

- describe limitations of current methods

## SUMMARY

- introduce method for quantifying FFR
- describe multi-dimensional phase-contrast magnetic resonance sequence
- determine pressure gradient within blood vessel segment
- correlate pressure gradient to FFR value
- describe MRI system for executing sequence
- describe non-transitory machine-readable medium with instructions

## DETAILED DESCRIPTION

- define terms used in application
- describe FFR technique
- describe advantages of MRI over CT
- describe method of using MRI for quantifying FFR
- describe multi-dimensional PC-MR sequence
- calculate pressure gradient using Navier-Stokes equations
- describe image reconstruction using generic Fourier transform methods
- derive 4D flow velocity field
- calculate velocity derivatives and pressure gradient field
- obtain transtenotic pressure difference
- describe VOI in subject
- describe imaging parameters
- describe cardiac phase
- describe scan time
- describe acquisition window
- describe ECG-triggering and navigator-gating
- describe MRI system
- describe processor and its functions
- describe computer and its functions
- describe non-transitory machine-readable medium
- describe method for diagnosing cardiovascular disease
- describe stenosis
- describe mild, moderate, and severe stenosis
- describe alternative imaging systems
- describe equivalents to methods and materials

## EXAMPLES

### Example 1

- describe 3D PC-MR sequence
- describe acquisition window and gating
- describe imaging parameters

### Example 2

- describe phantom studies

### Volunteer Studies

- describe volunteer studies

### Example 3

- describe quantification of pressure gradient
- describe healthy human volunteer data
- describe noninvasive FFR measurement
- describe various methods and techniques
- describe objectives and advantages
- describe alternatives and equivalents
- describe applicability of various features
- describe skilled artisan recognition
- describe embodiments and modifications
```

You need to draft a complete patent application that strictly follows the outline's section order and headings. Do not skip any bullet points. Use formal patent language. The generated patent must not be shorter than the research paper in word count.

Here is the research paper that describes the invention:

```md
# Background

Fractional Flow Reserve (FFR) is an invasively determined index of the functional severity of an intermediate coronary stenosis by measuring the pressure drop across the lesion [1]. Noninvasive pressure gradient (ΔP) measurements using phase-contrast (PC)-MRI have been attempted in the aorta, carotid, and renal arteries [2-4]. The purpose of this study is to assess the reproducibility of PC-MRI and noninvasive ΔP calculations in the coronary artery, which is relevant for establishing the robustness of the noninvasive FFR technique.

# Methods

2D PC-MRI was used to acquire two-cardiac-phase data at mid-diastole and end-expiration via ECG-triggering and navigator-gating on 3T MAGNETOM Verio (Siemens). K-space phase-encoding ordering is designed to allow offline view sharing [5], which is applied in cases where the acquisition window exceeds the quiescent period (~100ms). The sequence measures the velocity field (vx, vy, vz) of a single cross-section per acquisition and 4-5 consecutive slices were obtained in the proximal LAD. Reproducibility was assessed with two repeat scans on 4 healthy subjects. VENC ranged 30-45 cm/s for each flow encoding direction was determined from a VENC scout scan. The Navier-Stokes equations were used to derive ΔP [6]. In addition, a flow phantom (gadolinium-doped water flow at 300 mL/min in a silicone tubing of 4.8mm ID) with 40% stenosis (VENC=130z30xy cm/s) was likewise tested for reproducibility. Imaging parameters were: in-plane resolution = 0.58-0.67mm, slice thickness = 3.2 mm, flip angle = 15°, 65-71 ms/phase with the first phase strictly coinciding with the quiescent period, scan time = 1-3 min per slice. Absolute maximum and averaged velocities at each slice in all three directions and the ΔP between adjacent slices obtained from both scans were statistically compared via intra-class correlation (ICC).

# Results

Volunteer studies: averaged maximum through-plane velocity over all healthy volunteers was 16.5±4.0 cm/s. A total of 19 slices were acquired from all subjects. For velocity measurements, excellent correlations were seen in the through-plane velocities (vz), with ICCs of 0.93/0.96 and slightly lower in vx and vy with ICCs of 0.83/0.86 and 0.80/0.78 for cardiac phases 1 and 2, respectively. For ΔPs, ICC was 0.51 with an average of 0.1039±0.28 mmHg among all subjects. Phantom studies: stenosis with 40% narrowing showed excellent correlations in all velocity directions and ΔPs (table 1).

# Conclusions

Our preliminary results suggest that the noninvasive quantification of flow velocities and ΔPs are reproducible in the coronary arteries, demonstrating the robustness and feasibility of 2D PC-MRI. Patient studies are underway to determine ΔP and FFR thresholds between healthy and patient populations. Further technical improvements are warranted to reduce noise and improve reproducibility.

# Funding

N/A.
```
