# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to a method for the quantitative measurement of peripheral tissue perfusion using near-infrared (NIR) fluorescence imaging with indocyanine green (ICG). More specifically, the invention provides a method for predicting tissue necrosis and evaluating the therapeutic effects of proangiogenic factors in ischemic tissues.

## BACKGROUND

Peripheral tissues are particularly susceptible to necrosis under conditions that promote vascular insufficiency. Peripheral vascular insufficiencies are highly prevalent and often result from diabetic complications or systemic atherosclerosis. Functional perfusion imaging tools are essential for visualizing the structure of the vasculature and assessing functional perfusion levels, which are crucial for preclinical drug discovery research and clinical applications. Functional perfusion imaging is superior to structural vascular imaging because the prognosis of vascular insufficiency is directly linked to the functional perfusion level rather than the vascular structure. For instance, highly varied outcomes of ischemic hindlimbs can arise from differences in preexisting collaterals. Therefore, measurements of functional tissue perfusion through vessels, including collaterals, are necessary to evaluate vascular insufficiency and predict the prognosis.

Quantitative measurements of peripheral tissue perfusion are required for comparative studies, such as evaluating the effects of drugs in vivo or making clinical decisions based on patient information. Minimizing the invasiveness of perfusion imaging tools is a primary concern for clinical applications. Several methods have been used to quantify perfusion, including scintigraphic and positron emission tomography (PET) imaging and magnetic resonance imaging (MRI). However, these methods are too expensive for use in the diagnosis of tissue perfusion, especially for animal studies. The standard quantitative method for measuring perfusion in animals is microsphere perfusion, but this method is suitable for ex vivo, rather than in vivo, measurements.

Optical imaging, particularly near-infrared (NIR) fluorescence imaging, has proven effective for in vivo imaging of the vasculature and estimating functional perfusion. Indocyanine green (ICG) has been clinically used as a NIR fluorophore for intravital imaging, a marker for liver function, and a sensitizer for photodynamic therapy. ICG is FDA-approved for vascular imaging, and its NIR spectra enable deep tissue imaging. Intravenously injected ICG shows minimal extravasation except in abnormally permeable vasculature because it binds to the major serum protein, albumin. Albumin-ICG complexes are segregated in the liver with first-order pharmacokinetics and excreted via the hepatobiliary pathway. Despite its rapid clearance, the altered pharmacokinetics of ICG can be useful for estimating cerebral oxygenation and hemodynamics and measuring cerebral blood flow (CBF).

However, the lack of quantitative information has limited the use of these techniques in the clinical setting. The main purpose of this invention is to develop a new method for functional and quantitative measurement of peripheral tissue perfusion and to demonstrate the feasibility of this method.

## DISCLOSURE

### Technical Problem

The technical problem addressed by the present invention is the need for a cost-effective, non-invasive, and quantitative method to measure peripheral tissue perfusion. Existing methods, such as scintigraphy, PET, and MRI, are either too expensive or invasive. Microsphere perfusion, while quantitative, is suitable only for ex vivo measurements. The invention aims to provide a method that can quantitatively measure tissue perfusion, predict tissue necrosis, and evaluate the therapeutic effects of proangiogenic factors in ischemic tissues.

### Technical Solution

The present invention provides a method for the quantitative measurement of peripheral tissue perfusion using near-infrared (NIR) fluorescence imaging with indocyanine green (ICG). The method involves the following steps:

1. **Induction of Ischemia**: Inducing ischemia in a subject, such as a murine model, by ligating and excising the femoral artery and vein.
2. **ICG Injection**: Injecting an intravenous bolus of ICG into the subject.
3. **Time-Series Imaging**: Obtaining time-series NIR fluorescence images of the subject at regular intervals after the ICG injection.
4. **Image Analysis**: Analyzing the time-series images to determine the spatiotemporal dynamics of ICG pharmacokinetics.
5. **Perfusion Rate Calculation**: Using a mathematical model to translate the ICG dynamics into perfusion rates.
6. **Perfusion Map Construction**: Constructing a perfusion map to visualize the spatial distribution of perfusion rates.
7. **Necrosis Prediction**: Predicting the probability of tissue necrosis based on the perfusion rates.
8. **Therapeutic Evaluation**: Evaluating the therapeutic effects of proangiogenic factors, such as VEGF and cAng1, on perfusion and the subsequent prognosis of ischemic tissues.

The mathematical model used in the invention accounts for the vascular input function (VIF) of ICG in normal tissues and the dynamics of ICG fluorescence intensity in the ischemic hindlimbs. The perfusion rate is defined as the fraction of blood exchanged per minute in the vascular volume of the region of interest (ROI). The time-to-peak (Tmax) and the ICG half-life in the trunk are used to calculate the perfusion rate of each pixel in the ROI.

## BEST MODE

### MODE FOR INVENTION

The mode for the invention involves the following detailed steps:

1. **Murine Hindlimb Ischemia Model**:
   - Obtain BalB/cAnNCriBgi-nu nude male mice aged 7-8 weeks (15-20 g).
   - Induce hindlimb ischemia by ligating and excising the right femoral artery and vein under ketamine-xylazine anesthesia.
   - Divide the mice into groups for intramuscular injection with saline solution (control), cAng1, VEGF, or both cAng1 and VEGF.
   - Perform serial ICG perfusion imaging immediately after surgery and on postoperative days (POD) 3 and 7.

2. **NIR Fluorescence Imaging**:
   - Use two optical systems for NIR fluorescence imaging: Image Station 4000 MM (Eastman Kodak Co.) and a customized system with a CCD digital camera (PIXIS 1024; Princeton Instruments), a custom-made 830-nm band-pass filter, and 760-nm light-emitting diode arrays.
   - Inject an intravenous bolus of ICG (0.1 mL of 400 µmol/L) into the tail vein of the mice.
   - Obtain time-series ICG fluorescence images for 12 minutes at 20-second intervals (Kodak imaging system) or 1-second intervals (custom optical imaging system).
   - Take a silhouette image of the mouse under white light to obtain the ROI mask of the ischemic and normal hindlimbs.

3. **In Silico Modeling**:
   - Perform in silico studies to identify the relationships between the spatiotemporal ICG profiles and the perfusion rates in peripheral tissue.
   - Define the perfusion rate as the fraction of blood exchanged per minute in the vascular volume of the ROI.
   - Assume time-invariant blood volume through the vasculature and equal volumetric inflow and outflow.
   - Divide the rear half of the mouse into three compartments: the trunk, the normal hindlimb, and the ischemic hindlimb.
   - Use the vascular input function (VIF) of ICG in normal tissues, described as a uniexponential function of ICG excretion by the liver.
   - Derive the dynamics of ICG fluorescence intensity in the ischemic hindlimbs using Fick's law.
   - Calculate the perfusion rate (P) of each pixel in the ROI using the time-to-peak (Tmax) and the ICG half-life in the trunk.

4. **Image Analysis**:
   - Draw ROI masks of the ischemic and normal limbs on the silhouette image to extract regions corresponding to the limbs in the acquired time-series NIR fluorescence images.
   - Measure the half-life of ICG in the square mask region on the trunk and determine the time-to-peak of each pixel to obtain the perfusion rate of a pixel in the ROI.
   - Calculate the perfusion rate in each pixel using the equation derived by modeling.
   - Visualize the result as a pseudocolor-coded perfusion map or a histogram.

5. **Histological Analysis**:
   - Perfuse the ischemic muscles with 4% (w/v) paraformaldehyde and embed them in paraffin.
   - Stain 10 µm thick calf muscle sections with hematoxylin-eosin (H&E) and double-stain with anti-CD31 antibody and a monoclonal anti-α-smooth muscle-actin (α-SMA) antibody conjugated to fluorescein isothiocyanate (FITC).
   - Stain proteins immunoreactive with the anti-CD31 antibody with a hamster anti-mouse IgG antibody conjugated with rhodamine.
   - Visualize the stained sections by confocal microscopy and express vessel density as the number of α-SMA and CD31 double-positive micro- (16-63 µm in diameter) and macrovessels (>63 µm in diameter) per high-power field (magnification ×400).

6. **Laser Doppler Imaging (LDI)**:
   - Scan the mice using LDI after ICG perfusion imaging at POD 0.
   - Perform three consecutive scans until blood flow measurements are stable.
   - Subject the images to computer-assisted quantification of blood flow.

7. **Micro-CT Angiography**:
   - Place the mice in an induction chamber with 4% isofluorane in oxygen to induce anesthesia.
   - Inject PEG-conjugated gold nanoparticles intravenously and place the mice on a volumetric CT scanner.
   - Acquire 600 images at 65 kVp, 55 µA, and 800 ms per frame.
   - Reconstruct the images using the Feldkamp cone-beam reconstruction algorithm.
   - Convert the final reconstructed data to the DICOM format to create three-dimensional (3-D) rendered images using 3-D-rendering software.

8. **Statistical Analysis**:
   - Express data as means ± SEM.
   - Assess statistical significance using Student's two-tailed t-test for two groups or one-way ANOVA and Bonferroni post hoc test for three or more groups.

## Comparative Example 1

### Prediction for Tissue Necrosis through Doppler Imaging

In a comparative example, laser Doppler imaging (LDI) was used to predict tissue necrosis in ischemic limbs. Mice were scanned using LDI after ICG perfusion imaging at POD 0. Three consecutive scans were performed until blood flow measurements were stable. The images were subjected to computer-assisted quantification of blood flow. The representative data of the ischemic/normal blood flow ratio using LDI did not provide enough sensitivity to show correlations between the initial blood flow and the necrosis level in the ischemic limbs. This indicates that LDI, while useful for qualitative assessments, lacks the sensitivity required for precise predictions of tissue necrosis.

## Example 1

### Establishment of Method of Measuring Perfusion Using ICG

In this example, the method for measuring perfusion using ICG was established. Murine hindlimb ischemia models were created by ligating and excising the right femoral artery and vein. Time-series NIR fluorescence images were obtained after intravenous injection of ICG. The spatiotemporal dynamics of ICG pharmacokinetics were analyzed using a mathematical model to calculate the perfusion rate of each pixel in the ROI. The perfusion rates were then visualized as a pseudocolor-coded perfusion map. The perfusion map showed the spatial distribution of perfusion rates with high spatial resolution, and histogram analysis revealed a bimodal distribution of perfusion rates in the normal limb, while only one peak with a lower perfusion rate was observed in the ischemic hindlimb. This method demonstrated high sensitivity and accuracy in predicting future necrosis of ischemic tissues.

## Example 2

### Measurement of Perfusion Using Indocyanine Green and the Construction of Perfusion Map and Tissue Necrosis Probability Map Based on Correlation Coefficient

In this example, the measurement of perfusion using indocyanine green (ICG) and the construction of perfusion and tissue necrosis probability maps were demonstrated. Time-series NIR fluorescence images were obtained from murine hindlimb ischemia models after intravenous injection of ICG. The spatiotemporal dynamics of ICG pharmacokinetics were analyzed using a mathematical model to calculate the perfusion rate of each pixel in the ROI. The perfusion rates were visualized as a pseudocolor-coded perfusion map, and the relationship between the regional perfusion rate and the probability of necrosis was determined. A necrosis map was generated by mapping the predictive necrosis of individual regions as a color-coded picture. The perfusion maps of representative mice showed considerable contrasts in ischemic limb perfusion rates, and the necrosis maps showed a remarkably high correlation with the actual necrotic regions determined on POD 7. This method provided superior sensitivity and accuracy compared to conventional LDI, confirming its feasibility and accuracy in predicting tissue necrosis.

## INDUSTRIAL APPLICABILITY

The present invention has significant industrial applicability in the fields of medical diagnostics and therapeutic evaluation. The method for the quantitative measurement of peripheral tissue perfusion using near-infrared (NIR) fluorescence imaging with indocyanine green (ICG) can be used to predict tissue necrosis and evaluate the therapeutic effects of proangiogenic factors in ischemic tissues. This method is cost-effective, non-invasive, and highly sensitive, making it suitable for both preclinical and clinical applications. It can be used to diagnose patients with peripheral vascular insufficiencies and to monitor the effectiveness of treatments aimed at improving tissue perfusion and preventing necrosis. The invention has the potential to revolutionize the multidisciplinary analysis of biomedical imaging and contribute to the development of clinical diagnostic techniques for peripheral vascular disease.