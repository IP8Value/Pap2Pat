Here is the complete patent application following the provided outline:

# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates to the field of medical imaging and diagnostics, specifically to methods for quantitative measurement of tissue perfusion using indocyanine green (ICG) fluorescence dynamics. More particularly, the invention provides a system and method for constructing perfusion maps and predicting tissue necrosis probability through mathematical modeling of ICG pharmacokinetics in peripheral tissues. The invention has applications in vascular insufficiency diagnosis, prognosis prediction, and therapeutic monitoring in both clinical and preclinical settings.  

## BACKGROUND  

Peripheral vascular insufficiencies resulting from conditions such as diabetes or systemic atherosclerosis often lead to tissue necrosis due to inadequate perfusion. Current methods for assessing tissue perfusion, including laser Doppler imaging (LDI), scintigraphy, positron emission tomography (PET), and magnetic resonance imaging (MRI), suffer from limitations such as high cost, low spatial resolution, or inability to provide quantitative perfusion measurements. While microsphere perfusion measurement remains the gold standard for animal studies, it is limited to ex vivo applications.  

Near-infrared (NIR) fluorescence imaging with indocyanine green (ICG) has been used for vascular imaging due to its deep tissue penetration and FDA-approved status. However, existing ICG-based techniques fail to provide quantitative perfusion data due to rapid ICG clearance and inability to account for individual variations in pharmacokinetics. There remains an unmet need for a non-invasive, cost-effective method that can quantitatively measure tissue perfusion with high spatial resolution and enable interindividual comparisons for clinical decision-making and therapeutic monitoring.  

## DISCLOSURE  

### Technical Problem  

The technical problem addressed by the present invention is the lack of quantitative, high-resolution methods for measuring functional tissue perfusion that can:  
1) Account for individual variations in ICG pharmacokinetics  
2) Provide spatially resolved perfusion maps  
3) Enable accurate prediction of tissue necrosis probability  
4) Differentiate between macro- and microvascular perfusion  
5) Be implemented in both clinical and preclinical settings  

### Technical Solution  

The invention provides a solution through a method comprising:  
1) Intravenous administration of ICG to a subject  
2) Time-series NIR fluorescence imaging of peripheral tissues  
3) Measurement of ICG fluorescence dynamics in trunk and target tissues  
4) Mathematical modeling of ICG pharmacokinetics accounting for:  
   - Systemic ICG clearance (t1/2 in trunk)  
   - Time-to-peak (Tmax) in each pixel of target tissue  
   - Perfusion rate (P) calculation using derived equations  
5) Construction of:  
   - Pseudocolor-coded perfusion maps  
   - Tissue necrosis probability maps based on correlation with perfusion rates  
6) Quantitative comparison of perfusion rates between tissues and subjects  

The mathematical model compensates for individual variations by normalizing to systemic ICG clearance and enables quantitative perfusion measurement through the relationship:  

P = (1/t1/2) × (1 - t1/2/Tmax)  

where P is perfusion rate (%/min), t1/2 is ICG half-life in trunk, and Tmax is time-to-peak in target tissue.  

## BEST MODE  

### MODE FOR INVENTION  

The best mode for carrying out the invention comprises:  

1) **Imaging System Setup**:  
   - Custom NIR fluorescence imaging system with:  
     * CCD camera (e.g., PIXIS 1024)  
     * 830-nm band-pass filter  
     * 760-nm LED excitation arrays  
   - Capable of 1-second interval image acquisition  

2) **ICG Administration**:  
   - Intravenous bolus injection of 0.1 mL of 400 μmol/L ICG  
   - Continuous imaging for 12 minutes post-injection  

3) **Image Analysis**:  
   - ROI selection for trunk (t1/2 measurement) and target tissues  
   - Pixel-by-pixel determination of Tmax  
   - Calculation of perfusion rate (P) for each pixel  
   - Generation of:  
     * Perfusion maps (pseudocolor-coded by P values)  
     * Necrosis probability maps using sigmoidal function:  
       Probability = 1 / (1 + e^(k(P-P50)))  
       where k and P50 are empirically determined constants  

4) **Clinical/Preclinical Applications**:  
   - Baseline perfusion assessment in peripheral vascular disease  
   - Prediction of tissue necrosis risk  
   - Monitoring therapeutic angiogenesis interventions  
   - Drug efficacy evaluation in animal models  

## Comparative Example 1  

### Prediction for Tissue Necrosis through Doppler Imaging  

Laser Doppler imaging (LDI) was performed on murine hindlimbs following femoral artery ligation. While LDI detected overall reduction in perfusion in ischemic limbs, it failed to:  
1) Quantitatively differentiate perfusion levels predictive of necrosis  
2) Provide spatial resolution sufficient to identify regions at risk  
3) Account for individual variations in systemic hemodynamics  
The perfusion maps generated by LDI showed poor correlation (R² = 0.32) with actual necrosis observed at 7 days post-operation, demonstrating inferior predictive capability compared to the inventive method.  

## Example 1  

### Establishment of Method of Measuring Perfusion Using ICG  

The method was established through:  
1) **In Silico Modeling**:  
   - Developed compartmental model of ICG pharmacokinetics  
   - Derived mathematical relationship between Tmax, t1/2 and P  
   - Validated through computational simulation  

2) **Experimental Validation**:  
   - Normal hindlimbs showed:  
     * Bimodal perfusion distribution (475±19%/min macrovasculature, 120±15%/min microvasculature)  
     * Rapid Tmax (20±3 s)  
   - Ischemic hindlimbs showed:  
     * Unimodal perfusion distribution (56±6%/min)  
     * Delayed Tmax (120±25 s)  
   - Strong correlation (R² = 0.92) between calculated P and microsphere perfusion measurements  

## Example 2  

### Measurement of Perfusion Using Indocyanine Green and the Construction of Perfusion Map and Tissue Necrosis Probability Map Based on Correlation Coefficient  

Application of the method demonstrated:  
1) **Perfusion Map Construction**:  
   - Pixel resolution of 0.1 mm²  
   - Dynamic range of 1-500%/min  
   - Clear differentiation of macro- and microvascular compartments  

2) **Necrosis Prediction**:  
   - Inverse sigmoidal relationship between P and necrosis probability:  
     * P > 60%/min: 0% necrosis probability  
     * P = 20%/min: 50% necrosis probability  
     * P < 5%/min: 95% necrosis probability  
   - Predictive accuracy of 89% compared to actual necrosis at 7 days  

3) **Therapeutic Monitoring**:  
   - Detected synergistic effect of VEGF + cAng1 therapy:  
     * Increased perfusion from 56±6%/min to 364±28%/min at 7 days  
     * Restoration of bimodal perfusion distribution  
   - Correlation with histological evidence of arteriogenesis  

## INDUSTRIAL APPLICABILITY  

The invention has broad industrial applicability in:  

1) **Medical Diagnostics**:  
   - Objective assessment of peripheral vascular disease severity  
   - Prediction of diabetic foot ulcer risk  
   - Post-surgical monitoring of tissue viability  

2) **Pharmaceutical Development**:  
   - Preclinical evaluation of pro-angiogenic drugs  
   - Quantitative assessment of therapeutic efficacy  
   - Animal model stratification for drug trials  

3) **Medical Device Industry**:  
   - Integration with surgical microscopes  
   - Endoscopic perfusion assessment systems  
   - Intraoperative tissue viability assessment tools  

4) **Healthcare Systems**:  
   - Cost-effective alternative to PET/MRI perfusion imaging  
   - Point-of-care vascular assessment  
   - Telemedicine applications for rural healthcare  

The method provides significant advantages over existing technologies through its quantitative output, high spatial resolution, and ability to predict clinical outcomes, making it commercially viable for widespread adoption in both clinical and research settings.