Here is the drafted patent application following the provided outline:

# DESCRIPTION  

## FIELD OF THE DISCLOSURE  

The present disclosure relates generally to forensic science and more specifically to methods and systems for analyzing bloodstain patterns at crime scenes. The disclosure pertains particularly to techniques for determining the time at which a blood pool was formed by analyzing its drying characteristics. The invention provides a novel approach to bloodstain pattern analysis by utilizing evaporation dynamics, shape factors, and diffusion coefficients to estimate the time of blood pool formation with high accuracy. This technology finds particular utility in forensic investigations where establishing an accurate timeline of events is critical but where traditional methods relying on body temperature or rigor mortis are unavailable due to the absence of a corpse.  

## TECHNICAL BACKGROUND  

Bloodstain pattern analysis is a well-established forensic discipline focused on interpreting blood traces found at crime scenes. Current methodologies allow investigators to deduce certain information from blood pools, such as approximate volume or potential blood loss severity. However, existing techniques fail to address the critical temporal question of when a blood pool was formed. Traditional forensic methods for estimating time of death or event timing—such as body temperature measurement, rigor mortis assessment, or forensic entomology—require the presence of a corpse. When no body is present, investigators lack reliable tools to establish when a bloodshed event occurred.  

Prior research has examined various aspects of blood drying, including morphological changes during evaporation and volume determination for reconstruction purposes. These studies have identified distinct stages in blood pool drying: coagulation, gelation, rim desiccation, center desiccation, and final desiccation. While this work has contributed to understanding blood drying mechanics, it has not yielded practical methods for temporal estimation of blood pool formation.  

The physics of blood pool drying presents unique challenges. Blood behaves as a colloidal suspension with red blood cells dispersed in plasma, undergoing complex phase changes during evaporation. Unlike simple liquids, blood transitions through gel-like states during drying, influenced by coagulation factors and fibrin formation. The evaporation dynamics differ significantly from both small droplets (where surface tension dominates) and large liquid bodies (where gravity prevails), placing blood pools in an intermediate regime requiring specialized analysis.  

Existing approaches to liquid evaporation analysis fail to account for these complexities. Standard evaporation models based on Fick's law or free surface convection cannot accurately predict blood pool drying behavior due to the interplay of biochemical, morphological, and physical factors. There exists a pressing need in forensic science for a reliable, physics-based method to determine blood pool formation time that considers these unique characteristics.  

## SUMMARY OF THE DISCLOSURE  

The present disclosure provides systems and methods for determining the time of formation of a blood pool through analysis of its drying characteristics. The invention recognizes that blood pool evaporation follows predictable dynamics that can be quantified through specific parameters including shape factors, diffusion coefficients, and drying front progression.  

Key aspects of the disclosure include:  

A method for estimating blood pool formation time by monitoring the progression of a drying front across the pool surface. The method involves capturing images of the blood pool at known time intervals, measuring the wet area bounded by the drying front, and calculating elapsed time since formation based on evaporation dynamics.  

The recognition that blood pool evaporation occurs in distinct stages analogous to sol-gel transitions, with characteristic evaporation rates during each stage. The disclosure establishes that these rates can be normalized across different pool sizes and shapes through a defined shape factor (L*) that accounts for pool geometry.  

The determination that blood pools exhibit an approximately constant diffusion coefficient when evaporation rates are properly normalized by shape factors and environmental parameters. This discovery enables reliable time estimation regardless of pool size or initial shape.  

A computational approach that correlates wet area reduction with mass loss over time, allowing back-calculation of pool formation time from single or multiple observations of the drying front position.  

Practical implementations including photographic analysis protocols and reference tables accounting for environmental conditions (temperature, humidity) and surface characteristics that affect evaporation rates.  

The disclosed methods provide significant advantages over prior approaches, including:  

The ability to estimate blood pool formation time without requiring a body or other traditional forensic timing methods.  

Accuracy within approximately ±30 minutes under controlled conditions, sufficient for forensic timeline reconstruction.  

Non-destructive analysis using standard crime scene photography equipment without altering evidence.  

Adaptability to various pool sizes and shapes through the incorporated shape factor normalization.  

The technology finds particular application in crime scene investigations where establishing event timing is crucial but traditional methods are unavailable. It provides investigators with a novel tool for bloodstain pattern analysis that addresses a critical gap in current forensic capabilities.  

## DETAILED DESCRIPTION OF EMBODIMENTS  

The following detailed description presents specific embodiments of the invention with reference to the accompanying drawings. These embodiments demonstrate the principles of the invention and its practical applications, enabling those skilled in the art to make and use the invention.  

### Blood Pool Evaporation Dynamics  

Blood pools undergo characteristic evaporation stages that form the basis for temporal analysis. When deposited on a surface, blood first enters a coagulation stage where platelets aggregate and fibrin forms a gel-like matrix. This is followed by gelation, where the blood transitions to a semi-solid state. Subsequent drying occurs through distinct phases:  

1. Rim desiccation: Evaporation begins at the pool periphery, creating a visible drying front that separates wet (red) and dry (black) regions.  
2. Center desiccation: The drying front progresses inward as the rim fully dries.  
3. Final desiccation: The remaining central area dries completely, often with crack formation.  

These stages reflect underlying physical processes where evaporation initially occurs at the liquid-vapor interface (constant rate period), followed by evaporation through the porous gel matrix (falling rate periods). The transitions between stages correspond to changes in evaporation mechanism that affect drying rates.  

### Shape Factor Normalization  

The invention introduces a shape factor (L*) that normalizes evaporation rates across different pool geometries:  

L* = A/(hP)  

Where:  
A = pool area  
h = pool height  
P = pool perimeter  

This dimensionless factor accounts for how pool shape influences evaporation dynamics. Elongated pools with higher perimeter-to-area ratios evaporate faster than circular pools of equal volume due to increased edge effects. The shape factor enables comparison and prediction of evaporation rates regardless of initial pool morphology.  

### Diffusion Coefficient Determination  

Through experimental analysis, the disclosure establishes that blood pools exhibit an approximately constant effective diffusion coefficient (D_blood ≈ 1×10^-9 m^2/s) when evaporation rates are properly normalized. This coefficient is derived by:  

1. Measuring evaporation rates (J*) as mass loss per unit area over time  
2. Normalizing by shape factor (L*) and environmental parameters (temperature, humidity)  
3. Incorporating the Knudsen layer thickness to account for vapor diffusion near the surface  

The resulting diffusion coefficient remains consistent across different pool sizes and shapes when drying conditions are constant, providing a fundamental parameter for time estimation.  

### Time Estimation Methodology  

The invention provides a method to calculate time since pool formation (t_x) using the following relationship:  

t_x = (αRk_B T^2 A_i^(1/2) h^(1/2) ρ[1-(A_x/A_i)]^β)/(Md^2 πP^(1/2) D_blood P_w P_a)  

Where:  
α, β = empirically determined constants  
R = universal gas constant  
k_B = Boltzmann constant  
T = temperature  
A_i = initial pool area  
h = pool height  
ρ = blood density  
A_x = wet area at time of observation  
M = molar mass  
d = molecular diameter  
P = pool perimeter  
D_blood = diffusion coefficient  
P_w = saturation vapor pressure  
P_a = atmospheric pressure  

Implementation involves:  

1. Photographic documentation of the blood pool at known time intervals  
2. Image analysis to measure total area (A_i), perimeter (P), and wet area (A_x)  
3. Input of environmental parameters (temperature, humidity)  
4. Calculation using the above equation to determine time since formation  

### Practical Implementation  

For crime scene application, the invention provides:  

1. Standardized photographic protocols using reference scales and controlled lighting  
2. Image processing software to precisely measure pool dimensions and drying front position  
3. Reference tables correlating diffusion coefficients with environmental conditions  
4. Calibration procedures for different surface types (linoleum, wood, tile etc.)  

The method can utilize either single observations (with reduced precision) or multiple time-series measurements (for improved accuracy). Field implementation requires only a digital camera and measurement scale, making it practical for crime scene investigators.  

### Experimental Validation  

Controlled experiments validate the method's accuracy:  

- Testing across pool sizes from 0.3g to 31.4g initial mass  
- Various shapes (circular, elongated, irregular)  
- Different surfaces (linoleum, varnished wood, tile)  
- Environmental conditions (20-30% relative humidity, 22-24°C)  

Results demonstrate time estimation accuracy within ±30 minutes for pools dried under consistent conditions. The method shows particular reliability when using multiple observations to track drying front progression.  

### Environmental Compensation  

The disclosure provides techniques to account for varying conditions:  

1. Temperature effects: Diffusion coefficients follow Arrhenius-type temperature dependence  
2. Humidity effects: Evaporation rates scale inversely with ambient humidity  
3. Surface effects: Contact angle and porosity influence are incorporated through calibrated parameters  

Reference data enables adjustment of calculations based on measured crime scene conditions. Future implementations may incorporate environmental sensors for real-time parameter input.  

This detailed description illustrates the principles and applications of the invention. Specific parameters and equations may be adjusted by those skilled in the art while maintaining the core inventive concepts. The embodiments demonstrate how blood pool drying physics can be harnessed to provide forensic investigators with reliable time estimation capabilities previously unavailable.