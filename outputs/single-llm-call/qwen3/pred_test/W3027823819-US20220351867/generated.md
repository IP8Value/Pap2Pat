## FIELD OF THE DISCLOSURE

- define dating blood pools

The present disclosure relates to a method and system for determining the time elapsed since the formation of a blood pool on a surface, thereby enabling forensic investigators to establish a temporal sequence of events at a crime scene. Specifically, the invention provides a quantitative, physics-based approach to date blood pools by analyzing their drying dynamics through the measurement of physical parameters such as wet area, drying front position, environmental conditions, and the diffusion characteristics of blood. This method is applicable to non-porous surfaces commonly encountered in forensic contexts, including tiles, linoleum, and varnished wood, and is designed to operate under ambient environmental conditions without requiring invasive sampling or destruction of evidence. The invention is intended for use in criminal investigations, accident reconstructions, and other scenarios where the timing of bloodshed events is critical to establishing timelines, corroborating witness statements, or identifying inconsistencies in narratives. By transforming visual observations of drying blood into computable physical metrics, the disclosure overcomes longstanding limitations in forensic science by introducing a reproducible, objective, and mathematically grounded technique for blood pool dating.

## TECHNICAL BACKGROUND

- limitations of current methods

Current forensic methods for estimating the time of bloodshed events rely heavily on indirect biological and environmental indicators such as body temperature, rigor mortis, livor mortis, and forensic entomology. These techniques are often inapplicable when no body is present, which is common in cases involving cleanup, relocation of victims, or crimes occurring in public or unoccupied spaces. Alternative approaches, such as visual assessment of bloodstain morphology or subjective estimation based on color change and crust formation, lack scientific rigor, are highly variable between observers, and cannot be quantified or validated. No standardized, peer-reviewed method exists to determine the time since a blood pool was deposited based on its physical drying behavior. As a result, investigators are forced to rely on circumstantial evidence or educated guesswork, which undermines the reliability of temporal reconstructions and compromises the integrity of judicial proceedings. The absence of a reliable, non-destructive, and universally applicable technique for blood pool dating represents a critical gap in forensic science.

- summarize spectroscopy-based techniques

Spectroscopy-based techniques have been explored in forensic contexts to analyze the chemical composition of dried bloodstains, particularly for identifying species, detecting drugs, or estimating age through hemoglobin degradation. While these methods offer valuable biochemical insights, they require specialized equipment such as Raman or infrared spectrometers, are sensitive to surface contamination, and do not provide temporal information about when the blood was deposited. Furthermore, spectroscopic signals are influenced by substrate material, ambient lighting, and surface reflectivity, making them unsuitable for rapid, on-site deployment. These techniques are also destructive or semi-destructive, requiring small sample removal, which is often undesirable in crime scene investigations where evidence preservation is paramount. Consequently, spectroscopy is not a viable tool for dating blood pools in real-world forensic settings.

- summarize hyperspectral imaging

Hyperspectral imaging has been applied in forensic analysis to distinguish between different types of biological fluids based on their spectral signatures across hundreds of narrow wavelength bands. Although this technology can differentiate blood from other fluids with high accuracy, it does not provide information regarding the time elapsed since deposition. The technique requires expensive, bulky imaging systems, controlled lighting conditions, and extensive post-processing, rendering it impractical for field use. Additionally, hyperspectral imaging is sensitive to environmental variables such as humidity and temperature, which can alter spectral reflectance independently of the drying state of the blood. Thus, while useful for fluid identification, hyperspectral imaging is not capable of resolving temporal dynamics of drying and cannot be employed as a standalone method for dating blood pools.

- summarize Laan et al. article

Laan et al. investigated the morphological evolution of drying blood pools and identified five distinct stages of desiccation: coagulation, gelation, rim desiccation, centre desiccation, and final desiccation. Their work established that drying is not uniform but progresses from the periphery inward, forming a visible drying front that separates wet, red blood from dried, blackened residue. This observation provided the foundational insight that the position and progression of the drying front could serve as a measurable indicator of time elapsed. However, Laan et al. did not develop a quantitative model to correlate the position of the drying front with time, nor did they account for environmental variables or substrate effects. Their study remained descriptive and observational, lacking the mathematical framework necessary for practical application in forensic casework.

- summarize Choi et al. and Thanakiatkrai et al. articles

Choi et al. examined the evaporation dynamics of small blood drops, focusing on the role of Marangoni flows and particle redistribution during the drying process. Their findings highlighted the importance of contact line dynamics and radius-dependent evaporation rates in micro-scale bloodstains. Thanakiatkrai et al. similarly studied the drying patterns of blood drops on various substrates, noting the formation of characteristic ring-like deposits due to capillary flow. While both studies contributed valuable knowledge regarding the behavior of individual blood drops, they are not directly applicable to larger blood pools, where gravity dominates over surface tension and evaporation occurs through porous media rather than along a pinned contact line. The scaling laws and physical models derived from drop studies fail to account for the gel-like structure, volume-dependent mass loss, and three-dimensional drying front propagation observed in pools of blood exceeding several milliliters in volume. Consequently, these studies do not provide a basis for dating larger, more forensically relevant blood pools.

## SUMMARY OF THE DISCLOSURE

- define "comprise" and "about"

For the purposes of this disclosure, the term “comprise” and its grammatical variants, such as “comprising” or “comprised,” are used in an inclusive, open-ended sense, meaning that the described system, method, or composition may include the recited elements or steps, but is not limited to them and may include additional elements or steps not explicitly mentioned. The term “about” when used in reference to a numerical value, parameter, or range indicates a permissible deviation of ±10% from the stated value, unless otherwise specified, to account for natural variability in measurement, environmental fluctuation, or instrument precision inherent in field-deployable forensic tools.

- introduce objective of disclosure

The objective of this disclosure is to provide a scientifically rigorous, repeatable, and field-applicable method for determining the time elapsed since the formation of a blood pool on a surface, based on the physical principles governing the evaporation and diffusion of blood under ambient environmental conditions. The method is designed to be non-invasive, non-destructive, and compatible with standard forensic photography equipment, enabling its use by crime scene investigators without requiring specialized laboratory infrastructure. By integrating measurable physical parameters with a validated drying model, the disclosure enables the calculation of elapsed time with a margin of error of less than ±30 minutes under controlled environmental conditions, thereby providing a reliable temporal anchor for forensic reconstruction.

- describe method for dating blood pools

The method for dating blood pools involves capturing one or more high-resolution images of the blood pool at a given time, measuring the wet area delimited by the drying front, determining the environmental conditions surrounding the pool, and applying a mathematical drying model that correlates the wet area with the mass of residual liquid and the time elapsed since deposition. The drying model is derived from the diffusion of water vapor through the porous gel matrix formed after coagulation, and incorporates the effects of temperature, humidity, substrate material, and pool geometry. The wet area is determined from image analysis, and the initial mass of the blood pool is estimated from the total area and average height of similar pools on the same substrate. The elapsed time is then calculated by comparing the current wet area to a predetermined functional relationship between wet area and normalized mass, which has been empirically established across a range of experimental conditions.

- define "blood pool"

For the purposes of this disclosure, a “blood pool” refers to a volume of blood deposited on a surface that has spread beyond the dimensions of a single drop, typically exceeding 0.1 mL in volume, and has undergone coagulation and gelation to form a semi-solid, porous matrix. The blood pool exhibits a distinct drying front separating a wet, red interior from a dried, darkened periphery, and retains sufficient liquid content to allow for measurable mass loss over time. Blood pools are distinguished from blood drops by their larger volume, dominant gravitational influence, gel-like rheological behavior, and non-uniform drying dynamics characterized by inward progression of the drying front.

- describe limitations of method

The method is limited to blood pools that are still in the process of drying and have not yet reached complete desiccation. Once the drying front has fully receded and the pool is entirely dry, the method cannot be applied. Similarly, if the blood pool is still entirely liquid and no drying front is visible, the method requires waiting for the onset of desiccation, which may take several minutes depending on environmental conditions. The accuracy of the method is also dependent on the precision of environmental measurements and the quality of the image used to determine the wet area. Substrates with high porosity or absorbency are not suitable, as they alter the evaporation dynamics and invalidate the model assumptions. Furthermore, the method assumes that the blood originates from a healthy human donor without pathological conditions or pharmacological interference that significantly alter clotting kinetics or plasma composition.

- describe application of method

The method finds primary application in forensic investigations where the temporal sequence of events must be reconstructed, such as in homicide, assault, or traffic accident cases. It enables investigators to determine whether a blood pool was formed before or after a suspected time of injury, whether a suspect’s account of events is consistent with the drying state of the blood, or whether multiple pools were deposited at different times. The method may also be used to corroborate or challenge alibis, assess the plausibility of cleaning attempts, or determine the duration of a victim’s exposure to a location. The method is particularly valuable in cases where no body is present, where traditional methods of time estimation are inapplicable, or where multiple independent timelines must be reconciled.

- describe drying process

The drying process of a blood pool begins with the deposition of liquid blood, followed by rapid coagulation due to platelet aggregation and fibrin network formation, resulting in a gel-like structure. Water within the plasma then evaporates from the surface, initiating a drying front that propagates inward from the periphery. As evaporation continues, the gel matrix shrinks, and liquid is transported through capillary channels as vapor, leading to a gradual reduction in wet area. The rate of evaporation is governed by the diffusion of water vapor through the porous medium, modulated by environmental temperature, humidity, and the geometry of the pool. The process progresses through distinct stages: initial constant-rate evaporation, followed by falling-rate phases as the internal liquid becomes increasingly isolated, culminating in final desiccation when only a residual biological deposit remains.

- describe drying model

The drying model is a mathematical framework that relates the time elapsed since deposition to the wet area of the blood pool, the environmental conditions, and the physical properties of blood. The model is derived from Fick’s second law of diffusion, adapted to account for the gel-like structure of dried blood and the influence of the drying front. It incorporates a diffusion coefficient specific to blood under given environmental conditions, a shape factor that accounts for pool geometry, and a Knudsen layer correction to model the vapor transport near the liquid-vapor interface. The model is expressed as a function that calculates elapsed time based on the normalized wet area, the initial mass of the pool, and the measured environmental parameters, allowing for the derivation of a precise time estimate without requiring direct mass measurements.

- describe correlation of time elapsed with diffusion coefficient

The time elapsed since the formation of a blood pool is directly correlated with the diffusion coefficient of water vapor through the gel matrix of the blood. A higher diffusion coefficient results in faster evaporation and a more rapid reduction in wet area, thereby shortening the time required to reach a given drying state. Conversely, a lower diffusion coefficient, which occurs at higher humidity or lower temperature, slows the drying process and increases the time elapsed for the same wet area. The diffusion coefficient is not a constant but varies predictably with environmental conditions, and its value is determined empirically from controlled experiments across a range of temperatures and humidities. The correlation between time and diffusion coefficient is embedded within the drying model as a multiplicative factor that scales the rate of wet area reduction.

- describe database of environmental and initial conditions

A comprehensive database has been compiled containing measured values of blood pool parameters under controlled environmental conditions, including temperature, relative humidity, substrate material, initial mass, initial area, initial height, drying front progression, wet area over time, and corresponding elapsed times. The database includes data from over 200 blood pools deposited on linoleum, tile, and varnished wood surfaces, with humidity levels ranging from 15% to 70% and temperatures from 18°C to 30°C. Each entry is associated with the hematocrit level of the donor blood, the shape factor of the pool, and the calculated diffusion coefficient. This database serves as the reference foundation for the drying model, enabling interpolation and extrapolation of unknown conditions during forensic analysis.

- describe determination of environmental conditions

Environmental conditions are determined using portable, calibrated instruments such as digital thermometers and hygrometers placed adjacent to the blood pool at the time of imaging. Temperature and relative humidity are recorded to the nearest 0.1°C and 1% respectively. These values are then matched against the database to select the appropriate diffusion coefficient and shape factor correction. In cases where environmental conditions are unknown, the method may be applied iteratively using assumed values, with sensitivity analysis performed to determine the range of possible elapsed times consistent with the observed wet area.

- describe determination of initial area

The initial area of the blood pool is estimated from the total area of the pool as captured in the image, assuming that the pool was initially circular or elliptical and had not yet begun to shrink. The initial area is calculated using image processing software that identifies the outer boundary of the wet region at the time of deposition, which is approximated by the maximum area observed in the earliest available image or by extrapolation from the final wet area using the established mass-area correlation function.

- describe comparison of conditions and area

The measured environmental conditions and the current wet area are compared against the database to identify the closest matching experimental conditions. The corresponding relationship between normalized wet area and normalized mass is then retrieved, and the current mass of the pool is calculated. This mass is compared to the estimated initial mass to determine the mass loss, which is then used in the drying model to compute the elapsed time. The comparison is performed algorithmically to ensure consistency and minimize human bias.

- describe determination of blood pool parameters

Blood pool parameters include the wet area, total area, perimeter, average height, shape factor, and drying front position. These are determined from high-resolution images using automated image processing software that applies edge detection, thresholding, and contour analysis to delineate the wet and dry regions. The average height is derived from empirical data for the specific substrate, and the shape factor is calculated as the ratio of area to the product of perimeter and height. All parameters are computed in real time during image analysis.

- describe measurement of blood pool parameters

Measurement of blood pool parameters is performed by capturing one or more digital images of the blood pool using a calibrated camera with a reference scale placed adjacent to the pool. The images are imported into image processing software that automatically detects the wet area by identifying the transition between red and black regions using color segmentation algorithms. The perimeter and total area are computed from the detected contours, and the shape factor is calculated using the established geometric formula. The average height is retrieved from the database based on substrate type and pool volume.

- describe predetermined environmental conditions

Predetermined environmental conditions refer to the set of temperature and humidity values for which the drying model and associated diffusion coefficients have been experimentally validated and stored in the database. These conditions span the typical range encountered in indoor forensic environments, from 18°C to 30°C and 15% to 70% relative humidity. The model is not valid outside this range without additional calibration, and the system alerts the user if input conditions fall beyond the validated range.

- describe calculation of time elapsed

The calculation of time elapsed is performed using a derived equation that integrates the diffusion coefficient, the shape factor, the Knudsen layer thickness, the saturation vapor pressure, and the normalized mass loss. The equation is implemented in calculation software that accepts input values for wet area, environmental conditions, and substrate type, and outputs the elapsed time in hours and minutes. The calculation is based on the correlation between mass loss and wet area, and is validated against the database to ensure accuracy within the specified margin of error.

- describe drying model equation

The drying model equation is expressed as:  
\[ t_x = \frac{\alpha R k_B T^2 A_i^{1/2} h^{1/2} \rho \left[1 - \left( \frac{A_x}{A_i} \right) \right]^\beta}{M d^2 \pi P^{1/2} D_{\text{blood}} P_w P_a} \]  
where \( t_x \) is the elapsed time, \( \alpha \) and \( \beta \) are empirical constants, \( R \) is the universal gas constant, \( k_B \) is Boltzmann’s constant, \( T \) is temperature, \( A_i \) is initial area, \( h \) is average height, \( \rho \) is blood density, \( A_x \) is wet area, \( M \) is molar mass of water, \( d \) is molecular diameter, \( P \) is perimeter, \( D_{\text{blood}} \) is the diffusion coefficient of blood, \( P_w \) is saturation vapor pressure, and \( P_a \) is atmospheric pressure.

- describe blood's diffusion coefficient equation

The diffusion coefficient of blood, \( D_{\text{blood}} \), is determined from the equation:  
\[ D_{\text{blood}} = K_i \cdot L_K \cdot \sqrt{L^*} \]  
where \( K_i \) is the mass transfer coefficient, \( L_K \) is the Knudsen layer thickness, and \( L^* \) is the shape factor. The value of \( D_{\text{blood}} \) is pre-calculated for each combination of temperature and humidity and stored in the database.

- describe coefficient of transfer equation

The coefficient of transfer, \( K_i \), is calculated using the equation:  
\[ K_i = \frac{J^* \cdot R T}{M P_w} \]  
where \( J^* \) is the evaporation rate of water, \( R \) is the universal gas constant, \( T \) is temperature, \( M \) is the molar mass of water, and \( P_w \) is the saturation vapor pressure of water at the surface.

- describe evaporation rate equation

The evaporation rate, \( J^* \), is defined as the mass loss per unit area per unit time, and is calculated as:  
\[ J^* = \frac{\Delta m}{A_{\text{wet}} \cdot \Delta t} \]  
where \( \Delta m \) is the change in mass, \( A_{\text{wet}} \) is the wet area, and \( \Delta t \) is the time interval between measurements.

- describe Knudsen layer equation

The Knudsen layer thickness, \( L_K \), is given by:  
\[ L_K = \frac{k_B T}{\pi d^2 P_a} \]  
where \( k_B \) is Boltzmann’s constant, \( T \) is absolute temperature, \( d \) is the molecular diameter of water vapor, and \( P_a \) is atmospheric pressure.

- describe shape factor equation

The shape factor, \( L^* \), is defined as:  
\[ L^* = \frac{A}{h P} \]  
where \( A \) is the total area of the pool, \( h \) is the average height, and \( P \) is the perimeter.

- describe measurement of blood's diffusion coefficient

The diffusion coefficient of blood is measured experimentally by monitoring the evaporation rate of water from blood pools under controlled environmental conditions and applying the relationship between \( J^* \), \( K_i \), \( L_K \), and \( L^* \). Multiple pools of varying shapes and sizes are dried on different substrates, and the resulting evaporation rates are normalized to yield a consistent diffusion coefficient value of approximately \( 1.1 \times 10^{-9} \, \text{m}^2/\text{s} \) under standard conditions.

- describe determination of wet perimeter

The wet perimeter is determined by image processing software that identifies the boundary between the wet and dry regions of the blood pool. Edge detection algorithms, such as Canny or Sobel filters, are applied to the image after color segmentation to isolate the drying front. The perimeter is then computed as the length of the detected contour line separating the red and black regions.

- describe determination of wet area

The wet area is determined by applying a color threshold to the image to isolate pixels corresponding to the red hue of undried blood. The resulting binary mask is processed to remove noise and fill gaps, and the total area of the connected region is calculated in square millimeters. The wet area is updated with each new image taken during the drying process.

- describe taking pictures of blood pool

Pictures of the blood pool are taken using a digital camera with a resolution of at least 5 megapixels, positioned perpendicularly above the pool at a fixed distance. A reference scale of known dimensions is placed adjacent to the pool to enable accurate scaling. Images are captured at regular intervals, typically every 10 to 30 minutes, to track the progression of the drying front. Lighting is kept consistent using ambient or diffuse artificial illumination to avoid shadows and glare.

- describe image processing of pictures

Image processing involves importing the captured images into software that performs color segmentation to distinguish the wet blood region from the dried residue. The software applies a hue-saturation-value threshold to isolate the red component of the blood, followed by morphological operations to clean the binary mask. The wet area, perimeter, and total area are then computed automatically. The software also corrects for lens distortion and ensures scale accuracy using the reference marker.

- describe drying stages of blood pool

The drying stages of a blood pool include coagulation, gelation, rim desiccation, centre desiccation, and final desiccation. During coagulation, the blood transitions from liquid to gel within minutes of deposition. Gelation is followed by rim desiccation, where the periphery dries first, creating a visible drying front. The drying front then propagates inward toward the center during centre desiccation. Finally, in the last stage, only isolated pockets of moisture remain, and the pool becomes fully desiccated. The method is applicable only during the rim and centre desiccation stages, when the drying front is clearly visible.

- describe correlation of mass variation with wet area

Mass variation is strongly correlated with wet area through a non-linear empirical function derived from experimental data. As the pool dries, the wet area decreases in a predictable manner relative to the mass lost. This correlation is expressed as:  
\[ \frac{m}{m_i} = 1 - \alpha \left[1 - \left( \frac{A_x}{A_i} \right) \right]^\beta \]  
where \( m \) is the current mass, \( m_i \) is the initial mass, \( A_x \) is the wet area, \( A_i \) is the initial area, and \( \alpha \) and \( \beta \) are constants determined from regression analysis across multiple datasets.

- describe function correlating mass variation with wet area

The function correlating mass variation with wet area is a power-law relationship derived from statistical fitting of normalized mass versus normalized wet area data from over 200 experimental trials. The function is expressed as:  
\[ \frac{m}{m_i} = 1 - 0.78 \left[1 - \left( \frac{A_x}{A_i} \right) \right]^{0.16} \]  
and has a correlation coefficient greater than 0.97, demonstrating high predictive reliability across diverse pool geometries and environmental conditions.

- describe statistical value of blood pool's height

The statistical value of the blood pool’s height is derived from the mean and standard deviation of height measurements across a large dataset of blood pools deposited on a specific substrate under controlled conditions. For example, on a white tile, the average height is 1.44 mm with a standard deviation of 0.19 mm. This statistical value is used to estimate the initial volume of the pool when direct measurement is not possible.

- describe average value of blood pool's height

The average value of the blood pool’s height is determined empirically from measurements of pools deposited on identical substrates under controlled laboratory conditions. For linoleum, the average height is 1.38 mm; for varnished wood, it is 1.41 mm; and for tile, it is 1.44 mm. These values are stored in the database and used to calculate initial volume from the measured area.

- describe correlation coefficients equation

The correlation coefficient, \( r \), between normalized mass and normalized wet area is calculated using the Pearson correlation formula:  
\[ r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}} \]  
where \( x_i \) and \( y_i \) are the normalized wet area and normalized mass values, respectively, and \( \bar{x} \) and \( \bar{y} \) are their respective means. Values of \( r > 0.95 \) indicate a strong linear relationship suitable for predictive modeling.

- describe normalized mass equation

The normalized mass is defined as the ratio of the current mass of the blood pool to its initial mass:  
\[ \frac{m}{m_i} = \frac{m_i - \Delta m}{m_i} \]  
where \( \Delta m \) is the mass lost due to evaporation. This value is used to normalize drying behavior across pools of different initial sizes.

- describe measurement of correlation coefficients

Correlation coefficients are measured by plotting the normalized wet area against the normalized mass for each experimental trial and computing the Pearson correlation coefficient using statistical software. The resulting values are used to validate the robustness of the mass-area relationship and to establish confidence intervals for time estimation.

- describe fitting curve for correlation coefficients

A fitting curve for the correlation between normalized mass and normalized wet area is generated using non-linear regression analysis with a power function of the form \( y = 1 - \alpha (1 - x)^\beta \). The curve is fitted to the experimental data using least-squares optimization, yielding the constants \( \alpha = 0.78 \) and \( \beta = 0.16 \) with a coefficient of determination \( R^2 = 0.97 \).

- introduce blood pool dating method

The blood pool dating method is a systematic procedure that combines image acquisition, environmental measurement, parameter extraction, and computational modeling to determine the time elapsed since the formation of a blood pool. The method is designed for field use, requiring only a camera, thermometer, hygrometer, and a portable computing device running specialized software. It transforms visual observations into quantitative temporal estimates with a margin of error of ±30 minutes under validated conditions.

- derive drying model equation

The drying model equation is derived by combining Fick’s law of diffusion with the empirical mass-area correlation, the Knudsen layer correction, and the shape factor. The evaporation flux is expressed as a function of the diffusion coefficient, which is modulated by environmental conditions and pool geometry. By integrating this flux over time and relating it to the change in mass, the equation for elapsed time is obtained, yielding a closed-form solution that depends only on measurable parameters.

- explain blood's diffusion coefficient

The diffusion coefficient of blood represents the rate at which water vapor moves through the porous gel matrix formed after coagulation. It is not a property of pure water but of the blood gel, which exhibits reduced permeability due to the presence of red blood cells and fibrin networks. The coefficient is experimentally determined to be approximately \( 1.1 \times 10^{-9} \, \text{m}^2/\text{s} \) under standard conditions and varies predictably with temperature and humidity.

- define Knudsen layer

The Knudsen layer is a thin region of vapor immediately adjacent to the liquid-vapor interface where molecular collisions dominate over bulk fluid dynamics. In the context of blood pool drying, it represents the zone through which water molecules diffuse before entering the ambient air. Its thickness is inversely proportional to atmospheric pressure and directly proportional to temperature, and it serves as the characteristic length scale for converting the mass transfer coefficient into a diffusion coefficient.

- define shape factor

The shape factor is a dimensionless parameter that quantifies the geometric influence of a blood pool on its drying dynamics. It is defined as the ratio of the pool’s area to the product of its perimeter and average height. A higher shape factor indicates a more compact, circular pool, while a lower value indicates a more elongated or irregular shape. The shape factor corrects for differences in evaporation efficiency caused by pool geometry.

- describe method for dating blood pools

The method for dating blood pools involves capturing an image of the pool, measuring the wet area and environmental conditions, retrieving the corresponding diffusion coefficient from the database, calculating the initial mass from the area and average height, determining the current mass using the mass-area correlation function, and applying the drying model equation to compute the elapsed time. The process is automated in software that integrates image processing, environmental input, and mathematical computation into a single user interface.

- determine drying front of blood pool

The drying front is determined by analyzing the image of the blood pool using color segmentation to identify the transition zone between the red, wet region and the black, dried residue. Edge detection algorithms are applied to isolate the contour of this transition, which defines the boundary of the wet area. The position and shape of the drying front are critical for accurate wet area measurement.

- determine time elapsed between given time and initial time

The time elapsed between the given time of image capture and the initial time of blood deposition is calculated by solving the drying model equation using the measured wet area, environmental parameters, and known constants. The result is expressed in hours and minutes and displayed to the user as the estimated time since deposition.

- describe system for dating blood pools

The system for dating blood pools comprises a digital camera, a portable thermometer and hygrometer, a mobile computing device running image processing and calculation software, and a cloud-accessible database of blood pool parameters. The system is designed for field deployment and requires no external power beyond the device batteries. All components are compact, rugged, and calibrated for forensic use.

- determine wet perimeter of blood pool

The wet perimeter is determined by applying edge detection algorithms to the binary mask of the wet region generated by image processing software. The software traces the contour of the drying front and computes its length in millimeters, which is then used in the calculation of the shape factor.

- determine wet area of blood pool

The wet area is determined by applying a color threshold to the image to isolate pixels corresponding to the red hue of undried blood. The resulting binary mask is processed to remove noise and fill gaps, and the total area of the connected region is calculated in square millimeters. The wet area is updated with each new image taken during the drying process.

- correlate mass variation with wet area

Mass variation is correlated with wet area through a power-law function derived from empirical data, which allows the current mass of the pool to be estimated from the measured wet area. This correlation is the cornerstone of the method, enabling time estimation without direct mass measurement.

- describe environmental conditions

Environmental conditions include ambient temperature, relative humidity, and atmospheric pressure, all of which affect the rate of water vapor diffusion from the blood pool. These parameters are measured using calibrated instruments and input into the system to select the appropriate diffusion coefficient from the database.

- describe initial blood pool conditions

Initial blood pool conditions include the initial volume, area, and height of the pool at the moment of deposition. These are estimated from the final wet area and the known average height for the substrate, assuming the pool was initially circular and had not yet begun to shrink.

- provide database of measured blood pool parameters

The database contains over 200 entries of measured blood pool parameters, including temperature, humidity, substrate type, initial mass, initial area, wet area over time, drying front progression, shape factor, and calculated diffusion coefficient. Each entry is tagged with the donor’s hematocrit level and the elapsed time. The database is continuously updated with new experimental data to improve accuracy.

- compare determined conditions with database

The system compares the measured environmental conditions and wet area with the database entries using an interpolation algorithm to find the closest matching conditions. The corresponding mass-area correlation function and diffusion coefficient are then retrieved for use in the time calculation.

- repeat steps to obtain average elapsed time

To improve accuracy, the method may be repeated multiple times using images taken at different intervals. Each calculation yields a time estimate, and the system computes the mean and standard deviation of these estimates to provide a statistically robust time value with an associated confidence interval.

- describe system for dating blood pools

The system for dating blood pools is a portable, integrated hardware-software platform comprising a high-resolution camera, digital thermometer, hygrometer, and a tablet or smartphone running proprietary software. The software automates image analysis, environmental data input, database lookup, and time calculation, presenting the result in a user-friendly format.

- use image processing software

The system employs image processing software that automatically detects the wet area and perimeter of the blood pool from digital images using color segmentation and edge detection algorithms. The software corrects for scale, lighting, and perspective distortion, ensuring accurate geometric measurements.

- use calculation software

The calculation software implements the drying model equation and the mass-area correlation function to compute the elapsed time based on input parameters. The software is optimized for speed and accuracy, performing all calculations in under 10 seconds on a standard mobile device.

- include database in system

The database of measured blood pool parameters is embedded within the system software and is accessible offline. It is regularly updated via secure cloud synchronization to incorporate new experimental data and improve predictive accuracy.

- include camera in system

The system includes a calibrated digital camera with a resolution of at least 5 megapixels, capable of capturing high-contrast images of blood pools under ambient lighting. The camera is mounted on a tripod or handheld with a reference scale for accurate scaling.

- include thermometer and hygrometer in system

The system includes a compact, calibrated digital thermometer and hygrometer that measure ambient temperature and relative humidity to the nearest 0.1°C and 1% respectively. These devices are wirelessly synchronized with the calculation software to ensure automatic data input.

- describe method for dating blood pools

The method for dating blood pools involves capturing an image of the blood pool, measuring environmental conditions, extracting the wet area and perimeter using image processing, retrieving the appropriate diffusion coefficient from the database, estimating the initial mass, calculating the current mass using the mass-area correlation, and applying the drying model equation to determine the elapsed time.

- determine drying front of blood pool

The drying front is determined by analyzing the image of the blood pool using color segmentation to identify the transition zone between the red, wet region and the black, dried residue. Edge detection algorithms are applied to isolate the contour of this transition, which defines the boundary of the wet area.

- determine time elapsed between given time and initial time

The time elapsed between the given time of image capture and the initial time of blood deposition is calculated by solving the drying model equation using the measured wet area, environmental parameters, and known constants. The result is expressed in hours and minutes and displayed to the user as the estimated time since deposition.

- describe system for dating blood pools

The system for dating blood pools is a portable, integrated hardware-software platform comprising a high-resolution camera, digital thermometer, hygrometer, and a tablet or smartphone running proprietary software. The software automates image analysis, environmental data input, database lookup, and time calculation, presenting the result in a user-friendly format.

- use image processing software

The system employs image processing software that automatically detects the wet area and perimeter of the blood pool from digital images using color segmentation and edge detection algorithms. The software corrects for scale, lighting, and perspective distortion, ensuring accurate geometric measurements.

- use calculation software

The calculation software implements the drying model equation and the mass-area correlation function to compute the elapsed time based on input parameters. The software is optimized for speed and accuracy, performing all calculations in under 10 seconds on a standard mobile device.

- include database in system

The database of measured blood pool parameters is embedded within the system software and is accessible offline. It is regularly updated via secure cloud synchronization to incorporate new experimental data and improve predictive accuracy.

- include camera in system

The system includes a calibrated digital camera with a resolution of at least 5 megapixels, capable of capturing high-contrast images of blood pools under ambient lighting. The camera is mounted on a tripod or handheld with a reference scale for accurate scaling.

- include thermometer and hygrometer in system

The system includes a compact, calibrated digital thermometer and hygrometer that measure ambient temperature and relative humidity to the nearest 0.1°C and 1% respectively. These devices are wirelessly synchronized with the calculation software to ensure automatic data input.

- describe method for dating blood pools

The method for dating blood pools involves capturing an image of the blood pool, measuring environmental conditions, extracting the wet area and perimeter using image processing, retrieving the appropriate diffusion coefficient from the database, estimating the initial mass, calculating the current mass using the mass-area correlation, and applying the drying model equation to determine the elapsed time.

- determine drying front of blood pool

The drying front is determined by analyzing the image of the blood pool using color segmentation to identify the transition zone between the red, wet region and the black, dried residue. Edge detection algorithms are applied to isolate the contour of this transition, which defines the boundary of the wet area.

- determine time elapsed between given time and initial time

The time elapsed between the given time of image capture and the initial time of blood deposition is calculated by solving the drying model equation using the measured wet area, environmental parameters, and known constants. The result is expressed in hours and minutes and displayed to the user as the estimated time since deposition.

- describe system for dating blood pools

The system for dating blood pools is a portable, integrated hardware-software platform comprising a high-resolution camera, digital thermometer, hygrometer, and a tablet or smartphone running proprietary software. The software automates image analysis, environmental data input, database lookup, and time calculation, presenting the result in a user-friendly format.

- use image processing software

The system employs image processing software that automatically detects the wet area and perimeter of the blood pool from digital images using color segmentation and edge detection algorithms. The software corrects for scale, lighting, and perspective distortion, ensuring accurate geometric measurements.

- use calculation software

The calculation software implements the drying model equation and the mass-area correlation function to compute the elapsed time based on input parameters. The software is optimized for speed and accuracy, performing all calculations in under 10 seconds on a standard mobile device.

- include database in system

The database of measured blood pool parameters is embedded within the system software and is accessible offline. It is regularly updated via secure cloud synchronization to incorporate new experimental data and improve predictive accuracy.

## DETAILED DESCRIPTION OF EMBODIMENTS

- introduce embodiments of the present disclosure

Embodiments of the present disclosure encompass a complete system and method for dating blood pools using a combination of image capture, environmental sensing, and computational modeling. These embodiments are designed for practical, field-deployable use by forensic investigators and are implemented through integrated hardware and software components that operate in concert to deliver accurate, repeatable, and legally defensible time estimates.

- provide non-limiting example of a system for dating a blood pool

A non-limiting example of the system comprises a smartphone equipped with a high-resolution camera, a Bluetooth-connected digital thermometer and hygrometer, and a dedicated application running image processing and calculation software. The smartphone is used to capture images of the blood pool, while the sensors automatically transmit environmental data to the application. The application processes the image, extracts the wet area and perimeter, retrieves the appropriate diffusion coefficient from the embedded database, and computes the elapsed time using the drying model equation.

- describe system components, including smartphone and image processing software

The system components include a smartphone with a minimum 12-megapixel camera, a calibrated Bluetooth thermometer and hygrometer, and a mobile application that integrates image processing and calculation modules. The image processing software is built using OpenCV and TensorFlow libraries to perform color segmentation, edge detection, and contour analysis. The software corrects for lens distortion, applies scale calibration from a reference marker, and outputs the wet area and perimeter in real time.

- explain image processing software functionality

The image processing software functions by first applying a hue-saturation-value threshold to isolate the red component of the blood pool, then using morphological operations to remove noise and fill gaps in the binary mask. The software detects the outer boundary of the wet region using the Canny edge detector and computes the area and perimeter of the resulting contour. The wet area is displayed to the user with a visual overlay on the original image.

- describe calculation software functionality

The calculation software receives the wet area, perimeter, and environmental data from the image processing module and retrieves the corresponding diffusion coefficient from the embedded database. It estimates the initial mass using the average height for the substrate and computes the current mass using the mass-area correlation function. The software then solves the drying model equation to determine the elapsed time, displaying the result with a confidence interval.

- introduce database of measured blood pool parameters

The database contains over 200 experimental records of blood pools dried under controlled conditions, including temperature, humidity, substrate type, initial mass, wet area over time, shape factor, and calculated diffusion coefficient. Each record is associated with the donor’s hematocrit level and the exact elapsed time. The database is stored locally on the device and is updated via secure cloud sync.

- describe database contents, including environmental and initial blood pool conditions

The database includes entries for temperature ranging from 18°C to 30°C, relative humidity from 15% to 70%, and substrates including tile, linoleum, and varnished wood. For each condition, the database stores the average initial height, the shape factor distribution, the diffusion coefficient, and the empirical mass-area correlation constants. The data is stratified by donor sex and hematocrit level to account for biological variability.

- provide example of system configuration, including thermometer and hygrometer

An example system configuration includes a Samsung Galaxy S21 smartphone, a Bluetooth-enabled HOBO U12-012 thermometer and hygrometer, and a custom Android application. The thermometer and hygrometer are placed within 10 cm of the blood pool and automatically synchronize with the app upon startup. The app displays real-time environmental readings and prompts the user to capture an image once conditions are stable.

- introduce method for dating a blood pool

The method for dating a blood pool begins with the placement of the thermometer and hygrometer near the pool, followed by capturing a high-resolution image with a reference scale. The user inputs the substrate type, and the application automatically processes the image to determine the wet area and perimeter. The system then computes the elapsed time using the embedded database and drying model, displaying the result within 10 seconds.

- describe method steps, including determining drying front and time elapsed

The method steps include: (1) placing environmental sensors near the blood pool; (2) capturing a digital image with a reference scale; (3) selecting the substrate type in the application; (4) initiating automated image processing to determine wet area and perimeter; (5) retrieving the diffusion coefficient and mass-area correlation from the database; (6) computing the initial mass from area and average height; (7) calculating the current mass using the correlation function; (8) solving the drying model equation to determine elapsed time; and (9) displaying the result with a confidence interval.

- explain drying model, including blood's diffusion coefficient

The drying model is based on the diffusion of water vapor through the porous gel matrix of the blood, with the diffusion coefficient corrected for environmental conditions and pool geometry. The coefficient is derived from experimental measurements and is stored in the database as a function of temperature and humidity. The model accounts for the Knudsen layer thickness and the shape factor to ensure accurate scaling across different pool sizes and shapes.

- describe experimental measurement of blood's diffusion coefficient

The diffusion coefficient of blood was measured experimentally by drying over 200 pools under controlled conditions and calculating the evaporation rate as a function of wet area. The mass transfer coefficient was derived from the evaporation rate, and the Knudsen layer thickness was computed from temperature and pressure. The diffusion coefficient was then calculated as the product of the transfer coefficient, Knudsen layer, and square root of the shape factor, yielding a consistent value of \( 1.1 \times 10^{-9} \, \text{m}^2/\text{s} \).

- introduce equation correlating mass variation with wet area

The equation correlating mass variation with wet area is:  
\[ \frac{m}{m_i} = 1 - 0.78 \left[1 - \left( \frac{A_x}{A_i} \right) \right]^{0.16} \]  
This equation was derived from regression analysis of 200 experimental datasets and has a correlation coefficient of 0.97, making it highly reliable for estimating current mass from wet area.

- describe determination of wet perimeter and wet area

The wet perimeter and wet area are determined by image processing software that applies color segmentation to isolate the red region of the blood pool, followed by edge detection to trace the drying front. The perimeter is computed as the length of the contour, and the area is computed as the number of pixels within the contour, scaled to square millimeters using the reference marker.

- explain calculation of time elapsed

The calculation of time elapsed is performed by substituting the measured wet area, initial area, environmental conditions, and known constants into the drying model equation. The software performs the computation in real time, using pre-validated constants and database values to ensure accuracy. The result is displayed in hours and minutes with a margin of error of ±30 minutes.

- provide numerical values for blood pool parameters

For a blood pool on tile at 23°C and 20% humidity, the average height is 1.44 mm, the shape factor is 1.21, the diffusion coefficient is \( 1.1 \times 10^{-9} \, \text{m}^2/\text{s} \), and the constants \( \alpha = 0.78 \) and \( \beta = 0.16 \). The Knudsen layer thickness is 0.098 μm, and the saturation vapor pressure is 2.81 kPa.

- introduce validation test of the method

A validation test was conducted by depositing 10 mL of blood on a white tile under controlled conditions and capturing images every 15 minutes for 10 hours. The system was used to compute the elapsed time for each image, and the results were compared to the actual time. The mean absolute error was 19 minutes, with a standard deviation of 12 minutes, confirming the method’s accuracy within the required ±30 minute margin.

- describe test setup, including blood spill on white tile

The test setup involved a controlled environment chamber with temperature maintained at 23.5°C and humidity at 20%. A 10 mL blood sample from a healthy donor was deposited on a clean white ceramic tile. A reference scale and environmental sensors were placed adjacent to the pool. Images were captured every 15 minutes using a calibrated smartphone camera.

- explain user input, including uploading picture and filling out environmental parameters

The user uploads the captured image to the application via the smartphone interface and selects the substrate type from a dropdown menu. The application automatically retrieves the environmental data from the Bluetooth sensors. The user is prompted to confirm the image quality and may retake the image if necessary. No manual measurement or calculation is required.

- describe image processing and extraction of blood pool parameters

The image processing software segments the red region of the blood pool, applies edge detection to trace the drying front, and computes the wet area and perimeter. The software corrects for scale using the reference marker and outputs the parameters to the calculation module. The average height is retrieved from the database based on substrate type.

- explain calculation of correlation coefficients and blood's diffusion coefficient

The software retrieves the diffusion coefficient from the database based on temperature and humidity. The correlation coefficient between wet area and mass is computed using the pre-established equation. The system does not recalculate these values in real time but uses validated constants stored in the database to ensure consistency.

- describe calculation of elapsed time

The elapsed time is calculated by solving the drying model equation using the wet area, initial area, diffusion coefficient, and environmental parameters. The software performs the computation in under 5 seconds and displays the result as “Estimated time since deposition: 8 hours 18 minutes ± 19 minutes.”

- provide results of validation test, including margin of error

The validation test results showed a mean absolute error of 19 minutes across 40 measurements, with a standard deviation of 12 minutes. The maximum error was 42 minutes, and the minimum error was 4 minutes. The method achieved an accuracy of ±30 minutes in 95% of trials, meeting the required forensic standard.

- introduce possibility of taking multiple pictures to obtain average elapsed time

Multiple images may be taken at different times, and the system computes an elapsed time for each. The application then calculates the mean and standard deviation of these estimates, providing a statistically robust time value. This feature reduces the impact of measurement error and improves reliability.

- describe potential for automated repetition of calculation steps

The system can be programmed to automatically capture images at regular intervals (e.g., every 10 minutes) using a timer function. The software processes each image in sequence, computes the elapsed time, and updates the running average. This enables continuous monitoring of the drying process without user intervention.

- explain importance of rich database for method accuracy

The accuracy of the method is critically dependent on the size and quality of the database. A larger, more diverse database improves interpolation accuracy, accounts for biological variability, and extends the range of valid environmental conditions. Continuous updates ensure the system remains reliable as new data becomes available.

- introduce diagram correlating KiLkL*0.5 with percentage of water left in blood pools

A diagram is included in the system interface that plots the product of the transfer coefficient, Knudsen layer, and square root of the shape factor against the percentage of water remaining in the blood pool. This diagram shows a plateau at approximately \( 1.1 \times 10^{-9} \, \text{m}^2/\text{s} \), confirming the consistency of the diffusion coefficient across varying conditions.

- present table of blood's diffusion coefficients for different humidities and substrates

| Humidity (%) | Temperature (°C) | Substrate | Diffusion Coefficient (m²/s) |
|--------------|------------------|-----------|------------------------------|
| 15           | 20               | Tile      | 1.21 × 10⁻⁹                  |
| 20           | 23               | Tile      | 1.10 × 10⁻⁹                  |
| 30           | 25               | Tile      | 9.8 × 10⁻¹⁰                  |
| 20           | 23               | Linoleum  | 1.13 × 10⁻⁹                  |
| 20           | 23               | Wood      | 1.09 × 10⁻⁹                  |
| 40           | 28               | Tile      | 8.5 × 10⁻¹⁰                  |

This table is embedded in the system database and used to select the appropriate diffusion coefficient during calculation.