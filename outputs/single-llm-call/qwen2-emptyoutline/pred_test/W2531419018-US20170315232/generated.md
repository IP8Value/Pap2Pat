# DESCRIPTION

## BACKGROUND

High-frequency (HF) radars have been utilized for ocean observations since the 1960s. These radars, located on the coast and transmitting vertically polarized radiations, exploit the high conductivity of seawater to propagate signals beyond the visible or microwave-radar horizon. They are widely used for mapping surface currents and monitoring sea state. In 1979, Barrick suggested that these radars could detect tsunamis by measuring the orbital wave velocity as they approach the coast. However, the sparse distribution of radars until the 1990s hindered the practical implementation of this concept. The catastrophic 2004 Banda Aceh earthquake in Indonesia, which claimed a quarter of a million lives, reignited interest in this application. Although no radars were in place to observe that event, subsequent work focused on quantifying the radar tsunami response. The 2011 Tohoku (Japan) tsunami provided the first opportunity to capture real tsunami data with multiple HF radars around the Pacific Rim, leading to the development of robust detection and warning algorithms. These algorithms have been refined using a database of actual HF radar tsunami observations, including both strong and weak tsunamis. Over 350 HF radar stations operate globally, providing continuous measurement of surface current velocities and waves. Tsunami detection software can run in a background mode, issuing warnings before the tsunami strikes the coast.

The first possible indication of a tsunami might be the seismic detection of an earthquake. However, not all subsea earthquakes generate tsunamis, and the magnitude of an earthquake cannot be used to forecast the detailed generation or intensity of a resulting tsunami. Currently, the only operational sensor that detects and measures the intensity of a tsunami is a bottom pressure sensor connected to a buoy overhead. Developed by the National Oceanic and Atmospheric Administration (NOAA), networks of these sensors, known as DART™ (Deep-ocean Assessment and Reporting of Tsunami), were deployed after the 2004 event. These sensors observe the height of the tsunami wave as it passes above them. The tsunami height measured by these buoys is then entered into numerical tsunami models to provide rough forecasts of the tsunami's arrival time and intensity at coastal points around the world. However, as these networks are located in the deep ocean, not all tsunamis are observable by DART before coastal impact. Moreover, the model's forecast of intensity at the coast is often coarse, necessitating more accurate estimates of intensity at specific locations. HF radars, which make areal observations over the local near-field, provide an ideal solution to this need. They can detect and warn of an approaching tsunami in the near-shore region over which these radars observe the sea surface. A total of 21 offline radar detections of tsunamis have been made to date, and many are described in the literature. A tsunami's orbital velocity appears as part of the surface current as the wave approaches the coast. Tsunami periods typically range from 20 to 50 minutes. A tsunami originates from a massive displacement of water, which can result from subsea earthquakes, landslides, or atmospheric anomalies. The spatial scales of water displacement are usually large horizontally but small vertically. As the displaced water mass leaves its source region under the influence of gravity, it becomes a freely propagating shallow-water wave. Tsunami warning times are primarily dependent on the width of the adjacent continental shelf, ranging from minutes for a narrow shelf (e.g., California) to hours for a broad shelf (e.g., New Jersey). Some sites may be less suitable for tsunami monitoring by radar, as the tsunami signature can be masked by large, variable background currents. Tsunami detection is favored by shallow water extending far offshore and by slowly varying background current fields. We describe a method for evaluating a coastal site for tsunami warning based on simulated tsunami velocities superimposed on the site's measured velocities. Factors affecting radar detection of tsunamis are discussed, and difficulties that can occur in tsunami detection and methods for alleviation are described. Work on the evaluation of coastal sites for tsunami warning using HF radars is being performed in a partnership between Codar Ocean Sensors and NOAA.

## SUMMARY

The invention relates to a method and system for detecting and warning of approaching tsunamis using high-frequency (HF) coastal radars. The method involves analyzing the orbital velocity of the tsunami wave as it approaches the coast, which is measured by the radar. The system includes a network of HF radars that continuously monitor surface currents and waves. The invention utilizes an empirical detection algorithm that runs in the background on these radars, identifying the tsunami's orbital velocity signature in the background ocean current velocity field. The algorithm is based on pattern recognition in the velocity time series and can issue a warning before the tsunami strikes the coast. The invention also includes methods for evaluating the suitability of coastal sites for tsunami detection using simulated tsunami velocities and for reducing false alarms through correlation with earthquake notifications and data from adjacent radars. The invention aims to provide early local detection of incoming tsunamis, thereby enhancing the effectiveness of tsunami warning systems.

## DETAILED DESCRIPTION

The invention provides a method and system for detecting and warning of approaching tsunamis using high-frequency (HF) coastal radars. The method involves the continuous monitoring of surface currents and waves by a network of HF radars, which are strategically placed along coastal areas. The system utilizes an empirical detection algorithm that runs in the background on these radars, identifying the tsunami's orbital velocity signature in the background ocean current velocity field. The algorithm is based on pattern recognition in the velocity time series and can issue a warning before the tsunami strikes the coast.

### Tsunami Theory and Modeling Applicable to HF Radar Observations

#### Fundamental Equations Describing a Tsunami in the Near-Field Region

Two primary equations form the basis of tsunami wave theory and propagation modeling. The first is Newton's second law, which, for fluids, gives the Navier-Stokes vector equation to the lowest order. For horizontal coordinates \(x\) and \(y\) and time \(t\), this is given by:

\[
\frac{\partial \eta}{\partial t} + \nabla \cdot (\eta \mathbf{u}) = 0
\]

where \(\eta(x, y, t)\) is the tsunami wave height, \(g\) is the acceleration due to gravity, and \(\mathbf{u}\) is the orbital velocity of the wave. The second equation is the continuity equation that expresses the incompressibility of water:

\[
\frac{\partial \eta}{\partial t} + \nabla \cdot (d \mathbf{u}) = 0
\]

where \(d(x, y)\) is the depth below a mean datum reference, from which the tsunami wave height is measured. The left side of this equation is the net horizontal water volume transport per unit time into a vertical column. Under the assumption that the orbital velocity at a particular location is independent of depth, the equation becomes linear.

#### Reduction to Partial Differential Equations (PDEs)

The coupled equations in the unknown tsunami wave height and orbital velocity can be simplified by differentiating with respect to time and/or space, eliminating one variable. This results in the following two hyperbolic PDEs for tsunami height and velocity:

\[
\frac{\partial^2 \eta}{\partial t^2} - g \nabla \cdot (d \nabla \eta) = 0
\]

\[
\frac{\partial \mathbf{u}}{\partial t} + g \nabla \eta = 0
\]

These equations are well-known for waves in shallow water and are justified when the water depth is much less than the horizontal scale of the water wave. The time scales (periods) for tsunami waves that represent hazards vary from 20 to 50 minutes.

#### Ray Optics and Green's Law Approximations for Tsunami Waves

Tsunami waves can sometimes follow simple ray optics approximations, where they refract and change direction continuously with the refractive index. The refractive index for waves of any nature propagating through media with different or changing properties is defined as the ratio of the reference phase velocity to the phase velocity at the specific point in the medium. For light or electromagnetic waves, the reference velocity is the speed of light in a vacuum. For acoustic waves or water waves in the shallow-depth limit, a convenient reference velocity is typically selected. For example, if the 4000-meter depth is taken as typical of a deep ocean basin, the refractive index becomes \(4000/d\), where \(d\) is the depth at a specific point.

When depth and refractive index vary slowly, ray tracing allows a version of Fresnel's law such that the advancing wave continuously refracts, so that its direction of propagation follows the gradient of the refractive index perpendicular to the isobath depth contours. This approximation has the consequence that there is always only one set of ray paths, which end up perpendicular to the coastline. The coastline boundary will reflect, and outgoing rays also cross the contours perpendicularly.

When depth and refractive index vary abruptly, it is valid to use the PDEs as an alternative. Models based on these equations will predict the direction correctly and will generate components parallel to the contours and coastline, as has been observed by radars.

Green's Law applies when the refractive index (depth) varies slowly. In this limit, tsunami height and orbital velocity follow simple relationships in terms of depth. The approximate tsunami wave height and scalar tsunami orbital speed in water of depth \(d\) are given by:

\[
\eta(d) = \eta_{4000} \left( \frac{4000}{d} \right)^{1/4}
\]

\[
u(d) = u_{4000} \left( \frac{4000}{d} \right)^{3/4}
\]

where \(\eta_{4000}\) and \(u_{4000}\) are the tsunami height and orbital velocity in water of depth 4000 meters.

### HF Radar Observations of the 2011 Japan Tsunami Leading to an Empirical Detection Algorithm

Radar echoes are produced by reflection of the radar wave from ocean waves with wavelengths half that of the radar, which have periods between 1.5 and 4.5 seconds. In contrast, the tsunami wave period lies between 20 and 50 minutes, corresponding to wavelengths between 400 and 800 kilometers in the open ocean. Tsunami orbital velocities add to the shortwave radial velocities producing the radar echo. In deep water, tsunami velocities are too small to be seen by the radar, but they increase as the tsunami moves onto the continental shelf and the water depth decreases below 200 meters, making them observable by radars located on the coast.

Radial current velocities are obtained from the first-order radar echo spectra measured at individual radar sites. In usual practice, several radar echo spectra are averaged over time before analysis. As time resolution is critical for local tsunami detection, unaveraged spectra are analyzed. The Doppler shift from the ideal Bragg frequency defines the radial current speed; spectral values at that frequency are interpreted to give the azimuth angles at which this speed occurs. Together with range defined by the time delay, estimates follow for the radial current velocity at locations spaced 1° apart around a circular range cell centered on the radar site.

Total current velocities are obtained by combining radial velocities from the radar sites. A grid is formed over the radar coverage area, and averaging circles surround each grid point. Total velocity vector components are calculated by fitting to radial velocities from the different radar sites that fall within the averaging circle.

On March 11, 2011, at 14:46 Japan Standard Time (JST), a magnitude-9 earthquake off Sendai, Japan, unleashed a large tsunami that was observed by HF radars around the Pacific Rim. The radars had a transmit frequency of 42 MHz and a range increment of 0.5 km. The water depth over the entire radar coverage area is less than 200 meters. The 42 MHz frequency band is used for high-resolution, short-range current observations and results in a radar range less than 15 km, due to significant attenuation of the surface wave passing across the sea at these higher HF frequencies.

#### Total Velocity Current Maps

The direction and strength of the flow were measured at approximately 4-minute intervals with a cell resolution of 0.5 km × 0.5 km. Total current-velocity maps show the arrival of the tsunami, indicated by strong inward flow, and an example of outward flow. The tsunami height is displayed in colors superimposed on the velocity vectors shown by the arrows. As noted, the accuracy of Green's Law estimates of height decreases close to shore.

A video showing the current velocity/height flow from March 11, 14:06 JST to March 12, 13:54 JST is available. The video shows the tsunami arriving at 15:53 JST and then sweeping in and out of Uchiura Bay.

#### Radial Velocity Components

The tsunami signal is also visible in the radar returns from a single radar site. To simplify the analysis of the data with the aim of developing objective detection criteria, the radial velocities are grouped into rectangular area bands 2 km wide approximately parallel to the depth contours. The radial velocities are resolved into components perpendicular and parallel to the area bands. These components are averaged over the band, and the averages are termed "band velocities." A time series of the band velocities is then formed, which displays the characteristic oscillations produced by the tsunami.

#### Detection of the Tsunami Signal in HF Radar Data

Two effects distinguish tsunami velocities from the background: (a) velocities in neighboring bands are strongly correlated after the arrival of the tsunami, and (b) the velocity oscillations are clearly visible above the background. These effects form the basis of a simple pattern detection procedure. At a given time, a factor (which we call the q-factor) is defined which signals the tsunami arrival when it exceeds a preset threshold. The steps in the detection algorithm are as follows:

1. Within each band, check whether the velocity increases or decreases by an amount greater than a preset level over two consecutive time intervals. If it does, increase/decrease the q-factor level for that band.
2. Do the maximum/minimum velocities for consecutive bands coincide (within a preset value) for consecutive time intervals? If so, increase/decrease the q-factor level further for that band and time.
3. Finally, check whether the velocity increases/decreases over two consecutive time intervals for three adjacent area bands. If so, increase/decrease the q-factor level further for that band/time.

Positive q-factor values indicate the tsunami velocity at the wave peak is moving toward the radar, negative values indicate that it is moving away.

To set the operational threshold signaling a tsunami detection, an extended data set obtained under normal conditions is analyzed to produce q-factors. A threshold value is then selected. There is a trade-off in the threshold selection: if the q-factor limit is set too low, the peak will certainly indicate a detection, but there may be many false-alarm detections. If the threshold is set too high, there will be few false alarms, but then the tsunami arrival may not be detected.

### Radar Detection of the 2011 Japan Tsunami

The Japan tsunami arrival was detected offline at radar sites around the northern Pacific Rim. Examples of q-factor tsunami detections and comparisons of arrival times at the radars with those measured by neighboring tide gauges are provided.

#### Hokkaido, Japan

Figure 6 shows the locations of two radars on the Kameda Peninsula, the neighboring tide gauge, and the offshore bathymetry. The water depth is less than 200 meters over the radar coverage area, and the tsunami signal is visible in the current velocities out to the radar range limits. Figure 7 shows A088 band velocities obtained over a 5-hour period and the q-factors resulting from the analysis. About an hour after the earthquake, the tsunami arrived at A088, resulting in distinctive correlated oscillations in the perpendicular band velocities, which lead to a q-factor peak indicating the tsunami arrival.

Close to shore, part of the tsunami flow is diverted by the steep bathymetry to move parallel to the coast, resulting in a reduced signal in the perpendicular component plotted in Figure 7. The analysis procedure was applied to A087 and A088 for all permutations of three band velocities that contained the tsunami signal, and the resulting q-factors were summed. The q-factor threshold was defined to be 500: the first q-factor to exceed this value was taken as defining the tsunami arrival time. Table 1 shows that the arrival times obtained from the radar q-factors reported are in the correct order: the tsunami arrives at Station A087 further from the earthquake location approximately 5 minutes after it reaches A088. Arrival times measured by the radars preceded those at the neighboring tide gauge by an average of 40 minutes, due to both the "quadrature relation" between velocity and height and the tsunami propagation delay between the two observations.

#### West Coast of USA

Radar spectra measured by 10 radars located along the US West Coast were analyzed to give band velocities and q-factors. Arrival times were compared with those at local tide gauges. Figure 8 shows radar and tide gauge locations and the offshore bathymetry. As the adjoining continental shelf is narrow off California and Oregon, the tsunami is often detectable only for close-in ranges. Two examples of measured band velocities and derived q-factors are provided.

The first example is the tsunami detection by the radar at YHS2, Oregon (transmit frequency 12 MHz). Figure 9 shows the band velocities and corresponding q-factors. The correlation is evident between the velocities in different bands starting at about 3:45 pm Coordinated Universal Time (UTC), resulting in a sharp decrease in the q-factor, which indicates the tsunami moving offshore, resulting in a decrease in water level. The neighboring South Beach tide gauge observed an initial water level increase due to the tsunami of just 0.3 meters, which was inadequate to produce a radar detection. However, the band velocities show the typical correlation due to the tsunami just before the sharp decrease.

The second example is the tsunami detection by the radar at ESTR in Southern California (transmit frequency 13 MHz). Figure 10 shows the band velocities and q-factors for ESTR. The observed background current velocities are quite variable for this site: it is the correlations between velocities in different bands that allow the tsunami to be detected by the pattern recognition algorithm described earlier. Table 2 shows that listed arrival times obtained from the radar q-factors reported are normally in the correct order; thus, it arrives in Southern California after it gets to Northern California and Oregon. Arrival times measured by the radars preceded those at neighboring tide gauges by an average of 15 minutes, due to both the "quadrature relation" between velocity and height and the tsunami propagation delay between the two observations. The tsunami was detected even though water-level changes at neighboring tide gauges were not large, varying between 0.3 and 2 meters.

#### Chile

A WERA radar system operating at 22 MHz at a site near Concepcion, Chile, observed the Japan tsunami. Current components pointing toward/away from the radar were measured within beams formed by the receiving antenna array. The orbital velocity of the shallow-water tsunami wave is part of the total signal, which also includes other background contributions such as tides and geostrophic flow. Figure 11 shows clear periodic disturbances produced by the tsunami in both radar and tide gauge observations. The tsunami component of these currents is identified from their typical periods that lie between 20 and 45 minutes, arriving at about 05:07 UTC on March 12, approximately 22 hours after the Japan earthquake. This arrival time was confirmed by NOAA's tsunami model and the tide gauge data.

### Radar Observations of the 2013 US East Coast Meteotsunami

An unusual storm system moved eastward across the US on June 13, 2013, commonly called a "derecho," and appears to have launched a meteotsunami that impacted the US East Coast. The existence of the meteotsunami was confirmed by several of the 30 tide gauges along the East Coast up through New England and was seen as far away as Puerto Rico and Bermuda. The event, which occurred during daylight hours, attracted widespread attention after several media reports were released focusing on local impacts, including people being swept off a breakwater at Barnegat Light, New Jersey, some damage to boat moorings, and minor inundation.

Meteotsunamis generally do not have sufficient heights/energies to cause catastrophic loss of life, as do severe seismic tsunamis, although damage to harbors and coastal structures is common. The June 13, 2013 event, however, attracted significant attention among many agencies and scientific groups, probably due to its proximity to heavily populated areas.

#### Origin of Meteotsunamis and Nature of the June 13, 2013 Event

A meteotsunami is generated by an atmospheric pressure disturbance traveling across the sea. An atmospheric anomaly (a low- or high-pressure center) will produce a small peak or trough moving at the same speed on the sea surface beneath it. This results in a freely propagating surface wave that increases in amplitude when the speed of the atmospheric anomaly \(v_{aa}\) matches the shallow-water wave phase velocity \(v_{ph}(d)\). This is known as Proudman resonance. The speed \(v_{aa}\) of the June 13, 2013 derecho was about 21.1 m/s. Substituting this value for \(v_{ph}(d)\) into the phase velocity equation, it follows that the onset of the independent wave occurs at a depth \(d\) equal to 45 meters, which lies about 60 kilometers off the New Jersey coast.

This meteotsunami was unusual because it was generated by a frontal pressure anomaly traveling offshore. Yet coastal sensors, including HF radars, indicate that the meteotsunami approached the coast. Numerical models indicate that a strong reflection occurred at the shelf edge about 110-120 kilometers offshore, where the depth decreases from 100 to 1200 meters over a distance of 20 kilometers. The reflection is greater when a wave interacts with a drop-off rather than a step-up with the same slope. Data from New Jersey radars confirm the existence of a wave reflected from the shelf edge back toward the coast. This wave was also detected by coastal tide gauges.

To explain these results, we consider the interaction of the tsunami with a hard boundary, assuming a single pulse of water approaching the coast, that is, a traveling wave. The forward velocity is maximum at the wave crest. As a boundary is approached, there is a hard reflection: the velocity goes to zero and the height doubles. This is known as the Neumann boundary condition. After a period of time from the reflection, a single wave travels outwards, with the crest velocity and height maxima in phase again. In reality, the situation is more complex. Instead of a single wave or soliton, a series of positive and negative tsunami peaks often resemble a sine wave for height and velocity. The hard-wall boundary condition causes the height peaks to lag the velocity peaks by as much as a quarter cycle, which is termed the "quadrature effect." After reflection, the height stays positive but the velocity amplitude becomes negative. The interaction of incoming and reflected waves constitutes a more complex partial standing-wave situation, which is well handled by numerical model solutions.

#### Radar Detection of the 2013 US East Coast Meteotsunami

We analyzed data sets from three SeaSonde HF radar systems located in New Jersey: BRNT, BRMR, and BELM. Radar transmit frequencies and range cell widths were approximately 13.5 MHz and 3 kilometers, respectively. Radar results were compared with data from NOAA tide gauges at Atlantic City and Sandy Hook, New Jersey. Figure 12 shows the locations of the radars and tide gauges, and the offshore bathymetry. The meteotsunami height at the neighboring DART buoy, located about 240 kilometers to the east, was only 5 cm. Atlantic City tide gauge data obtained from the NOAA website are shown in Figure 13. Readings show a maximum negative meteotsunami signal at approximately 18:42 UTC, indicated by the sharp water-level decrease. This is followed at approximately 22:00 UTC by a sharp increase in water level and subsequent oscillations.

The radar coverage area is divided into rectangular area bands 2 kilometers wide and approximately parallel to the depth contours. Radial vectors within each area band were resolved parallel and perpendicular to the depth contour. These velocity components are then averaged over the bands. Figure 14 shows time series of four perpendicular band velocities from BRNT and BRMR and the corresponding q-factors, obtained from the four bands. The arrival of the meteotsunami is signaled by a marked decrease in the perpendicular band velocity component, indicating an outflow, followed by correlation between different area bands. The parallel component did not display the tsunami signature. The water level measured by the closest tide gauge at Atlantic City decreases when the tsunami arrives, as shown in Figure 13, also indicating an outflow of water.

The tsunami signal at BELM was far less, which is consistent with tide gauge measurements at Sandy Hook, 30 kilometers to the north, which barely registered the tsunami arrival. About 4 hours later, after 22:00 UTC, BRNT velocities first increase and then sharply decrease, as is also shown by the Atlantic City tide gauge. This effect was not seen at BRMR or BELM. To demonstrate more clearly the meteotsunami velocity trough as it approached the coast, BRNT band velocities were further processed as follows: the band velocities were first detrended over time, removing effects with time scales longer than 1.5 hours, such as those due to tides. The detrended band velocities were then low-pass filtered and, to further reduce noise, averaged over two adjacent bands.

Figure 15 shows the smoothed velocities plotted as a function of time vs. range from shore, the dashed line indicating the progression of the first tsunami trough. Tsunami hindcast modeling confirms this time-distance progression of the meteotsunami as it moved toward shore.

Figures 14 and 15 show that the tsunami arrived first at the most distant ranges and progressively later moved toward the coast. To compare these results with theory, the tsunami arrival time at BRNT was calculated using the phase velocity equation, based on an initial detection at range 23 kilometers. The bathymetry contours offshore from BRNT were approximated by parallel contours, giving depth as a function of distance. As discussed, this approximation is valid, as the tsunami is not affected by perturbations in depth with spatial scales far less than its wavelength. This analysis assumes no coastal boundary and results are expected to differ somewhat from radar-observed arrival times, as the orbital velocities are affected by shallow water. The initial velocity observed by the radars was offshore, indicating a "trough" on the ocean surface. This was also observed by the closest tide gauge at Atlantic City. However, as shown in Figure 15, the tsunami wave itself approached the coast due to a strong reflection occurring at the shelf edge 110-120 kilometers from shore. The meteotsunami was detected by the radars 23 kilometers from the coast. It arrived at the shore 47 minutes later, as indicated by the tide gauge measurement of water level shown in Figure 13. The measured tsunami height was approximately 50 centimeters. These observations suggest that for similar tsunami height and bathymetry conditions, HF radar can provide a three-quarter hour warning alert before the wave strikes the shore.

### Calculation of Simulated Tsunami Velocities and Heights

Tsunami simulation provides an understanding of many of the factors affecting the capability of coastal HF radars to provide tsunami observation and warning. Ultimately, this can lead to performance assessment for a radar at a given site based on local bathymetry. Orbital velocities are tracked vs. time and related to the tsunami wave intensity. Comparisons with the background current field allow the assessment of possible warning time and wave amplitude as the tsunami approaches the coast near the radar. In this section, we describe two methods for simulating tsunami velocities: the first based on solving the fundamental equations to give total velocity/height maps and the second based on application of Green's Law to give simulated band velocities.

#### Simulation Based on Solution of the Fundamental Equations of Motion

To simulate tsunami height and velocity, the fundamental equations of motion are solved numerically within the radar coverage area, typically out to 50 kilometers from the coast. The offshore bathymetry is included as the depth variable \(d(x, y)\), and the coastline becomes a boundary for the domain. First, the scalar equation is solved for the tsunami wave height. Then, velocity is obtained by integrating the left side of the continuity equation over time, after linearization. This establishes the relations between the orbital velocity measured by the radar and the tsunami wave height, as well as provides the time of arrival at the coast from any point in the near-field region.

##### One-Dimensional Tsunami

We examine how a simple one-dimensional wave approaching normally to the coast behaves when it encounters a steep continental slope starting from a depth of 1000 meters at a distance of 70 kilometers from shore and sloping upwards to a depth of 100 meters at a distance of 50 kilometers from shore. How do both height and orbital velocity change as they traverse this shelf? How much is transmitted across the shelf and how much gets reflected? On the return of the ray reflected by the coast, is there a second reflection going back toward shore? Answers to these questions are provided by a video that can be viewed. Elapsed time in minutes is given in each frame of the movie. Colors represent the wave height (blue) and the orbital velocities (red). The coast was taken to be a Neumann reflecting boundary, that is, velocity stops perpendicular to the coast, where its magnitude is zero. The bottom profile including the shelf is shown as the heavy black curve at the top in the video. We note several points indicated by the video:

- The normalized height wave and orbital velocity wave come in from the right, toward the coast at the left. For this exact solution, the orbital velocity grows much faster than height as the wave advances onto the shallower shelf. This also follows from the Green's Law approximation, which indicates that height depends on depth \(d\) as \(d^{-1/4}\), while velocity varies as \(d^{-3/4}\).
- After coastal reflection, the height remains positive, while the direction of the velocity for the outgoing wave reverses. This is also true of the initial reflection at the bottom of the shelf-edge slope, although this is below the visibility level in the movie.
- The offshore retreating waves after coastal reflection encounter another strong reflection as they reach the top of the shelf edge 50 kilometers offshore. In fact, these backward-reflected waves explain the meteotsunami that was observed in June 2013 from the New Jersey coast. The original tsunami was launched by Proudman resonance from the eastward-moving low-pressure center. When an atmospheric anomaly like a low-pressure center travels across the sea at the same speed as the shallow-water phase velocity (which depends on the inverse square root of depth), a match or "resonance" is achieved. This causes the mound of water uplifted by the atmospheric low to break free and propagate as a tsunami soliton wave on its own. It was the returning reflected tsunami that impacted the coast and was reported by several radars and coastal tide gauges.

It is from the output files of this one-dimensional simulation that we deduced the transmission coefficient cited earlier, which was compared with predictions from the Green's Law approximation.

##### Two-Dimensional Tsunamis

We now examine two more realistic scenarios: in the first, a plane wave tsunami approaches the Portuguese West Coast, and in the second, a tsunami is generated by a point source in the Alboran Sea. Videos are provided, which show tsunami height and velocity normalized by their initial values, as the tsunami is refracted by the bathymetry and reflects from the coast. In these videos, the background color represents the tsunami wave height normalized by its initial value, with the magnitude indicated by the color bar. Velocity vectors overlain on top of the height background represent the orbital velocities normalized by the initial value, with the magnitudes indicated by the vector length. Elapsed time shown on each frame indicates the time taken for the tsunami to reach various points along the coast. These simulated values can be tested against radar observations of real tsunamis. In both cases, reflection of the wave from the coast is clearly visible, and the tsunami velocity increases more rapidly than the height as depth decreases, indicating that observed velocities provide a sensitive alert flag for a tsunami approaching the coast. Future work on this simulation will study results related to actual heights and velocities, rather than normalized values; output values can then be tested against radar observations of real tsunamis.

###### Portugal

The 1755 Lisbon earthquake in combination with subsequent fires and a tsunami almost totally destroyed Lisbon and adjoining areas. Tsunamis as tall as 20 meters swept the coast of North Africa, and struck Martinique and Barbados across the Atlantic. Using the actual offshore bathymetry, we simulated a tsunami approaching the Portuguese coastline from the west. Results are shown in a video. Bathymetry contours are shown to understand the tsunami refraction. The epicenter was located more than 200 kilometers to the west of the map. When the source is so distant, the initial condition for solving the PDE can be taken to be a plane wave, corresponding to a ridge of water traveling eastward. This approximation is reasonable whenever the source is distant from the near-field region and is convenient to model for numerical solutions. The domain for the numerical solution consists of the coastline of interest and the open box edges over the ocean. The coastline was assumed to have a Neumann (reflective) boundary condition.

This region was also selected for study because there are three 13.5-MHz SeaSonde HF radars operating at nearby locations, which are shown by green squares in the video. Tsunami observation software is being installed at these sites. The three radars would not see the tsunami if its propagation followed the line of sight from the source because the coast of southern Portugal would shadow those paths. In fact, the model output shows how the tsunami wave refracts and approaches the sites from the south. Reflection of the wave from the coast is clearly visible.

###### Alboran Sea

Another region of recent interest is the Alboran Sea, which is enclosed on three sides by Gibraltar on the west, Spain on the north, and Morocco on the south. There are seismically active regions near tiny Alboran Island that could raise a localized mound of water, which would spread out under the influence of gravity, initially radiating a near-circular tsunami wave.

A point source is another initial condition that is easy to handle in the PDE solution. Resulting maps are shown in a video. The tsunami point source is located near Alboran Island (the green square marker) in water that is 1000 meters deep. The tsunami radiates in all directions, intensifying in height and velocity as it moves into shallow water, as indicated by the bathymetry contours. As before, background color represents normalized height and vector length represents normalized velocity.

One can see as the movie progresses how different coastal regions are affected, as the approaching tsunami intensity increases. Offshore reflections and along-shore tsunami vectors are clearly seen from the vectors. The island is small compared to the tsunami wavelength, causing little observable effect.

#### Band-Velocity Simulation Based on Green's Law

This approximate procedure is based on the theory given earlier. To simulate velocities for a given test radar site over a period of time close to the arrival of a tsunami, band velocities from a real tsunami (termed reference velocities \(V_{Ref}\)) are superimposed on band velocities measured at the site (termed site velocities \(V_{Site}\)). Before adding to the site velocities, the reference velocities are adjusted for the site bathymetry using Green's Law. They are then multiplied by an arbitrary factor \(F\) that can be varied to adjust the height of the simulated tsunami approaching the test site. This process is encapsulated in the following equation for a given band:

\[
V_{Sim} = V_{Site} + F \cdot V_{Ref} \left( \frac{Depth_{Site}}{Depth_{Ref}} \right)^{3/4}
\]

where \(V_{Sim}\) is the simulated velocity that for \(F = 1\) would be observed if the reference tsunami approached the test site and \(Depth_{Site}\) and \(Depth_{Ref}\) are the average depths across the band for the test and reference sites. Increasing/decreasing the value of \(F\) will increase/decrease the size of the simulated velocity.

Simulated band velocities calculated using this approximate method are currently used to evaluate the suitability of radar sites for tsunami detection, as described in the next section.

### Evaluation of Radar Sites for Tsunami Detection Using Simulated Tsunami Velocities

As some sites are less suitable than others for tsunami monitoring with coastal radar systems, we are developing a site-dependent method that uses simulated tsunami velocities to estimate the size of a tsunami required to trigger a detection as a function of distance from the shore. This leads to an estimate of the warning time available. The tsunami simulation methods currently available have been discussed in the previous section. Tsunami simulation based on PDE model solutions of equations of motion is under development at this time with early results