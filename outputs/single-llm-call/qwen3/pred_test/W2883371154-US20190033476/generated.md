# DESCRIPTION

## FIELD

- define forecasting method

The present invention relates to a novel forecasting method for predicting the likelihood of subsequent seismic events following a primary earthquake, based on statistically significant spatial and temporal correlations observed across global seismic networks. This method leverages empirical data derived from historical earthquake catalogs to identify non-random patterns of increased seismic activity occurring at specific angular distances and time intervals after a high-magnitude seismic event. Unlike conventional approaches that assume seismic events are spatially and temporally independent beyond local aftershock zones, this method introduces a probabilistic framework that accounts for global-scale, delayed triggering effects extending up to three days following a primary event. The forecasting method operates by analyzing the distribution of secondary earthquakes relative to the epicenter of a primary earthquake, using arc-distance binning and baseline rate normalization to detect statistically significant deviations from Poisson-distributed background seismicity. It further incorporates magnitude-dependent weighting and antipodal enhancement factors to refine risk assessments, enabling the generation of actionable forecasts for regions previously considered immune to remote triggering. The method is applicable to real-time seismic monitoring systems and is designed to operate independently of mechanistic assumptions regarding wave propagation or stress transfer, relying instead on observed statistical anomalies in global seismic catalogs spanning over four decades. This approach transforms seismic forecasting from a purely localized, short-term warning paradigm into a globally informed, probabilistic risk assessment tool capable of identifying elevated hazard zones thousands of kilometers from the source event.

## BACKGROUND

- motivate earthquake prediction

Earthquake prediction remains one of the most critical challenges in geophysics due to the devastating human and economic consequences of large-scale seismic events. While early warning systems have improved response times for local populations by detecting P-waves and issuing alerts seconds before S-wave arrival, these systems provide no advance notice for regions distant from the epicenter, nor do they account for the possibility that a single large earthquake may increase the probability of subsequent events elsewhere on the planet. The societal need for extended-range forecasting is amplified by the increasing frequency of high-magnitude events in recent decades, many of which occur in densely populated coastal and tectonically active regions where infrastructure is vulnerable to cascading failures. The inability to anticipate secondary seismic risks beyond the immediate aftershock zone leaves communities, emergency responders, and critical infrastructure operators unprepared for events that may unfold hours or days later in geographically remote but seismically sensitive locations. Current risk models, grounded in the assumption of statistical independence between distant earthquakes, fail to capture the systemic nature of global seismic coupling, resulting in underestimation of hazard exposure and inefficient allocation of disaster preparedness resources. A method capable of identifying elevated risk zones at teleseismic distances would significantly enhance global resilience by enabling proactive mitigation measures, such as temporary shutdowns of industrial facilities, rerouting of transportation networks, and deployment of emergency personnel to regions previously deemed low-risk.

- limitations of early warning systems

Existing early warning systems are fundamentally limited by their reliance on near-field detection and rapid signal propagation analysis. These systems are effective only within a narrow radius—typically less than 100 kilometers—around the source event, and provide no predictive capability for regions beyond the immediate seismic wavefront. They are incapable of detecting or forecasting triggered events that occur in distant tectonic zones hours or days after the initial rupture, even when those events are statistically correlated with the primary shock. Furthermore, current models disregard the influence of global stress perturbations, treating all seismic activity as independent unless it occurs within the well-defined spatial and temporal bounds of classical aftershock sequences. This limitation results in a systemic blind spot in hazard assessment, where regions such as the antipode of a major subduction zone or areas near major transform faults may experience elevated seismic risk without any prior indication from conventional monitoring networks. Additionally, existing systems do not account for magnitude-dependent triggering thresholds or the cumulative effect of multiple primary events occurring in close succession, leading to an incomplete understanding of seismic cascades. As a result, emergency management protocols remain localized and reactive, failing to leverage the predictive power inherent in global seismic patterns that have been empirically documented but not yet operationalized in public safety infrastructure.

## SUMMARY

- introduce statistical association

The invention introduces a statistically validated association between large-magnitude primary earthquakes and an increased probability of subsequent seismic events occurring at specific angular distances and time intervals globally. This association is not explained by conventional aftershock dynamics or local stress transfer mechanisms, but instead reflects a systemic, long-range coupling effect that manifests as a non-random redistribution of seismic energy across the Earth’s lithosphere. The statistical association is quantified through the analysis of over 5,000 three-day observation windows derived from a global earthquake catalog spanning 1973 to 2016, revealing consistent patterns of elevated seismic activity at angular distances of approximately 45 degrees and 150 degrees from the primary event, with a pronounced suppression of activity near 90 degrees. These patterns persist across multiple magnitude thresholds and are statistically significant at p-values below 0.01 when aggregated across hundreds of independent events, rejecting the null hypothesis of spatial and temporal independence with high confidence.

- determine enhanced earthquake risk

The invention enables the determination of enhanced earthquake risk by comparing the observed frequency of secondary seismic events within defined spatial and temporal bins against a baseline rate derived from a control population of non-triggered time periods. Risk enhancement is calculated as a relative rate, normalized to account for variations in background seismicity and catalog completeness, and is further refined by incorporating magnitude-dependent scaling factors that reflect the increased likelihood of triggering larger events following higher-magnitude primaries. The method identifies zones of elevated risk not only in proximity to the primary event but also in antipodal regions, where the cumulative effect of seismic wave interactions may induce failure in tectonically stressed fault systems. Risk enhancement is not a binary condition but a continuous probability function, allowing for graded alerts based on the magnitude of the primary event, the elapsed time since its occurrence, and the proximity of the target region to the predicted high-risk angular zones.

- generate earthquake alerts

Upon detection of a qualifying primary earthquake, the system automatically generates an earthquake alert that communicates the probability of follow-on seismic events occurring within the next 72 hours in predefined geographic regions. These alerts are not mere notifications of potential activity but are calibrated probabilistic forecasts derived from the validated statistical model, expressed as risk enhancement factors relative to the regional baseline. The alert includes a spatial map indicating zones of elevated risk, with color-coded intensity levels corresponding to the magnitude of the primary event and the strength of the statistical association. The system further distinguishes between single-event triggering and cascading scenarios, where a secondary event may itself become a source for additional triggering, thereby extending the temporal horizon of risk.

- disseminate alerts

The earthquake alerts are disseminated through multiple communication channels, including government emergency management networks, international seismic monitoring agencies, mobile alert applications, satellite broadcast systems, and automated infrastructure control interfaces. Alerts are prioritized based on the predicted risk level and the population density of affected regions, ensuring that high-probability zones receive immediate notification while lower-risk areas receive consolidated updates. The dissemination protocol is designed to be interoperable with existing public warning systems and includes multilingual text, audio cues, and visual indicators to ensure accessibility across diverse user populations and technological environments.

- embodiment of method

An embodiment of the method involves the automated ingestion of real-time seismic data from a global network of seismometers, followed by the application of a declustering algorithm to remove known aftershocks and foreshocks, ensuring that only independent seismic events are included in the statistical analysis. The system then calculates the arc-distance distribution of all events ≥M5.0 occurring within three days of a primary event ≥M6.5, bins these events into ten-degree angular intervals, and compares the observed counts against a baseline derived from 5,355 non-triggered time windows. A binomial model is applied to compute p-values for each bin, and a polynomial fit is used to model the global average risk enhancement profile. When the p-value for any bin falls below a predetermined threshold, and the relative rate exceeds 1.5, an alert is triggered for the corresponding geographic region.

- system for generating alerts

The system for generating alerts comprises a central processing unit configured to receive seismic event data from distributed sensor networks, execute the statistical model described herein, and output alerts through a secure communication protocol. The system includes a database of historical seismic events, a real-time event classifier, a risk calculation engine, and a multi-channel alert distributor. It operates continuously, evaluating every qualifying primary event as it occurs and updating risk assessments in near real-time. The system is self-calibrating, adjusting baseline rates annually to account for changes in detection thresholds and catalog completeness, and includes failover mechanisms to ensure uninterrupted operation during network outages or data corruption events.

## DETAILED DESCRIPTION

### I. Terms

- define aftershock

An aftershock is defined as a subsequent seismic event that occurs in the vicinity of a larger primary earthquake and is temporally and spatially correlated with it, typically within a defined window determined by the magnitude of the primary event. Aftershocks are excluded from the statistical analysis of this invention to prevent contamination of baseline rates and to isolate the signal of remote triggering from local clustering effects. The spatial and temporal extent of an aftershock is calculated using empirical equations derived from the magnitude of the primary event, with larger events permitting longer durations and greater distances for aftershock inclusion.

- define amplitude measure

An amplitude measure refers to the maximum ground displacement or velocity recorded by a seismometer in response to seismic waves, typically expressed in millimeters per second or centimeters. In this invention, amplitude measures are not used as direct inputs to the forecasting model, as the method relies on event counts and spatial distributions rather than wave energy. However, amplitude data may be used in ancillary analyses to validate the physical plausibility of triggering mechanisms or to distinguish between tectonic and non-tectonic seismic signals.

- define angular coordinates

Angular coordinates refer to the great-circle arc distance measured in degrees between the epicenter of a primary earthquake and the epicenter of a secondary event, computed along the surface of the Earth. These coordinates are used to bin seismic events into 10-degree intervals for statistical analysis, enabling the identification of spatial patterns such as antipodal enhancement and equatorial suppression. Angular coordinates are independent of absolute latitude and longitude, allowing the method to be applied uniformly across the globe regardless of the primary event’s location.

- define antipodal

Antipodal refers to the point on the Earth’s surface that is diametrically opposite to the epicenter of a primary earthquake. In this invention, the antipodal region is identified as a zone of elevated seismic risk, typically extending within 30 degrees of the antipode, where secondary events occur at rates significantly higher than the global baseline. This phenomenon is observed consistently across multiple high-magnitude events and is a key feature of the statistical model.

- define binomial model

The binomial model is a statistical framework used to calculate the probability of observing a given number of seismic events within a spatial bin under the assumption that events occur randomly and independently. In this invention, the binomial model is applied to compare observed event counts against expected baseline counts, with the probability of success defined as the reciprocal of the number of control time windows. The model is adjusted using a mid-p-value correction to account for the discrete nature of earthquake counts.

- define body waves

Body waves are seismic waves that propagate through the interior of the Earth, including P-waves and S-waves. While body waves are detected by seismometers and used to locate earthquakes, they are not directly utilized in the forecasting method of this invention. The method operates independently of wave propagation mechanics, relying instead on statistical correlations in event distributions.

- define cluster

A cluster is a group of seismic events occurring in close spatial and temporal proximity, typically associated with aftershock sequences or localized stress release. Clusters are excluded from the analysis using a declustering algorithm to ensure that only independent events contribute to the statistical model. The method distinguishes between clusters generated by local aftershocks and those potentially induced by remote triggering.

- define earthquake

An earthquake is defined as a sudden release of energy in the Earth’s crust that generates seismic waves, measured by its moment magnitude (Mw) and recorded by seismic networks. For the purposes of this invention, earthquakes with magnitudes of M5.0 or greater are included in the analysis, with primary events required to be M6.5 or greater to trigger the forecasting algorithm.

- define epicenter

The epicenter is the point on the Earth’s surface directly above the hypocenter of an earthquake, determined from seismic wave arrival times recorded by multiple stations. In this invention, the epicenter is used as the reference point for calculating angular coordinates and spatial binning of secondary events.

- define fault

A fault is a fracture or zone of fractures in the Earth’s crust along which displacement has occurred. While the nature of the fault (e.g., thrust, strike-slip, normal) is not a direct input to the forecasting model, the method may be enhanced in future embodiments by incorporating fault type as a weighting factor based on observed triggering susceptibility.

- define fit

A fit refers to the mathematical curve, typically a polynomial, that best describes the observed pattern of relative seismic rates as a function of angular distance. In this invention, a fourth-order polynomial is fitted to the average relative rates across multiple events to model the global risk enhancement profile, including maxima near 45 and 150 degrees and a minimum near 90 degrees.

- define forecast

A forecast is a probabilistic prediction of increased seismic activity in a specific geographic region over a defined time window, generated by the method of this invention. Unlike deterministic predictions, a forecast expresses the likelihood of an event occurring relative to the background rate, and is accompanied by a confidence metric derived from statistical significance.

- define foreshock

A foreshock is a smaller seismic event that precedes a larger primary earthquake and occurs within the same spatial and temporal cluster. Foreshocks are excluded from the analysis to prevent bias in the baseline rate calculation and to ensure that only events occurring after the primary earthquake are considered as potential triggered events.

- define free oscillations

Free oscillations are global vibrations of the Earth that persist after a large earthquake, lasting for hours or days. While these oscillations may be detected by sensitive instruments, they are not used as inputs to the forecasting method, which relies solely on discrete event counts and their spatial distribution.

- define hypocenter

The hypocenter is the point within the Earth where an earthquake rupture initiates, located at a specific depth below the surface. The hypocenter is used to determine the epicenter, but depth is not a parameter in the statistical model of this invention.

- define p-value

A p-value is a statistical measure indicating the probability that an observed result could have occurred under the null hypothesis of random, independent event distribution. In this invention, p-values below 0.05 are considered statistically significant, and aggregated p-values across multiple bins are used to confirm the presence of a global triggering signal.

- define Poisson model

The Poisson model is a statistical distribution used to describe the probability of a given number of events occurring in a fixed interval of time or space, assuming events occur independently and at a constant average rate. In this invention, the Poisson model serves as the baseline assumption against which observed seismic patterns are tested, and its validity is confirmed through declustering of the earthquake catalog.

- define seismic

Seismic refers to phenomena related to the generation and propagation of elastic waves through the Earth’s interior, including earthquakes, explosions, and other sources of ground vibration. In this context, the term is used to describe events recorded by seismometers and included in the global catalog for analysis.

- define surface wave and time window

A surface wave is a type of seismic wave that travels along the Earth’s surface, often responsible for the most destructive shaking during an earthquake. A time window is a defined period, in this invention typically three days, during which secondary seismic events are monitored following a primary earthquake. The time window is fixed across all analyses to ensure consistency in statistical comparisons.

### II. Method for Determining Enhancement of Earthquake Risk

- define primary earthquake event

A primary earthquake event is defined as a seismic event with a moment magnitude of M6.5 or greater, serving as the initiating event in the forecasting method. The epicenter and precise timing of the primary event are used as the reference point for all subsequent spatial and temporal analyses. Only events meeting this magnitude threshold are considered capable of triggering the global risk enhancement signal.

- introduce spatial region and time window

The spatial region is defined as the entire surface of the Earth, partitioned into 180 angular bins of 1-degree width, which are aggregated into 10-degree intervals for statistical analysis. The time window is fixed at 72 hours following the occurrence of the primary earthquake, during which all secondary events ≥M5.0 are recorded and analyzed. This window was selected based on empirical observation of peak triggering activity and statistical power.

- motivate question of enhanced earthquake risk

The motivation for this method arises from the observation that traditional seismic risk models fail to account for the possibility that a single large earthquake may influence seismic activity thousands of kilometers away. The question of whether such remote triggering exists at a statistically significant level has remained unresolved due to the lack of a robust analytical framework capable of distinguishing true triggering from random clustering.

- describe first exemplary method

The first exemplary method involves a prospective analysis in which a set of primary earthquakes is identified, and all subsequent events within the 72-hour window are cataloged. The angular distance from each primary event to each secondary event is calculated, and events are binned into 10-degree intervals. The observed count in each bin is compared to the baseline rate derived from 5,355 non-triggered time windows of equal duration.

- introduce historical record of earthquake events

The historical record consists of all earthquakes ≥M5.0 recorded globally between January 1, 1973, and December 31, 2016, sourced from the United States Geological Survey (USGS) comprehensive earthquake catalog. This dataset provides sufficient spatial and temporal coverage to establish a reliable baseline of seismic activity.

- define test events and corpus events

Test events are the primary earthquakes ≥M6.5 used to initiate the forecasting analysis. Corpus events are all secondary earthquakes ≥M5.0 that occur within the 72-hour window following a test event. The corpus events are analyzed for spatial distribution relative to the test events.

- describe properties of test events

Test events are characterized by their moment magnitude, date and time of occurrence, epicentral coordinates, and focal mechanism. Only events with well-constrained hypocentral parameters and magnitudes ≥M6.5 are included. Events with incomplete data or those occurring in regions with poor seismic coverage are excluded.

- describe properties of corpus events

Corpus events are characterized by their magnitude, location, and time of occurrence. All events ≥M5.0 are included regardless of their tectonic setting, provided they are not classified as aftershocks or foreshocks. Their spatial distribution relative to the test event epicenter is the primary variable of interest.

- assign events to space-time bins

Each corpus event is assigned to a spatial bin based on its angular distance from the test event epicenter and to a temporal bin based on its occurrence time within the 72-hour window. Spatial bins are 10 degrees wide, centered at 30, 35, 40, ..., 175 degrees. Temporal bins are not used in the primary analysis but may be employed in extended embodiments.

- disregard azimuthal coordinate

The azimuthal coordinate, or compass direction, is disregarded in the analysis to ensure that the method is isotropic and applicable globally. Only the arc distance from the primary event is used, eliminating directional bias and enabling uniform application across all latitudes and longitudes.

- bin events relative to every test event

For each test event, all corpus events are binned according to their angular distance relative to that event’s epicenter. This process is repeated for every qualifying primary event, resulting in a cumulative distribution of observed event counts across all test events.

- determine observed rates and baseline rates

Observed rates are calculated as the total number of corpus events in each spatial bin divided by the number of test events. Baseline rates are calculated as the average number of events in each bin across all 5,355 non-triggered time windows, normalized to match the number of test events.

- calculate baseline event rates

Baseline event rates are computed by summing the counts of all corpus events in each bin across the control group, dividing by the number of control windows (5,355), and adjusting for the number of test events to ensure comparability.

- normalize baseline rates

Baseline rates are normalized so that the total expected count across all bins equals the total observed count, ensuring that the comparison between observed and expected rates is statistically valid and not skewed by variations in global seismicity.

- determine observed event rates

Observed event rates are determined by summing the number of corpus events in each spatial bin across all test events and dividing by the number of test events. These rates reflect the empirical distribution of triggering effects.

- normalize observed event rates

Observed event rates are normalized using the same scaling factor applied to the baseline rates to maintain consistency in the ratio of observed to expected counts.

- evaluate p-values

P-values are calculated for each spatial bin using a binomial distribution, where the number of trials is the number of test events plus control windows, and the probability of success is the reciprocal of the total number of windows. The mid-p-value correction is applied to reduce bias from discrete counting.

- describe statistical model of earthquake events

The statistical model assumes that, absent triggering, seismic events follow a Poisson distribution in space and time. The method tests whether the observed distribution of secondary events deviates significantly from this assumption, indicating the presence of a global triggering mechanism.

- assume Poisson statistics

Poisson statistics are assumed as the null hypothesis, meaning that the occurrence of earthquakes is random and independent across space and time. The validity of this assumption is confirmed through declustering, which removes aftershocks and restores the Poisson nature of the remaining catalog.

- approximate Poisson model with binomial model

The Poisson model is approximated using a binomial model for computational tractability, where each test event represents a trial and the occurrence of a corpus event in a given bin represents a success. The binomial model allows for precise calculation of cumulative probabilities under the null hypothesis.

- calculate p-value for each spatial zone

For each 10-degree spatial zone, the p-value is calculated as the probability of observing the number of corpus events or more under the null hypothesis. This is done using the binomial cumulative distribution function with mid-p-value correction.

- interpret p-value results

A p-value less than 0.05 indicates that the observed event count in that bin is statistically unlikely to have occurred by chance, supporting the alternative hypothesis of triggering. Aggregated p-values across multiple bins are used to confirm the robustness of the signal.

- accept or reject null hypothesis

If the aggregated p-value across multiple bins is less than 0.01, the null hypothesis of random, independent seismicity is rejected, and the presence of a global triggering effect is accepted.

- illustrate disclosed method

The disclosed method is illustrated through a series of spatial maps showing the distribution of corpus events relative to test events, with overlays indicating bins with p-values below 0.05 and relative rates exceeding 1.5. These visualizations demonstrate the consistent emergence of high-risk zones at 45 and 150 degrees.

- show primary test event on map

A primary test event is shown on a global map as a red circle at its epicenter, with surrounding concentric circles marking angular distances of 30, 45, 90, 150, and 180 degrees. Corpus events are plotted as green dots, with their density clearly elevated in the 45- and 150-degree zones.

- show corpus events on map

Corpus events are displayed as individual points on the map, color-coded by magnitude and clustered in the antipodal and 45-degree zones, with minimal density near the 90-degree equatorial region.

- illustrate time bins for baseline rate estimation

Time bins for baseline rate estimation are shown as non-overlapping 72-hour windows extracted from the historical catalog, spaced at least 10 days apart to avoid contamination. Each window is centered on a random date and location, with no correlation to any primary event.

- exclude time bins correlated with primary event

Time bins that overlap with any known primary event or its aftershock sequence are excluded from the control group to ensure that the baseline remains uncontaminated by triggering effects.

- count events in remaining time bins

Events in the remaining control time bins are counted and aggregated by spatial bin to form the baseline distribution, which is then used to compute expected rates.

- fit Poisson distribution to baseline counts

The baseline counts for each spatial bin are fitted to a Poisson distribution to confirm that the declustered catalog follows the expected statistical behavior, validating the use of the Poisson model as the null hypothesis.

- calculate baseline rates for each spatial bin

Baseline rates are calculated as the mean number of events per bin across all control windows, scaled to match the number of test events for direct comparison.

- describe forward-looking experiment 1

Forward-looking experiment 1 involves identifying a set of primary events and then observing the distribution of subsequent events within the 72-hour window. This experiment tests whether large earthquakes increase the probability of future events at specific distances.

- extract test events with various magnitudes

Test events are extracted in narrow magnitude bands (e.g., M6.0–6.1, M6.5–6.6, etc.) to assess whether triggering sensitivity varies with magnitude.

- prepare corpus of events with magnitude ≥5.0M

The corpus is prepared by extracting all events ≥M5.0 occurring within 72 hours of each test event, regardless of their tectonic setting or depth.

- choose time window of zero to three days

The time window is fixed at 0 to 72 hours to capture the peak of triggering activity, as determined by empirical analysis of lag-time distributions.

- search corpus for observed events within time window

The corpus is searched for all events occurring within the 72-hour window following each test event, and their locations are recorded.

- compute angular distance from test event to observed events

The angular distance between the test event epicenter and each observed event is computed using the haversine formula on the Earth’s spherical surface.

- spatially bin observed events

Observed events are assigned to 10-degree angular bins centered at 30, 35, 40, ..., 175 degrees, with no binning in azimuth.

- determine baseline counts and rates

Baseline counts are determined by summing events in each bin across all control windows, and baseline rates are computed as the average per window.

- calculate p-values using binomial distribution

P-values are calculated for each bin using the binomial distribution, with the probability of success set to 1/5,356, representing the ratio of test to total windows.

- describe filtering of foreshocks and aftershocks

Foreshocks and aftershocks are filtered using a declustering algorithm based on magnitude-dependent spatial and temporal windows, as defined by Gardner and Knopoff. Events falling within these windows are removed from the corpus.

- remove aftershocks from analysis

Aftershocks are removed to prevent inflation of observed counts and to isolate the signal of remote triggering from local clustering.

- remove foreshocks from analysis

Foreshocks are removed to ensure that only events occurring after the primary event are considered, preserving the causal directionality of the analysis.

- filter clustered events

Clustered events are identified using a spatial-temporal density algorithm and removed if they are determined to be part of a local sequence rather than a globally triggered signal.

- check for physical proximity of observed events

Physical proximity is checked to ensure that events in the same bin are not duplicates or mislocated. Only events with well-constrained epicenters are retained.

- introduce backward-looking experiment 2

Backward-looking experiment 2 reverses the analysis: corpus events are selected as targets, and the distribution of primary events occurring within the preceding 72 hours is analyzed.

- design and implement experiment 2

Experiment 2 is implemented by selecting all events ≥M5.0 and searching backward in time for any primary event ≥M6.5 that occurred within 72 hours. The angular distance from each primary to its potential target is computed and binned.

- extract test events

Test events are extracted as the potential triggering events ≥M6.5, while corpus events are the targets ≥M5.0.

- prepare corpus of events

The corpus consists of all events ≥M5.0 from the catalog, with no restriction on their origin or tectonic context.

- identify time window

The time window is again fixed at 72 hours prior to each corpus event.

- perform statistical analysis

Statistical analysis is performed identically to experiment 1, with p-values calculated for each angular bin.

- determine p-values

P-values are calculated using the same binomial model, confirming that the same spatial patterns emerge regardless of whether the analysis is prospective or retrospective.

- present results of experiment 2

Results show elevated p-values in the 45- and 150-degree zones, consistent with experiment 1, reinforcing the robustness of the findings.

- introduce exemplary results

Exemplary results include a 2.1-fold increase in seismicity at 150 degrees following M8.0 events, with a p-value of 0.0003, and a 1.8-fold increase at 45 degrees for M7.5 events.

- describe FIGS. 3A-3D

FIGS. 3A–3D illustrate the relative rates, control counts, p-values, and aggregated significance for experiment 2, demonstrating the consistency of the antipodal and 45-degree enhancement across multiple magnitude classes.

- present relative rates and p-values

Relative rates are presented as ratios of observed to expected counts, with p-values shown as negative logarithms to emphasize significance. Zones with p-values <0.01 are highlighted.

- discuss results of experiment 2

The results confirm that the triggering effect is not an artifact of the forward-looking design but is a real, bidirectional statistical association.

- introduce second exemplary method

The second exemplary method involves a real-time implementation that continuously monitors seismic networks and generates alerts as soon as a qualifying primary event is detected.

- describe flowchart 400

Flowchart 400 depicts the sequence of operations: data ingestion, declustering, binning, rate calculation, p-value determination, risk threshold comparison, and alert generation.

- set up analysis

The analysis is set up by initializing the database of historical events and the declustering parameters.

- collect sets of events

Real-time event data are collected from global seismic networks and stored in a buffer.

- remove foreshocks and aftershocks

Events are filtered using the declustering algorithm before inclusion in the analysis.

- set up spatial bins

The 18 angular bins are initialized with counters for observed and baseline counts.

- mark time steps

Time steps are marked at 1-hour intervals to allow for dynamic updating of the risk profile as the 72-hour window progresses.

- classify corpus events

Events are classified as either test or corpus based on magnitude and temporal relationship.

- accumulate events

Events are accumulated in their respective spatial bins as they occur.

- calculate p-values

P-values are recalculated at each time step using the updated observed and baseline counts.

- analyze p-value results

Results are analyzed to determine whether any bin has crossed the significance threshold.

- determine risk enhancement factors

Risk enhancement factors are calculated as the ratio of observed to baseline rates for each bin.

- fit risk enhancement factors

A fourth-order polynomial is fitted to the risk enhancement factors across all bins to model the global risk profile.

- introduce exemplary results

Exemplary results show that for an M8.0 event, the risk enhancement factor reaches 2.3 at 150 degrees within 24 hours and remains elevated for 72 hours.

- describe FIG. 5

FIG. 5 shows the fitted risk enhancement curve, with the 45- and 150-degree maxima clearly visible, and the 90-degree suppression zone as a trough.

- present relative rates and fitted risk enhancement

Relative rates are plotted as discrete points, and the fitted polynomial is overlaid, demonstrating the predictive power of the model.

- discuss results of experiment 2

The results confirm that the risk enhancement is not random but follows a predictable, repeatable spatial pattern.

- introduce cascaded earthquake events

Cascaded earthquake events occur when a secondary event triggered by a primary event itself becomes a source for additional triggering, creating a chain reaction.

- describe FIGS. 6A-6C

FIGS. 6A–6C illustrate examples of cascading events, where a primary earthquake triggers a secondary event, which in turn triggers a third event in its own antipodal zone.

- illustrate spatial and temporal relationships

The figures show the epicenters of each event, the angular distances between them, and the time intervals, demonstrating that cascades can span multiple continents.

- show proximate and antipodal risk enhancement zones

The figures highlight that both proximate and antipodal zones can be activated in sequence, with the second event’s antipodal zone becoming a new high-risk area.

- discuss cascaded earthquake events

Cascades demonstrate that the triggering effect is not limited to single events but can propagate through the global fault network, extending the temporal and spatial horizon of risk.

- conclude on risk enhancement

The method conclusively demonstrates that enhanced earthquake risk is not localized but globally distributed, with predictable spatial patterns that can be quantified and forecasted.

- conclude on cascaded earthquake events

Cascaded events represent a previously unrecognized mode of seismic hazard propagation, and the method provides the first framework for detecting and warning against such multi-stage triggering.

- conclude on method for determining enhanced earthquake risk

The method provides a statistically rigorous, globally applicable, and operationally feasible means of determining enhanced earthquake risk beyond the traditional aftershock domain, transforming seismic forecasting from a reactive to a predictive science.

### III. Method for Generating and Disseminating an Earthquake Alert

- receive indication of primary earthquake

The system receives an indication of a primary earthquake through real-time data feeds from global seismic networks, including the USGS, EMSC, and IRIS, with latency less than 10 seconds.

- receive indication from seismometer or seismometer network

The indication is received via automated data streams from seismometers, which transmit waveform data and preliminary magnitude estimates upon detection of a seismic event.

- receive indication from news service or messaging service

In the event of a data outage, the system may receive indications from trusted news services or emergency messaging platforms that report large earthquakes with sufficient accuracy.

- determine probability measure for follow-on earthquake

The probability measure is determined by computing the relative rate for each angular bin and comparing it to the fitted risk enhancement curve, then applying a magnitude scaling factor based on the primary event’s size.

- determine probability measure for second spatial area and second time window

The system determines the probability measure for a second spatial area by identifying the 45- and 150-degree zones and for a second time window by extending the analysis to 72 hours, with dynamic updates every hour.

- characterize second spatial area

The second spatial area is characterized as a 30-degree radius centered at 45 and 150 degrees from the primary epicenter, with higher risk in the antipodal zone.

- define second time window

The second time window is defined as the 72-hour period following the primary event, divided into 24-hour segments for incremental alert updates.

- determine probability measure using formula

The probability measure is determined using the formula: P = R × M × T, where R is the relative rate, M is the magnitude scaling factor, and T is the time decay factor.

- determine probability measure dependent on baseline rate

The probability measure is inversely proportional to the baseline rate, ensuring that regions with naturally high seismicity are not falsely flagged as elevated risk.

- determine probability measure dependent on spatial distance

The probability measure is highest at 45 and 150 degrees, with a sharp decline near 90 degrees, as defined by the fitted polynomial.

- determine probability measure dependent on time difference

The probability measure peaks at 24–48 hours after the primary event and decays exponentially beyond 72 hours.

- determine probability measure dependent on magnitude of primary earthquake

The magnitude scaling factor increases logarithmically with the primary event’s magnitude, with M8.0+ events producing a 2.5-fold increase in risk relative to M6.5 events.

- determine probability measure dependent on type of primary earthquake

The type of primary earthquake, such as subduction, transform, or intraplate, is used as a secondary weighting factor based on historical triggering efficiency.

- determine probability measure dependent on type of fault

The type of fault is inferred from focal mechanism data and used to adjust the risk enhancement factor, with thrust faults showing higher triggering potential.

- determine probability measure dependent on number of foreshocks or aftershocks

The number of foreshocks or aftershocks is used to validate the declustering process; if excessive, the event is flagged for manual review.

- generate earthquake alert

An earthquake alert is generated when the probability measure exceeds a threshold of 0.05 in any spatial bin, with severity levels assigned based on the magnitude of the enhancement.

- include audible alert in earthquake alert

The alert includes an audible tone, such as a high-pitched siren or voice announcement, to attract attention in noisy environments.

- include visual alert in earthquake alert

The alert includes a visual display on mobile devices and public screens, showing a global map with color-coded risk zones and a countdown timer.

- include message in earthquake alert

The alert includes a textual message such as “Elevated seismic risk detected in antipodal region. Prepare for possible M5+ event within 72 hours.”

- transmit earthquake alert to destination

The alert is transmitted via SMS, email, satellite broadcast, emergency alert systems, and API integrations with government and infrastructure operators.

### IV. System for Providing Earthquake Alerts

- describe system for generating and disseminating earthquake alerts

The system comprises a central server cluster connected to global seismic data feeds, a real-time processing engine, a statistical risk model, and a multi-channel alert distributor. It operates continuously, evaluating every seismic event ≥M6.5 and generating alerts within 30 seconds of detection.

- describe components of system

Components include a data ingestion module, a declustering engine, a spatial binning module, a p-value calculator, a risk enhancement fitter, an alert generator, and a communication interface. All components are redundant and geographically distributed for fault tolerance.

### V. A Generalized Computer Environment

- describe computing environment

The computing environment consists of a distributed network of servers located in multiple data centers, each equipped with redundant power, cooling, and network connectivity to ensure 99.99% uptime.

- describe processing unit

The processing unit is a multi-core CPU cluster capable of executing the binomial and polynomial calculations in parallel across hundreds of test events simultaneously.

- describe memory

Memory includes high-speed RAM for real-time data buffering and persistent storage for historical catalogs and model parameters.

- describe storage

Storage is provided by solid-state drives with encrypted, replicated backups across three geographic regions to prevent data loss.

- describe input devices

Input devices include network interfaces for seismic data feeds, API endpoints for emergency services, and manual override terminals for system administrators.

- describe output devices

Output devices include public alert displays, mobile notification servers, satellite transmitters, and web portals for emergency management agencies.

- describe communication ports

Communication ports include TCP/IP, MQTT, and satellite uplink interfaces, with failover protocols to maintain connectivity during network outages.

- describe computing cloud

The computing cloud is a hybrid architecture combining on-premises servers with secure cloud instances for scalability and disaster recovery.

- describe software instructions

Software instructions are written in Python and C++, compiled into executable modules that run on Linux-based operating systems.

- describe program modules

Program modules include event classifier, declusterer, binning engine, statistical analyzer, alert generator, and communicator, each modular and independently testable.

- describe computer-executable instructions

Computer-executable instructions are stored in non-volatile memory and loaded into RAM for execution, with version control and digital signatures to prevent tampering.

### VI. General Considerations

- define comprising

The term “comprising” is used in its open-ended sense, meaning that the described method or system may include additional elements not explicitly listed.

- define or

The term “or” is used in its inclusive sense, meaning that any combination of the listed elements may be employed.

- define about

The term “about” refers to a range of ±10% unless otherwise specified, allowing for minor variations in measurement or calculation.

- describe numerical parameters

Numerical parameters such as magnitude thresholds, time windows, and angular bins are empirically derived and may be adjusted based on updated catalogs or improved statistical models.

- describe ranges

Ranges for magnitude, time, and distance are defined with precision to ensure reproducibility, with the primary range being M6.5–M8.0 for primary events and 0–72 hours for the time window.

- describe limitations

Limitations include dependence on catalog completeness, sensitivity to declustering parameters, and the exclusion of events in regions with sparse seismic coverage.

- describe novel and non-obvious features

The novel and non-obvious features include the use of antipodal enhancement as a predictive signal, the aggregation of p-values across multiple bins to confirm global triggering, and the application of a polynomial risk profile to generate real-time alerts.

- describe computer operations

Computer operations include data ingestion, filtering, binning, statistical modeling, alert generation, and transmission, all performed automatically without human intervention.

- describe theories of operation

Theories of operation are not required for the method’s implementation, as the invention is based on empirical statistical correlation rather than mechanistic theory.

- describe apparatus and methods

The apparatus and methods are inseparable in their operation; the method cannot be implemented without the apparatus, and the apparatus is designed solely to execute the method.

- describe computer-readable storage media

Computer-readable storage media include hard drives, solid-state drives, optical discs, and cloud storage, all capable of storing the software instructions and historical data.

- describe computer-executable instructions

Computer-executable instructions are encoded in binary form and loaded into memory for execution by the processing unit, with checksums to ensure integrity.

- describe software application

The software application is a standalone program that runs on dedicated servers and does not require internet connectivity once initialized.

- describe network environment

The network environment includes secure, encrypted connections between seismic sensors, data centers, and alert recipients, with firewalls and intrusion detection systems.

- describe suitable programming language

Suitable programming languages include Python for data analysis, C++ for performance-critical modules, and JavaScript for web-based alert interfaces.

- describe communication means

Communication means include terrestrial fiber, satellite links, cellular networks, and radio broadcast, with redundancy to ensure delivery under all conditions.