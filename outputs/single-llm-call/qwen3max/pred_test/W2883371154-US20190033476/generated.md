# DESCRIPTION

## FIELD

The present invention relates to a forecasting method for determining enhanced earthquake risk following a primary earthquake event. More specifically, the invention provides a statistically rigorous, data-driven approach to assess whether a significant earthquake—typically of magnitude 5.0 or greater—increases the likelihood of subsequent earthquakes at teleseismic distances within a defined time window, such as up to three days after the initiating event. The method leverages historical seismic catalogs, spatial-temporal binning, and probabilistic modeling to evaluate deviations from baseline seismicity rates under a null hypothesis of independence. This forecasting method enables the generation of actionable alerts regarding elevated seismic risk in specific geographic regions following large earthquakes, thereby supporting improved short-term hazard assessment and emergency preparedness.

## BACKGROUND

Earthquake prediction remains one of the most challenging problems in geophysics. While early warning systems can provide seconds to minutes of advance notice based on the detection of initial seismic waves, they are inherently reactive and limited to regions near the rupture zone. These systems do not address the broader question of whether a major earthquake can influence seismic activity globally over longer time scales—hours to days—beyond its immediate aftershock sequence. Historically, the prevailing scientific view has held that earthquakes are spatially and temporally independent beyond local fault interactions, governed by Poisson statistics wherein each event occurs randomly with a constant average rate. This assumption underpins most seismic hazard models used for long-term risk assessment.

However, anecdotal and observational evidence has increasingly suggested possible remote triggering of seismicity by large earthquakes, even at antipodal distances. For instance, dynamic stresses from surface waves of great earthquakes have been linked to increased microseismicity in distant volcanic or tectonically active regions. Despite these reports, systematic statistical evidence for the triggering of moderate-to-large earthquakes (≥M5.0) at global distances within days of a mainshock has remained elusive. Prior studies examining windows of 16 to 100 hours post-mainshock found no significant increase in M>5 events beyond 1,000 km, reinforcing skepticism about global interconnectivity. A key limitation of existing approaches is their reliance on narrow temporal bins (e.g., 24 hours) and aggregated magnitude ranges, which may obscure subtle but consistent patterns. Furthermore, inadequate declustering of aftershocks can inflate baseline rates or contaminate test observations, leading to false negatives. Thus, there exists a critical need for a robust, reproducible forecasting method capable of detecting statistically significant enhancements in earthquake risk at teleseismic distances following major seismic events.

## SUMMARY

The present invention introduces a novel statistical association between a primary earthquake event and an enhanced probability of subsequent earthquakes occurring within a defined spatial region and time window—specifically, up to three days and at arc distances ranging from approximately 30° to 175° from the epicenter of the primary event. This association is quantified by comparing observed seismicity rates following the primary event against long-term baseline rates derived from historical earthquake catalogs, while rigorously excluding aftershocks and foreshocks to isolate non-local effects.

In accordance with the invention, enhanced earthquake risk is determined by evaluating whether the number of observed earthquakes in specific spatial bins exceeds what would be expected under a null hypothesis of random, independent occurrence. Statistical significance is assessed using p-values calculated via a binomial approximation to a Poisson process, enabling rejection of the null hypothesis when p < 0.05 or other predetermined thresholds. Upon confirmation of statistically significant risk enhancement, the system generates an earthquake alert that characterizes the elevated risk in terms of geographic location, time window, and magnitude dependence.

These alerts are disseminated through audible, visual, or textual means to relevant stakeholders, including emergency management agencies, infrastructure operators, and the public. In one embodiment, the method is implemented as a computer-executable algorithm that processes real-time or near-real-time seismic data, applies declustering filters, performs spatial-temporal binning, computes relative rates and p-values, and triggers alerts when risk thresholds are exceeded. The invention further encompasses a system comprising seismometers, communication networks, processing units, and output devices configured to execute this method continuously and autonomously.

## DETAILED DESCRIPTION

### I. Terms

As used herein, an *aftershock* refers to a secondary earthquake that occurs in close spatial and temporal proximity to a larger mainshock, typically within a distance defined by empirical scaling laws related to the mainshock’s magnitude and within a time window extending days to years. An *amplitude measure* denotes any quantitative representation of seismic wave strength, such as peak ground acceleration or velocity. *Angular coordinates* describe positions on the Earth’s surface in terms of latitude and longitude, or equivalently, arc distance and azimuth from a reference point. *Antipodal* refers to locations on the Earth’s surface diametrically opposite one another, separated by approximately 180 degrees of arc.

A *binomial model* is a discrete probability distribution that describes the number of successes in a fixed number of independent trials with the same probability of success. *Body waves* are seismic waves that travel through the Earth’s interior, including P-waves and S-waves. A *cluster* denotes a group of earthquakes occurring in close spatial and temporal proximity, often indicative of aftershock sequences or swarm activity. An *earthquake* is a sudden release of energy in the Earth’s crust that creates seismic waves, typically measured by magnitude scales such as moment magnitude (Mw).

The *epicenter* is the point on the Earth’s surface directly above the hypocenter, which is the actual location where rupture initiates underground. A *fault* is a fracture or zone of fractures between two blocks of rock, along which displacement has occurred. To *fit* means to apply a mathematical function or model to observed data to estimate parameters or trends. A *forecast* is a probabilistic or deterministic prediction of future seismic activity based on statistical or physical models.

A *foreshock* is an earthquake that precedes a larger mainshock in the same location and is identified retrospectively. *Free oscillations* are the natural vibrational modes of the Earth excited by large earthquakes. *P-value* is the probability, under a null hypothesis, of obtaining test results at least as extreme as the observed results. A *Poisson model* assumes events occur independently at a constant average rate, commonly used to model background seismicity.

*Seismic* pertains to phenomena related to earthquakes or earth vibrations. A *surface wave* is a seismic wave that propagates along the Earth’s surface, typically with larger amplitude and longer duration than body waves. A *time window* is a defined interval during which events are observed or analyzed, such as zero to three days following a primary earthquake.

### II. Method for Determining Enhancement of Earthquake Risk

The method begins by defining a *primary earthquake event*, typically of magnitude ≥M6.0, which serves as the potential trigger. A *spatial region* is established around the globe, excluding the immediate aftershock zone (e.g., within 25° arc distance for M≥8.0 events), and a *time window* of zero to three days post-event is selected. The central question is whether this primary event enhances the risk of subsequent earthquakes in distant regions during this window.

A first exemplary method involves compiling a *historical record of earthquake events* from authoritative catalogs such as the USGS database spanning 1973–2016, including all events ≥M5.0. From this corpus, *test events*—potential triggers—are extracted based on narrowly defined magnitude ranges (e.g., M8.0 ≤ M < 8.1). *Corpus events*—potential triggered events—are all earthquakes ≥M5.0 occurring within the time window following each test event.

Properties of test events include precise origin time, epicenter, and magnitude; corpus events share these attributes but are evaluated for their spatial relationship to each test event. Events are assigned to *space-time bins*: ten-degree arc-distance bins centered every five degrees from 30° to 175°, with the *azimuthal coordinate disregarded* to focus on radial dependence. Binning is performed *relative to every test event*, treating each as a local origin.

*Observed rates* are computed by counting corpus events in each spatial bin across all test events. *Baseline rates* are derived from a control group: for each test event, 5,355 non-overlapping three-day periods from the historical record are sampled, and earthquake counts in corresponding bins are aggregated. These baseline counts are normalized by the number of control periods to yield expected rates per three-day window.

Observed and baseline rates are normalized to account for varying sample sizes. *P-values* are then calculated using a *binomial model* that approximates the underlying *Poisson statistics* of global seismicity. Under the null hypothesis of independence, the probability of observing *x* or more events in a bin is given by the complement of the cumulative binomial distribution, with success probability *p* = 1/(N_control + 1). Mid-p-value correction is applied to reduce bias.

If the p-value for a spatial bin falls below a significance threshold (e.g., 0.05), the null hypothesis is rejected, indicating *enhanced earthquake risk*. The method is illustrated by mapping a primary test event and surrounding corpus events, showing elevated counts in antipodal zones (e.g., 140°–175°). Time bins used for baseline estimation exclude periods correlated with known large earthquakes to avoid contamination.

In *forward-looking Experiment 1*, test events of varying magnitudes (M6.0 to ≥M8.0) are analyzed. Corpus events ≥M5.0 are searched within 0–3 days, binned by angular distance, and compared to baselines. Foreshocks and aftershocks are removed using the Gardner-Knopoff declustering method, which defines space-time windows based on magnitude-dependent formulas. Clustered events are filtered to ensure independence.

A *backward-looking Experiment 2* reverses the logic: test events are potential triggered earthquakes (≥M5.0), and corpus events are potential sources (≥M6.5) in the preceding three days. Similar binning and statistical analysis yield p-values confirming risk enhancement, particularly in antipodal regions. Results show collective p-values as low as 0.0004, strongly rejecting randomness.

Exemplary results (FIGS. 3A–3D) display relative rates exceeding 2.0 in antipodal bins with p-values <0.01. A second method (flowchart 400) automates the process: collect events, remove clusters, set spatial bins, mark time steps, classify events, accumulate counts, compute p-values, and determine *risk enhancement factors* by fitting observed-to-baseline ratios. FIG. 5 shows higher enhancement for larger source magnitudes.

*Cascaded earthquake events* occur when a triggered earthquake itself becomes a new source, creating chains (FIGS. 6A–6C). Overlapping antipodal zones show concentrated activity. The method conclusively demonstrates that large earthquakes enhance global seismic risk for up to three days, with spatial patterns inconsistent with random models.

### III. Method for Generating and Disseminating an Earthquake Alert

Upon detection of a primary earthquake—via a *seismometer* or *network*—or notification from a *news or messaging service*—the system determines a *probability measure* for a follow-on earthquake in a *second spatial area* (e.g., antipodal cap) and *second time window* (e.g., 0–72 hours). This area is characterized by arc distance (e.g., 140°–175°) and the window by duration post-mainshock.

The probability measure depends on: (1) the *baseline rate* of seismicity in the target zone; (2) *spatial distance* from the primary event; (3) *time difference* since the primary event; (4) *magnitude of the primary earthquake*; (5) *type of primary earthquake* (e.g., thrust vs. strike-slip); (6) *fault type* in the target zone; and (7) *number of foreshocks or aftershocks*, which may indicate stress state.

An *earthquake alert* is generated when the probability exceeds a threshold. The alert includes an *audible signal* (e.g., siren), *visual indicator* (e.g., map overlay), and *message* specifying location, time window, and risk level. The alert is *transmitted* to destinations such as emergency operations centers, mobile devices, or broadcast systems.

### IV. System for Providing Earthquake Alerts

The system comprises a seismic data ingestion module, a declustering filter, a spatial-temporal binning engine, a statistical analyzer computing p-values and risk factors, an alert generator, and a dissemination network. Components operate in real time, integrating data from global seismic networks and historical catalogs.

### V. A Generalized Computer Environment

The invention operates in a computing environment featuring a *processing unit* (CPU/GPU), *memory* (RAM), *storage* (SSD/HDD), *input devices* (network interfaces), *output devices* (displays, speakers), and *communication ports* (Ethernet, cellular). It may reside in a *computing cloud* with distributed resources. *Software instructions* implement the method via *program modules* executing *computer-executable instructions* in languages like Python or C++.

### VI. General Considerations

The term *comprising* is open-ended, allowing additional elements. *Or* is inclusive unless context dictates exclusivity. *About* permits ±10% variation. Numerical parameters include ranges (e.g., 30°–175°), which are exemplary, not limiting. The invention’s novelty lies in its statistical detection of non-local triggering, use of dual experimental designs, and operational alerting framework. It encompasses apparatus, methods, *computer-readable media*, and *software applications* deployable in *network environments*.