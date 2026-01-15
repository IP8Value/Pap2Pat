Here is the complete patent application following your outline:

# DESCRIPTION  

## FIELD  

The present invention relates generally to the field of cardiac electrophysiology and specifically to systems and methods for analyzing intracardiac electrograms (IEGMs) to detect and characterize abnormalities in atrial fibrillation (AF) wavefront propagation. More particularly, the invention provides novel signal processing techniques for extracting regional dominant frequency (RDF) and wave break rate (WBR) metrics from IEGMs, enabling improved identification of AF sources and optimization of ablation therapy.  

## BACKGROUND  

Atrial fibrillation represents the most prevalent cardiac arrhythmia and constitutes a major cause of thromboembolic stroke. The condition manifests through complex spatiotemporal patterns of electrical wavefront propagation within the atrial myocardium, resulting in highly fragmented and heterogeneous electrogram morphologies.  

Current clinical practice employs catheter ablation therapy targeting pulmonary vein isolation and other anatomical structures, often guided by electrogram characteristics such as complex fractionated electrograms (CFE) or dominant frequency measurements. However, these conventional approaches demonstrate significant limitations, particularly in persistent AF cases where long-term success rates remain suboptimal. The fundamental challenge stems from an incomplete understanding of the underlying mechanisms sustaining AF, particularly regarding the dynamic interactions between wavefront propagation patterns and tissue substrate properties.  

Existing methods for analyzing IEGMs typically rely on either time-domain features (e.g., activation timing, electrogram fractionation) or frequency-domain characteristics (e.g., dominant frequency) calculated over extended recording periods. Such approaches fail to capture the transient variations in wavefront dynamics that may indicate critical sites for AF perpetuation. There exists a pressing clinical need for improved analytical methods that can better characterize the spatiotemporal heterogeneity of AF wavefront propagation and identify mechanistically significant regions for targeted ablation therapy.  

## SUMMARY  

The present invention provides a novel method for detecting and characterizing abnormalities in atrial fibrillation through advanced analysis of intracardiac electrograms. The method involves extracting multiple features from IEGMs using specialized signal processing techniques, followed by fusion of these features to generate comprehensive regional characterizations of wavefront dynamics.  

Key aspects of the invention include performing time-frequency analysis on IEGM signals to detect spatiotemporal heterogeneity in wavefront propagation patterns. The system calculates both electrode-level dominant frequencies (EDF) and regional dominant frequencies (RDF) through optimized spectral estimation techniques. A particularly innovative feature involves the quantification of wave break rate (WBR) as a metric for characterizing discontinuities in wavefront propagation.  

The processing pipeline incorporates mechanisms for excluding irrelevant signal components and artifacts through advanced filtering techniques. The system outputs results through graphical displays including color-coded maps that visually identify regions of interest based on calculated RDF and WBR values. These outputs enable clinicians to identify potential sources of cardiac atrial fibrillation by detecting characteristic changes in wavefront dynamics.  

The invention provides quantitative measures of wave break rate that correlate with critical sites for AF termination. The integrated analysis of both RDF and WBR metrics offers superior characterization of AF mechanisms compared to conventional single-parameter approaches. Display outputs are optimized for clinical interpretation, facilitating rapid identification of optimal ablation targets during electrophysiology procedures.  

## DETAILED DESCRIPTION  

The following detailed description provides comprehensive information about the systems and methods comprising the present invention. The described embodiments represent preferred implementations but do not limit the scope of the invention, which encompasses various alternative configurations and applications.  

The invention provides methods for analyzing intracardiac electrograms to determine wavefront characteristics during atrial fibrillation. FIG. 1A presents a block diagram overview of the processing pipeline, which includes several innovative components for feature extraction and fusion. The system acquires IEGM signals from multi-electrode mapping catheters and applies specialized preprocessing to enhance signal quality and extract relevant features.  

A critical innovation involves the time-frequency and/or time-scale analysis of regional features derived from multiple simultaneous electrode recordings. This approach enables detection of transient variations in wavefront propagation that would be obscured in conventional whole-segment analyses. The system was validated through an embodiment involving twenty patients undergoing catheter ablation for AF, with rigorous correlation between MATLAB-generated maps and procedural outcomes.  

### 1A. Regional Dominant Frequency and Wave Break Rate  

The invention calculates electrode dominant frequency (EDF) through advanced spectral analysis of preprocessed IEGMs. The preprocessing stage involves applying specialized filters to remove baseline wander and high-frequency noise while preserving critical signal components. For each electrode pair, the system estimates the instantaneous EDF (iEDF) using short-time Fourier transform (STFT) with optimized windowing parameters.  

Regional dominant frequency (RDF) represents a novel metric derived through fusion of multiple electrode signals. The preprocessing for RDF calculation involves averaging processed signals from all catheter electrodes, followed by application of a two-sided exponential finite impulse response (FIR) filter. This unique filtering approach enables detection of discontinuities in wavefront propagation while maintaining temporal resolution.  

The system calculates instantaneous RDF (iRDF) using STFT analysis with a 1-second window duration and 95% overlap, providing optimal balance between time and frequency resolution. The upper quartile of iRDF values within each analysis segment is extracted as the final RDF metric. Wave break (WB) events are identified as transient drops in iRDF exceeding 3 Hz below the baseline RDF value or falling below 0.5 Hz, with durations exceeding 100 ms.  

Wave break rate (WBR) constitutes a key innovation, quantifying the frequency of wavefront discontinuities per unit time. The system calculates WBR by counting valid WB events within each analysis segment and normalizing by segment duration. WBR serves as a robust feature for characterizing wavefront propagation quality at each sampled site, with higher values indicating greater disorganization.  

### 1B. Example of RDF-Based Wave Break Identification  

An exemplary implementation demonstrates the system's ability to identify wave breaks through RDF analysis. FIG. 2 illustrates sample electrograms recorded from the left atrial roof during persistent AF, along with processed outputs at each analysis stage. During periods of organized wavefront propagation, the preprocessed signals from all electrodes exhibit synchronized peaks, producing a coherent averaged signal with well-defined spectral components.  

In contrast, wave break events manifest through temporal dispersion of electrode activations, resulting in multiple small peaks in the averaged signal. These high-frequency components are effectively attenuated by the specialized FIR filter, producing characteristic drops in iRDF. The example clearly shows three distinct wave breaks occurring within a 30-second recording segment, with the system correctly calculating a WBR of 0.1 WB/s for this region.  

### 1C. Minimum Required Segment Duration for Accurate RDF Estimation  

The invention establishes optimized recording durations for reliable parameter estimation. Through systematic analysis comparing shorter segments to a 30-second gold standard, the system determines that RDF can be accurately estimated from just 4 seconds of data (Pearson correlation >0.9 with 30-second reference). This represents a significant improvement over conventional approaches requiring much longer recordings.  

### 1D. Minimum Required Segment Duration for Accurate WBR Estimation  

Similar analysis for WBR estimation determines that 25-second segments provide reliable quantification (correlation >0.85 with 50-second reference). The differential requirements for RDF and WBR estimation reflect their distinct temporal characteristics, with WBR benefiting from longer observation periods to capture intermittent wave break events.  

### 1E. Statistics  

The system employs robust statistical methods for parameter comparison and validation. Non-parametric tests (Mann-Whitney U) compare metrics between patient groups, while Spearman's rank correlation assesses relationships between parameters. Statistical significance is established at p<0.05, with results reported as mean ± standard deviation or median with ranges as appropriate.  

### 1F. Implementation  

The invention is implemented through specialized software operating on standard computer hardware. The data processing system includes a user interface for parameter adjustment and display control, input devices for operator interaction, and a central processing unit executing the analysis algorithms.  

The system architecture incorporates memory for temporary data storage, display devices for visualization of results, and interface devices for connection to electrophysiology recording systems. Network connections enable data sharing and remote access, while database systems provide long-term storage of procedural data.  

Computer-executable programmed instructions implement the analytical methods described herein, with a graphical user interface (GUI) providing intuitive control and visualization. The GUI displays color-coded maps of calculated parameters overlaid on anatomical models, facilitating rapid clinical interpretation.  

### 1G. Results  

Clinical validation involved fifteen patients after excluding five for poor data quality. The patient cohort included five paroxysmal and ten persistent AF cases, with comprehensive left atrial mapping performed prior to ablation. Recordings averaged 24.4±7 sites per patient with 29.9±9.8 second durations.  

Analysis revealed significant differences between paroxysmal and persistent AF, with higher mean RDF (5.99±0.8 Hz vs 5.32±0.75 Hz) and WBR (0.24±0.14 vs 0.14±0.11 WB/s) in paroxysmal cases (p<0.001). Spatial heterogeneity was evident for both parameters, with weak correlation between RDF and WBR (r=0.3, p<0.001).  

Procedural outcomes demonstrated clinical relevance, with ablation terminating AF at sites exhibiting high RDF and low WBR in 8/9 cases. These "↑RDF,↓WBR" sites showed distinct spatial distributions, being more prevalent in pulmonary veins for paroxysmal AF (63%) compared to persistent cases (33%).  

### 1H. Discussion  

The invention introduces novel metrics for AF investigation, overcoming limitations of conventional approaches. Regional dominant frequency provides robust characterization of wavefront dynamics without requiring precise activation timing, while wave break rate quantifies propagation discontinuities that may indicate critical sites.  

The combined analysis of RDF and WBR offers mechanistic insights, with ↑RDF,↓WBR sites correlating strongly with AF termination during ablation. This pattern suggests these regions may represent stable sources surrounded by areas of wave break, consistent with experimental rotor models.  

The system's sequential mapping approach provides practical advantages over simultaneous panoramic techniques, enabling higher resolution analysis with standard clinical catheters. The optimized recording durations (4s for RDF, 25s for WBR) facilitate efficient integration into clinical workflows.  

### 2. Computer Modelling of Spiral Rotor and Associated Wave Break Analysis  

Computer simulations validate the invention's analytical principles using a modified FitzHugh-Nagumo model of cardiac excitation. The model generates spiral rotor activity resembling clinical AF, with calculated unipolar and bipolar electrograms demonstrating characteristic iRDF drops at wave break locations.  

### 3. Clinical Example of an Identified Rotor During Wave Break  

FIG. 3 presents a clinical example showing rotational activity coinciding with wave break detection. The propagation map demonstrates organized rotation centered on a region exhibiting ↑RDF,↓WBR characteristics, subsequently terminated by ablation at this site.  

## EQUIVALENTS  

The scope of the invention encompasses all modifications, variations, and equivalents that utilize the essential principles and achieve substantially similar results. This includes alternative signal processing techniques, different catheter configurations, and various implementations of the analytical algorithms while maintaining the core innovation of regional wavefront characterization through combined RDF and WBR analysis.