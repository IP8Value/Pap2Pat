Here is the patent application following your outline:

# DESCRIPTION  

The present application claims the benefit of European Patent Application No. [Application Number], filed [Filing Date], which is hereby incorporated by reference in its entirety.  

## TECHNICAL FIELD  

The present invention relates generally to radar-based sensing systems and more particularly to systems and methods for accurately counting people within a monitored environment using radar measurements processed through specialized neural network architectures.  

## BACKGROUND  

People counting has become an essential capability across numerous applications, including occupancy monitoring, security surveillance, retail analytics, and energy management systems. Traditional approaches relying on image-based sensors suffer from well-documented limitations regarding privacy concerns, lighting dependency, and occlusion challenges in crowded environments. While radar sensors provide inherent advantages by operating independently of visible light conditions and preserving anonymity, conventional radar signal processing techniques struggle with low-resolution data interpretation, signal superposition from multiple body reflections, and unstable detection reliability—particularly in dense or confined spaces like vehicle cabins.  

Prior attempts to apply deep learning techniques to radar-based people counting have improved upon traditional methods but remain constrained by their inability to leverage the inherent ordinal relationships present in counting tasks. Existing approaches treat people counting as either a classification problem without preserving label ordering or a regression problem without structured embedding spaces, resulting in suboptimal accuracy and robustness—especially in challenging real-world scenarios with frequent occlusions and dynamic movements.  

## SUMMARY  

The present invention provides a novel radar-based people counting system that overcomes these limitations through an advanced neural network architecture specifically designed to process distinct radar measurement maps while preserving ordinal relationships in the embedding space.  

The system operates by first obtaining a first range-Doppler measurement map (RDI) optimized for macro-Doppler features representing bulk body movements and a second range-Doppler measurement map optimized for micro-Doppler features representing finer limb motions. These complementary representations are processed through separate but interconnected data processing pipelines within a neural network architecture comprising specialized range-Doppler convolutional layers.  

The network architecture incorporates:  
- An encoder branch with spatial contraction through pooling layers  
- Parallel macro-Doppler and micro-Doppler processing pipelines with 2D convolutional layers  
- Strategically placed connecting sections that fuse features while preserving distinct processing pathways  
- A regression block that consolidates processed features into predictive outputs  

Key innovations include:  
1) A label-aware ranked (LAR) loss function that explicitly structures the embedding space to maintain ordinal relationships between people counts, mathematically enforcing uniform angular separation between label representations  
2) Temporal smoothing techniques including exponential moving averages and Kalman filtering to stabilize predictions across sequential radar frames  
3) A specialized preprocessing chain generating optimized macro-Doppler and micro-Doppler RDIs through:  
   - MTI filtering and 2D FFT processing for macro-Doppler RDIs  
   - Multi-frame integration and Hamming window application for micro-Doppler RDIs  

The complete system implementation includes program code for:  
- Loading and executing the computer-implemented people counting method  
- Determining the first and second range-Doppler measurement maps from raw radar data  
- Inputting the measurement maps into the neural network algorithm  
- Applying tracking filters to monitor evolution within the structured embedding space  
- Outputting both instantaneous and smoothed people count predictions  

Training methodologies incorporate:  
- Acquisition of multiple labeled training radar measurement datasets  
- Joint optimization of macro-Doppler and micro-Doppler pipelines using the LAR loss  
- Label-aware ranked loss formulations that exploit distance relationships between count values  

## DETAILED DESCRIPTION OF ILLUSTRATIVE EMBODIMENTS  

### Electrical Devices and Circuits  

The invention utilizes standard radar front-end components including:  
- FMCW radar transceivers operating in 60 GHz frequency bands  
- Programmable chirp generators with adjustable repetition intervals  
- Multi-channel receiver arrays with analog-to-digital converters  
- Digital signal processors implementing MTI filtering and 2D FFT operations  

Limitations of conventional implementations include fixed velocity resolution constraints and inability to simultaneously capture both macro and micro-Doppler signatures effectively.  

### Radar Measurement Operation  

The radar sensor operates in pulsed FMCW mode with:  
- Adjustable chirp repetition times (CRT) between 100μs-1ms  
- Frequency sweeps covering 1-4 GHz bandwidths  
- Frame repetition frequencies of 10-30 Hz  
- Fast-time sampling across 256-512 samples per chirp  
- Slow-time integration across 32-128 chirps per frame  

Doppler frequency shifts are calculated through phase analysis of consecutive chirps, with maximum resolvable velocities determined by CRT selection according to:  
v_max = λ/(4·T_c) where λ is wavelength and T_c is chirp repetition time  

### Machine Learning Architecture  

The neural network processes:  
1) Macro-Doppler RDIs obtained through:  
   - MTI filtering along slow-time dimension  
   - 2D FFT with Hamming windowing  
   - Magnitude calculation from complex outputs  

2) Micro-Doppler RDIs obtained through:  
   - Multi-frame integration (5-10 frames)  
   - Slow-time FFT with increased zero-padding  
   - Logarithmic magnitude scaling  

The network architecture features:  
- Encoder branches with 3-5 convolutional layers (32-64 filters, 3×3 kernels)  
- Max pooling between convolutional stages  
- Bottleneck layers reducing spatial dimensions while increasing feature depth  
- Connecting sections implementing either:  
   a) Parameterized combination layers with learned weighting  
   b) Concatenation followed by 1×1 convolutions  

### Training Methodology  

The system is trained through:  
1) Acquisition of labeled datasets covering 0-5 people counts  
2) Smart batch construction ensuring equal label representation  
3) Joint optimization using:  
   - LAR loss component enforcing angular separation  
   - MSE loss component for regression accuracy  
4) Label-aware ranked loss formulations incorporating:  
   - Logarithmic scaling of label distance multipliers  
   - Uniform angular separation constraints  

### Inference Operation  

During deployment, the system:  
1) Captures radar frames at 10 Hz  
2) Generates macro/micro-Doppler RDIs  
3) Feeds RDIs through trained network  
4) Applies exponential smoothing to outputs  
5) Tracks positions in embedding space using Kalman filters  
6) Outputs stabilized people counts  

### Use Case Implementations  

1) Vehicle Occupancy Monitoring:  
   - Installation on windshield upper surface  
   - HVAC control based on real-time counts  
   - Seatbelt reminder systems  

2) Retail Traffic Analysis:  
   - Overhead mounting in store entrances  
   - Conversion rate tracking  
   - Peak hour staffing adjustments  

3) Building Energy Management:  
   - Conference room occupancy tracking  
   - Lighting/HVAC zone control  
   - Security alert generation  

The complete system provides robust people counting in challenging environments while maintaining privacy compliance and operational reliability across varying conditions.