# DESCRIPTION

- claim benefit of European Patent Application

This patent application claims the benefit of priority under 35 U.S.C. § 119(e) and/or 35 U.S.C. § 120 to European Patent Application No. EP 23 198 765.2, filed on September 28, 2023, entitled “Method and System for People Counting Using Radar Measurements and Label-Aware Ranked Loss,” the entire disclosure of which is hereby incorporated by reference in its entirety. The present invention builds upon the technical disclosures and experimental validations contained therein, extending their scope through formalized claims directed to novel architectures, training methodologies, and system implementations for estimating human occupancy using radar-based sensing and deep learning. The invention is not merely an incremental improvement but a fundamentally structured approach to regression-based people counting that leverages the intrinsic ordinal relationship among label values to enforce a geometrically ordered embedding space, thereby achieving unprecedented accuracy and temporal stability in dynamic environments.

## TECHNICAL FIELD

- introduce people counting based on radar measurements

The present invention relates generally to the field of sensor-based occupancy estimation, and more particularly to systems and methods for counting the number of individuals within a confined or semi-confined space using radar measurements. The invention is especially suited for environments where visual sensing is impractical or undesirable, such as within automotive cabins, public transit vehicles, elevators, or enclosed workspaces, due to constraints imposed by privacy regulations, lighting conditions, or occlusion. Unlike optical cameras, radar sensors operate effectively in complete darkness, through non-metallic materials, and under adverse weather conditions, making them uniquely suited for continuous, unobtrusive monitoring. The invention employs frequency-modulated continuous wave (FMCW) radar technology to capture subtle reflections from human bodies, and processes these signals through a multi-stream deep neural network architecture that distinguishes between macro-Doppler and micro-Doppler motion signatures to infer accurate and robust people counts without requiring direct line-of-sight or identifiable visual features.

## BACKGROUND

- motivate people counting

Accurate estimation of human occupancy is critical for optimizing energy consumption in heating, ventilation, and air conditioning systems, enhancing safety and comfort in transportation environments, enabling intelligent access control, and supporting crowd management in public infrastructure. In commercial and industrial settings, real-time occupancy data allows for dynamic resource allocation, predictive maintenance scheduling, and improved operational efficiency. In automotive applications, occupancy information enables adaptive climate control, seatbelt reminders, airbag deployment logic, and driver monitoring systems. The demand for reliable, privacy-preserving, and low-cost occupancy sensing has grown substantially as smart environments become more pervasive, necessitating solutions that transcend the limitations of conventional approaches.

- limitations of image-based people counting

Image-based systems, while widely adopted, suffer from inherent vulnerabilities that limit their applicability in real-world deployments. These systems are highly sensitive to illumination variations, occlusions caused by objects or body posture, and environmental obstructions such as fog, smoke, or glare. Furthermore, they raise significant privacy concerns, as they capture and process visual data that may include identifiable facial features, clothing, or behavioral patterns, often violating data protection statutes such as the General Data Protection Regulation. Additionally, image sensors require substantial computational resources for real-time inference, and their performance degrades significantly in low-light or high-contrast conditions. These limitations render them unsuitable for continuous, unobtrusive, and legally compliant deployment in sensitive environments such as private vehicles, medical facilities, or residential spaces.

## SUMMARY

- summarize people counting based on radar measurements

The invention provides a computer-implemented method and system for estimating the number of individuals present in a scene using radar measurements, wherein the estimation is performed by a neural network architecture that processes distinct radar-derived feature maps corresponding to macro-Doppler and micro-Doppler motion signatures. The method achieves high accuracy and temporal stability by leveraging a label-aware ranked loss function during training, which enforces a geometrically ordered embedding space where the angular separation between feature vectors of different occupancy levels corresponds to the ordinal relationship of their labels. The system operates without requiring visual data, is resilient to environmental interference, and produces continuous, smoothed estimates suitable for real-time control applications.

- define 1st range-Doppler measurement map

The invention defines a first range-Doppler measurement map as a two-dimensional representation derived from a fast-time and slow-time radar signal frame, wherein the fast-time dimension corresponds to range bins generated by the Fourier transform of individual chirps, and the slow-time dimension corresponds to Doppler frequency bins generated by the Fourier transform across successive chirps, with a high-pass filtering operation applied to suppress static clutter and direct leakage, thereby emphasizing moving targets and capturing gross body motion signatures.

- define 2nd range-Doppler measurement map

The invention defines a second range-Doppler measurement map as a two-dimensional representation derived from an integrated sequence of multiple radar frames, wherein the integration spans a longer temporal window to enhance velocity resolution and capture subtle micro-motions associated with respiration, limb movement, or torso oscillation, with a moving target indication filter applied to isolate dynamic components and a Hamming window applied to reduce spectral leakage, thereby emphasizing fine-grained biological motion signatures.

- estimate people count using neural network algorithm

The invention estimates the number of people present in a scene by inputting the first and second range-Doppler measurement maps into a neural network algorithm that processes them through separate, parallel data processing pipelines, each optimized for macro-Doppler and micro-Doppler features, respectively, and fuses their outputs in a joint regression block to produce a single scalar prediction of occupancy.

- input 1st range-Doppler measurement map into 1st data processing pipeline

The invention inputs the first range-Doppler measurement map into a first data processing pipeline configured to extract spatial and velocity patterns associated with large-scale human movement, such as walking, shifting posture, or overall displacement, using a series of two-dimensional convolutional layers with increasing receptive fields and non-linear activation functions to encode macro-Doppler context.

- input 2nd range-Doppler measurement map into 2nd data processing pipeline

The invention inputs the second range-Doppler measurement map into a second data processing pipeline configured to extract fine-grained temporal patterns associated with biological micro-motions, such as breathing, finger movement, or head nodding, using a series of two-dimensional convolutional layers with smaller kernel sizes and higher temporal resolution to encode micro-Doppler context.

- include range-Doppler convolutional layers

The invention includes range-Doppler convolutional layers within both the first and second data processing pipelines, wherein each layer applies a set of learnable filters to the range-Doppler maps to detect localized patterns of motion energy across range and velocity dimensions, with each layer followed by batch normalization and a rectified linear unit activation function to enhance feature discrimination and training stability.

- process outputs in regression block

The invention processes the outputs of the first and second data processing pipelines in a regression block that combines their encoded representations through a connecting section comprising a concatenation layer and a subsequent convolutional layer, followed by a fully-connected layer with a single neuron that outputs a continuous scalar value representing the predicted number of individuals.

- output 1-dimensional value predicting people count

The invention outputs a one-dimensional value that directly predicts the number of people present in the scene, wherein the value is unconstrained by discrete classification boundaries and is trained to minimize a label-aware ranked loss function that enforces ordinal ordering of the embedding space according to the true occupancy labels.

- include fully-connected layer with single neuron

The invention includes a fully-connected layer with a single neuron positioned at the final stage of the regression block, wherein the neuron receives the fused feature vector from the connecting section and produces a scalar output that corresponds to the estimated people count, with no intermediate classification layers, thereby enabling continuous regression and preserving the ordinal relationship between predicted values.

- output higher-dimensional value

The invention outputs a higher-dimensional value prior to the final fully-connected layer, wherein the higher-dimensional value represents an embedding vector in a latent space that encodes the combined macro-Doppler and micro-Doppler features of the input radar data, and wherein the dimensionality of this embedding is sufficient to support the geometric separation of labels according to their ordinal ranking.

- predict people count based on position in embedding space

The invention predicts the people count based on the position of the embedding vector within a learned latent space, wherein the spatial arrangement of embedding vectors for different occupancy levels is constrained during training such that the angular separation between vectors corresponds to the difference in their label values, and wherein the final prediction is derived by projecting the embedding onto a one-dimensional axis aligned with the ordinal structure of the labels.

- define micro-Doppler range-Doppler measurement map

The invention defines a micro-Doppler range-Doppler measurement map as a time-integrated representation of radar returns that captures sub-meter-scale oscillatory motions of the human body, such as respiration and slight limb movements, obtained by stacking multiple consecutive radar frames and applying a moving target indication filter to isolate dynamic components while preserving fine temporal variations.

- define macro-Doppler range-Doppler measurement map

The invention defines a macro-Doppler range-Doppler measurement map as a short-duration representation of radar returns that captures gross body motion, such as walking or repositioning, obtained by processing individual radar frames with a high-pass filter to remove static clutter and applying a two-dimensional fast Fourier transform to resolve range and velocity components.

- determine velocity resolution

The invention determines the velocity resolution of the radar system by calculating the inverse of the total observation time across the slow-time dimension, wherein the observation time is defined as the product of the number of chirps per frame and the repetition time between consecutive chirps, and wherein the velocity resolution is further refined by the use of windowing functions to minimize spectral leakage.

- include program code for computer-implemented method

The invention includes non-transitory computer-readable program code stored on a tangible medium, wherein the program code, when loaded and executed by a processing unit, causes the system to perform the steps of acquiring radar measurement data, preprocessing the data to generate first and second range-Doppler measurement maps, inputting the maps into a trained neural network architecture, and outputting a scalar prediction of the number of individuals present.

- load and execute program code

The invention includes the step of loading the program code into a memory of a processing device and executing it in real time, wherein the processing device is integrated with a radar sensor and operates independently of external network connectivity, enabling autonomous, low-latency occupancy estimation in embedded environments.

- perform computer-implemented method

The invention performs a computer-implemented method comprising the steps of determining a first range-Doppler measurement map from a set of radar chirps, determining a second range-Doppler measurement map from an integrated sequence of radar frames, inputting both maps into a neural network algorithm trained with a label-aware ranked loss function, and outputting a single scalar value representing the estimated number of people.

- input measurement maps into neural network algorithm

The invention inputs the first and second range-Doppler measurement maps simultaneously into a neural network algorithm that processes them through parallel data processing pipelines, wherein each pipeline is optimized for a distinct type of motion signature, and wherein the outputs of the pipelines are fused before being passed to a regression block for final prediction.

- apply tracking filter

The invention applies a tracking filter to the sequence of predicted people counts over time, wherein the tracking filter models the temporal evolution of occupancy as a constant-velocity motion process, and wherein the filter smooths abrupt changes and reduces noise by incorporating prior estimates into the current prediction.

- track evolution of output in embedding space

The invention tracks the evolution of the output embedding vector in the latent space over successive time steps, wherein the embedding space is structured such that adjacent occupancy levels are uniformly spaced in angular distance, and wherein the tracking filter uses the predicted embedding position to infer the most likely occupancy state based on motion dynamics.

- obtain multiple training radar measurement datasets

The invention obtains multiple training radar measurement datasets, each comprising a sequence of radar frames labeled with the true number of individuals present, wherein the datasets are collected across diverse environmental conditions, seating arrangements, and human postures to ensure robust generalization.

- perform training of neural network algorithm

The invention performs training of the neural network algorithm using a label-aware ranked loss function that minimizes the angular distance between embedding vectors of the same label while maximizing the angular separation between vectors of different labels in proportion to the difference in their ordinal values.

- use label-aware ranked loss

The invention uses a label-aware ranked loss function that incorporates the logarithmic difference between label values as a weighting factor in the loss computation, thereby enforcing a geometrically ordered embedding space where the angular separation between embeddings corresponds to the relative magnitude of their occupancy labels, and wherein the loss function is optimized using gradient descent over a batch of triplets sampled from the training dataset.

## DETAILED DESCRIPTION OF ILLUSTRATIVE EMBODIMENTS

- introduce electrical devices and circuits

The invention introduces electrical devices and circuits designed for the acquisition, processing, and interpretation of radar signals in real time, including a frequency-modulated continuous wave radar transceiver, an analog-to-digital converter, a field-programmable gate array for real-time signal preprocessing, and a microprocessor unit for executing the neural network inference and tracking algorithms.

- describe functionality of electrical devices and circuits

The functionality of the electrical devices and circuits is to convert electromagnetic reflections from human bodies into digital radar data, preprocess the data to extract range-Doppler features, and feed these features into a neural network for occupancy estimation, with all operations performed locally on the device without reliance on cloud-based computation.

- explain limitations of electrical devices and circuits

The limitations of conventional electrical devices and circuits for people counting include their inability to distinguish between overlapping motion signatures, their sensitivity to multipath reflections, and their lack of temporal coherence in output predictions, which result in noisy, unstable estimates that are unsuitable for control applications.

- motivate people counting based on radar measurements

People counting based on radar measurements is motivated by the need for a privacy-preserving, robust, and low-power sensing modality that functions reliably in environments where visual sensing is prohibited or impractical, such as in private vehicles, public transit, or healthcare facilities.

- introduce radar measurement operation

The radar measurement operation involves transmitting a series of frequency-modulated chirps and receiving the reflected signals, which are then processed to generate a two-dimensional representation of target range and radial velocity, forming the basis for subsequent feature extraction.

- describe radar sensor and its functionality

The radar sensor is a 60 GHz FMCW transceiver capable of emitting chirps with a bandwidth of 4 GHz and a repetition rate of 10 Hz, with four transmit and six receive antennas arranged in a linear array to enable beamforming and angular resolution.

- explain pulsed operation of radar sensor

The pulsed operation of the radar sensor consists of transmitting a sequence of linear frequency-modulated pulses, each lasting a few microseconds, followed by a silent interval during which the received echoes are sampled, with the repetition of this cycle forming the basis of the slow-time dimension in the range-Doppler map.

- describe Doppler frequency shift and its application

The Doppler frequency shift is the change in frequency of the reflected signal caused by the relative motion between the radar and the target, and is used to determine the radial velocity of moving objects, enabling the separation of static clutter from dynamic human motion.

- motivate machine-learning algorithm for people counting

A machine-learning algorithm is motivated for people counting because traditional signal processing techniques fail to disentangle overlapping reflections from multiple individuals, whereas deep learning can learn complex, non-linear mappings from raw radar features to occupancy labels.

- introduce range-doppler measurement maps

Range-Doppler measurement maps are two-dimensional matrices that represent the distribution of reflected signal energy across range bins and Doppler bins, providing a compact representation of the spatial and velocity characteristics of targets in the scene.

- describe 2-D Fourier transformation for obtaining RDIs

The two-dimensional Fourier transformation is applied to the raw radar data along both the fast-time (range) and slow-time (Doppler) dimensions to convert time-domain samples into frequency-domain representations, yielding the range-Doppler images used as input to the neural network.

- introduce 2-D angular measurement maps

Two-dimensional angular measurement maps are generated by applying a beamforming algorithm to the multi-antenna radar data, enabling the resolution of target angles and the creation of a spatial map of reflectivity across azimuth and elevation.

- describe beamforming algorithm for obtaining 2-D angular measurement maps

The beamforming algorithm applies phase shifts to the signals received by each antenna element to steer the sensor’s sensitivity in specific directions, and then sums the coherently delayed signals to form a spatially resolved map of reflectivity.

- introduce range-angle measurement maps

Range-angle measurement maps are generated by combining range information from the fast-time Fourier transform with angular information from the beamforming algorithm, providing a three-dimensional representation of target location in range and angle.

- motivate differentiation between micro-Doppler and macro-Doppler features

The differentiation between micro-Doppler and macro-Doppler features is motivated by the distinct physical origins of these signals: macro-Doppler arises from gross body motion, while micro-Doppler arises from internal biological motions, and their separation enhances the discriminative power of the occupancy estimation.

- describe macro-Doppler features and micro-Doppler features

Macro-Doppler features correspond to low-frequency velocity components associated with whole-body movement, while micro-Doppler features correspond to high-frequency modulations caused by respiration, limb oscillation, or torso vibration, each contributing unique information to the occupancy prediction.

- introduce first RDI for macro-Doppler features

The first range-Doppler image is introduced as the primary input for capturing macro-Doppler features, generated from a single radar frame with high temporal resolution and minimal integration, emphasizing transient motion events.

- introduce second RDI for micro-Doppler features

The second range-Doppler image is introduced as the primary input for capturing micro-Doppler features, generated by integrating multiple consecutive frames to enhance velocity resolution and isolate subtle periodic motions.

- describe input to ML algorithm

The input to the machine learning algorithm consists of four channels derived from the real and imaginary components of both the first and second range-Doppler images, forming a six-dimensional tensor that encodes both spatial and dynamic information.

- introduce macro-Doppler data processing pipeline

The macro-Doppler data processing pipeline is a convolutional neural network branch designed to extract features from the first range-Doppler image, consisting of three convolutional layers with increasing receptive fields, each followed by batch normalization and ReLU activation.

- introduce micro-Doppler data processing pipeline

The micro-Doppler data processing pipeline is a convolutional neural network branch designed to extract features from the second range-Doppler image, consisting of three convolutional layers with smaller kernel sizes and higher temporal sensitivity to capture fine-grained oscillations.

- describe neural network architecture

The neural network architecture comprises two parallel encoder branches for macro- and micro-Doppler features, a connecting section that fuses their outputs, and a regression block that produces a scalar occupancy prediction, with all components trained end-to-end using a label-aware ranked loss function.

- explain spatial contraction and expansion in encoder and decoder branches

The encoder branches exhibit spatial contraction through successive pooling operations that reduce the resolution of feature maps while increasing their depth, enabling the extraction of abstract, high-level representations of motion patterns.

- describe 2-D convolutional layers in macro-Doppler and micro-Doppler data processing pipelines

The two-dimensional convolutional layers in both pipelines apply learnable filters to the range-Doppler maps to detect localized patterns of motion energy, with the macro-Doppler pipeline using larger kernels to capture broad motion trends and the micro-Doppler pipeline using smaller kernels to resolve fine oscillations.

- introduce output section for processing combined outputs

The output section is introduced as the final component of the neural network, comprising a concatenation layer, a convolutional layer for feature fusion, and a fully-connected layer with a single neuron that outputs the predicted number of individuals.

- describe joint training of macro-Doppler and micro-Doppler data processing pipelines

The macro-Doppler and micro-Doppler data processing pipelines are jointly trained using a single loss function that simultaneously optimizes both branches to produce embeddings that are ordered according to the true occupancy labels, ensuring complementary feature extraction.

- motivate separate processing of micro-Doppler and macro-Doppler features

Separate processing of micro-Doppler and macro-Doppler features is motivated by their distinct physical origins and temporal scales, which, when processed independently, allow the network to learn specialized representations that are more discriminative than a unified representation.

- introduce connecting sections for joining macro-Doppler and micro-Doppler data processing pipelines

Connecting sections are introduced to merge the encoded representations from the macro- and micro-Doppler pipelines, enabling the network to exploit correlations between gross and fine motion patterns that are indicative of human presence.

- describe feature fusion at connecting sections

Feature fusion at the connecting sections is achieved through concatenation of the feature maps followed by a convolutional layer that learns to weight and combine the contributions of macro- and micro-Doppler features into a unified embedding.

- introduce people counting using radar sensor

People counting using a radar sensor is introduced as a privacy-preserving alternative to visual sensing, capable of operating in complete darkness, through clothing and non-metallic materials, and without capturing identifiable biometric data.

- motivate correlation between macro-Doppler and micro-Doppler features

The correlation between macro-Doppler and micro-Doppler features is motivated by the observation that human motion is inherently hierarchical, with gross movement often accompanied by consistent micro-motions, and that their joint modeling improves robustness to occlusion and sensor misalignment.

- describe use of connecting sections to capture correlations

The connecting sections are used to capture correlations between macro- and micro-Doppler features by enabling the flow of information between the two pipelines, allowing the network to learn joint representations that are invariant to individual motion variations.

- enhance robustness against variation in radar sensor pose

The system enhances robustness against variation in radar sensor pose by learning features that are invariant to orientation and position, as demonstrated by consistent performance across multiple mounting configurations and viewing angles.

- provide options for implementing connecting sections

Options for implementing connecting sections include concatenation followed by convolution, element-wise addition, or attention-based weighting, each providing different trade-offs between computational complexity and representational capacity.

- describe combination layer with filter parameters

The combination layer is described as a convolutional layer with learnable filter parameters that operate on the concatenated feature maps from the two pipelines, producing a fused representation that encodes the joint statistical structure of macro- and micro-motion.

- describe concatenation layer and convolutional layer

The concatenation layer merges the feature maps from the two pipelines along the channel dimension, and the subsequent convolutional layer applies a set of filters to this combined representation to extract high-level joint features.

- combine different implementations for combination layers

Different implementations of the combination layer are combined in a modular architecture, allowing the system to be adapted to varying computational constraints and performance requirements.

- include regression block in neural network

The regression block is included in the neural network as the final stage, comprising the combination layer, a fully-connected layer with a single neuron, and a linear activation function to produce a continuous scalar prediction of occupancy.

- describe output of regression block

The output of the regression block is a single real-valued number that represents the estimated number of individuals present, with no discrete classification thresholds, enabling fine-grained estimation and smooth temporal transitions.

- control dimensionality of output using fully-connected layers

The dimensionality of the output is controlled using a fully-connected layer with a single neuron, ensuring that the final prediction is constrained to a one-dimensional space aligned with the ordinal structure of the labels.

- determine feature vector and people count

The feature vector is determined by the latent representation produced by the encoder and connecting sections, and the people count is determined by projecting this vector onto a learned one-dimensional axis that reflects the ordinal ranking of the labels.

- describe postprocessing techniques for people counting

Postprocessing techniques include the application of a smoothing filter to the sequence of predictions to suppress transient noise and a tracking filter to enforce temporal consistency based on a constant-velocity motion model.

- apply smoothing filter to avoid artificial changes

A smoothing filter is applied to the sequence of predicted people counts to avoid artificial fluctuations caused by sensor noise or momentary occlusions, thereby improving the reliability of the output for control systems.

- apply tracking filter to track evolution of embedding output

A tracking filter is applied to the evolution of the embedding output in the latent space, wherein the filter predicts the next state based on the previous state and velocity, reducing jitter and enhancing long-term stability.

- predict position in embedding space using tracking filter

The position in the embedding space is predicted using a Kalman filter that models the motion of the embedding vector as a linear process with Gaussian noise, enabling accurate estimation even under intermittent signal degradation.

- order predefined regions in embedding space

Predefined regions in the embedding space are ordered such that each region corresponds to a specific occupancy level, with the angular distance between regions proportional to the difference in label values, ensuring that the embedding space preserves the ordinal structure of the labels.

- use label-aware ranked loss to achieve ordering

The label-aware ranked loss is used to achieve this ordering by penalizing deviations in angular separation between embedding vectors in proportion to the difference in their true labels, thereby enforcing a geometrically consistent ranking.

- describe system including radar sensor and processing device

The system includes a radar sensor mounted on the interior surface of a vehicle cabin and a processing device integrated with the sensor, wherein the processing device executes the neural network algorithm and outputs the estimated people count to a control unit.

- illustrate radar sensor and its components

The radar sensor is illustrated as a compact module comprising a 60 GHz FMCW transceiver, a multi-antenna array, a local oscillator, and an analog-to-digital converter, all housed in a single printed circuit board.

- describe data processing for people counting

Data processing for people counting involves preprocessing the raw radar data to generate range-Doppler maps, feeding them into the neural network, and applying postprocessing filters to produce a stable, continuous estimate of occupancy.

- illustrate data processing using neural network and smoothing filter

The data processing is illustrated as a pipeline wherein the radar data is first converted into range-Doppler maps, then processed by the neural network to produce a raw prediction, and finally smoothed by an exponential moving average filter to yield the final output.

- preprocess radar measurement dataset to obtain macro-Doppler RDI and micro-Doppler RDI

The radar measurement dataset is preprocessed by applying a moving target indication filter and a two-dimensional fast Fourier transform to obtain the macro-Doppler RDI, and by integrating multiple frames and applying a high-pass filter to obtain the micro-Doppler RDI.

- execute MTI filtering and 2-D FFT for macro-Doppler preprocessing

Macro-Doppler preprocessing involves executing a moving target indication filter to remove static clutter and applying a two-dimensional fast Fourier transform to resolve range and velocity components from individual radar frames.

- implement high-pass filter to form macro-Doppler RDI

A high-pass filter is implemented in the slow-time dimension to eliminate low-frequency components associated with stationary objects and direct leakage, thereby isolating dynamic motion signatures for the macro-Doppler RDI.

- integrate multiple frames for micro-Doppler preprocessing

Multiple consecutive radar frames are integrated to enhance the velocity resolution and isolate periodic micro-motions, with the integration window selected to match the typical frequency of human respiration.

- apply moving target indication filter and 2-D FFT for micro-Doppler preprocessing

The moving target indication filter and two-dimensional fast Fourier transform are applied to the integrated frame sequence to generate a high-resolution micro-Doppler RDI that captures fine oscillatory patterns.

- use Hamming window to reduce spectral leakage

A Hamming window is applied along both the range and Doppler dimensions before each Fourier transform to reduce spectral leakage and improve the resolution of closely spaced reflectors.

- introduce radar measurement data

Radar measurement data is introduced as a sequence of complex-valued samples acquired from the radar receiver, organized into a three-dimensional tensor comprising fast-time samples, slow-time chirps, and antenna channels.

- explain structure of radar measurement frame

The structure of the radar measurement frame consists of a matrix of complex samples, where each row corresponds to a chirp and each column corresponds to a sample within the chirp duration, forming the basis for subsequent range-Doppler transformation.

- describe fast-time and slow-time dimensions

The fast-time dimension corresponds to the time within a single chirp and is used to determine range, while the slow-time dimension corresponds to the sequence of chirps and is used to determine Doppler velocity.

- explain antenna dimension

The antenna dimension refers to the number of receive channels in the radar array, each providing a distinct view of the reflected signal, enabling spatial resolution and beamforming capabilities.

- define duration of radar measurement frames

The duration of each radar measurement frame is defined as the time required to transmit and receive a complete set of chirps, typically 100 milliseconds, with a frame repetition frequency of 10 Hz.

- describe chirps repetition time

The chirps repetition time is the interval between the start of consecutive chirps, typically 1 millisecond, and determines the maximum unambiguous Doppler velocity that can be measured.

- calculate maximum resolve Doppler velocity

The maximum resolve Doppler velocity is calculated as half the product of the chirp repetition frequency and the speed of light divided by the carrier frequency, yielding a maximum detectable velocity of approximately 1.5 meters per second.

- explain frequency range of chirps

The frequency range of the chirps spans from 57 GHz to 61 GHz, with a bandwidth of 4 GHz, enabling a theoretical range resolution of 3.75 centimeters.

- calculate range resolution

The range resolution is calculated as the speed of light divided by twice the chirp bandwidth, resulting in a resolution of 3.75 centimeters, sufficient to distinguish between closely spaced individuals.

- describe frame repetition frequency

The frame repetition frequency is the rate at which complete radar frames are acquired, set to 10 Hz to balance temporal resolution with computational load and power consumption.

- illustrate neural network architecture

The neural network architecture is illustrated as a symmetric encoder-decoder structure with two parallel branches, each containing three convolutional layers, followed by a connecting section and a regression block with a single output neuron.

- explain encoder branch

The encoder branch is explained as a sequence of convolutional and pooling layers that progressively reduce the spatial dimensions of the input while increasing the number of feature channels, extracting hierarchical representations of motion patterns.

- describe regression block

The regression block is described as the final component of the network, comprising a fusion layer, a fully-connected layer with a single neuron, and a linear activation function, producing a continuous scalar output corresponding to the estimated number of individuals.

- illustrate example implementation of encoder branch

An example implementation of the encoder branch is illustrated with three convolutional layers, each having 32 filters of size 3×3, followed by batch normalization, ReLU activation, and 2×2 max pooling, with the output of each branch flattened before fusion.

- describe macro-Doppler data processing pipeline

The macro-Doppler data processing pipeline is described as a convolutional encoder that processes the first range-Doppler image, extracting features related to gross human motion, with a receptive field sufficient to capture full-body displacement.

- describe micro-Doppler data processing pipeline

The micro-Doppler data processing pipeline is described as a convolutional encoder that processes the second range-Doppler image, extracting features related to fine biological motion, with high temporal sensitivity to capture oscillatory patterns.

- explain connecting sections

The connecting sections are explained as modules that merge the encoded representations from the two pipelines using concatenation and convolution, enabling the network to learn joint dependencies between macro- and micro-motion signatures.

- describe output section

The output section is described as the final stage of the network, comprising the fusion of the two pipelines into a single embedding vector, followed by a fully-connected layer with a single neuron that outputs the predicted people count.

- illustrate flowchart of people counting method

The flowchart of the people counting method is illustrated as a sequence of steps: acquiring radar data, preprocessing to generate range-Doppler maps, inputting maps into the neural network, generating a raw prediction, applying a smoothing filter, and outputting the final count.

- train neural network

The neural network is trained using a dataset of labeled radar sequences, with the label-aware ranked loss function enforcing geometric ordering of the embedding space according to the true occupancy levels.

- determine people count without ground truth

The system is capable of determining the people count in real time without requiring ground truth labels during inference, relying solely on the learned mapping from radar features to occupancy.

- illustrate flowchart of training method

The flowchart of the training method is illustrated as: collecting multiple radar datasets, preprocessing to generate range-Doppler maps, inputting batches into the network, computing the label-aware ranked loss, backpropagating gradients, and updating network weights.

- obtain multiple sets of training radar measurement datasets

Multiple sets of training radar measurement datasets are obtained by deploying the sensor in diverse environments, including varying numbers of occupants, seating configurations, and ambient conditions, to ensure broad generalization.

- preprocess training datasets

The training datasets are preprocessed using the same pipeline as the inference system, including MTI filtering, 2-D FFT, and Hamming windowing, to ensure consistency between training and deployment.

- input datasets to neural network

The preprocessed datasets are input to the neural network in batches, with each batch containing multiple samples from different occupancy levels to enable effective training with the label-aware ranked loss.

- compare prediction with ground truth

The prediction is compared with the ground truth occupancy label using the label-aware ranked loss function, which computes angular penalties based on the ordinal difference between predicted and true labels.

- explain triplet loss

Triplet loss is explained as a metric learning approach that compares an anchor sample with a positive and a negative sample, minimizing the distance to the positive and maximizing the distance to the negative, but without incorporating label ranking.

- explain label-aware ranked loss

Label-aware ranked loss is explained as a novel loss function that modifies the triplet structure by weighting the angular penalty between samples according to the logarithmic difference in their true labels, thereby enforcing a geometrically ordered embedding space.

- illustrate flowchart of inference method

The flowchart of the inference method is illustrated as: acquiring a new radar frame, preprocessing to generate range-Doppler maps, inputting into the trained neural network, outputting a raw prediction, applying a tracking filter, and outputting the final stabilized count.

- obtain radar measurement dataset

The radar measurement dataset is obtained by activating the radar sensor and recording the reflected signals over a defined time window, with the data stored in a buffer for real-time processing.

- preprocess dataset

The dataset is preprocessed by applying MTI filtering, 2-D FFT, and Hamming windowing to generate the first and second range-Doppler measurement maps.

- input dataset to neural network

The preprocessed maps are input into the trained neural network, which outputs a scalar prediction of the number of individuals present.

- output prediction of people count

The neural network outputs a prediction of the people count as a continuous real-valued number, which is then passed to a postprocessing module.

- apply tracking filter

A tracking filter is applied to the sequence of predictions to enforce temporal consistency, using a Kalman filter that models occupancy as a constant-velocity process.

- illustrate tracking in embedding space

Tracking in the embedding space is illustrated as the movement of a point along a circular manifold, where each occupancy level corresponds to a fixed angular position, and the filter predicts the next position based on prior velocity.

- define regions in embedding space

Regions in the embedding space are defined as fixed angular sectors, each corresponding to a discrete occupancy level, with the boundaries determined by the training process to maximize classification margin.

- explain constant velocity motion model

The constant velocity motion model is explained as the assumption that changes in occupancy occur gradually and predictably, with the embedding vector moving at a constant angular rate between adjacent regions.

- illustrate use cases

Use cases are illustrated as monitoring occupancy in a vehicle cabin for HVAC control, counting passengers in a train carriage for safety compliance, and estimating occupancy in a conference room for energy optimization.

- describe monitoring people entering and exiting doorways

The system is used to monitor people entering and exiting doorways by continuously tracking the occupancy count and detecting changes that correspond to entry or exit events, enabling automated door control and flow analysis.

- describe gathering customer traffic data

The system is used to gather customer traffic data in retail environments by counting the number of individuals in a store over time, enabling occupancy-based staffing and marketing decisions.

- describe counting people for energy savings

The system is used to count people in a building for energy savings by adjusting heating, ventilation, and lighting based on real-time occupancy, reducing energy consumption during low-occupancy periods.