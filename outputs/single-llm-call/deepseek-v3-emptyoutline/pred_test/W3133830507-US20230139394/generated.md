# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates to the field of speech recognition systems, particularly those designed to process speech from individuals with speech disorders such as aphasia, apraxia, and dysarthria. More specifically, the invention pertains to a deep learning-based algorithm that enhances the accuracy of automatic speech recognition (ASR) systems by incorporating neural signals, specifically electroencephalography (EEG) data, alongside traditional acoustic features. The invention addresses the limitations of conventional ASR systems, which struggle to accurately interpret distorted, incomplete, or broken speech patterns characteristic of individuals with speech impairments. By integrating EEG-derived acoustic representations into the ASR training and inference pipeline, the invention significantly improves recognition performance, thereby enabling broader accessibility to voice-assisted technologies for individuals with speech disorders.  

## BACKGROUND  

Automatic speech recognition (ASR) systems are widely deployed in modern voice assistants such as Apple's Siri, Amazon's Alexa, and Samsung's Bixby. However, these systems are predominantly trained on uniform speech from individuals without speech disorders, leading to degraded performance when processing speech from individuals with conditions such as aphasia, apraxia, or dysarthria. Aphasia, often caused by stroke or head trauma, disrupts language comprehension and formulation. Apraxia involves impaired motor planning for speech production, while dysarthria arises from neurological damage affecting motor speech control. These disorders result in speech that is fragmented, distorted, or otherwise difficult for conventional ASR systems to interpret accurately.  

Existing research has explored the use of neural signals, such as EEG and electrocorticography (ECoG), to capture speech-related neural activity. While EEG offers non-invasive, high-temporal-resolution measurements, its spatial resolution and signal-to-noise ratio are inferior to invasive methods like ECoG. Prior studies have demonstrated that EEG signals contain valuable information about speech perception and production, but their application has been limited to individuals without speech disorders. Efforts to improve ASR performance for disordered speech have yielded poor results, with word error rates (WER) as high as 97.5% for aphasia speech and phoneme error rates (PER) between 75% and 89% for severe cases.  

The present invention builds upon these findings by introducing a novel deep learning algorithm that leverages EEG features to augment acoustic models. The algorithm trains a regression model to predict acoustic features (e.g., Mel-frequency cepstral coefficients, or MFCCs) from EEG signals, which are then combined with traditional speech features to enhance ASR performance. This approach has demonstrated a more than 50% improvement in isolated speech recognition accuracy and marginal gains in continuous speech recognition for disordered speech.  

## SUMMARY  

The invention provides a deep learning-based algorithm for improving speech recognition performance in individuals with aphasia, apraxia, or dysarthria by integrating EEG-derived acoustic representations with conventional speech features. The algorithm comprises the following key components:  

1. **EEG Feature Extraction and Processing**: EEG signals are recorded synchronously with speech, filtered to remove noise and artifacts (e.g., electromyography, or EMG), and processed to extract relevant neural features such as root mean square, zero-crossing rate, moving window average, kurtosis, and power spectral entropy.  

2. **Dimensionality Reduction**: Non-linear techniques such as kernel principal component analysis (KPCA) are applied to reduce the high-dimensional EEG feature space, improving computational efficiency without sacrificing discriminative power.  

3. **Regression Model Training**: A gated recurrent unit (GRU)-based regression model is trained to predict acoustic features (e.g., MFCCs) from EEG signals, generating auxiliary representations that complement traditional speech features.  

4. **Speech Recognition Model Integration**: The EEG-derived acoustic representations are concatenated with standard acoustic features and fed into an ASR model, which may be configured for isolated or continuous speech recognition. The ASR model employs deep learning architectures such as GRUs or connectionist temporal classification (CTC) frameworks, optimized for disordered speech.  

Experimental results demonstrate that the proposed algorithm significantly outperforms baseline ASR systems, achieving over 50% improvement in isolated speech recognition accuracy and reducing WER in continuous speech recognition tasks. The invention also introduces a large-scale dataset of synchronized speech-EEG recordings from individuals with speech disorders, facilitating further research in this domain.  

## DETAILED DESCRIPTION  

The invention is further described with reference to the following detailed embodiments, which illustrate the implementation and advantages of the proposed algorithm.  

### EEG and Speech Data Collection  

Data collection involved recording synchronized EEG, EMG, and speech signals from individuals undergoing speech therapy. Participants performed two tasks: (1) reading English sentences displayed on a screen, and (2) listening to and repeating recorded sentences. A total of 8,854 data samples were collected from nine subjects, covering 57 unique English sentences. EEG signals were captured using 29 wet electrodes arranged according to the 10-20 system, with additional EMG sensors to monitor articulatory artifacts. Speech was recorded via a mono-channel microphone at 16 kHz, while EEG was sampled at 1 kHz.  

### Signal Processing and Feature Extraction  

EEG signals were bandpass-filtered (0.1–70 Hz) and notch-filtered (60 Hz) to remove noise. EMG artifacts were regressed out using linear regression. Five features per EEG channel were extracted: root mean square, zero-crossing rate, moving window average, kurtosis, and power spectral entropy. These features were sampled at 100 Hz to align with the speech feature extraction rate. Speech signals were processed to extract 13-dimensional MFCCs, also sampled at 100 Hz.  

### Dimensionality Reduction and Model Training  

The high-dimensional EEG feature space (145 dimensions) was reduced to 10 dimensions using KPCA with a polynomial kernel. A GRU-based regression model (128 hidden units) was trained to predict MFCCs from EEG features, optimized via mean squared error over 70 epochs. The ASR model for isolated speech recognition comprised a GRU layer (512 units), dropout regularization (rate = 0.2), and a dense layer (57 units, matching the vocabulary size), trained for 10 epochs with categorical cross-entropy loss. For continuous speech recognition, a CTC-based encoder-decoder architecture was employed, trained for 100 epochs with a 4-gram language model for beam search decoding.  

### Performance Evaluation  

The algorithm was evaluated using accuracy, F1-score, precision, recall, and WER metrics. Results demonstrated that integrating EEG-derived features with MFCCs reduced WER by 4–7% in continuous speech recognition and improved isolated speech recognition accuracy by over 50%. Statistical analysis confirmed the significance of these improvements (p = 0.0000213).  

### Applications and Future Directions  

The invention enables the development of speech prosthetics and enhanced therapy tools for individuals with speech disorders. Future work will focus on real-time deployment, latency optimization, and expansion to larger datasets. The public release of the collected speech-EEG dataset will further advance research in this field.  

This concludes the detailed description of the invention. The claims section will further define the scope of the patented technology.