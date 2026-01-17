# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of automatic speech recognition (ASR) systems, particularly for individuals with speech disorders such as aphasia, apraxia, and dysarthria. The invention specifically addresses the improvement of ASR performance by integrating non-invasive neural signals, such as electroencephalography (EEG), with traditional acoustic features.

## BACKGROUND

Automatic speech recognition (ASR) systems have become integral components of various modern technologies, including virtual voice assistants like Apple's Siri, Amazon's Alexa, and Samsung's Bixby. These systems are designed to convert spoken language into written text, facilitating user interaction with devices and applications. However, the performance of ASR systems is significantly compromised when dealing with incomplete, distorted, or broken speech, which is common among individuals with speech disorders such as aphasia, apraxia, and dysarthria.

Aphasia is a condition characterized by impaired language comprehension and production, often resulting from brain damage due to strokes or head injuries. Apraxia is a motor planning disorder affecting speech production, while dysarthria is a motor speech disorder caused by neurological damage. Individuals with these conditions produce speech that is often distorted and incomplete, making it challenging for conventional ASR systems to accurately transcribe their speech.

Previous research has explored the use of electrophysiological signals, such as EEG and electrocorticography (ECoG), to enhance speech recognition. EEG, being non-invasive, offers a high temporal resolution, making it suitable for capturing rapid speech-related neural activities. However, the poor spatial resolution and signal-to-noise ratio of EEG signals pose challenges. Despite these limitations, several studies have demonstrated the potential of EEG features to improve ASR performance, especially for isolated speech recognition tasks.

Despite these advancements, there remains a significant gap in the ability of ASR systems to effectively handle the speech of individuals with aphasia, apraxia, and dysarthria. The high word error rates (WER) and phoneme error rates (PER) reported in existing literature highlight the need for innovative solutions to enhance the robustness and accuracy of ASR systems for these user groups.

## SUMMARY

The present invention provides a deep learning-based algorithm to improve the performance of speech recognition systems for individuals with aphasia, apraxia, and dysarthria by integrating EEG features with traditional acoustic features. The key aspects of the invention include:

1. **Data Collection**: A large-scale aphasia, apraxia, and dysarthria Speech-EEG dataset is collected, which will be made publicly available to advance research in this area.
2. **Feature Extraction**: Acoustic features (e.g., Mel frequency cepstral coefficients, MFCC) and EEG features (e.g., root mean square, zero-crossing rate, moving window average, kurtosis, and power spectral entropy) are extracted from the collected data.
3. **Dimension Reduction**: Non-linear dimension reduction techniques, such as kernel principal component analysis (KPCA), are applied to reduce the dimensionality of the EEG feature space.
4. **Regression Model**: A regression model, based on gated recurrent units (GRUs), is trained to predict acoustic features from EEG features.
5. **Speech Recognition Models**: Isolated and continuous speech recognition models, also based on GRUs, are trained using the concatenated acoustic and EEG features to improve recognition performance.

The invention achieves a significant improvement in the test-time decoding performance of aphasia, apraxia, and dysarthria speech recognition, with a performance enhancement of over 50% for isolated speech recognition and a slight improvement for continuous speech recognition. This innovation has the potential to enhance the accessibility and effectiveness of speech recognition technology for individuals with speech disorders, leading to better speech prosthetics and speech therapy tools.

## DETAILED DESCRIPTION

### Data Collection

The invention involves collecting a large-scale dataset of aphasia, apraxia, and dysarthria speech and corresponding EEG signals. The data collection process includes two tasks performed by subjects during speech therapy sessions:

1. **Reading Task**: Subjects read out loud English sentences displayed on a computer screen, and their EEG, electromyography (EMG), and speech signals are recorded simultaneously.
2. **Listening Task**: Subjects listen to recorded audio of English sentences and then repeat them aloud, with their EEG, EMG, and speech signals recorded simultaneously.

The dataset comprises 8854 data samples from 9 subjects, with a vocabulary of 57 unique daily used common English sentences. The EEG signals are recorded using 29 wet EEG sensors placed according to the standard 10-20 EEG sensor placement guidelines. The EMG signals are recorded using two sensors placed to monitor EMG artifacts during articulation. The speech signals are recorded using a mono-channel microphone. The dataset is split into 70% training, 10% validation, and 20% test sets, ensuring no overlap between the sets.

### Feature Extraction

#### EEG Feature Extraction

The recorded EEG signals are preprocessed to remove noise and artifacts. The preprocessing steps include:
- Sampling at 1000 Hz.
- Applying a fourth-order IIR bandpass filter with cut-off frequencies of 0.1 Hz and 70 Hz.
- Using a notch filter with a cut-off frequency of 60 Hz to remove power line noise.
- Removing EMG artifacts using linear regression.

Five features are extracted per EEG channel:
- Root Mean Square (RMS)
- Zero-Crossing Rate (ZCR)
- Moving Window Average (MWA)
- Kurtosis
- Power Spectral Entropy (PSE)

These features are extracted at a sampling frequency of 100 Hz per channel.

#### Speech Feature Extraction

The speech signals are sampled at 16 kHz. Mel frequency cepstral coefficients (MFCC) of dimension 13 are extracted as features for the speech signal. The MFCC features are also extracted at a sampling frequency of 100 Hz to align with the EEG feature extraction.

### Dimension Reduction

The high-dimensional EEG feature space is reduced using kernel principal component analysis (KPCA) with a polynomial kernel of degree 3. The optimal dimension is determined by plotting the cumulative explained variance against the number of components. The EEG feature space is reduced from 145 dimensions (five features per each of the 29 channels) to a final dimension of 10. Before applying KPCA, the EEG features are normalized by removing the mean and scaling to unit variance.

### Regression Model

A regression model is trained to predict acoustic features (MFCC) from EEG features. The model architecture consists of:
- A single layer of gated recurrent units (GRUs) with 128 hidden units.
- A time-distributed dense layer with 13 hidden units and a linear activation function.

The regression model is trained for 70 epochs using mean squared error (MSE) as the loss function and the Adam optimizer. The model generates additional acoustic features that are concatenated with the original acoustic features to form the input for the speech recognition models.

### Speech Recognition Models

#### Isolated Speech Recognition Model

The isolated speech recognition model architecture includes:
- A single layer of GRUs with 512 hidden units.
- A dropout regularization layer with a drop-out rate of 0.2.
- A dense layer with 57 hidden units and a linear activation function.
- A softmax activation function to obtain label prediction probabilities.

The model is trained for 10 epochs with a batch size of 50, using categorical cross-entropy as the loss function and the Adam optimizer. Early stopping is employed to prevent overfitting.

#### Continuous Speech Recognition Model

The continuous speech recognition model architecture consists of:
- A GRU layer with 512 hidden units acting as an encoder.
- A decoder composed of a dense layer with a linear activation function and a softmax activation function.
- An external 4-gram language model and a CTC beam search decoder are used during inference.

The model is trained for 100 epochs with a batch size of 50 to optimize the connectionist temporal classification (CTC) loss function, using the Adam optimizer. The model predicts a character at every time-step.

### Results and Discussion

The performance of the isolated and continuous speech recognition models is evaluated using various metrics:
- **Isolated Speech Recognition**: Percentage accuracy, F1-score, precision, and recall.
- **Continuous Speech Recognition**: Word error rate (WER).

Baseline results are obtained by training and testing the models using only acoustic (MFCC) features. The addition of EEG features significantly improves the performance of both models. Specifically:
- For isolated speech recognition, the proposed method outperforms the baseline by more than 50% in terms of accuracy, F1-score, precision, and recall.
- For continuous speech recognition, the proposed method reduces the WER compared to the baseline, with a statistically significant improvement (p-value = 0.0000213).

### Conclusion, Limitation, and Future Work

The present invention demonstrates a significant improvement in the performance of ASR systems for individuals with aphasia, apraxia, and dysarthria by integrating EEG features with traditional acoustic features. The proposed deep learning-based algorithm outperforms the baseline results for both isolated and continuous speech recognition tasks. The release of the aphasia, apraxia, and dysarthria Speech-EEG dataset will facilitate further research in this area.

However, the proposed algorithm has limitations, particularly regarding real-time deployment due to latency issues. Future work will focus on optimizing the algorithm for real-time performance, validating the results on larger datasets, and exploring additional methods to enhance the robustness and accuracy of the ASR system.