# DEEP LEARNING ALGORITHM FOR IMPROVED SPEECH RECOGNITION USING EEG SIGNALS

## Background and Introduction

Speech recognition systems for individuals with speech disorders such as aphasia, apraxia, and dysarthria often struggle due to the distorted and inconsistent nature of their speech. Traditional methods rely solely on acoustic features, which may not capture the full complexity of these conditions. This paper introduces a deep learning algorithm that leverages non-invasive neural EEG signals to enhance the performance of speech recognition systems for these disorders.

## Describe the Problem

Speech disorders like aphasia, apraxia, and dysarthria result in distorted speech patterns, making it challenging for traditional speech recognition systems to accurately transcribe spoken words. These systems typically rely on acoustic features such as Mel Frequency Cepstral Coefficients (MFCCs), which may not capture the neural underpinnings of speech production. This limitation can lead to poor performance and reduced usability for individuals with these conditions.

## Proposed Solution

The proposed solution involves a deep learning algorithm that integrates EEG signals, which carry neural information about speech perception and production, with traditional acoustic features. By training a regression model to extract acoustic representations from EEG signals, the system can provide additional context to the speech recognition model, leading to improved accuracy and robustness.

## Dataset Description

The dataset used for this study consists of synchronized EEG and speech recordings from individuals with aphasia, apraxia, and dysarthria. The data was collected using 29 EEG sensors placed according to the standard 10-20 system and a mono-channel microphone. The dataset is split into training (70%), validation (10%), and test (20%) sets, ensuring no overlap between the sets.

## Data Collection Process

Data collection involved participants wearing an EEG cap with 29 sensors placed according to the 10-20 system guidelines. Speech was recorded using a mono-channel microphone. Two EMG sensors were also used to monitor muscle activity during articulation. The data was sampled at 1000 Hz for EEG and 16 kHz for speech, with appropriate filters applied to remove noise and artifacts.

## Data Preprocessing

The collected EEG signals were preprocessed by applying a fourth-order IIR bandpass filter (0.1-70 Hz) and a notch filter (60 Hz) to remove power line noise. EMG artifacts were removed using linear regression. Five features—root mean square, zero-crossing rate, moving window average, kurtosis, and power spectral entropy—were extracted from each EEG channel at 100 Hz.

## Feature Extraction

For the speech signals, Mel Frequency Cepstral Coefficients (MFCCs) of dimension 13 were extracted at a sampling frequency of 100 Hz. For the EEG signals, five features per channel were extracted: root mean square, zero-crossing rate, moving window average, kurtosis, and power spectral entropy. These features capture neural information about speech perception and production.

## Dimension Reduction

To reduce the dimensionality of the EEG feature space, kernel principal component analysis (KPCA) with a polynomial kernel of degree 3 was applied. The cumulative explained variance plot was used to determine the optimal number of components, reducing the 145-dimensional feature space to 10 dimensions.

## Model Architecture

The proposed model consists of two main components: a regression model and a speech recognition model. The regression model is trained to extract acoustic representations from EEG signals, which are then concatenated with MFCC features. The combined features are fed into a GRU-based speech recognition model for text decoding.

## Training Process

The regression model was trained using the training set to predict acoustic representations from EEG signals. The loss function used was mean squared error (MSE). The speech recognition model was trained using the combined features of MFCCs and high-frequency EEG representations. Early stopping and learning rate decay were used to prevent overfitting.

## Evaluation Metrics

The performance of the isolated speech recognition model was evaluated using accuracy, F1-score, precision, and recall. For continuous speech recognition, word error rate (WER) was used as the primary metric. Baseline results were obtained by training and testing the models using only MFCC features for comparison.

## Results

The proposed method significantly outperformed the baseline for isolated speech recognition, with improvements in accuracy, F1-score, precision, and recall. For continuous speech recognition, the model achieved a lower WER compared to the baseline, demonstrating its effectiveness in handling longer sequences of speech.

## Discussion

The results indicate that integrating EEG signals with acoustic features can significantly enhance speech recognition performance for individuals with aphasia, apraxia, and dysarthria. The use of high-frequency EEG representations was particularly effective, suggesting that these neural signals carry important information about speech production. Future work will focus on real-time implementation and validation on larger datasets.

## Conclusion

This paper presents a deep learning algorithm that leverages non-invasive EEG signals to improve the performance of speech recognition systems for individuals with aphasia, apraxia, and dysarthria. The proposed method outperformed baseline results, demonstrating its potential to enhance communication for these users. Future work will address real-time implementation and further validation on larger datasets.

## Limitations

One limitation is the potential latency in real-time deployment due to the additional step of extracting acoustic representations from EEG signals. This latency depends on input sequence length, model size, and computational resources. Additionally, the current results are based on offline analysis, and real-time performance needs to be evaluated.

## Future Work

Future work will focus on validating the proposed method on larger datasets and developing tools to improve real-time performance. The aphasia, apraxia, and dysarthria speech-EEG dataset will be released to the public to facilitate further research in this area.