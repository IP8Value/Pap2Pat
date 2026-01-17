# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to the field of multimedia content analysis and summarization, particularly focusing on the detection and representation of events in unscripted multimedia content. More specifically, the invention provides a content-adaptive analysis and representation framework for audio event discovery from unscripted multimedia, such as sports and surveillance videos.

## BACKGROUND OF THE INVENTION

The goals of multimedia content summarization are twofold: to capture the essence of the content in a succinct manner and to provide a top-down access into the content for browsing. Signal processing and statistical learning tools are used to generate a suitable representation for the content, from which summaries can be created. For scripted content, such as news, movies, and dramas, a representation that captures the sequence of semantic units has been shown to be useful. Past work on summarization of scripted content has primarily focused on creating a table of contents (ToC) representation, where summaries are constructed using abstractions (e.g., skims, keyframes) from each detected semantic unit.

For unscripted content, such as sports and surveillance videos, interesting events occur sparsely in a background of usual events. Therefore, past work on summarization of unscripted content has mainly focused on detecting these specific events. Various approaches have been proposed for different types of unscripted content:

1. **Sports Video**:
   - Detection of domain-specific events and objects using audio-visual cues.
   - Unsupervised extraction of play-break segments from sports video.

2. **Surveillance Video**:
   - Detection of "unusual" events using object segmentation and tracking from video.

These approaches typically rely on domain-specific knowledge and manual selection of key audio-visual objects. However, a more desirable framework is one that is content-adaptive and genre-independent, postponing content-specific processing to as late a stage as possible.

## SUMMARY OF THE INVENTION

The present invention addresses the challenges of event discovery in unscripted multimedia by proposing a content-adaptive analysis and representation framework. The framework is based on the observation that "interesting" events are outliers in a background of usual events. The invention uses audio features to perform an inlier/outlier-based temporal segmentation of the content, which can be applied to various genres of unscripted multimedia.

Key aspects of the invention include:

1. **Feature Extraction**: Low-level features are extracted from the input content to generate a time series from which events are to be discovered. For example, Mel-frequency cepstral coefficients (MFCC) can be extracted from the audio stream.

2. **Classification/Clustering**: The low-level features are classified using supervised models for classes that span the whole domain, generating a discrete time series of mid-level classification/cluster labels.

3. **Detection of Outlier Subsequences**: Outlier subsequences are detected from the time series of low-level features or mid-level classification labels. This is achieved by eigenvector analysis of the affinity matrix constructed from statistical models estimated from the subsequences of the time series.

4. **Ranking Outlier Subsequences**: The detected outliers are ranked based on their statistical deviation from the inliers, allowing for the generation of summaries of desired length.

5. **Summarization**: Domain knowledge is incorporated to prune and modify the ranks of the detected outliers, ensuring that only the "interesting" events are included in the summary.

The invention is particularly useful for sports and surveillance content, where it can effectively detect and rank unusual events without requiring extensive domain-specific knowledge. The framework is also computationally efficient and can be adapted to various types of unscripted multimedia.

## BRIEF DESCRIPTION OF THE DRAWINGS

- **Figure 1**: Illustration of a table of contents (ToC) representation for scripted content.
- **Figure 2**: Hierarchical representation for unscripted content based on domain-specific key audio-visual objects.
- **Figure 3**: Inlier/outlier-based representation for unscripted content.
- **Figure 4**: Detailed illustration of the clustering and outlier detection blocks in the proposed framework.
- **Figure 5**: Framework for generating synthetic time series data.
- **Figure 6**: Results of normalized cut for synthetic time series data.
- **Figure 7**: Results of alphabet-constrained K-means clustering for synthetic time series data.
- **Figure 8**: Results of dendrogram-based clustering for synthetic time series data.
- **Figure 9**: Results of modified normalized cut (foreground cut) for synthetic time series data.
- **Figure 10**: Results of normalized cut for synthetic time series data with multiple foreground processes.
- **Figure 11**: Results of hierarchical clustering using normalized cut and foreground cut for synthetic time series data.
- **Figure 12**: PDF estimates for different context models and context sizes.
- **Figure 13**: Affinity matrix for a 3-hour-long golf game.
- **Figure 14**: Cluster indicator vector and affinity matrix for sports audio content.
- **Figure 15**: Temporal segmentation of a 20-minute Japanese baseball clip.
- **Figure 16**: Systematic choice of key audio classes using the proposed framework.
- **Figure 17**: Precision of highlights extraction using the discovered highlight class.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENT

### Overview of Invention

The present invention provides a content-adaptive analysis and representation framework for audio event discovery from unscripted multimedia. The framework is designed to detect and rank unusual events in a background of usual events, making it particularly useful for sports and surveillance content. The invention operates in several stages, each contributing to the overall goal of identifying and summarizing "interesting" events.

#### Feature Extraction

The first stage involves extracting low-level features from the input content. For audio content, Mel-frequency cepstral coefficients (MFCC) are commonly used. These features are extracted from the audio stream to generate a time series, which serves as the input for subsequent stages of the framework.

#### Classification/Clustering

In the second stage, the low-level features are classified using supervised models. For example, Gaussian mixture models (GMMs) can be used to classify every frame of audio into one of several predefined classes, such as applause, cheering, music, speech, and speech with music. This classification generates a discrete time series of mid-level classification/cluster labels, which can be used to detect events at different scales.

#### Detection of Outlier Subsequences

The third stage focuses on detecting outlier subsequences from the time series of low-level features or mid-level classification labels. This is achieved by eigenvector analysis of the affinity matrix constructed from statistical models estimated from the subsequences of the time series. The affinity matrix represents the similarity between different subsequences, and eigenvector analysis helps identify distinct clusters and outliers.

#### Ranking Outlier Subsequences

Once the outliers are detected, they are ranked based on their statistical deviation from the inliers. This ranking allows for the generation of summaries of desired length, as the most unusual events are given higher priority.

#### Summarization

The final stage involves incorporating domain knowledge to prune and modify the ranks of the detected outliers. This ensures that only the "interesting" events are included in the summary. For example, in sports content, a supervised detector for excited speech can be used to filter out non-highlight events.

### Detailed Implementation

#### Feature Extraction

Low-level features, such as MFCC, are extracted from the audio stream. The extraction process involves the following steps:

1. **Preprocessing**: The audio signal is preprocessed to remove noise and normalize the volume.
2. **Frame Extraction**: The audio signal is divided into overlapping frames, typically 20-40 ms in duration.
3. **Spectral Analysis**: For each frame, the power spectrum is computed using a Fourier transform.
4. **Mel-Frequency Filtering**: The power spectrum is filtered using a set of triangular filters spaced linearly on the mel scale.
5. **Logarithmic Transformation**: The output of the mel filters is transformed using a logarithmic function.
6. **Discrete Cosine Transform (DCT)**: The logarithmic values are transformed using a DCT to produce the MFCC coefficients.

#### Classification/Clustering

The extracted MFCC coefficients are classified using supervised models, such as GMMs. The classification process involves the following steps:

1. **Model Training**: GMMs are trained using a labeled dataset of audio clips. Each class (e.g., applause, cheering, music, speech, speech with music) is represented by a GMM with a specified number of mixture components.
2. **Feature Extraction**: MFCC coefficients are extracted from the test audio.
3. **Classification**: For each frame of the test audio, the likelihood of the observed features under each GMM is computed, and the frame is assigned to the class with the highest likelihood.

#### Detection of Outlier Subsequences

The detection of outlier subsequences is based on eigenvector analysis of the affinity matrix. The process involves the following steps:

1. **Context Modeling**: The input time series is sampled on a uniform grid, and a statistical model (e.g., GMM) is estimated for each context.
2. **Affinity Matrix Construction**: The affinity matrix is constructed using a distance metric (e.g., likelihood distance) between the context models. Each element of the affinity matrix represents the similarity between two contexts.
3. **Eigenvector Analysis**: The second generalized eigenvector of the affinity matrix is computed, which serves as the cluster indicator vector.
4. **Thresholding**: A threshold is applied to the eigenvector values to identify inliers and outliers. The optimal threshold is selected based on the normalized cut value.

#### Ranking Outlier Subsequences

The detected outliers are ranked based on their deviation from the inliers. The ranking process involves the following steps:

1. **Confidence Measure**: The confidence on the detected outliers is quantified using the PDF of the distance metric. The confidence measure is computed for each outlier context.
2. **Ranking**: The outliers are ranked based on their confidence measures, with higher confidence indicating a more unusual event.

#### Summarization

The final stage involves incorporating domain knowledge to prune and modify the ranks of the detected outliers. The process involves the following steps:

1. **Supervised Detection**: A supervised detector (e.g., excited speech detector) is used to filter out non-highlight events.
2. **Rank Modification**: The ranks of the detected outliers are modified based on the output of the supervised detector.
3. **Summary Generation**: The top-ranked outliers are selected to generate a summary of the desired length.

### Experimental Results

The proposed framework has been tested on a variety of sports and surveillance audio content. The results demonstrate the effectiveness of the framework in detecting and ranking unusual events.

#### Sports Audio Content

- **Baseball**: The framework was tested on 4 hours of baseball audio from 5 different games. The detected outliers were classified by hand, and the results showed that the most indicative highlight events were excited speech with cheering and cheering alone.
- **Soccer**: The framework was tested on 6 hours of soccer audio from 7 different games. The results were similar to those obtained for baseball, with excited speech with cheering and cheering alone being the most indicative of highlight events.
- **Golf**: The framework was tested on 90 minutes of a golf game. The detected outliers included applause segments, silences during the commentator's speech, and interviews with new speakers.

#### Surveillance Audio Content

- **Elevator Surveillance**: The framework was tested on 1.5 hours of elevator surveillance data. The detected outliers included banging sounds and excited speech, which were correlated with suspicious activities.
- **Traffic Intersection**: The framework was tested on 2.5 hours of traffic intersection video. The detected outliers included ambulance sirens and a car crash, which were unusual events in the context of normal traffic.

### Systematic Choice of Key Audio Classes

The proposed framework not only detects and ranks unusual events but also helps in the systematic choice of key audio classes. By examining the detected outliers, consistent patterns in the data can be identified, and supervised statistical learning models can be built. This is particularly useful for less understood domains, such as surveillance, where the audio classes cannot be anticipated.

### Conclusion

The present invention provides a content-adaptive analysis and representation framework for audio event discovery from unscripted multimedia. The framework is based on the observation that "interesting" events are outliers in a background of usual events. The invention uses audio features to perform an inlier/outlier-based temporal segmentation of the content, which can be applied to various genres of unscripted multimedia. The framework is computationally efficient and can be adapted to different types of content, making it a valuable tool for multimedia content summarization.