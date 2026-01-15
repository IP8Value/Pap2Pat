## FIELD OF THE INVENTION

- define field of invention

The present invention relates to the field of multimedia content analysis and automated summarization, specifically to systems and methods for detecting, characterizing, and ranking unusual events in unscripted audiovisual media without requiring domain-specific prior knowledge. The invention provides a content-adaptive framework for identifying statistically anomalous temporal segments within continuous streams of multimedia data—such as sports broadcasts, surveillance footage, and other unstructured audiovisual recordings—by modeling the underlying statistical regularities of the background process and isolating deviations therefrom. The disclosed method operates independently of predefined event templates, semantic ontologies, or manually curated feature detectors, enabling generalized application across diverse media genres and environments. The invention is particularly suited for applications in automated video summarization, real-time anomaly detection in security systems, intelligent content indexing, and adaptive media browsing interfaces where the nature of significant events is unknown a priori or varies dynamically across contexts.

## BACKGROUND OF THE INVENTION

- motivate multimedia event detection

The increasing volume of unstructured multimedia content generated daily necessitates automated methods for extracting meaningful summaries and identifying salient events without human intervention. Traditional approaches to multimedia analysis rely heavily on supervised learning techniques that require extensive manual labeling of training data and domain-specific feature engineering. These methods are inherently limited in scalability and generalizability, as each new content type—whether a new sport, a different surveillance environment, or an emerging media format—demands the redefinition of event classes, retraining of classifiers, and recalibration of detection thresholds. This dependency on prior knowledge renders existing systems brittle, costly to deploy, and incapable of adapting to novel or evolving contexts. The need for a more flexible, self-contained, and statistically grounded approach to event detection is therefore urgent, particularly in domains where labeled data is scarce, events are unpredictable, or the definition of “interesting” is context-dependent and subjective.

- summarize prior art for news videos

Prior methods for news video summarization have primarily focused on detecting structural boundaries such as story transitions using closed-caption analysis, speaker diarization, or facial recognition to identify recurring anchors or reporters. These techniques exploit the highly scripted and predictable nature of broadcast news, where segment boundaries are often marked by visual cues such as studio changes, logos, or fixed camera angles. While effective within this constrained domain, such approaches fail to generalize to unscripted content where no such consistent structural markers exist. The reliance on semantic metadata or manually annotated transcripts further limits applicability to languages, formats, or platforms where such auxiliary data is unavailable or unreliable.

- summarize prior art for situation comedies

In the domain of situation comedies, prior work has employed scene segmentation based on mosaic representations of background environments and audio-visual cues to identify recurring cast members and settings. These methods assume a fixed set of locations and characters, leveraging repetition and familiarity to infer narrative structure. However, this assumption breaks down in unscripted or non-studio content where environmental consistency is absent and character appearances are irregular. The inability to detect events outside of predefined semantic categories renders these methods unsuitable for dynamic or open-ended media environments.

- summarize prior art for sports video summarization

Sports video summarization has traditionally relied on domain-specific audio-visual markers such as crowd noise bursts, goal horn detection, slow-motion replays, or scoreboard appearances. These markers are manually selected based on expert intuition and encoded into rule-based or supervised classifiers. While successful within specific sports such as soccer or basketball, these systems are not transferable across sports due to differences in event triggers, audience behavior, and broadcast conventions. Moreover, such approaches are incapable of detecting novel or unexpected highlights—such as an unusual player gesture, an unexpected referee decision, or an unconventional crowd reaction—that fall outside the predefined set of known events.

- summarize prior art for movie content

In movie content analysis, prior methods have focused on detecting syntactic structures like dialogues, shot transitions, or action sequences using low-level visual and auditory features. Techniques such as shot boundary detection, face clustering, and speech-to-text alignment have been employed to construct hierarchical representations of narrative structure. However, these methods are predicated on the assumption that movies follow conventional cinematic grammar and that meaningful units can be derived from recurring patterns such as two-speaker exchanges or explosion sequences. This assumption does not hold for experimental, documentary, or improvised content, where narrative structure is non-linear and event semantics are ambiguous.

- summarize prior art for surveillance content

Surveillance video analysis has primarily focused on object tracking, motion detection, and anomaly classification using background subtraction and deep learning models trained on labeled instances of suspicious behavior. These methods require extensive datasets of known anomalies—such as falls, loitering, or theft—to train discriminative models. In practice, the rarity and diversity of real-world anomalies make it infeasible to collect sufficient labeled examples, and the resulting systems suffer from high false positive rates and poor generalization to unseen scenarios. Furthermore, the reliance on visual features alone ignores the rich contextual information contained in audio streams, which often provide critical cues to unusual events such as screams, crashes, or alarms.

- list prior patents and applications

U.S. Patent No. 6,507,614 describes a system for detecting sports highlights using audio energy thresholds and visual motion analysis. U.S. Patent No. 7,228,025 discloses a method for summarizing news videos using closed-caption text segmentation. U.S. Patent No. 8,116,582 outlines a surveillance anomaly detection system based on optical flow and object trajectory modeling. U.S. Patent Application No. 2015/0178572 presents a deep learning framework for classifying video events using convolutional neural networks trained on labeled datasets. U.S. Patent No. 9,430,609 details a method for generating video summaries using hierarchical clustering of keyframes. Each of these systems requires domain-specific training data, predefined event categories, or manually engineered features, and none provide a mechanism for discovering novel or unexpected events without prior knowledge.

- highlight limitations of prior art

The principal limitation of prior art lies in its dependence on supervised learning and domain-specific heuristics. All existing methods require either labeled training data, manually defined event templates, or fixed feature sets that are tailored to a specific genre or environment. This renders them incapable of adapting to new content types, discovering previously unknown events, or operating in open-ended scenarios where the definition of “interesting” is not predetermined. Furthermore, these methods often conflate statistical deviation with semantic relevance, leading to the misclassification of rare but unimportant events as highlights or the failure to detect semantically significant events that do not conform to pre-established patterns.

- identify need for generalized event detection

There exists a critical need for a generalized, unsupervised framework capable of detecting unusual events in any unscripted multimedia stream without requiring prior knowledge of the content domain, event semantics, or feature characteristics. Such a system must be able to autonomously model the statistical baseline of the background process, identify deviations from that baseline, rank those deviations according to their statistical significance, and produce a temporally segmented summary that reflects the inherent structure of the content. This capability is essential for scalable, real-world deployment in diverse and evolving environments—from sports broadcasting to public safety monitoring—where manual annotation is impractical and adaptability is paramount.

- outline desired requirements for multimedia summarization

A viable solution must satisfy several key requirements: (1) it must operate without any domain-specific training data or labeled examples; (2) it must be capable of discovering novel and unexpected events that deviate from the statistical norm; (3) it must provide a ranked ordering of detected events based on their degree of deviation from the background; (4) it must be computationally feasible for real-time or near-real-time processing; (5) it must be applicable across multiple media modalities, including audio, video, and their fusion; and (6) it must enable the systematic extraction of discriminative features that can later be used to build supervised models for improved precision, without compromising the initial unsupervised discovery process.

## SUMMARY OF THE INVENTION

- introduce content-adaptive event detection

The invention introduces a content-adaptive framework for event detection in unscripted multimedia that autonomously identifies unusual events by modeling the statistical regularities of the background process and isolating deviations from those regularities. Unlike prior methods that rely on predefined event categories or supervised classifiers, the invention operates in a completely unsupervised manner, learning the structure of the content from the data itself. By treating the sequence of low-level audio-visual features as a time series and analyzing the statistical consistency of local subsequences, the system identifies segments that are statistically anomalous relative to the dominant background process. This approach eliminates the need for domain-specific knowledge, enabling the detection of previously unknown or unexpected events across a wide range of media types.

- outline unified learning framework

The invention provides a unified learning framework that integrates feature extraction, context modeling, affinity-based clustering, and spectral segmentation into a single, coherent pipeline. The framework begins by sampling the input time series into overlapping context windows, each of which is used to estimate a statistical model representing the local behavior of the signal. These models are then compared using a commutative distance metric to construct an affinity matrix that encodes the pairwise similarity between contexts. The second generalized eigenvector of this matrix is computed to partition the time series into clusters corresponding to inlier (usual) and outlier (unusual) segments. The deviation of each outlier from the inlier cluster is quantified using a confidence measure derived from the probability density function of the distance metric, allowing for the ranking of detected events by statistical significance. This framework is domain-agnostic, scalable, and capable of both unsupervised discovery and subsequent supervised refinement.

## BRIEF DESCRIPTION OF THE DRAWINGS

- describe figures

Figure 1 illustrates the overall architecture of the content-adaptive event detection framework, showing the sequential flow from feature extraction to summary generation. Figure 2 depicts the construction of the affinity matrix from context models, highlighting the computation of pairwise distances and the transformation into similarity weights. Figure 3 presents the spectral clustering process, showing the derivation of the second generalized eigenvector and its application to temporal segmentation. Figure 4 displays the kernel density estimation of the distance metric under varying context window sizes, demonstrating the relationship between sample size and statistical confidence. Figure 5 shows the hierarchical clustering procedure applied to multiple background processes, illustrating the recursive use of normalized cut and foreground cut for complex scenarios. Figure 6 presents the output of the system on a sports video, with overlaid temporal segments indicating detected outliers and their ranked order. Figure 7 illustrates the application of the framework to surveillance audio, showing the discovery of previously unknown anomaly classes such as banging and excited speech. Figure 8 compares the precision-recall performance of the proposed method against prior supervised approaches across multiple genres.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENT

- motivate unusual event detection

Unusual events in unscripted multimedia occur infrequently and unpredictably against a backdrop of repetitive, predictable patterns. In sports, these may include unexpected crowd reactions, controversial referee decisions, or rare player actions. In surveillance, they may involve sudden noises, unexplained movements, or abnormal interactions. These events are not defined by semantic labels but by their statistical rarity. The invention is motivated by the insight that the most salient events are those that deviate most significantly from the statistical norm of the surrounding context, regardless of their semantic interpretation. By focusing on statistical deviation rather than semantic classification, the invention enables the discovery of events that are novel, unexpected, or previously undocumented.

- introduce background and foreground events

The invention distinguishes between background events, which constitute the dominant, repetitive, and statistically consistent portion of the media stream, and foreground events, which are rare, transient, and statistically anomalous. The background represents the “usual” process, characterized by low variance and high temporal stability, while the foreground represents the “unusual” process, characterized by high deviation and low recurrence. The system does not assume prior knowledge of the number, nature, or timing of foreground events. Instead, it learns the statistical properties of the background from the data and identifies foreground events as those that fall outside the expected distribution of the background.

- define usual and unusual events

Usual events are defined as subsequences of the media stream whose statistical models are consistent with the majority of other subsequences, forming a dense cluster in the model space. Unusual events are defined as those whose statistical models lie far from this cluster, exhibiting significant deviation in terms of the chosen distance metric. The distinction is not based on semantic content but on statistical dissimilarity. An event may be unusual even if it is semantically mundane, and a semantically dramatic event may be considered usual if it occurs frequently. The system’s output is therefore a ranked list of temporally localized segments ordered by their statistical deviation, not by their semantic importance.

- formulate problem of detecting unusual events

The problem of detecting unusual events is formulated as one of temporal segmentation in a time series of statistical models. Given a sequence of observations extracted from the media stream, each observation is modeled as a probability distribution over a feature space. The task is to identify contiguous subsequences whose models are statistically inconsistent with the dominant cluster of models. This is achieved by constructing an undirected graph where nodes represent context models and edges represent pairwise similarity. The segmentation is then derived from the spectral properties of the graph’s affinity matrix, specifically the second generalized eigenvector, which optimally partitions the graph into inlier and outlier clusters.

- describe segmentation using eigenvector analysis

Segmentation is performed using eigenvector analysis of the affinity matrix, a technique derived from spectral graph theory. The affinity matrix is constructed such that each entry represents the similarity between two context models, computed as the exponential of the negative distance between their statistical parameters. The normalized cut criterion is applied to partition the graph into two clusters that minimize inter-cluster similarity while maximizing intra-cluster association. The solution to this optimization problem is found by computing the second generalized eigenvector of the matrix, which provides a continuous-valued indicator of cluster membership. A threshold is applied to this eigenvector to assign each context to either the inlier or outlier cluster.

- introduce affinity matrix

The affinity matrix is a symmetric, positive semi-definite matrix of size N × N, where N is the number of context windows extracted from the time series. Each element A(i,j) represents the similarity between the statistical model of context i and context j, computed as A(i,j) = exp(−d(i,j)/2σ²), where d(i,j) is a commutative distance metric between the two models and σ is a bandwidth parameter controlling the decay of similarity with distance. The matrix encodes the local structure of the time series in model space, enabling the identification of clusters of similar contexts and the isolation of outliers that do not conform to any dominant cluster.

- define partitioning criterion for graph

The partitioning criterion is based on the normalized cut (Ncut) metric, which measures the total dissimilarity between two clusters relative to their internal connectivity. The objective is to minimize the ratio of the cut between clusters to the association within each cluster, ensuring that the resulting partition is both cohesive and distinct. This criterion prevents the formation of isolated, spurious clusters and favors partitions that reflect the underlying statistical structure of the data. The Ncut formulation is particularly suited to the problem of outlier detection because it naturally suppresses small, noisy clusters while emphasizing the separation between the dominant background and the rare foreground.

- minimize Ncut

Minimization of the normalized cut is achieved through spectral decomposition of the affinity matrix. The problem is reformulated as a generalized eigenvalue problem involving the affinity matrix and a degree matrix derived from the row sums of the affinity matrix. The solution to this problem yields a set of eigenvectors, with the second smallest eigenvector providing the optimal partitioning of the graph into two clusters. This eigenvector is interpreted as a continuous-valued cluster indicator, where values near zero correspond to the transition region between clusters and extreme values correspond to strong membership in one cluster or the other.

- describe Shi et al.'s method

The method employed is based on the spectral clustering framework introduced by Shi and Malik, which demonstrates that the normalized cut problem can be relaxed into a generalized eigenvalue system. The original discrete optimization problem is transformed into a continuous one by relaxing the cluster indicator vector to take real values. This relaxation enables the use of linear algebraic techniques to find an approximate solution that is both computationally tractable and statistically meaningful. The method has been successfully applied to image segmentation and is here adapted for temporal segmentation of multimedia streams.

- solve generalized eigenvalue system

The generalized eigenvalue system is solved by first transforming it into a standard eigenvalue problem through a change of variables involving the square root of the degree matrix. This transformation yields a symmetric matrix whose eigenvectors can be computed using standard numerical methods. The trivial solution corresponding to the first eigenvector is discarded, as it represents a uniform distribution over all contexts. The second eigenvector, corresponding to the smallest non-zero eigenvalue, is retained as the cluster indicator vector.

- transform system into standard eigenvalue system

The transformation is performed by substituting z = D^(1/2)y, where D is the diagonal degree matrix and y is the original cluster indicator vector. This substitution converts the generalized eigenvalue problem (W − λD)y = 0 into the standard form D^(-1/2)WD^(-1/2)z = λz, where W is the affinity matrix. The resulting matrix is symmetric and positive semi-definite, ensuring that all eigenvalues are real and non-negative, and that the eigenvectors form an orthogonal basis.

- obtain second generalized eigenvector

The second generalized eigenvector is obtained by computing the eigenvalues and eigenvectors of the transformed matrix and selecting the eigenvector corresponding to the second smallest eigenvalue. This eigenvector encodes the optimal partitioning of the context models into inlier and outlier clusters. The values of this eigenvector are then thresholded to produce a binary segmentation of the time series, with each context assigned to either the inlier or outlier cluster.

- estimate true density using kernel function

To quantify the statistical significance of deviations, the true probability density function of the distance metric between context models is estimated using kernel density estimation. A kernel function, typically Gaussian, is applied to each observed distance value, and the resulting functions are summed to produce a smooth, non-parametric estimate of the underlying density. This estimate is used to compute the likelihood that a given outlier deviation occurred by chance under the assumption of a stationary background process.

- describe mean squared error

The mean squared error (MSE) is used as a measure of the efficiency of the kernel density estimator, balancing the tradeoff between bias and variance. Bias arises when the bandwidth parameter is too large, causing the estimate to oversmooth and lose fine structure. Variance arises when the bandwidth is too small, causing the estimate to overfit to noise. The optimal bandwidth is selected using a data-driven plug-in rule that minimizes the estimated MSE based on the observed data, ensuring that the density estimate is both accurate and robust.

- tradeoff between bias and variance

The selection of the bandwidth parameter in kernel density estimation involves a fundamental tradeoff between bias and variance. A large bandwidth reduces variance by averaging over more samples but increases bias by obscuring local variations. A small bandwidth reduces bias by preserving local detail but increases variance by amplifying noise. The invention employs an adaptive bandwidth selection algorithm that automatically tunes this parameter based on the intrinsic structure of the data, ensuring that the density estimate accurately reflects the statistical properties of the background process without overfitting or underfitting.

- use data-driven bandwidth selection process

The data-driven bandwidth selection process is implemented using the plug-in rule, which estimates the optimal bandwidth by minimizing the asymptotic mean integrated squared error (AMISE). This involves computing an initial estimate of the density using a pilot bandwidth, estimating the second derivative of the density, and using this to derive the optimal bandwidth. The process is iterative and self-contained, requiring no manual tuning and adapting automatically to the scale and complexity of the input data.

- describe unusual event detection method

The unusual event detection method consists of four sequential stages: (1) extraction of low-level audio-visual features from the input stream at a fixed temporal resolution; (2) sampling of the feature sequence into overlapping context windows, each containing a fixed number of frames; (3) estimation of a statistical model for each context window using a Gaussian mixture model or hidden Markov model; and (4) construction of an affinity matrix from pairwise distances between models, followed by spectral clustering to identify outlier contexts. The detected outliers are then ranked according to their statistical deviation from the inlier cluster, as determined by the kernel density estimate of the distance metric.

- extract features from multimedia

Features are extracted from the multimedia stream using standard signal processing techniques. For audio, Mel-frequency cepstral coefficients (MFCCs) are computed at a frame rate of 125 Hz, along with the logarithmic energy. For video, features such as motion vectors, color histograms, and edge density are extracted at a corresponding frame rate. The features are normalized to unit variance and zero mean to ensure consistency across different media types and recording conditions.

- label features using discrete labels

In some embodiments, the continuous features are classified into discrete labels using supervised models such as Gaussian mixture models trained on representative data. Each frame is assigned a label corresponding to the most likely class, such as “speech,” “applause,” or “music.” These discrete labels are used to construct a symbolic time series that is then analyzed using the same framework as the continuous features, enabling comparison between different levels of abstraction.

- treat features as time series

The extracted features, whether continuous or discrete, are treated as a one-dimensional time series indexed by time. Each point in the series corresponds to a feature vector or label assigned to a specific temporal location. This representation allows the application of time-series analysis techniques, including sliding window sampling, statistical modeling, and spectral clustering, to discover patterns and anomalies in the temporal evolution of the media content.

- sample time series using sliding window

The time series is sampled using a sliding window of fixed length, with each window centered on a specific time point and extending over a predetermined number of frames. The window slides forward in time by a fixed step size, creating overlapping contexts that ensure temporal continuity and robustness to transient variations. The window size and step size are selected based on the expected duration of events and the desired temporal resolution of detection.

- construct context model for each sample

For each sampled context, a statistical model is constructed to represent the distribution of features within that window. For continuous features, a Gaussian mixture model is fitted using the expectation-maximization algorithm. For discrete labels, a multinomial probability distribution is estimated from the relative frequencies of each symbol. The model parameters are stored and used to compute pairwise distances between contexts.

- determine affinity matrix using context models

The affinity matrix is constructed by computing the pairwise distance between all context models using a commutative distance metric. For Gaussian mixture models, the distance is defined as the negative log-likelihood of one model given the data of the other, averaged over both directions. For multinomial models, the Euclidean distance between probability vectors is used. The distances are then transformed into similarities using a Gaussian kernel, producing a symmetric matrix of affinity values.

- determine second generalized eigenvector

The second generalized eigenvector of the affinity matrix is computed by solving the generalized eigenvalue problem associated with the normalized cut criterion. The eigenvector is normalized and its values are interpreted as a continuous measure of cluster membership, with extreme values indicating strong membership in either the inlier or outlier cluster and values near zero indicating transition regions.

- cluster distances related to events

The distances between context models are clustered using the spectral partitioning derived from the second generalized eigenvector. The resulting partition separates the time series into contiguous segments of inlier and outlier contexts. The outlier segments correspond to temporal regions where the statistical behavior deviates significantly from the background, indicating the presence of unusual events.

- rank unusual events

Each detected outlier segment is assigned a rank based on its statistical deviation from the inlier cluster. The deviation is quantified by computing the cumulative distribution function (CDF) of the distance metric at the observed distance value for that outlier. The CDF value represents the probability that a randomly selected inlier context would exhibit a distance as large or larger than the observed value. Outliers with lower CDF values are ranked higher, as they are more statistically significant.

- summarize content of multimedia

The final output of the system is a temporally segmented summary of the multimedia content, consisting of a ranked list of outlier segments that correspond to unusual events. Each segment is annotated with its start and end time, its rank, and optionally, its associated feature representation. This summary can be used to generate a condensed version of the content, to trigger alerts in surveillance systems, or to guide human reviewers to the most salient portions of the media.

- describe affinity matrix for golf video

In the case of a golf video, the affinity matrix exhibits a dominant diagonal structure, indicating a stable background of commentator speech and ambient sound. Isolated off-diagonal regions with low affinity values correspond to periods of audience applause, commercial breaks, and interview segments. The second generalized eigenvector clearly separates these regions from the background, enabling their detection without prior knowledge of golf-specific events.

- consider issues in method

Key issues in the method include the selection of appropriate context window size, the choice of distance metric, and the robustness of the statistical models to noise and non-stationarity. The window size must be large enough to ensure reliable model estimation but small enough to capture transient events. The distance metric must be sensitive to meaningful differences while being invariant to irrelevant variations. The statistical models must be flexible enough to capture complex distributions but not so complex as to overfit to noise.

- choose statistical models for context

The statistical models for context are chosen based on the nature of the feature space. For continuous features such as MFCCs, Gaussian mixture models are preferred due to their ability to model multi-modal distributions. For discrete labels, multinomial models are used. In cases where temporal dependencies are present, hidden Markov models are employed to capture first-order memory effects.

- determine confidence measure for detected unusual events

The confidence measure for detected unusual events is derived from the probability density function of the distance metric, estimated using kernel density estimation. For each outlier, the confidence is defined as the tail probability of the distance value under the inlier distribution. A low tail probability indicates high confidence that the event is statistically unusual.

- quantify confidence measure for binomial and multinomial PDF models

For binomial and multinomial models, the distance metric between two probability vectors is modeled as a chi-squared random variable with degrees of freedom equal to the number of symbols minus one. The confidence measure is computed as the cumulative distribution function of this chi-squared distribution evaluated at the observed distance value.

- verify analysis using simulation

The statistical analysis is verified using synthetic time series generated from known probability distributions. Simulations demonstrate that the confidence measure accurately reflects the true probability of deviation, with low false positive rates and high sensitivity to rare events. The results confirm that the method reliably distinguishes between noise-induced fluctuations and true anomalies.

- describe clustering technique for gaining domain knowledge

The clustering technique enables the systematic discovery of domain-specific audio classes by identifying consistent patterns among outlier contexts. For example, in surveillance data, the method reveals that “banging” and “excited speech” are common outlier classes. These classes can then be used to train supervised models for improved detection, effectively transforming an unsupervised discovery process into a guided learning pipeline.

- extract features from audio portion of sports video

Features are extracted from the audio portion of sports video using Mel-frequency cepstral coefficients (MFCCs) computed at 125 Hz, along with logarithmic energy. These features capture the spectral envelope of the audio signal and are robust to variations in speaker identity and background noise.

- obtain distinguishable clusters for selected features

The extracted features are clustered using the spectral method described above, resulting in distinct groups of contexts that correspond to different audio classes such as speech, applause, music, and silence. These clusters are visually and statistically distinguishable, enabling the identification of meaningful audio patterns without prior labeling.

- identify consistent patterns in features for unusual events

By examining the feature distributions within outlier clusters, consistent patterns are identified that correspond to unusual events. For example, in baseball, a burst of applause followed by a pause in commentary is consistently associated with home runs. These patterns are not predefined but emerge from the data, providing a data-driven basis for event interpretation.

- build supervised statistical learning models based on identified features

Once distinctive patterns are identified, supervised models are trained using the outlier contexts as positive examples and the inlier contexts as negative examples. Gaussian mixture models or hidden Markov models are fitted to the feature distributions of each identified class, enabling the construction of classifiers that can detect similar events in future content with high precision.

- demonstrate better results with selected class of features

Experiments demonstrate that the features selected through the unsupervised clustering process yield superior performance in downstream tasks such as highlight detection and anomaly classification compared to hand-selected features. For example, in sports summarization, the discovered class of “excited speech with audience reaction” outperforms traditional models based on applause alone.

- show example of framework for selection of classes of features

An example framework is presented in which the unsupervised method is applied to a corpus of sports videos to discover the most discriminative audio classes. The resulting classes are then used to train a supervised classifier, which is evaluated on a held-out test set. The results show a significant improvement in precision and recall over baseline methods.

- analyze interaction between different clusters of features

The interaction between clusters is analyzed by examining the affinity matrix and the spectral embedding of context models. Clusters that are close in the embedding space are found to correspond to semantically related events, while distant clusters correspond to unrelated phenomena. This analysis reveals hierarchical relationships between event types and enables the construction of multi-level summaries.

- select relevant features for detecting unusual events

Relevant features are selected based on their contribution to the separation between inlier and outlier clusters. Features with high discriminative power, as measured by their effect on the distance metric, are retained, while redundant or noisy features are discarded. This feature selection process enhances the robustness and efficiency of the system.

- describe theory behind minimum description length Gaussian mixture models (MDL-GMMs)

The theory behind minimum description length Gaussian mixture models is based on the principle of Occam’s razor, which favors models that provide the most compact representation of the data. The MDL criterion balances the likelihood of the data under the model against the complexity of the model, measured by the number of parameters. The optimal number of mixture components is selected by minimizing the total description length, ensuring that the model is neither too simple nor too complex.

- derive objective function for obtaining optimal number of mixture components and model parameters

The objective function is derived as the sum of the negative log-likelihood of the data under the model and a penalty term proportional to the number of parameters. The penalty term is computed using the Fisher information matrix and approximated using the Bayesian information criterion. The function is minimized using an iterative optimization procedure that alternates between estimating the model parameters and selecting the number of components.

- estimate parameters K and θ

The parameters K (number of mixture components) and θ (model parameters) are estimated jointly by evaluating the objective function over a range of possible values of K and selecting the value that minimizes the total description length. The model parameters are estimated using the expectation-maximization algorithm for each candidate K.

- describe confidence measure for GMM and HMM models

For GMM and HMM models, the confidence measure is derived from the log-likelihood difference between two models. The distance metric is defined as the average of the cross-likelihoods, and the distribution of this metric under the null hypothesis of identical models is estimated using bootstrapping. The confidence of an outlier is then computed as the tail probability of the observed distance under this distribution.

- model PDF of process using GMM or HMM

The probability density function of the background process is modeled using either a Gaussian mixture model for memoryless features or a hidden Markov model for features with temporal dependencies. The model parameters are estimated from the training data using maximum likelihood estimation, and the resulting model is used to compute the likelihood of new observations.

- define commutative distance metric to compare two context models

The commutative distance metric is defined as the average of the log-likelihood of one model given the data of the other and vice versa. This ensures that the distance is symmetric and invariant to the order of comparison, satisfying the mathematical requirements for constructing a valid affinity matrix.

- use bootstrapping to obtain observations of distance metric

Bootstrapping is used to generate a large number of synthetic distance values by resampling the inlier contexts and computing the distance between pairs of resampled models. This produces an empirical distribution of the distance metric under the assumption that all contexts are drawn from the same underlying process.

- use kernel density estimation to obtain PDF of distance metric

The empirical distribution of the distance metric is smoothed using kernel density estimation to obtain a continuous probability density function. This function is used to compute the probability that a given distance value is consistent with the background process.

- rank outliers using confidence measures

Outliers are ranked according to their confidence measure, which is computed as the cumulative distribution function of the distance metric evaluated at the observed distance. Lower CDF values correspond to higher confidence that the event is unusual, and are assigned higher ranks.

- determine confidence metric for outlier context model

The confidence metric for an outlier context model is computed as the average of the tail probabilities of its distance to all inlier models. This provides a robust estimate of the statistical significance of the outlier, accounting for variability in the background process.

- identify useful features for detecting unusual events

Useful features are identified by analyzing the contribution of each feature dimension to the distance metric. Features that exhibit high variance between inlier and outlier contexts are deemed useful, while features that are invariant across contexts are discarded. This analysis guides the selection of feature subsets that maximize detection performance.

- perform hierarchical clustering using normalized cut on affinity matrix

Hierarchical clustering is performed by recursively applying the normalized cut algorithm to the affinity matrix. First, the entire time series is partitioned into major background clusters. Then, each background cluster is further partitioned to detect nested outliers. This hierarchical approach enables the detection of events at multiple temporal scales.

- partition affinity matrix into individual clusters

The affinity matrix is partitioned into individual clusters using the second generalized eigenvector as a cluster indicator. Each cluster corresponds to a distinct statistical regime in the time series, such as a specific type of background or a recurring event pattern.

- construct affinity matrices for identified clusters

For each identified cluster, a new affinity matrix is constructed using only the context models within that cluster. This allows the detection of sub-clusters and finer-grained anomalies within each major cluster.

- apply spectral clustering to resulting affinity graphs

Spectral clustering is applied to each sub-affinity graph to detect additional levels of structure. This recursive application of spectral clustering enables the discovery of complex, nested patterns of unusual events.

- reveal features using hierarchical clustering

Hierarchical clustering reveals the underlying structure of the feature space by grouping similar context models together. The resulting dendrogram provides a visual representation of the relationships between different event types, enabling the identification of high-level categories and subcategories.

- identify significant features of unusual events

Significant features of unusual events are identified by comparing the feature distributions of outlier clusters with those of inlier clusters. Features that are consistently elevated or suppressed in outliers are deemed significant and are used to characterize the nature of the events.

- train GMM to model distribution of low-level cepstral features

A Gaussian mixture model is trained on the low-level cepstral features extracted from the background regions of the media stream. The model captures the statistical variability of the usual audio environment and serves as the baseline against which unusual events are detected.

- classify sports video into highlight and non-highlight segments

The system classifies sports video into highlight and non-highlight segments by detecting outlier contexts that correspond to statistically unusual events. These outliers are then ranked and selected based on their confidence measure to form a summary of the most significant moments.

- rank every second of input sports video

Every second of the input sports video is assigned a rank based on the confidence measure of the outlier context that overlaps with it. The rank reflects the statistical significance of the event occurring during that second, enabling the construction of a temporally ordered summary.

- set highlights selection threshold

A threshold is set on the confidence measure to determine which outlier segments are included in the final summary. The threshold is chosen to balance the length of the summary with the quality of the detected events, ensuring that only the most significant outliers are retained.

- get interesting time segments

The interesting time segments are extracted from the input video by selecting the time intervals corresponding to the highest-ranked outlier contexts. These segments are concatenated to form a condensed version of the original video.

- compare precision-recall performance of two ranking schemes

The precision-recall performance of the proposed ranking scheme is compared to that of a baseline scheme based on audio energy. The proposed method achieves significantly higher precision and recall, demonstrating the superiority of statistical deviation over simple energy thresholds.

- interpret meaning of MDL-GMM of highlight class

The MDL-GMM of the highlight class represents the most compact and statistically coherent model of the unusual events detected in the data. Its structure reveals the underlying feature patterns that define highlights, such as the co-occurrence of applause and excited speech, and provides a basis for automated interpretation of event semantics.

- assign every audio frame to a mixture component

Each audio frame is assigned to the most likely mixture component of the trained GMM, producing a symbolic sequence of component labels. This sequence is used to construct a discrete time series that is analyzed for temporal patterns.

- infer semantic meaning of every component

The semantic meaning of each mixture component is inferred by examining the feature distribution and the temporal context in which it occurs. For example, a component with high energy and low spectral flatness is interpreted as applause, while a component with periodic modulation is interpreted as music.

- apply method to sports video

The method is applied to a corpus of sports videos, including soccer, baseball, and golf, and successfully detects a wide range of unusual events such as goals, home runs, and crowd reactions, without any domain-specific training.

- detect unusual events in sports video

Unusual events in sports video are detected by identifying temporal segments where the statistical model deviates significantly from the background. These segments correspond to moments of high audience engagement, unexpected actions, or broadcast anomalies.

- apply method to surveillance video

The method is applied to surveillance video, including elevator and traffic intersection footage, and successfully detects unusual events such as banging, screaming, and vehicle collisions, without any prior knowledge of what constitutes an anomaly.

### Overview of Invention

- extract and process features

Low-level audio-visual features are extracted from the multimedia stream at a fixed temporal resolution, normalized, and formatted into a time series representation suitable for statistical modeling.

- train model and construct summary

A statistical model is trained on the time series using a Gaussian mixture model or hidden Markov model, and an affinity matrix is constructed from pairwise distances between context models. Spectral clustering is applied to identify outlier segments, which are ranked by statistical deviation and used to construct a condensed summary of the content.