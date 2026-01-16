# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to methods and systems for classifying biological particles, particularly cancer cells, based on their electrical properties. More specifically, the invention provides a device and method for rapidly assessing the viability of cancer cells in response to targeted therapy using multifrequency impedance spectroscopy in combination with machine learning algorithms.

## BACKGROUND OF THE INVENTION

Cancer remains one of the leading causes of mortality globally. Traditional treatments such as chemotherapy, while effective, often result in significant side effects due to their non-specific nature. Targeted therapies, which aim to selectively eliminate tumor cells while sparing healthy tissue, offer a promising alternative. However, the effectiveness of these therapies can vary significantly among patients, necessitating rapid and accurate methods to assess drug efficacy.

Current methods for assessing cell viability, such as the trypan blue dye exclusion method, require staining and bulky optical instrumentation, limiting their utility in point-of-care settings. Microfluidic and impedance-based techniques have emerged as potential alternatives, offering label-free and miniaturized solutions. However, these methods often rely on single-frequency analysis, which may not provide sufficient information for accurate classification.

The present invention addresses these limitations by providing a multifrequency impedance cytometry system combined with machine learning algorithms. This system can rapidly and accurately assess the viability of cancer cells in response to targeted therapy, enabling personalized treatment decisions.

## SUMMARY OF THE INVENTION

The present invention provides a method and system for classifying biological particles, particularly cancer cells, based on their electrical properties. The method involves treating the cells with a targeted therapeutic agent, measuring the impedance response of the cells at multiple frequencies, and classifying the cells as live or dead using a machine learning algorithm.

In one aspect, the invention provides a method for assessing the viability of cancer cells, comprising:
1. Providing a sample of cancer cells.
2. Treating the cancer cells with a targeted therapeutic agent.
3. Measuring the impedance response of the cells at multiple frequencies using a multifrequency impedance cytometry system.
4. Extracting features from the impedance data, including amplitude change and phase change.
5. Classifying the cells as live or dead using a support vector machine (SVM) classifier trained on the extracted features.

In another aspect, the invention provides a system for assessing the viability of cancer cells, comprising:
1. A microfluidic channel with embedded electrodes.
2. A multifrequency lock-in amplifier for measuring the impedance response of the cells.
3. A data processing unit for extracting features from the impedance data and classifying the cells using a machine learning algorithm.

The invention further provides definitions for key terms used in the detailed description of the invention.

## DETAILED DESCRIPTION OF THE INVENTION

### A. Methods and Systems for Classifying Biological Particles

The present invention provides a comprehensive method and system for classifying biological particles, particularly cancer cells, based on their electrical properties. The method involves treating the cells with a targeted therapeutic agent, measuring their impedance response at multiple frequencies, and classifying the cells as live or dead using a machine learning algorithm. The system includes a microfluidic channel with embedded electrodes, a multifrequency lock-in amplifier, and a data processing unit.

#### Microfluidic Channel and Electrode Fabrication

The microfluidic channel is fabricated using standard photolithography on a 3-inch fused silica wafer. The process involves photo-patterning resist on the wafer, electron beam metal evaporation, and liftoff processing. The photo-patterning process includes wafer cleaning, spin coating the photoresist, soft bake of the resist, ultraviolet light exposure through a chromium mask, resist development, and hard bake of the resist. A 100-nm gold layer is deposited on the substrate using electron beam evaporation, with a 10-nm chromium layer used to enhance adhesion. The width of the electrodes is 20 μm, and the spacing between the two electrodes is 25 μm.

The microfluidic channel itself is fabricated in PDMS (polydimethylsiloxane) using soft lithography. A layer of SU-8 is patterned onto a 3-inch silicon wafer to act as a master mold. The SU-8 photo-patterning process involves standard cleaning, spin coating, soft baking, exposure, development, and hard baking. PDMS (10:1 prepolymer/curing agent) is poured onto the master mold and baked at 80°C for 2 hours to cure. The PDMS channel is then peeled off from the mold, and holes are punched to form the inlet and outlet. The PDMS substrate is aligned and bonded to the electrode chip after both substrates undergo oxygen plasma treatment. The bonded chip is baked at 70°C for 30 minutes to form an irreversible bond. The microfluidic channel has a width of 100 μm and a height of 30 μm.

#### Multifrequency Impedance Cytometry System

The multifrequency impedance cytometry system includes a multifrequency lock-in amplifier (e.g., Zurich Instruments®) and software for recording and analyzing the data. The system measures the impedance response of the cells at discrete frequencies ranging from 300 kHz to 30 MHz. For each cell type, a series of measurements is performed at four discrete frequencies simultaneously, with 500 kHz always included as one of the frequencies.

The recorded data is post-processed using an algorithm to detrend and denoise the data, which helps to analyze the cytometry data with minimal error. Two significant features are extracted from the data: amplitude change and phase change. Amplitude change is defined as the change in amplitude level when a cell passes by, obtained by finding the difference between the baseline voltage and the peak voltage of a cell passing by. Phase change is defined as the change in angular position of the excitation frequency when a cell passes by, calculated from the real and imaginary data points obtained from the data.

#### Machine Learning Algorithm

The machine learning algorithm used in the invention is a support vector machine (SVM) with a Gaussian kernel. SVMs are supervised learning models that can efficiently perform nonlinear classification by implicitly mapping their inputs to high-dimensional feature spaces. The data used for training consists of features extracted from 100% live and 100% dead cells. For training, the features from live cells are labeled as 1, and the features from dead cells are labeled as 0. The training data set size is more than 1000 events (peaks from the impedance data corresponding to a cell passing over the electrodes) to prevent overfitting.

The SVM classifier is evaluated using a confusion matrix on a set of test data for which the true values are known. The performance metrics used to evaluate the SVM classifier include accuracy, true positive rate (TP), and true negative rate (TN). The confusion matrix is built using a portion of the training data, which includes features from 100% live and 100% dead cells.

#### Performance Evaluation

The performance of the SVM classifier is evaluated using three different tumor cell test samples with varying viability percentages (90% live, 50% live, and 82% live). The classifier's accuracy is assessed by comparing the predicted viability percentages with the ground truth obtained using the trypan blue staining method. The results show that the SVM classifier using both amplitude change and phase change as features achieves the highest accuracy, with an overall accuracy of 95.9%, a TP rate of 95%, and a TN rate of 97%.

### B. Definitions

- **Biological Particles**: Any cellular or subcellular entity, including but not limited to cancer cells, bacteria, and viruses.
- **Impedance Spectroscopy**: A technique used to measure the electrical impedance of a material over a range of frequencies.
- **Multifrequency Impedance Cytometry**: A method that measures the impedance response of individual cells at multiple frequencies to obtain a comprehensive profile of their electrical properties.
- **Support Vector Machine (SVM)**: A supervised learning model used for classification and regression analysis. SVMs find an optimal hyperplane in a high-dimensional space that distinctly classifies the data points.
- **Amplitude Change**: The change in the amplitude level of the impedance signal when a cell passes by the electrodes, calculated as the difference between the baseline voltage and the peak voltage.
- **Phase Change**: The change in the angular position of the excitation frequency when a cell passes by the electrodes, calculated from the real and imaginary components of the impedance data.
- **Machine Learning**: A field of computer science that uses statistical techniques to enable machines to improve their performance on a specific task over time.
- **Gaussian Kernel**: A kernel function used in SVMs to map input data into a higher-dimensional space, enabling the SVM to handle nonlinear classification problems.
- **Confusion Matrix**: A table used to describe the performance of a classification model, showing the number of true positives, true negatives, false positives, and false negatives.
- **True Positive Rate (TP)**: The proportion of actual positive cases that are correctly identified by the classifier.
- **True Negative Rate (TN)**: The proportion of actual negative cases that are correctly identified by the classifier.
- **Point-of-Care Diagnostic**: A medical test that is performed close to or at the site of a patient, providing rapid results to guide immediate treatment decisions.

The present invention provides a robust and accurate method and system for assessing the viability of cancer cells in response to targeted therapy. By combining multifrequency impedance spectroscopy with machine learning algorithms, the invention offers a label-free, miniaturized, and rapid solution for personalized cancer treatment.