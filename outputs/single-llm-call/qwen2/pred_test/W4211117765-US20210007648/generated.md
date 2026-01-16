# DESCRIPTION

## BACKGROUND

Hemoglobin (Hb) abnormalities are a significant cause of various blood disorders, leading to fatal and chronic health problems such as heart attacks, strokes, and pregnancy complications. Maintaining an adequate Hb blood level (men ≥ 13 g/dL, women ≥ 12 g/dL) is crucial for the proper functioning of major organs like the kidneys, brain, and heart. Anemia, a common Hb disorder, can result from blood loss, decreased red blood cell (RBC) production, and increased RBC destruction. The prevalence of Hb-related disorders is substantial, affecting millions globally, with significant economic and health burdens.

Traditional invasive methods for Hb measurement, such as the cyan-methemoglobin method, are reliable but have limitations, including lack of portability, delayed results, and high costs. These methods are particularly impractical in low- and middle-income countries where frequent invasive testing is inconvenient and potentially risky due to pain, anxiety, and infection risks. Therefore, there is a pressing need for a reliable, affordable, and user-friendly noninvasive point-of-care (POC) tool for Hb measurement.

Recent advancements in smartphone technology have opened new possibilities for noninvasive Hb measurement. Smartphones, equipped with high-resolution cameras, powerful processors, and various sensors, can serve as a portable and cost-effective POC tool. This invention leverages the smartphone's capabilities to develop a noninvasive method for measuring Hb levels using photoplethysmography (PPG) signals and machine-learning algorithms.

## SUMMARY

The present invention relates to a noninvasive method and system for measuring hemoglobin (Hb) levels using a smartphone. The method involves capturing PPG signals from a user's fingertip using the smartphone's camera and processing these signals to estimate Hb levels. The system utilizes near-infrared (NIR) light-emitting diodes (LEDs) with specific wavelengths (850 nm, 940 nm, and 1070 nm) to illuminate the fingertip, capturing the PPG signals under these wavelengths. The captured signals are then processed to extract relevant features, and machine-learning algorithms are applied to predict Hb levels.

Key aspects of the invention include:
1. **Data Collection**: Capturing PPG signals from the fingertip using the smartphone's camera and NIR LEDs.
2. **Signal Processing**: Preprocessing the captured signals to remove noise and artifacts, and extracting features such as systolic and diastolic peaks, PPG rise time, pulse transit time, pulse shape, and amplitude.
3. **Machine Learning**: Applying machine-learning algorithms, such as multiple linear regression (MLR), partial least squares regression (PLSR), and support vector machine regression (SVR), to build a prediction model for Hb levels.
4. **Performance Metrics**: Evaluating the performance of the prediction model using metrics such as mean absolute percentage error (MAPE), mean squared error (MSE), correlation coefficient (r), and Bland-Altman plots.

The invention provides a user-friendly, portable, and cost-effective solution for noninvasive Hb measurement, making it particularly suitable for use in low-resource settings and for frequent monitoring of Hb levels.

## DETAILED DESCRIPTION

### Data Collection

The invention involves capturing PPG signals from a user's fingertip using a smartphone. The smartphone is equipped with a camera and NIR LEDs with specific wavelengths (850 nm, 940 nm, and 1070 nm) to illuminate the fingertip. The user places their fingertip on the smartphone's camera lens, ensuring that the light from the LEDs is directed onto the fingertip. The smartphone captures a video or a series of images of the illuminated fingertip, which are then processed to extract PPG signals.

### Signal Processing

The captured PPG signals are preprocessed to remove noise and artifacts. Common preprocessing techniques include:
1. **Smoothing Filters**: Applying Savitzky-Golay smoothing, Butterworth, and Gaussian filters to smooth the PPG signals and remove high-frequency noise.
2. **Independent Component Analysis (ICA)**: Using ICA to separate the PPG signal from motion artifacts and other sources of interference.
3. **Fourier Series Analysis**: Performing cycle-by-cycle Fourier series analysis to reduce measurement errors.

After preprocessing, features are extracted from the PPG signals. Key features include:
1. **Systolic and Diastolic Peaks**: Identifying the peaks corresponding to the systolic and diastolic phases of the heartbeat.
2. **PPG Rise Time**: Measuring the time taken for the PPG signal to rise from the diastolic to the systolic peak.
3. **Pulse Transit Time**: Calculating the time taken for the pulse wave to travel from the heart to the fingertip.
4. **Pulse Shape and Amplitude**: Analyzing the shape and amplitude of the PPG signal to extract additional information.

### Machine Learning

The extracted features are used to train machine-learning algorithms to predict Hb levels. The following machine-learning algorithms are particularly effective for this purpose:
1. **Multiple Linear Regression (MLR)**: Modeling the relationship between the extracted features and the Hb levels using a linear regression model.
2. **Partial Least Squares Regression (PLSR)**: Calculating latent factors that explain the variation in the target variable (Hb levels) and building a prediction model.
3. **Support Vector Machine Regression (SVR)**: Using a kernel function to create a hyperplane that maximizes the margin between different classes and predicting Hb levels.

### Performance Evaluation

The performance of the prediction model is evaluated using various metrics:
1. **Mean Absolute Percentage Error (MAPE)**: Calculating the average percentage difference between the predicted and actual Hb levels.
2. **Mean Squared Error (MSE)**: Measuring the average squared difference between the predicted and actual Hb levels.
3. **Correlation Coefficient (r)**: Determining the strength of the linear relationship between the predicted and actual Hb levels.
4. **Bland-Altman Plot**: Visualizing the agreement between the predicted and actual Hb levels and calculating the limits of agreement.

### User Interface and Data Transmission

The invention includes a user-friendly mobile application that guides the user through the data collection process. The application ensures that the user places their fingertip correctly on the smartphone's camera and captures the necessary PPG signals. The application also provides real-time feedback on the quality of the captured signals and the estimated Hb levels. The estimated Hb levels can be stored locally on the smartphone or transmitted to a cloud server for further analysis and record-keeping.

### Conclusion

The present invention provides a noninvasive, portable, and cost-effective method and system for measuring hemoglobin (Hb) levels using a smartphone. By leveraging the smartphone's camera and NIR LEDs, the invention captures PPG signals from the user's fingertip and processes these signals to estimate Hb levels using machine-learning algorithms. The invention addresses the limitations of traditional invasive methods and offers a practical solution for frequent Hb monitoring, particularly in low-resource settings.