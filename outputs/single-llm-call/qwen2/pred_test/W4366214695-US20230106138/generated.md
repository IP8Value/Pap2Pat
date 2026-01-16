# DESCRIPTION

## FIELD OF INVENTION

The present invention relates to a multi-modal bio-signal acquisition system and method for predicting and diagnosing surgical site infections (SSIs) in patients. Specifically, the invention provides a system equipped with clinical-grade sensors capable of continuously monitoring peri-wound tissue oxygen saturation (StO2), temperature, and bioimpedance. The system is designed to detect and predict the onset of superficial incisional infections, particularly those caused by Methicillin Susceptible Staphylococcus Aureus (MSSA), using a machine-learning model. This technology aims to enhance the early detection and management of SSIs, thereby reducing healthcare costs and improving patient outcomes.

## BACKGROUND

Surgical site infections (SSIs) are a significant concern in modern healthcare, affecting a substantial number of patients undergoing surgical procedures. SSIs can be classified into superficial incisional, deep incisional, and organ/space infections, with superficial incisional SSIs being the most common. These infections not only prolong hospital stays but also increase the risk of mortality and impose a substantial economic burden on healthcare systems. In the United States alone, SSIs extend hospital stays by an average of 9.7 days and are associated with a 2- to 11-fold increase in the risk of mortality. The economic impact is substantial, with SSIs costing the healthcare system approximately $10 billion annually, or about $25,000 per case.

Traditional methods for post-discharge SSI surveillance, such as direct observation by healthcare professionals, telephone interviews, patient questionnaires, and outpatient clinic follow-ups, are often unstructured and can lead to delayed detection of infections. A 45-hour delay in the detection and treatment of an SSI can increase the odds of infection-related deaths by 3.8 times. Therefore, there is a critical need for more reliable and timely methods to predict and diagnose SSIs.

Recent advancements in sensor technology and machine learning have shown promise in addressing this challenge. For instance, near-infrared spectroscopy (NIRS) has been used to measure tissue oxygenation, and wearable sensors have been developed to monitor wound pH and temperature. However, these approaches typically focus on a single parameter and lack the comprehensive multi-modal monitoring required for early and accurate SSI detection.

The present invention addresses these limitations by providing a multi-modal bio-signal acquisition system that integrates multiple sensors to continuously monitor peri-wound tissue oxygen saturation, temperature, and bioimpedance. The system is designed to work in conjunction with a machine-learning model to predict the onset of SSIs, thereby enabling timely intervention and improved patient outcomes.

## SUMMARY

The present invention provides a multi-modal bio-signal acquisition system and method for predicting and diagnosing surgical site infections (SSIs). The system comprises a sensor apparatus that continuously measures peri-wound tissue oxygen saturation (StO2), temperature, and bioimpedance. The sensor apparatus includes a printed circuit board (PCB) with a custom-built near-infrared spectroscopy (NIRS) subsystem, a digital temperature sensor, a bioimpedance system, and an inertial measurement unit. The system is designed to interface with a bio-signal acquisition device that features onboard data storage and wireless communication capabilities.

The invention further includes a method for using the multi-modal bio-signal acquisition system to predict and diagnose SSIs. The method involves collecting bio-signal data from the peri-wound site, processing the data to filter out motion artifacts, and extracting relevant features. The extracted features are then input into a multi-modal, bagged, stacked, and balanced ensemble logistic regression machine-learning model to predict the presence of a current SSI, as well as the likelihood of an SSI occurring 24 hours and 48 hours in advance of a clinical diagnosis.

Key aspects of the invention include:
1. **Sensor Apparatus**: A multi-modal sensor system capable of continuously monitoring peri-wound tissue oxygen saturation, temperature, and bioimpedance.
2. **Data Acquisition and Processing**: Techniques for filtering and processing raw bio-signal data to extract meaningful features.
3. **Machine-Learning Model**: A multi-modal, bagged, stacked, and balanced ensemble logistic regression model for predicting and diagnosing SSIs.
4. **Clinical Validation**: Evaluation of the system's performance in a porcine model infected with Methicillin Susceptible Staphylococcus Aureus (MSSA).

The invention offers several advantages over existing methods, including:
- **Early Detection**: The system can predict the onset of SSIs 24 to 31 hours before clinical signs become apparent.
- **Continuous Monitoring**: Real-time, continuous monitoring of peri-wound bio-signals allows for timely intervention.
- **High Sensitivity and Specificity**: The machine-learning model demonstrates high accuracy, sensitivity, and specificity in predicting and diagnosing SSIs.
- **Non-Invasive**: The system is non-invasive, making it suitable for use in both clinical and home settings.

## DETAILED DESCRIPTION OF THE DISCLOSURE

### Sensor Apparatus

The multi-modal bio-signal acquisition system of the present invention includes a sensor apparatus designed to continuously monitor peri-wound tissue oxygen saturation (StO2), temperature, and bioimpedance. The sensor apparatus comprises a printed circuit board (PCB) that integrates a custom-built near-infrared spectroscopy (NIRS) subsystem, a digital temperature sensor, a bioimpedance system, and an inertial measurement unit.

#### Near-Infrared Spectroscopy (NIRS) Subsystem

The NIRS subsystem is designed to measure tissue oxygen saturation (StO2) using near-infrared light. The subsystem includes an analog front-end (AFE4420) that collects optical biosensing information. Two integrated light-emitting diodes (LEDs) emit red light at 730 nm and infrared light at 850 nm, while a silicon photodiode with a large active area (7.5 mm²) serves as the photodetector. The LEDs and photodiode are soldered onto the PCB at a source-detector separation distance of 50 mm. The NIRS subsystem operates by emitting near-infrared light into the tissue and measuring the reflected light to determine tissue oxygenation levels.

#### Digital Temperature Sensor

The digital temperature sensor (TMP117, Texas Instruments) is used to measure the temperature of the peri-wound site. The sensor provides high-accuracy temperature measurements and is integrated into the PCB to ensure reliable and consistent data collection.

#### Bioimpedance System

The bioimpedance system (AD5941) is capable of generating high-frequency signals up to 200 kHz and measuring the electrical impedance of the tissue. The system is designed to provide detailed information about the tissue's electrical properties, which can be indicative of changes in tissue physiology associated with infection.

#### Inertial Measurement Unit

The inertial measurement unit (IMU) contains a 3D accelerometer and a 3D gyroscope (LSM6DS33, STMicroelectronics). The IMU is used to detect and compensate for motion artifacts that can affect the accuracy of the bio-signal measurements. The IMU data is processed in real-time to ensure that the bio-signal data remains clean and reliable.

### Data Acquisition and Processing

The sensor apparatus interfaces with a bio-signal acquisition device (CrelyPRO) that features onboard data storage and wireless communication capabilities. The CrelyPRO device utilizes a single-chip solution (Feather nRF52840 by Adafruit) and a real-time clock (RTC, PCF8523 by Adafruit) to manage data acquisition and storage. The device is powered by a 3.7V Lithium Polymer battery and is designed to be lightweight and portable.

#### Data Collection

The sensor apparatus collects bio-signal data from the peri-wound site at regular intervals. Data is acquired via regular burst-mode sampling at a frequency of 50 Hz, with data captured in one-minute bursts over a 10-minute interval. The electronic components are encased in silicon molding to protect them from environmental factors and ensure durability.

#### Data Processing

The raw bio-signal data is processed to filter out motion artifacts and other noise. Signal processing techniques, such as bandpass filtering and wavelet denoising, are applied to ensure that the data is clean and reliable. Relevant features, such as the slope and percentiles of the bio-signals, are extracted using a moving window approach. The extracted features are then used as input for the machine-learning model.

### Machine-Learning Model

The multi-modal, bagged, stacked, and balanced ensemble logistic regression machine-learning model is designed to predict and diagnose SSIs. The model is trained using a dataset of bio-signal measurements from infected and non-infected surgical sites. The dataset includes 5,860 observations from 35 wound sites, with each observation consisting of 90 features extracted from the processed bio-signal data.

#### Model Training

The model is trained using a bootstrap aggregation (bagging) technique to handle class imbalance in the data. The minority classes (infected sites) are up-sampled to ensure that the model is trained on a balanced dataset. The model is validated using leave-one-out cross-validation (LOOCV) to ensure robustness and generalizability.

#### Model Performance

The performance of the machine-learning model is evaluated using several metrics, including the area under the receiver operating characteristic curve (AUC), accuracy, sensitivity, and specificity. The model demonstrates high accuracy, sensitivity, and specificity in predicting the presence of a current SSI, as well as the likelihood of an SSI occurring 24 hours and 48 hours in advance of a clinical diagnosis.

### Clinical Validation

The multi-modal bio-signal acquisition system was evaluated in a porcine model infected with Methicillin Susceptible Staphylococcus Aureus (MSSA). The study involved two skeletally mature pigs, each with 14 incisions: seven inoculated with MSSA, six sham sites, and one control site. The sensor apparatus was sutured to the skin around the incision sites, and bio-signal data was collected continuously from Day -3 to Day 7.

#### Data Analysis

The bio-signal data was analyzed to explore differences in tissue oxygenation, temperature, and bioimpedance between infected and non-infected sites. Statistical analysis using two-tailed unpaired Student's t-tests and cross-correlation analysis was performed to determine the time lag between changes in bio-signals and clinical signs of infection.

#### Results

The results of the study demonstrated that the multi-modal bio-signal acquisition system was capable of detecting changes in tissue physiology associated with MSSA infection 24 to 31 hours before clinical signs became apparent. The machine-learning model achieved an AUC of 0.77 for detecting current SSIs, with higher performance for predicting SSIs 24 hours in advance (AUC = 0.80) and slightly lower performance for predicting SSIs 48 hours in advance (AUC = 0.74).

### Conclusion

The multi-modal bio-signal acquisition system of the present invention provides a reliable and non-invasive method for predicting and diagnosing surgical site infections (SSIs). The system integrates multiple sensors to continuously monitor peri-wound tissue oxygen saturation, temperature, and bioimpedance, and uses a machine-learning model to predict the onset of SSIs. The system has been validated in a porcine model and demonstrates high accuracy, sensitivity, and specificity in detecting and predicting SSIs. The invention has the potential to significantly improve the early detection and management of SSIs, thereby reducing healthcare costs and improving patient outcomes.