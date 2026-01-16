# DESCRIPTION

## FIELD

The present invention relates to a method and apparatus for real-time monitoring of plasma processes in semiconductor manufacturing using Radio Emission Spectroscopy (RES). Specifically, the invention provides a non-invasive, contact-free technique for monitoring power variations, pressure variations, chamber wall cleanliness, and plasma conditions in a multiple frequency chamber.

## BACKGROUND

Semiconductor device manufacturing involves complex plasma processes that require precise control and monitoring. Traditional methods for monitoring these processes, such as optical emission spectroscopy (OES) and Langmuir probes, often suffer from limitations such as sensitivity to viewport transparency, invasiveness, and limited real-time capabilities. These limitations can lead to inaccuracies and inefficiencies in process control, affecting the quality and yield of semiconductor devices.

Radio Emission Spectroscopy (RES) is a promising alternative that addresses these issues. RES involves measuring the radio frequency (RF) signals emitted by the plasma, which can provide valuable insights into the plasma's behavior without the need for physical contact or optical access. The technique has been shown to be highly sensitive and capable of real-time monitoring, making it particularly suitable for semiconductor manufacturing environments.

However, the application of RES in semiconductor manufacturing has not been fully explored, and there is a need for a comprehensive method and apparatus that can effectively utilize RES for various monitoring tasks. The present invention aims to fill this gap by providing a detailed and robust approach to using RES for real-time monitoring of power variations, pressure variations, chamber wall cleanliness, and plasma conditions in multiple frequency chambers.

## SUMMARY

The present invention provides a method and apparatus for real-time monitoring of plasma processes in semiconductor manufacturing using Radio Emission Spectroscopy (RES). The invention includes the following aspects:

1. **Real-Time Monitoring of Power Variations in the Process Chamber Using RES**: The method involves placing a loop antenna near the plasma viewport to measure the RF signals at the fundamental frequency (e.g., 13.56 MHz). The amplitude of the RES signal is proportional to the plasma currents within the bulk of the discharge, allowing for the detection of power variations with high sensitivity and accuracy.

2. **Real-Time Monitoring of Pressure Variations in the Process Chamber Using RES**: The method involves measuring the RES signal amplitude at the fundamental frequency while varying the process pressure. The sensitivity of the RES signal to pressure changes allows for the detection of small pressure variations, which is crucial for maintaining process stability.

3. **Real-Time Monitoring of Chamber Wall Cleanliness Using RES**: The method involves measuring the RES signal amplitude before and after contaminating the chamber wall. The change in signal amplitude provides a quantitative measure of the degree of contamination, enabling real-time monitoring of chamber wall cleanliness.

4. **Use of RES to Monitor Plasmas in a Multiple Frequency Chamber**: The method involves measuring the RES signals in a multiple frequency chamber, where the plasma is driven by multiple RF power supplies. The RES signals exhibit frequency mixing components, which can be used to monitor and control the plasma conditions, including the sheath characteristics and ion energy distributions.

5. **Use of RES to Remotely Monitor Changes in Stray Capacitance, Chamber Conditions, or Changes in the Sheath Characteristics of a Plasma**: The method involves analyzing the RES signals to detect changes in stray capacitance, chamber conditions, or sheath characteristics. The non-invasive nature of RES makes it ideal for remote monitoring, ensuring that the plasma process remains stable and controlled.

The invention provides a comprehensive and robust solution for real-time monitoring of plasma processes in semiconductor manufacturing, offering significant improvements over existing methods in terms of sensitivity, accuracy, and non-invasiveness.

## DETAILED DESCRIPTION OF THE DRAWINGS

The detailed description of the drawings is provided to illustrate the various embodiments of the invention. The drawings are not intended to limit the scope of the invention but to provide visual aids to understand the concepts and implementations described herein.

## 1. Real-Time Monitoring of Power Variations in the Process Chamber Using RES

The present invention provides a method for real-time monitoring of power variations in the process chamber using Radio Emission Spectroscopy (RES). The method involves placing a loop antenna near the plasma viewport to measure the RF signals at the fundamental frequency, typically 13.56 MHz. The amplitude of the RES signal is proportional to the plasma currents within the bulk of the discharge, allowing for the detection of power variations with high sensitivity and accuracy.

### Experimental Setup

The experimental setup includes a capacitively coupled plasma (CCP) chamber, such as the Oxford Instruments Plasmalab 100, operated with an oxygen plasma. The chamber is equipped with a loop antenna placed near the plasma viewport, which is used to capture the RF signals. The loop antenna is positioned at a distance of approximately 1 mm from the viewport, with the plane of the loop oriented perpendicular to the viewport. The RF signals are collected using a data acquisition system capable of performing fast Fourier transforms (FFTs) at a high data analysis rate, typically 133 kHz.

### Data Collection and Analysis

The RF signals at the fundamental frequency (13.56 MHz) are collected while varying the applied electrode RF power. The chamber is operated by feeding oxygen gas at a flow rate of 50 sccm and a pressure of 100 mTorr. The RES signals are collected by varying the electrode power from 50 W to 500 W. The variation in RES signal amplitude is found to range from approximately 10 dB, which represents a ten-fold change in signal amplitude on a linear scale.

### Sensitivity and Accuracy

The sensitivity of the RES technique to power variations is demonstrated by the ability to detect a power change as low as 5 W with an error of less than 0.4%. Within the 50-150 W power range, a logarithmic signal change of approximately 5.5 dB corresponds to a linear change in signal intensity of 350%, allowing for a sensitivity estimate of approximately 3.5% per watt. This high sensitivity and accuracy make the RES technique a reliable and effective method for real-time monitoring of power variations in the process chamber.

### Correlation with Fundamental Plasma Parameters

To understand the variations in the RES signal with power, the fundamental plasma parameters, such as electron density (ne) and electron temperature (Te), are analyzed. The conduction current (Ic) in the bulk plasma is responsible for ohmic heating and is a key indicator of power dissipation. The conduction current is given by the equation:

\[ J_c = \frac{ne \cdot e \cdot \nu_d}{\omega_{rf} \cdot \epsilon_0} \]

where \( J_c \) is the conduction current density, \( ne \) is the electron density, \( e \) is the electron charge, \( \nu_d \) is the electron drift velocity, \( \omega_{rf} \) is the applied RF frequency, and \( \epsilon_0 \) is the vacuum permittivity. The conduction current is found to range from 0.214 A at 50 W to 1.1 A at 500 W, which is in good agreement with the spatially averaged current (I.cosΦ) measured using a V-I probe. The RES signal (IRES) also shows an increasing trend as a function of power, confirming that the RES probe is a current sensor that measures localized current in the plasma chamber.

## 2. Real-Time Monitoring of Pressure Variations in the Process Chamber Using RES

The present invention provides a method for real-time monitoring of pressure variations in the process chamber using Radio Emission Spectroscopy (RES). The method involves measuring the RES signal amplitude at the fundamental frequency while varying the process pressure. The sensitivity of the RES signal to pressure changes allows for the detection of small pressure variations, which is crucial for maintaining process stability.

### Experimental Setup

The experimental setup is similar to that described for monitoring power variations. The Oxford Instruments Plasmalab 100 CCP chamber is operated with an oxygen plasma, and the loop antenna is placed near the plasma viewport to capture the RF signals. The RF signals are collected using a data acquisition system capable of performing fast Fourier transforms (FFTs) at a high data analysis rate, typically 133 kHz.

### Data Collection and Analysis

The RF signals at the fundamental frequency (13.56 MHz) are collected while varying the process pressure. The chamber is operated by feeding oxygen gas at a flow rate of 50 sccm and an applied electrode RF power of 200 W. The RES signals are collected by varying the pressure from 10 mTorr to 250 mTorr. The variation in RES signal amplitude is found to range from approximately 4 dB, which represents a 250% change in signal intensity on a linear scale.

### Sensitivity and Accuracy

The sensitivity of the RES technique to pressure variations is demonstrated by the ability to detect a pressure change as low as 1 mTorr with an error of less than 0.1%. Within the 10-25 mTorr pressure range, a logarithmic signal change of approximately 4 dB corresponds to a linear change in signal intensity of 250%, allowing for a sensitivity estimate of approximately 2.5% per mTorr. This high sensitivity and accuracy make the RES technique a reliable and effective method for real-time monitoring of pressure variations in the process chamber.

### Correlation with Fundamental Plasma Parameters

To understand the variations in the RES signal with pressure, the fundamental plasma parameters, such as electron density (ne) and electron temperature (Te), are analyzed. The conduction current (Ic) in the bulk plasma is given by the equation:

\[ J_c = \frac{ne \cdot e \cdot \nu_d}{\omega_{rf} \cdot \epsilon_0} \]

The electron density (ne) is found to range from 2.75 × 10^15 m^-3 at 10 mTorr to 2.5 × 10^16 m^-3 at 200 mTorr, with corresponding electron plasma frequencies (ωp) ranging from 0.47 GHz to 1.74 GHz. The electron temperature (Te) is found to range from 4.5 eV at 10 mTorr to 0.8 eV at 200 mTorr. The conduction current is found to range from 0.1 A at 10 mTorr to 0.59 A at 200 mTorr, which is in good agreement with the spatially averaged current (I.cosΦ) measured using a V-I probe. The RES signal (IRES) also shows an increasing trend as a function of pressure, confirming that the RES probe is a current sensor that measures localized current in the plasma chamber.

## 3. Real-Time Monitoring of Chamber Wall Cleanliness Using RES

The present invention provides a method for real-time monitoring of chamber wall cleanliness using Radio Emission Spectroscopy (RES). The method involves measuring the RES signal amplitude before and after contaminating the chamber wall. The change in signal amplitude provides a quantitative measure of the degree of contamination, enabling real-time monitoring of chamber wall cleanliness.

### Experimental Setup

The experimental setup is similar to that described for monitoring power and pressure variations. The Oxford Instruments Plasmalab 100 CCP chamber is operated with an oxygen plasma, and the loop antenna is placed near the plasma viewport to capture the RF signals. The RF signals are collected using a data acquisition system capable of performing fast Fourier transforms (FFTs) at a high data analysis rate, typically 133 kHz.

### Data Collection and Analysis

The RF signals at the fundamental frequency (13.56 MHz) are collected before and after contaminating the chamber wall. The contamination is simulated by applying positive photoresist (Microposit S1818 TM G2) to a section of aluminum foil, which is placed on the wall of the process chamber. The oxygen plasma is ignited at an applied power of 500 W and a pressure of 50 mTorr, with an oxygen flow rate of 50 sccm. The RES signals are collected continuously for an interval of 4.3 hours. The percentage coverage of the contaminated area is approximately 1.5%.

### Sensitivity and Accuracy

The sensitivity of the RES technique to chamber wall cleanliness is demonstrated by the clear and measurable difference in the amplitudes of the RES signals collected before and after the contamination of the chamber wall. The RES signal amplitude from the contaminated wall slowly approaches that of the clean wall as the photoresist is removed during cleaning by the oxygen plasma. The percentage change in the RES signal amplitude provides a quantitative measure of the degree of contamination, enabling real-time monitoring of chamber wall cleanliness.

### Qualitative Model

A qualitative model is used to describe the relationship between the RES signal and the chamber wall cleanliness. The model considers the effective plasma-to-chamber wall capacitance (Ceff) and the additional capacitance due to the dielectric coating (Cf). The effective impedance (Zeff) decreases as the film thickness (tf) decreases, leading to an increase in the RES signal amplitude as the contaminant film thickness is progressively reduced to zero. This qualitative model provides a theoretical basis for understanding the observed behavior of the RES signal in response to chamber wall contamination.

## 4. Use of RES to Monitor Plasmas in a Multiple Frequency Chamber

The present invention provides a method for using Radio Emission Spectroscopy (RES) to monitor plasmas in a multiple frequency chamber. The method involves measuring the RES signals in a multiple frequency chamber, where the plasma is driven by multiple RF power supplies. The RES signals exhibit frequency mixing components, which can be used to monitor and control the plasma conditions, including the sheath characteristics and ion energy distributions.

### Experimental Setup

The experimental setup includes a multiple frequency chamber, such as the Lam EXELAN 2300, which consists of a combination of driving frequencies at 2 MHz, 27 MHz, and 162 MHz. The chamber is operated with an Ar/O2 plasma, and the loop antenna is placed near the plasma viewport to capture the RF signals. The RF signals are collected using a data acquisition system capable of performing fast Fourier transforms (FFTs) at a high data analysis rate, typically 133 kHz.

### Data Collection and Analysis

The RF signals are collected while operating the plasma using a combination of 162 MHz and 2 MHz frequencies with applied powers of 250 W and 50 W, respectively. The majority of the captured RES signal is found within an approximately 30 MHz frequency span of the main drive frequency at 162 MHz. Frequency mixing components of the 162 MHz signal with the lower 2 MHz frequency are clearly seen in the captured RES data, indicating the presence of beat frequencies with a regular frequency shift (Δf) of 2 MHz.

### Qualitative Model

A qualitative model is used to explain the observed frequency mixing components in the RES signal. The model considers the non-linear sheath response, which acts as an effective "diode mixer" for the multiple applied power supplies. The small-signal analysis of the diode mixer model shows that multiple heterodyning (sideband) components are generated, which are related to the frequency of the lower frequency "ion control" power supply. The intensities of these sidebands can be correlated to the electron temperature (Te) and electron density (ne) in the plasma chamber, providing valuable insights into the plasma conditions.

## 5. Use of RES to Remotely Monitor Changes in Stray Capacitance, Chamber Conditions, or Changes in the Sheath Characteristics of a Plasma

The present invention provides a method for using Radio Emission Spectroscopy (RES) to remotely monitor changes in stray capacitance, chamber conditions, or changes in the sheath characteristics of a plasma. The method involves analyzing the RES signals to detect changes in these parameters, which can affect the stability and performance of the plasma process.

### Experimental Setup

The experimental setup is similar to that described for monitoring plasmas in a multiple frequency chamber. The Lam EXELAN 2300 multiple frequency chamber is operated with an Ar/O2 plasma, and the loop antenna is placed near the plasma viewport to capture the RF signals. The RF signals are collected using a data acquisition system capable of performing fast Fourier transforms (FFTs) at a high data analysis rate, typically 133 kHz.

### Data Collection and Analysis

The RF signals are collected while varying the process conditions, such as the applied power, pressure, and gas composition. The changes in the RES signals are analyzed to detect variations in stray capacitance, chamber conditions, or sheath characteristics. The non-invasive nature of RES allows for remote monitoring, ensuring that the plasma process remains stable and controlled.

### Qualitative Model

A qualitative model is used to explain the relationship between the RES signals and the changes in stray capacitance, chamber conditions, or sheath characteristics. The model considers the effective plasma-to-chamber wall capacitance (Ceff) and the additional capacitance due to changes in the plasma conditions. The changes in the effective impedance (Zeff) lead to variations in the RES signal amplitude, providing a quantitative measure of the changes in the plasma conditions. This qualitative model provides a theoretical basis for understanding the observed behavior of the RES signal in response to changes in the plasma environment.

### Conclusion

The present invention provides a comprehensive and robust method and apparatus for real-time monitoring of plasma processes in semiconductor manufacturing using Radio Emission Spectroscopy (RES). The invention offers significant improvements over existing methods in terms of sensitivity, accuracy, and non-invasiveness, making it an ideal solution for maintaining process stability and improving the quality and yield of semiconductor devices. The invention is particularly useful for monitoring power variations, pressure variations, chamber wall cleanliness, and plasma conditions in multiple frequency chambers, and for remotely detecting changes in stray capacitance, chamber conditions, or sheath characteristics.