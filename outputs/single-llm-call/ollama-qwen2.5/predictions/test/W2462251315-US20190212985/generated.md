# Patent Application: Quantum Random Number Generator Using Quanta Image Sensor (QIS)

## Abstract

This patent application describes a quantum random number generator (QRNG) using a Quanta Image Sensor (QIS). The QIS leverages the intrinsic randomness in photon emission and its single-photon detection capability to generate high-quality random numbers. The method includes data collection, threshold setting, entropy maximization, and post-processing with a randomness extractor to ensure uniform randomness.

## Background

### Field of Invention
The present invention relates to quantum random number generators (QRNGs) and more specifically to a QRNG using a Quanta Image Sensor (QIS).

### Description of Related Art
Random number generation is crucial for various applications, including cryptography, simulations, and gaming. Traditional methods often rely on pseudo-random number generators (PRNGs), which are deterministic and can be predictable. Quantum random number generators (QRNGs) offer true randomness by leveraging quantum phenomena, such as the intrinsic randomness in photon emission.

## Summary of Invention

### Overview
The invention provides a method for generating high-quality random numbers using a Quanta Image Sensor (QIS). The QIS's single-photon detection capability and low read noise make it an ideal candidate for QRNGs. The method includes collecting data from the QIS, setting a threshold to maximize entropy, and post-processing the raw data with a randomness extractor.

## Detailed Description

### QIS Jot Device
The Quanta Image Sensor (QIS) is composed of individual pixels called jots. Each jot can detect single photons and has low read noise, making it suitable for high-precision measurements. The QIS jot device used in this application has an analog readout approach, where the output signal from multiple columns is selected by a multiplexer and amplified before being digitized.

### Data Collection
The data collection process involves illuminating the QIS with a stable light source, such as an array of green LEDs. The distance between the light source and the sensor is controlled to ensure consistent illumination. The readout electronics include a programmable gain amplifier (PGA) and an off-chip 14-bit ADC for digitizing the analog signal. Raw data is collected at a rate of 10 ksamples/s, with each sample being a 14-bit digital value.

### Threshold Setting
To generate binary random numbers, a threshold \(U_t\) is set based on the median of the testing samples. The quanta exposure \(H\) is determined using the Photon Counting Histogram (PCH) method. For optimal entropy and stability, \(H \cong 0.7\) and \(U_t = 0.5\) e\(^-\) are preferred. These settings ensure that the output data has a balanced distribution of 0s and 1s.

### Entropy Maximization
The minimum quantum entropy per output bit is calculated using the formula:
\[ \overline{S} = -p_0 \log_2(p_0) - p_1 \log_2(p_1) \]
where \(p_0\) and \(p_1\) are the probabilities of 0s and 1s, respectively. For \(H = 0.7\) and read noise of 0.24 e\(^-\), the minimum quantum entropy is approximately 0.9845.

### Randomness Extractor
A randomness extractor based on Universal-2 hash functions is used to post-process the raw data. The extractor computes a number \(q\) of high-entropy output bits from a larger number \(n\) of lower-entropy input bits. The compression factor is adjusted to ensure that the output deviates minimally from perfect uniform randomness. For example, with \(n = 1024\), the compression factor is 1.23, resulting in a loss of only 18% of the input raw bits.

### Statistical Tests
The generated random numbers are subjected to NIST statistical tests to ensure their quality. These tests evaluate the proportion of 0s and 1s, the presence of patterns, and the possibility of compression without loss of information. The QIS-based QRNG passed all these tests, confirming its high-quality randomness.

### Comparison with Other Technologies
Compared to other QRNG technologies, such as Single Photon Avalanche Diodes (SPADs) and conventional CMOS image sensors (CIS), the QIS offers a better tradeoff between data rate and scalability. SPADs require high supply voltages and suffer from after-pulsing phenomena, while CIS has lower randomness quality due to its inability to detect single photons. The QIS combines the advantages of both technologies, providing high-speed, high-quality random number generation.

## Claims

1. A method for generating quantum random numbers using a Quanta Image Sensor (QIS), comprising:
   - Illuminating the QIS with a stable light source.
   - Collecting raw data from the QIS at a predetermined rate.
   - Setting a threshold \(U_t\) based on the median of testing samples to generate binary random numbers.
   - Post-processing the raw data using a randomness extractor to ensure uniform randomness.

2. The method of claim 1, wherein the quanta exposure \(H\) is set to approximately 0.7 to maximize entropy and stability.

3. The method of claim 1, further comprising performing NIST statistical tests on the generated random numbers to verify their quality.

4. A system for generating quantum random numbers using a Quanta Image Sensor (QIS), comprising:
   - A QIS with single-photon detection capability.
   - Readout electronics including a multiplexer and a programmable gain amplifier (PGA).
   - An off-chip 14-bit ADC for digitizing the analog signal.
   - A processor for setting a threshold \(U_t\) and post-processing raw data using a randomness extractor.

5. The system of claim 4, wherein the quanta exposure \(H\) is set to approximately 0.7 to maximize entropy and stability.

6. The system of claim 4, further comprising a stable light source and a precision voltage source for controlling the intensity of the light source.

## Conclusion
The invention provides a robust method for generating high-quality random numbers using a Quanta Image Sensor (QIS). By leveraging the intrinsic randomness in photon emission and the single-photon detection capability of the QIS, this method offers significant advantages over existing QRNG technologies. The combination of data collection, threshold setting, entropy maximization, and post-processing ensures that the generated random numbers are truly random and suitable for various applications.