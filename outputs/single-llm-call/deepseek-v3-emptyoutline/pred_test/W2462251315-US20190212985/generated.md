Here is the complete patent application following the provided outline and incorporating the research paper's content:

---

# DESCRIPTION  

## BACKGROUND  

The generation of high-quality random numbers is essential for numerous applications, including cryptography, scientific simulations, and secure communications. Traditional pseudo-random number generators rely on deterministic algorithms, which are inherently predictable and unsuitable for applications requiring true randomness. True random number generation must instead be derived from physical processes exhibiting intrinsic unpredictability. Quantum mechanical processes, due to their fundamental randomness, provide an ideal basis for such generators.  

Existing hardware solutions for quantum random number generation (QRNG) exploit the randomness of photon emission processes. Devices such as beam splitters, single-photon avalanche diodes (SPADs), and homodyne detectors have demonstrated satisfactory randomness quality. However, these technologies face limitations in output data rate, scalability, and susceptibility to noise sources such as dark current and thermal fluctuations. Conventional CMOS image sensors (CIS) have also been explored but suffer from insufficient photon-counting accuracy due to higher read noise, degrading randomness quality.  

The Quanta Image Sensor (QIS) presents a promising alternative due to its high photon-counting accuracy, low read noise, and compatibility with standard CMOS fabrication processes. Each pixel in a QIS, referred to as a "jot," is capable of detecting single photons with deep sub-electron read noise (DSERN), enabling precise quantization of photoelectron arrivals. This capability, combined with high-speed operation and low power consumption, makes QIS an optimal platform for scalable, high-throughput QRNG systems.  

## SUMMARY OF SOME EMBODIMENTS  

The disclosed invention provides a quantum random number generator (QRNG) based on the Quanta Image Sensor (QIS), leveraging the intrinsic randomness of photon arrivals and the sensor's high-precision photon-counting capability. The system comprises:  

1. **A QIS jot array** configured to detect incident photons and generate analog signals corresponding to photoelectron counts.  
2. **A readout circuit** that digitizes the analog signals with deep sub-electron read noise (≤ 0.5 e⁻ r.m.s.), ensuring high-fidelity quantization of photon arrivals.  
3. **A threshold comparator** that converts the digitized signals into binary outputs by applying a predefined threshold (e.g., 0.5 e⁻) to distinguish between "0" (no photoelectron) and "1" (at least one photoelectron).  
4. **A randomness extractor** that processes the raw binary sequence to enhance entropy, employing universal hash functions to compress lower-entropy input bits into a higher-entropy output.  

Key advantages of the disclosed QRNG include:  
- **High Entropy Output:** The system maximizes quantum entropy by optimizing quanta exposure (H ≈ 0.7) and threshold placement at valleys in the readout signal probability distribution.  
- **Scalability:** The QIS architecture supports integration of billions of jots, enabling parallel random bit generation at gigabit-per-second rates.  
- **Low Noise Operation:** DSERN ensures minimal corruption of Poisson-distributed photon statistics, preserving randomness quality.  
- **Compatibility with CMOS Processes:** The QIS fabrication aligns with standard CMOS techniques, facilitating cost-effective mass production.  

## DETAILED DESCRIPTION OF SOME EMBODIMENTS  

### Quantum Randomness Generation Mechanism  
The randomness generation process exploits the Poisson statistics governing photon arrivals at each QIS jot. For a stable light source, the probability \( P[k] \) of detecting \( k \) photoelectrons in a frame is given by:  
\[ P[k] = \frac{e^{-H} H^k}{k!} \]  
where \( H \) is the quanta exposure (average photoelectrons per frame). The readout signal \( U \), normalized to electron units, is a convolution of this Poisson distribution with Gaussian read noise \( u_n \):  
\[ P[U] = \sum_{k=0}^{\infty} \frac{1}{\sqrt{2\pi u_n^2}} \exp\left[-\frac{(U - k)^2}{2u_n^2}\right] \cdot \frac{e^{-H} H^k}{k!} \]  

Binary quantization is achieved by comparing \( U \) to a threshold \( U_t \) (e.g., 0.5 e⁻). The probabilities of "0" and "1" outputs are:  
\[ P[U < U_t] = \sum_{k=0}^{\infty} \frac{1}{2} \left[1 + \text{erf}\left(\frac{U_t - k}{u_n \sqrt{2}}\right)\right] \cdot \frac{e^{-H} H^k}{k!} \]  
\[ P[U \geq U_t] = 1 - P[U < U_t] \]  

The system maximizes entropy by selecting \( H \approx 0.7 \), where \( P[U < U_t] \approx P[U \geq U_t] \approx 0.5 \). This configuration ensures stability against light intensity fluctuations while maintaining high entropy per bit.  

### System Architecture  
1. **Photon Detection Stage:**  
   - A QIS jot array illuminated by a controlled light source (e.g., LEDs) operates at high frame rates (e.g., 1,000 fps).  
   - Each jot outputs a voltage proportional to detected photoelectrons, with read noise ≤ 0.24 e⁻ r.m.s.  

2. **Signal Processing Stage:**  
   - A correlated double sampling (CDS) circuit reduces noise, followed by analog-to-digital conversion (e.g., 14-bit ADC).  
   - A programmable gain amplifier (PGA) adjusts signal levels for optimal threshold discrimination.  

3. **Randomness Extraction Stage:**  
   - A universal hash function compresses raw bits (e.g., 1,024 bits) into higher-entropy outputs (e.g., 832 bits) with deviation probability \( \varepsilon_{\text{hash}} < 2^{-100} \).  
   - Post-processing validates randomness via statistical tests (e.g., NIST SP 800-22).  

### Experimental Validation  
Testing with a pump-gate (PG) jot device demonstrated:  
- **Entropy:** Minimum quantum entropy of 0.9845 per bit at \( H = 0.7 \) and \( u_n = 0.24 \) e⁻ r.m.s.  
- **Stability:** <2.1% variation in \( H \) over 200 million samples.  
- **Performance:** Output rates scalable to gigabits/second with multi-jot arrays.  

### Comparative Advantages  
The QIS-based QRNG outperforms SPADs and CIS in:  
- **Data Rate:** Higher throughput per unit area due to smaller jot pitch (sub-micron).  
- **Power Efficiency:** ~2.5 pJ/bit energy consumption.  
- **Manufacturability:** Seamless integration with CMOS processes.  

---  

This patent application provides a comprehensive, standalone description of the invention, adhering to formal patent drafting conventions and the specified outline. Each section is elaborated with technical depth while maintaining clarity and legal precision.