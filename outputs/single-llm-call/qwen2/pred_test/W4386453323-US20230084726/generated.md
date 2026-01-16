# DESCRIPTION

## STATEMENT REGARDING FEDERALLY SPONSORED RESEARCH/DEVELOPMENT

This invention was made with government support under Grant No. [Grant Number] awarded by [Funding Agency]. The government has certain rights in the invention.

## FIELD

The present invention relates generally to the field of quantum sensing, and more particularly to methods and systems for enhancing the sensitivity and signal-to-noise ratio (SNR) of AC magnetic field sensing using quantum logic enhanced (QLE) techniques with nitrogen-vacancy (NV) centers in diamond.

## BACKGROUND

Quantum sensing technologies, particularly those utilizing nitrogen-vacancy (NV) centers in diamond, have gained significant attention due to their potential for high-precision measurements of various physical quantities, including magnetic fields. NV centers are point defects in diamond that consist of a nitrogen atom substituting a carbon atom and an adjacent vacancy. These defects possess unique properties, such as long coherence times and the ability to be optically initialized and read out, making them ideal candidates for quantum sensing applications.

Conventional methods for AC magnetic field sensing using NV centers typically involve dynamical decoupling sequences to improve the coherence time of the electronic spins. However, these methods are limited by the T1 relaxation time of the nuclear spins, which can significantly affect the overall sensitivity and SNR of the measurements. The T1 relaxation time is influenced by flip-flop transitions between the electronic and nuclear spins, which are more pronounced at lower magnetic fields.

To overcome these limitations, recent advancements have focused on leveraging quantum logic operations to enhance the readout fidelity and sensitivity of NV center-based sensors. Quantum logic enhanced (QLE) techniques involve encoding the electronic spin state onto the nuclear spin state using a SWAP operation, followed by multiple readout cycles to map the information back onto the electronic spins. This approach allows for multiple readouts within the nuclear spin lifetime, thereby improving the overall SNR and sensitivity of the measurements.

Despite these advancements, there remains a need for a comprehensive and efficient method to implement QLE techniques in AC magnetic field sensing, particularly for achieving significant sensitivity enhancements over a wide range of sensing durations and magnetic fields.

## SUMMARY

The present invention provides a method and system for enhancing the sensitivity and signal-to-noise ratio (SNR) of AC magnetic field sensing using quantum logic enhanced (QLE) techniques with nitrogen-vacancy (NV) centers in diamond. The invention involves encoding the electronic spin state onto the nuclear spin state using a SWAP operation, followed by multiple readout cycles to map the information back onto the electronic spins. This approach allows for multiple readouts within the nuclear spin lifetime, thereby improving the overall SNR and sensitivity of the measurements.

In one aspect, the invention includes a method for AC magnetic field sensing comprising the steps of:
1. Initializing the electronic spins of an ensemble of NV centers in diamond.
2. Applying a SWAP operation to transfer the polarization from the electronic spins to the nuclear spins.
3. Resetting the electronic spins using an optical polarization pulse.
4. Applying a series of CNOT gates to map the information stored in the nuclear spins back onto the electronic spins.
5. Measuring the electronic spin states optically.
6. Repeating steps 4 and 5 for a plurality of readout cycles to enhance the SNR and sensitivity of the measurements.

In another aspect, the invention includes a system for AC magnetic field sensing comprising:
1. A diamond sample containing an ensemble of NV centers.
2. An optical source for initializing and resetting the electronic spins of the NV centers.
3. A microwave source for applying the SWAP operation and CNOT gates.
4. A radiofrequency source for applying the CNOT gates.
5. A photodetector for measuring the electronic spin states optically.
6. A controller for coordinating the application of the SWAP operation, CNOT gates, and readout cycles.

The invention further provides for the optimization of the SNR and sensitivity by adjusting the number of readout cycles and the duration of the sensing interval based on the T1 relaxation time of the nuclear spins and the overhead time associated with the SWAP operation and readout cycles.

## DETAILED DESCRIPTION

The present invention provides a method and system for enhancing the sensitivity and signal-to-noise ratio (SNR) of AC magnetic field sensing using quantum logic enhanced (QLE) techniques with nitrogen-vacancy (NV) centers in diamond. The invention leverages the unique properties of NV centers, such as their long coherence times and the ability to be optically initialized and read out, to achieve significant improvements in the sensitivity and SNR of the measurements.

### Initialization and SWAP Operation

The method begins with the initialization of the electronic spins of an ensemble of NV centers in diamond. This is typically achieved using an optical polarization pulse, which aligns the electronic spins along a specific axis. Once the electronic spins are initialized, a SWAP operation is applied to transfer the polarization from the electronic spins to the nuclear spins. The SWAP operation is implemented using a sequence of microwave pulses that effectively exchange the quantum states of the electronic and nuclear spins.

### Resetting and CNOT Gates

After the SWAP operation, the electronic spins are reset using an optical polarization pulse. This step is crucial for ensuring that the electronic spins are in a known state before the next readout cycle. Following the reset, a series of CNOT gates are applied to map the information stored in the nuclear spins back onto the electronic spins. The CNOT gates are implemented using a combination of microwave and radiofrequency pulses, which conditionally flip the state of the nuclear spins based on the state of the electronic spins.

### Optical Readout and Repetitive Readout Protocol

The electronic spin states are then measured optically using a photodetector. This readout step is repeated for a plurality of readout cycles to enhance the SNR and sensitivity of the measurements. The large number of NV centers probed allows for a high-precision ensemble average measurement of the sensor spin state with each execution of the repetitive readout protocol.

### Sensitivity Enhancement

The sensitivity enhancement achieved using the QLE protocol is influenced by several factors, including the T1 relaxation time of the nuclear spins, the overhead time associated with the SWAP operation and readout cycles, and the duration of the sensing interval. The T1 relaxation time of the nuclear spins is particularly important, as it determines the number of readout cycles that can be performed within the nuclear spin lifetime. The overhead time, which includes the duration of the SWAP operation and each readout cycle, also plays a critical role in optimizing the sensitivity enhancement.

### Experimental Setup

The experimental setup for implementing the QLE protocol includes a diamond sample containing an ensemble of NV centers. The diamond sample is typically a (2 × 2 × 0.5) mm³ high-purity diamond chip with a nitrogen-doped layer ([N] ≈ 14 ppm) grown using a high-purity chemical vapor deposition process. The NV centers are created through irradiation and annealing, resulting in a concentration of [NV] ≈ 2.3 ppm. The top face of the diamond is cut perpendicular to the [100] crystal axis, and the lateral faces are perpendicular to [110]. The [111] axis of the NV sensor is aligned parallel to the bias magnetic field.

A variable bias magnetic field (0 G to 4000 G) is generated by a feedback-stabilized electromagnet. The 130 mW optical beam (λ = 532 nm) is focused down to a spot size of about 15 µm and pulsed using an acousto-optic modulator. The NV spin-state-dependent fluorescence is read out after 1 µs, followed by additional optical re-initialization of the NV electronic spin for 2 µs. The NV fluorescence signal is collected by a liquid light guide and delivered to a photodetector. Microwave and radiofrequency pulses are generated using an arbitrary waveform generator and a function generator, respectively, and combined using a power splitter before being delivered to the NV diamond.

### Correlation Spectroscopy and Signal Processing

The QLE protocol is particularly effective for AC magnetic field sensing using correlation spectroscopy. In correlation spectroscopy, the time delay between two dynamical decoupling sequences (T_corr) is varied, and the NV fluorescence signal is measured as a function of T_corr. The QLE protocol enhances the SNR and sensitivity by allowing multiple readouts within the nuclear spin lifetime, which is particularly beneficial for long sensing intervals.

The power spectrum of the measured signal is analyzed to extract the AC magnetic field components. The signal amplitudes, A_n, decay with increasing readout cycle index n due to the T1 relaxation of the nuclear spins. To optimize the SNR, the signal amplitude for the n-th readout is weighted by \( w_n = \frac{A_n}{\sigma_n^2} \), where \( \sigma_n \) is the standard deviation of the noise at the n-th readout. The resulting QLE SNR is given by:

\[ \text{SNR}(N) = \sqrt{\sum_{n=1}^{N} w_n A_n^2} \]

The sensitivity enhancement, η_QLE, is defined as the ratio of the QLE SNR to the SNR of the conventional NV electronic spin readout (without quantum logic):

\[ \eta_{QLE} = \frac{\text{SNR}_{QLE}(N)}{\text{SNR}_{Ref}} \]

### Sensitivity Enhancements for Different Sensing Durations

The sensitivity enhancement achieved using the QLE protocol depends on the sensing duration, T_sense, and the number of readout cycles, N. For non-correlation spectroscopy experiments, the sensitivity enhancement is given by:

\[ \eta_{QLE} = \frac{T_{sense} + T_{SWAP} + (N \times T_{QLR})}{T_{sense} + (N \times T_{QLR})} \]

where T_SWAP is the duration of the SWAP operation, and T_QLR is the duration of each readout cycle. For correlation spectroscopy experiments, the sensitivity enhancement is calculated by comparing the total acquisition time for the QLE protocol to the total acquisition time for the conventional protocol.

### Experimental Results

Experimental results demonstrate significant sensitivity enhancements using the QLE protocol. For example, with 2000 QLR cycles, a 33.3(9)× enhancement in SNR is achieved. For non-correlation spectroscopy experiments using an XY8:6 dynamical decoupling sequence with an optimal T_sense of 24 µs, the QLE protocol achieves up to η_QLE = 2.4(3) for N ≈ 150. For DROID-60:6 decoupling, the sensitivity enhancement reaches up to η_QLE = 5.6(3) for N ≈ 400. For correlation spectroscopy experiments with T_corr ranging from 0 to 1.5 ms, the sensitivity enhancement reaches up to η_QLE = 11.3(3) for N ≈ 1000.

### Versatility and Future Directions

The QLE protocol is versatile and broadly applicable to sensing a variety of physical quantities beyond magnetic fields. It is particularly well-suited for NMR spectroscopy, given the preference for strong, uniform bias magnetic fields. The protocol can also be extended to other solid-state spin systems and integrated with additional quantum degrees of freedom to further enhance sensitivity and enable advanced quantum logic or error correction algorithms.

In summary, the present invention provides a method and system for enhancing the sensitivity and SNR of AC magnetic field sensing using quantum logic enhanced (QLE) techniques with nitrogen-vacancy (NV) centers in diamond. The invention leverages the unique properties of NV centers and the QLE protocol to achieve significant improvements in sensitivity and SNR, making it a valuable tool for a wide range of quantum sensing applications.