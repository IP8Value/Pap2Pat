Here is the patent application following the provided outline and research paper:

# DESCRIPTION  

## INTRODUCTION  

The present invention relates generally to the field of audio signal processing and specifically to methods and systems for estimating the location of an active speaker within an enclosed environment such as a vehicle cabin. Accurate location estimation enables improved beamforming techniques for speech enhancement by allowing targeted noise suppression and interference cancellation.  

Beamforming applications particularly benefit from precise location estimation when implemented in environments with fixed speaker positions, such as automobile cabins where passengers occupy defined seating locations. Conventional beamforming techniques like minimum variance distortionless response (MVDR) beamforming and linearly constrained minimum variance (LCMV) beamforming rely on relative transfer functions (RTFs) between microphones to effectively steer beams toward desired speakers while suppressing noise and interference. The present invention provides novel methods for determining speaker location through RTF analysis, enabling more effective beamforming implementations in constrained environments.  

## SUMMARY  

The present invention discloses a method for estimating the location of an active speaker within an enclosed environment using an array of microphones. The method comprises designating one microphone as a reference microphone and storing pre-computed relative transfer functions (RTFs) corresponding to known locations within the environment. During operation, the system obtains a voice sample from the active speaker through the microphone array and calculates speaker RTFs relative to the reference microphone.  

The method performs RTF projection by comparing the calculated speaker RTFs against the stored RTFs using cosine distance measurements. The location of the active speaker is determined by identifying which stored RTF most closely matches the speaker RTFs through this projection analysis. The system is particularly suited for implementation in automobiles where RTFs can be stored for each seat location during a calibration process.  

The invention further discloses a complete system for location estimation comprising a microphone array, storage for pre-computed RTFs, and processing means for performing RTF projection and location determination. Additional features include methods for updating stored RTFs and techniques for handling multiple simultaneous speakers through voice activity detection and speaker identification algorithms.  

## DETAILED DESCRIPTION  

The present invention addresses the technical challenge of accurately estimating speaker location within constrained environments like vehicle cabins where conventional localization techniques may prove inadequate. The disclosed system enables robust location estimation even in noisy conditions through innovative use of relative transfer function analysis.  

The system operates through a calibration process followed by real-time location determination. During calibration, one microphone in the array is designated as the reference microphone. Sound samples are obtained at each microphone position while emitting test signals from known locations within the environment, typically corresponding to seat positions in a vehicle. For each test location, the system performs RTF estimation by calculating the acoustic transfer functions (ATFs) between the source and each microphone, then deriving RTFs as ratios between these ATFs and the reference microphone ATF. These location-specific RTFs are stored in a library for use during real-time operation.  

During real-time operation, the system samples an active speaker through the microphone array. Speaker RTFs are obtained by calculating the ratio between each microphone's received signal and the reference microphone's signal. The system then performs RTF projection by computing the cosine distance between the speaker RTFs and each stored RTF in the library. The location corresponding to the stored RTF with the smallest cosine distance to the speaker RTFs is identified as the active speaker's location.  

The RTF estimation process involves obtaining acoustic transfer functions (ATFs) that characterize how sound propagates from a source to each microphone. These ATFs capture both direct path and reflected path components of the acoustic environment. The system calculates RTFs by taking the ratio between ATFs of microphone pairs, with one microphone serving as the reference. This ratio normalizes out common components of the acoustic paths, making the RTFs more robust to environmental variations.  

For cosine distance calculation, the system treats RTFs as vectors in a high-dimensional space and computes the cosine of the angle between the speaker RTF vector and each stored RTF vector. This measurement provides a robust similarity metric that is insensitive to absolute magnitude differences. The location determination is made by identifying the stored RTF that yields the smallest cosine distance (largest cosine value), indicating the closest match to the speaker's acoustic signature.  

The system's performance is particularly enhanced in automotive applications where seat positions are fixed and stable RTF libraries can be established during vehicle manufacturing or through owner calibration. The fixed geometry of vehicle cabins allows for highly accurate location estimation that can support advanced beamforming implementations for in-vehicle communication systems.  

Additional refinements include techniques for handling multiple simultaneous speakers through voice activity detection and methods for updating stored RTFs to account for environmental changes over time. The system's modular design allows integration with various beamforming implementations while providing critical location information to enhance their performance.