Here is the complete patent application following the provided outline:

# DESCRIPTION

## STATEMENT OF GOVERNMENT SPONSORED SUPPORT  
The invention described herein was made with government support under [Grant Number] awarded by [Agency Name]. The government has certain rights in the invention.

## FIELD OF THE INVENTION  
The present invention relates generally to brain-machine interfaces (BMIs) and neural decoding systems. More specifically, the invention pertains to a novel multiplicative recurrent neural network (MRNN) architecture and training methodology that provides robust neural decoding performance across varying recording conditions. The system demonstrates improved stability when faced with electrode signal loss, natural neural variability over time, and other common challenges in BMI operation.

## BACKGROUND OF THE INVENTION  
Current brain-machine interface systems suffer from significant limitations in maintaining consistent performance over time. Conventional neural decoders, particularly those based on linear models like the Kalman filter, are highly sensitive to changes in recording conditions. These changes may include electrode signal degradation, array movement, neural reorganization, or variations in the user's physiological state. The clinical translation of BMI technology has been hindered by this instability, necessitating frequent recalibration sessions that disrupt continuous use.

Prior approaches to address this problem have focused on three main strategies: improving electrode hardware stability, using adaptive decoders that update parameters during use, and employing signal processing techniques to extract more stable neural features. While these methods provide partial solutions, they fail to fully address the fundamental challenge of creating a decoder that can inherently accommodate the wide range of neural variability encountered in practical BMI use.

There exists an unmet need for a neural decoding system that maintains high performance across diverse recording conditions without requiring constant recalibration or adaptation. Such a system would significantly advance the clinical viability of BMI technology by reducing downtime and improving reliability for end users.

## SUMMARY OF THE INVENTION  
The present invention provides a novel neural decoding system based on a multiplicative recurrent neural network (MRNN) architecture that demonstrates unprecedented robustness to recording condition changes. The system comprises three key innovations: (1) a specialized neural network architecture that enables multiplicative interactions between inputs and hidden states, (2) a training methodology that incorporates extensive historical neural data spanning multiple recording sessions, and (3) a data augmentation technique that intentionally perturbs training inputs to increase decoder robustness.

The MRNN decoder transforms neural spike counts into continuous cursor kinematics through a recurrent neural network whose weights are dynamically modulated by the input signals themselves. This multiplicative architecture allows the network to effectively learn and switch between multiple neural-to-kinematic mappings corresponding to different recording conditions. The system is trained using an optimized Hessian-Free algorithm on large datasets comprising months of historical neural recordings, enabling it to recognize and appropriately respond to patterns of neural activity encountered in previous sessions.

Experimental results demonstrate that the invented system maintains superior performance compared to conventional decoders when challenged with electrode signal loss, naturally occurring neural variability, and other common BMI failure modes. Notably, this robustness is achieved without sacrificing peak performance under ideal recording conditions. The invention represents a significant advance in BMI technology by providing reliable, continuous control without requiring frequent recalibration.

## DETAILED DESCRIPTION OF THE INVENTION  

### MRNN Definition  
The multiplicative recurrent neural network (MRNN) at the core of the invention represents a novel architecture for neural decoding. The MRNN is characterized by an N-dimensional vector of activation variables (x) and corresponding firing rates (r = tanh x), where both quantities are continuous in time. Unlike conventional recurrent neural networks where inputs affect dynamics through additive biases, the MRNN architecture allows inputs to parameterize the recurrent weight matrix itself, enabling multiplicative interactions between inputs and hidden states. This critical architectural innovation is mathematically expressed through a specialized dynamics equation governing the activation vector.

### MRNN Output Definition  
The MRNN produces two primary outputs used for cursor control: decoded position and velocity in two-dimensional space. These outputs are generated through separate but parallel MRNN networks, each trained to specialize in either position or velocity estimation. The position output is derived from a weighted sum of network firing rates plus a bias term, while the velocity output follows a similar structure but is trained on numerically differentiated hand velocity data. During operation, these outputs are intelligently blended to provide smooth, stable cursor control.

### Network Construction for Cursor BMI Decoder  
The BMI decoder system combines the position and velocity MRNN outputs through a carefully designed control law. The on-screen cursor position is updated according to a weighted combination of decoded velocity and position, with the velocity component dominating (typically 99% weighting) to provide responsive control while the position component provides stabilization against drift. This blending approach leverages the complementary strengths of both MRNN outputs while mitigating their individual limitations.

### MRNN Initialization  
The MRNN is initialized with network sizes tailored to the specific recording configuration (typically 50-100 units). Weight matrices are initialized with Gaussian-distributed values scaled appropriately for each matrix type, while output weights and biases begin at zero. The time constant τ is set to a physiologically relevant value in the hundreds of milliseconds range. During operation, both the hidden state and output kinematics are initialized to zero at system startup.

### Concatenating Neural Trials for Seeding the MRNN During Training  
The training methodology employs an innovative approach to state initialization where sequences of actual monkey reaching trials are concatenated to form extended training examples. Each training example comprises five consecutive actual trials, with the first two used exclusively for seeding the network's hidden state and the remaining three used for parameter learning. This approach ensures the network develops appropriate dynamics before being evaluated on training targets.

### Perturbing the Neural Input During Training  
A key innovation in the training process is the systematic perturbation of neural inputs during training. This data augmentation technique introduces both global (across all electrodes) and local (electrode-specific) variations in firing rates by randomly adding or removing spikes according to Gaussian-distributed scaling factors. These perturbations simulate various recording condition changes, forcing the network to learn robust mappings that generalize across different input statistics.

### Using Many Days Training Data  
The invention's robustness stems in part from training on extensive historical datasets spanning many months of recordings. During training, minibatches are constructed by randomly sampling trials from across the entire historical dataset, ensuring each update step exposes the network to diverse recording conditions. This approach allows the MRNN to learn a rich set of neural-to-kinematic mappings and recognize when current inputs resemble patterns from previous sessions.

### Network Output  
The complete BMI system generates two-dimensional cursor control signals by combining the outputs of the position and velocity MRNNs. The velocity output provides the primary drive for cursor movement, while the position output contributes a small stabilizing influence. This combined output demonstrates superior smoothness and stability compared to velocity-only decoders while maintaining the responsiveness needed for precise control.

### Training and Running the Networks  
The MRNN parameters are optimized using a modified Hessian-Free algorithm adapted for continuous-time networks. Training proceeds through approximately 200 optimization steps with periodic validation checks to select the best-performing network snapshot. During real-time operation, the trained networks process binned spike counts (typically 20ms bins) to generate continuous cursor kinematics. The system maintains fixed parameters during use, with updates only occurring during offline retraining sessions.