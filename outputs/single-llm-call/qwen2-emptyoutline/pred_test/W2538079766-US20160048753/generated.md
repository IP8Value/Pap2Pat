# DESCRIPTION

## STATEMENT OF GOVERNMENT SPONSORED SUPPORT

This invention was made with government support under Grant Number [Insert Grant Number] awarded by [Insert Agency Name]. The government has certain rights in the invention.

## FIELD OF THE INVENTION

The present invention relates to the field of brain-machine interfaces (BMIs) and, more specifically, to methods and systems for enhancing the robustness and performance of BMIs by utilizing a Multiplicative Recurrent Neural Network (MRNN) decoder trained on large datasets and perturbed neural inputs.

## BACKGROUND OF THE INVENTION

Brain-Machine Interfaces (BMIs) are systems designed to translate neural activity into control signals for external devices, such as robotic limbs or computer cursors. A critical challenge in the development and clinical translation of BMIs is maintaining decoder performance in the face of recording condition changes. These changes can arise from various sources, including electrode failures, changes in neural activity patterns, and variations in the user's cognitive state. Traditional decoders, such as the Kalman Filter (KF), often require frequent recalibration to maintain performance, which can be cumbersome and time-consuming.

Recent advancements in machine learning, particularly in the domain of recurrent neural networks (RNNs), offer promising solutions to enhance the robustness and performance of BMIs. One such advancement is the Multiplicative Recurrent Neural Network (MRNN) decoder. The MRNN is a computationally powerful nonlinear decoder that can be trained on large datasets representing a variety of recording conditions. By incorporating data augmentation techniques, the MRNN can be made robust to unexpected changes in neural inputs, such as the loss of informative electrodes or day-to-day variations in neural activity.

## SUMMARY OF THE INVENTION

The present invention provides a method and system for enhancing the robustness and performance of brain-machine interfaces (BMIs) using a Multiplicative Recurrent Neural Network (MRNN) decoder. The MRNN decoder is trained on a large dataset of neural recordings collected over multiple days, which includes a variety of recording conditions. Additionally, the training data is augmented with perturbed neural inputs to simulate potential changes in recording conditions. This approach ensures that the MRNN decoder can maintain high performance and robustness under a wide range of conditions, including unexpected electrode failures and day-to-day variations in neural activity.

The invention includes the following key features:
1. **Multiplicative Recurrent Neural Network (MRNN) Architecture**: The MRNN is designed to handle large datasets and learn complex neural-to-kinematic mappings. The input to the MRNN directly parameterizes the recurrent weight matrix, allowing for a multiplicative interaction between the input and the hidden state.
2. **Data Augmentation**: The training data is augmented with perturbed neural inputs to simulate various recording condition changes. This includes adding and removing spikes from each electrode, which helps the MRNN learn to be robust to firing rate reductions and increases.
3. **Training with Large Datasets**: The MRNN is trained on a large dataset of neural recordings collected over multiple days. This ensures that the decoder can generalize to a wide range of recording conditions.
4. **Closed-Loop Control**: Once trained, the MRNN decoder is used in a closed-loop system to control a BMI cursor. The decoder outputs cursor position and velocity, which are blended to update the cursor position on the screen.

## DETAILED DESCRIPTION OF THE INVENTION

### MRNN Definition

The Multiplicative Recurrent Neural Network (MRNN) is a type of recurrent neural network designed to handle large datasets and learn complex neural-to-kinematic mappings. The MRNN architecture is defined by an N-dimensional vector of activation variables, \( \mathbf{x} \), and a vector of corresponding ‘firing rates', \( \mathbf{r} = \tanh(\mathbf{x}) \). Both \( \mathbf{x} \) and \( \mathbf{r} \) are continuous in time and take continuous values. In the MRNN model, the input directly parameterizes the recurrent weight matrix, allowing for a multiplicative interaction between the input and the hidden state. This interaction can be mathematically represented as:

\[
\tau \frac{d\mathbf{x}}{dt} = -\mathbf{x} + \mathbf{J}_u(t) \mathbf{r} + \mathbf{b}_x
\]

where \( \mathbf{J}_u(t) \) is an \( N \times N \times |u| \) tensor describing the weights of the recurrent connections, which are dependent on the E-dimensional input \( \mathbf{u}(t) \). The symbol \( |u| \) denotes the number of unique values \( \mathbf{u}(t) \) can take. To make these computations tractable, the input is linearly combined into F factors, and \( \mathbf{J}_u(t) \) is factorized according to the following formula:

\[
\mathbf{J}_u(t) = \mathbf{J}_{xf} \cdot \text{diag}(\mathbf{J}_{fu} \mathbf{u}(t)) \cdot \mathbf{J}_{fx}
\]

where \( \mathbf{J}_{xf} \) has dimension \( N \times F \), \( \mathbf{J}_{fu} \) has dimension \( F \times E \), \( \mathbf{J}_{fx} \) has dimension \( F \times N \), and \( \text{diag}(\mathbf{v}) \) takes a vector \( \mathbf{v} \) and returns a diagonal matrix with \( \mathbf{v} \) along the diagonal. The network units receive a bias \( \mathbf{b}_x \). The constant \( \tau \) sets the time scale of the network, typically in the range of hundreds of milliseconds.

### MRNN Output Definition

The output of the MRNN is read out from a weighted sum of the network firing rates plus a bias. The output \( \mathbf{z}(t) \) is defined by the equation:

\[
\mathbf{z}(t) = \mathbf{W}_o \mathbf{r} + \mathbf{b}_z
\]

where \( \mathbf{W}_o \) is an \( M \times N \) matrix, and \( \mathbf{b}_z \) is an M-dimensional bias. In the context of BMI control, the MRNN is trained to output the normalized hand position and velocity in the horizontal (x) and vertical (y) spatial dimensions.

### Network Construction for Cursor BMI Decoder

The MRNN is constructed to decode neural activity into cursor control signals. Two separate MRNN networks are trained to output the normalized hand position and velocity. The first network learns to output the normalized hand position through time in both the horizontal (x) and vertical (y) spatial dimensions. The second network learns to output the hand velocity through time, also in the x and y dimensions. The training data for the velocity decoder is calculated from the hand positions using numerical differentiation.

### MRNN Initialization

The MRNN is initialized with random weights and biases. For a network of size \( N \), the non-zero elements of the non-sparse matrices \( \mathbf{J}_{xf} \), \( \mathbf{J}_{fu} \), and \( \mathbf{J}_{fx} \) are drawn independently from a Gaussian distribution with zero mean and variances \( \frac{g_{xf}}{F} \), \( \frac{g_{fu}}{E} \), and \( \frac{g_{fx}}{N} \), respectively. The elements of \( \mathbf{W}_o \) are initialized to zero, and the bias vectors \( \mathbf{b}_x \) and \( \mathbf{b}_z \) are also initialized to zero.

### Concatenating Neural Trials for Seeding the MRNN During Training

To seed the hidden state of the MRNN during training, data from five consecutive actual monkey-reaching trials are concatenated together to form one ‘MRNN training' trial. The first two actual trials in an MRNN training trial are used for seeding the hidden state, while the next three actual trials are used for learning. This process ensures that the MRNN is initialized with a meaningful hidden state before the learning phase begins.

### Perturbing the Neural Input During Training

To enhance the robustness of the MRNN to recording condition changes, the training data is augmented with perturbed neural inputs. The concatenated input \( \mathbf{u}(t) \) is perturbed by adding and removing spikes from each electrode. For electrode \( c \) of the \( j \)-th training trial, the number of actual observed spikes \( n_c^j \) is perturbed according to:

\[
n_c^j = \left\lfloor n_c^j \cdot \eta_j \cdot \eta_c \right\rfloor
\]

where \( \eta_j \) and \( \eta_c \) are Gaussian variables with a mean of one and standard deviations \( \sigma_{\text{trial}} \) and \( \sigma_{\text{electrode}} \), respectively. If \( n_c^j \) is less than zero or greater than the maximum number of spikes, it is resampled to ensure the average number of perturbed spikes is roughly equal to the average number of true (unperturbed) spikes. If \( n_c^j \) is greater than the maximum number of spikes, additional spikes are added to random time bins of the training trial. If \( n_c^j \) is less than the actual number of spikes, spikes are randomly removed from time bins that already have spikes.

### Using Many Days Training Data

The MRNN is trained on a large dataset of neural recordings collected over multiple days. This dataset includes a variety of recording conditions, ensuring that the MRNN can generalize to a wide range of conditions. When training data sets include data from more than one day, a small number of trials from each day are randomly selected for each minibatch. This sampling strategy ensures that every minibatch of training data represents the input distributions from all training days.

### Network Output

The output of the MRNN is used to control a BMI cursor. The decoded velocity and position are initialized to zero, as is the MRNN hidden state. At each decode time step, the MRNN receives binned spike counts as input and outputs the cursor position and velocity. The on-screen position that the cursor moves to during BMI control, \( \Delta x(t) \) and \( \Delta y(t) \), is defined by:

\[
\Delta x(t) = \gamma_v v_x(t) + \beta \gamma_p p_x(t)
\]
\[
\Delta y(t) = \gamma_v v_y(t) + \beta \gamma_p p_y(t)
\]

where \( v_x \) and \( v_y \) are the normalized velocity in the x and y dimensions, \( p_x \) and \( p_y \) are the normalized position in the x and y dimensions, \( \gamma_v \) and \( \gamma_p \) are factors that convert from the normalized velocity and position to the coordinates of the virtual-reality workspace, and \( \beta \) is a factor that sets the amount of position versus velocity decoding. In this setup, the decode is almost entirely dominated by velocity, with a slight position contribution to stabilize the cursor in the workspace.

### Training and Running the Networks

The MRNN is trained using the Hessian-Free (HF) optimization method, which is an exact second-order method that uses backpropagation through time to compute the gradient of the error with respect to the network parameters. The set of trained parameters is \( \{\mathbf{J}_{xf}, \mathbf{J}_{fu}, \mathbf{J}_{fx}, \mathbf{b}_x, \mathbf{W}_o, \mathbf{b}_z\} \). The HF algorithm has critical parameters such as the minibatch size, initial lambda setting, and maximum number of conjugate-gradient iterations. These parameters are set to ensure efficient and effective training.

Once trained, the MRNN is compiled into the embedded real-time operating system and run in closed-loop to provide online BMI cursor control. The MRNN's recurrent connections mean that previous inputs affect how subsequent near-term inputs are processed, but the parameters of the network are fixed during closed-loop use. This ensures that the MRNN is robust to input changes without requiring adaptive updates during use.

By combining a powerful MRNN architecture, large datasets, and data augmentation techniques, the present invention provides a robust and high-performance solution for brain-machine interfaces, addressing the critical challenge of maintaining decoder performance in the face of recording condition changes.