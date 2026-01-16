# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of digital communication systems, particularly to methods and apparatuses for encoding and decoding data for transmission over noisy channels. More specifically, the invention pertains to the use of deep learning techniques to design and implement autoencoders (AEs) for reliable communication over such channels. The invention introduces a novel framework called list autoencoder (listAE) that enhances the error correction capabilities of traditional AEs by allowing the decoder to output a list of candidate message words. This approach leverages the concept of list decoding from classical coding theory and is applicable to various AE architectures, including a specific architecture known as incremental redundancy autoencoder (IR-AE).

## SUMMARY

The invention provides a method and system for designing and implementing a list autoencoder (listAE) for reliable communication over noisy channels. The listAE framework allows the decoder to output a list of candidate message words, thereby improving the error correction performance compared to traditional autoencoders that output a single candidate. The invention includes a specific loss function designed to optimize the performance of a genie-aided (GA) decoder, which counts a block error event only if the transmitted message word is not present in the output list. During the testing phase, the functionality of the GA decoder is emulated using cyclic redundancy check (CRC) code to select a single candidate from the list. The invention also presents a more general architecture, incremental redundancy autoencoder (IR-AE), which decodes the received word on a sequence of component codes with non-increasing rates, demonstrating improved performance over existing architectures, especially for smaller list sizes.

## DETAILED DESCRIPTION

### Introduction

Reliable transmission over noisy channels has been a significant challenge in digital communication systems. Traditional channel coding techniques, such as Turbo, Low-Density Parity-Check (LDPC), and polar codes, have been developed to achieve reliable transmission by mapping input data to higher-dimensional representations. These codes are designed using mathematical analysis and tailored to specific channel models, often the Additive White Gaussian Noise (AWGN) channel. However, the design of these codes has been sporadic and heavily reliant on human ingenuity.

Recent advancements in deep learning have opened new avenues for automating the design of channel encoders and decoders. Deep learning frameworks, particularly autoencoders (AEs), have shown promise in designing codes that can compete with or even outperform classical codes, especially for channels with complex or undefined models. The present invention builds on this foundation by introducing a novel framework called list autoencoder (listAE), which extends the capabilities of AEs by incorporating the concept of list decoding from classical coding theory.

### Problem Definition

The problem of reliable transmission over a noisy channel can be formally defined as follows. A message word \( \mathbf{u} = [u_1, u_2, \ldots, u_K] \) of \( K \) bits, where each \( u_i \) takes binary values from \{0, 1\}, is encoded using an encoder neural network with an encoding function \( f_{\theta}(\cdot) \) to obtain a real-valued codeword \( \mathbf{x} = [x_1, x_2, \ldots, x_N] = f_{\theta}(\mathbf{u}) \), where \( \theta \) denotes the weights of the encoder neural network and \( N \) denotes the code length. A power normalization block is applied to \( \mathbf{x} \) to ensure that the codeword has zero mean and unit variance code symbols, i.e., \( E(x_i) = 0 \) and \( E(x_i^2) = 1 \) for \( i = 1, 2, \ldots, N \). The codeword \( \mathbf{x} \) is then transmitted over the channel.

The channel introduces noise to the codeword, producing a noisy version \( \mathbf{y} = [y_1, y_2, \ldots, y_N] \), where each \( y_i \) takes real values. The channel can be modeled by a transition probability density function (pdf) \( W_N(\mathbf{y}|\mathbf{x}) \). For the AWGN channel, the output \( y_i = x_i + w_i \), where \( w_i \) is a Gaussian random variable with zero mean and variance \( \sigma^2 \). The decoder network receives the channel output vector \( \mathbf{y} \) and applies a decoding function \( g_{\phi}(\cdot) \) to produce the decoded message word \( \hat{\mathbf{u}} = [\hat{u}_1, \hat{u}_2, \ldots, \hat{u}_K] = g_{\phi}(\mathbf{y}) \), where \( \phi \) denotes the weights of the decoder neural network. The goal is to minimize the block error rate (BLER) or bit error rate (BER) for different levels of channel impairment, such as signal-to-noise ratio (SNR) defined as \( 10 \log_{10} \left( \frac{1}{\sigma^2} \right) \) for the AWGN channel.

### List Autoencoder (listAE)

#### General Framework

The list autoencoder (listAE) framework is designed to address the limitations of traditional AEs by allowing the decoder to output a list of \( L \) candidate message words instead of a single candidate. This approach is inspired by the concept of list decoding in classical coding theory, where the decoder outputs a list of potential message words, and a selection mechanism is used to choose the correct one. Figure 2 illustrates a general listAE with a list size \( L \). A conventional AE is a special case of listAE with a list size of \( L = 1 \).

During the testing phase, the decoder must output a single candidate \( \hat{\mathbf{u}} \) from the list. This is achieved using a selection process, such as a genie-aided (GA) decoder or cyclic redundancy check (CRC)-aided (CA) decoder. The GA decoder outputs \( \hat{\mathbf{u}} = \mathbf{u} \) if \( \mathbf{u} \) is equal to one of the rows in the list, otherwise it outputs a randomly chosen row from the list. Mathematically, this can be expressed as:

\[ \hat{\mathbf{u}} = \begin{cases} 
\mathbf{u} & \text{if } \mathbf{u} \in \{\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_L\} \\
\mathbf{u}_r & \text{otherwise}
\end{cases} \]

where \( r \) is a random number chosen uniformly from 1 to \( L \).

During the training phase, the output list \( \mathbf{u}^{(list)} \) is made to take real values between zero and one, for example, by passing through a Sigmoid activation function. In the testing phase, the outputs are rounded to the nearest integer to give binary values. The performance metric to optimize is the BLER, which is calculated between the transmitted message word \( \mathbf{u} \) and the selected candidate \( \hat{\mathbf{u}} \).

#### Loss Function

A key challenge in designing the listAE is defining a loss function that reflects the performance of the GA decoder without explicitly modeling the genie operation. The proposed loss function is designed to minimize the distance between the transmitted message word and the closest candidate in the list. The loss function is defined as:

\[ \text{loss}(\mathbf{u}^{(list)}, \mathbf{u}) = \min_{l \in \{1, 2, \ldots, L\}} \rho(\mathbf{u}_l, \mathbf{u}) \]

where \( \rho \) is the average binary cross-entropy (BCE) loss function, which takes two vectors \( \mathbf{x} \) and \( \mathbf{x}' \) of length \( K \):

\[ \rho(\mathbf{x}, \mathbf{x}') = -\frac{1}{K} \sum_{i=1}^K \left[ x_i \log(x_i') + (1 - x_i) \log(1 - x_i') \right] \]

The min function is non-differentiable at points where an equality holds between the input arguments, but these points occur with zero probability, so they do not pose a significant issue for backpropagation during training.

#### Cyclic Redundancy Check (CRC)-Aided Decoding

For practical implementation, the functionality of the GA decoder is emulated using CRC. A CRC of length \( Z \) bits is generated using a polynomial \( g(x) = g_0 + g_1 x + \ldots + g_Z x^Z \). The CRC bits are appended to the message word to form a length-\( K \) vector \( \mathbf{u} \) as the encoder input. At the decoder side, each candidate in the list is checked for passing the CRC equations. Among the candidates that pass the CRC, one is randomly chosen as the final output of the decoder.

### Incremental Redundancy Autoencoder (IR-AE)

#### Architecture

The incremental redundancy autoencoder (IR-AE) is a specific architecture within the listAE framework. The encoder of IR-AE is similar to the Turbo-AE architecture, and the decoder relies on information exchange between decoding blocks. A rate-1/\( n \) IR-AE uses \( n \) encoding blocks applied to interleaved length-\( K \) message words to produce a length-\( N \) codeword \( \mathbf{x} = [x_1, x_2, \ldots, x_n] \) after power normalization. The IR-AE decoder consists of \( I \) iterations, where each iteration involves a series of decoding blocks that take subsets of the received vector \( \mathbf{y} \) and a list matrix as input and output an updated list matrix.

The IR-AE architecture is designed to allow more powerful codes with smaller rates to attempt decoding the message word based on an improved list matrix provided by previous weaker codes with higher rates. In this paper, we primarily focus on a rate-1/3 IR-AE. The detailed encoder and decoder architecture and training methodology are as follows:

1. **Rate-1/3 IR-AE Encoder**: The encoder is identical to the rate-1/3 Turbo-AE encoder. The output is a length-\( N = 3K \) codeword with normalized power \( \mathbf{x} = [x_1, x_2, \ldots, x_n] \).

2. **Decoder**: The decoder consists of \( I \) iterations. At iteration \( i \), a series of decoding blocks take subsets of \( \{y_1, y_2, \ldots, y_n\} \) and a list matrix as input and output an updated list matrix. The same architecture is replicated in every iteration, but with independent learnable weights. The list matrix \( P_I \), output by iteration \( I \), is passed through a Sigmoid function to give the output list of message word candidates.

3. **Power Normalization**: The output \( \mathbf{b} = [b_1, b_2, \ldots, b_3] \) is given to a power normalization block to meet the power constraint requirements. Batch-wise normalization is used, where each codeword is normalized as \( \mathbf{x} = \mathbf{b} - \mu \gamma \), where \( \mu \) is the mean and \( \gamma \) is the standard deviation of the code symbols in the batch.

4. **Training Methodology and Hyperparameters**: The model is trained for a maximum of 500 epochs. At each epoch, the encoder is trained \( T_{\text{enc}} \) times while freezing the weights of the decoder, and then the decoder is trained \( T_{\text{dec}} \) times while freezing the weights of the encoder. A batch size of \( B \) is used, and for each training, a set of \( B \) randomly generated message words of length \( K = 100 \) are generated and encoded by the encoder network. Noise vectors of length \( N \) are generated and added to the codewords. A fixed SNR is used for training the encoder, while a range of SNRs is used to train the decoder. The hyperparameters are detailed in Table I.

### Experimental Results

The performance of the listAE with both Turbo-AE and IR-AE architectures is evaluated and compared to classical codes. For IR-AE, the hyperparameters are given in Table I. The parameters for Turbo-AE are the same as the relevant blocks of the IR-AE. Figures 4 and 5 show the training loss trajectories for different list sizes. When the list size is increased from 8 to 64, the converged value of the loss drops significantly for Turbo-AE, while the change is smaller for IR-AE. This suggests an advantage of IR-AE over Turbo-AE at smaller list sizes. The test loss generally decreases with larger list sizes, although this trend is not consistent at every epoch due to the increased model size.

Figure 6 demonstrates the BLER of the List Turbo-AE and List IR-AE under GA decoding for different list sizes. The performance trends align with the test loss trajectories, indicating that IR-AE outperforms Turbo-AE for smaller list sizes, while both architectures have similar performance for larger list sizes. The improvement in IR-AE comes at the cost of a larger decoding network size.

Next, the performance of the List IR-AE under CA decoding is evaluated and compared to classical codes and Turbo-AE. The code dimensions for Turbo-AE and polar codes are \( (N = 300, K = 100) \). For IR-AE, a length-8 CRC generated by the polynomial \( g(x) = 1 + x^2 + x^4 + x^6 + x^7 + x^8 \) is appended to the \( K = 92 \) message bits before encoding. The comparison is made using \( E_b / \sigma^2 \) instead of SNR to account for the rate reduction caused by the CRC. Figure 7 shows that the List IR-AE with a list size of 64 outperforms Turbo-AE and polar codes at BLERs smaller than 0.03, achieving a coding gain of up to 0.5 dB over Turbo-AE and 0.3 dB over polar codes at high SNRs.

### Conclusion

In summary, the present invention introduces the list autoencoder (listAE) framework, which extends the capabilities of traditional autoencoders by allowing the decoder to output a list of candidate message words. The invention includes a specific loss function designed to optimize the performance of a genie-aided (GA) decoder and a more general architecture, incremental redundancy autoencoder (IR-AE), which demonstrates improved performance over existing architectures, especially for smaller list sizes. The experimental results show that the List IR-AE outperforms Turbo-AE and polar codes, making it a promising approach for designing AEs that can compete or outperform classical codes. Future research directions include refining the loss function to better reflect the performance under cyclic redundancy check (CRC)-aided list decoding and exploring other AE architectures for further improvements.