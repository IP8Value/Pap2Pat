# DESCRIPTION

## STATEMENT REGARDING PRIOR DISCLOSURES BY THE INVENTOR OR A JOINT INVENTOR

This application claims the benefit of priority to U.S. Provisional Application No. 63/XXXXXXX, filed on [DATE], which is incorporated herein by reference in its entirety. The inventors have not made any public disclosures of the invention prior to the filing of this application.

## BACKGROUND

Event stream data is collected to explore the dynamics and behavior of a wide variety of systems, including social networks, biochemical networks, electronic health records, and computer logs in data centers. In a multivariate event stream, events of multiple types (labels) occur at irregularly spaced time stamps on a common timeline. Event models seek to capture the joint stochastic dynamics of such event streams. Multivariate point processes and conditional intensity functions provide the mathematical framework for capturing event dynamics. The model fitting task for event stream data has a long history in machine learning, including prior work in temporal point process modeling. Various models have been proposed to capture history-dependent arrival rates, including graphical event models, forest-based point processes, piecewise-constant conditional intensity models, and Poisson networks. However, these approaches often make assumptions about the parametric form of the corresponding model, which can be challenging in practice without first-hand knowledge of the underlying data-generating process.

On the other hand, researchers have also proposed sequential deep learning techniques for event data sets, such as recurrent marked temporal point processes and neural Hawkes processes. These models use a recurrent neural network to capture the history dependency of the conditional intensity function. While these methods are semi-parametric and can be powerful and expressive, they may not adequately learn general-purpose forms of history dependence, such as history dependence that doesn't involve exponential decay or processes involving time lags with delayed excitation or inhibition.

## SUMMARY

The present invention provides a non-parametric deep learning approach to model multivariate event data sets in continuous time. The invention seeks to learn history-dependent conditional intensity functions in a fully data-driven, non-parametric manner, using only network weights and activation functions, and via learning a suitable representation of all available (strict) histories. The invention introduces a deep learning model that reinforces the negative evidence from each successive inter-event duration and develops an efficient multi-scale, multi-channel internal state representation. Additionally, the invention proposes a spatiotemporal attention model to capture the most influential histories and points in time for determining the instantaneous arrival rate for any chosen label.

The key contributions of the invention include:
1. A simple, yet effective, non-parametric way to approximately capture the continuous-time variation of historical influence on conditional intensity by exploiting the negative evidence from each successive inter-event duration.
2. An efficient multi-scale, multi-channel internal state representation that aligns the architecture with graphical event models.
3. A spatiotemporal attention model to capture the most influential histories and points in time for determining the instantaneous arrival rate for any chosen label.

The resulting multi-channel neural graphical event model (MCN-GEM) demonstrates non-trivial gains in evaluating point process log likelihood estimates on test data over state-of-the-art models.

## DETAILED DESCRIPTION

### Continuous Time in Deep Event Models

Given a dataset \( D \) consisting of event epochs \((l_i, t_i)\) where \( t_i \) is the occurrence time of the \( i \)-th event and \( l_i \) is an event label from a finite set \( L \) of possible labels, the invention aims to train a deep neural network to produce the instantaneous conditional intensity \( \lambda_k^t | H_t \) for each event type \( k \). The invention uses a sequence modeling approach with recurrent neural networks (RNNs) and long short-term memory (LSTM) cells. The sequence of tokens fed into the LSTM network is the temporally ordered event sequence \( D \), where each token corresponds to an event arrival, i.e., a label and a timestamp. Each token is represented in the raw input as a concatenation of its one-hot encoded event label and its continuous-valued timestamp.

The internal states of the recurrent LSTM cell evolve in response to each current raw input. The hidden state \( h_k^i \) for each event label \( k \) in the augmented set \( L \) is computed as follows:
\[ h_k^i = LSTM(Emb(l_i), t_i, h_{k-1}, c_{k-1}), \forall k \in L \]
where \( Emb \) denotes the embedding matrix for label \( l_i \). The embedding consists of one embedding layer on top of one-hot encoding of labels. \( h_k^{-1} \) is initialized to be all-zero vectors. For practical computation reasons, the LSTM parameters are shared among different event labels \( k \).

### Modeling Continuous Time History with Negative Evidence

The invention introduces the concept of "fake epochs" to reinforce the negative evidence of no observable events within each inter-event dead-space. An auxiliary \((M + 1)\)-th label is introduced into the label set \( L \), referred to as the "fake label." Within each inter-event interval, a certain number \( K \) of fake epochs (with label \( M + 1 \)) are spread uniformly in time over the interval. These fake event epochs participate in the recurrent computations like any other real event epoch, allowing the LSTM dynamics to further evolve the internal states within each inter-event interval.

The resulting finer sequence of internal states is a summary of all the event trace history as well as the passage of time in the intervening dead-space intervals. The fake event epochs also allow the computation of the integral terms in the log-likelihood function using a numerical quadrature procedure with the fake epoch timestamps as sampling time points.

### A Multi-Channel View for Spatial and Temporal Attention

In multivariate event data, different labels may have different arrival rates and may mutually influence the arrival rates of each other. The invention achieves this by associating each label with a corresponding partition of a single hidden state vector. The hidden state dimension is chosen to be an integer multiple of the number of labels, i.e., \( m(M + 1) \) for some positive integer \( m \), effectively realizing an \( m \)-dimensional hidden state for each label in \( L \).

These label-specific sub-vectors are selectively channeled for computing the label-specific rates through the network layer. The multi-channel view also enables modeling temporal and spatial attention within the \( \lambda \)-network. Spatial attention models inter-label dependence, while temporal attention models the lagged dependence on parental event history. This is achieved by maintaining a memory bank \( M \) of historical label-specific hidden states that span the most recent \( J \) event arrivals.

The raw hidden states in the memory bank are combined using an attention mechanism into a net hidden state that becomes input for the \( \lambda \)-network. The net attentive hidden state is given as:
\[ h_{\text{net},k}^i = \sum_{m=1}^{JM} \alpha_{k,im} h_m, \forall k \in L \]
where \( c_i \) is the context vector at epoch \( i \), computed as a weighted average of the raw hidden states in \( M_i \). The weighting is done by an alignment vector \( \alpha_{k,im} \):
\[ \alpha_{k,im} = \frac{\exp(e_{k,im})}{\sum_{m'=1}^{JM} \exp(e_{k,im'})} \]
where \( e_{k,im} \) is derived by comparing the current raw hidden state \( h_k^i \) to each raw hidden state \( h_m \) in the memory bank at time \( i \):
\[ e_{k,im} = v^\top \tanh(W_1 h_k^i + W_2 h_m) \]

### Training

The invention uses two feed-forward layers to learn the intensity rate \( \lambda_k^{t_{i+1}} \) given the net hidden state \( h_{\text{net},k}^i \) and time interval \( \Delta t_{i+1} = t_{i+1} - t_i \):
\[ \lambda_k^{t_{i+1}} = \sigma_2(f_2(\sigma_1(f_1(h_{\text{net},k}^i, \Delta t_{i+1})))) \]
where \( f_1 \) and \( f_2 \) are feed-forward neural layers, and \( \sigma_1 \) and \( \sigma_2 \) are activation functions (ReLU and softplus, respectively). The softplus activation ensures a positive conditional intensity.

To train the MCN-GEM, the invention uses the same log-likelihood function as in the standard point process model, with the assumption of constant intensity in between two consecutive events, real or fake:
\[ \mathcal{L}(D) = \sum_{i=1}^N \log \lambda_{l_i}^{t_i} - \int_0^T \sum_{k \in L} \lambda_k^t dt \]
where \( \Delta t_i = t_i - t_{i-1} \) is the time interval since the last event.

In addition, the invention adds two regularization terms to help with generalization. The first is the target prediction loss \( L_p \) of the next event label \( l_{i+1} \) given \( \lambda_k^{t_{i+1}} \), for which a cross-entropy loss is used. The second term \( L_w \) penalizes the \( L_2 \)-norm of the weights on \( f_1 \) and \( f_2 \). The overall regularized objective for training is:
\[ \mathcal{L}_{\text{total}} = \mathcal{L}(D) - \lambda_1 L_p - \lambda_2 L_w \]

### Empirical Evaluation

The invention was empirically evaluated on synthetic datasets generated from proximal graphical event models (PGEMs) and real-world datasets, including the Integrated Crisis and Early Warning System (ICEWS) political event dataset and the MIMIC-II healthcare dataset. The results demonstrated that the MCN-GEM outperformed other state-of-the-art models by a significant margin, particularly when using fake epochs to reinforce negative evidence.

### Ablation Study

An ablation study was conducted to evaluate the effectiveness of different components of the proposed model, including the multi-channel vs. single-channel modeling, the usage of attention vs. no attention, and the impact of attention memory bank sizes. The results showed that the multi-channel modeling and the use of attention significantly improved the performance of the model. The optimal memory bank size varied depending on the dataset, with a size of 3 being the best performing in the ICEWS Argentina dataset.

### Impact of the Number of Fake Epochs

The impact of the number of fake epochs on the performance of the MCN-GEM was studied on the ICEWS Argentina dataset. Introducing one fake epoch resulted in a large gain in log-likelihood, while introducing more fake epochs led to minimal improvements and could even degrade performance. The introduction of fake epochs allowed the model to better approximate the variation of hidden states in continuous time, leading to a sharper intensity rate landscape.

### Graph Visualization of the Attention

The attention mechanism in the MCN-GEM enables the visualization of the relationships among the variables as a graph. The average attention of all event channels across time was used to compute the graph connections, and the resulting graph was visualized for the ICEWS Argentina dataset. The visualization revealed meaningful chains of events, such as the cooperation between the head of the Argentina government and the Brazil government leading to internal conflicts and subsequent citizen unhappiness.

### Conclusion

The invention introduces a new multi-scale multi-channel neural graphical event model (MCN-GEM) with two-dimensional attentions for modeling event sequences. The model exploits the negative evidence of no observable events in each successive inter-event duration by introducing fake epochs, eliminating the need to assume specific functional forms. This makes the approach practically appealing for approximately capturing the variation of hidden states in continuous time in a non-parametric manner. The model combines the framework of graphical event models and the modeling power of deep neural networks, demonstrating significant performance gains over state-of-the-art models on synthetic and benchmark datasets.