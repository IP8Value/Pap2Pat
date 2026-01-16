Here is the complete patent application following the provided outline and incorporating the invention described in the research paper:

# DESCRIPTION  

## STATEMENT REGARDING PRIOR DISCLOSURES BY THE INVENTOR OR A JOINT INVENTOR  

The inventor has not made any prior public disclosures of the claimed invention that would preclude the granting of a patent. All disclosures of the invention have been made under circumstances that preserve the novelty and non-obviousness of the claimed subject matter for patentability purposes.  

## BACKGROUND  

The field of the invention relates generally to machine learning systems for modeling multivariate event stream data, and more particularly to non-parametric deep learning approaches for learning history-dependent conditional intensity functions in continuous-time event streams.  

Event stream data is collected across numerous domains including social networks, biochemical networks, electronic health records, and computer system logs. Such data consists of events of multiple types occurring at irregular time intervals on a common timeline. Existing approaches for modeling these multivariate event streams suffer from significant limitations. Traditional methods rely on parametric assumptions about the form of history-dependent arrival rates, which may not match the true underlying data generating process. While some semi-parametric neural network approaches have been proposed, these still incorporate functional assumptions about how network states translate to event arrival rates, typically using exponential decay patterns similar to Hawkes processes.  

These existing techniques fail to adequately capture more complex history-dependence patterns that may involve piecewise constant intensities, delayed excitation or inhibition effects, or varying time scales across different event types. There exists a need for a fully non-parametric approach that can learn arbitrary forms of history-dependence directly from data without imposing restrictive parametric assumptions. Additionally, current methods do not effectively utilize the information contained in inter-event intervals - the periods between observable events - which represent important negative evidence about event dynamics.  

## SUMMARY  

The present invention provides a multi-channel neural graphical event model (MCN-GEM) that addresses these limitations through several key innovations. First, the invention introduces a novel approach to modeling continuous-time history dependence by incorporating "fake epochs" between actual events. These artificial data points reinforce the negative evidence from inter-event intervals and allow the model to better capture the continuous variation of historical influence on conditional intensity functions.  

Second, the invention implements a multi-scale, multi-channel internal state representation that aligns with graphical event model frameworks. This architecture maintains separate hidden state channels for each event type, enabling the model to handle different base rates and influence patterns across event types.  

Third, the invention incorporates a spatiotemporal attention mechanism that identifies which historical events (spatial attention) and which points in time (temporal attention) are most influential for determining current event arrival rates. This attention mechanism operates on a memory bank of recent hidden states to selectively weight historical information when computing current conditional intensities.  

The complete system combines these components into an end-to-end trainable neural network that learns to predict event arrival rates in a fully data-driven manner. The model is trained using a composite objective function that includes both the event stream likelihood and regularization terms for improved generalization. Experimental results demonstrate that this approach achieves superior performance in modeling synthetic and real-world event streams compared to existing state-of-the-art methods.  

## DETAILED DESCRIPTION  

The detailed description begins with an overview of the system architecture, followed by in-depth explanations of each key innovation: the fake epoch approach for modeling negative evidence, the multi-channel state representation, and the spatiotemporal attention mechanism. Implementation details and training procedures are then provided.  

**System Architecture Overview**  

The MCN-GEM system takes as input a sequence of timestamped events D = {(l_i, t_i)} where each l_i represents an event label from a finite set L and each t_i represents the event's occurrence time. The system outputs a set of conditional intensity functions {λ_k(t|H_t)} for each event type k ∈ L, where H_t represents the strict history of all events occurring before time t.  

At the core of the system is a recurrent neural network (typically using LSTM cells) that processes the event sequence while maintaining an evolving internal state representation. The network's hidden states are partitioned into multiple channels, with each channel corresponding to a particular event type. This multi-channel architecture enables the model to maintain separate representations for different event types while still allowing cross-type influences through the attention mechanism.  

**Modeling Negative Evidence with Fake Epochs**  

A key innovation of the invention is the introduction of artificial "fake epochs" between actual observed events. These fake epochs serve to reinforce the negative evidence contained in inter-event intervals - the periods when no events are observed.  

The system augments the original label set L with an additional fake label (M+1). During processing, it inserts K fake epochs uniformly within each inter-event interval. These fake epochs participate in the recurrent computations just like real events, causing the internal states to evolve continuously across time rather than remaining fixed between events.  

This approach provides several advantages:  
1. It allows the model to better approximate the continuous-time variation of conditional intensity functions between events.  
2. It enables more accurate computation of the integral terms in the event stream likelihood function by providing additional sampling points.  
3. It serves as a form of adversarial training, with the fake epochs acting as negative samples that help regularize the model.  

The number of fake epochs K is a hyperparameter that can be tuned based on the characteristics of the target dataset. Empirical results show that using just one fake epoch per interval provides most of the benefit, with diminishing returns from additional fake epochs.  

**Multi-Channel State Representation**  

The invention implements a multi-channel architecture where the hidden state vector is partitioned into separate sub-vectors for each event type (including the fake label). Specifically, if the hidden state dimension is m(M+1), then each of the M+1 labels gets its own m-dimensional subspace.  

This design provides several benefits:  
1. It allows the model to maintain separate representations for different event types, accommodating their potentially different base rates and influence patterns.  
2. It enables selective processing of label-specific information when computing conditional intensities.  
3. It provides a natural framework for implementing spatial attention across event types.  

The multi-channel architecture is particularly well-suited for graphical event models, where different parent events may influence child events in type-specific ways.  

**Spatiotemporal Attention Mechanism**  

The invention incorporates a two-dimensional attention mechanism that operates both spatially (across event types) and temporally (across historical time points). This attention mechanism works as follows:  

1. The system maintains a memory bank M_i containing the raw hidden states from the most recent J event epochs (both real and fake) for each event type.  
2. For each current event, the system computes attention weights by comparing the current hidden state to each historical state in the memory bank.  
3. These attention weights are used to compute a context vector as a weighted sum of historical states.  
4. The context vector is combined with the current hidden state to produce a "net hidden state" that captures the most relevant historical information.  

The attention mechanism provides several advantages:  
1. It allows the model to identify which historical events (spatial attention) and which time points (temporal attention) are most relevant for predicting current event rates.  
2. It enables visualization of event-type relationships through the learned attention patterns.  
3. It provides flexibility in modeling different forms of history-dependence, including delayed excitation/inhibition effects.  

**Implementation and Training**  

The complete system is implemented as follows:  

1. Each input event is represented as a concatenation of its one-hot encoded label and timestamp.  
2. The recurrent network processes this input sequence while maintaining its multi-channel hidden state.  
3. At each step (for both real and fake epochs), the attention mechanism computes a net hidden state.  
4. The net hidden state is passed through feedforward layers (the λ-network) to produce the conditional intensity vector.  

The model is trained using a composite objective function:  

L_total = LL(D) + αL_p + βL_w  

Where:  
- LL(D) is the event stream log-likelihood  
- L_p is a prediction loss (cross-entropy for next event type)  
- L_w is an L2 weight regularization term  
- α and β are weighting hyperparameters  

The log-likelihood term is computed using the intensities at both real and fake epochs, with the integral terms approximated via numerical quadrature between consecutive epochs.  

Experimental results demonstrate that this approach achieves superior performance compared to existing methods across synthetic and real-world datasets, with particularly strong gains when fake epochs are incorporated. The attention mechanism also provides interpretable insights into event-type relationships through visualization of the learned attention patterns.  

The invention represents a significant advance in event stream modeling by combining the flexibility of non-parametric deep learning with the structured approach of graphical event models. Its innovations in handling negative evidence, multi-channel representations, and spatiotemporal attention provide a powerful framework for learning complex history-dependent patterns in continuous-time event data.