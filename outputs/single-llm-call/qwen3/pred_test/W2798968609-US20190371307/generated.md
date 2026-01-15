# DESCRIPTION

## FIELD

- relate to automated spoken language understanding

The present invention relates to automated spoken language understanding systems designed to interpret human utterances in natural language, particularly in conversational and interactive environments such as voice-activated assistants, customer service chatbots, and smart home control interfaces. Specifically, the invention concerns the automated identification of user intent and extraction of semantic slot values from spoken or text-based input sequences without reliance on manually annotated word-level alignments. Traditional approaches to spoken language understanding require extensive labeling of individual words with slot tags, such as BIO (Begin, Inside, Outside) schemes, which are labor-intensive, domain-specific, and brittle when applied to out-of-vocabulary terms or noisy speech recognition outputs. The invention overcomes these limitations by introducing a neural generative architecture that jointly models intent classification and slot-filling through a unified, end-to-end trainable framework capable of dynamically generating or copying slot values directly from the input utterance, thereby eliminating the need for pre-defined slot value dictionaries and reducing dependency on annotated training data.

## BACKGROUND

- introduce spoken language understanding

Spoken language understanding is a critical component of conversational artificial intelligence systems, enabling machines to extract semantic meaning from human utterances by identifying the user’s intent and the associated parameters, or slot values, that qualify that intent. For example, in a restaurant reservation system, the utterance “I want to book a table at the Blue Garden for four people tonight” requires the system to recognize the intent as “book_table” and extract slot values such as “restaurant_name=Blue Garden,” “party_size=four,” and “time=tonight.” Accurate interpretation of these components is essential for downstream task execution, such as querying databases, scheduling appointments, or initiating transactions. Without precise understanding, systems risk misinterpretation, failed commands, or user frustration.

- describe intent determination and slot filling

Intent determination refers to the classification of the overall purpose behind a user’s utterance, while slot filling involves identifying and extracting the specific contextual parameters that modify or qualify that intent. These two tasks are traditionally treated as separate processes: intent classification is often performed using supervised classifiers such as support vector machines or convolutional neural networks, while slot filling is approached as a sequence labeling problem using conditional random fields or recurrent neural networks with token-level annotations. However, this modular pipeline introduces error propagation, as inaccuracies in slot extraction directly impact the reliability of intent classification and vice versa.

- motivate joint training model

Joint training of intent and slot tasks has been proposed to mitigate error accumulation and improve overall performance by allowing shared representations to inform both components simultaneously. However, existing joint models still rely on fully annotated training corpora with word-level slot tags, which are expensive to produce and difficult to scale across domains. Moreover, they are fundamentally constrained by fixed vocabularies of slot values, rendering them ineffective when encountering previously unseen entities such as novel restaurant names, street addresses, or product titles.

- explain attention mechanism

Attention mechanisms have emerged as a powerful tool in sequence-to-sequence learning, allowing the decoder to selectively focus on relevant portions of the input sequence during generation. By computing a dynamic weighting over encoder hidden states, attention enables the model to align input tokens with output predictions without requiring explicit alignment labels. This has proven particularly effective in machine translation and summarization, where long-range dependencies and variable-length inputs are common. In the context of spoken language understanding, attention allows the model to associate phrases in the utterance with corresponding slot values, even when those values are not pre-defined in a vocabulary.

- describe slot-filling operation

Slot-filling operation traditionally involves assigning a label to each word or subword unit in an utterance indicating its role in a semantic frame. For instance, in the phrase “find a coffee shop near Central Park,” the words “Central Park” are tagged as the value of the “location” slot. This requires a one-to-one mapping between input tokens and output labels, which is problematic when slot values consist of multi-word phrases, rare entities, or phrases not seen during training. Furthermore, in real-world applications, speech recognition errors, dialectal variations, and paraphrasing introduce noise that breaks the alignment between surface form and semantic annotation.

- limitations of prior-art SLU systems

Prior-art systems suffer from three fundamental limitations. First, they require large volumes of manually annotated training data with precise word-level slot labels, which are costly and time-consuming to produce. Second, they are unable to generalize to slot values not present in the training vocabulary, leading to poor performance on out-of-vocabulary (OOV) entities such as newly opened businesses or uncommon proper nouns. Third, they rely on handcrafted rules or heuristics to infer alignments between sentence-level annotations and word sequences, which are domain-specific and fail under noisy conditions such as those produced by automatic speech recognition systems.

- motivate improvements to SLU systems

There is a pressing need for a more robust, scalable, and adaptive approach to spoken language understanding that can operate effectively with minimal supervision, handle arbitrary slot values without predefined dictionaries, and maintain high accuracy even in the presence of speech recognition noise or data scarcity. The invention addresses these challenges by introducing a neural generative model that unifies intent classification and slot filling into a single, end-to-end trainable architecture capable of both generating slot values from a limited vocabulary and directly copying them from the input utterance, thereby overcoming the OOV problem and eliminating the need for manual alignment.

## SUMMARY

- introduce neural generative model

The invention introduces a neural generative model for spoken language understanding that predicts user intent and slot values from raw utterances without requiring word-level annotations or predefined slot value dictionaries. This model leverages a hybrid architecture combining a sequence-to-sequence decoder with a pointer network, enabling it to dynamically choose between generating slot values from a fixed vocabulary or copying them directly from the input utterance. This dual-mode operation allows the system to handle both common, enumerable slot values and rare, unenumerable entities such as proper names, addresses, or product identifiers with equal efficacy.

- describe Seq2Seq learning

The model employs a sequence-to-sequence learning framework, wherein an encoder processes the input utterance as a sequence of word embeddings and produces a contextual representation of the entire input. A decoder then generates the output slot value sequence token by token, using the encoder’s hidden states to inform each prediction. Unlike traditional Seq2Seq models that rely solely on a fixed target vocabulary, this invention extends the decoder’s output space to include the possibility of referencing words directly from the input sequence, thereby enabling the model to reproduce previously unseen slot values without prior exposure during training.

- incorporate pointer network

The pointer network component of the invention allows the decoder to select a position within the input utterance and copy the word at that position as a slot value. This mechanism is particularly effective for handling out-of-vocabulary entities, as it bypasses the need to generate them from a fixed lexical set. The pointer network operates in parallel with the standard Seq2Seq decoder, sharing the same attention mechanism and encoder states, but producing a probability distribution over input positions rather than over a vocabulary.

- handle OOV slots

By integrating the pointer network, the invention effectively mitigates the out-of-vocabulary problem that plagues conventional slot-filling systems. When a slot value such as “The Spice Route” or “123 Maple Street” is encountered during inference, the model does not attempt to generate it from a limited dictionary but instead identifies its occurrence in the input utterance and copies it directly. This eliminates the need for manual expansion of slot vocabularies and enables seamless adaptation to new domains with minimal training data.

- predict slot values

The model predicts slot values as sequences of tokens, where each token is either generated from a predefined slot vocabulary or copied from the input utterance. This allows for the accurate extraction of multi-word slot values, such as “next Thursday at 7 PM” or “Italian cuisine with gluten-free options,” without requiring segmentation or alignment preprocessing. The output sequence is generated autoregressively, with each predicted token conditioned on the previous ones and the full context of the input.

- use neural generative model without annotations

The system requires only sentence-level semantic annotations—such as “intent=book_table, restaurant_name=Blue Garden, party_size=four”—and does not require any word-level labeling. This dramatically reduces the annotation burden and enables deployment in domains where labeled data is scarce or unavailable. The model learns to infer the alignment between sentence-level annotations and input words through the joint optimization of intent classification and slot value prediction.

- handle OOV problem

The invention solves the OOV problem by allowing the model to reference any word in the input utterance as a potential slot value, regardless of whether it appeared in the training corpus. This is achieved through the pointer network’s ability to assign non-zero probability to input positions, even for rare or unseen words, thereby enabling robust performance on novel entities without retraining or dictionary updates.

- identify user intent and slot types

The system simultaneously identifies the user’s intent and assigns slot types to extracted values using a unified neural architecture. Intent classification is performed based on the final hidden state of the encoder, while slot types are inferred from the structure of the generated slot value sequence and its alignment with the input. This joint modeling ensures consistency between intent and slot assignments, reducing conflicts and improving overall accuracy.

- describe pipeline or joint framework

Unlike prior systems that treat intent and slot tasks as sequential or independent modules, the invention employs a fully joint framework in which both tasks are optimized under a single loss function. The encoder produces a shared representation used by both the intent classifier and the slot value decoder, ensuring that semantic information flows coherently between components. This eliminates error propagation and enables the model to leverage contextual cues from one task to improve performance in the other.

## DETAILED DESCRIPTION

- introduce patent application

The present patent application describes a novel system and method for automated spoken language understanding that enables accurate and robust extraction of user intent and slot values from natural language utterances without requiring word-level annotations or predefined slot value dictionaries. The system is implemented as a neural generative model that combines a bidirectional recurrent neural network encoder, an attention-based sequence-to-sequence decoder, and a pointer network to dynamically generate or copy slot values from the input utterance.

- reference drawings and descriptions

The invention is illustrated in a series of figures, including a system architecture diagram showing the flow of data from audio input to semantic output, a block diagram of the neural network components, and flowcharts depicting the training and inference processes. These figures are referenced throughout the description to clarify the structure and operation of the system.

- define Recurrent Neural Network (RNN)

A recurrent neural network is a class of artificial neural network designed to process sequential data by maintaining an internal state that captures information from previous elements in the sequence. In this invention, the RNN is implemented using gated recurrent units (GRUs), which efficiently model long-range dependencies while mitigating the vanishing gradient problem common in standard RNNs.

- describe RNN encoder/decoder network

The encoder is a bidirectional GRU that processes the input utterance from both forward and backward directions, producing a sequence of hidden states that encode contextual information for each word. The decoder is a unidirectional GRU that generates the output slot value sequence one token at a time, using the encoder’s hidden states and an attention mechanism to guide each prediction.

- explain encoder internal hidden state

The internal hidden state of the encoder at each time step represents a compressed, context-aware encoding of the input utterance up to that point. These hidden states are used by the attention mechanism to compute relevance scores between the decoder’s current state and each input word, enabling the model to focus on the most pertinent parts of the utterance during slot value generation.

- describe decoder output

The decoder produces, at each time step, a probability distribution over two possible sources of output: a fixed slot vocabulary and the positions in the input utterance. This dual-output distribution is combined using a trainable gating mechanism that determines whether to generate a word from the vocabulary or copy it from the input.

- introduce out-of-vocabulary (OOV) words

Out-of-vocabulary words are terms that do not appear in the predefined slot value dictionary and are therefore unrepresentable by traditional classification-based slot fillers. These include proper nouns, newly coined terms, and domain-specific entities such as restaurant names, street addresses, or product codes. The invention handles OOV words by allowing the pointer network to select them directly from the input utterance, thereby circumventing the need for lexical expansion.

- describe uni-directional and bi-directional RNNs

The encoder employs a bidirectional RNN to capture both past and future context for each word, while the decoder uses a unidirectional RNN to generate output tokens sequentially, conditioned only on previously generated tokens and the input context. This asymmetry ensures that the decoder maintains causal consistency while the encoder maximizes contextual richness.

- explain bi-directional RNN output

The output of the bidirectional encoder is a sequence of concatenated forward and backward hidden states, each representing a word in the context of its entire surrounding utterance. These states serve as the basis for the attention mechanism and are used by both the slot value decoder and the intent classifier.

- introduce attention mechanism

The attention mechanism computes a dynamic weighting over the encoder’s hidden states at each decoding step, allowing the decoder to selectively focus on the most relevant parts of the input utterance when predicting the next slot value. This mechanism is computed using a learned score function that measures compatibility between the decoder’s current state and each encoder state.

- describe score function

The score function is implemented as a feedforward neural network that takes as input the concatenation of the decoder’s current hidden state and an encoder hidden state, and outputs a scalar score representing their alignment. These scores are then normalized using a softmax function to produce attention weights.

- explain attention weights calculation

Attention weights are calculated by applying the softmax function to the output of the score function across all encoder positions. Each weight indicates the relative importance of the corresponding input word in generating the current output token. These weights are used to compute a context vector that summarizes the most relevant parts of the input.

- describe context vector and attention vector

The context vector is a weighted sum of the encoder’s hidden states, where the weights are the attention scores. It serves as a dynamic summary of the input relevant to the current decoding step. The attention vector is the vector of attention weights themselves, used to track which input positions are being referenced during generation.

- introduce pointer network

The pointer network is a specialized component that, at each decoding step, produces a probability distribution over the positions of the input utterance. Unlike the standard decoder, which generates tokens from a vocabulary, the pointer network selects an input word directly, enabling the model to copy rare or unseen slot values without prior exposure.

- describe pointer network output

The pointer network outputs a probability distribution over the input sequence indices, indicating the likelihood that each word in the utterance should be copied as the next slot value. This distribution is computed using the same attention mechanism as the Seq2Seq decoder, ensuring consistency in alignment.

- explain combining RNN decoder and pointer network outputs

The final prediction at each time step is a weighted combination of the probabilities generated by the Seq2Seq decoder and the pointer network. A trainable gate, computed from the decoder’s current state, the context vector, and the input embedding, determines the proportion of each source to use, allowing the model to adaptively switch between generation and copying based on context.

- introduce extended vocabulary

The extended vocabulary is defined as the union of the fixed slot vocabulary and all unique words appearing in the training utterances. This expanded set enables the model to generate slot values from a broader lexical base while retaining the ability to copy unseen words via the pointer network.

- describe SLU system 100

The spoken language understanding system 100 comprises an audio input device, a speech recognizer, a processor, a memory, and an output device. The system receives spoken or text-based input, converts it into a sequence of word embeddings, and processes it through a neural network architecture to extract intent and slot values. The output is used to trigger actions such as executing commands, initiating searches, or controlling devices.

- introduce audio input device 104

The audio input device 104 captures spoken utterances from a user and converts them into digital audio signals. These signals are transmitted to the speech recognizer for transcription into text.

- describe output device 112

The output device 112 may include a speaker, display, or actuator that responds to the system’s semantic interpretation. For example, it may play music, display search results, or adjust thermostat settings based on the inferred intent and slot values.

- introduce processor 128

The processor 128 executes programmed instructions stored in memory to perform the operations of the neural network, including encoding, attention computation, decoding, and slot value prediction.

- describe memory 132

The memory 132 stores the trained neural network parameters, the extended slot vocabulary database, the slot database, and the programmed instructions that implement the system’s functionality.

- introduce programmed instructions 134

The programmed instructions 134 comprise software modules that orchestrate the flow of data through the system, including speech recognition, intent classification, slot value generation, and command execution.

- describe speech recognizer 136

The speech recognizer 136 converts spoken input into a sequence of text tokens, which are then processed by the neural network. It may be implemented using a conventional automatic speech recognition engine or a neural acoustic model.

- introduce intent classifier 138

The intent classifier 138 is a neural network module that receives the final encoder hidden state and outputs a probability distribution over possible intents, such as “book_table,” “find_restaurant,” or “set_alarm.”

- describe RNN encoder 140

The RNN encoder 140 is a bidirectional GRU that processes the input token sequence and produces a sequence of contextualized hidden states for use by the attention mechanism and decoder.

- introduce slot word decoder 144

The slot word decoder 144 is the core component responsible for generating the sequence of slot values. It combines the outputs of the RNN decoder and the pointer network to produce a final probability distribution over the extended vocabulary and input positions.

- describe RNN decoder 146

The RNN decoder 146 is a unidirectional GRU that generates slot value tokens autoregressively, using the context vector and previous predictions to inform each step.

- introduce pointer network 148

The pointer network 148 computes a probability distribution over the input utterance positions, enabling the system to copy slot values directly from the input without relying on a fixed vocabulary.

- describe extended slot vocabulary database 162

The extended slot vocabulary database 162 contains all slot values observed during training, including both enumerable values (e.g., “Italian,” “high”) and words from the input utterances. It serves as the source for the Seq2Seq decoder’s generation component.

- introduce slot database 166

The slot database 166 maps slot types (e.g., “restaurant_name,” “time”) to their corresponding values and is used to structure the final semantic output for downstream applications.

- describe speech recognition process

The speech recognition process converts raw audio into a sequence of text tokens, which are then embedded and fed into the RNN encoder. The system is robust to minor transcription errors, as the pointer network can still copy correct words even if they are misrecognized.

- explain slot and intent classification

Slot and intent classification are performed jointly by the neural network, with the encoder providing a shared representation that informs both the intent classifier and the slot word decoder. This ensures semantic consistency and reduces error propagation.

- describe RNN encoder output

The RNN encoder output consists of a sequence of hidden states, each representing a word in the context of the entire utterance. These states are used by the attention mechanism to compute relevance scores for slot value generation.

- explain slot word decoder output

The slot word decoder output is a sequence of slot values, each generated either by copying from the input utterance or by generating from the extended vocabulary. The sequence is terminated when a stop token is predicted or a maximum length is reached.

- describe combining RNN decoder and pointer network outputs

The outputs of the RNN decoder and pointer network are combined using a trainable gate that determines, at each time step, whether to generate a word from the vocabulary or copy it from the input. This gate is computed from the decoder’s hidden state, the context vector, and the current input embedding.

- introduce final output of slot word decoder

The final output of the slot word decoder is a sequence of slot values, each associated with a slot type, forming a complete semantic frame that describes the user’s request.

- describe using slot words for slot-filling operations

Slot words are used to populate structured data fields that are then used to execute commands, query databases, or trigger actions. For example, the slot values “restaurant_name=Blue Garden” and “party_size=four” are used to book a table at the specified restaurant for the specified number of guests.

- describe memory components

The memory components store the trained neural network weights, the extended vocabulary, the slot database, and the programmed instructions that implement the system’s logic.

- describe extended slot vocabulary database

The extended slot vocabulary database is dynamically updated during training to include all unique words from the input utterances, ensuring that the system can generate or copy any term encountered during inference.

- describe slot database

The slot database contains predefined slot types and their permissible value formats, enabling the system to validate and structure the output of the slot word decoder.

- describe slot classifier

The slot classifier assigns a slot type to each predicted slot value based on its context and position in the sequence, using a lightweight neural network that operates on the decoder’s hidden states.

- describe RNN encoder and decoder

The RNN encoder and decoder operate in tandem to transform the input utterance into a semantic representation. The encoder captures global context, while the decoder generates structured output using attention and pointer mechanisms.

- describe attention mechanism

The attention mechanism dynamically aligns input words with output slot values by computing relevance scores between the decoder’s current state and each encoder hidden state, enabling precise localization of slot boundaries.

- describe pointer network

The pointer network enables the system to copy slot values directly from the input utterance, bypassing the limitations of fixed vocabularies and enabling robust handling of out-of-vocabulary entities.

- describe training process

The system is trained end-to-end using a combined loss function that penalizes incorrect intent predictions and slot value predictions, whether generated or copied. Training is performed using stochastic gradient descent with backpropagation through time.

- describe hyperparameters

Hyperparameters include the dimensionality of word embeddings (100), the size of hidden states (128), the dropout rate (0), and the learning rate (0.001), all optimized on a development set.

- describe GRU functions

Gated recurrent units update their hidden state using update and reset gates that control the flow of information, allowing the model to retain long-term dependencies while avoiding vanishing gradients.

- describe process 300 overview

Process 300 describes the overall operation of the system: receiving an input utterance, encoding it with a bidirectional RNN, computing attention weights, generating slot values via a hybrid decoder, and outputting a structured semantic frame.

- describe receiving input sequence

The system receives a sequence of word tokens derived from either spoken input transcribed by a speech recognizer or direct text input from a user interface.

- describe generating encoded output

The bidirectional RNN encoder processes the input sequence and generates a sequence of contextualized hidden states that represent the meaning of each word in its full utterance context.

- describe generating probability distribution outputs

At each decoding step, the system generates two probability distributions: one over the extended vocabulary and one over the input positions, which are combined to produce the final prediction.

- describe generating attention weights

Attention weights are computed by applying a softmax function to the alignment scores between the decoder’s current state and each encoder hidden state.

- describe generating probability distribution outputs for out-of-vocabulary words

For out-of-vocabulary words, the system assigns non-zero probability through the pointer network, which selects the word directly from the input utterance, thereby enabling accurate prediction without prior exposure.

- describe combining outputs of RNN decoder and pointer network

The outputs are combined using a sigmoid-gated linear combination, where the gate is learned from the decoder’s state, the context vector, and the input embedding.

- describe identifying slot word at each time step

At each time step, the system identifies the most probable slot word by selecting the token with the highest combined probability from the generator and pointer distributions.

- describe generating weighted combination of outputs

The weighted combination is computed as a convex combination of the generator and pointer probabilities, with the mixing coefficient learned during training.

- describe identifying slot label

A slot label is assigned to each predicted slot word by a slot classifier that maps the word and its context to a predefined slot type, such as “restaurant_name” or “time.”

- describe generating sequence of slot words

The system generates a sequence of slot words autoregressively, with each word conditioned on the previous ones and the input context, until a stop condition is met.

- describe stopping criteria for slot word decoder

The decoder stops when it predicts a special end-of-sequence token or when a maximum sequence length is reached, ensuring that the output remains bounded and interpretable.

- describe example of predicted slot words

For the utterance “I’d like to dine at The Spice Route tonight,” the system correctly predicts the slot value “The Spice Route” by copying it from the input, even though it was not present in the training vocabulary.

- describe performing command specified in input phrase

The system executes the command specified by the intent and slot values—for example, initiating a restaurant reservation with the identified name, party size, and time.

- describe identifying intent of input phrase

The intent is identified by applying a classifier to the final hidden state of the encoder, which encodes the overall semantic meaning of the utterance.

- describe using slot words as parameters for execution of command

The extracted slot words serve as parameters for downstream systems, such as database queries, API calls, or control signals for smart devices.

- describe generating search query

The system generates a structured search query from the intent and slot values—for example, “search for Italian restaurants near Central Park with vegan options”—and submits it to a backend service.

- describe controlling output device

The system sends control signals to an output device, such as a smart speaker or home automation hub, to perform actions like playing music, adjusting lighting, or unlocking doors.

- describe improving precision and recall of slot assignment process

By combining generation and copying, the system achieves higher precision by avoiding spurious predictions and higher recall by correctly identifying out-of-vocabulary slot values.

- describe F1 metric

The F1 metric is used to evaluate the overall performance of the slot-filling component, balancing precision and recall across all slot types and domains.

- describe results of slot value prediction

The system achieves state-of-the-art performance on benchmark datasets, outperforming prior methods in both in-vocabulary and out-of-vocabulary conditions.

- describe comparison with prior-art systems

Compared to CNN-based and SVM-based systems, the invention demonstrates superior performance, particularly under low-data and high-OOV conditions, with up to 15% higher F1 scores.

- describe advantages of system 100

The system requires no manual annotation of word-level slot labels, adapts seamlessly to new domains, handles arbitrary slot values, and operates robustly under noisy input conditions.

- describe variants of system 100

Variants include multi-turn dialogue support, integration with dialogue state tracking, and use of transformer-based encoders instead of RNNs.

- describe applications of system 100

Applications include virtual assistants, customer service bots, voice-controlled home automation, automotive infotainment systems, and healthcare appointment schedulers.

- describe modifications of system 100

Modifications include replacing GRUs with LSTMs, incorporating external knowledge bases, or using beam search for improved decoding.

- describe variations of system 100

Variations include training on multiple languages, supporting multilingual slot values, or integrating with speaker identification for personalized responses.

- describe improvements of system 100

Improvements include fine-tuning with reinforcement learning, incorporating confidence scores for uncertain predictions, or enabling active learning for iterative data collection.

- describe subsequent developments of system 100

Subsequent developments include extending the model to handle nested slots, multi-slot dependencies, and temporal reasoning over conversational history.

- describe unforeseen alternatives of system 100

Unforeseen alternatives include using the model for non-linguistic sequence labeling tasks, such as extracting entities from sensor logs or medical records.

- describe unanticipated modifications of system 100

Unanticipated modifications include deploying the system on edge devices with quantized weights, enabling real-time inference on low-power hardware.

- describe intended scope of claims

The intended scope of the claims encompasses all systems and methods that implement the joint neural generative model for spoken language understanding comprising a bidirectional encoder, an attention-based decoder, and a pointer network for out-of-vocabulary slot value prediction, regardless of the specific implementation details of the recurrent units, attention functions, or training procedures.