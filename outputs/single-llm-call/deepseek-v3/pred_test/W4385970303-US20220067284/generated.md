**PATENT APPLICATION**  

# **DESCRIPTION**  

## **CROSS REFERENCE(S)**  
The present application claims priority to U.S. Provisional Patent Application No. [XXXXXX], filed on [DATE], the contents of which are incorporated herein by reference in their entirety.  

## **TECHNICAL FIELD**  
The present invention relates generally to the field of natural language processing (NLP) and, more specifically, to systems and methods for controllable text summarization using neural networks. The invention enables dynamic control over generated summaries based on user-defined parameters, such as entity focus, length constraints, and content emphasis.  

## **BACKGROUND**  
Text summarization is a critical task in NLP, aiming to condense lengthy documents into concise summaries while retaining key information. Traditional summarization systems fall into two categories: extractive summarization, which selects and concatenates important sentences from the source document, and abstractive summarization, which generates novel sentences that paraphrase the original content.  

Despite advancements in neural summarization models, existing systems produce generic summaries without accommodating user preferences. For instance, a news article about a sports event may contain multiple entities (e.g., players, teams), but standard summarization models arbitrarily select content without regard to user interest in specific entities. This limitation reduces the utility of automated summarization in personalized applications.  

Prior attempts at controllable summarization require predefined control aspects (e.g., length, topic) and specialized training for each control dimension. These approaches lack flexibility and cannot adapt to new control tasks without retraining. There is a need for a unified framework that enables dynamic control over summarization without modifying the underlying model architecture.  

## **DETAILED DESCRIPTION**  

### **Limitations of Existing Summarization Systems**  
Conventional summarization models generate fixed outputs without user input, limiting their applicability in scenarios requiring tailored summaries. For example, medical professionals may need summaries focused on specific symptoms, while legal analysts may require emphasis on particular case precedents. Existing systems fail to provide such adaptability.  

### **Motivation for Controllable Summarization**  
The disclosed invention addresses these limitations by introducing a controllable summarization system that conditions output on user-provided keywords. This approach decouples control logic from model training, allowing the same model to support diverse control aspects (e.g., entity focus, summary length) through keyword manipulation.  

### **Controllable Summarization System Overview**  
The system, termed **CTRLSUM**, leverages a neural sequence-to-sequence architecture trained to predict summaries conditioned on both the source document (**x**) and a set of keywords (**z**). The conditional distribution is denoted as **p(y|x, z)**, where **y** is the summary. At inference time, a control function **g<sub>control</sub>(x, c)** maps user input (**c**) to keywords, enabling dynamic control without model retraining.  

### **System Components**  
1. **Neural Network Model**: The backbone is a transformer-based architecture (e.g., BART) fine-tuned to generate summaries from **x** and **z**.  
2. **Keyword Manipulation Mechanism**: Keywords are extracted from the source document or provided by the user. The system prepends keywords to **x** during training and inference.  
3. **User Interaction Module**: A control center allows users to input control signals (e.g., entity names, desired length) translated into keywords via **g<sub>control</sub>**.  

### **Workflow**  
1. **Training Phase**:  
   - Extract keywords (**z<sub>train</sub>**) from reference summaries using ROUGE-guided sentence selection.  
   - Train the model to maximize **p(y|x, z)**.  
   - Apply keyword dropout to prevent over-reliance on keywords.  

2. **Inference Phase**:  
   - Receive document **x** and control signal **c** (e.g., "focus on LeBron James").  
   - Generate keywords **z<sub>test</sub> = g<sub>control</sub>(x, c)**.  
   - Decode summary **y** conditioned on **x** and **z<sub>test</sub>**.  

### **Controllable Summarization Overview**  

#### **Traditional vs. Controllable Summarization**  
Traditional models learn **p(y|x)**, whereas CTRLSUM learns **p(y|x, z)**, enabling control via **z**. The system architecture comprises:  
- An encoder processing **x** and **z**.  
- A decoder generating **y** with attention to keyword-guided content.  

#### **Probability Distribution p(y|x, z)**  
The model is trained to maximize the likelihood of **y** given **x** and **z**, ensuring summaries align with user intent.  

#### **Keyword Extraction**  
Keywords are identified via:  
1. **Training**: Select sentences maximizing ROUGE with the reference summary, then extract longest sub-sequences.  
2. **Inference**: Use a BERT-based tagger to predict keyword probabilities.  

#### **User Interaction**  
Users interact via a graphical interface or API, specifying control tokens (e.g., "summarize contributions") or entities (e.g., "Miami Heat").  

#### **Flexibility**  
The same model supports:  
- **Entity Control**: Direct keyword injection (e.g., **z = "LeBron James"**).  
- **Length Control**: Adjust keyword count to modulate summary length.  
- **Multi-Task Prompts**: Combine keywords with task-specific prompts (e.g., "Q: What happened to [entity]? A:").  

### **Computer Environment**  

#### **Hardware Architecture**  
The system operates on computing devices comprising:  
- **Processor**: CPU/GPU for neural network inference.  
- **Memory**: Storage for model weights and input data.  
- **I/O Interfaces**: APIs for user input and summary output.  

#### **Software Modules**  
1. **Controllable Summarization Module**: Executes model inference.  
2. **Keyword Tagger**: Extracts keywords from documents.  
3. **Control Center**: Translates user input into control signals.  

### **Controllable Summarization Workflows**  

#### **Training Process**  
1. Receive (**x**, **y**) pairs from a summarization dataset.  
2. Extract keywords **z** from **y** via ROUGE-based selection.  
3. Train model to predict **y** from **x** and **z** with keyword dropout.  

#### **Inference Process**  
1. Input document **x** and control signal **c**.  
2. Generate **z = g<sub>control</sub>(x, c)**.  
3. Decode summary **y** from **x** and **z**.  

#### **Keyword Strategies**  
- **Entity Control**: **z = c** (e.g., **c = "Miami Heat"**).  
- **Length Control**: Bucket summaries by length and extract **K<sub>l</sub>** keywords per bucket.  
- **Prompt-Based Control**: Prepend task-specific prompts (e.g., "The purpose of this invention is").  

### **Example Performance**  

#### **Qualitative Examples**  
- **Entity Control**: A sports article summarized with focus on "LeBron James" yields stats-centric output.  
- **Length Control**: Adjusting **K<sub>l</sub>** produces variable-length summaries.  

#### **Quantitative Metrics**  
- **ROUGE/Length Control**: Achieves Pearson Correlation Coefficient (PCC) of 0.89 between target and actual lengths.  
- **Human Evaluation**: Users rate CTRLSUM summaries higher in control accuracy (4.2/5) vs. baselines (3.1/5).  

#### **Comparative Results**  
CTRLSUM outperforms BART in:  
- Entity inclusion (95% success rate vs. 61%).  
- Length adherence (PCC 0.89 vs. 0.72).  

## **CONCLUSION**  
The disclosed invention provides a flexible framework for controllable summarization, enabling dynamic user control without model retraining. By decoupling control logic from training, CTRLSUM supports diverse applications, from personalized news summaries to technical document condensation.  

---  
*This application includes additional embodiments, experimental details, and claims not reproduced here for brevity.*