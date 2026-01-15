Here is the patent application following the provided outline and research paper:

# DESCRIPTION  

## STATEMENT REGARDING PRIOR DISCLOSURES BY THE INVENTOR OR A JOINT INVENTOR (IF APPLICABLE)  

The present invention relates to improvements in explainable artificial intelligence systems, particularly for generating contrastive explanations in natural language processing applications. Prior art includes various feature-based and exemplar-based explanation methods for machine learning models, such as LIME (Ribeiro et al. 2016) and SHAP (Lundberg and Lee 2017). Additional relevant prior art includes contrastive explanation methods like Generate Your Counterfactuals (GYC) (Madaan et al. 2021) and Minimal Contrastive Editing (MICE) (Ross et al. 2021), which modify input text to generate explanations. However, these existing methods lack the ability to provide semantically meaningful attributes that explain why particular contrasts were generated, and they often require extensive fine-tuning or specific model architectures.  

## BACKGROUND OF THE INVENTION  

As artificial intelligence systems become increasingly prevalent in decision-making applications, there is growing need for explainable AI (XAI) methods that provide interpretable explanations of model behavior. This is particularly important in natural language processing applications where black-box models make critical decisions about text classification, sentiment analysis, and other language understanding tasks. Current contrastive explanation methods generate modified versions of input text that result in different model predictions, but they fail to provide additional insight into why these particular modifications led to changed predictions. There exists an unmet need for explanation systems that not only generate fluent contrastive examples but also identify the underlying semantic attributes responsible for the contrast, enabling better understanding of model behavior and potential biases.  

## SUMMARY  

The present invention provides a novel Contrastive Attributed Text (CAT) program that generates contrastive explanations for text classification models while identifying semantically meaningful attributes that drive the contrasts. The system operates by receiving an input text, determining important words using classifier gradients, generating candidate perturbations through masked language modeling, and selecting the optimal perturbation based on an objective function that considers prediction change, attribute modification, edit distance, and fluency. A key innovation is the use of attribute classifiers that identify relevant subtopics or themes in the text, allowing the system to explain contrasts in terms of added or removed attributes. The CAT program provides significant advantages over prior methods including better fluency, higher content preservation, fewer required edits, and the unique ability to explain contrasts through meaningful attributes.  

## DETAILED DESCRIPTION  

The present invention comprises a computer program product for generating contrastive attributed explanations for text, implemented through a series of interconnected modules and processes. The computer program product includes computer readable storage medium having computer readable program instructions embodied therewith for execution by a processor.  

The computer readable storage medium may include, but is not limited to, electronic storage devices such as random access memory (RAM), read-only memory (ROM), electrically erasable programmable read-only memory (EEPROM), flash memory, optical storage devices, magnetic storage devices, or any suitable combination thereof. The computer readable program instructions may be downloaded to the storage medium from another computer or external storage device via a network interface card or similar network component.  

The network components for instruction transmission may include wired connections such as Ethernet or fiber optic cables, or wireless connections such as Wi-Fi, Bluetooth, or cellular networks. The computer readable program instructions may include executable code, scripts, or configuration files that implement the various functions of the CAT program.  

Execution of the instructions on a user's computer causes the processor to implement the CAT program functionality. Alternatively, the instructions may be executed on a remote computer or server accessed through a network connection. In some embodiments, certain operations may be performed by specialized electronic circuitry such as application-specific integrated circuits (ASICs) or field-programmable gate arrays (FPGAs) optimized for natural language processing tasks.  

The functionality of the CAT program is illustrated through flowchart and block diagram representations that show how various functions and acts are implemented. The computer readable storage medium contains instructions that, when loaded onto a computer, cause the processor to perform a series of operational steps that implement the contrastive explanation generation process.  

The computing environment for the CAT program includes one or more computing devices connected through network components. Each computing device contains standard components including a processor, memory, storage devices, input/output interfaces, and network interfaces. The network components facilitate communication between devices and may include routers, switches, firewalls, and other networking equipment.  

The CAT program functions by first receiving text input from a user or application. The program then determines classifiers for the text data by analyzing the output of a black-box classification model. A mask module identifies important words in the input text using techniques such as integrated gradients or other feature attribution methods. The system generates candidate perturbations by inserting mask tokens around important words and using a language model to fill the masks with alternative words.  

For each candidate perturbation, the system determines edit distance from the original text, evaluates fluency using language model probabilities, and calculates attribute changes using specialized attribute classifiers. The system selects the candidate perturbation that maximizes an objective function balancing prediction change, attribute modification, edit distance, and fluency. The final output includes both the perturbed text and data about which attributes were added or removed to create the contrast.  

The system architecture includes components for identifying important words, generating perturbations, evaluating language models, and reclassifying perturbed text. The computing device running the CAT program contains computer-readable storage media storing the program instructions, language models, and classifier parameters.  

The contrastive attributed text (CAT) program implements an objective function (E.1) that combines multiple factors:  

E.1: maximize [λ·1(f(x)≠f(x')) + β·(Σ_i 1(|ζ_i(x')-ζ_i(x)|>τ)) - η·log p_LM(x') - ν·d_Lev(x,x')]  

Where λ controls the importance of prediction change, β weights attribute modification, η weights fluency, and ν weights edit distance. The attribute classifiers ζ_i provide scores indicating presence of specific attributes in the text.  

The edit distance calculation uses word-level Levenshtein distance to quantify differences between original and perturbed text. Fluency is quantified by the likelihood of the generated sentence according to the language model. Hyperparameters λ, β, η, and ν are tuned to balance these competing objectives.  

In one embodiment, the CAT program operates on text classification tasks such as news categorization or sentiment analysis. Example perturbations demonstrate how changing a few key words can alter model predictions while maintaining fluency. A table of results shows quantitative improvements over prior methods in terms of flip rate, edit distance, content preservation, and fluency.  

The CAT program can detect classifier model bias by revealing which attributes most easily change predictions. The operational processes include: receiving text input; determining classifiers; inserting masks for important words; generating candidate perturbations; evaluating edit distance and fluency; selecting the optimal perturbation; and providing both the perturbed text and attribute modification data.  

The system architecture includes components for identifying important words through gradient analysis, perturbing those words through masked language modeling, evaluating perturbations through language models and attribute classifiers, and selecting the best contrastive explanation. The computing device components include processors, memory, storage media, and network interfaces required to execute these functions.  

Computer-readable storage media contain the instructions for implementing the CAT program, including the language models, classifier parameters, and optimization algorithms. The program loads these instructions into memory and executes a series of operational steps to generate contrastive explanations.  

The flowchart and block diagram functionality illustrates how the system receives input, processes it through various modules, and generates output. The computing environment supports distributed execution where some components run on local devices and others on remote servers.  

In conclusion, the detailed description specifies how the CAT program implements novel methods for generating contrastive explanations with attribute information. The system provides technical improvements over prior methods through its unique combination of perturbation generation, attribute analysis, and optimization-based selection of contrastive examples.