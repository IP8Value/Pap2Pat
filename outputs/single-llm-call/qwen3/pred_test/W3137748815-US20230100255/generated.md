# DESCRIPTION

## FIELD OF INVENTION

- relate to medical imaging and radiotherapy planning

The present invention relates to systems and methods for medical imaging and radiotherapy planning, particularly to an interactive contouring framework that leverages contextual deep learning to predict anatomical boundaries in three-dimensional medical image datasets. The invention enables clinicians to rapidly and accurately delineate structures of interest in computed tomography, magnetic resonance, or other volumetric imaging modalities by providing a single manually drawn contour on one image slice and allowing a machine learning model to propagate that contour across adjacent or distant slices based on learned spatial and structural context. This approach is particularly valuable in radiotherapy treatment planning, where precise segmentation of tumors and critical organs directly impacts dose delivery accuracy and patient outcomes. Unlike conventional segmentation tools that require pre-defined anatomical templates or are limited to structures included in their training data, the disclosed system generalizes across anatomical entities regardless of prior exposure, enabling seamless contouring of both common and previously unseen structures without retraining. The invention integrates contextual information from image intensity, spatial relationships, and user-provided contours to generate high-fidelity segmentations that reduce manual effort, minimize inter-observer variability, and accelerate clinical workflows.

## BACKGROUND OF INVENTION

### Image Segmentation

- motivate image segmentation
- describe applications of image segmentation
- define contour and contour set
- discuss limitations of manual contouring
- introduce atlas-based auto-segmentation
- discuss machine learning-based approaches
- highlight need for interactive segmentation

Accurate image segmentation is a foundational requirement in modern medical imaging, particularly in radiotherapy planning, surgical navigation, and diagnostic assessment, where the delineation of anatomical structures determines the spatial precision of clinical interventions. A contour, as used herein, refers to a closed boundary defined on a two-dimensional image slice that encloses a region of interest, such as a tumor, organ, or tissue type, while a contour set denotes the collection of such boundaries across multiple slices that collectively define a three-dimensional volume. Manual contouring, though widely practiced, is labor-intensive, time-consuming, and subject to significant variability between observers due to subjective interpretation, fatigue, and anatomical complexity. Atlas-based segmentation methods attempt to mitigate these issues by registering patient images to pre-labeled anatomical atlases, but they often fail when anatomical variations exceed the representational capacity of the atlas population. Machine learning-based approaches, particularly convolutional neural networks, have demonstrated remarkable success in automating segmentation tasks by learning patterns from large annotated datasets. However, these methods remain constrained by their dependence on training data that must encompass the full spectrum of anatomical appearances, sizes, and locations. When applied to structures not represented in the training corpus, such as rare tumors or variant anatomy, these models frequently produce inaccurate or unusable segmentations, necessitating extensive manual correction. This limitation underscores the persistent need for interactive segmentation techniques that combine the efficiency of automation with the adaptability of human expertise.

### The Need for Interactive Segmentation

- motivate interactive segmentation
- discuss limitations of fully automatic techniques
- introduce semi-automatic methods
- discuss benefits of interactive techniques
- highlight difficulty with interaction
- discuss 2D vs 3D interaction
- introduce contour-based interaction
- discuss stroke-based interaction
- motivate use of spatial context
- introduce concept of contextual information

Fully automatic segmentation techniques, while fast, lack the flexibility to adapt to novel anatomical configurations or unexpected pathologies, rendering them unreliable in clinical settings where variability is the norm. Semi-automatic methods bridge this gap by incorporating user input as a guiding constraint, thereby improving segmentation accuracy and reducing post-processing time. Interactive techniques allow clinicians to initiate segmentation with minimal input—such as a single contour on one slice—and rely on the system to propagate that boundary across the volume. The primary challenge in interactive segmentation lies in designing intuitive, responsive, and context-aware interaction mechanisms. Two-dimensional interaction, where users draw on individual slices, is common but inefficient for volumetric data, while three-dimensional interaction, though more natural, often requires specialized hardware and suffers from poor spatial resolution. Contour-based interaction, in which users provide a closed boundary on a slice, has proven more reliable than stroke-based or point-based inputs due to its explicit definition of structure boundaries. Crucially, the effectiveness of such methods depends on the system’s ability to interpret and utilize spatial context—the relationship between the provided contour and the surrounding image intensities, texture gradients, and anatomical topology—rather than relying solely on predefined shape priors. The concept of contextual information, as employed herein, refers to the ensemble of image features and spatial relationships that enable a model to infer the likely extent of a structure beyond the provided input, independent of its identity or prior training exposure.

### Application Examples of Segmentation:

- describe manual contouring process
- describe semi-automatic contouring process

In manual contouring, a clinician sequentially examines each slice of a three-dimensional image dataset and draws a boundary around the target structure using a graphical interface, often requiring hundreds of individual annotations per patient. This process can take upwards of thirty to sixty minutes per structure, depending on complexity and anatomical location. In semi-automatic contouring, the clinician provides an initial contour on a representative slice, and the system employs an algorithm to propagate that boundary to adjacent slices, either through interpolation, edge detection, or model-based prediction. Existing semi-automatic tools often rely on deformable models or graph cuts that require manual tuning and frequently fail when image contrast is low or when the structure exhibits irregular morphology. These methods typically assume a fixed anatomical template and are unable to adapt to structures not explicitly modeled during development, resulting in poor generalization and high user correction burden.

## PRIOR ART

- introduce prior art in medical imaging and ML

### Fully Automated Segmentation Using ML

- describe fully automatic segmentation method by China et al.
- describe fully automatic segmentation method by WO2017/091833
- discuss limitations of fully automatic segmentation methods

China et al. disclosed a fully automated segmentation method employing a deep convolutional neural network trained on a large dataset of annotated abdominal CT scans to delineate liver, kidneys, and spleen. The model achieved high Dice scores on structures present in the training set but exhibited significant performance degradation when applied to structures not included in the training data, such as pancreatic tumors or adrenal glands. Similarly, WO2017/091833 describes a multi-organ segmentation system using a U-Net architecture with multi-scale feature fusion, optimized for predefined anatomical classes. While effective for known structures, these approaches are fundamentally limited by their reliance on fixed class labels and cannot generalize to previously unseen anatomical entities. Their performance is contingent upon the completeness of the training corpus, and any deviation from the learned distribution—such as atypical tumor shapes or variant organ positions—results in erroneous or incomplete segmentations. These limitations render fully automated methods unsuitable for clinical environments where anatomical diversity and unpredictability are inherent.

### Interactive 2D Image Segmentation Approaches

- overview of interactive 2D segmentation approaches

Interactive 2D segmentation approaches have historically relied on user-initiated seed points or scribbles to guide region-growing algorithms or level-set methods. These methods operate on single slices and require repeated user input for each slice in a volume, resulting in a fragmented and inefficient workflow. While computationally lightweight, they lack the capacity to leverage contextual information across slices, leading to inconsistent boundaries and high inter-slice discontinuity. Furthermore, these techniques do not learn from user corrections or adapt over time, limiting their utility in dynamic clinical workflows.

### Interactive 3D Segmentation in Medical Images

- discuss interactive 3D segmentation techniques
- discuss limitations of interactive 3D segmentation techniques

Interactive 3D segmentation techniques have been developed to allow users to manipulate volumetric boundaries directly in three dimensions using specialized input devices. These include volumetric brush tools, surface deformation widgets, and haptic feedback interfaces. While offering greater anatomical fidelity, these methods demand substantial computational resources, require specialized hardware, and suffer from poor spatial resolution due to the coarse discretization of 3D interaction space. Moreover, they do not incorporate machine learning to predict boundary evolution based on context, instead relying on geometric constraints that often fail in low-contrast regions or complex anatomical junctions.

### Interactive Contour Propagation Methods

- discuss interactive contour propagation methods
- discuss limitations of interactive contour propagation methods
- motivate need for interactive contouring method
- discuss ML models for slice-by-slice image segmentation
- discuss limitations of ML models
- discuss Léger et al.'s ML model
- discuss Zheng et al.'s ML model
- discuss limitations of Léger et al.'s and Zheng et al.'s ML models
- train ML model for heart segmentation
- test ML model on different organs
- discuss results of testing ML model
- discuss limitations of ML model
- motivate need for generalized approach
- discuss state-of-the-art tools for image segmentation
- discuss limitations of state-of-the-art tools
- motivate need for fast and intuitive interaction method
- discuss importance of contour quality and speed
- discuss benefits of automated deep learning contouring
- discuss limitations of automated deep learning contouring
- motivate need for interactive contouring method
- introduce disclosed interactive contouring method
- discuss receiving input 2D image slice and input contour
- discuss predicting target contour data
- discuss using machine learning model for prediction
- discuss providing contextual information
- discuss learning contextual information from training data set
- discuss using consecutive image slices
- discuss updating machine learning model based on user edits
- discuss creating 3D contour from predicted 2D contours

Existing interactive contour propagation methods, such as those described by Léger et al. and Zheng et al., employ machine learning models to predict contours on adjacent slices based on a single user-drawn contour. However, these models are trained to recognize specific anatomical structures, such as the heart or liver, and are incapable of generalizing to novel entities. Léger et al. trained a recurrent neural network on cardiac CT data and demonstrated accurate propagation along the cardiac axis, but performance collapsed when applied to pulmonary nodules or renal structures. Zheng et al. extended this approach to multiple organs but retained structure-specific output layers, preventing the model from extrapolating beyond its training classes. These models are thus constrained by their architecture: they learn structure-specific features rather than contextual relationships. As a result, even minor deviations in anatomy or pathology lead to segmentation failure. Furthermore, these methods do not update their predictions in response to user corrections, treating each slice as an independent prediction rather than a dynamic, iterative process. The disclosed invention overcomes these limitations by training a single, unified machine learning model on a diverse set of anatomical structures without assigning class labels to contours, thereby enabling the model to learn abstract, structure-agnostic contextual cues from image intensity, spatial gradients, and boundary topology. The model receives as input a target image slice, a previously segmented image slice, and its corresponding binary mask, and predicts a new contour without reference to anatomical identity. This approach allows the model to generalize to previously unseen structures, dynamically adapt to user edits, and iteratively propagate contours across the entire volume to generate a coherent three-dimensional contour set. The system is designed to be intuitive, requiring only a single initial contour and minimal user intervention, while maintaining high segmentation fidelity and rapid inference speed.