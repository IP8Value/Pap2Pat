Here is the patent application following the provided outline:

# DESCRIPTION  

## FIELD OF INVENTION  

The present invention relates to the field of medical imaging and radiotherapy planning. More specifically, the invention pertains to systems and methods for interactive segmentation of anatomical structures in medical images using contextual deep learning techniques. The disclosed technology enables efficient contouring of both known and previously unseen anatomical structures by incorporating user-provided contextual information into a machine learning-based segmentation process.  

## BACKGROUND OF INVENTION  

### Image Segmentation  

Image segmentation plays a critical role in various medical applications, particularly in radiotherapy planning where accurate delineation of anatomical structures is essential for treatment optimization. The process of image segmentation involves partitioning medical images into meaningful regions corresponding to specific anatomical structures or pathological findings. In clinical practice, segmentation results are typically represented as contours that outline the boundaries of structures of interest. These contours form contour sets that collectively define three-dimensional volumes for treatment planning purposes.  

Manual contouring, while considered the gold standard, presents significant limitations including time consumption and inter-observer variability. The labor-intensive nature of manual segmentation has motivated the development of automated techniques. Atlas-based auto-segmentation methods represent an early approach to automation, where new images are segmented by mapping them to pre-labeled atlases. More recently, machine learning-based approaches have demonstrated superior performance by learning complex patterns directly from training data. However, fully automatic segmentation methods often fail when presented with anatomical variations or pathologies not well-represented in their training data.  

### The Need for Interactive Segmentation  

The limitations of fully automatic techniques have created a need for interactive segmentation approaches that combine the efficiency of automation with the accuracy of human oversight. Interactive methods address the shortcomings of fully automatic techniques by incorporating user guidance during the segmentation process. Compared to manual approaches, interactive techniques improve repeatability and consistency across multiple observers while maintaining clinical relevance.  

Current interactive segmentation methods face challenges in effectively utilizing user inputs. Two-dimensional interaction techniques, while intuitive, fail to capture the three-dimensional nature of medical imaging data. Three-dimensional interaction methods often prove cumbersome in clinical workflows. Contour-based interaction approaches show promise but require sophisticated mechanisms to propagate user edits across image slices. Stroke-based interaction methods provide flexibility but lack spatial context preservation.  

The disclosed invention addresses these challenges by introducing a contextual interaction framework that leverages spatial relationships between image slices. This approach maintains anatomical consistency while allowing efficient user correction of automated segmentation results.  

### Application Examples of Segmentation:  

In clinical practice, manual contouring typically involves a clinician reviewing sequential two-dimensional image slices and manually delineating structures of interest on each slice. This process may require hours of expert time for complex cases. Semi-automatic contouring methods reduce this burden by automatically propagating user-drawn contours from reference slices to adjacent slices. However, existing semi-automatic methods often require structure-specific tuning and fail when applied to anatomical structures not included in their training.  

## PRIOR ART  

The field of medical image segmentation has seen significant advances through machine learning techniques, with various approaches demonstrating different strengths and limitations.  

### Fully Automated Segmentation Using ML  

Prior work by China et al. demonstrated fully automatic segmentation using deep convolutional neural networks trained on large datasets of annotated medical images. Similarly, WO2017/091833 disclosed methods for automatic organ segmentation in CT scans using three-dimensional neural networks. While these methods achieve high accuracy for specific anatomical structures, they exhibit limited generalizability to unseen structures and fail to incorporate user feedback during the segmentation process.  

### Interactive 2D Image Segmentation Approaches  

Various interactive two-dimensional segmentation approaches have been proposed, primarily focusing on natural images rather than medical applications. These methods typically employ user-provided scribbles or bounding boxes to guide segmentation algorithms. While effective for simple two-dimensional tasks, these approaches lack the three-dimensional contextual understanding required for medical image analysis.  

### Interactive 3D Segmentation in Medical Images  

Three-dimensional interactive segmentation techniques for medical images have been developed using graph cuts, random walks, and other optimization-based methods. These approaches allow some degree of user interaction but require extensive parameter tuning and often demonstrate poor performance on complex anatomical structures. The computational complexity of these methods further limits their clinical utility.  

### Interactive Contour Propagation Methods  

Existing interactive contour propagation methods typically employ simple interpolation techniques between user-drawn contours on key slices. More advanced methods, such as those disclosed by Léger et al., use machine learning models trained on specific anatomical structures to propagate contours slice-by-slice. Zheng et al. proposed improvements to this approach through specialized network architectures. However, these methods remain limited to predefined anatomical structures and fail to generalize to unseen structures.  

Training machine learning models for heart segmentation has demonstrated the potential of structure-specific approaches, but testing these models on different organs reveals significant limitations in generalizability. State-of-the-art tools for image segmentation continue to face trade-offs between contour quality and interaction speed. Automated deep learning contouring methods provide fast results but lack mechanisms for intuitive user correction.  

The disclosed invention addresses these limitations through a novel interactive contouring method that receives an input two-dimensional image slice and corresponding contour data, predicts target contour data using a contextual machine learning model, and creates three-dimensional contours from predicted two-dimensional contours. This approach uniquely leverages contextual information learned from diverse training data to enable segmentation of both known and previously unseen anatomical structures.  

## DETAILED DESCRIPTION  

The present invention introduces an innovative deep learning-based contouring system that overcomes limitations of prior art methods through contextual learning and interactive refinement. The system employs a medical image database and contour prediction engine to generate accurate segmentations while allowing efficient user interaction.  

At the core of the invention is a contextual deep learning model trained on multiple organs and capable of predicting contours for various anatomical structures. The medical image contouring system receives input two-dimensional image slices and contour data, then predicts target contour data using learned contextual relationships. For example, when contouring a three-dimensional medical image, the system identifies a target image slice and uses the machine learning model to generate a target contour based on provided contextual information.  

The model architecture employs a convolutional neural network modified to process three input channels: the target image slice, a reference image slice, and corresponding contour data. This design enables the model to learn contextual relationships rather than structure-specific features. The network distinguishes structures through learned contextual patterns rather than explicit labels, allowing generalization to previously unseen anatomical structures.  

Training methodology represents a critical aspect of the invention. The model is trained using diverse anatomical structures with careful selection of slice distances between reference and target images. This training strategy ensures the model learns meaningful contextual relationships rather than simple contour copying. Data augmentation techniques including affine transformations further enhance model robustness.  

The contouring system workflow includes several key components: a contour prediction engine for automated segmentation, manual contouring and editing tools for user refinement, and an image rendering engine for visualization. The system supports multiple input variants including single or multiple reference slices and can incorporate empty masks when appropriate.  

An exemplary application workflow begins with loading a patient's three-dimensional image volume. The user selects an initial two-dimensional image slice and provides manual contours for structures of interest. The contour prediction engine then generates contours on adjacent slices using either direct propagation from the reference slice or iterative propagation through intermediate slices. The user may edit predicted contours, with corrections automatically incorporated into subsequent predictions.  

The training process for the machine learning model emphasizes generalization across anatomical structures. Training data is carefully constructed to include diverse structures with varied appearances and spatial relationships. This diversity enables the model to learn fundamental contextual patterns rather than memorizing specific anatomical features.  

System performance evaluation demonstrates significant advantages over prior approaches. Multi-structure models outperform single-structure models in both accuracy and generalizability. Quantitative measures including Dice similarity coefficient and relative added path length show substantial improvements, particularly for previously unseen anatomical structures.  

The invention may be implemented as a computer program product comprising computer-executable instructions stored on non-transitory computer-readable media. The system operates on standard computer hardware and may integrate with existing medical imaging platforms. Various modifications and optimizations may be applied to accommodate different imaging modalities or clinical workflows while maintaining the core inventive concepts.  

The disclosed technology represents a significant advance in medical image segmentation by combining the efficiency of machine learning with the precision of interactive correction. Unlike prior approaches limited to predefined anatomical structures, the contextual learning framework enables accurate segmentation of both known and novel structures based on learned spatial relationships. This capability has particular importance in radiotherapy planning where new or atypical anatomical presentations frequently require manual contouring.  

By reducing the need for exhaustive manual segmentation while maintaining clinical accuracy, the invention addresses critical challenges in medical image analysis. The integration of contextual learning with interactive refinement creates a practical solution for clinical workflows, offering substantial time savings without compromising segmentation quality. The system's ability to generalize to unseen structures further enhances its clinical utility across diverse patient populations and anatomical variations.