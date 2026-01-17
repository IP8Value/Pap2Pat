# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of image processing and computer vision, particularly to methods and systems for segmenting whole cells in 2D microscopy images using a single channel marker. The invention is particularly useful in biological and medical research, where accurate and automated cell segmentation is crucial for various applications such as drug discovery, disease diagnosis, and cellular dynamics studies.

## BACKGROUND

The cell is the fundamental unit of life, and the ability to image, extract, and study cells and their sub-cellular compartments is essential in numerous research areas. High-resolution fluorescent microscopy has revolutionized the visualization of cells and their sub-cellular structures, providing detailed insights into cellular dynamics and drug efficacy. However, manual analysis of these images is time-consuming and subjective, leading to a need for automated and accurate image analysis techniques.

Automated analysis of 2D cellular images enables high-throughput cell quantification and reproducibility, addressing various biological questions. Traditional methods often focus on segmenting cell nuclei, which are relatively uniform in shape and size, but whole cell segmentation, including the cytoplasm, remains challenging due to the variability in cell shape and size, and the presence of touching cells with weak boundary gradients.

Several algorithms have been proposed for cell segmentation, including watershed-based methods, levelset methods, and active contour models. However, these methods often struggle with the complexity of whole cell segmentation, especially in single-channel images where the cytoplasm is hyperintense and the nuclei are hypointense. Deep learning techniques have shown promise in this domain, but most existing approaches rely on multiple channels, which limits their applicability in scenarios where only a single channel is available.

There is a persistent need for robust and automated algorithms that can segment whole cells in 2D microscopy images using a single channel marker, accommodating the high variability in cell appearance and the challenges posed by touching cells.

## SUMMARY

The present invention addresses the aforementioned needs by providing a deep learning-based framework for the automated segmentation of whole cells in 2D microscopy images using a single channel marker. The invention includes the following key components:

1. **Deep Learning Predictive Model**: A deep convolutional neural network (CNN) is trained to predict per-pixel probabilities for nuclei, cytoplasm, and background using single-channel images. The model is designed to handle the variability in cell appearance and the challenges of segmenting touching cells.

2. **Nuclei Seed Detection**: An efficient algorithm is employed to detect individual nuclei from the nucleus prediction map. This step involves multi-scale Laplacian of Gaussian (LoG) blob detection and shape-based watershed segmentation to accurately identify and separate touching nuclei.

3. **Seed-Based Cell Segmentation**: A seeded-watershed algorithm is used to segment individual cells using the cell prediction map and the segmented nuclei as seeds. This ensures robust segmentation of the entire cell, including the cytoplasm and nucleus.

The invention is particularly advantageous for its ability to handle a wide range of cell markers, drug treatment conditions, and magnifications, making it a versatile tool for various biological and medical applications.

## DETAILED DESCRIPTION

### System Overview

The system for automated whole cell segmentation in 2D microscopy images using a single channel marker comprises several key components: a deep learning predictive model, a nuclei seed detection algorithm, and a seed-based cell segmentation algorithm. The system is designed to handle the variability in cell appearance and the challenges of segmenting touching cells, providing accurate and robust segmentation results.

### Development of a Training Model

The deep learning predictive model is developed using a UNet-like architecture, which is trained on labeled images to predict per-pixel probabilities for nuclei, cytoplasm, and background. The training process involves the following steps:

1. **Image Preprocessing**: Images are preprocessed to correct for uneven illumination and to standardize the pixel size. Top-hat filtering with a kernel size of 200x200 pixels is applied to suppress the image background. Images are down-sampled to approximately 10x magnification to ensure consistency.

2. **Model Architecture**: The UNet-like architecture consists of a contracting path and an expanding path. The contracting path includes a series of convolution and pooling layers, while the expanding path includes deconvolution layers. The architecture is designed to handle the variability in cell appearance and to provide accurate predictions.

3. **Training Process**: The model is trained using image patches of 160x160 pixels. The loss function is defined as the root mean square deviation (RMSD) of the prediction and label, with additional constraints to ensure the relationship between the different labels. The training data is augmented by rotating the original patches by 90 degrees to increase the diversity of the training set.

### Single Channel Whole Cell Segmentation Workflow

The workflow for segmenting whole cells in 2D microscopy images using a single channel marker involves the following steps:

1. **Deep Learning Inference**: Unseen images are divided into 176x176 patches, which are processed through the trained deep learning model to create probability maps for nuclei, cytoplasm, and background. The predicted patches are stitched together to build the prediction of the full image.

2. **Nuclei Seed Detection**: The nuclei prediction map is processed to detect individual nuclei seeds. A multi-scale Laplacian of Gaussian (LoG) blob detector is applied to enhance regions containing blob-like nuclei. Automated multi-level Otsu thresholding is used to extract the binary nuclear mask. A shape-based watershed approach is then applied to segment individual nuclei from the binary mask.

3. **Seed-Based Cell Segmentation**: The cytoplasm prediction map is combined with the cell marker image to enhance the cells. The background is determined using three-level Otsu thresholding. The segmented nuclei are used as seeds in a seeded-watershed transform to segment the entire cells. This approach ensures robust segmentation of the entire cell, including the cytoplasm and nucleus.

### Assessment of Segmentation Results

The quality of the automated segmentation results is assessed using a custom cell segmentation similarity metric. The metric compares the automated segmentation to the ground truth or reference segmentation, taking into account the large number of segmented cells. The similarity metric is defined as follows:

\[ \text{SM}(R, T) = k \left( \frac{1}{N} \sum_{i=1}^{N} \max_{t_j \in T^{r_i}} \frac{2 | r_i \cap t_j |}{| r_i | + | t_j |} \right) + (1 - k) \left( \frac{2 | P_T^R |}{N + M} \right) \]

where \( R \) and \( T \) are the sets of labels in the reference and target segmentations, respectively, and \( k \) is a weighting factor. The first term computes the average maximum overlap between each label in the reference segmentation and the corresponding labels in the target segmentation, while the second term computes the ratio of true positive labels to all the labels.

### Examples

#### Example 1: Segmentation of Hela Cells

In this example, the system was applied to segment Hela cells stained with a green-dsRed marker. The input image was preprocessed to correct for uneven illumination and standardized to 10x magnification. The deep learning model was used to predict the nuclei, cytoplasm, and background. The nuclei seed detection algorithm successfully identified individual nuclei, and the seed-based cell segmentation algorithm accurately segmented the entire cells, including the cytoplasm and nucleus.

#### Example 2: Segmentation of Fibroblasts

In this example, the system was applied to segment fibroblasts stained with a TexasRed marker. The input image was preprocessed and the deep learning model was used to predict the nuclei, cytoplasm, and background. The nuclei seed detection algorithm effectively identified individual nuclei, and the seed-based cell segmentation algorithm accurately segmented the entire cells, handling the variability in cell shape and size.

#### Example 3: Segmentation of HEPG2 Cells

In this example, the system was applied to segment HEPG2 cells stained with a Cy5 marker. The input image was preprocessed and the deep learning model was used to predict the nuclei, cytoplasm, and background. The nuclei seed detection algorithm successfully identified individual nuclei, and the seed-based cell segmentation algorithm accurately segmented the entire cells, even in images with tightly-packed cells.

These examples demonstrate the robustness and versatility of the system in segmenting whole cells in 2D microscopy images using a single channel marker, accommodating the high variability in cell appearance and the challenges of segmenting touching cells.