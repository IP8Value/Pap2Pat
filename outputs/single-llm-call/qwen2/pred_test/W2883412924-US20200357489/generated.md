# DESCRIPTION

## TECHNICAL FIELD OF THE INVENTION

The present invention relates to a deep learning-based drug screening system, particularly for evaluating the efficacy of nanoformulated drugs. The system leverages convolutional neural networks (CNNs) to analyze single-cell images obtained from flow cytometry, providing a rapid, accurate, and cost-effective method for drug screening.

## BACKGROUND OF THE INVENTION

Nanomaterials have become a focal point in the development of novel drugs due to their ability to enhance the efficacy of traditional medications. However, conventional methods for evaluating nanoformulated drugs, such as the 3-(4,5-dimethylthiazol-2-yl)-2,5-diphenyltetrazolium bromide (MTT) assay and lactate dehydrogenase release assays, often fall short due to the unique physicochemical properties of nanoparticles. These methods can be distorted by the high absorbance, optical activity, and large surface area of nanoparticles, leading to inaccurate results. Consequently, there is an urgent need for a new method that can effectively evaluate the efficacy of nanoformulated drugs.

Machine learning, particularly deep learning, has shown promise in medical research by identifying subtle changes in cells and predicting their behavior. Convolutional neural networks (CNNs), a subset of deep learning, are particularly well-suited for image-based tasks, making them an ideal choice for developing a drug screening system. CNNs can automatically extract features from single-cell images, enabling a more precise and objective evaluation of drug efficacy.

The present invention addresses the limitations of existing methods by providing a deep learning-based drug screening system that is both accurate and efficient. This system, referred to as DeepScreen, utilizes single-cell images obtained from flow cytometry and a CNN to classify the efficacy of nanoformulated drugs. By reducing the need for manual analysis and improving the sensitivity of drug evaluation, DeepScreen offers a significant advancement in the field of drug discovery.

## SUMMARY OF THE INVENTION

The present invention provides a deep learning-based drug screening system, DeepScreen, for evaluating the efficacy of nanoformulated drugs. The system comprises the following key components:

1. **Data Collection**: Single-cell images are obtained from flow cytometry after treating cells with nanoformulated drugs. These images are labeled based on the efficacy of the drug treatment, which is determined using a 24-hour MTT assay.

2. **Image Preprocessing**: The single-cell images are preprocessed to ensure they are suitable for input into the CNN. This includes concatenating the single-channel images and resizing them to a standard size.

3. **Convolutional Neural Network (CNN)**: A CNN is trained using the preprocessed images. The CNN is designed with a "network-in-network" structure, which includes convolution layers, pooling layers, and regularization layers. This structure allows the network to extract high-order features from the images and perform classification tasks accurately.

4. **Model Training**: The CNN is trained using a balanced sampling strategy and various regularization techniques to improve its performance and prevent overfitting. The Adam optimizer is used to update the training parameters.

5. **Evaluation and Validation**: The trained model is evaluated using a separate set of testing data to assess its accuracy and robustness. The system demonstrates high accuracy in classifying the efficacy of nanoformulated drugs, even with short treatment times.

6. **Class Activation Mapping (CAM)**: CAM is used to visualize the areas of the cell that the model focuses on during classification. This provides insights into the biological processes underlying the drug's effects.

The invention further includes methods for preparing the nanoformulated drugs, collecting and preprocessing the single-cell images, and training the CNN. The system is designed to be user-friendly and adaptable, allowing for the addition of additional markers to improve accuracy or simplify the process for convenience.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENT

### Example 1 A Deep Learning-Based Quick and Precise High-Throughput Drug Screening System

#### Introduction

The present invention, DeepScreen, is a novel deep learning-based drug screening system designed to evaluate the efficacy of nanoformulated drugs. The system utilizes single-cell images obtained from flow cytometry and a convolutional neural network (CNN) to classify the efficacy of the drugs. This approach offers several advantages over traditional methods, including increased accuracy, reduced manual intervention, and the ability to detect subtle changes in cells at early stages of drug treatment.

#### Data Collection

To develop the DeepScreen system, two types of nanoformulated drugs were used: inorganic-layered double hydroxide loaded with etoposide (LDH-VP16) and lipid-based materials solid lipid nanoparticles loaded with curcumin (SLN-Cur). These drugs were chosen for their potential in cancer treatment and their unique physicochemical properties.

A549 (human pulmonary adenocarcinoma) and HEpG2 (human hepatocellular carcinoma) cell lines were used in the study. The cells were treated with the nanoformulated drugs for 2 and 6 hours, and single-cell images were collected using flow cytometry. The images were labeled based on the efficacy of the drug treatment, which was determined using a 24-hour MTT assay. The labels included ineffective (IE), low efficacy (LE), medium efficacy (ME), and high efficacy (HE).

#### Image Preprocessing

The single-cell images obtained from flow cytometry were preprocessed to prepare them for input into the CNN. Each cell image sample contained three single-channel images: bright-field, Annexin V-APC, and, optionally, anti-EGFR-FITC. The single-channel images were concatenated channelwise, and the resulting images were resized to 70x70 pixels using bicubic interpolation. The resized images were then standardized to form the input for the CNN.

#### Convolutional Neural Network (CNN)

The CNN used in the DeepScreen system is based on the "Google Inception" network architecture. The network has a "network-in-network" structure, which includes convolution layers, pooling layers, and regularization layers. This structure allows the network to extract high-order features from the input images and perform classification tasks accurately.

The CNN was implemented using the TensorFlow framework and trained on two NVIDIA GTX 1080Ti GPUs. The model was trained using a balanced sampling strategy to ensure that each class was equally represented in the training data. Various regularization techniques, including weight decay, dropout, and batch normalization, were applied to improve the model's performance and prevent overfitting. The Adam optimizer was used to update the training parameters, with a learning rate of 0.001, beta1 of 0.9, beta2 of 0.999, and epsilon of 1 × 10^-8.

#### Model Training

The CNN was trained using the preprocessed single-cell images and their corresponding labels. The training process involved feeding the images into the network, performing forward and backward passes, and updating the model parameters using the Adam optimizer. The training was conducted until the model achieved high accuracy on the training data.

#### Evaluation and Validation

The trained model was evaluated using a separate set of testing data to assess its accuracy and robustness. The testing data included single-cell images from A549 and HEpG2 cells treated with the nanoformulated drugs for 2 and 6 hours. The model demonstrated high accuracy in classifying the efficacy of the drugs, with accuracies of 0.851, 0.864, and 0.908 for mixed cells, HEpG2, and A549, respectively.

The model's performance was compared to traditional methods, including the MTT assay and flow cytometry. The MTT assay and flow cytometry failed to show significant differences in the efficacy of the drugs when the treatment time was shortened to 2 and 6 hours. In contrast, the DeepScreen model was able to accurately classify the efficacy of the drugs, even with short treatment times.

#### Class Activation Mapping (CAM)

To gain insights into the biological processes underlying the drug's effects, class activation mapping (CAM) was used to visualize the areas of the cell that the model focuses on during classification. The CAMs showed that the model tended to focus on specific areas of the cell, particularly the cell membrane and nuclei. This suggests that the model is able to detect subtle changes in the cell structure, which are indicative of the drug's efficacy.

#### Anti-Interference Capabilities

The DeepScreen system was tested for its ability to handle interference from the self-fluorescence of drugs and the physicochemical properties of nanoparticles. The model demonstrated high accuracy even when the self-fluorescence channel was added to the input data, indicating its robustness against interference. Additionally, the model was able to accurately classify the efficacy of nanoformulated drugs with different morphologies and compositions, further demonstrating its reliability.

#### Potential Applications

The DeepScreen system has the potential to be applied in various areas of drug discovery and development. Its ability to quickly and accurately evaluate the efficacy of nanoformulated drugs makes it a valuable tool for researchers and pharmaceutical companies. The system can be adapted to include additional markers to improve accuracy or simplified for convenience. Future developments may include the application of DeepScreen to other cellular state changes, such as stem cell differentiation, and the evaluation of emerging nanocarriers like modularized extracellular vehicles (EVs).

In conclusion, the DeepScreen system represents a significant advancement in the field of drug screening, offering a rapid, accurate, and cost-effective method for evaluating the efficacy of nanoformulated drugs. The system's robust performance and flexibility make it a promising tool for advancing medical research and drug development.