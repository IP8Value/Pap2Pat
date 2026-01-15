- A novel screening system called DeepScreen was introduced for evaluating nanoformulated drugs. Based on deep learning, it outperforms existing laboratory methods in precision, speed, and convenience for detecting nanomedications.

- The study utilized A549 (lung cancer) and HEpG2 (liver cancer) cell lines, testing etoposide (VP16) and curcumin (Cur), encapsulated within layered double hydroxides (LDHs) or solid lipid nanoparticles (SLNs).

- Cell cytotoxicity was assessed via MTT assay. Cells were co-cultured with drugs for 2 or 6 hours, stained with Annexin-V-APC, and analyzed using flow cytometry and image cytometry.

- For DeepScreen, cell images were processed by concatenating single-channel images channelwise, resizing them to 70x70 pixels, and standardizing the inputs. The model used an inception-style convolutional neural network architecture with a "network-in-network" structure.

- Training techniques included balanced sampling, regularization (weight decay, dropout, batch normalization), and Adam optimizer for parameter updates, all implemented using TensorFlow on NVIDIA GPUs.

- Class activation mapping was employed to visualize how models recognize cell images by extracting weights from the final convolution layer and computing linear combinations of feature maps. 

- The code for DeepScreen is publicly available at a GitHub repository.
