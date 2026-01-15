- The study explores using deep learning with three input channels for interactive CT image contouring: the target image and two context channels. Experiments show a model trained on multiple structures can accurately predict contours both within and outside the training set based on provided context.

- Two prediction methods, direct and iterative, were compared. Iterative prediction performed better for unseen structures while showing similar results to direct prediction for included structures. The error propagation in iterative prediction remained small over large interslice distances.

- Key findings include that a diverse training set improves generalization performance even for excluded structures. This suggests the model learns contextual features rather than structure-specific ones, enabling it to generalize. 

- Compared to fully automatic approaches, the proposed method achieved similar or better DSC values for some structures but lower performance on others like spleen. However, this approach can decrease delineation times for unseen structures where no automated solutions exist.

- The authors conclude that contextual deep learning facilitates interactive contouring and may enable faster de-novo clinical contouring when manual input is required and insufficient data exists to train structure-specific models. Future work will examine interobserver variability using the proposed tool.
  
- Funding was provided by the European Union's Horizon 2020 programme, CRUK, and the CRUK Radnet Centre. Data was sourced from The Cancer Imaging Archive and Medical Segmentation Decathlon. The authors declare no conflicts of interest.

- Supplemental materials provide additional experimental results, investigate the necessity of all three input channels, demonstrate generalization to unseen structures, and illustrate the importance of a diverse training set for model performance.
