# DESCRIPTION

## FIELD OF THE INVENTION

- relate to image-guided radiation therapy

The present invention relates to image-guided radiation therapy systems designed for the precise delivery of ionizing radiation to moving tumors within the thoracic and abdominal regions of a patient. Specifically, the invention pertains to a non-invasive, real-time tumor tracking methodology that enables continuous monitoring of tumor position during radiation delivery without the use of implanted fiducial markers, external surrogates, or patient-specific calibration. The system operates using orthogonal X-ray imaging sequences acquired during treatment and leverages a learned regression model to correlate image-based features with three-dimensional tumor motion parameters. This approach is particularly suited for clinical environments where minimizing treatment time, reducing radiation exposure to healthy tissues, and eliminating invasive procedures are critical objectives. The invention integrates computational vision, machine learning, and biomechanical motion modeling to achieve sub-pixel accuracy in tumor localization under clinically realistic imaging conditions, including low contrast, limited resolution, and noise-corrupted radiographic data.

## BACKGROUND OF THE INVENTION

- motivate image-guided radiation therapy

Image-guided radiation therapy has emerged as a cornerstone of modern oncological treatment, enabling the precise targeting of tumors while sparing surrounding healthy tissues. Traditional radiation protocols require large safety margins around the target volume to account for uncertainties in tumor position due to respiratory motion, organ deformation, and patient setup variability. These margins often result in unnecessary irradiation of critical organs, leading to acute and long-term complications. Image-guided radiation therapy addresses this limitation by providing real-time feedback on tumor location during treatment, allowing for dynamic adjustment of the radiation beam to follow the tumor’s motion. This capability permits the reduction of planning target volume margins, thereby improving therapeutic ratios and enhancing patient outcomes.

- describe limitations of conventional tumor tracking

Conventional tumor tracking methods rely on either internal fiducial markers or external surrogate signals to infer tumor position. Internal fiducials, typically metallic implants placed via CT-guided surgery, provide direct positional information but require invasive procedures that carry risks such as pneumothorax, hemorrhage, and infection. Moreover, these markers may migrate over the course of multiple treatment sessions, introducing significant uncertainty into the tracking process. External surrogates, such as reflective markers placed on the chest or abdominal wall, are less invasive but suffer from poor correlation with internal tumor motion due to complex respiratory biomechanics, abdominal wall compliance, and inter- and intra-session variability in breathing patterns. These limitations render surrogate-based methods unreliable in patients with irregular respiration or anatomical changes during treatment.

- describe disadvantages of implanted fiducial markers

The implantation of fiducial markers introduces additional clinical burdens beyond procedural risk. The surgical placement requires specialized equipment, trained personnel, and additional time prior to treatment initiation. Once implanted, markers may shift due to tissue relaxation, tumor regression, or patient movement, necessitating repeated imaging and re-calibration. Furthermore, fiducials can cause image artifacts in X-ray and CT modalities, obscuring adjacent anatomical structures and complicating treatment planning. Their presence also restricts the use of certain imaging protocols and limits patient eligibility for subsequent interventions. These drawbacks have motivated the development of markerless tracking techniques that eliminate the need for physical implants altogether.

- describe limitations of markerless tumor tracking methods

Existing markerless tumor tracking methods include template matching, optical flow, particle filtering, and parametric motion modeling. Template matching techniques are highly sensitive to changes in image contrast, illumination, and tumor shape, and often fail in low-contrast regions common in lung and liver tumors. Optical flow methods, while capable of dense motion estimation, are computationally intensive, prone to drift, and require iterative optimization that precludes real-time performance. Particle filtering approaches rely on sampling large hypothesis spaces to approximate tumor motion, which demands substantial computational resources and frequently results in jittery trajectories or mode collapse when texture information is insufficient. Parametric models, though faster, require extensive offline training with manually annotated tumor trajectories, making them impractical for clinical deployment where rapid setup and adaptability are essential. None of these methods simultaneously achieve high accuracy, real-time speed, robustness to low contrast, and zero reliance on prior patient data or invasive markers.

## SUMMARY OF THE INVENTION

- describe method for tracking objects without fiducial markers

The invention discloses a method for tracking the motion of a tumor in real time during image-guided radiation therapy without the use of fiducial markers, external surrogates, or patient-specific training. The method employs orthogonal X-ray imaging sequences acquired during treatment and constructs a learned regression model that maps image features extracted from tumor-adjacent regions directly to three-dimensional affine motion parameters. The regression model is trained online using a single pair of initial X-ray images, wherein a set of random affine motion hypotheses is generated based on biomechanical constraints typical of respiratory motion. For each hypothesis, the corresponding image features are computed using histogram of oriented gradients within optimally sized tracking windows, and a ridge regression is solved to establish a linear mapping between feature vectors and motion parameters. During subsequent imaging, the model is applied to newly acquired image pairs to predict tumor motion through a single matrix multiplication, enabling sub-100-millisecond tracking with pixel-level accuracy. The method operates in two modes: a three-dimensional regression mode that jointly estimates motion from both X-ray views, and a two-dimensional regression mode that independently tracks motion in each view and reconstructs the three-dimensional position via back-projection. Both modes require no iterative optimization, no template matching, and no prior knowledge of tumor shape or boundary, making the system robust to low-contrast, low-resolution, and noisy imaging conditions.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS

- introduce tumor tracking method

The tumor tracking method operates on a sequence of orthogonal X-ray images acquired during radiation delivery. The system begins by identifying an initial tumor location through manual or automated segmentation of the first image pair. A tracking window is then defined around this region, and a regression model is trained to correlate image features within this window with underlying affine motion parameters. The model is constructed using a set of randomly sampled motion hypotheses that respect physiological constraints on tumor displacement, such as maximum translational and rotational bounds observed in respiratory motion. These hypotheses are projected onto the two orthogonal imaging planes, and corresponding image features are extracted from each view. The concatenated feature vectors are then used to solve a regularized least-squares problem that yields a single regression matrix mapping features to motion parameters.

- describe non-invasive tumor tracking

The method is entirely non-invasive, requiring no implanted markers, external sensors, or prior patient-specific calibration. It does not rely on anatomical landmarks, tumor boundaries, or intensity-based segmentation, which are often unreliable in low-contrast regions. Instead, it exploits the statistical regularity of image textures within the tumor region and their consistent correlation with motion. The absence of fiducials eliminates risks associated with surgical implantation and avoids the temporal instability inherent in marker migration. The system is capable of tracking tumors regardless of their shape, size, or internal density, provided they are visible in at least one of the orthogonal X-ray projections.

- describe regression model fitting

The regression model is formulated as a linear mapping between a high-dimensional image feature vector and a low-dimensional motion parameter vector representing the incremental affine transformation between consecutive frames. The model is trained by minimizing the geodesic distance between the predicted and ground-truth motion matrices, which lie on a Riemannian manifold, rather than using Euclidean norms. This ensures that the learned mapping respects the intrinsic geometry of affine transformations. The solution is obtained via ridge regression, which incorporates Tikhonov regularization to prevent overfitting and enhance numerical stability. The resulting regression matrix is applied in real time to new image pairs, enabling instantaneous motion estimation without iterative refinement.

- describe image feature extraction

Image features are extracted using the histogram of oriented gradients (HOG) descriptor, which captures local edge orientations within a sliding grid of 5×5 pixel blocks. The HOG representation is invariant to illumination changes and provides discriminative power even in low-contrast regions. Features are computed within a dynamically optimized tracking window, whose size is determined by analyzing the self-similarity of image patches in the vicinity of the tumor. The optimal window size is selected to maximize the mean discriminative power of feature distances while minimizing the influence of surrounding tissue variability.

- describe motion constraints

Motion hypotheses are generated within bounds derived from physiological studies of respiratory tumor motion, limiting translation to ±15 mm and rotation to ±5 degrees in each axis. These constraints reduce the hypothesis space, accelerate training, and stabilize trajectories by preventing implausible motion predictions. The constraints are not fixed but can be adapted based on patient-specific motion profiles observed during initial imaging.

- describe 3D setup parameters

The system operates with two orthogonal X-ray imaging systems positioned at 90 degrees relative to one another, typically aligned with the coronal and sagittal planes. The source-to-detector distances, gantry angles, and imaging frame rates are configured to ensure sufficient spatial resolution and temporal coverage of the respiratory cycle. The coordinate systems of both imaging modalities are calibrated to a common world frame using known geometric relationships between the radiation sources and detectors.

- describe initial object location and shape

The initial tumor location is determined from the first acquired image pair using manual delineation or automated thresholding. The shape of the tumor is not explicitly modeled; instead, a bounding box is defined around the region of interest, and the tracking window is centered on this box. No segmentation, boundary fitting, or contour estimation is performed during tracking.

- describe alignment of initial set of sequences

The initial image pair is aligned to a common reference frame using rigid registration based on anatomical landmarks external to the tumor. This ensures that the coordinate systems of the two X-ray views are consistent for the purpose of joint feature concatenation and 3D motion estimation.

- describe 3D motion parameters

The motion of the tumor is represented as a 4×4 affine transformation matrix, parameterized by 12 independent degrees of freedom: three translations, three rotations, three scales, and three shears. These parameters are encoded as a vector in a Lie algebra space, and the incremental motion between frames is computed via the matrix exponential map to ensure that the predicted transformations remain within the group of affine transformations.

- describe training module

The training module generates a set of n random affine motion hypotheses around the initial tumor position, projects each hypothesis onto both X-ray views, extracts HOG features from corresponding windows, and constructs a feature matrix X and motion matrix Y. Ridge regression is then applied to compute the optimal regression matrix Ω that minimizes the squared geodesic error between XΩ and Y, with regularization parameter λ selected via cross-validation.

- describe tracking module

The tracking module receives a new pair of X-ray images and extracts HOG features from the tracking window defined by the previous frame’s predicted position. These features are concatenated and multiplied by the precomputed regression matrix Ω to produce an incremental motion estimate. The current tumor position is updated by applying the exponential map of this motion to the previous pose.

- describe regression function

The regression function is a linear operator Ω ∈ ℝ^(m×d), where m is the dimensionality of the concatenated feature vector and d is the number of motion parameters. The function is trained once at the start of treatment and remains fixed throughout the session, as no drift or model degradation is observed in clinical data.

- describe feature extraction

Feature extraction is performed using a fixed HOG configuration: 8 orientation bins per 5×5 pixel block, with block stride of 2 pixels, and no normalization. The resulting feature vector has dimensionality m = 2,880 for a 128×128 pixel window. Features are computed in real time using optimized GPU-accelerated libraries.

- describe motion parameters estimation

Motion parameters are estimated by multiplying the feature vector from the current image pair with the regression matrix Ω, yielding a vector in the Lie algebra of affine transformations. This vector is exponentiated to produce a 4×4 affine transformation matrix, which is then applied to the previous tumor position to obtain the current estimate.

- describe motion constraints application

Motion constraints are enforced during training by limiting the range of sampled motion hypotheses. During tracking, any predicted motion exceeding physiological bounds is clipped to the nearest valid value, ensuring biologically plausible trajectories.

- describe window update

The tracking window is not re-initialized during the session but is dynamically adjusted in size based on the optimal window size determined during training. The window center follows the predicted tumor position, and its dimensions remain fixed unless manually overridden by the operator.

- describe 3D regression tracking

In the 3D regression mode, feature vectors from both X-ray views are concatenated into a single input vector, and a single regression matrix maps this combined vector directly to a 4×4 3D affine motion matrix. This eliminates the need for back-projection and ensures consistency between the two views.

- describe 3D motion representation

The 3D motion is represented as a 4×4 homogeneous affine matrix, parameterized in exponential coordinates to preserve the group structure of rigid and non-rigid transformations. This representation enables accurate interpolation and composition of motion increments.

- describe affine transformation

Affine transformation is used to model the incremental motion of the tumor between consecutive frames, accounting for translation, rotation, scaling, and shearing. This model is sufficient to capture the dominant modes of respiratory motion observed in clinical data.

- describe feature vector extraction

The feature vector is extracted by dividing the tracking window into a grid of overlapping 5×5 pixel blocks, computing an 8-bin histogram of gradient orientations for each block, and concatenating all histograms into a single column vector. The vector is normalized by its ℓ2 norm to reduce sensitivity to illumination variations.

- describe training set construction

The training set consists of 600 randomly sampled affine motion hypotheses, each associated with a corresponding concatenated HOG feature vector from the initial image pair. The hypotheses are sampled uniformly within the constrained motion space defined by physiological bounds.

- describe regression function training

The regression function is trained by solving a ridge regression problem: Ω = (XᵀX + λI)⁻¹XᵀY, where X is the feature matrix, Y is the motion matrix, and λ is a regularization parameter set to 10⁻⁴. The solution is computed using singular value decomposition for numerical stability.

- describe tumor motion prior probability

The prior probability of tumor motion is implicitly encoded in the distribution of sampled motion hypotheses, which are constrained to reflect known respiratory biomechanics. This prior ensures that the regression model does not predict implausible motion patterns.

- describe linear motion constraint

Linear motion constraint refers to the assumption that the tumor’s motion path is approximately linear along the primary direction of respiration, with minimal hysteresis. This assumption is used to limit the range of rotational and shear components during hypothesis generation.

- describe motion prior probability estimation

The motion prior is estimated from population-level studies of respiratory tumor motion and is implemented as bounds on the magnitude and direction of sampled motion hypotheses. These bounds are not learned from data but are fixed based on clinical literature.

- describe optimal tracking window

The optimal tracking window size is determined by evaluating the mean discriminative power of HOG feature distances across a range of candidate window sizes. The size that maximizes the mean of the 20% smallest feature distances is selected, as it best distinguishes the tumor region from its surroundings.

- describe self-similarity in tumor region

Self-similarity is quantified by computing the ℓ2 distance between all pairs of image patches within a search region surrounding the tumor. Regions with high self-similarity exhibit low feature distance variance, indicating homogeneity, while regions with high discriminative power exhibit high variance, indicating texture richness.

- describe HOG distance computation

HOG distance is computed as the ℓ2 norm between two HOG feature vectors extracted from corresponding image patches. This distance serves as a measure of visual dissimilarity and is used to evaluate the discriminative power of candidate tracking windows.

- describe discriminative-power score computation

The discriminative-power score is defined as the mean of the 20% smallest HOG distances within the search region. A higher score indicates greater texture distinctiveness and is used to select the optimal window size.

- describe 2D regression tracking

In the 2D regression mode, separate regression matrices are trained for each X-ray view. The tumor position in each view is tracked independently, and the 3D position is reconstructed by back-projecting the 2D estimates through the radiation source points and computing the midpoint of the shortest line segment connecting the two projection rays.

### Effect of the Invention

- describe accuracy of tracking results

The invention achieves an average 3D tracking error of 1.05 pixels on low-resolution, 8-bit DRR sequences derived from real patient 4DCT data, corresponding to a positioning accuracy of 92.5% relative to the maximum tumor displacement observed in the dataset. This performance is maintained across diverse tumor shapes, locations, and breathing patterns, including irregular respiration and low-contrast regions where other methods fail.

- describe speed of tracking method

The method processes each image pair in less than 0.03 seconds on a standard clinical workstation, enabling real-time tracking at frame rates exceeding 30 Hz. This speed is achieved without parallelization, optimization, or hardware acceleration, making it deployable on standard radiotherapy systems.

- compare with prior art methods

Compared to state-of-the-art optical flow and particle filtering methods, the invention reduces tracking error by more than 70% and increases processing speed by more than 5,000-fold. Optical flow methods require over two minutes per frame and exhibit systematic underestimation of large motions, while particle filters produce jittery trajectories and frequently lose track in low-texture regions.

- describe advantages of the invention

The invention provides a fully non-invasive, markerless, patient-adaptive, and real-time tumor tracking solution that requires no manual labeling, no offline training, no fiducial implants, and no patient-specific calibration. It is robust to low contrast, low resolution, and imaging noise, and operates with sub-pixel accuracy across a wide range of clinical scenarios. The method eliminates procedural risks, reduces treatment time, and enables margin reduction in radiation planning, thereby improving therapeutic outcomes and patient safety.