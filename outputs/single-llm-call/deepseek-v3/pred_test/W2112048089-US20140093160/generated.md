Here is the patent application following your outline and incorporating the research paper's invention:

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates to image-guided radiation therapy (IGRT), specifically to non-invasive methods for tracking tumors or other objects in real-time during radiation treatment without requiring fiducial markers. The invention provides a regression-based tracking system that correlates image features from orthogonal X-ray sequences with underlying tumor motion parameters, enabling highly accurate and computationally efficient position estimation.  

## BACKGROUND OF THE INVENTION  

Image-guided radiation therapy has become essential for treating tumors in the lung and abdominal regions, where respiratory motion causes significant displacement. Accurate tracking allows radiation beams to continuously target the tumor while minimizing exposure to surrounding healthy tissue.  

Conventional tumor tracking methods rely on implanted fiducial markers, which serve as internal surrogates for motion estimation. However, these markers require invasive surgical placement, risking complications such as pneumothorax. Additionally, fiducials may migrate over multiple treatment sessions, introducing positional uncertainty.  

Alternative approaches use external markers on the chest or abdomen to indirectly estimate tumor position through correlation models. However, respiratory biomechanics often violate these correlations, reducing tracking accuracy. Parametric motion models and template matching methods have also been explored, but these require extensive manual labeling or fail with low-contrast tumor regions.  

Existing markerless tracking techniques, such as optical flow or particle filters, suffer from computational inefficiency, sensitivity to noise, and inability to handle low-contrast regions. These limitations highlight the need for a robust, non-invasive tracking method that operates in real-time without patient-specific training or manual intervention.  

## SUMMARY OF THE INVENTION  

The invention provides a method for tracking tumors or other objects in real-time using regression-based motion estimation from orthogonal X-ray sequences. The method eliminates the need for fiducial markers by learning an online regression model that maps image features to tumor motion parameters.  

Key aspects include:  
- Generation of motion hypotheses constrained by respiratory biomechanics.  
- Extraction of discriminative image features, such as Histograms of Oriented Gradients (HOG).  
- Training a regression model using ridge regression on a Riemannian manifold to account for affine motion matrices.  
- Real-time position estimation through matrix multiplication, avoiding iterative optimization.  

The system supports both 3D regression (jointly modeling orthogonal views) and 2D regression (independently tracking each view). An adaptive window selection mechanism ensures optimal feature extraction.  

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS  

### Tumor Tracking Method  

The invention treats tumor tracking as a regression problem, where image features extracted from orthogonal X-ray sequences are linearly correlated with tumor motion parameters. The method operates in two phases: training and tracking.  

### Non-Invasive Tumor Tracking  

Unlike fiducial-based approaches, the invention relies solely on image features from X-ray sequences. No prior patient-specific data or manual annotations are required. The system initializes using the first pair of orthogonal X-ray images, where the tumor position is known (e.g., via table alignment).  

### Regression Model Fitting  

The regression model Ω is learned by minimizing the geodesic distance between motion hypotheses and their estimates:  
\[ \min_{\mathbf{\Omega}} ||\mathbf{X}\mathbf{\Omega} - \mathbf{Y}||^2 + \lambda||\mathbf{\Omega}||^2 \]  
where:  
- **X** is a matrix of concatenated feature vectors.  
- **Y** is a matrix of logarithmic motion hypotheses.  
- **λ** is a regularization parameter.  

The solution is obtained via ridge regression:  
\[ \mathbf{\Omega} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{Y} \]  

### Image Feature Extraction  

Histograms of Oriented Gradients (HOG) are used to describe tumor regions. Features are computed within adaptive windows, ensuring sensitivity to motion while remaining robust to low contrast.  

### Motion Constraints  

Motion hypotheses are generated under biomechanical constraints (e.g., limited translation/rotation) to reflect typical respiratory motion. This stabilizes tracking and reduces jitter.  

### 3D Setup Parameters  

For 3D regression, a joint model maps features from both orthogonal views directly to 3D affine motion parameters. The 3D affine matrix ΔM* is a 4×4 transformation estimated via:  
\[ \mathbf{\Omega}^* = (\mathbf{X}^{*\top}\mathbf{X}^* + \lambda\mathbf{I})^{-1}\mathbf{X}^{*\top}\mathbf{Y}^* \]  
where **X*** and **Y*** are constructed from combined orthogonal features and 3D motion hypotheses.  

### Initial Object Location and Shape  

The tumor's initial position and bounding box are defined in the first X-ray pair. Random affine hypotheses are generated around this location to train the regression model.  

### Alignment of Initial Sequences  

Orthogonal views are aligned using known geometry (e.g., source-detector positions). Features are extracted from corresponding regions in both views.  

### 3D Motion Parameters  

Tumor motion is modeled as a 4×4 affine matrix accounting for translation, rotation, scaling, and skew. Incremental updates are computed via Lie group exponential maps:  
\[ \Delta M_t = \exp(\mathbf{h}_t^\top \mathbf{\Omega}) \]  

### Training Module  

The training phase involves:  
1. Generating **n** random motion hypotheses.  
2. Extracting features for each hypothesis.  
3. Solving the ridge regression problem to obtain **Ω**.  

### Tracking Module  

For new X-ray pairs, the tumor position is estimated by applying the learned model to extracted features:  
\[ M_t = M_{t-1} \cdot \exp(\mathbf{h}_t^\top \mathbf{\Omega}) \]  

### Regression Function  

The function linearly maps features to motion parameters. For 2D regression, separate models are trained for each view, and 3D position is obtained via back-projection.  

### Feature Extraction  

HOG features are computed in 5×5 blocks within the tracking window, concatenated into a single vector.  

### Motion Parameters Estimation  

The current tumor position is derived from the previous position and incremental motion.  

### Motion Constraints Application  

Hypotheses are bounded by maximum displacement/rotation values typical for respiratory motion.  

### Window Update  

The tracking window adapts dynamically to maintain optimal feature discriminability.  

### 3D Regression Tracking  

Joint modeling of orthogonal views ensures consistency and improves speed (0.03 sec/frame).  

### 3D Motion Representation  

Affine matrices lie on a Riemannian manifold; distances are measured via geodesics.  

### Affine Transformation  

Tumor motion is approximated as affine to balance generality and computational efficiency.  

### Feature Vector Extraction  

HOG vectors are normalized to ensure invariance to intensity variations.  

### Training Set Construction  

Hypotheses are sampled uniformly within motion constraints to avoid overfitting.  

### Regression Function Training  

The model is trained once at initialization and optionally updated during tracking.  

### Tumor Motion Prior Probability  

Respiratory mechanics inform motion constraints (e.g., linear pathways during regular breathing).  

### Linear Motion Constraint  

For simplicity, translational motion is prioritized in hypothesis generation.  

### Motion Prior Probability Estimation  

Bounds are derived from clinical observations of tumor motion ranges.  

### Optimal Tracking Window  

Window size is selected based on self-similarity metrics to maximize feature discriminability.  

### Self-Similarity in Tumor Region  

Local patch distances quantify the uniqueness of image features for a given window size.  

### HOG Distance Computation  

ℓ₂ norms between HOG vectors determine feature similarity.  

### Discriminative-Power Score Computation  

Mean error of small distances (20% percentile) identifies optimal window sizes.  

### 2D Regression Tracking  

Independent tracking in each view (0.06 sec/frame) with 3D intersection for position estimation.  

## Effect of the Invention  

### Accuracy of Tracking Results  

Experiments on 10 patient datasets show an average error of 1.05 pixels (7.5% of maximum displacement), outperforming optical flow (3.57 pixels) and particle filters (5.01 pixels).  

### Speed of Tracking Method  

The 3D regression operates at 0.03 sec/frame, enabling real-time clinical use. Training completes in <0.05 sec.  

### Comparison with Prior Art  

The invention eliminates fiducial markers, manual labeling, and iterative optimization while improving accuracy and speed.  

### Advantages of the Invention  

- Non-invasive: No fiducials or external markers required.  
- Robust: Handles low-contrast tumors and imaging artifacts.  
- Efficient: Matrix multiplication enables real-time operation.  
- Generalizable: No patient-specific training needed.  

This method reduces treatment margins to millimeter precision, minimizing radiation exposure to healthy tissue.