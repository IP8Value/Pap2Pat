Here is the complete patent application following the provided outline and incorporating the research paper's content:

---

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates generally to medical imaging and radiation therapy systems, and more specifically, to a noninvasive method and system for real-time tumor tracking during image-guided radiation therapy (IGRT). The invention utilizes regression-based motion estimation from orthogonal X-ray video sequences to precisely track tumor movement in three-dimensional space without requiring implanted fiducial markers or external surrogates. The disclosed system achieves high accuracy in determining tumor position while operating at speeds suitable for clinical real-time applications.  

## BACKGROUND OF THE INVENTION  

Conventional approaches to tumor tracking in radiation therapy face significant limitations that impact both treatment efficacy and patient safety. Existing methods typically rely on either invasive internal fiducials or external surrogate markers, each presenting distinct disadvantages. Internal fiducials, whether passive metallic markers or active electromagnetic transponders, require surgical implantation procedures that carry risks including tissue damage, pneumothorax during placement, and potential marker migration over multiple treatment sessions. These factors introduce uncertainty in reference positions and compromise tracking accuracy.  

Alternative approaches using external surrogates attempt to correlate chest or abdominal surface movements with internal tumor position through correspondence models. However, the complex biomechanics of respiration frequently violate these assumed correlations, leading to tracking inaccuracies. Parametric motion models and template matching techniques present additional challenges, as they require extensive manual labeling for training and often fail when tracking low-contrast tumors in poor quality images.  

Prior attempts at noninvasive tracking have employed computationally intensive methods such as optical flow algorithms, particle filters, and iterative optimization techniques. These approaches suffer from either insufficient accuracy (particularly for large tumor displacements) or impractical computational demands that prevent real-time operation in clinical settings. There exists a clear need for a tracking system that combines high positional accuracy with computational efficiency while eliminating the risks associated with invasive marker placement.  

## SUMMARY OF THE INVENTION  

The present invention provides a novel solution to the aforementioned problems through a regression-based tumor tracking system that operates on orthogonal X-ray video sequences. The system comprises three principal innovations: (1) a manifold-aware regression model that maps image features to tumor motion parameters while respecting the Riemannian geometry of affine transformations; (2) a joint feature representation from orthogonal views that enables direct 3D position estimation; and (3) an adaptive window sizing mechanism that optimizes tracking performance for individual patient anatomy.  

At the core of the invention lies the mathematical formulation that treats tumor tracking as a regression problem on a Riemannian manifold. The system generates multiple affine motion hypotheses around an initial tumor location, constrained by respiratory biomechanics to efficiently cover the parameter space. These hypotheses are projected onto orthogonal X-ray planes where discriminative image features (such as Histograms of Oriented Gradients) are extracted. A ridge regression model then learns the optimal mapping between feature vectors and motion parameters using geodesic distances on the manifold of affine transformations.  

The tracking process operates in two distinct phases: initialization and continuous tracking. During initialization, the system learns the regression model from the first frame pair where tumor position is known. For subsequent frames, tracking requires only feature extraction and matrix multiplication, enabling real-time operation. The system maintains tumor position estimates in 3D space either through direct 3D regression (3DR) or by intersecting independently tracked 2D positions from orthogonal views (2DR).  

Key advantages over prior art include elimination of invasive procedures, robustness to low-contrast tumors, computational efficiency suitable for real-time clinical use, and consistent sub-millimeter accuracy across varying respiratory patterns. The invention represents a significant advancement in precision radiation therapy by enabling continuous tumor tracking without the safety concerns and practical limitations of existing approaches.  

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS  

The preferred embodiments of the invention will now be described with reference to the mathematical formulations and clinical implementation details. Figure 1 illustrates the overall system architecture, while Figure 2 shows the processing flow from image acquisition to 3D position estimation.  

**Initialization Phase:**  
The system begins by acquiring orthogonal X-ray image pairs (typically anterior-posterior and lateral views) with known initial tumor position M₀. For the 3DR embodiment, the processor generates n random 3D affine motion hypotheses ΔMᵢ* within biomechanical constraints (e.g., limited translation/rotation magnitudes characteristic of respiratory motion). Each 4×4 hypothesis matrix is projected to both X-ray views, creating corresponding 2D regions. Within these regions, the system extracts and concatenates feature vectors (h₀,₁¹; h₀,₁²) using Histograms of Oriented Gradients computed over 5×5 pixel blocks.  

The regression model Ω* is learned by solving the minimization problem:  

min_Ω ||X*Ω - Y*||² + λ||Ω||²  

where X* = [(h₀,₁¹;h₀,₁²)ᵀ; ... ; (h₀,ₙ¹;h₀,ₙ²)ᵀ] contains concatenated feature vectors, Y* = [(log ΔM₁*)ᵀ; ... ; (log ΔMₙ*)ᵀ] contains logarithmically mapped motion matrices, and λ provides Tikhonov regularization. The solution Ω* = (X*ᵀX* + λI)⁻¹X*ᵀY* is computed via ridge regression.  

**Continuous Tracking Phase:**  
For each new X-ray pair at time t > 0, the system:  
1. Extracts features hₜ¹ and hₜ² within previous tumor windows  
2. Computes incremental motion ΔMₜ = exp((hₜ¹;hₜ²)ᵀΩ*)  
3. Updates tumor position as Mₜ = Mₜ₋₁·ΔMₜ  

The 2DR embodiment operates similarly but learns separate models Ω¹ and Ω² for each view, then intersects back-projected positions to obtain 3D estimates.  

**Optimal Window Selection:**  
An adaptive mechanism determines the optimal tracking window size wₓ × wᵧ by analyzing local self-similarity. For candidate sizes, the system:  
1. Defines a 2wₓ × 2wᵧ search region centered on the tumor  
2. Computes feature distances between all patch pairs  
3. Selects size maximizing discriminative power (preferring smaller distances)  

Typical optimal windows range from 0.9× to 1.3× the tumor bounding box size across different views and patients.  

### Effect of the Invention  

The disclosed tumor tracking system provides multiple clinical and technical benefits compared to existing approaches:  

1. **Noninvasive Operation:** By eliminating fiducial markers, the invention removes risks associated with implantation procedures including tissue damage and pneumothorax. Patients experience improved safety and comfort during treatment.  

2. **Superior Tracking Accuracy:** Experimental results demonstrate average tracking errors of 1.05 pixels (7.5% of maximum displacement), corresponding to sub-millimeter precision in clinical imaging systems. This enables significant reduction of treatment margins from centimeters to millimeter range.  

3. **Computational Efficiency:** The regression approach requires only 0.03-0.06 seconds per frame (including feature extraction and matrix multiplication), enabling real-time operation at >30 fps. This represents a 100× speed improvement over optical flow methods.  

4. **Robustness to Challenging Conditions:** The system maintains accuracy for low-contrast tumors, irregular breathing patterns, and imaging artifacts that compromise conventional trackers. The manifold-aware formulation prevents error accumulation common in Euclidean space methods.  

5. **Clinical Practicality:** With no requirement for patient-specific training or manual annotation, the system integrates seamlessly into existing IGRT workflows. The consistent performance across varying tumor locations and respiratory dynamics makes it suitable for broad clinical adoption.  

The invention's technical advancements directly translate to improved radiation therapy outcomes - enabling higher dose delivery to tumors while sparing healthy tissue, ultimately improving cancer treatment efficacy and patient quality of life.  

--- 

This complete patent application maintains all required sections from the outline while expanding each to meet the specified word count requirements. The language follows formal patent drafting conventions, and the technical descriptions provide sufficient detail to support broad claims while enabling implementation by those skilled in the art.