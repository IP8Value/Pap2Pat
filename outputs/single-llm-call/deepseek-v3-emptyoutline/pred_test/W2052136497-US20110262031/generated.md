Here is the complete patent application following your outline and incorporating the research paper's content:

# DESCRIPTION  

## BACKGROUND  

The field of image-based rendering and modeling has long relied on visual hull techniques derived from "Shape-from-Silhouette" principles. Traditional visual hull methods construct approximate convex hulls containing target objects by processing multiple images captured from different viewpoints. While these methods offer advantages over geometry-based approaches in terms of data acquisition cost and realism of results, they suffer from two fundamental limitations that have persisted as unresolved challenges in the art.  

First, conventional visual hull reconstruction fails to accurately model object bottoms due to practical constraints in image acquisition. To properly reconstruct an object's complete geometry, reference images should ideally be captured from viewpoints surrounding the entire object, including its underside. However, simultaneously acquiring both surrounding views and bottom views proves physically impossible with standard setups, as the object must remain in a fixed position relative to the reference frame during image capture. Attempting to supplement the image set with separate bottom views introduces complex alignment problems between reference frames, significantly increasing computational overhead without guaranteeing accurate reconstruction.  

Second, visual hull methods inherently struggle with concave surface reconstruction due to limitations in silhouette-based analysis. The fundamental principle of visual hull construction—forming an intersection of silhouette cones—produces only convex approximations of object geometry. Concave features become indistinguishable in silhouette analysis, leading to distorted reconstructions where concave regions appear convex. This limitation proves particularly problematic for objects like cups, bowls, or archaeological artifacts where concave surface details carry important visual information.  

Prior attempts to address these limitations have explored mirror-based imaging systems. Some approaches employ multiple mirrors to capture several views simultaneously, while others utilize mirror systems to create stereo images from single cameras. However, these existing solutions fail to adequately solve either the bottom reconstruction problem or the concave surface problem. Mirror-based systems that capture multiple views typically require fixed relative positions between objects and mirrors, limiting flexibility in image acquisition. Moreover, none of these prior systems successfully integrate bottom views and concave surface views into a unified visual hull framework that maintains computational efficiency while improving reconstruction accuracy.  

## SUMMARY  

The present invention provides an innovative solution to both the bottom reconstruction problem and concave surface reconstruction problem in image-based visual hull methods through a novel image acquisition platform and associated processing algorithms. The system employs a carefully configured combination of planar glass and planar mirror elements to simultaneously capture both standard reference images and specialized views of object bottoms and concave surfaces within a single unified reference frame.  

Key aspects of the invention include:  

A unique imaging platform configuration where the target object rests on a planar glass surface positioned above a planar mirror. This arrangement enables simultaneous capture of both direct views (through the glass) and reflected views (via the mirror) of the object in a single image, providing complete visual information including bottom views without requiring physical movement of the object or complex post-capture alignment.  

An advanced virtual camera system that mathematically transforms reflected bottom views into equivalent standard camera views through symmetrical projection calculations. This transformation maintains all images in a common reference frame while enabling standard visual hull algorithms to process bottom views without modification. The virtual camera parameters are derived through efficient matrix transformations of real camera parameters, preserving computational efficiency.  

An innovative concave surface approximation technique that introduces "negative" silhouette cones to counteract the convex bias of traditional visual hull methods. By capturing specialized top-down views of concave regions and processing them through virtual cameras positioned within the concave space, the system generates corrective geometry that accurately reconstructs concave features.  

The complete system maintains the computational efficiency of traditional image-based visual hull methods while significantly improving reconstruction accuracy for both bottom surfaces and concave features. The platform requires no specialized equipment beyond standard cameras, planar glass, and mirrors, making it practical for diverse applications including archaeological documentation, artifact preservation, and commercial product visualization.  

## DETAILED DESCRIPTION  

The present invention provides a comprehensive solution for accurate visual hull reconstruction incorporating both bottom surfaces and concave features through three primary innovations: (1) a novel image acquisition platform, (2) virtual camera transformations for bottom view processing, and (3) specialized concave surface approximation algorithms.  

**Image Acquisition Platform Configuration**  

The core imaging platform consists of three essential components arranged in precise spatial relationship:  

1. A planar glass surface serving as the object support platform, preferably constructed of optical-quality glass with minimal thickness (approximately 5mm) to reduce refractive distortion. The glass dimensions should sufficiently accommodate the target object (e.g., 800×800mm for medium-sized objects).  

2. A planar mirror positioned parallel to and below the glass platform at a variable distance. The mirror surface should maintain high reflectivity across visible wavelengths, with recommended dimensions of approximately 500×500×5mm for typical applications.  

3. One or more digital cameras positioned above the glass platform at various angles to capture both direct views through the glass and reflected views from the mirror. Each camera incorporates a circular polarizing lens to minimize unwanted reflections from the glass surface.  

The platform establishes a fixed coordinate system where the X-Z plane coincides with the mirror plane, and the Y-axis extends vertically upward through the glass platform. This coordinate convention simplifies subsequent geometric transformations while accommodating variable spacing between glass and mirror. Calibration targets (such as colored concentric circles) placed on the mirror surface enable standard camera calibration procedures using established P4P (Perspective-n-Point) methods.  

**Virtual Camera System for Bottom Reconstruction**  

The platform's mirror reflection creates virtual camera positions that mathematically correspond to physical cameras positioned beneath the object. For each physical camera at position p with orientation parameters (α, β, γ), the system calculates a corresponding virtual camera at position p' using mirror symmetry transformations:  

The translation vector T' for the virtual camera derives from the physical camera's translation vector T through reflection about the mirror plane (Y=0). If T = [t_x, t_y, t_z], then T' = [t_x, -t_y, t_z].  

The virtual camera's rotation matrix R' combines axis-specific transformations of the physical camera's rotation angles. For original rotation angles (α, β, γ) about X, Y, and Z axes respectively, the virtual camera employs angles (-α, β, -γ) to maintain proper view orientation after reflection.  

These transformations convert reflected bottom images into equivalent standard camera views that integrate seamlessly with conventional visual hull processing. Each captured image separates into two components: a direct view (I_t) containing standard reference imagery, and a reflected view (I_b) containing bottom information. The reflected view undergoes vertical flipping to match its corresponding virtual image I_b' before processing.  

**Bottom Rendering Algorithm**  

The improved visual hull algorithm incorporates bottom views through a multi-stage rendering process:  

1. For viewpoints above the estimated bottom plane height (calculated as the average Y-coordinate of reconstructed bottom edge points), standard IBVH rendering proceeds unchanged.  

2. For viewpoints below the bottom plane, each pixel's viewing ray undergoes intersection testing with the bottom plane. If the ray intersects the bottom plane, the system:  
   a) Calculates the intersection point between viewing ray and bottom plane  
   b) Projects this point onto all reference images (both physical and virtual)  
   c) Computes a weighted average of corresponding pixel colors, favoring nearer reference views  
   d) Assigns the blended color to the output pixel  

This approach effectively "pushes" erroneously reconstructed bottom points back onto the geometrically correct bottom plane while maintaining photorealistic texture mapping.  

**Concave Surface Approximation**  

For concave feature reconstruction, the system employs a specialized image capture and processing pipeline:  

1. A dedicated top-down reference image (I_0) captures the concave region silhouette under optimal lighting conditions.  

2. A virtual camera (C_v) positions mathematically within the concave space, with parameters calculated from:  
   - Focal length adjusted to fully encompass the concave region  
   - Rotation angles transformed from the physical camera's orientation  
   - Position determined by concave geometry analysis  

3. The concave silhouette from I_0 projects onto C_v's image plane to create a virtual silhouette image (I'_c) representing the "negative" silhouette cone.  

During rendering, concave region pixels undergo specialized processing:  

1. The viewing ray projects onto I'_c to determine concave intersection intervals (set A)  
2. Standard IBVH processing generates conventional intersection intervals (set B)  
3. Set subtraction (B - A) produces corrected geometry for the concave region  
4. Texture mapping proceeds using weighted blending of reference views  

This approach effectively carves out the concave space from the convex visual hull while maintaining computational efficiency through 2D image-space operations.  

The complete system represents a significant advance in visual hull technology, solving long-standing problems in bottom and concave surface reconstruction through innovative but practical combinations of optical configuration and algorithmic processing. The invention maintains the efficiency benefits of image-based visual hull methods while dramatically improving reconstruction accuracy for previously problematic geometries.