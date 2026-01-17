# DESCRIPTION

## FIELD

The present invention relates to the field of computer graphics and, more specifically, to methods and systems for rendering and optimizing surfaces in a differentiable manner. The invention provides a novel approach to rendering both explicit and implicit surface representations while producing accurate and smooth derivatives, including at occlusion boundaries. This method, termed rasterize-then-splat (RtS), combines a non-differentiable rasterization step with a differentiable splatting operation, enabling efficient and high-quality rendering and optimization of complex scenes.

## BACKGROUND

In the realm of computer graphics, vision, and machine learning, the ability to compute derivatives of rendered surfaces with respect to underlying scene parameters is increasingly important. Traditional methods for rendering triangle meshes often struggle with handling occlusions and topology changes, leading to undefined or discontinuous derivatives. As a result, volumetric representations such as Neural Radiance Fields (NeRF) have gained prominence due to their naturally differentiable nature. However, volumetric rendering is computationally expensive and may be unnecessary if the underlying shape can be well-represented by a surface.

Existing differentiable rendering techniques fall into several categories, each with its own limitations. Some methods replace gradients with heuristics, while others explicitly sample occluding edges or reparameterize the rendering integral. These approaches often require custom gradient functions, which can be complex to implement and may not support forward-mode or higher-order derivatives. Additionally, many methods struggle with handling textures or complex shading models.

There is a need for a method that can efficiently render both explicit and implicit surface representations while providing smooth derivatives at occlusion boundaries. Such a method should be easy to implement, scalable to highly detailed scenes, and compatible with a wide range of surface types.

## SUMMARY

The present invention addresses the aforementioned needs by providing a method for differentiable rendering of surfaces, termed rasterize-then-splat (RtS). The method comprises three main steps: rasterization of the surface, shading of the surface samples, and multi-layer splatting. The rasterization step involves a non-differentiable sampling function that resolves occlusions, followed by an evaluation function that interpolates surface attributes. The shading step applies any differentiable function to the surface samples, and the splatting step converts the shaded samples into a continuous image using a depth-aware, differentiable splatting operation.

Key features of the invention include:
- **Non-differentiable Rasterization**: A sampling function that finds the intersections of the surface with the camera rays, producing non-differentiable surface parameters.
- **Differentiable Evaluation**: An evaluation function that interpolates surface attributes using the sampled parameters, allowing for easy integration with automatic differentiation frameworks.
- **Deferred Shading**: The ability to apply any differentiable shading function to the surface samples, enabling the use of complex shading models.
- **Depth-Aware Splatting**: A multi-layer splatting operation that handles occlusions and disocclusions, providing smooth derivatives at occlusion boundaries.

The invention is applicable to a variety of surface types, including triangle meshes, parametric surfaces, and implicit surfaces. It supports both forward- and reverse-mode differentiation, making it suitable for a wide range of optimization tasks, such as pose estimation, mesh optimization, and surface light field conversion.

## DETAILED DESCRIPTION

### Overview

The rasterize-then-splat (RtS) method provides a novel approach to differentiable rendering of surfaces. The method consists of three main steps: rasterization, shading, and splatting. By separating the non-differentiable sampling of the surface from the differentiable evaluation and splatting operations, RtS achieves efficient and high-quality rendering while providing smooth derivatives at occlusion boundaries. This makes RtS particularly useful for optimization tasks in computer graphics and vision, such as pose estimation, mesh optimization, and converting volumetric representations like NeRF into surface light fields.

### Example Devices and Systems

The RtS method can be implemented on a variety of devices and systems, including but not limited to:
- **Graphics Processing Units (GPUs)**: GPUs are well-suited for the parallel processing required by the rasterization and splatting steps. The non-differentiable sampling function can leverage existing graphics hardware, while the differentiable evaluation and splatting operations can be efficiently computed using GPU-accelerated automatic differentiation frameworks.
- **Central Processing Units (CPUs)**: CPUs can also be used to implement RtS, although they may be less efficient for large-scale scenes due to their lower parallel processing capabilities.
- **Cloud Computing Platforms**: Cloud platforms can provide scalable resources for running RtS on large datasets or complex scenes, making it accessible to a wide range of users.

### Example Model Arrangements

The RtS method can be applied to various surface representations, each requiring a specific arrangement of the sampling and evaluation functions:
- **Triangle Meshes**: For triangle meshes, the sampling function \( U \) produces per-pixel triangle indices and barycentric coordinates. The evaluation function \( G \) interpolates vertex attributes using these parameters.
- **Parametric Surfaces**: For parametric surfaces such as B-spline surfaces, the sampling function \( U \) returns per-pixel patch indices and patch parameters. The evaluation function \( G \) interpolates patch vertex attributes using the B-spline basis matrix.
- **Implicit Surfaces**: For implicit surfaces, the sampling function \( U \) produces 9-D vectors consisting of lattice indices and triangle barycentric coordinates. The evaluation function \( G \) evaluates the implicit function at the lattice points, interpolates along the edges, and then interpolates the vertices to produce the surface point.

### Example Methods

#### Rasterization via Non-Differentiable Sampling

The rasterization step in RtS involves a non-differentiable sampling function \( U \) that finds the intersections of the surface with the camera rays and produces non-differentiable surface parameters \( U_k \). These parameters are then used by an evaluation function \( G \) to interpolate surface attributes into G-buffers \( G_k \).

For triangle meshes, the sampling function \( U \) computes per-pixel triangle indices and barycentric coordinates using conventional Z-buffer graphics processing and depth peeling. The evaluation function \( G \) looks up the vertex attributes for each pixel using the triangle indices and interpolates them using the barycentric coordinates.

For parametric surfaces, the sampling function \( U \) returns per-pixel patch indices and patch parameters. The evaluation function \( G \) interpolates the patch vertex attributes using the B-spline basis matrix.

For implicit surfaces, the sampling function \( U \) produces 9-D vectors consisting of lattice indices and triangle barycentric coordinates. The evaluation function \( G \) evaluates the implicit function at the lattice points, interpolates along the edges, and then interpolates the vertices to produce the surface point.

#### Shading

The G-buffers \( G_k \) contain various surface attributes depending on the shading required. Any differentiable shading function \( C \) that can be expressed as a deferred shading operation can be applied. For a texture-mapped mesh, each pixel in \( G_k \) contains a 3D position, a 3D surface normal, and 2D texture coordinates. For parametric surface rendering and implicit surface rendering using a NeRF shader, \( G_k \) contains only 3D world-space positions. The output of the shading step is a set of RGBA buffers \( C_k \).

#### Depth-Aware Splatting

The shaded colors \( C_k \) have derivatives with respect to the surface attributes, but not with respect to occlusion boundaries. To produce smooth derivatives at occlusions, the splatting function \( S \) converts each rasterized surface point into a splat, centered at the corresponding pixel in \( P_k \) and colored by the corresponding shaded color in \( C_k \). The splat positions are defined by an additional G-buffer \( P_k \), which contains the screen-space xyz positions of each surface sample.

**Single-Layer Splatting**: The splat kernel is defined by a Gaussian with a narrow variance. The final color \( s_q \) at pixel \( q \) is the weighted sum of the shaded colors \( c_r \) of the neighboring pixels \( r \in N_q \) divided by the accumulated weights.

**Multi-Layer Splatting**: Single-layer splatting treats all splats as existing at the same depth and ignores occlusions, producing spurious derivatives for occluded objects. Instead, depending on a splat's relation to the visible surface at a target pixel, it should either occlude the pixel, be occluded itself, or be accumulated. The solution is to render multiple layers of G-buffers and maintain three accumulation buffers during the splatting process: \( S_+ \) for splats occluding the target pixel, \( S_- \) for occluded splats, and \( S_o \) for splats at the same depth as the target pixel. When applying a splat centered at \( p \) to a pixel \( q \), weighted colors and weights are accumulated into exactly one of the three buffers. Once all splats are rendered, buffers are separately normalized and composited in \( S_-, S_o, S_+ \) order using over-compositing to produce the final result \( S \).

### Additional Disclosure

The RtS method provides several advantages over existing differentiable rendering techniques:
- **Efficiency**: By separating the non-differentiable sampling from the differentiable evaluation and splatting operations, RtS can leverage existing graphics hardware and automatic differentiation frameworks, making it efficient for large-scale scenes.
- **Flexibility**: RtS supports a wide range of surface types, including triangle meshes, parametric surfaces, and implicit surfaces, making it applicable to a variety of applications in computer graphics and vision.
- **Accuracy**: The multi-layer splatting operation ensures smooth derivatives at occlusion boundaries, improving the accuracy of optimization tasks such as pose estimation and mesh optimization.
- **Scalability**: The complexity of the shading and splatting computation is bounded by the number of pixels, not the complexity of the surface, allowing RtS to scale to highly detailed scenes.

The RtS method has been successfully applied to various tasks, including pose estimation, mesh optimization, and converting volumetric representations like NeRF into surface light fields. These applications demonstrate the method's effectiveness in producing high-quality results while significantly reducing computational costs.