# DESCRIPTION

## FIELD

- relate to differentiable rendering

The present invention relates to systems and methods for generating differentiable two-dimensional renderings from three-dimensional geometric representations, particularly in the context of computer graphics, computer vision, and machine learning applications. The invention enables the computation of smooth, accurate gradients of rendered pixel values with respect to parameters of three-dimensional shapes, including vertex positions, surface normals, texture coordinates, camera pose, and implicit field parameters, even at occlusion boundaries where traditional rasterization methods fail to produce meaningful derivatives. Unlike prior approaches that rely on custom gradient functions, edge sampling, or volumetric rendering, the disclosed method decouples the non-differentiable sampling of surface geometry from the differentiable computation of image-space contributions, thereby enabling seamless integration with automatic differentiation frameworks such as TensorFlow and PyTorch. This architecture supports both forward-mode and reverse-mode differentiation without modification, allowing for efficient optimization of complex scene parameters using gradient-based algorithms such as Levenberg-Marquardt and Adam. The method is applicable to a wide range of surface representations, including triangle meshes, parametric surfaces such as B-splines and Bézier patches, and implicit surfaces defined by isosurfaces of scalar fields, making it uniquely versatile across domains ranging from inverse rendering and 3D reconstruction to neural radiance field optimization and pose estimation.

## BACKGROUND

- introduce limitations of contemporary solutions

Contemporary approaches to differentiable rendering face fundamental limitations when attempting to compute gradients with respect to three-dimensional geometry, particularly at occlusion boundaries where surface topology changes abruptly. Traditional methods that rely on explicit sampling of silhouette edges, such as Redner and nvdiffrast, incur computational costs that scale with the number of mesh edges, rendering them impractical for high-complexity scenes with millions of triangles. Other techniques, including Soft Rasterizer and Neural Mesh Renderer, modify the forward rendering process to produce smooth visibility functions, but do so at the cost of introducing artifacts, requiring expensive closest-point queries, or failing to support complex shading models and textures. These approaches often produce inconsistent gradients that diverge from the true derivative of the rendered image, leading to unstable convergence during optimization. Furthermore, volumetric rendering methods such as Neural Radiance Fields provide differentiable rendering by construction but require hundreds of ray samples per pixel, resulting in prohibitive computational overhead that limits their utility in real-time or iterative optimization settings. Methods that attempt to differentiate through Marching Cubes or other isosurface extraction algorithms are hindered by singularities in the gradient when neighboring grid values are nearly identical, preventing reliable optimization of implicit surface parameters. Additionally, existing solutions rarely support multi-layered geometry or self-occlusions in a physically consistent manner, and none provide a unified framework capable of handling triangle meshes, parametric surfaces, and implicit surfaces within the same differentiable pipeline. As a result, researchers are forced to choose between computational efficiency, geometric fidelity, and differentiability, with no existing solution offering all three simultaneously.

## SUMMARY

- introduce method for differentiable rendering
- obtain three-dimensional mesh
- rasterize three-dimensional mesh
- determine initial color values
- construct splats
- determine updated color values
- introduce computing system
- obtain three-dimensional mesh
- rasterize three-dimensional mesh
- determine initial color values
- construct splats
- determine updated color values
- introduce non-transitory computer-readable media

The invention introduces a novel method for generating differentiable two-dimensional renderings by separating the non-differentiable sampling of surface geometry from a differentiable image-space splatting operation. The method begins by obtaining a three-dimensional mesh or other surface representation, which may be defined by vertices and faces, parametric control points, or an implicit scalar field. This representation is then rasterized using a conventional, non-differentiable rendering pipeline to produce a set of screen-space sample points, each associated with interpolated surface attributes such as position, normal, and texture coordinates. Initial color values are determined for each sample point by applying a differentiable shading function, which may include texture lookup, neural network evaluation, or physically based lighting models, using only the rasterized attributes and not the original surface geometry. For each pixel in the output image, one or more splats are constructed, each centered at the screen-space position of a corresponding sample and weighted according to a spatially decaying kernel, typically a Gaussian with standard deviation of approximately half a pixel. These splats are accumulated in a multi-layered fashion, distinguishing between splats that occlude the pixel, are occluded by it, or lie at coincident depth, ensuring accurate derivative propagation across occlusion boundaries. Updated color values for each pixel are computed as a normalized weighted sum of the shaded colors from all contributing splats, with normalization factors computed over a local neighborhood to preserve gradient consistency between forward and backward passes. The entire process is implemented using standard automatic differentiation libraries, requiring no custom gradient kernels or manual Jacobian derivation. The invention further encompasses a computing system comprising a processor and memory configured to execute instructions that implement the foregoing steps, including the generation of three-dimensional meshes, rasterization, shading, splat construction, and color accumulation. The system may be embodied in a server, workstation, or mobile device, and may be coupled to a non-transitory computer-readable medium storing executable instructions that, when executed, cause the system to perform the method, thereby enabling efficient, scalable, and accurate differentiable rendering for inverse rendering, 3D reconstruction, and machine learning applications.

## DETAILED DESCRIPTION

### Overview

- introduce differentiable rendering
- relate to splat-based forward rendering
- describe three-dimensional mesh
- explain rasterization
- generate coordinates for pixels
- determine initial color value
- construct splat for each pixel
- preserve derivatives for vertices
- compute splat center points
- apply perspective division and viewport transformation
- determine updated color value
- weight splats based on proximity
- generate differentiable two-dimensional rendering
- utilize rendering for smooth derivatives
- describe three-dimensional mesh generation
- relate to machine-learned model output
- explain rasterization scheme
- determine initial color value
- apply shading and/or texturing scheme
- construct splat with smooth falloff
- compute weight of splat
- determine updated color value
- select subset of splats
- weigh splats based on distance
- generate differentiable two-dimensional rendering
- utilize rendering for smooth derivatives
- describe machine-learned model training
- evaluate loss function
- adjust model parameters
- describe machine-learned pose estimation model
- generate image data
- describe machine-learned three-dimensional mesh generation model
- generate second three-dimensional mesh
- compare to point-based rendering
- explain limitations of rasterization
- describe technical effects and benefits
- enable generation of differentiable rendering
- reduce computational resources
- provide accurate derivatives

Differentiable rendering is a computational technique that enables the optimization of three-dimensional scene parameters by computing gradients of a rendered two-dimensional image with respect to those parameters. The present method achieves this through a two-stage process: rasterization followed by splatting. The three-dimensional mesh, whether explicit or implicit, is first processed by a non-differentiable rasterization engine that resolves visibility and produces a set of screen-space samples, each associated with interpolated surface attributes. These samples are not differentiated through directly; instead, their attributes are used to compute initial color values via deferred shading, which may involve texture maps, neural networks, or analytic lighting models. Each sample is then converted into a splat—a small, differentiable disk centered at its screen-space position—with a spatially decaying weight function that ensures smooth transitions across pixel boundaries. The splat center points are computed by applying perspective division and viewport transformation to the world-space positions of the surface samples, ensuring that any change in the underlying geometry directly influences the position of the splat and thus the final image. The weight of each splat is determined by a Gaussian kernel with a small standard deviation, ensuring that only neighboring pixels are affected, while normalization over a 3×3 neighborhood ensures that the forward rendering and its gradient remain consistent. A subset of splats is selected for each pixel based on depth layering, distinguishing between those that occlude, are occluded, or coincide with the target pixel’s surface, thereby preserving correct visibility relationships. The updated color value for each pixel is computed as a weighted sum of the splat contributions, normalized by the sum of weights to prevent gradient bias. This process generates a differentiable two-dimensional rendering that retains the sharpness of traditional rasterization while providing smooth, accurate derivatives at occlusion boundaries, enabling optimization of complex geometries with minimal computational overhead. The three-dimensional mesh may be generated by a machine-learned model, such as a neural network that predicts vertex positions from image inputs, and the rendered output may be used to train the model by comparing it to ground truth images and backpropagating errors through the differentiable rendering pipeline. Unlike point-based rendering methods, which suffer from under- or over-sampling as optimization progresses, this method resamples the surface at every iteration, ensuring consistent coverage. The technical effect is the ability to generate differentiable renderings with the efficiency of rasterization and the gradient fidelity of volumetric methods, reducing computational resources by orders of magnitude compared to raymarching while providing accurate derivatives even for highly detailed or topologically complex surfaces.

### Example Devices and Systems

- introduce computing system 100
- describe user computing device 102
- detail processor 112 and memory 114
- explain data 116 and instructions 118
- motivate two-dimensional differentiable rendering 124
- describe rasterization of three-dimensional mesh
- generate coordinates for pixels
- determine initial color value for pixels
- construct splat for each pixel
- determine updated color value for pixels
- generate two-dimensional differentiable rendering 124
- generate derivative for splat
- discuss server computing system 130
- describe processor 132 and memory 134
- explain data 136 and instructions 138
- introduce machine-learned models 120
- describe neural networks
- discuss training of machine-learned models 120
- introduce server computing system 130
- describe machine-learned models 140
- discuss training computing system 150
- describe processor 152 and memory 154
- explain data 156 and instructions 158
- introduce model trainer 160
- discuss training of machine-learned models 120 and 140
- describe backwards propagation of errors
- discuss generalization techniques
- introduce training data 162
- describe ground truth data
- discuss machine-learned pose estimation model
- evaluate loss function
- discuss personalizing model
- introduce network 180
- describe communication over network 180
- discuss alternative computing systems
- illustrate example computing devices

The invention is implemented on a computing system comprising a processor and memory, wherein the processor executes instructions stored on a non-transitory computer-readable medium to perform the differentiable rendering method. A user computing device, such as a desktop workstation or mobile device, may include a central processing unit and memory configured to receive a three-dimensional mesh, rasterize it into a pixel grid, compute initial color values using deferred shading, construct splats with spatially varying weights, and accumulate these splats to generate a differentiable two-dimensional rendering. The system may further generate derivatives of the splat contributions with respect to vertex positions, camera parameters, or neural network weights, enabling gradient-based optimization. In a server-based implementation, a server computing system may host machine-learned models that generate three-dimensional meshes from input images or predict camera poses, wherein the differentiable rendering pipeline serves as a differentiable layer within the model architecture. These machine-learned models may be neural networks trained using supervised learning, with training data comprising pairs of input images and corresponding ground truth three-dimensional meshes or poses. The server system includes a model trainer that performs backward propagation of errors from the rendered output to the model parameters, adjusting weights to minimize a loss function defined as the difference between rendered and target images. Generalization techniques such as dropout, data augmentation, and regularization are applied to prevent overfitting. The system may communicate over a network to distribute rendering tasks, synchronize model updates, or transmit rendered outputs to remote clients. Training may occur on a dedicated training computing system with high-performance graphics processors and large memory capacity, enabling batched processing of multiple scenes. Alternative computing systems, including embedded devices, cloud instances, or distributed clusters, may be configured to execute the method, with variations in memory allocation, parallelization strategy, or precision of floating-point operations depending on deployment context.

### Example Model Arrangements

- depict splat construction for a one-dimensional line segment
- define barycentric coordinates
- construct splat at splat position
- update pixel color value
- generate derivative from splat
- define derivative of weight of splat
- determine sign of derivative
- increase weight of splat
- decrease weight of splat
- generate derivative with respect to v2
- construct splat for pixel
- generate derivatives based on splat location
- depict data flow diagram of method for generating two-dimensional differentiable rendering
- obtain three-dimensional mesh
- rasterize three-dimensional mesh
- obtain two-dimensional raster
- determine initial color values for pixels
- apply shading data and/or texture data
- construct splat for each pixel
- compute coordinates at which splats are constructed
- determine updated color value for each pixel
- generate two-dimensional differentiable rendering
- find smooth derivatives at occlusion boundaries
- depict data flow diagram of method for training machine-learned model
- obtain two-dimensional differentiable rendering and 3D mesh training data
- generate derivatives for respective splats
- process two-dimensional differentiable rendering with machine-learned model
- generate machine-learned output
- evaluate loss function
- adjust parameters of machine-learned model
- generate machine-learned output using machine-learned model
- evaluate differences between machine-learned output and training data
- generate parameter adjustments to optimize machine-learned model
- depict entity represented by three-dimensional mesh

The splat construction process is illustrated through a one-dimensional line segment, where each sample point along the segment is projected into screen space and converted into a splat with a Gaussian weight profile. Barycentric coordinates are used to interpolate vertex attributes during rasterization, and the splat center is computed by transforming these interpolated positions through the camera’s projection matrix. When a splat is applied to a pixel, its weight is computed based on the Euclidean distance between the splat center and the pixel center, and the pixel’s color value is updated by accumulating the weighted contributions of all splats overlapping its neighborhood. The derivative of the splat weight with respect to a vertex position is computed analytically using the chain rule, with the sign of the derivative determined by the direction of movement of the splat center relative to the pixel. An increase in weight occurs when the splat moves toward the pixel center, and a decrease occurs when it moves away, ensuring that the gradient correctly reflects the geometric influence of the vertex. Derivatives with respect to vertex v2 are computed by propagating the gradient through the barycentric interpolation and projection steps, preserving the dependency of the splat location on the original geometry. The data flow diagram for generating a differentiable rendering begins with the input of a three-dimensional mesh, proceeds through rasterization to generate a two-dimensional raster, applies shading and texture data to compute initial color values, constructs splats at computed screen-space coordinates, and accumulates them to determine updated color values, culminating in a differentiable two-dimensional rendering. Smooth derivatives at occlusion boundaries are achieved by the multi-layer splatting scheme, which distinguishes between occluding, occluded, and coincident splats. For training a machine-learned model, the system obtains a set of training data comprising rendered images and corresponding ground truth three-dimensional meshes. The model processes the rendered output, generates a predicted mesh or pose, and the loss function evaluates the difference between the predicted and target renderings. Derivatives of the splats are used to backpropagate error gradients through the rendering pipeline to adjust the model parameters, enabling end-to-end optimization of the entire system from image to geometry.

### Example Methods

- obtain three-dimensional mesh
- generate three-dimensional mesh using machine-learning techniques
- rasterize three-dimensional mesh
- obtain two-dimensional raster
- determine initial color value for each pixel
- apply shading and/or texture data
- construct splat for each pixel
- compute splat center points
- apply perspective division and viewport transformation
- determine updated color value for each pixel
- weight splats based on proximity
- generate two-dimensional differentiable rendering
- select subset of splats for each pixel
- weigh splats based on distance
- normalize weights
- compute updated color value
- generate derivatives at occlusion boundaries
- process two-dimensional differentiable rendering with machine-learned model
- generate machine-learned output
- evaluate loss function
- adjust machine-learned model parameters
- train machine-learned pose estimation model
- train machine-learned three-dimensional mesh generation model
- generate image data with second pose/orientation
- generate second three-dimensional mesh
- allow for training of machine-learned models
- make reference to servers, databases, software applications
- discuss actions taken and information sent to/from systems
- describe flexibility of computer-based systems
- provide variations and equivalents to embodiments

The method begins by obtaining a three-dimensional mesh, which may be generated by a machine-learned model that predicts geometry from input images or sensor data. The mesh is rasterized using a conventional graphics pipeline to produce a two-dimensional raster containing sample points with interpolated attributes. For each pixel in the raster, an initial color value is determined by applying shading and/or texture data to the sample attributes, such as position, normal, and texture coordinates. A splat is constructed for each sample, with its center computed by applying perspective division and viewport transformation to the world-space position of the sample. The splat is weighted based on its proximity to the pixel center using a Gaussian kernel, and a subset of splats is selected for each pixel based on depth layering to ensure correct occlusion handling. Weights are normalized over a local neighborhood to ensure gradient consistency, and the updated color value for each pixel is computed as the sum of normalized splat contributions. Derivatives at occlusion boundaries are generated automatically through the differentiable splatting operation, without requiring edge sampling or custom gradient functions. The resulting two-dimensional differentiable rendering is processed by a machine-learned model, such as a convolutional neural network, to generate a machine-learned output, such as a predicted pose or geometry. A loss function is evaluated by comparing the rendered output to ground truth data, and the model parameters are adjusted using gradient descent or other optimization algorithms. The method supports training of machine-learned pose estimation models and three-dimensional mesh generation models by iteratively rendering images from varying poses, generating new meshes, and refining predictions. These operations may be executed on servers, databases, or software applications that store training data, manage model checkpoints, and distribute rendering tasks across multiple computing nodes. The flexibility of the system allows for variations in splat kernel shape, normalization strategy, shading model, or rasterization technique, and encompasses all equivalents and modifications that fall within the scope of the disclosed method.

### Additional Disclosure

- discuss inherent flexibility of computer-based systems
- describe alterations, variations, and equivalents to embodiments
- intend to cover such alterations, variations, and equivalents

The disclosed invention is implemented on computer-based systems whose architecture may be varied without departing from the essential functionality. The rasterization step may be performed using any conventional graphics API, including OpenGL, Vulkan, or DirectX, and the splatting operation may be implemented in any automatic differentiation framework, such as TensorFlow, PyTorch, or JAX. The Gaussian splat kernel may be replaced with other smooth, differentiable weighting functions, such as quadratic or cubic B-splines, and the normalization scheme may be adapted to use adaptive neighborhood sizes or depth-aware weighting. The multi-layer splatting strategy may be extended to support semi-transparent layers, motion blur, or temporal coherence across frames. The method may be applied to non-rigid deformations, dynamic scenes, or animated meshes by incorporating time as an additional parameter in the optimization. The machine-learned models may be replaced with physics-based simulators, optimization solvers, or hybrid architectures combining neural networks with geometric constraints. The system may be deployed in real-time rendering engines, augmented reality applications, robotics perception systems, or medical imaging pipelines. All such alterations, variations, and equivalents are intended to be encompassed within the scope of the claims, as the invention is not limited to the specific embodiments described but extends to any implementation that performs the core function of generating differentiable renderings through non-differentiable rasterization followed by differentiable splatting.