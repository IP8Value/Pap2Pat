# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to the field of computer graphics and, more specifically, to the animation and rendering of deformable real-world objects using surface reflectance fields. The invention provides a method for animating and rendering deformable objects with complex surface properties, such as specularity, transparency, and self-shadowing, under arbitrary illumination conditions.

## BACKGROUND OF THE INVENTION

Visualization and animation of realistic 3D computer graphics models are essential for various applications, including computer games, movies, advertising, virtual environments, and e-commerce. Traditionally, there are three main approaches to reproducing the visual appearance of real objects: explicit modeling with parametric representations, pure image-based approaches, and hybrid approaches that combine both.

Explicit appearance models use parametric Bidirectional Reflectance Distribution Functions (BRDFs) fitted to acquired data. These models can be efficiently rendered on modern graphics hardware, and the underlying geometry can be animated and deformed using well-established techniques such as skinning and vertex blending. However, parametric BRDFs often fail to capture many effects of real-world materials, such as translucency, interreflections, self-shadowing, and subsurface scattering.

Pure image-based techniques, on the other hand, are well-suited for acquiring and representing complex object appearance. However, they typically impose restrictions on the viewpoints and lack a 3D geometry model, making deformations very difficult or impossible.

Hybrid approaches, which parameterize an image-based model on an impostor geometry, have become increasingly popular. One such representation is the surface reflectance field, which captures the object appearance for many possible light configurations. While surface reflectance fields can render objects with arbitrary reflectance properties from any viewpoint under new illumination, animating these fields has been a significant challenge.

Previous work in this area has focused on static illumination or limited deformations. For instance, Wood et al. described arbitrary deformations on a surface light field but did not properly handle the diffuse component of the surface color and worked only for purely reflective isotropic BRDFs. Feature-based light field morphing required substantial user input and was not applicable to general animation settings. Furukawa et al. presented a system to capture objects and spatially varying BRDFs, but their method relied on a tight impostor geometry and did not explicitly address appearance preservation under non-uniform, skewed deformations.

The present invention addresses these limitations by providing a method for animating surface reflectance fields with arbitrary geometric deformations while preserving the visual appearance of the object materials under different lighting conditions.

## SUMMARY OF THE INVENTION

The present invention provides a method for animating and rendering deformable surface reflectance fields (SRFs) with arbitrary geometric deformations while preserving the visual appearance of the object materials under different lighting conditions. The method involves the following steps:

1. **Deformation of Impostor Geometry**: Apply a 3D warp to the impostor geometry of the SRF.
2. **Local Parameterization**: Augment the impostor geometry with a local parameterization that allows the correct evaluation of acquired reflectance images.
3. **Look-Up Function**: Develop a look-up function to map queries from the deformed object space to the original acquisition space, ensuring the preservation of the object's BRDFs.
4. **Shading Scheme**: Implement a deferred shading scheme to handle the increased amount of data involved in shading the deformable SRF.

The invention ensures that the visual appearance of the deformed object is preserved by approximately maintaining the spatially varying BRDFs of the undeformed object. This is achieved through a novel local impostor parameterization and a look-up function that accounts for the deformation of the object's surface.

## BRIEF DESCRIPTION OF THE DRAWINGS

FIG. 1 illustrates the concept of a surface reflectance field, showing reflectance images from multiple viewpoints under varying directional illumination.

FIG. 2 depicts the two-step procedure for rendering a surface reflectance field: computing an image of the object under new illumination for each viewpoint and rendering these images using the impostor geometry.

FIG. 3 shows the process of shading a point on the impostor surface by applying the look-up function to the surface point, the viewing ray, and the incident light direction.

FIG. 4 illustrates the mapping of a query from object space to the original acquisition space using the inverse warp function.

FIG. 5 demonstrates the intersection of viewing rays in object and acquisition space with the object and impostor geometries.

FIG. 6 explains the preservation of the angles and azimuthal orientations of the lighting and viewing directions relative to the surface normal.

FIG. 7 shows the local impostor parameterization, including the tangential systems in the acquisition and object spaces.

FIG. 8 illustrates the effect of deforming the surfel geometry using elliptical surface splats.

FIG. 9 defines the structure of a surfel, including its position and tangential system.

FIG. 10 demonstrates the viewing ray divergence in distorted regions of the SRF.

FIG. 11 shows the models used in the experiments, including the doll and the beer mug.

FIG. 12 illustrates view extrapolation in deformed SRFs.

FIG. 13 demonstrates the quality of appearance preservation in the doll data set.

FIG. 14 shows various deformations of the beer mug model.

FIG. 15 illustrates the rendering of large deformations without visible holes.

FIG. 16 visualizes the camera and light source blending fields in a deformed SRF.

FIG. 17 shows a sequence of frames from a surface reflectance field animation.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENT

### Approximate BRDF Preservation

In the present invention, the goal is to preserve the visual appearance of the deformed object by approximately maintaining the spatially varying BRDFs of the undeformed object. This is achieved through a novel look-up function that maps queries from the deformed object space to the original acquisition space.

The look-up function is designed to meet the following conditions:
1. **Intersection of Viewing Rays**: The viewing ray in object space should intersect the warped object at a point corresponding to the intersection of the viewing ray in the acquisition space with the original object.
2. **Angle Preservation**: The angles between the lighting and viewing directions and the surface normal should be preserved.
3. **Azimuthal Orientation Preservation**: The azimuthal orientations of the lighting and viewing directions relative to the surface should be preserved.

These conditions ensure that the reflectance characteristics of the object are maintained during deformation. The look-up function is a rigid transformation that translates the warped impostor point back to its original position and rotates the lighting and viewing directions accordingly.

### Local Impostor Parameterization

To enable the look-up function, the impostor geometry is augmented with a local parameterization. This parameterization includes:
- **Original Position**: The original position of the impostor point in the acquisition space.
- **Tangential System**: The tangential system in the acquisition space, stored as a Rodrigues vector.
- **Warped Tangential System**: The tangential system in the object space, which is updated after each deformation.

The local parameterization allows the determination of the look-up function during rendering without requiring an explicit inverse warp function. The tangential system is mapped using directional derivatives of the warp function, ensuring surface coverage and minimizing the squared angular differences between corresponding basis vectors.

### Shading

Once the look-up function is determined, the impostor point can be shaded using the reflectance images from the SRF. The shading process involves:
1. **Environment Mapping**: Filtering the environment map according to the spatial resolution of the reflectance images to avoid aliasing artifacts.
2. **Point Light Sources**: Simulating lighting by point light sources, which avoids the need for refiltering the environment map for each surface point.
3. **Reflectance Image Interpolation**: Interpolating between the acquired reflectance images to handle novel lighting directions and camera views.

The shading scheme is optimized for performance using a deferred shading approach, which collects and sorts shading operations to minimize cache misses and improve rendering times.

### Implementation

The invention is implemented using an extended surfel representation for the impostor geometry. The surfel representation includes:
- **Elliptical Surface Splats**: Each surfel is defined by its position and two tangential vectors, allowing for seamless rendering of deformations.
- **Local Impostor Parameters**: The surfel attributes include the original position, orientation, and tangential system, enabling the look-up function to be applied during rendering.

The reflectance image interpolation is performed using unstructured lumigraph interpolation, which handles view and light direction interpolation. The shading process is further optimized using a reordered evaluation technique that minimizes cache misses and improves rendering performance.

### Results

The invention has been tested on two models: a doll and a beer mug. The doll model, with its various materials, is well-suited for analyzing appearance preservation. The beer mug model, with its refractive effects, demonstrates the limitations of the approach. The results show that the invention effectively preserves the visual appearance of the deformed objects under different lighting conditions, with realistic results for reasonable deformations.

### Effect of the Invention

The present invention provides a robust method for animating and rendering deformable surface reflectance fields while preserving the visual appearance of the object materials under different lighting conditions. The method is particularly useful for real-world models with complex surface properties, where exact surface normals are hard to determine. By using a local parameterization and a novel look-up function, the invention ensures that the reflectance characteristics of the object are maintained during deformation, leading to realistic and visually appealing animations.