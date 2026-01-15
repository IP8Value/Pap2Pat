## FIELD OF THE INVENTION

- relate to computer graphics

The present invention relates to the field of computer graphics, specifically to methods and systems for rendering animated and deformed three-dimensional objects using image-based representations that capture spatially varying reflectance properties under arbitrary illumination conditions. The invention pertains to the generation of photorealistic visualizations in applications such as digital film production, virtual reality environments, interactive entertainment systems, e-commerce product visualization, and augmented reality interfaces. It addresses the challenge of preserving the perceptual fidelity of material appearance—such as specular highlights, anisotropic reflections, and complex light transport effects—when the underlying geometry of an object undergoes nonrigid deformation. The invention operates within the domain of hybrid rendering techniques that combine implicit surface geometry with dense, acquired reflectance data sampled across multiple viewpoints and illumination configurations. It further extends the capabilities of surface reflectance field representations by introducing a novel parameterization and shading framework that enables physically plausible animation of objects with complex, nonuniform, and highly detailed surface properties without requiring explicit knowledge of the true object geometry or its surface normals.

## BACKGROUND OF THE INVENTION

- introduce visualization and animation

Visualization and animation of photorealistic three-dimensional objects have long been central goals in computer graphics, driven by demands in cinematic production, immersive simulations, and consumer-facing digital content. Accurate representation of real-world materials requires capturing not only geometric shape but also the intricate ways in which light interacts with surfaces—through reflection, refraction, scattering, and interreflection. Traditional approaches rely on explicit geometric modeling combined with parametric bidirectional reflectance distribution functions (BRDFs), which, while efficient for rendering, are fundamentally limited in their ability to reproduce complex material behaviors such as subsurface scattering, translucency, self-shadowing, and non-Lambertian anisotropy. As a result, many real-world objects—including fabrics, metals with microstructure, biological tissues, and painted surfaces—cannot be faithfully rendered using these simplified models.

- describe model-based rendering

Model-based rendering employs mathematical functions to describe the optical properties of surfaces based on physical principles, typically parameterized by material coefficients such as roughness, metallicness, and anisotropy. These models are often integrated with deformable mesh geometries using techniques like skinning, vertex blending, or finite element simulations to animate objects. While computationally efficient and compatible with modern graphics pipelines, model-based rendering suffers from a lack of expressiveness: it cannot capture emergent optical phenomena that arise from microscale surface structure, layered materials, or global light transport effects. Consequently, the rendered appearance often appears artificial or lacks the subtle variations observed in real materials under dynamic lighting.

- describe image-based rendering

Image-based rendering, by contrast, bypasses the need for explicit material models by directly acquiring and storing a dense set of images of an object under varying viewing and illumination conditions. This approach captures the full spectral and angular response of the surface, including all nonlinear and nonlocal optical effects. However, without an underlying geometric model, such representations are inherently static and cannot be deformed or animated without significant modification or resampling. The absence of a parametric surface description renders traditional animation techniques inapplicable, making image-based rendering unsuitable for dynamic scenes unless augmented with proxy geometry.

- describe hybrid rendering

Hybrid rendering strategies bridge these two paradigms by associating image-based reflectance data with an approximate geometric proxy, commonly referred to as an impostor. This impostor, typically derived from a visual hull or a simplified mesh, serves as a deforming scaffold onto which the acquired image data is mapped. The most prominent hybrid representation is the surface light field, which encodes radiance as a function of view direction and surface position under fixed illumination. While effective for static scenes, surface light fields are constrained to a single lighting environment and cannot support relighting under novel illumination conditions.

- limitations of parametric BRDFs

Parametric BRDFs are fundamentally limited by their inability to represent nonlocal optical phenomena such as interreflections, self-shadowing, and subsurface scattering. Their functional form is constrained by mathematical tractability, leading to oversimplified assumptions about surface microstructure. Even advanced models such as the Cook-Torrance or GGX distributions fail to capture the spatial variation of reflectance across a single object’s surface, which is common in real-world materials like brushed metal, patterned textiles, or weathered paint. Furthermore, fitting these models to measured data often requires extensive optimization and remains inaccurate for complex or heterogeneous surfaces.

- limitations of image-based rendering

Image-based rendering, while rich in detail, is inherently non-deformable. When the underlying geometry is altered, the mapping between the acquired images and the deformed surface becomes ambiguous. Direct warping of image data without geometric consistency leads to visual artifacts such as texture stretching, blurring, or misalignment of highlights. Moreover, the lack of a continuous surface representation prevents the accurate computation of surface normals, lighting directions, and viewing vectors necessary for physically plausible shading under new illumination.

- describe surface light fields

Surface light fields extend image-based rendering by parameterizing radiance measurements onto a continuous impostor surface, enabling view interpolation and efficient rendering via unstructured lumigraph techniques. However, they are restricted to a single, fixed illumination setup. This limitation renders them incapable of supporting dynamic lighting environments, which are essential for realistic animation in changing scenes. The inability to relight objects under arbitrary illumination severely constrains their utility in interactive applications.

- limitations of surface light fields

The primary limitation of surface light fields lies in their dependence on a single lighting configuration. When an object is rendered under new illumination, the resulting appearance is not physically accurate because the reflectance properties of the surface are not modeled as a function of incident light direction. Attempts to approximate lighting changes through post-processing or color scaling introduce artifacts and fail to preserve the angular structure of specular lobes or the spatial variation of anisotropic reflections.

- describe surface reflectance fields

Surface reflectance fields overcome this limitation by capturing the full bidirectional reflectance function across multiple viewing angles and a dense set of incident illumination directions. Each point on the impostor geometry is associated with a multidimensional reflectance sample that encodes how light is reflected from every possible incoming direction to every possible outgoing direction. This representation enables the rendering of objects under arbitrary, dynamic, or even environment-mapped lighting conditions, making it suitable for photorealistic animation in changing environments. The data is typically acquired using a multi-camera, multi-light setup that samples the object from hundreds of viewpoints and dozens of illumination angles.

- limitations of surface reflectance fields

Despite their expressive power, surface reflectance fields have not been successfully animated under nonrigid deformations. Existing methods for deforming image-based representations either assume rigid transformations or rely on exact knowledge of the object’s true geometry and surface normals—information that is typically unavailable in acquired data. Without a principled method to preserve the spatially varying BRDFs during deformation, the appearance of the object becomes distorted: specular highlights shift incorrectly, anisotropic patterns misalign, and material properties appear to change unnaturally. This has rendered surface reflectance fields unusable for animation until the present invention.

- prior art for 3D rendering

Prior art in 3D rendering includes the lumigraph, which combines a visual hull with a dense light field to enable view interpolation while mitigating ghosting artifacts. View-dependent texture mapping reduces data requirements by storing sparse texture views on simplified geometry, but fails on highly specular surfaces due to insufficient sampling. Surface light fields and opacity light fields improve upon these by parameterizing data directly on the surface, yet remain constrained to static lighting. Surface reflectance fields, as introduced in prior work, provide the necessary foundation for relighting but lack any mechanism for deformation.

- describe lumigraph

The lumigraph represents a collection of images captured from a dense set of viewpoints around an object, stored in a four-dimensional light field structure. It is rendered by interpolating between these views using the object’s visual hull as a geometric guide. While effective for viewpoint interpolation, the lumigraph does not encode illumination variation and cannot be used to simulate new lighting conditions. Its utility is therefore limited to fixed-light environments.

- describe view-dependent texture mapping

View-dependent texture mapping associates a small set of texture images with a low-resolution geometry, allowing for efficient rendering by selecting the most appropriate texture based on the viewer’s position. This technique significantly reduces memory usage but sacrifices visual fidelity, particularly for materials with complex reflectance anisotropy or high-frequency specular details. The sparse sampling leads to visible discontinuities and incorrect highlight placement during motion.

- describe surface light fields and opacity light fields

Surface light fields store radiance as a function of surface position and view direction under fixed illumination, enabling high-quality rendering with unstructured interpolation. Opacity light fields extend this by incorporating view-dependent transparency data, allowing for the rendering of semi-transparent or translucent objects. However, both representations are fundamentally static with respect to illumination and cannot support dynamic lighting changes, which are essential for realistic animation.

- describe surface reflectance fields and interpolation

Surface reflectance fields extend surface light fields by adding illumination as a third dimension, resulting in a five-dimensional function that maps surface position, incident light direction, and view direction to reflected radiance. Interpolation is performed using unstructured lumigraph techniques, which weight nearby samples based on geometric and angular proximity. This allows for smooth transitions between acquired data points and enables rendering under novel lighting conditions. However, prior methods do not address how to preserve the local reflectance characteristics when the underlying impostor geometry is deformed.

- prior art for animation of image-based data

Prior attempts to animate image-based data include feature-based morphing of light fields, which requires manual correspondence between two static configurations, and tensor-based compression of bidirectional texture functions (BTFs) for limited deformations. These methods are either restricted to rigid transformations, require exact geometry, or fail to preserve BRDF structure under nonuniform warping. None provide a general, automated, and appearance-preserving method for animating surface reflectance fields under arbitrary deformations.

## SUMMARY OF THE INVENTION

- provide method for rendering deformed and animated surface reflectance fields

The present invention provides a novel method for rendering deformed and animated surface reflectance fields while preserving the perceptual appearance of material properties under arbitrary geometric transformations. The method introduces a local parameterization of the impostor geometry that enables the consistent mapping of reflectance queries from the deformed object space back to the original acquisition space. This mapping, termed the look-up function, enforces three critical conditions for appearance preservation: alignment of surface points, preservation of angles between lighting and viewing directions relative to the surface normal, and maintenance of azimuthal orientation of anisotropic reflectance lobes. By storing at each impostor point its original position, a fixed tangential reference frame, and a deformed tangential system, the method reconstructs a local rigid transformation that aligns the deformed geometry with the acquisition frame without requiring knowledge of the true object surface or its normals. This transformation is applied during shading to remap incident lighting and viewing directions, enabling the correct retrieval of reflectance values from the acquired dataset. The invention further introduces a deferred shading scheme optimized for cache coherence, allowing efficient rendering of large-scale reflectance fields on standard hardware. The result is a robust, automated, and general-purpose framework for animating photorealistic objects with complex, spatially varying reflectance properties under dynamic lighting conditions.

## BRIEF DESCRIPTION OF THE DRAWINGS

- describe figures of the invention

The figures illustrate the components and operational flow of the invention. Figure 1 depicts the acquisition setup, showing multiple cameras and a rotating array of light sources capturing reflectance images of an object from numerous viewpoints. Figure 2 illustrates the structure of a surface reflectance field, representing the multidimensional data as a collection of reflectance images indexed by viewpoint, lighting direction, and surface position. Figure 3 presents the deformation pipeline, showing the mapping from a deformed impostor point to its corresponding location in acquisition space via the look-up function. Figure 4 details the coordinate systems used in the local parameterization, including the original tangential frame R₀ and the deformed tangential vectors u and v. Figure 5 demonstrates the geometric relationship between viewing rays in object space and acquisition space, highlighting the approximation of surface point correspondence. Figure 6 illustrates the preservation of BRDF characteristics through angular alignment of lighting and viewing directions relative to the surface normal. Figure 7 shows the reconstruction of the rotation matrix Lp* using the orthogonalized tangential system derived from the deformed and reference frames. Figure 8 compares circular and elliptical surfel splats under deformation, demonstrating the improved surface coverage achieved by the invention. Figure 9 provides a detailed view of the elliptical surfel defined by position and two nonorthogonal tangential vectors. Figure 10 visualizes the divergence of viewing rays in acquisition space following deformation, illustrating the increased data access complexity. Figure 11 displays the two test objects used in validation: a doll with heterogeneous materials and a mug with refractive properties. Figure 12 shows view extrapolation during deformation, where previously unseen regions are interpolated using unstructured lumigraph techniques. Figure 13 compares the appearance of the doll under deformation, highlighting the preservation of wood grain, fabric diffuse, and specular braid. Figure 14 presents deformations of the mug, demonstrating realistic rendering despite the absence of refraction modeling. Figure 15 illustrates the seamless rendering of large deformations using elliptical splats without visible holes. Figure 16 visualizes the blending fields for camera and light source interpolation, revealing the spatial variation in data weighting. Figure 17 shows a sequence of animated frames with varying lighting and large-scale deformation, demonstrating the full capability of the invention.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENT

- introduce surface reflectance field

A surface reflectance field is a comprehensive representation of an object’s optical behavior, defined as a discrete sampling of the bidirectional reflectance function across a dense set of viewing directions, incident illumination directions, and surface positions. Each point on the impostor geometry is associated with a multidimensional dataset containing reflectance measurements for every combination of incoming and outgoing light direction. This representation captures not only diffuse and specular components but also complex phenomena such as anisotropy, interreflections, and self-shadowing, all encoded without assuming any parametric model of material properties. The data is acquired using a multi-camera, multi-light system that captures hundreds of views under dozens of illumination configurations, resulting in a high-fidelity, relightable model of the object’s appearance.

- define 3D model construction

The three-dimensional model used in this invention is constructed as an impostor geometry derived from the visual hull of the object, computed from silhouette images captured during the acquisition process. This geometry is not intended to represent the true object surface with submillimeter precision but rather to serve as a coarse, deformable scaffold that approximates the object’s overall shape. The impostor may be represented as a point cloud, a triangular mesh, or a surfel-based structure, with each element carrying additional attributes necessary for the rendering pipeline. The choice of representation is independent of the core method, as long as the geometry supports local parameterization and deformation.

- acquire reflectance images

Reflectance images are acquired by capturing a sequence of high dynamic range images of the object under systematically varied illumination conditions. Each image corresponds to a unique combination of camera viewpoint and light source direction. For each viewpoint, a stack of images is recorded, one for each lighting condition, forming a reflectance volume. These images are stored in a structured dataset indexed by position, view direction, and light direction. The acquisition system employs calibrated cameras and programmable light arrays to ensure consistent spatial and angular sampling, producing a comprehensive record of the object’s reflectance behavior under diverse lighting environments.

- define shading operation

The shading operation is the process of determining the color of a point on the deformed impostor geometry under a given lighting and viewing condition. This involves querying the surface reflectance field to retrieve the appropriate reflectance value for the current point, incident light direction, and viewing direction. In the case of a deformed object, the query is first transformed from the deformed object space into the original acquisition space using the look-up function. The retrieved reflectance coefficient is then multiplied by the intensity of the incident light and attenuated by distance to produce the final pixel color. This operation is performed for every visible point on the impostor during rendering.

- motivate deformable surface reflectance fields

The motivation for animating surface reflectance fields arises from the need to render real-world objects with complex, spatially varying reflectance properties under dynamic lighting and deformation. Traditional methods either sacrifice appearance fidelity through parametric modeling or are incapable of deformation due to the static nature of image-based data. The invention enables the animation of objects such as clothing, hair, skin, or organic forms that undergo nonrigid motion while preserving the subtle material characteristics that define their realism. This capability is essential for applications in digital cinema, virtual production, and interactive simulations where photorealism must be maintained during motion.

- define surface reflectance field notation

The surface reflectance field is denoted as SRF(p, l, v), where p represents a point on the impostor geometry, l is the unit vector pointing from the point to the light source, and v is the unit vector pointing from the point to the viewer. This function returns the scalar reflectance value, representing the fraction of incident light from direction l that is reflected toward direction v. During animation, the deformed geometry introduces a new point p*, with corresponding lighting l* and viewing v*. The invention defines a mapping L that transforms the query (p*, l*, v*) into the original acquisition space (p, l, v), enabling the correct retrieval of SRF(p, l, v) without requiring knowledge of the true object surface.

- outline animation process

The animation process begins with the application of a deformation function to the impostor geometry, producing a new configuration of surface points. For each point on the deformed geometry, the invention computes a local rigid transformation that maps the deformed tangential frame back to the original acquisition frame. This transformation is derived from stored parameters including the original position, the fixed reference tangential system, and the deformed tangential vectors. Using this transformation, the incident lighting and viewing directions are rotated to align with the acquisition frame. The surface reflectance field is then queried at the mapped position and directions, and the resulting reflectance value is used to shade the point. This process is repeated for all visible points, with deferred shading and cache-optimized data access ensuring efficient rendering.

### Approximate BRDF Preservation

- motivate BRDF preservation

The preservation of bidirectional reflectance distribution functions during deformation is essential to maintain the perceptual realism of material appearance. Without this preservation, specular highlights shift incorrectly, anisotropic patterns rotate unnaturally, and the overall texture of the surface appears distorted. The invention recognizes that exact preservation is impossible due to the absence of true surface geometry and microstructure information. Instead, it proposes an approximate method that preserves the most perceptually salient characteristics of the BRDF: the angular relationships between lighting, viewing, and surface orientation.

- limitations of exact object geometry and material properties

Exact preservation of BRDFs would require complete knowledge of the object’s true surface geometry, surface normals, and microstructural properties such as microfacet distribution. These are rarely available in image-based acquisition systems, which typically capture only silhouette and radiance data. The invention circumvents this limitation by operating solely on the impostor geometry, which is an approximation of the true surface. This makes the method applicable to a wide range of real-world objects acquired without high-precision scanning.

- introduce local BRDFs

The invention assumes that the appearance of the object can be locally modeled as a spatially varying BRDF, where each point on the surface has a unique reflectance function that depends only on the local lighting and viewing geometry. This assumption permits the use of a local mapping function to preserve the BRDF characteristics during deformation, without requiring global knowledge of the object’s structure or light transport.

- define look-up function L

The look-up function L is a mapping from the deformed query space (p*, l*, v*) to the acquisition space (p, l, v). It is defined per point on the impostor geometry and consists of a translation component that maps p* back to its original position p₀, and a rotational component that aligns the deformed tangential frame with the original acquisition frame. This function is computed using the stored local parameters and does not require the inverse of the deformation function.

- enforce appearance preservation by three conditions

Appearance preservation is enforced through three geometric conditions. First, the viewing ray in object space must intersect the impostor at a point that corresponds as closely as possible to the original surface point. Second, the angle between the lighting direction and the surface normal must be preserved, ensuring that the shape of reflectance lobes remains unchanged. Third, the azimuthal orientation of the lighting and viewing directions relative to the surface tangent plane must be preserved, which is critical for maintaining anisotropic reflectance patterns.

- motivate first condition: same surface point with same BRDF

The first condition ensures that the reflectance value retrieved during shading corresponds to the same physical location on the object as it was originally captured. Although the true surface point is unknown, the invention approximates this by using the impostor point p = É⁻¹(p*) as the origin of the viewing ray in acquisition space. This approximation is valid because the impostor geometry is typically close to the true surface, and the resulting error is negligible compared to the spatial sampling resolution of the reflectance field.

- select p as approximation of q

The point p, derived by applying the inverse deformation to the deformed impostor point p*, is selected as the best available approximation of the true surface point q. This selection is grounded in the observation that the impostor geometry is constructed from the visual hull, which closely bounds the object’s surface. The use of p instead of q introduces minimal error, especially when the deformation is smooth and the impostor is densely sampled.

- motivate second condition: same angle between lighting and surface normal

The second condition ensures that the angle between the incident lighting direction and the surface normal remains unchanged after deformation. This is critical because the shape of the BRDF’s reflectance lobe—whether Gaussian, specular, or anisotropic—is determined primarily by this angle. Preserving this angle ensures that highlights retain their size, intensity, and spread, preventing unnatural brightening or blurring during deformation.

- preserve shape of reflectance lobes

By preserving the angle between lighting and surface normal, the invention ensures that the shape of the reflectance lobe is maintained. This is particularly important for materials with complex BRDFs, such as brushed metal or satin fabric, where the angular distribution of reflected light varies significantly with orientation. Failure to preserve this angle would cause the lobe to stretch or compress, leading to visually implausible artifacts.

- preserve azimuthal orientations

The azimuthal orientation refers to the rotational alignment of the lighting and viewing directions within the tangent plane of the surface. For anisotropic materials, this orientation determines the directionality of reflectance patterns—such as the orientation of hair fibers or brushed metal grain. The invention preserves this orientation by aligning the deformed tangent frame with the original acquisition frame, ensuring that anisotropic features remain correctly oriented relative to the surface.

- motivate third condition: preserve effect of anisotropic BRDFs

Anisotropic BRDFs exhibit directional reflectance patterns that depend on the orientation of the surface microstructure. If the azimuthal alignment between lighting and viewing directions is not preserved, these patterns will rotate or distort unnaturally during deformation. The third condition ensures that the relative orientation of l* and v* in the deformed frame matches that of l and v in the acquisition frame, thereby preserving the perceptual signature of anisotropic materials.

- determine mapping (p*, l*, v*)(l, v)

The mapping from (p*, l*, v*) to (l, v) is formulated as a locally affine transformation that acts on the lighting and viewing vectors in the vicinity of each impostor point. This transformation is derived from the difference between the deformed and original tangential frames and is constrained to be an isometry to preserve both angles and vector lengths.

- express mapping as locally affine

The mapping is expressed as a linear transformation applied to the lighting and viewing vectors, with the transformation matrix Lp* depending on the local deformation at point p*. This matrix is computed from the stored reference and deformed tangential systems and is applied independently at each point, allowing for nonuniform and skewed deformations to be handled without global constraints.

- enforce angle preservation

Angle preservation is enforced by requiring that the transformation Lp* be conformal, meaning it preserves the angles between all pairs of vectors. This is achieved by ensuring that the transformation matrix is orthogonal, which guarantees that the dot product between any two vectors remains unchanged after transformation.

- enforce length preservation

Length preservation is enforced by requiring that the transformation be norm-preserving, meaning that the magnitude of any unit vector remains unchanged. This ensures that the direction of lighting and viewing vectors is not scaled or distorted during the mapping, which would otherwise lead to incorrect reflectance values.

- conclude isometry of function Lp*

The combination of angle preservation and length preservation implies that Lp* is an isometry, meaning it is a rigid transformation composed of a rotation and/or reflection. This conclusion is critical, as it ensures that the local reflectance characteristics are preserved without distortion, even under complex deformations.

- interpret as rotation or reflection

The isometric transformation Lp* can be interpreted as a rotation or reflection of the lighting and viewing vectors relative to the surface. In practice, reflections are rare in physical deformations and are therefore excluded for simplicity, with the transformation assumed to be a pure rotation.

- assume rotation

The invention assumes that the transformation Lp* is a rotation, which is consistent with most physical deformations encountered in real-world objects. This assumption simplifies the implementation and ensures that the handedness of the coordinate system is preserved, avoiding unnatural flipping of reflectance patterns.

- express total effect as rigid transformation

The total effect of the mapping is expressed as a rigid transformation that translates the deformed point p* back to its original position p₀ and rotates the lighting and viewing vectors to align with the original acquisition frame. This rigid transformation is applied locally at each point, enabling the method to handle arbitrary deformations without requiring global consistency.

- restrict to rigid transformations mapping deformed normal to original normal

The rotation component of Lp* is restricted to those transformations that map the deformed normal n* at p* to the original normal n₀ at p₀. This ensures that the surface orientation is correctly aligned with the acquisition frame, even when the impostor geometry deviates from the true surface.

- introduce local model parameterization

The local model parameterization stores, at each impostor point, a set of vectors that define the original and deformed tangential frames. These include the original position p₀, the original tangential reference frame R₀, and the deformed tangential vectors u and v. This parameterization allows the reconstruction of Lp* during rendering without requiring the inverse of the deformation function.

- associate parameters with each point p*

Each point p* on the deformed impostor geometry is associated with a tuple of parameters: p₀, R₀, u, and v. These parameters are computed during the initial deformation and remain fixed for the duration of the animation. The original position p₀ and reference frame R₀ are static, while u and v are updated whenever the geometry is deformed.

- enable look-up function L without inverse deformation function

By storing the original position and reference frame, the invention eliminates the need to compute the inverse of the deformation function É⁻¹. Instead, the mapping from p* to p₀ is achieved by direct lookup, and the rotation is computed from the relative orientation of the deformed and original tangential systems.

- store original position p₀

The original position p₀ is stored as a three-dimensional coordinate that represents the location of the point on the impostor geometry before deformation. This allows the translation component of the mapping to be computed without reference to the deformation function.

- align deformed tangential system with acquisition frame

The deformed tangential system, defined by the vectors u and v, is aligned with the acquisition frame by computing the rotation that maps the deformed basis to the original basis. This alignment ensures that the azimuthal orientation of lighting and viewing directions is preserved.

- provide two coordinate systems: R0 and (u, v)

The invention uses two coordinate systems: the fixed reference system R₀ = (u₀, v₀, n₀) defined in the acquisition frame, and the deformed system (u, v) defined in the object space. These systems are used to compute the rotation Lp* that aligns the deformed frame with the original.

- reconstruct rotation Lp* during rendering

During rendering, the rotation Lp* is reconstructed by orthogonalizing the deformed tangential vectors u and v to form a new basis (û, v̂), and then computing the rotation that maps (u₀, v₀) to (û, v̂). This rotation is applied to the lighting and viewing vectors to align them with the acquisition frame.

- define normal and bisecting vector

The normal vector n is computed as the cross product of the deformed tangential vectors u and v, normalized. The bisecting vector is defined as the normalized sum of the lighting and viewing directions, and is used to orient the tangential system in a way that minimizes angular distortion.

- construct orthogonalized tangential system

The orthogonalized tangential system is constructed by applying a Gram-Schmidt process to the deformed vectors u and v, producing an orthonormal pair (û, v̂) that spans the same plane but is free of skew. This system is used to compute the rotation Lp*.

- give rotation Lp*

The rotation Lp* is given by the matrix that maps the original basis (u₀, v₀, n₀) to the orthogonalized deformed basis (û, v̂, n). This matrix is computed as the product of the transpose of the original basis and the deformed basis, yielding a 3×3 orthogonal matrix that represents the required rotation.

- note approximation of tangential orientation preservation

The method introduces a small approximation in the preservation of tangential orientation, particularly under shearing deformations. This is because the rotation is derived from the impostor geometry rather than the true object surface. However, this approximation is consistent with the assumption that the microstructure of the material is unknown, and thus the original BRDF is retained without modification.

- discuss rigid transformation for rigid object deformations

For rigid object deformations, such as translation, rotation, or uniform scaling, the proposed method reduces to the standard practice of rotating the reflectance data according to the inverse object transformation. In such cases, the method is equivalent to prior art, but extends naturally to nonrigid deformations.

- discuss limitations of prior art

Prior art methods for animating image-based data either assume rigid transformations, require exact geometry, or fail to preserve BRDF structure under nonuniform deformation. The invention overcomes these limitations by providing a general, parameterization-based method that works with approximate geometry and arbitrary deformations.

- introduce shading

Shading is the process of computing the final color of a surface point by combining the retrieved reflectance value with the incident lighting. In the invention, shading is performed by querying the surface reflectance field using the mapped lighting and viewing directions, then multiplying the result by the light intensity and attenuation factor.

- discuss problems with environment map filtering

When using environment maps for lighting, standard filtering techniques are required to avoid aliasing due to the discrete sampling of the reflectance field. However, applying these filters for every surface point during deformation is computationally prohibitive, as the environment map must be re-filtered after each rotation.

- provide alternative method for shading

The invention provides an alternative shading method that simulates environment lighting using a dense set of point light sources placed at the same angular positions as the original acquisition lights. This eliminates the need for environment map filtering, as the lighting is now represented as discrete samples matching the reflectance field’s sampling density.

- simulate lighting with point light sources

Lighting is simulated by defining a set of point light sources at the same spatial positions as the original acquisition lights. Each light source is assigned a color corresponding to the radiance measured at that direction during acquisition. The reflectance field is then queried for each point using these point light directions.

- apply look-up scheme to reflectance query

The look-up scheme is applied to each reflectance query by transforming the point, lighting, and viewing directions into the acquisition space. The resulting query is then used to interpolate the reflectance value from the stored dataset.

- yield reflectance coefficients

The interpolated reflectance value yields a coefficient that represents the fraction of light reflected from the incident direction to the viewing direction. This coefficient is multiplied by the color of the point light source to produce the final radiance contribution.

- shade point with color

The final color of the point is computed by summing the contributions from all active point light sources, each weighted by its distance-based attenuation. This produces a photorealistic shading result that matches the appearance of the object under the simulated environment.

- discuss error in incident lighting direction

The use of the impostor geometry instead of the true object surface introduces a small error in the incident lighting direction. However, this error is typically smaller than the angular resolution of the reflectance field’s sampling, making the resulting artifacts imperceptible.

- discuss interpolation of novel lighting directions

The invention relies on the reflectance field’s interpolation mechanism to handle lighting directions that were not explicitly sampled during acquisition. This interpolation is performed using unstructured lumigraph techniques, which weight nearby samples based on angular and spatial proximity.

- discuss environment mapping with point light sources

Environment mapping is implemented by subsampling the environment map at the same angular resolution as the acquisition lights and assigning each sample to a point light source. This transforms the continuous environment into a discrete set of lights that can be handled by the existing shading pipeline.

- discuss sub-sampling environment map

Sub-sampling the environment map at a density matching the acquisition lighting ensures that the reconstructed lighting approximates the original environment with minimal error. The interpolation inherent in the reflectance field acts as a reconstruction filter, eliminating the need for explicit filtering.

- conclude advantages of method

The method provides a computationally efficient, appearance-preserving framework for animating surface reflectance fields under arbitrary deformations. It requires no knowledge of true surface normals, works with approximate geometry, avoids expensive environment map filtering, and is compatible with standard rendering pipelines. The result is a scalable, robust, and general-purpose solution for photorealistic animation of complex real-world objects.

## EFFECT OF THE INVENTION

- summarize invention benefits

The invention enables the animation of surface reflectance fields with unprecedented fidelity, preserving the perceptual characteristics of complex materials under arbitrary deformations and dynamic lighting. It eliminates the need for exact geometry or surface normals, making it applicable to objects acquired through non-invasive, image-based methods. The method requires no manual parameter tuning, operates automatically on any deformable impostor geometry, and is compatible with existing image-based rendering pipelines. The deferred shading and cache-optimized data access ensure efficient rendering even with large datasets, enabling real-time or near-real-time animation on standard hardware. The invention significantly expands the scope of photorealistic animation, allowing for the realistic depiction of clothing, hair, skin, and other organic or textured materials in digital cinema, virtual production, and interactive applications. It represents a fundamental advancement in hybrid rendering, bridging the gap between high-fidelity image-based capture and dynamic deformation.