Here is the complete patent application following the provided outline and research paper:

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates generally to computer graphics and image-based rendering. More specifically, the invention pertains to systems and methods for rendering deformable surfaces while preserving reflectance properties under arbitrary illumination conditions. The invention enables photorealistic animation of real-world objects captured through surface reflectance fields by maintaining bidirectional reflectance distribution function (BRDF) characteristics during geometric deformation.  

## BACKGROUND OF THE INVENTION  

Animation of photorealistic computer graphics models is critical for applications such as films, video games, virtual environments, and e-commerce. Traditional approaches for reproducing real-world object appearance fall into three categories: explicit parametric modeling, pure image-based techniques, and hybrid methods combining geometry with image data.  

Parametric BRDF models allow efficient rendering on modern graphics hardware and support animation through established deformation techniques. However, they fail to capture complex real-world material properties like subsurface scattering, interreflections, and self-shadowing. Pure image-based methods excel at capturing intricate appearance details but lack geometric representation, making deformation impossible.  

Hybrid approaches parameterize image data onto impostor geometry, enabling animation while preserving visual fidelity. Surface light fields represent one such hybrid technique but are limited to fixed illumination conditions. Surface reflectance fields extend this concept by capturing appearance under varying illumination, but prior to this invention, no method existed for properly animating them while preserving material properties during deformation.  

A fundamental challenge in animating image-based representations involves evaluating captured data to simulate deformations while maintaining visual appearance. Prior techniques either required accurate geometry unavailable in reflectance fields or failed to preserve BRDF characteristics under non-uniform warps. The present invention solves these problems through a novel local parameterization and shading scheme that maintains reflectance properties during arbitrary deformations of approximate geometry.  

## SUMMARY OF THE INVENTION  

The invention provides a system and method for rendering deformable surface reflectance fields while preserving material appearance under arbitrary illumination. Key aspects include:  

A local impostor parameterization that stores original surface positions and orientations at each point of the impostor geometry. During deformation, this parameterization enables reconstruction of an appearance-preserving mapping between deformed object space and original acquisition space.  

An approximate BRDF preservation technique that maintains reflectance characteristics through a rigid transformation of lighting and viewing directions. The transformation rotates directions according to local surface deformation while preserving angles between vectors critical for BRDF evaluation.  

A deferred shading scheme optimized for cache-coherent access to reflectance field data. By reordering shading operations and implementing a compact cache, the system minimizes disk access when evaluating numerous reflectance images required for deformed surface rendering.  

An extended surfel representation using elliptical surface splats that deform according to local geometric distortion. The splats maintain surface coverage during large deformations while providing the tangential frame information needed for proper BRDF evaluation.  

The invention enables photorealistic animation of real-world objects captured through surface reflectance fields, preserving material properties under new lighting conditions and arbitrary geometric deformations. This represents a significant advancement over prior methods limited to static illumination or requiring precise geometry knowledge.  

## BRIEF DESCRIPTION OF THE DRAWINGS  

FIG. 1 illustrates reflectance images comprising stacks of camera images showing an object under varying illumination.  

FIG. 2 depicts the two-step rendering process for surface reflectance fields involving illumination computation followed by impostor rendering.  

FIG. 3 shows the shading process for a warped surface reflectance field using the invented look-up function.  

FIG. 4 demonstrates mapping between deformed object space and original acquisition space queries.  

FIG. 5 illustrates viewing ray intersection conditions for appearance preservation.  

FIG. 6 diagrams angle preservation requirements for BRDF characteristics.  

FIG. 7 displays the local impostor parameterization coordinate systems.  

FIG. 8 compares circular and elliptical splats under deformation.  

FIG. 9 defines the elliptical surfel region of influence.  

FIG. 10 shows viewing ray divergence in acquisition space due to deformation.  

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENT  

The preferred embodiment implements a system for rendering deformable surface reflectance fields comprising three main components: an extended surfel representation with local parameterization, a BRDF-preserving shading algorithm, and an optimized rendering pipeline.  

The surfel representation stores for each point: the original position p0 in acquisition space, original orientation r0 as a Rodrigues vector, and current tangential vectors u and v. During deformation, u and v are updated using central differences of the warp function while p0 and r0 remain fixed. This parameterization enables reconstruction of the rigid transformation L needed for proper BRDF evaluation.  

The shading algorithm implements approximate BRDF preservation through three key conditions:  

1) Viewing rays in deformed and acquisition space should intersect corresponding points on the impostor geometry. This is achieved by mapping point p' in deformed space to p0 in acquisition space.  

2) Lighting/viewing directions and surface normals must maintain equivalent angles in both spaces to preserve reflectance lobe characteristics. The transformation L enforces this by being an isometry.  

3) For anisotropic materials, azimuthal orientations relative to the surface must be preserved. The orthogonalized tangential system ("u, "v) maintains this relationship.  

The transformation L is computed as:  
L = R0 * [ "u/||u||  "v/||v||  n/||n|| ]^T  
where R0 is the original orientation, ("u, "v, n) form the orthogonalized current tangential frame, and n is the normalized cross product of u and v.  

### Approximate BRDF Preservation  

The BRDF preservation technique approximates the complex light transport of real materials while working within the constraints of surface reflectance field representation. Although global effects like interreflections cannot be perfectly preserved without exact geometry knowledge, the method maintains local reflectance properties critical for visual fidelity.  

For each shading query (p', l', v') in deformed space:  
1) Apply translation p = p0  
2) Transform directions using rotation L: l = L(l'), v = L(v')  
3) Evaluate SRF(p,l,v) through reflectance image interpolation  
4) Apply lighting calculation using transformed directions  

The system handles environment maps by subsampling into point lights and implements a deferred shading scheme where all shading operations are collected, sorted for cache coherence, then executed in optimal order. This minimizes disk access when processing the numerous reflectance images required for deformed surface evaluation.  

## EFFECT OF THE INVENTION  

The invention enables several important advancements in computer graphics:  

1) Photorealistic animation of real-world objects captured through surface reflectance fields, maintaining visual fidelity under arbitrary deformations and new illumination conditions.  

2) Practical rendering of complex material properties like specular highlights, anisotropic reflections, and subsurface scattering during deformation, without requiring precise geometric knowledge.  

3) Efficient handling of large reflectance field datasets through optimized caching and reordered evaluation, making deformable surface rendering feasible on conventional hardware.  

4) Flexible representation working with various geometric forms (meshes, point clouds) while maintaining surface coverage during large deformations through elliptical surfel adaptation.  

These effects collectively solve the long-standing problem of animating relightable image-based representations while preserving material appearance, opening new possibilities for realistic computer graphics in film, games, and virtual environments.