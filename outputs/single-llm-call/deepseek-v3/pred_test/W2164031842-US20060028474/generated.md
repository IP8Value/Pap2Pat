Here is the complete patent application following your outline and research paper:

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates generally to the field of computer graphics, and more particularly to methods for rendering deformable and animated surface reflectance fields under arbitrary illumination conditions. The invention provides techniques for preserving material appearance during deformation of image-based representations while enabling dynamic relighting effects.  

## BACKGROUND OF THE INVENTION  

Visualization and animation of realistic 3D computer graphics models are critical for applications including computer games, films, virtual environments, and e-commerce. Traditional approaches to reproducing object appearance fall into three categories: explicit modeling with parametric representations, pure image-based rendering, and hybrid methods combining both approaches.  

Model-based rendering using parametric bidirectional reflectance distribution functions (BRDFs) enables efficient rendering on modern graphics hardware and supports established animation techniques like skinning and vertex blending. However, parametric BRDFs cannot accurately capture many real-world material properties including translucency, interreflections, self-shadowing, and subsurface scattering.  

Pure image-based techniques excel at capturing complex appearance but typically restrict viewpoint selection and lack geometric models needed for deformation. Hybrid approaches address these limitations by parameterizing image data on impostor geometry, enabling animation while maintaining photorealism.  

Surface light fields represent a prominent hybrid approach, parameterizing appearance data on geometric proxies. While effective for fixed illumination, surface light fields cannot adapt to new lighting environments. Surface reflectance fields extend this concept by capturing appearance across multiple lighting conditions, enabling rendering under novel illumination.  

Prior animation techniques for image-based representations include lumigraph rendering, view-dependent texture mapping, and surface light field deformation methods. However, these approaches either maintain fixed illumination or fail to properly preserve material appearance during deformation. Existing methods for animating surface reflectance fields either require precise geometry or do not address appearance preservation under non-uniform deformations.  

The limitations of current techniques create a need for improved methods that enable animation of surface reflectance fields while preserving material appearance across arbitrary deformations and lighting conditions.  

## SUMMARY OF THE INVENTION  

The present invention provides a method for rendering deformed and animated surface reflectance fields that preserves material appearance under arbitrary illumination. The invention employs a local parameterization of impostor geometry that enables correct evaluation of acquired reflectance images during deformation. A novel look-up function approximately preserves spatially varying BRDFs by maintaining key reflectance characteristics through rigid transformations of lighting and viewing directions.  

The method includes acquiring reflectance images under multiple lighting conditions, constructing a 3D model of the object, and augmenting the geometry with local parameterization data. During rendering, deformations are applied to the impostor geometry, and a shading operation maps queries in object space to corresponding reflectance data in acquisition space. The technique supports point-based rendering with elliptical surface splats that adapt to local deformations while maintaining surface coverage.  

## BRIEF DESCRIPTION OF THE DRAWINGS  

Figure 1 illustrates reflectance image acquisition showing an object under varying directional illumination from a fixed viewpoint.  

Figure 2 depicts the two-step rendering process for surface reflectance fields involving image generation under novel illumination followed by geometry-based view interpolation.  

Figure 3 shows the shading pipeline for warped surface reflectance fields including the look-up function that maps object space queries to acquisition space.  

Figure 4 demonstrates the coordinate transformation between object space and acquisition space required for shading deformed surface reflectance fields.  

Figure 5 illustrates viewing ray intersections with both impostor geometry and (unknown) actual object surface during deformation.  

Figure 6 diagrams key angles and orientations that must be preserved to maintain BRDF characteristics during deformation.  

Figure 7 shows the local impostor parameterization including coordinate systems in both acquisition and object space.  

Figure 8 compares circular and elliptical surface splats under deformation, demonstrating improved coverage with the latter.  

Figure 9 defines the elliptical region of influence for a surfel using its tangential coordinate system.  

Figure 10 visualizes viewing ray divergence in acquisition space resulting from impostor deformation.  

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENT  

The invention provides a comprehensive method for animating surface reflectance fields while preserving material appearance. A surface reflectance field comprises reflectance images captured under multiple lighting conditions from various viewpoints, combined with impostor geometry used for rendering.  

The 3D model construction begins with acquisition of reflectance images showing the object under controlled illumination. Each reflectance image represents a stack of camera images from a fixed viewpoint under varying directional lighting. The impostor geometry, typically a visual hull derived from silhouette images, serves as a proxy for the actual object surface.  

The shading operation represents a core innovation, enabling correct evaluation of reflectance data during deformation. For a query (p*, l*, v*) in object space - specifying a surface point, light direction, and view direction - the shading operation computes a corresponding query (p, l, v) in acquisition space. This mapping preserves material appearance by maintaining key BRDF characteristics through three conditions: surface point correspondence, angle preservation between lighting/viewing directions and surface normals, and azimuthal orientation preservation for anisotropic materials.  

The animation process involves applying arbitrary differentiable warp functions to the impostor geometry. To support these deformations, each point in the impostor geometry is augmented with a local parameterization storing its original position and orientation. During rendering, this parameterization enables reconstruction of the rigid transformation needed to map object space queries to acquisition space while preserving appearance.  

### Approximate BRDF Preservation  

Preserving material appearance during deformation requires maintaining the perceptual impression of BRDFs despite lacking exact object geometry and material properties. The invention achieves this through an approximate preservation scheme focusing on local BRDF characteristics rather than global illumination effects.  

The look-up function L maps queries from object space to acquisition space while enforcing three appearance preservation conditions. First, viewing rays in both spaces should intersect corresponding points on their respective impostor geometries. This is approximated by setting p = Φ⁻¹(p*), where Φ represents the deformation function. Although this doesn't guarantee intersection with actual surface points, the impostor proximity provides reasonable approximation.  

Second, the angles between lighting/viewing directions and surface normals must be preserved to maintain reflectance lobe characteristics. This condition ensures that specular highlights and other angle-dependent effects appear consistent after deformation. The invention enforces this by requiring that the mapping between l* and l (and similarly between v* and v) preserves these critical angles.  

Third, for anisotropic materials, the azimuthal orientations of lighting and viewing directions relative to the surface must be maintained. This preserves directional reflectance properties like the appearance of brushed metals or woven fabrics. The mapping achieves this by treating the local transformation as a rigid rotation that maintains these relative orientations.  

Mathematically, the mapping (p*, l*, v*) → (l, v) is expressed as a locally affine function L_p* that must be both angle-preserving and length-preserving - properties characteristic of isometries. The invention interprets this mapping as a rotation (though reflections could alternatively be considered) that rigidly transforms the local frame while translating p* back to its original position p.  

The look-up function construction aligns the warped impostor normal at p* with the original normal at p, using the impostor normal as a practical approximation when actual surface normals are unavailable. This rotation maps the tangential plane at p* to that at p while preserving in-plane orientations to maintain anisotropic effects.  

## EFFECT OF THE INVENTION  

The present invention provides significant advantages over prior animation techniques for image-based representations. By enabling deformation of surface reflectance fields while preserving material appearance under arbitrary illumination, the invention supports new creative possibilities in computer graphics.  

Key benefits include:  
1) Faithful preservation of material properties including specular highlights and anisotropic effects during deformation  
2) Support for dynamic lighting environments beyond the fixed illumination of surface light fields  
3) Compatibility with approximate geometry representations without requiring precise surface normals  
4) Efficient rendering through cache-optimized shading and point-based representations  
5) Robust handling of non-uniform deformations through local parameterization  

The invention finds application in entertainment media, virtual prototyping, architectural visualization, and other domains requiring realistic animation of complex real-world materials. By bridging the gap between image-based capture and geometric deformation, the technique enables new workflows for creating photorealistic animated content.