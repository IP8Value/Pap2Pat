- The paper introduces a novel method to deform and relight surface reflectance fields (SRFs) while approximately preserving material properties. It uses an impostor geometry with a local parameterization that doesn't require accurate normals. This is particularly useful for real-world models with complex surfaces.

- An extended surfel representation with elliptical splats allows seamless rendering of large deformations while maintaining texture quality. The surfels' tangential vectors directly provide parameters needed for the SRF look-up, complementing the local impostor parameterization approach.

- To handle increased reflectance image queries during deformation, the method uses a small cache and reorders shading operations to improve cache coherency. This reduces shading times by an order of magnitude compared to naive shading approaches.

- The technique was tested on two models - a doll with various materials and a beer mug showing limitations for refractive effects. Results demonstrate realistic appearance preservation under deformations, even when extrapolating views not captured during acquisition. 

- Potential future work includes developing approximate methods to preserve global illumination effects by leveraging the SRF geometry or collecting additional information like opacity hulls during acquisition. Stereo algorithms could also enhance impostor geometry to reduce texture parallax issues.
